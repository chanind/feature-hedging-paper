import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from sae_lens import (
    LanguageModelSAERunnerConfig,
    TrainingSAE,
    TrainingSAEConfig,
)
from sae_lens.sae import SAE_CFG_FILENAME, SAE_WEIGHTS_FILENAME, SPARSITY_FILENAME
from sae_lens.training.sae_trainer import SAETrainer
from sae_lens.training.training_sae import SAE_WEIGHTS_PATH, Step, TrainStepOutput
from safetensors.torch import load_file, save_file
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES

from hedging_paper.saes.util import from_sae_runner_config, get_extra_field_items


@dataclass
class BaseSAERunnerConfig(LanguageModelSAERunnerConfig):
    sa_initial_temp: float = 1.0
    sa_final_temp: float = 0.01
    sa_cooling_rate: float = 0.99
    sa_step_size: float = 0.01
    fast_forward_steps: int = 0
    disable_enc_bias: bool = False
    reconstruction_loss: Literal["MSE", "L2"] = "MSE"
    extend_sae_path: str | None = None
    extend_sae_latents: int | None = None
    min_l1_during_warm_up: float = 0.0


@dataclass
class BaseSAEConfig(TrainingSAEConfig):
    disable_enc_bias: bool = False
    reconstruction_loss: Literal["MSE", "L2"] = "MSE"
    lr: float = None  # type: ignore
    extend_sae_latents: int | None = None
    min_l1_during_warm_up: float = 0.0

    @classmethod
    def from_sae_runner_config(  # type: ignore
        cls, cfg: BaseSAERunnerConfig
    ) -> "BaseSAEConfig":
        return from_sae_runner_config(cfg, cls)

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), **get_extra_field_items(self)}


class BaseSAE(TrainingSAE):
    cfg: BaseSAEConfig  # type: ignore
    # hacky way to let the SAE save a checkpoint itself
    trainer: SAETrainer | None = None

    def __init__(self, cfg: BaseSAEConfig):
        super().__init__(cfg)
        self.cfg = cfg  # type: ignore
        self.step_num = 0

    def initialize_weights_complex(self):
        super().initialize_weights_complex()
        if self.cfg.disable_enc_bias:
            self.b_enc.data = torch.zeros_like(self.b_enc)
            self.b_enc.requires_grad = False

    def extend_sae(self, num_latents: int, initial_magnitude: float = 0.1):
        dtype = self.W_dec.dtype
        device = self.W_dec.device
        d_in = self.cfg.d_in

        # Ensure d_in is correctly inferred if not directly available initially
        if d_in is None:
            raise ValueError(
                "Cannot extend SAE if d_in is not specified in the config."
            )

        with torch.no_grad():
            new_latents_weights = torch.randn(
                (num_latents, d_in),
                device=device,
                dtype=dtype,
            )
            new_latents_weights = new_latents_weights / torch.norm(
                new_latents_weights, dim=-1, keepdim=True
            )
            new_latents_weights = new_latents_weights * initial_magnitude

        # Create new tensors by concatenating old data and new data
        new_W_dec_data = (
            torch.cat([self.W_dec.data, new_latents_weights], dim=0)
            .detach()
            .clone()
            .contiguous()
        )
        # Assuming W_enc is tied or initialized based on W_dec extension
        new_W_enc_data = (
            torch.cat([self.W_enc.data.T, new_latents_weights], dim=0)
            .T.detach()
            .clone()
            .contiguous()
        )
        new_b_enc_data = (
            torch.cat(
                [self.b_enc.data, torch.zeros(num_latents, device=device, dtype=dtype)],
                dim=0,
            )
            .detach()
            .clone()
            .contiguous()
        )

        # Reassign parameters instead of modifying .data and detaching
        self.W_dec = torch.nn.Parameter(new_W_dec_data)
        self.W_enc = torch.nn.Parameter(new_W_enc_data)
        self.b_enc = torch.nn.Parameter(new_b_enc_data)

        # Update the config d_sae value stored on the SAE instance
        self.cfg.d_sae = self.W_dec.shape[0]

    def save_model_final(self, path: str | Path, sparsity: torch.Tensor | None = None):
        path = Path(path)

        if not path.exists():
            path.mkdir(parents=True)

        # generate the weights
        state_dict = self.state_dict()
        self.process_state_dict_for_saving_final(state_dict)
        model_weights_path = path / SAE_WEIGHTS_FILENAME
        save_file(state_dict, model_weights_path)

        # save the config
        config = self.cfg.to_dict()

        # use the tlens hooks/names if we trained on huggingface models
        if "model.layers." in config["hook_name"]:
            layer_num = config["hook_name"].split("model.layers.")[1]
            config["hook_name"] = f"blocks.{layer_num}.hook_resid_post"
            config["model_from_pretrained_kwargs"] = {"center_writing_weights": False}
        if (
            "/" in config["model_name"]
            and config["model_name"] not in OFFICIAL_MODEL_NAMES
        ):
            config["model_name"] = config["model_name"].split("/")[1]

        cfg_path = path / SAE_CFG_FILENAME
        with open(cfg_path, "w") as f:
            json.dump(config, f)

        if sparsity is not None:
            sparsity_in_dict = {"sparsity": sparsity}
            sparsity_path = path / SPARSITY_FILENAME
            save_file(sparsity_in_dict, sparsity_path)
            return model_weights_path, cfg_path, sparsity_path

        return model_weights_path, cfg_path

    def skip_main_losses(self) -> bool:
        return False

    def _get_mse_loss_fn(self):
        if self.cfg.reconstruction_loss == "L2":
            return l2_loss
        return super()._get_mse_loss_fn()

    def training_forward_pass(
        self,
        sae_in: torch.Tensor,
        current_l1_coefficient: float,
        dead_neuron_mask: torch.Tensor | None = None,
    ) -> TrainStepOutput:
        # do a forward pass to get SAE out, but we also need the
        # hidden pre.
        l1_coefficient = max(current_l1_coefficient, self.cfg.min_l1_during_warm_up)
        feature_acts, hidden_pre = self.encode_with_hidden_pre_fn(sae_in)
        sae_out = self.decode(feature_acts)

        # MSE LOSS
        per_item_mse_loss = self.mse_loss_fn(sae_out, sae_in)
        if self.skip_main_losses():
            mse_loss = torch.tensor(0.0, device=sae_in.device, dtype=sae_in.dtype)
        else:
            mse_loss = per_item_mse_loss.sum(dim=-1).mean()
        loss = mse_loss

        losses: dict[str, float | torch.Tensor] = {}

        if self.cfg.architecture == "gated":
            # Gated SAE Loss Calculation

            # Shared variables
            sae_in_centered = (
                self.reshape_fn_in(sae_in) - self.b_dec * self.cfg.apply_b_dec_to_input
            )
            pi_gate = sae_in_centered @ self.W_enc + self.b_gate
            pi_gate_act = torch.relu(pi_gate)

            # SFN sparsity loss - summed over the feature dimension and averaged over the batch
            l1_loss = (
                l1_coefficient
                * torch.sum(pi_gate_act * self.W_dec.norm(dim=1), dim=-1).mean()
            )

            # Auxiliary reconstruction loss - summed over the feature dimension and averaged over the batch
            via_gate_reconstruction = pi_gate_act @ self.W_dec + self.b_dec
            aux_reconstruction_loss = torch.sum(
                (via_gate_reconstruction - sae_in) ** 2, dim=-1
            ).mean()
            loss = mse_loss + l1_loss + aux_reconstruction_loss
            losses["auxiliary_reconstruction_loss"] = aux_reconstruction_loss
            losses["l1_loss"] = l1_loss
        elif self.cfg.architecture == "jumprelu":
            if not self.skip_main_losses():
                threshold = torch.exp(self.log_threshold)
                l0 = torch.sum(
                    Step.apply(hidden_pre, threshold, self.bandwidth),  # type: ignore
                    dim=-1,
                )
                l0_loss = (l1_coefficient * l0).mean()
                loss = mse_loss + l0_loss
                losses["l0_loss"] = l0_loss
        elif self.cfg.architecture == "topk":
            topk_loss = self.calculate_topk_aux_loss(
                sae_in=sae_in,
                sae_out=sae_out,
                hidden_pre=hidden_pre,
                dead_neuron_mask=dead_neuron_mask,
            )
            losses["auxiliary_reconstruction_loss"] = topk_loss
            loss = mse_loss + topk_loss
        else:
            # default SAE sparsity loss
            if not self.skip_main_losses():
                weighted_feature_acts = feature_acts
                if self.cfg.scale_sparsity_penalty_by_decoder_norm:
                    weighted_feature_acts = feature_acts * self.W_dec.norm(dim=1)
                sparsity = weighted_feature_acts.norm(
                    p=self.cfg.lp_norm, dim=-1
                )  # sum over the feature dimension

                l1_loss = (l1_coefficient * sparsity).mean()
                loss = mse_loss + l1_loss
                losses["l1_loss"] = l1_loss

        losses["mse_loss"] = mse_loss

        if (
            self.cfg.extend_sae_latents is not None
            and self.cfg.extend_sae_latents > 0
            and self.trainer is not None
        ):
            dead_extension_neurons = self.trainer.dead_neurons[
                -self.cfg.extend_sae_latents :
            ]
            losses["dead_extension_neurons"] = dead_extension_neurons.sum()
            extension_latents_enc_norm = self.W_enc[
                :, -self.cfg.extend_sae_latents :
            ].norm(dim=0)
            losses["extension_latents_enc_norm"] = extension_latents_enc_norm.mean()
            extension_latents_dec_norm = self.W_dec[
                -self.cfg.extend_sae_latents :
            ].norm(dim=-1)
            losses["extension_latents_dec_norm"] = extension_latents_dec_norm.mean()
            losses["extension_latents_norm"] = (
                extension_latents_enc_norm * extension_latents_dec_norm
            ).mean()
        self.step_num += 1

        return TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=loss,
            losses=losses,
        )

    def process_state_dict_for_saving_final(self, state_dict: dict[str, Any]) -> None:
        return self.process_state_dict_for_saving(state_dict)

    def load_from_checkpoint(self, checkpoint_path: str | Path):
        state_dict = load_file(Path(checkpoint_path) / SAE_WEIGHTS_PATH)
        self.process_state_dict_for_loading(state_dict)
        self.load_state_dict(state_dict)


def l2_loss(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (preds - target).norm(dim=-1, keepdim=True)
