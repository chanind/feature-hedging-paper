from typing import Any, Callable

import torch
from sae_lens.training.training_sae import TrainStepOutput
from torch import nn

from hedging_paper.saes.base_sae import BaseSAE, BaseSAEConfig


class BatchTopK(nn.Module):
    def __init__(
        self, k: int, postact_fn: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()
    ):
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts = x.relu()
        acts_topk = torch.topk(acts.flatten(), self.k * acts.shape[0], dim=-1)
        return (
            torch.zeros_like(acts.flatten())
            .scatter(-1, acts_topk.indices, acts_topk.values)
            .reshape(acts.shape)
        )


class BatchTopkSAE(BaseSAE):
    topk_threshold: torch.Tensor

    def __init__(self, cfg: BaseSAEConfig):
        super().__init__(cfg)
        if cfg.architecture != "topk":
            raise ValueError("BatchTopkSAE requires topk architecture")
        self.register_buffer(
            "topk_threshold",
            torch.tensor(0.0, dtype=torch.double, device=self.W_dec.device),
        )
        self.activation_fn = BatchTopK(cfg.activation_fn_kwargs["k"])

    def training_forward_pass(
        self,
        sae_in: torch.Tensor,
        current_l1_coefficient: float,
        dead_neuron_mask: torch.Tensor | None = None,
    ) -> TrainStepOutput:
        # this is hacky, but can't call super() here because this is extended by BatchTopkMatryoshkaSAE and that breaks things
        output = BaseSAE.training_forward_pass(
            self, sae_in, current_l1_coefficient, dead_neuron_mask
        )
        self.update_topk_threshold(output.feature_acts)
        # not technically a loss, but useful to track in wandb
        output.losses["topk_threshold"] = self.topk_threshold
        return output

    @torch.no_grad()
    @torch.autocast("cuda", enabled=False)
    def update_topk_threshold(self, acts_topk, lr=0.01):
        positive_mask = acts_topk > 0
        if positive_mask.any():
            min_positive = acts_topk[positive_mask].min().to(self.topk_threshold.dtype)
            self.topk_threshold = (1 - lr) * self.topk_threshold + lr * min_positive

    def save_model_final(self, *args, **kwargs):
        # hacky way to make this save out as a jumprelu SAE
        self.cfg.activation_fn_str = "relu"
        self.cfg.architecture = "jumprelu"
        res = super().save_model_final(*args, **kwargs)
        self.cfg.activation_fn_str = "topk"
        self.cfg.architecture = "topk"
        return res

    def process_state_dict_for_saving_final(self, state_dict: dict[str, Any]) -> None:
        # turn this into a jumprelu SAE with a uniform threshold
        with torch.no_grad():
            threshold = (
                self.topk_threshold.expand(self.cfg.d_sae).clone().to(self.W_dec.dtype)
            )
            state_dict["threshold"] = threshold
            del state_dict["topk_threshold"]

    def process_state_dict_for_loading(self, state_dict: dict[str, Any]) -> None:
        super().process_state_dict_for_loading(state_dict)
        if "threshold" in state_dict:
            state_dict["topk_threshold"] = (
                state_dict["threshold"][0].clone().contiguous()
            )
            del state_dict["threshold"]

    def encode_standard(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if len(original_shape) == 3:
            x = x.reshape(-1, original_shape[-1])

        result = super().encode_standard(x)

        if len(original_shape) == 3:
            result = result.reshape(original_shape[0], original_shape[1], -1)

        return result
