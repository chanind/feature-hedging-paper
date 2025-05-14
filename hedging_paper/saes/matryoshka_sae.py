from collections.abc import Generator, Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch
from sae_lens.training.training_sae import (
    Step,
    TrainStepOutput,
    _calculate_topk_aux_acts,
)

from hedging_paper.saes.base_sae import BaseSAE, BaseSAEConfig, BaseSAERunnerConfig
from hedging_paper.saes.batch_topk_sae import BatchTopkSAE
from hedging_paper.saes.util import (
    from_sae_runner_config,
    get_extra_field_items,
)


@dataclass
class MatryoshkaSAERunnerConfig(BaseSAERunnerConfig):
    matryoshka_steps: list[int] = field(default_factory=list)
    include_matryoshka_inner_l1_loss: bool = True
    matryoshka_inner_l1_loss_multipliers: list[float] | float = 1.0
    matryoshka_inner_loss_multipliers: list[float] | float = 1.0
    outer_loss_multiplier: float = 1.0
    skip_outer_loss: bool = False
    matryoshka_reconstruction_loss: Literal["L2", "MSE"] = "MSE"
    matryoshka_reconstruction_l2_power: float = 1.0
    use_matryoshka_aux_loss: bool = False
    # match original matryoshka loss normalization
    normalize_losses_by_num_matryoshka_steps: bool = False
    normalize_reconstruction_losses_by_d_in: bool = False
    include_bdec_loss: bool = False
    use_delta_loss: bool = False
    # loss balancing
    inner_loss_balancing_enabled: bool = False
    matryoshka_loss_multiplier_logits: list[float] | float = 0.0
    update_inner_multipliers_every_n_steps: int = 50
    update_inner_multipliers_start_step: int = 0
    update_inner_multipliers_lr: float = 0.01


@dataclass
class MatryoshkaSAEConfig(BaseSAEConfig):
    matryoshka_steps: list[int] = field(default_factory=list)
    include_matryoshka_inner_l1_loss: bool = True
    matryoshka_inner_l1_loss_multipliers: list[float] | float = 1.0
    matryoshka_inner_loss_multipliers: list[float] | float = 1.0
    outer_loss_multiplier: float = 1.0
    skip_outer_loss: bool = False
    matryoshka_reconstruction_loss: Literal["L2", "MSE"] = "MSE"
    matryoshka_reconstruction_l2_power: float = 1.0
    use_matryoshka_aux_loss: bool = False
    # match original matryoshka loss normalization
    normalize_losses_by_num_matryoshka_steps: bool = False
    normalize_reconstruction_losses_by_d_in: bool = False
    include_bdec_loss: bool = False
    use_delta_loss: bool = False
    # loss balancing
    inner_loss_balancing_enabled: bool = False
    matryoshka_loss_multiplier_logits: list[float] | float = 0.0
    update_inner_multipliers_every_n_steps: int = 50
    update_inner_multipliers_start_step: int = 0
    update_inner_multipliers_lr: float = 0.01

    @classmethod
    def from_sae_runner_config(  # type: ignore
        cls, cfg: MatryoshkaSAERunnerConfig
    ) -> "MatryoshkaSAEConfig":
        return from_sae_runner_config(cfg, cls)

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), **get_extra_field_items(self)}


class MatryoshkaSAE(BaseSAE):
    cfg: MatryoshkaSAEConfig  # type: ignore
    inner_l1_loss_multipliers: list[float]
    reconstruction_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    matryoshka_loss_multiplier_logits: torch.Tensor

    def __init__(
        self,
        cfg: MatryoshkaSAEConfig,
    ):
        super().__init__(cfg)
        self.matryoshka_steps = sorted(set(cfg.matryoshka_steps))
        self.reconstruction_loss_fn = loss_fn(
            cfg.matryoshka_reconstruction_loss, cfg.matryoshka_reconstruction_l2_power
        )
        if isinstance(cfg.matryoshka_inner_l1_loss_multipliers, list):
            self.inner_l1_loss_multipliers = cfg.matryoshka_inner_l1_loss_multipliers
        else:
            self.inner_l1_loss_multipliers = [
                cfg.matryoshka_inner_l1_loss_multipliers
            ] * len(self.matryoshka_steps)
        expected_num_losses = len(self.matryoshka_steps)
        if not self.cfg.skip_outer_loss:
            expected_num_losses += 1
        if isinstance(cfg.matryoshka_loss_multiplier_logits, list):
            matryoshka_loss_multiplier_logits = cfg.matryoshka_loss_multiplier_logits

        else:
            matryoshka_loss_multiplier_logits = [
                cfg.matryoshka_loss_multiplier_logits
            ] * expected_num_losses

        if len(matryoshka_loss_multiplier_logits) != expected_num_losses:
            raise ValueError(
                f"Expected {expected_num_losses} loss multiplier logits, got {len(matryoshka_loss_multiplier_logits)}"
            )
        self.register_buffer(
            "matryoshka_loss_multiplier_logits",
            torch.tensor(matryoshka_loss_multiplier_logits),
        )
        self.matryoshka_loss_multiplier_logits.requires_grad = True
        self.infrequent_losses = {}

    def get_matryoshka_loss_multipliers(self) -> tuple[float, list[float]]:
        if self.cfg.inner_loss_balancing_enabled:
            base_multipliers = self.matryoshka_loss_multiplier_logits.softmax(
                dim=0
            ).tolist()
            if self.cfg.skip_outer_loss:
                return (1.0, base_multipliers)
            return (base_multipliers[-1], base_multipliers[:-1])
        if isinstance(self.cfg.matryoshka_inner_loss_multipliers, list):
            return (
                self.cfg.outer_loss_multiplier,
                self.cfg.matryoshka_inner_loss_multipliers,
            )
        return (
            self.cfg.outer_loss_multiplier,
            [self.cfg.matryoshka_inner_loss_multipliers] * len(self.matryoshka_steps),
        )

    def decode_partial(
        self,
        feature_acts: torch.Tensor,
        portion: int,
        portion_start: int = 0,
        add_bdec: bool = True,
    ) -> torch.Tensor:
        """Decodes SAE feature activation tensor using part of the decoder"""
        # "... d_sae, d_sae d_in -> ... d_in",
        decoded = (
            self.apply_finetuning_scaling_factor(feature_acts)[:, portion_start:portion]
            @ self.W_dec[portion_start:portion]
        )
        if add_bdec:
            decoded = decoded + self.b_dec
        return decoded

    def skip_main_losses(self) -> bool:
        return self.cfg.skip_outer_loss

    def iterable_decode(
        self,
        feature_acts: torch.Tensor,
        max_stage: int | None = None,
        use_full_width_for_final_portion: bool = False,
        detach_prev: bool = False,
        include_bdec: bool = True,
    ) -> Generator[torch.Tensor, None, None]:
        """Decodes each portion of the SAE and yields the result per stage"""
        decoded = self.b_dec if include_bdec else None
        prev_portion = 0
        acts = self.apply_finetuning_scaling_factor(feature_acts)
        for i, portion in enumerate(self.matryoshka_steps):
            if use_full_width_for_final_portion and i == len(self.matryoshka_steps) - 1:
                portion = self.cfg.d_sae
            if max_stage is not None and i > max_stage:
                break
            current_delta = (
                acts[:, prev_portion:portion] @ self.W_dec[prev_portion:portion]
            )
            if decoded is None:
                decoded = current_delta
            else:
                if detach_prev and prev_portion > 0:
                    decoded = decoded.detach()
                decoded = decoded + current_delta
            prev_portion = portion
            yield decoded

    def inner_reconstruction_loss(
        self,
        partial_sae_error: torch.Tensor,
        matryoshka_step_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        return self.reconstruction_loss_fn(partial_sae_error)

    def training_forward_pass(
        self,
        sae_in: torch.Tensor,
        current_l1_coefficient: float,
        dead_neuron_mask: torch.Tensor | None = None,
    ) -> TrainStepOutput:
        output = super().training_forward_pass(
            sae_in, current_l1_coefficient, dead_neuron_mask
        )
        return self.matryoshka_train_step(output, current_l1_coefficient)

    def weight_feature_acts(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return feature_acts * self.W_dec[: feature_acts.shape[-1]].norm(dim=1)

    def matryoshka_train_step(
        self, output: TrainStepOutput, current_l1_coefficient: float
    ) -> TrainStepOutput:
        feature_acts = output.feature_acts
        sae_in = output.sae_in
        weighted_feature_acts = feature_acts
        if (
            self.cfg.scale_sparsity_penalty_by_decoder_norm
            and self.cfg.include_matryoshka_inner_l1_loss
        ):
            weighted_feature_acts = self.weight_feature_acts(feature_acts)

        # always start with the full SAE loss
        inner_reconstruction_losses = []
        inner_l1_losses = []
        outer_loss_multiplier, inner_loss_multipliers = (
            self.get_matryoshka_loss_multipliers()
        )

        if self.cfg.include_bdec_loss:
            bdec_err = self.b_dec - sae_in
            bdec_loss = self.reconstruction_loss_fn(bdec_err)
            if self.cfg.normalize_reconstruction_losses_by_d_in:
                bdec_loss = bdec_loss / self.cfg.d_in
            inner_reconstruction_losses.append(bdec_loss)

        # add an additional loss for each portion of the full SAE
        prev_portion = 0
        matryoshka_losses = []
        for i, (
            portion,
            partial_sae_out,
            inner_l1_loss_multiplier,
            inner_loss_multiplier,
        ) in enumerate(
            zip(
                self.matryoshka_steps,
                self.iterable_decode(
                    feature_acts,
                    max_stage=self.max_stage(),
                    detach_prev=self.cfg.use_delta_loss,
                    include_bdec=False,
                ),
                self.inner_l1_loss_multipliers,
                inner_loss_multipliers,
            )
        ):
            if inner_loss_multiplier == 0:
                matryoshka_losses.append(output.loss.new_tensor(0.0))
                continue
            partial_sae_out = partial_sae_out + self.b_dec
            partial_sae_error = partial_sae_out - sae_in

            inner_reconstruction_loss = (
                inner_loss_multiplier
                * self.inner_reconstruction_loss(partial_sae_error, i)
            )

            # bart's SAE takes the mean of the loss over the portion of the SAE
            if self.cfg.normalize_reconstruction_losses_by_d_in:
                inner_reconstruction_loss = inner_reconstruction_loss / self.cfg.d_in
            inner_reconstruction_losses.append(inner_reconstruction_loss)

            inner_loss = inner_reconstruction_loss
            if (
                self.cfg.include_matryoshka_inner_l1_loss
                and inner_l1_loss_multiplier > 0
            ):
                portion_start = prev_portion if self.cfg.use_delta_loss else 0
                if self.cfg.architecture == "jumprelu":
                    # this is hacky, technically should be called l0 loss, but whatever. Will fix if jumprelu works
                    inner_l1_loss = partial_l0_loss(
                        hidden_pre=output.hidden_pre,
                        log_threshold=self.log_threshold,
                        bandwidth=self.bandwidth,
                        l0_coefficient=current_l1_coefficient
                        * inner_l1_loss_multiplier
                        * inner_loss_multiplier,
                        portion=portion,
                    )
                else:
                    inner_l1_loss = partial_l1_loss(
                        weighted_feature_acts,
                        current_l1_coefficient
                        * inner_l1_loss_multiplier
                        * inner_loss_multiplier,
                        portion=portion,
                        portion_start=portion_start,
                        lp_norm=self.cfg.lp_norm,
                    ) * self.inner_l1_multiplier(i)
                inner_l1_losses.append(inner_l1_loss)
                inner_loss = inner_loss + inner_l1_loss
            matryoshka_losses.append(inner_loss / inner_loss_multiplier)
            prev_portion = portion
        if len(inner_reconstruction_losses) > 0:
            output.losses["inner_recons_loss"] = torch.stack(
                inner_reconstruction_losses
            ).sum()
        else:
            output.losses["inner_recons_loss"] = output.loss.new_tensor(0.0)
        inner_loss = output.losses["inner_recons_loss"]

        if len(inner_l1_losses) > 0:
            output.losses["inner_l1_loss"] = torch.stack(inner_l1_losses).sum()
            inner_loss = inner_loss + output.losses["inner_l1_loss"]

        output.loss = inner_loss
        output.losses["mse_loss"] = output.losses["mse_loss"] * outer_loss_multiplier
        if "l1_loss" in output.losses:
            output.losses["l1_loss"] = output.losses["l1_loss"] * outer_loss_multiplier
        if "l0_loss" in output.losses:
            output.losses["l0_loss"] = output.losses["l0_loss"] * outer_loss_multiplier
        if self.cfg.normalize_reconstruction_losses_by_d_in:
            output.losses["mse_loss"] = output.losses["mse_loss"] / self.cfg.d_in
        if self.cfg.skip_outer_loss:
            del output.losses["mse_loss"]
            if "l1_loss" in output.losses:
                del output.losses["l1_loss"]
            if "l0_loss" in output.losses:
                del output.losses["l0_loss"]
        else:
            outer_loss = output.losses["mse_loss"]
            output.loss = output.loss + output.losses["mse_loss"]
            if "l1_loss" in output.losses:
                outer_loss = outer_loss + output.losses["l1_loss"]
                output.loss = output.loss + output.losses["l1_loss"]
            if "l0_loss" in output.losses:
                outer_loss = outer_loss + output.losses["l0_loss"]
                output.loss = output.loss + output.losses["l0_loss"]
            matryoshka_losses.append(
                outer_loss / outer_loss_multiplier
            )  # we want this loss before the outer loss multiplier is applied

        # Bart's SAE normalizes the loss by the number of steps
        # and also divides the mse and l1 losses by the size of the SAE
        if self.cfg.normalize_losses_by_num_matryoshka_steps:
            denom = len(self.matryoshka_steps)
            if not self.cfg.skip_outer_loss:
                denom += 1
            if self.cfg.include_bdec_loss:
                denom += 1
            for loss_name, loss in output.losses.items():
                output.losses[loss_name] = loss / denom
            output.loss = output.losses["inner_recons_loss"]
            if "mse_loss" in output.losses:
                output.loss = output.loss + output.losses["mse_loss"]
            if "l1_loss" in output.losses:
                output.loss = output.loss + output.losses["l1_loss"]
            if "l0_loss" in output.losses:
                output.loss = output.loss + output.losses["l0_loss"]
            if "inner_l1_loss" in output.losses:
                output.loss = output.loss + output.losses["inner_l1_loss"]

        if (
            self.cfg.inner_loss_balancing_enabled
            and self.step_num % self.cfg.update_inner_multipliers_every_n_steps == 0
        ):
            # we still want to log the kurtosis loss even if we don't update the coefficients
            perform_update = (
                self.step_num >= self.cfg.update_inner_multipliers_start_step
            )
            kurtosis_loss = self.update_inner_coefficients(
                matryoshka_losses, perform_update=perform_update
            )
            # hacky way to log kurtosis loss, even though we don't update it often
            self.infrequent_losses["kurtosis_loss"] = kurtosis_loss
        # not exactly losses, but useful for logging
        for i, coeff in enumerate(inner_loss_multipliers):
            output.losses[f"inner_coeff.{i}"] = coeff
        if self.cfg.inner_loss_balancing_enabled:
            for i, logit in enumerate(self.matryoshka_loss_multiplier_logits):
                output.losses[f"inner_coeff_logits.{i}"] = logit
        output.losses["outer_loss_multiplier"] = outer_loss_multiplier
        for loss_name, loss in self.infrequent_losses.items():
            output.losses[loss_name] = loss

        return output

    def update_inner_coefficients(
        self,
        matryoshka_losses: list[torch.Tensor],
        perform_update: bool = True,
    ) -> torch.Tensor:
        multipliers = self.matryoshka_loss_multiplier_logits.softmax(dim=0)
        kurtosis_loss = self._calculate_kurtosis_loss(multipliers, matryoshka_losses)
        if perform_update:
            multiplier_grads = torch.autograd.grad(
                kurtosis_loss, self.matryoshka_loss_multiplier_logits, allow_unused=True
            )[0]
            with torch.no_grad():
                self.matryoshka_loss_multiplier_logits -= (
                    self.cfg.update_inner_multipliers_lr * multiplier_grads
                ).detach()
        return kurtosis_loss.detach()

    def _calculate_kurtosis_loss(
        self, multipliers: torch.Tensor, matryoshka_losses: list[torch.Tensor]
    ) -> torch.Tensor:
        W_enc_grad = torch.zeros_like(self.W_enc)
        assert len(multipliers) == len(matryoshka_losses)
        for multiplier, loss in zip(multipliers, matryoshka_losses):
            W_enc_grad = W_enc_grad + (
                multiplier
                * torch.autograd.grad(
                    loss, self.W_enc, allow_unused=True, retain_graph=True
                )[0].detach()
            )
        # simulate gradient clipping
        W_enc_grad = W_enc_grad / (W_enc_grad.norm() + 1e-6)
        modified_W_enc = self.W_enc.detach() - self.cfg.lr * W_enc_grad
        norm_w_enc = torch.nn.functional.normalize(modified_W_enc, dim=0)
        prev_portion = 0
        portion_mean_kurtosis = []
        portions: list[int | None] = [*self.matryoshka_steps]
        if not self.cfg.skip_outer_loss:
            portions.append(None)
        for portion in portions:
            kurtosis = _enc_cos_kurtosis(norm_w_enc[:, prev_portion:portion])
            portion_mean_kurtosis.append(kurtosis.mean())
            prev_portion = portion

        # we want minimum kurtosis
        kurtosis_loss = torch.stack(portion_mean_kurtosis).sum()
        return kurtosis_loss

    def max_stage(self) -> int | None:  # noqa: ARG002
        return None

    def inner_l1_multiplier(self, i: int) -> float:  # noqa: ARG002
        return 1.0

    def calculate_topk_aux_loss(
        self,
        sae_in: torch.Tensor,
        sae_out: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.cfg.use_matryoshka_aux_loss:
            return super().calculate_topk_aux_loss(
                sae_in, sae_out, hidden_pre, dead_neuron_mask
            )

        # calculate a separate aux loss for each new matryoshka portion of the SAE
        if dead_neuron_mask is not None and int(dead_neuron_mask.sum()) > 0:
            k_aux = sae_in.shape[-1] // 2
            acts = self.activation_fn(hidden_pre)
            prev_portion = 0
            aux_losses = []
            for i, (portion, partial_sae_out) in enumerate(
                zip(
                    self.matryoshka_steps,
                    self.iterable_decode(
                        acts,
                        max_stage=self.max_stage(),
                        use_full_width_for_final_portion=True,
                    ),
                )
            ):
                if i == len(self.matryoshka_steps) - 1:
                    portion = hidden_pre.shape[-1]
                partial_dead_neuron_mask = dead_neuron_mask[prev_portion:portion]
                partial_num_dead = int(partial_dead_neuron_mask.sum())
                if partial_num_dead == 0:
                    continue

                partial_k_aux = min(k_aux, partial_num_dead)
                partial_hidden_pre = hidden_pre[:, prev_portion:portion]
                residual = (sae_in - partial_sae_out).detach()
                auxk_acts = _calculate_topk_aux_acts(
                    k_aux=partial_k_aux,
                    hidden_pre=partial_hidden_pre,
                    dead_neuron_mask=partial_dead_neuron_mask,
                )

                # Reduce the scale of the loss if there are a small number of dead latents
                scale = min(partial_num_dead / partial_k_aux, 1.0)

                # Encourage the top ~50% of dead latents to predict the residual of the
                # top k living latents
                recons = auxk_acts @ self.W_dec[prev_portion:portion] + self.b_dec
                auxk_loss = (recons - residual).pow(2).sum(dim=-1).mean()
                aux_losses.append(scale * auxk_loss)
                prev_portion = portion
            stacked_losses = torch.stack(aux_losses)
            return (
                stacked_losses.mean()
                if self.cfg.normalize_losses_by_num_matryoshka_steps
                else stacked_losses.sum()
            )
        return sae_out.new_tensor(0.0)

    def process_state_dict_for_saving(self, state_dict: dict[str, Any]) -> None:
        state_dict["step_num"] = torch.tensor(self.step_num)

    def process_state_dict_for_saving_final(self, state_dict: dict[str, Any]) -> None:
        self.process_state_dict_for_saving(state_dict)
        if "step_num" in state_dict:
            del state_dict["step_num"]
        if "matryoshka_loss_multiplier_logits" in state_dict:
            del state_dict["matryoshka_loss_multiplier_logits"]

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        if "step_num" in state_dict:
            self.step_num = state_dict["step_num"].item()
            del state_dict["step_num"]  # type: ignore
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def _enc_cos_kurtosis(norm_enc: torch.Tensor) -> torch.Tensor:
    enc_cos = norm_enc.T @ norm_enc
    return _calculate_kurtosis_per_row(enc_cos)


def _calculate_kurtosis_per_row(matrix: torch.Tensor) -> torch.Tensor:
    # Calculate mean and std per row
    # keepdim=True preserves dimensions for broadcasting
    means = matrix.mean(dim=1, keepdim=True)
    stds = matrix.std(dim=1, keepdim=True)

    # Calculate standardized values per row
    standardized = (matrix - means) / stds

    # Calculate kurtosis (4th moment)
    # Subtract 3 for excess kurtosis (normal distribution has kurtosis of 3)
    kurtosis = torch.mean(standardized**4, dim=1) - 3

    return kurtosis


class BatchTopkMatryoshkaSAE(BatchTopkSAE, MatryoshkaSAE):
    def __init__(
        self,
        cfg: MatryoshkaSAEConfig,
    ):
        super().__init__(cfg)

    def training_forward_pass(
        self,
        sae_in: torch.Tensor,
        current_l1_coefficient: float,
        dead_neuron_mask: torch.Tensor | None = None,
    ) -> TrainStepOutput:
        output = BatchTopkSAE.training_forward_pass(
            self, sae_in, current_l1_coefficient, dead_neuron_mask
        )
        aux_loss = output.losses["auxiliary_reconstruction_loss"]
        output = self.matryoshka_train_step(output, current_l1_coefficient)
        output.loss = aux_loss + output.loss
        return output

    def process_state_dict_for_saving_final(self, state_dict: dict[str, Any]) -> None:
        BatchTopkSAE.process_state_dict_for_saving_final(self, state_dict)
        MatryoshkaSAE.process_state_dict_for_saving_final(self, state_dict)


def partial_l1_loss(
    weighted_feature_acts: torch.Tensor,
    l1_coefficient: float,
    portion: int | None,
    portion_start: int = 0,
    lp_norm: float = 1.0,
) -> torch.Tensor:
    sparsity = weighted_feature_acts[:, portion_start:portion].norm(p=lp_norm, dim=-1)
    return l1_coefficient * sparsity.mean()


def partial_l0_loss(
    hidden_pre: torch.Tensor,
    log_threshold: torch.Tensor,
    bandwidth: float,
    l0_coefficient: float,
    portion: int | None,
) -> torch.Tensor:
    threshold = torch.exp(log_threshold[:portion])
    l0 = torch.sum(Step.apply(hidden_pre[:, :portion], threshold, bandwidth), dim=-1)  # type: ignore
    l0_loss = (l0_coefficient * l0).mean()
    return l0_loss


def loss_fn(
    loss_fn_name: Literal["L2", "MSE"],
    l2_power: float = 1.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if loss_fn_name == "L2":
        return l2_loss(l2_power)
    elif loss_fn_name == "MSE":
        return mse_loss
    else:
        raise ValueError(f"Invalid loss function name: {loss_fn_name}")


def mse_loss(sae_error: torch.Tensor) -> torch.Tensor:
    return sae_error.pow(2).sum(dim=-1).mean()


def l2_loss(l2_power: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def _l2_loss_fn(sae_error: torch.Tensor) -> torch.Tensor:
        base_loss = sae_error.norm(p=2, dim=-1)
        if l2_power != 1:
            base_loss = base_loss.pow(l2_power)
        return base_loss.mean()

    return _l2_loss_fn
