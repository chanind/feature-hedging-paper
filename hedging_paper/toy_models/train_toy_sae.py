from typing import Callable

import torch
from sae_lens.training.geometric_median import compute_geometric_median
from sae_lens.training.sae_trainer import SAETrainer
from tqdm import tqdm

from hedging_paper.saes.base_sae import BaseSAERunnerConfig, TrainingSAE
from hedging_paper.toy_models.toy_model import ToyModel
from hedging_paper.util import DEFAULT_DEVICE


# We only need a way to get `next_batch()` from the ActivationsStore, but the real
# ActivationsStore expects tokens and a LLM rather than toy activations. For our
# purposes, this just implements the important interface to use our feature generator
class FakeActivationsStore:
    def __init__(self, model, generate_batch_fn, batch_size: int):
        self.model = model
        self.batch_size = batch_size
        self.generate_batch_fn = generate_batch_fn
        self.estimated_norm_scaling_factor = None

    def set_norm_scaling_factor_if_needed(self):
        pass

    @torch.no_grad()
    def next_batch(self):
        # the middle param is always 1 in SAELens, I think for legacy reasons
        return self.model(self.generate_batch_fn(self.batch_size)).unsqueeze(1)


# Ignore saving checkpoints, the toy models train very fast
def _save_checkpoint(
    trainer: SAETrainer,
    checkpoint_name: int | str,
    wandb_aliases: list[str] | None = None,
):
    pass


# this is copied from SAELens sae_training_runner.py. This should probably not be in the runner
def _init_sae_group_b_decs(
    sae: TrainingSAE,
    cfg: BaseSAERunnerConfig,
    store: FakeActivationsStore,
) -> None:
    if cfg.b_dec_init_method == "geometric_median":
        layer_acts = store.next_batch().detach()[:, 0, :]
        # get geometric median of the activations if we're using those.
        median = compute_geometric_median(
            layer_acts,
            maxiter=100,
        ).median
        sae.initialize_b_dec_with_precalculated(median)  # type: ignore
    elif cfg.b_dec_init_method == "mean":
        layer_acts = store.next_batch().detach().cpu()[:, 0, :]
        sae.initialize_b_dec_with_mean(layer_acts)  # type: ignore


def train_toy_sae(
    sae: TrainingSAE,
    toy_model: ToyModel,
    activations_batch_provider: Callable[[int], torch.Tensor],
    lr: float = 3e-4,
    normalize_sae_decoder: bool = False,
    training_tokens=100_000_000,
    lr_warm_up_steps: int = 0,
    lr_decay_steps: int = 0,
    log_to_wandb: bool = False,
    device: torch.device = DEFAULT_DEVICE,
    sa_cooling_rate: float = 0.99,
    sa_initial_temp: float = 1.0,
    sa_final_temp: float = 0.01,
    sa_step_size: float = 0.01,
    train_batch_size_tokens: int = 4096,
) -> None:
    tqdm._instances.clear()  # type: ignore

    cfg = BaseSAERunnerConfig(
        context_size=sae.cfg.context_size,
        train_batch_size_tokens=train_batch_size_tokens,
        d_in=sae.cfg.d_in,
        d_sae=sae.cfg.d_sae,
        device=str(device),
        training_tokens=training_tokens,
        eval_every_n_wandb_logs=99999999999,
        l1_coefficient=sae.cfg.l1_coefficient,
        normalize_sae_decoder=normalize_sae_decoder,
        scale_sparsity_penalty_by_decoder_norm=not normalize_sae_decoder,
        lr=lr,
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_decay_steps,
        log_to_wandb=log_to_wandb,
        b_dec_init_method="zeros",
        sa_cooling_rate=sa_cooling_rate,
        sa_initial_temp=sa_initial_temp,
        sa_final_temp=sa_final_temp,
        sa_step_size=sa_step_size,
    )
    toy_model.eval()
    store = FakeActivationsStore(
        toy_model, activations_batch_provider, sae.cfg.context_size
    )
    _init_sae_group_b_decs(sae, cfg, store)
    trainer = SAETrainer(
        model=toy_model,
        sae=sae,
        activation_store=store,  # type: ignore
        cfg=cfg,
        save_checkpoint_fn=_save_checkpoint,
    )
    trainer.fit()
