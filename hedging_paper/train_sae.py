import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, cast

import torch
import wandb
from datasets import Dataset
from sae_lens import SAE, SAETrainingRunner
from sae_lens.evals import run_evals
from sae_lens.training.sae_trainer import SAETrainer
from sae_lens.training.training_sae import SAE_WEIGHTS_PATH
from safetensors.torch import load_file
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from hedging_paper.build_sae_dashboard import (
    BuildSAEDashboardOptions,
    build_sae_dashboard,
)
from hedging_paper.evals.run_all_evals import run_all_evals
from hedging_paper.saes.base_sae import BaseSAE, BaseSAERunnerConfig
from hedging_paper.util import DEFAULT_DEVICE_STR, millify

DEFAULT_OUTPUT_TEMPLATE = "{hook_name}/w-{width}/t-{tokens_trained}/l0-{l0:.1f}"


@dataclass
class SAEStats:
    sparsity: torch.Tensor
    l0: float
    width: int
    hook_name: str
    tokens_trained: int
    all_metrics: dict[str, Any]


def output_template(
    base_path: str | Path, template: str = DEFAULT_OUTPUT_TEMPLATE
) -> Callable[[SAEStats], Path]:
    """Helper to create a callable to format the output path from the stats."""
    return lambda stats: Path(base_path) / template.format(
        hook_name=stats.hook_name,
        width=stats.width,
        l0=stats.l0,
        tokens_trained=millify(stats.tokens_trained),
    )


def save_checkpoint(
    trainer: SAETrainer,
    base_path: Path,
) -> None:
    """
    Copied from the base trainer class. This is a hacky way to let the SAE save a checkpoint itself.
    """
    base_path.mkdir(exist_ok=True, parents=True)

    trainer.activations_store.save(
        str(base_path / "activations_store_state.safetensors")
    )

    # Save training state including optimizer
    torch.save(
        {
            "optimizer": trainer.optimizer.state_dict(),
            "lr_scheduler": trainer.lr_scheduler.state_dict(),
            "l1_scheduler": trainer.l1_scheduler.state_dict(),
            "n_training_tokens": trainer.n_training_tokens,
            "n_training_steps": trainer.n_training_steps,
            "act_freq_scores": trainer.act_freq_scores,
            "n_forward_passes_since_fired": trainer.n_forward_passes_since_fired,
            "n_frac_active_tokens": trainer.n_frac_active_tokens,
            "started_fine_tuning": trainer.started_fine_tuning,
        },
        str(base_path / "training_state.pt"),
    )

    if trainer.sae.cfg.normalize_sae_decoder:
        trainer.sae.set_decoder_norm_to_unit_norm()

    weights_path, cfg_path, sparsity_path = trainer.sae.save_model(
        str(base_path),
        trainer.log_feature_sparsity,
    )

    # let's over write the cfg file with the trainer cfg, which is a super set of the original cfg.
    # and should not cause issues but give us more info about SAEs we trained in SAE Lens.
    config = trainer.cfg.to_dict()
    with open(cfg_path, "w") as f:
        json.dump(config, f)


class ExtendedSAETrainingRunner(SAETrainingRunner):
    cfg: BaseSAERunnerConfig  # type: ignore
    sae: BaseSAE  # type: ignore

    def run_and_eval(self) -> SAEStats:
        """
        Run the training of the SAE and evaluate.
        """

        if self.cfg.from_pretrained_path is not None:
            _load_pretrained_weights(
                self.sae,
                self.cfg.from_pretrained_path,
                self.cfg.device,
            )

        if self.cfg.extend_sae_path is not None:
            _load_pretrained_weights(
                self.sae,
                self.cfg.extend_sae_path,
                self.cfg.device,
            )
        # extend before trying to load from checkpoint
        if self.cfg.extend_sae_latents is not None and self.cfg.extend_sae_latents > 0:
            self.sae.extend_sae(self.cfg.extend_sae_latents)
            self.cfg.d_sae = self.sae.W_dec.shape[0]

        # ---- copied from SAELens directly ---
        if self.cfg.log_to_wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                config=cast(Any, self.cfg),
                name=self.cfg.run_name,
                id=self.cfg.wandb_id,
            )

        trainer = SAETrainer(
            model=self.model,
            sae=self.sae,
            activation_store=self.activations_store,
            save_checkpoint_fn=self.save_checkpoint,
            cfg=self.cfg,
        )

        self.sae.trainer = trainer

        # --- end copied section ---

        if self.cfg.fast_forward_steps > 0:
            trainer.n_training_steps = self.cfg.fast_forward_steps
            trainer.n_training_tokens = (
                self.cfg.fast_forward_steps * self.cfg.train_batch_size_tokens
            )
            print(
                f"Fast forwarding {self.cfg.fast_forward_steps} steps, {trainer.n_training_tokens} tokens"
            )
            for _ in tqdm(range(self.cfg.fast_forward_steps)):
                trainer.activations_store.next_batch()

        # ---- copied from SAELens directly ---

        self._compile_if_needed()
        sae = self.run_trainer_with_interruption_handling(trainer)

        if self.cfg.log_to_wandb:
            wandb.finish()

        # --- end copied section ---

        sae.fold_W_dec_norm()

        # skip special tokens in evals
        ignore_tokens: set[int | None] = (  # this typing seems off in saelens...
            set(self.model.tokenizer.all_special_ids) if self.model.tokenizer else set()  # type: ignore
        )
        all_metrics = run_evals(
            sae,
            self.activations_store,
            self.model,
            trainer.trainer_eval_config,
            ignore_tokens=ignore_tokens,
        )[0]

        stats = SAEStats(
            sparsity=trainer.log_feature_sparsity,
            l0=all_metrics["sparsity"]["l0"],
            width=sae.cfg.d_sae,
            hook_name=self.cfg.hook_name,
            all_metrics=all_metrics,
            tokens_trained=trainer.n_training_tokens,
        )

        return stats


def train_sae(
    sae: BaseSAE,
    cfg: BaseSAERunnerConfig,
    override_model: HookedTransformer | None = None,
    override_dataset: Dataset | None = None,
) -> SAEStats:
    runner = ExtendedSAETrainingRunner(
        cfg,
        override_sae=sae,
        override_model=override_model,
        override_dataset=override_dataset,
    )
    sae_stats = runner.run_and_eval()
    return sae_stats


def train_eval_and_save_sae(
    sae: BaseSAE,
    cfg: BaseSAERunnerConfig,
    output_path: Path | str | Callable[[SAEStats], Path | str],
    shared_path: Path | str,
    dashboard_cfg: BuildSAEDashboardOptions | None = None,
    run_evals: bool = True,
) -> SAEStats:
    checkpoint_path = Path(shared_path) / "checkpoints" / hash_sae_cfg(cfg)
    cfg.checkpoint_path = str(checkpoint_path)
    sae_stats = train_sae(sae, cfg)
    if isinstance(output_path, Callable):
        output_path = output_path(sae_stats)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    sae.save_model_final(output_path, sparsity=sae_stats.sparsity)

    stats_dict = asdict(sae_stats)
    del stats_dict["sparsity"]
    with open(output_path / "sae_stats.json", "w") as f:
        json.dump(stats_dict, f)
    with open(checkpoint_path / "sae_stats.json", "w") as f:
        json.dump(stats_dict, f)
    with open(checkpoint_path / "output_path.txt", "w") as f:
        f.write(str(output_path))

    # load the SAE as it will be loaded by users and run more evals on it
    loaded_sae = SAE.load_from_pretrained(str(output_path), device=DEFAULT_DEVICE_STR)
    loaded_sae.fold_W_dec_norm()

    if run_evals:
        run_all_evals(loaded_sae, output_path, Path(shared_path))

    if dashboard_cfg is not None:
        build_sae_dashboard(loaded_sae, output_path / "dashboard", dashboard_cfg)

    return sae_stats


def hash_sae_cfg(cfg: BaseSAERunnerConfig) -> str:
    fields = cfg.to_dict()
    fields.pop("run_name")
    fields.pop("checkpoint_path")
    return hashlib.sha256(json.dumps(fields).encode()).hexdigest()


def find_latest_checkpoint(checkpoints_path: Path) -> str | None:
    if not checkpoints_path.exists():
        return None
    # Only consider directories, not files
    all_checkpoints = [p for p in checkpoints_path.glob("*") if p.is_dir()]
    if not all_checkpoints:
        return None
    return str(max(all_checkpoints, key=lambda x: x.stat().st_mtime))


def _load_pretrained_weights(
    sae: BaseSAE,
    pretrained_path: str,
    device: str,
) -> None:
    weight_path = os.path.join(pretrained_path, SAE_WEIGHTS_PATH)
    state_dict = load_file(filename=weight_path, device=device)
    sae.process_state_dict_for_loading(state_dict)
    sae.load_state_dict(state_dict, strict=False)
