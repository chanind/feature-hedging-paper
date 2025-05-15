from dataclasses import dataclass
from pathlib import Path

from dataclasses_json import DataClassJsonMixin

from hedging_paper.build_sae_dashboard import BuildSAEDashboardOptions
from hedging_paper.saes.matryoshka_sae import (
    BatchTopkMatryoshkaSAE,
    MatryoshkaSAEConfig,
    MatryoshkaSAERunnerConfig,
)
from hedging_paper.train_sae import output_template, train_eval_and_save_sae


@dataclass
class TrainBatchTopkMatryoshkaSaeOptions(DataClassJsonMixin):
    sae_cfg: MatryoshkaSAERunnerConfig
    output_path: str | Path
    shared_path: str | Path
    dashboard_cfg: BuildSAEDashboardOptions | None = None
    run_evals: bool = True


def train_batch_topk_matryoshka_sae(
    cfg: TrainBatchTopkMatryoshkaSaeOptions,
):
    sae = BatchTopkMatryoshkaSAE(
        MatryoshkaSAEConfig.from_sae_runner_config(cfg.sae_cfg)
    )
    stats = train_eval_and_save_sae(
        sae,
        cfg.sae_cfg,
        output_path=output_template(cfg.output_path),
        shared_path=cfg.shared_path,
        dashboard_cfg=cfg.dashboard_cfg,
        run_evals=cfg.run_evals,
    )
    return sae, stats
