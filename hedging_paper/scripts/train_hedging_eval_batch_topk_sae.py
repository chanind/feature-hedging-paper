from dataclasses import dataclass, replace
from pathlib import Path

from dataclasses_json import DataClassJsonMixin
from sae_lens import SAE

from hedging_paper.build_sae_dashboard import BuildSAEDashboardOptions
from hedging_paper.evals.hedging_eval import HedgingEval
from hedging_paper.evals.run_all_evals import run_eval
from hedging_paper.saes.base_sae import BaseSAEConfig, BaseSAERunnerConfig
from hedging_paper.saes.batch_topk_sae import BatchTopkSAE
from hedging_paper.train_sae import train_eval_and_save_sae
from hedging_paper.util import DEFAULT_DEVICE_STR


@dataclass
class TrainHedgingEvalBatchTopkSaeOptions(DataClassJsonMixin):
    base_sae_cfg: BaseSAERunnerConfig
    base_output_path: str | Path
    new_latents: int
    shared_path: str | Path
    dashboard_cfg: BuildSAEDashboardOptions | None = None
    run_evals: bool = False
    run_hedging_eval: bool = True


def train_hedging_eval_batch_topk_sae(
    cfg: TrainHedgingEvalBatchTopkSaeOptions,
):
    initial_output_path = Path(cfg.base_output_path) / "initial"
    control_output_path = Path(cfg.base_output_path) / "control"
    extended_output_path = Path(cfg.base_output_path) / "extended"

    assert cfg.base_sae_cfg.d_sae is not None, "d_sae must be set"

    initial_sae_cfg = replace(
        cfg.base_sae_cfg,
        extend_sae_latents=0,
        extend_sae_path=None,
        run_name=f"initial-{cfg.base_sae_cfg.run_name}",
    )
    control_sae_cfg = replace(
        cfg.base_sae_cfg,
        extend_sae_latents=0,
        extend_sae_path=str(initial_output_path),
        run_name=f"control-{cfg.base_sae_cfg.run_name}",
    )
    extended_sae_cfg = replace(
        cfg.base_sae_cfg,
        extend_sae_latents=cfg.new_latents,
        extend_sae_path=str(initial_output_path),
        run_name=f"extended-{cfg.base_sae_cfg.run_name}",
    )

    if (initial_output_path / "cfg.json").exists():
        print(f"Initial SAE already exists at {initial_output_path}, skipping...")
    else:
        sae = BatchTopkSAE(BaseSAEConfig.from_sae_runner_config(initial_sae_cfg))
        train_eval_and_save_sae(
            sae,
            initial_sae_cfg,
            output_path=initial_output_path,
            shared_path=cfg.shared_path,
            dashboard_cfg=cfg.dashboard_cfg,
            run_evals=cfg.run_evals,
        )

    if (control_output_path / "cfg.json").exists():
        print(f"Control SAE already exists at {control_output_path}, skipping...")
    else:
        sae = BatchTopkSAE(BaseSAEConfig.from_sae_runner_config(control_sae_cfg))
        train_eval_and_save_sae(
            sae,
            control_sae_cfg,
            output_path=control_output_path,
            shared_path=cfg.shared_path,
            dashboard_cfg=cfg.dashboard_cfg,
            run_evals=cfg.run_evals,
        )

    if (extended_output_path / "cfg.json").exists():
        print(f"Extended SAE already exists at {extended_output_path}, skipping...")
    else:
        sae = BatchTopkSAE(BaseSAEConfig.from_sae_runner_config(extended_sae_cfg))
        train_eval_and_save_sae(
            sae,
            extended_sae_cfg,
            output_path=extended_output_path,
            shared_path=cfg.shared_path,
            dashboard_cfg=cfg.dashboard_cfg,
            run_evals=cfg.run_evals,
        )

    if cfg.run_hedging_eval:
        sae = SAE.load_from_pretrained(
            str(extended_output_path),
            device=DEFAULT_DEVICE_STR,
        )
        eval = HedgingEval()
        run_eval(
            eval,
            sae,
            results_dir=extended_output_path,
            shared_dir=Path(cfg.shared_path),
        )
