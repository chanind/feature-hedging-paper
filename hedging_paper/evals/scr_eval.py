from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sae_bench.evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig
from sae_bench.evals.scr_and_tpp.main import run_eval
from sae_lens import SAE

from hedging_paper.evals.eval import Eval
from hedging_paper.evals.util import glob_matches
from hedging_paper.util import DEFAULT_DEVICE_STR


@dataclass
class SCREvalOptions:
    device: str | None = None
    llm_dtype: str = "float32"
    llm_batch_size: int = 128


class SCREval(Eval):
    def __init__(self, options: SCREvalOptions | None = None):
        self.options = options or SCREvalOptions()

    def has_eval_run(self, results_dir: Path) -> bool:
        return glob_matches(results_dir, "**/saebench_scr_custom_sae_eval_results.json")

    def run(self, sae: SAE, results_dir: Path, shared_dir: Path) -> None:  # noqa: ARG002
        eval_saebench_scr(
            sae, self.options, results_dir, artifacts_path=shared_dir / "artifacts"
        )


def eval_saebench_scr(
    sae: SAE,
    options: SCREvalOptions,
    results_dir: str | Path,
    artifacts_path: str | Path,
) -> dict[str, Any]:
    sae.fold_W_dec_norm()
    cfg = ScrAndTppEvalConfig(
        model_name=sae.cfg.model_name,
        llm_dtype=options.llm_dtype,
        llm_batch_size=options.llm_batch_size,
        perform_scr=True,
        n_values=[2, 5, 10],
    )
    return run_eval(
        config=cfg,
        device=options.device or DEFAULT_DEVICE_STR,
        output_path=str(results_dir),
        selected_saes=[("saebench_scr", sae)],
        artifacts_path=str(artifacts_path),
    )["saebench_scr_custom_sae"]
