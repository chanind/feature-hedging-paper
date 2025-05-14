from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig
from sae_bench.evals.sparse_probing.main import run_eval
from sae_lens import SAE

from hedging_paper.evals.eval import Eval
from hedging_paper.evals.util import glob_matches
from hedging_paper.util import DEFAULT_DEVICE_STR


@dataclass
class SparseProbingEvalOptions:
    device: str | None = None
    llm_dtype: str = "float32"
    llm_batch_size: int = 128


class SparseProbingEval(Eval):
    def __init__(self, options: SparseProbingEvalOptions | None = None):
        self.options = options or SparseProbingEvalOptions()

    def has_eval_run(self, results_dir: Path) -> bool:
        return glob_matches(
            results_dir, "**/saebench_sparse_probing_custom_sae_eval_results.json"
        )

    def run(self, sae: SAE, results_dir: Path, shared_dir: Path) -> None:  # noqa: ARG002
        eval_saebench_sparse_probing(
            sae, self.options, results_dir, artifacts_path=shared_dir / "artifacts"
        )


def eval_saebench_sparse_probing(
    sae: SAE,
    options: SparseProbingEvalOptions,
    results_dir: str | Path,
    artifacts_path: str | Path,
) -> dict[str, Any]:
    sae.fold_W_dec_norm()
    cfg = SparseProbingEvalConfig(
        model_name=sae.cfg.model_name,
        llm_dtype=options.llm_dtype,
        llm_batch_size=options.llm_batch_size,
        k_values=[1, 2, 5, 10],
    )
    return run_eval(
        config=cfg,
        device=options.device or DEFAULT_DEVICE_STR,
        output_path=str(results_dir),
        selected_saes=[("saebench_sparse_probing", sae)],
        artifacts_path=str(artifacts_path),
    )["saebench_sparse_probing_custom_sae"]
