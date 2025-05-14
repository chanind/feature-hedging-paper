import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.evals.autointerp.main import run_eval
from sae_lens import SAE

from hedging_paper.evals.eval import Eval
from hedging_paper.evals.util import glob_matches
from hedging_paper.util import DEFAULT_DEVICE_STR


@dataclass
class AutointerpEvalOptions:
    device: str | None = None
    llm_dtype: str = "float32"
    llm_batch_size: int = 128


class AutointerpEval(Eval):
    def __init__(self, options: AutointerpEvalOptions | None = None):
        self.options = options or AutointerpEvalOptions()

    def has_eval_run(self, results_dir: Path) -> bool:
        return glob_matches(
            results_dir, "**/saebench_autointerp_custom_sae_eval_results.json"
        )

    def run(self, sae: SAE, results_dir: Path, shared_dir: Path) -> None:  # noqa: ARG002
        eval_saebench_autointerp(
            sae, self.options, results_dir, artifacts_path=shared_dir / "artifacts"
        )


def eval_saebench_autointerp(
    sae: SAE,
    options: AutointerpEvalOptions,
    results_dir: str | Path,
    artifacts_path: str | Path,
    api_key: str | None = None,
) -> dict[str, Any]:
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY is not set")
    sae.fold_W_dec_norm()
    cfg = AutoInterpEvalConfig(
        model_name=sae.cfg.model_name,
        llm_dtype=options.llm_dtype,
        llm_batch_size=options.llm_batch_size,
    )
    return run_eval(
        config=cfg,
        device=options.device or DEFAULT_DEVICE_STR,
        output_path=str(results_dir),
        selected_saes=[("saebench_autointerp", sae)],
        api_key=api_key,
        artifacts_path=str(artifacts_path),
    )
