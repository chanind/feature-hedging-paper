import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from sae_lens import SAE

from hedging_paper.evals.eval import Eval
from hedging_paper.evals.util import glob_matches
from hedging_paper.sparse_probing.parts_of_speech import collect_pos_activations
from hedging_paper.sparse_probing.probing import (
    train_and_evaluate_k_sparse_probes,
)
from hedging_paper.util import DEFAULT_DEVICE_STR

KS = [1, 2, 5, 10]
POS_LABELS = [
    "TO",
    "IN",
    "DT",
    "CC",
    "NNS",
    "PRP",
    "POS",
]


@dataclass
class POSSparseProbingEvalOptions:
    device: str | None = None
    max_rescale_samples: int | None = 50_000
    train_ratio: float = 0.7


class POSSparseProbingEval(Eval):
    def __init__(self, options: POSSparseProbingEvalOptions | None = None):
        self.options = options or POSSparseProbingEvalOptions()

    def has_eval_run(self, results_dir: Path) -> bool:
        return glob_matches(results_dir, "**/pos_sparse_probing_results.json")

    def run(self, sae: SAE, results_dir: Path, shared_dir: Path) -> None:  # noqa: ARG002
        eval_pos_sparse_probing(
            sae, self.options, results_dir, artifacts_path=shared_dir / "artifacts"
        )


@torch.inference_mode()
def eval_pos_sparse_probing(
    sae: SAE,
    options: POSSparseProbingEvalOptions,
    results_dir: str | Path,
    artifacts_path: str | Path,
) -> dict[str, Any]:
    sae.fold_W_dec_norm()
    pos_acts = collect_pos_activations(
        hook_name=sae.cfg.hook_name,
        model_name=sae.cfg.model_name,
        cache_dir=Path(artifacts_path) / "parts_of_speech",
        from_pretrained_kwargs=sae.cfg.model_from_pretrained_kwargs,
        device=options.device or DEFAULT_DEVICE_STR,
    )
    sae_acts = sae.encode(pos_acts.activations.to(sae.device))
    results = {}
    for pos_label in POS_LABELS:
        labels = torch.zeros(pos_acts.activations.shape[0], dtype=torch.bool)
        labels[pos_acts.pos_indices[pos_label]] = 1
        result = train_and_evaluate_k_sparse_probes(
            latent_activations=sae_acts,
            labels=labels,
            k_values=KS,
            train_ratio=options.train_ratio,
            rescale_before_mean_diff=True,
            max_rescale_samples=options.max_rescale_samples,
        )
        results[pos_label] = {k: v.to_dict() for k, v in result.items()}
    results_path = Path(results_dir) / "pos_sparse_probing_results.json"
    with open(results_path, "w") as f:
        json.dump({"results": results}, f)
    return results
