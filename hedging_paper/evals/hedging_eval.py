import json
from dataclasses import dataclass
from pathlib import Path

import torch
from sae_lens import SAE

from hedging_paper.evals.eval import Eval
from hedging_paper.evals.util import glob_matches
from hedging_paper.util import DEFAULT_DEVICE_STR

HEDGING_STATS_FILENAME = "hedging_eval_results.json"


@dataclass
class HedgingEvalOptions:
    control_sae_relative_path: str = "../control"
    device: str | None = None
    nbaselines: int = 50


class HedgingEval(Eval):
    def __init__(self, options: HedgingEvalOptions | None = None):
        self.options = options or HedgingEvalOptions()

    def has_eval_run(self, results_dir: Path) -> bool:
        return glob_matches(results_dir, f"**/{HEDGING_STATS_FILENAME}")

    def run(self, sae: SAE, results_dir: Path, shared_dir: Path) -> None:  # noqa: ARG002
        control_sae = SAE.load_from_pretrained(
            str((results_dir / self.options.control_sae_relative_path).resolve()),
            device=self.options.device or DEFAULT_DEVICE_STR,
        )
        control_sae.fold_W_dec_norm()
        sae.fold_W_dec_norm()
        stats = calculate_hedging_stats(
            control_sae=control_sae,
            extended_sae=sae,
            nbaselines=self.options.nbaselines,
        )
        with open(results_dir / HEDGING_STATS_FILENAME, "w") as f:
            json.dump(stats, f)


@torch.inference_mode()
def measure_subspace_projection(x, vectors):
    # Step 1: Find an orthonormal basis for the subspace using QR decomposition
    q, r = torch.linalg.qr(vectors.T)  # q will contain orthonormal basis vectors

    # The rank of the subspace might be less than 32 if vectors are linearly dependent
    # We can determine this from the rank of r
    rank = torch.linalg.matrix_rank(r)
    q = q[:, :rank]  # Only keep the basis vectors

    # Step 2: Project x onto the subspace
    projection = q @ (q.T @ x.T)
    return projection.T


def calculate_hedging_stats(control_sae: SAE, extended_sae: SAE, nbaselines: int):
    original_width = control_sae.cfg.d_sae
    nlatents = extended_sae.cfg.d_sae - control_sae.cfg.d_sae
    delta = extended_sae.W_dec[:original_width] - control_sae.W_dec[:original_width]
    cos = torch.nn.functional.cosine_similarity(
        extended_sae.W_dec[:original_width], control_sae.W_dec[:original_width]
    )
    just_added_feats = extended_sae.W_dec[original_width:]
    mean_control_dec_norm = control_sae.W_dec.norm(dim=-1).mean().item()
    delta_proj = (
        measure_subspace_projection(delta, just_added_feats).norm(dim=-1).mean().item()
    )
    delta_proj_portion = delta_proj / mean_control_dec_norm
    baseline_projs = [
        measure_subspace_projection(delta, torch.randn_like(just_added_feats))
        .norm(dim=-1)
        .mean()
        .item()
        for _ in range(nbaselines)
    ]
    baseline_proj = sum(baseline_projs) / nbaselines
    baseline_proj_std = torch.std(torch.tensor(baseline_projs)).item()
    baseline_proj_portion = baseline_proj / mean_control_dec_norm
    return {
        "width": original_width,
        "nlatents": nlatents,
        "delta_norm": delta.norm(dim=-1).mean().item(),
        "delta_proj": delta_proj,
        "delta_proj_portion": delta_proj_portion,
        "control_dec_norm": mean_control_dec_norm,
        "mean_dec_cos_sim_delta": cos.mean().item(),
        "nbaselines": nbaselines,
        "baseline_proj": baseline_proj,
        "baseline_proj_std": baseline_proj_std,
        "baseline_proj_portion": baseline_proj_portion,
        "hedging_rate": max(
            0.0,
            (delta_proj_portion - baseline_proj_portion) / (1 - baseline_proj_portion),
        ),
    }
