import json
from pathlib import Path

from sae_lens import SAE

from hedging_paper.evals.hedging_eval import (
    HEDGING_STATS_FILENAME,
    HedgingEval,
    HedgingEvalOptions,
)
from hedging_paper.saes.base_sae import BaseSAE
from tests.helpers import build_sae_cfg


def test_hedging_eval_run(tmp_path: Path) -> None:
    d_in = 16
    initial_d_sae = 32
    num_latents_to_add = 8

    # Create and save the control SAE
    control_cfg = build_sae_cfg(
        d_in=d_in,
        d_sae=initial_d_sae,
        device="cpu",
    )
    sae = BaseSAE(control_cfg)
    control_sae_dir = tmp_path / "control"
    sae.save_model_final(control_sae_dir)

    # Initialize extended SAE with control config dimensions first
    sae.extend_sae(num_latents_to_add)

    extended_sae_dir = tmp_path / "extended"
    sae.save_model_final(extended_sae_dir)

    # Prepare and run HedgingEval
    hedging_options = HedgingEvalOptions(
        control_sae_relative_path="../control", device="cpu"
    )
    hedging_eval = HedgingEval(options=hedging_options)

    # Load the extended SAE using the standard SAE class for the eval
    loaded_extended_sae = SAE.load_from_pretrained(str(extended_sae_dir), device="cpu")

    assert not hedging_eval.has_eval_run(extended_sae_dir)
    hedging_eval.run(
        sae=loaded_extended_sae, results_dir=extended_sae_dir, shared_dir=tmp_path
    )
    assert hedging_eval.has_eval_run(extended_sae_dir)

    # Check results file
    results_path = extended_sae_dir / HEDGING_STATS_FILENAME
    assert results_path.exists()

    with open(results_path) as f:
        results = json.load(f)

    # Basic assertions on results
    expected_keys = {
        "width",
        "nlatents",
        "delta_norm",
        "delta_proj",
        "delta_proj_portion",
        "control_dec_norm",
        "mean_dec_cos_sim_delta",
        "baseline_proj",
        "baseline_proj_portion",
        "hedging_rate",
        "nbaselines",
        "baseline_proj_std",
    }
    assert set(results.keys()) == expected_keys
    assert results["width"] == initial_d_sae
    assert results["nlatents"] == num_latents_to_add
    assert isinstance(results["delta_norm"], float)
    assert isinstance(results["delta_proj"], float)
    assert isinstance(results["delta_proj_portion"], float)
    assert isinstance(results["control_dec_norm"], float)
    assert isinstance(results["mean_dec_cos_sim_delta"], float)
    assert isinstance(results["baseline_proj"], float)
    assert isinstance(results["baseline_proj_portion"], float)
    assert isinstance(results["hedging_rate"], float)
    assert results["hedging_rate"] >= 0
    assert isinstance(results["baseline_proj_std"], float)
    assert results["nbaselines"] == 50
