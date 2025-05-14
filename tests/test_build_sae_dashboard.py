import zipfile
from pathlib import Path

from sae_lens import TrainingSAE
from transformer_lens import HookedTransformer

from hedging_paper.build_sae_dashboard import (
    BuildSAEDashboardOptions,
    build_sae_dashboard,
)


def test_build_sae_dashboard(
    gpt2_model: HookedTransformer,
    gpt2_l4_sae: TrainingSAE,
    tmp_path: Path,
):
    output_path = tmp_path / "dashboard-test"

    # Run with minimal settings for fast test
    options = BuildSAEDashboardOptions(
        n_prompts=2,
        store_batch_size_prompts=1,
        n_batches_in_buffer=1,
        minibatch_size_features=32,
        minibatch_size_tokens=32,
        device="cpu",
        dtype="float32",
        context_size=21,
        zip_output=True,
        ignore_special_tokens=True,
        features=[1, 3, 5, 7, 9, 123],
    )

    output_filename = build_sae_dashboard(
        sae=gpt2_l4_sae,
        model=gpt2_model,
        output_path=output_path,
        options=options,
    )

    assert output_filename == str(tmp_path / "dashboard-test.zip")
    # Verify output file exists
    assert Path(output_filename).exists()
    # Verify expected feature HTML files exist in zip

    with zipfile.ZipFile(output_filename, "r") as zf:
        filenames = zf.namelist()
        for feature_id in [1, 3, 5, 7, 9, 123]:
            assert f"dashboard_feature_{feature_id}.html" in filenames
