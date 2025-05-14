from pathlib import Path

import pytest
import torch
from datasets import Dataset
from pytest import approx
from sae_lens import (
    SAE,
    ActivationsStore,  # Keep for reference but will use BaseSAEConfig for initialization
)
from sae_lens.evals import get_sparsity_and_variance_metrics
from transformer_lens import HookedTransformer

from hedging_paper.saes.base_sae import BaseSAE, BaseSAEConfig
from hedging_paper.saes.batch_topk_sae import BatchTopK, BatchTopkSAE
from tests.helpers import build_runner_cfg, build_sae_cfg
from tests.saes._comparison_saes import BatchTopKSAE as BartBatchTopKSAE


def test_BatchTopK_with_same_number_of_top_features_per_batch():
    batch_topk = BatchTopK(k=2)
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0], [-5.0, 6.0, -7.0, 8.0]])

    # Expected output after:
    # 1. ReLU: [[1, 0, 3, 0], [0, 6, 0, 8]]
    # 2. Flatten: [1, 0, 3, 0, 0, 6, 0, 8]
    # 3. Top 4 values (k=2 * batch_size=2): [8, 6, 3, 1]
    # 4. Reshape back to original shape
    expected = torch.tensor([[1.0, 0.0, 3.0, 0.0], [0.0, 6.0, 0.0, 8.0]])

    output = batch_topk(x)

    assert output.shape == x.shape
    torch.testing.assert_close(output, expected)

    # Check that exactly k*batch_size values are non-zero
    assert (output != 0).sum() == batch_topk.k * x.shape[0]


def test_BatchTopK_with_imbalanced_top_features_per_batch():
    batch_topk = BatchTopK(k=2)
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # All positive values
            [-5.0, -6.0, 0.5, -8.0],  # Only one small positive value
        ]
    )

    # Expected output after:
    # 1. ReLU: [[1, 2, 3, 4], [0, 0, 0.5, 0]]
    # 2. Flatten: [1, 2, 3, 4, 0, 0, 0.5, 0]
    # 3. Top 4 values (k=2 * batch_size=2): [4, 3, 2, 1]
    # 4. Reshape back to original shape
    expected = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # Gets 4 non-zero values
            [0.0, 0.0, 0.0, 0.0],  # Gets 0 non-zero values
        ]
    )

    output = batch_topk(x)
    torch.testing.assert_close(output, expected)

    # Check that exactly k*batch_size values are non-zero across all batches
    assert (output != 0).sum() == batch_topk.k * x.shape[0]


def test_BatchTopK_output_must_be_positive():
    batch_topk = BatchTopK(k=2)
    x = torch.tensor(
        [
            [-1.0, 2.0, -3.0, -4.0],  # Only 1 positive value
            [5.0, -6.0, -7.0, -8.0],  # Only 1 positive value
        ]
    )
    # Total positive values (2) < k*batch_size (4)

    # Expected output after:
    # 1. ReLU: [[0, 2, 0, 0], [5, 0, 0, 0]]
    # 2. Flatten: [0, 2, 0, 0, 5, 0, 0, 0]
    # 3. Top 4 values (k=2 * batch_size=2): [5, 2, 0, 0]
    # 4. Reshape back to original shape
    expected = torch.tensor(
        [
            [0.0, 2.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
        ]
    )

    output = batch_topk(x)

    torch.testing.assert_close(output, expected)
    assert (output >= 0).all()

    # Check that number of non-zero values equals number of positive inputs
    # (which is less than k*batch_size)
    assert (output != 0).sum() == (x > 0).sum()


@pytest.mark.parametrize("sae_cls", [BatchTopkSAE])
def test_BatchTopkSAE_saves_as_jumprelu(tmp_path: Path, sae_cls: type[BaseSAE]) -> None:
    # Create BatchTopkSAE
    cfg = build_sae_cfg(
        d_in=5,
        d_sae=10,
        architecture="topk",
        activation_fn_kwargs={"k": 2},
    )
    sae = sae_cls(cfg)

    # Set non-zero threshold
    sae.topk_threshold = torch.tensor(0.5)

    # Save model
    sae.save_model_final(tmp_path)

    # Load model back
    loaded_sae = SAE.load_from_pretrained(str(tmp_path))

    # Verify architecture was changed to jumprelu
    assert loaded_sae.cfg.architecture == "jumprelu"
    assert loaded_sae.cfg.activation_fn_str == "relu"

    # Verify threshold was expanded to d_sae size and saved
    state_dict = loaded_sae.state_dict()
    assert "threshold" in state_dict
    assert state_dict["threshold"].shape == (cfg.d_sae,)
    assert torch.allclose(state_dict["threshold"], torch.full((cfg.d_sae,), 0.5))


def test_BatchTopkSAE_save_load_checkpoint(tmp_path: Path) -> None:
    """Test that a BatchTopkSAE can be saved and loaded from a checkpoint."""

    # Create BatchTopkSAE
    cfg = build_sae_cfg(
        d_in=8,
        d_sae=10,
        architecture="topk",
        activation_fn_kwargs={"k": 2},
    )

    original_sae = BatchTopkSAE(cfg)

    # Set threshold to an arbitrary value
    arbitrary_threshold = 0.75
    original_sae.topk_threshold = torch.tensor(
        arbitrary_threshold, dtype=original_sae.topk_threshold.dtype
    )

    # Create test input and get feature activations
    x = torch.randn(4, 8)
    original_feature_acts = original_sae.encode(x)
    original_output = original_sae.decode(original_feature_acts)

    # Save the model
    original_sae.save_model(tmp_path)

    # Create a new SAE with the same config
    loaded_sae = BatchTopkSAE(cfg)

    # Load from checkpoint
    loaded_sae.load_from_checkpoint(tmp_path)

    # Verify threshold was preserved
    assert torch.isclose(loaded_sae.topk_threshold, original_sae.topk_threshold)
    assert loaded_sae.topk_threshold.item() == arbitrary_threshold

    # Verify weights are identical
    torch.testing.assert_close(loaded_sae.W_enc, original_sae.W_enc)
    torch.testing.assert_close(loaded_sae.b_enc, original_sae.b_enc)
    torch.testing.assert_close(loaded_sae.W_dec, original_sae.W_dec)
    torch.testing.assert_close(loaded_sae.b_dec, original_sae.b_dec)

    # Verify behavior is identical
    loaded_feature_acts = loaded_sae.encode(x)
    loaded_output = loaded_sae.decode(loaded_feature_acts)

    torch.testing.assert_close(loaded_feature_acts, original_feature_acts)
    torch.testing.assert_close(loaded_output, original_output)


def test_BatchTopkSAE_training_step_updates_threshold() -> None:
    # Create BatchTopkSAE
    cfg = build_sae_cfg(
        d_in=5,
        d_sae=10,
        architecture="topk",
        activation_fn_kwargs={"k": 2},
    )
    sae = BatchTopkSAE(cfg)

    # Set initial threshold
    initial_threshold = 0.0
    sae.topk_threshold = torch.tensor(initial_threshold)
    sae.b_enc.data = torch.randn(10) + 10

    # Create input that will produce non-zero activations
    x = torch.randn(4, 5)

    # Do training step
    with torch.no_grad():
        output = sae.training_forward_pass(x, current_l1_coefficient=0.1)

    # Verify threshold changed
    assert sae.topk_threshold != initial_threshold

    # Verify threshold is included in losses dict
    assert "topk_threshold" in output.losses
    assert output.losses["topk_threshold"] == sae.topk_threshold


def test_BatchTopkSAE_matches_comparison_sae() -> None:
    # Create BatchTopkSAE
    cfg = build_sae_cfg(
        d_in=5,
        d_sae=10,
        architecture="topk",
        activation_fn_kwargs={"k": 2},
    )
    batch_topk_sae = BatchTopkSAE(cfg)

    # Create comparison SAE
    comparison_cfg = {
        "act_size": 5,
        "dict_size": 10,
        "top_k": 2,
        "top_k_aux": 2,
        "aux_penalty": 0.1,
        "l1_coeff": 0.1,
        "n_batches_to_dead": 100,
        "seed": 42,
        "input_unit_norm": False,
        "device": "cpu",
        "dtype": torch.float32,
    }
    comparison_sae = BartBatchTopKSAE(comparison_cfg)

    # Copy weights to make them identical
    comparison_sae.W_enc.data = batch_topk_sae.W_enc.data
    comparison_sae.W_dec.data = batch_topk_sae.W_dec.data
    comparison_sae.b_dec.data = batch_topk_sae.b_dec.data
    comparison_sae.threshold = batch_topk_sae.topk_threshold

    # Create random input
    x = torch.randn(4, 5)

    # Get outputs from both SAEs
    with torch.no_grad():
        # BatchTopkSAE output
        batch_topk_output = batch_topk_sae.training_forward_pass(
            x, current_l1_coefficient=0.1
        )

        # Comparison SAE output
        comparison_output = comparison_sae(x)

    # Compare feature activations
    torch.testing.assert_close(
        batch_topk_output.feature_acts, comparison_output["feature_acts"]
    )

    # Compare reconstructions
    torch.testing.assert_close(batch_topk_output.sae_out, comparison_output["sae_out"])


def test_BatchTopkSAE_encode_matches_encode_with_hidden_pre():
    # Create BatchTopkSAE
    cfg = build_sae_cfg(
        d_in=5,
        d_sae=10,
        architecture="topk",
        activation_fn_kwargs={"k": 2},
    )
    batch_topk_sae = BatchTopkSAE(cfg)

    # make sure there's lots of positive things to topk
    batch_topk_sae.b_enc.data = torch.randn(10) + 10.0

    # Create random input
    x = torch.randn(4, 5)

    # Get outputs from both encoding methods
    with torch.no_grad():
        encoded = batch_topk_sae.encode(x)
        feature_acts, hidden_pre = batch_topk_sae.encode_with_hidden_pre(x)

    # Check that there are k*batch_size non-zero values
    assert (encoded.flatten() > 0).sum() == 2 * 4

    # Compare outputs
    torch.testing.assert_close(encoded, feature_acts)


def test_BatchTopkSAE_get_sparsity_and_variance_metrics_matches_k(
    gpt2_model: HookedTransformer,
    example_dataset: Dataset,
):
    # Create BatchTopkSAE
    runner_cfg = build_runner_cfg(
        d_in=768,
        d_sae=10,
        architecture="topk",
        activation_fn_kwargs={"k": 2},
        context_size=10,
    )
    batch_topk_sae = BatchTopkSAE(BaseSAEConfig.from_sae_runner_config(runner_cfg))
    batch_topk_sae.b_enc.data = torch.randn(10) + 10.0

    store = ActivationsStore.from_config(
        gpt2_model, runner_cfg, override_dataset=example_dataset
    )

    # Get metrics
    sparsity, variance = get_sparsity_and_variance_metrics(
        sae=batch_topk_sae,
        model=gpt2_model,
        activation_store=store,
        n_batches=2,
        compute_l2_norms=False,
        compute_sparsity_metrics=True,
        compute_variance_metrics=False,
        compute_featurewise_density_statistics=False,
        eval_batch_size_prompts=2,
        model_kwargs={"device": "cpu"},
    )

    # Check that l0 close to k
    assert sparsity["l0"] == approx(2.0)
