from pathlib import Path

import pytest
import torch
from dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKSAE,
    MatryoshkaBatchTopKTrainer,
)
from sae_lens import SAE

from hedging_paper.saes.matryoshka_sae import (
    BatchTopkMatryoshkaSAE,
    MatryoshkaSAE,
    MatryoshkaSAEConfig,
    MatryoshkaSAERunnerConfig,
    mse_loss,
    partial_l1_loss,
)
from tests.helpers import build_sae_cfg
from tests.saes._comparison_saes import (
    GlobalBatchTopKMatryoshkaSAE as BartGlobalBatchTopKMatryoshkaSAE,
)

BATCH_TOPK_SAE_AND_CFG_VARIANTS = [
    (BatchTopkMatryoshkaSAE, MatryoshkaSAEConfig, MatryoshkaSAERunnerConfig),
]


@pytest.mark.parametrize(
    "sae_class, sae_cfg_cls, runner_cfg_cls", BATCH_TOPK_SAE_AND_CFG_VARIANTS
)
def test_BatchTopkSAE_matryoshka_variants_saves_as_jumprelu(
    sae_class, sae_cfg_cls, runner_cfg_cls, tmp_path: Path
) -> None:
    # Create BatchTopkSAE variant
    cfg = build_sae_cfg(
        runner_cfg_cls=runner_cfg_cls,
        sae_cfg_cls=sae_cfg_cls,
        d_in=5,
        d_sae=10,
        architecture="topk",
        activation_fn_kwargs={"k": 2},
        matryoshka_steps=[2, 3, 4],  # Required for matryoshka variants
    )
    sae = sae_class(cfg)

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


@pytest.mark.parametrize(
    "sae_class, sae_cfg_cls, runner_cfg_cls", BATCH_TOPK_SAE_AND_CFG_VARIANTS
)
def test_BatchTopkSAE_matryoshka_variants_training_step_updates_threshold(
    sae_class, sae_cfg_cls, runner_cfg_cls
) -> None:
    # Create BatchTopkSAE variant
    cfg = build_sae_cfg(
        runner_cfg_cls=runner_cfg_cls,
        sae_cfg_cls=sae_cfg_cls,
        d_in=5,
        d_sae=10,
        architecture="topk",
        activation_fn_kwargs={"k": 2},
        matryoshka_steps=[2, 3, 4],  # Required for matryoshka variants
    )
    sae = sae_class(cfg)

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


@pytest.mark.parametrize(
    "sae_class, sae_cfg_cls, runner_cfg_cls", BATCH_TOPK_SAE_AND_CFG_VARIANTS
)
def test_BatchTopkSAE_matryoshka_variants_dead_latents_increases_losses(
    sae_class, sae_cfg_cls, runner_cfg_cls
) -> None:
    # Create two identical SAEs, one with dead_latents=None and one with all latents marked dead
    cfg = build_sae_cfg(
        runner_cfg_cls=runner_cfg_cls,
        sae_cfg_cls=sae_cfg_cls,
        d_in=5,
        d_sae=10,
        architecture="topk",
        activation_fn_kwargs={"k": 2},
        matryoshka_steps=[2, 3, 4],  # Required for matryoshka variants
    )
    sae = sae_class(cfg)

    # Create random input
    x = torch.randn(4, 5)

    # Get outputs from both SAEs
    with torch.no_grad():
        output_no_dead = sae.training_forward_pass(
            x, current_l1_coefficient=0.1, dead_neuron_mask=None
        )
        output_all_dead = sae.training_forward_pass(
            x,
            current_l1_coefficient=0.1,
            dead_neuron_mask=torch.ones(cfg.d_sae, dtype=torch.bool),
        )

    # Verify aux loss increased when all latents marked as dead
    assert (
        output_all_dead.losses["auxiliary_reconstruction_loss"]
        > output_no_dead.losses["auxiliary_reconstruction_loss"]
    )

    # Verify total loss increased
    assert output_all_dead.loss > output_no_dead.loss


@pytest.mark.parametrize(
    "sae_class, sae_cfg_cls, runner_cfg_cls", BATCH_TOPK_SAE_AND_CFG_VARIANTS
)
def test_BatchTopkSAE_matryoshka_variants_matches_comparison_sae(
    sae_class, sae_cfg_cls, runner_cfg_cls
) -> None:
    # Create BatchTopkSAE variant
    cfg = build_sae_cfg(
        runner_cfg_cls=runner_cfg_cls,
        sae_cfg_cls=sae_cfg_cls,
        d_in=5,
        d_sae=10,
        architecture="topk",
        activation_fn_kwargs={"k": 2},
        include_matryoshka_inner_l1_loss=False,
        matryoshka_steps=[2, 3, 4],  # Required for matryoshka variants
        normalize_losses_by_num_matryoshka_steps=True,
        normalize_reconstruction_losses_by_d_in=True,
        include_bdec_loss=True,
    )
    batch_topk_sae = sae_class(cfg)

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
        "group_sizes": [2, 1, 1, 6],
    }
    comparison_sae = BartGlobalBatchTopKMatryoshkaSAE(comparison_cfg)

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


@pytest.mark.parametrize(
    "sae_class, sae_cfg_cls, runner_cfg_cls", BATCH_TOPK_SAE_AND_CFG_VARIANTS
)
def test_BatchTopkSAE_matryoshka_variants_matches_dictionary_learning(
    sae_class, sae_cfg_cls, runner_cfg_cls
) -> None:
    # Create BatchTopkSAE variant
    cfg = build_sae_cfg(
        runner_cfg_cls=runner_cfg_cls,
        sae_cfg_cls=sae_cfg_cls,
        d_in=5,
        d_sae=10,
        architecture="topk",
        activation_fn_kwargs={"k": 2},
        include_matryoshka_inner_l1_loss=False,
        matryoshka_steps=[2, 3, 4],  # Required for matryoshka variants
        normalize_losses_by_num_matryoshka_steps=True,
        normalize_reconstruction_losses_by_d_in=True,
        include_bdec_loss=True,
    )
    batch_topk_sae = sae_class(cfg)

    # Create comparison SAE
    comparison_sae = MatryoshkaBatchTopKSAE(
        activation_dim=5,
        dict_size=10,
        k=2,
        group_sizes=[2, 1, 1, 6],
    )

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
        comp_feats = comparison_sae.encode(x, use_threshold=False)
        comp_sae_out = comparison_sae.decode(comp_feats)  # type: ignore

    torch.testing.assert_close(batch_topk_output.feature_acts, comp_feats)
    torch.testing.assert_close(batch_topk_output.sae_out, comp_sae_out)


@pytest.mark.parametrize(
    "sae_class, sae_cfg_cls, runner_cfg_cls", BATCH_TOPK_SAE_AND_CFG_VARIANTS
)
def test_BatchTopkSAE_matryoshka_variants_matches_bart_losses(
    sae_class, sae_cfg_cls, runner_cfg_cls
) -> None:
    # Create our SAE
    cfg = build_sae_cfg(
        runner_cfg_cls=runner_cfg_cls,
        sae_cfg_cls=sae_cfg_cls,
        d_in=5,
        d_sae=10,
        l1_coefficient=0,
        architecture="topk",
        activation_fn_kwargs={"k": 2},
        include_matryoshka_inner_l1_loss=False,
        matryoshka_steps=[2, 3, 4],  # Required for matryoshka variants
        normalize_losses_by_num_matryoshka_steps=True,
        normalize_reconstruction_losses_by_d_in=True,
        include_bdec_loss=True,
    )
    batch_topk_sae = sae_class(cfg)

    # Create comparison SAE
    comparison_cfg = {
        "act_size": 5,
        "dict_size": 10,
        "top_k": 2,
        "top_k_aux": 2,
        "aux_penalty": 0.1,
        "n_batches_to_dead": 100,
        "seed": 42,
        "input_unit_norm": False,
        "device": "cpu",
        "dtype": torch.float32,
        "group_sizes": [2, 1, 1, 6],
        "l1_coeff": 0,
    }
    comparison_sae = BartGlobalBatchTopKMatryoshkaSAE(comparison_cfg)

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
            x, current_l1_coefficient=0
        )
        # Comparison SAE output
        comparison_output = comparison_sae(x)

    # Compare losses
    torch.testing.assert_close(
        batch_topk_output.losses["mse_loss"]
        + batch_topk_output.losses["inner_recons_loss"],
        comparison_output["l2_loss"],
    )
    torch.testing.assert_close(batch_topk_output.loss, comparison_output["loss"])


@pytest.mark.parametrize(
    "sae_class, sae_cfg_cls, runner_cfg_cls", BATCH_TOPK_SAE_AND_CFG_VARIANTS
)
def test_BatchTopkSAE_matryoshka_variants_matches_dictionary_learning_losses(
    sae_class, sae_cfg_cls, runner_cfg_cls
) -> None:
    # Create our SAE
    cfg = build_sae_cfg(
        runner_cfg_cls=runner_cfg_cls,
        sae_cfg_cls=sae_cfg_cls,
        d_in=5,
        d_sae=10,
        l1_coefficient=0,
        architecture="topk",
        activation_fn_kwargs={"k": 2},
        include_matryoshka_inner_l1_loss=False,
        matryoshka_steps=[2, 3, 4],  # Required for matryoshka variants
        normalize_losses_by_num_matryoshka_steps=True,
        normalize_reconstruction_losses_by_d_in=False,
        include_bdec_loss=False,
    )
    batch_topk_sae = sae_class(cfg)

    comparison_trainer = MatryoshkaBatchTopKTrainer(
        steps=100,
        activation_dim=5,
        dict_size=10,
        k=2,
        layer=0,
        warmup_steps=0,
        lm_name="gpt2",
        group_fractions=[0.2, 0.1, 0.1, 0.6],
        group_weights=[1.0, 1.0, 1.0, 1.0],
    )

    # Copy weights to make them identical
    comparison_trainer.ae.W_enc.data = batch_topk_sae.W_enc.data
    comparison_trainer.ae.W_dec.data = batch_topk_sae.W_dec.data
    comparison_trainer.ae.b_dec.data = batch_topk_sae.b_dec.data
    comparison_trainer.ae.threshold = batch_topk_sae.topk_threshold

    # Create random input
    x = torch.randn(4, 5)

    # Get outputs from both SAEs
    with torch.no_grad():
        # BatchTopkSAE output
        batch_topk_output = batch_topk_sae.training_forward_pass(
            x, current_l1_coefficient=0
        )
        # Comparison SAE output
        comp_losses = comparison_trainer.loss(x, logging=True, step=0).losses  # type: ignore

    # Compare losses
    torch.testing.assert_close(
        (
            batch_topk_output.losses["mse_loss"]
            + batch_topk_output.losses["inner_recons_loss"]
        ).item(),
        comp_losses["l2_loss"],
    )
    torch.testing.assert_close(batch_topk_output.loss.item(), comp_losses["loss"])
    torch.testing.assert_close(
        batch_topk_output.losses["auxiliary_reconstruction_loss"].item(),
        comp_losses["auxk_loss"],
    )


@pytest.mark.parametrize(
    "sae_class, sae_cfg_cls, runner_cfg_cls", BATCH_TOPK_SAE_AND_CFG_VARIANTS
)
def test_BatchTopkMatryoshkaSAE_save_load_checkpoint(
    sae_class, sae_cfg_cls, runner_cfg_cls, tmp_path: Path
) -> None:
    # Create SAE configuration
    cfg = build_sae_cfg(
        runner_cfg_cls=runner_cfg_cls,
        sae_cfg_cls=sae_cfg_cls,
        d_in=8,
        d_sae=10,
        architecture="topk",
        activation_fn_kwargs={"k": 2},
        matryoshka_steps=[2, 3, 4],  # required for matryoshka variants
    )

    # Create original SAE
    original_sae = sae_class(cfg)

    # Set arbitrary values for step_num and topk_threshold
    arbitrary_step_num = 42
    arbitrary_threshold = 0.75

    original_sae.step_num = arbitrary_step_num
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
    loaded_sae = sae_class(cfg)

    # Load from checkpoint
    loaded_sae.load_from_checkpoint(tmp_path)

    # Verify step_num was preserved
    assert loaded_sae.step_num == arbitrary_step_num

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


def test_decode_partial() -> None:
    num_feats: int = 4
    num_hidden: int = 10
    portion: int = 2

    base_cfg = build_sae_cfg(
        d_in=num_hidden,
        d_sae=num_feats,
        lr=0.1,
    )
    cfg = MatryoshkaSAEConfig(
        **base_cfg.to_dict(),
        matryoshka_steps=[2, 3, 4],
    )
    sae = MatryoshkaSAE(cfg)

    # Create test input
    feature_acts = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    # Get partial decode output
    partial_out = sae.decode_partial(feature_acts, portion)

    # Verify shape is correct
    assert partial_out.shape == (2, num_hidden)

    # Verify it only uses first `portion` features
    expected_out = (feature_acts[:, :portion] @ sae.W_dec[:portion]) + sae.b_dec
    assert torch.allclose(partial_out, expected_out)


def test_partial_l1_loss_weighted() -> None:
    # Test basic partial L1 loss
    feature_acts = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    l1_coeff = 0.1
    portion = 2

    # Should only consider first 2 columns
    # L1 norm of [1,2] and [4,5], then mean
    expected = l1_coeff * ((1.0 + 2.0 + 4.0 + 5.0) / 2)

    result = partial_l1_loss(feature_acts, l1_coeff, portion, lp_norm=1.0)
    assert torch.isclose(result, torch.tensor(expected))


def test_mse_loss_simple_case() -> None:
    error = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    # Sum over features (dim=-1), then mean over batch
    expected = torch.tensor((1.0**2 + (-2.0) ** 2 + 3.0**2 + (-4.0) ** 2) / 2)
    assert torch.isclose(mse_loss(error), expected)


def test_partial_l1_loss_basic() -> None:
    feature_acts = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    l1_coeff = 0.1
    portion = 2

    # Should only consider first 2 columns
    # L1 norm of [1,2] and [4,5], then mean
    expected = l1_coeff * ((1.0**1 + 2.0**1 + 4.0**1 + 5.0**1) / 2)

    result = partial_l1_loss(feature_acts, l1_coeff, portion, lp_norm=1.0)
    assert torch.isclose(result, torch.tensor(expected))


def test_partial_l1_loss_l2_norm() -> None:
    feature_acts = torch.tensor([[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]])
    l1_coeff = 0.1
    portion = 2

    # Should compute L2 norm of first 2 columns
    # L2 norm of [3,4] is 5, L2 norm of [6,8] is 10, mean is 7.5
    expected = l1_coeff * 7.5

    result = partial_l1_loss(feature_acts, l1_coeff, portion, lp_norm=2.0)
    assert torch.isclose(result, torch.tensor(expected))


def test_MatryoshkaSAE_iterable_decode_matches_decode_partial() -> None:
    base_cfg = build_sae_cfg(
        d_in=8,
        d_sae=10,
        lr=0.1,
    )
    cfg = MatryoshkaSAEConfig(
        **base_cfg.to_dict(),
        matryoshka_steps=[2, 4, 6],  # 3 stages
    )
    sae = MatryoshkaSAE(cfg)

    # Create test input and get feature activations
    x = torch.randn(4, 8)
    feature_acts = sae.encode(x)

    # Test each stage matches decode_partial
    for i, decoded in enumerate(sae.iterable_decode(feature_acts)):
        expected = sae.decode_partial(feature_acts, cfg.matryoshka_steps[i])
        torch.testing.assert_close(decoded, expected)


def test_MatryoshkaSAE_iterable_decode_max_stage() -> None:
    base_cfg = build_sae_cfg(
        d_in=8,
        d_sae=10,
        lr=0.1,
    )
    cfg = MatryoshkaSAEConfig(
        **base_cfg.to_dict(),
        matryoshka_steps=[2, 4, 6],  # 3 stages
    )
    sae = MatryoshkaSAE(cfg)

    x = torch.randn(4, 8)
    feature_acts = sae.encode(x)

    max_stage = 1
    decoded_stages = list(sae.iterable_decode(feature_acts, max_stage=max_stage))
    assert len(decoded_stages) == max_stage + 1


def test_MatryoshkaSAE_iterable_decode_full_width_final() -> None:
    base_cfg = build_sae_cfg(
        d_in=8,
        d_sae=10,
        lr=0.1,
    )
    cfg = MatryoshkaSAEConfig(
        **base_cfg.to_dict(),
        matryoshka_steps=[2, 4, 6],  # 3 stages
    )
    sae = MatryoshkaSAE(cfg)

    x = torch.randn(4, 8)
    feature_acts = sae.encode(x)

    decoded_stages = list(
        sae.iterable_decode(feature_acts, use_full_width_for_final_portion=True)
    )
    final_stage = decoded_stages[-1]
    full_decode = sae.decode(feature_acts)
    torch.testing.assert_close(final_stage, full_decode)


def test_MatryoshkaSAE_save_load_checkpoint(tmp_path: Path) -> None:
    base_cfg = build_sae_cfg(
        d_in=8,
        d_sae=10,
        lr=0.1,
    )
    cfg = MatryoshkaSAEConfig(
        **base_cfg.to_dict(),
        matryoshka_steps=[2, 4, 6],
    )

    original_sae = MatryoshkaSAE(cfg)
    original_sae.step_num = 42

    x = torch.randn(4, 8)
    feature_acts = original_sae.encode(x)
    original_output = original_sae.decode(feature_acts)
    original_sae.save_model(tmp_path)

    loaded_sae = MatryoshkaSAE(cfg)
    loaded_sae.load_from_checkpoint(tmp_path)

    # Verify step_num was preserved
    assert loaded_sae.step_num == original_sae.step_num

    # Verify weights are identical
    torch.testing.assert_close(loaded_sae.W_enc, original_sae.W_enc)
    torch.testing.assert_close(loaded_sae.b_enc, original_sae.b_enc)
    torch.testing.assert_close(loaded_sae.W_dec, original_sae.W_dec)
    torch.testing.assert_close(loaded_sae.b_dec, original_sae.b_dec)

    # Verify matryoshka-specific attributes
    assert loaded_sae.matryoshka_steps == original_sae.matryoshka_steps
    assert (
        loaded_sae.inner_l1_loss_multipliers == original_sae.inner_l1_loss_multipliers
    )

    # Verify behavior is identical
    loaded_feature_acts = loaded_sae.encode(x)
    loaded_output = loaded_sae.decode(loaded_feature_acts)
    torch.testing.assert_close(loaded_output, original_output)

    # Test iterable_decode behavior
    original_stages = list(original_sae.iterable_decode(feature_acts))
    loaded_stages = list(loaded_sae.iterable_decode(loaded_feature_acts))

    assert len(original_stages) == len(loaded_stages)
    for orig_stage, loaded_stage in zip(original_stages, loaded_stages):
        torch.testing.assert_close(orig_stage, loaded_stage)


def test_MatryoshkaSAE_save_model_final_and_load_from_pretrained(
    tmp_path: Path,
) -> None:
    # Create a MatryoshkaSAE
    base_cfg = build_sae_cfg(
        d_in=8,
        d_sae=10,
        lr=0.1,
    )
    cfg = MatryoshkaSAEConfig(
        **base_cfg.to_dict(),
        matryoshka_steps=[2, 4, 6],
    )

    original_sae = MatryoshkaSAE(cfg)
    original_sae.step_num = 42

    # Generate some test data
    x = torch.randn(4, 8)
    feature_acts = original_sae.encode(x)
    original_output = original_sae.decode(feature_acts)

    # Calculate sparsity for save_model_final
    sparsity = torch.mean((feature_acts.abs() > 0).float(), dim=0)

    # Save the model using save_model_final
    original_sae.save_model_final(tmp_path, sparsity)

    # Load the model using SAE.load_from_pretrained
    loaded_sae = SAE.load_from_pretrained(str(tmp_path))

    # Verify the loaded model has the correct dimensions
    assert loaded_sae.cfg.d_in == cfg.d_in
    assert loaded_sae.cfg.d_sae == cfg.d_sae

    # Verify weights are identical
    torch.testing.assert_close(loaded_sae.W_enc, original_sae.W_enc)
    torch.testing.assert_close(loaded_sae.b_enc, original_sae.b_enc)
    torch.testing.assert_close(loaded_sae.W_dec, original_sae.W_dec)
    torch.testing.assert_close(loaded_sae.b_dec, original_sae.b_dec)

    # Verify behavior is identical for encoding and decoding
    loaded_feature_acts = loaded_sae.encode(x)
    loaded_output = loaded_sae.decode(loaded_feature_acts)
    torch.testing.assert_close(loaded_feature_acts, feature_acts)
    torch.testing.assert_close(loaded_output, original_output)
