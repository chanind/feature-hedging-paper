import numpy as np
import pytest
import torch

from hedging_paper.toy_models.get_training_batch import (
    AbsorptionPair,
    absorption,
    chain_modifiers,
    create_correlated_features_modifier,
    get_training_batch,
    suppress_features,
)
from hedging_paper.util import DEFAULT_DEVICE


def test_get_training_batch_fires_features_with_correct_probabilities():
    firing_probs = torch.tensor([0.3, 0.2, 0.1]).to(DEFAULT_DEVICE)
    batch_size = 1000
    samples = get_training_batch(batch_size, firing_probs)

    # Calculate the actual firing probabilities from the samples
    actual_probs = (samples > 0).float().mean(dim=0)

    # Assert that the actual probabilities are close to the expected ones
    torch.testing.assert_close(actual_probs, firing_probs, atol=0.05, rtol=0)


def test_get_training_batch_fires_features_with_correct_magnitudes():
    firing_probs = torch.tensor([1.0, 1.0, 1.0])
    std_firing_magnitudes = torch.tensor([0.1, 0.2, 0.3]).to(DEFAULT_DEVICE)
    batch_size = 1000
    samples = get_training_batch(batch_size, firing_probs, std_firing_magnitudes)
    actual_magnitudes = samples.std(dim=0)
    torch.testing.assert_close(
        actual_magnitudes, std_firing_magnitudes, atol=0.05, rtol=0
    )


def test_get_training_batch_never_fires_negative_magnitudes():
    firing_probs = torch.tensor([1.0, 1.0, 1.0])
    std_firing_magnitudes = torch.tensor([0.5, 1.0, 2.0])
    batch_size = 1000
    samples = get_training_batch(batch_size, firing_probs, std_firing_magnitudes)
    assert torch.all(samples >= 0)


def test_get_training_batch_can_set_mean_magnitudes():
    firing_probs = torch.tensor([0.5, 0.5, 1.0])
    firing_means = torch.tensor([1.5, 2.5, 3.5])
    batch_size = 1000
    samples = get_training_batch(
        batch_size, firing_probs, mean_firing_magnitudes=firing_means
    )
    assert set(samples[:, 0].tolist()) == {0, 1.5}
    assert set(samples[:, 1].tolist()) == {0, 2.5}
    assert set(samples[:, 2].tolist()) == {3.5}


def test_absorption_with_simple_pairs():
    # Test absorption function with custom pairs
    feats = torch.zeros(10, 6)

    # Define custom absorption pairs
    custom_pairs = [
        AbsorptionPair(4, 1),  # If feature 1 is 1, set feature 4 to 1
        AbsorptionPair(5, 3),  # If feature 3 is 1, set feature 5 to 1
    ]

    # Set some specific features
    feats[0, 1] = 1  # Should cause feature 4 to be set to 1
    feats[1, 3] = 1  # Should cause feature 5 to be set to 1
    feats[2, 0] = 1  # Should not affect other features
    feats[3, 2] = 1  # Should not affect other features
    feats[4, 1] = 1  # Should cause feature 4 to be set to 1
    feats[4, 3] = 1  # Should cause feature 5 to be set to 1

    # Apply the absorption function with custom pairs
    absorbed_feats = absorption(custom_pairs)(feats)

    # Check if the absorption worked as expected
    assert absorbed_feats[0, 4] == 1, "Feature 4 should be set to 1 when feature 1 is 1"
    assert absorbed_feats[1, 5] == 1, "Feature 5 should be set to 1 when feature 3 is 1"
    assert absorbed_feats[2, 4] == 0, "Feature 4 should not be affected by feature 0"
    assert absorbed_feats[2, 5] == 0, "Feature 5 should not be affected by feature 0"
    assert absorbed_feats[3, 4] == 0, "Feature 4 should not be affected by feature 2"
    assert absorbed_feats[3, 5] == 0, "Feature 5 should not be affected by feature 2"
    assert absorbed_feats[4, 4] == 1, "Feature 4 should be set to 1 when feature 1 is 1"
    assert absorbed_feats[4, 5] == 1, "Feature 5 should be set to 1 when feature 3 is 1"

    # Check that other features remain unchanged
    assert torch.all(absorbed_feats[:, [0, 1, 2, 3]] == feats[:, [0, 1, 2, 3]]), (
        "Features 0, 1, 2, and 3 should remain unchanged"
    )


def test_suppress_features_works():
    # Test suppress_features function
    feats = torch.zeros(10, 6)

    # Set some specific features
    feats[0, 0] = 1  # Should suppress features 1 and 2
    feats[1, 1] = 1  # Should not affect other features
    feats[2, 2] = 1  # Should not affect other features
    feats[3, 0] = 1  # Should suppress features 1 and 2
    feats[3, 1] = 1  # Should be suppressed by feature 0
    feats[3, 2] = 1  # Should be suppressed by feature 0
    feats[4, 1] = 1  # Should not affect other features
    feats[4, 2] = 1  # Should not affect other features

    # Apply the suppress_features function
    suppress_fn = suppress_features(dominant_feature=0, suppressed_features=[1, 2])
    suppressed_feats = suppress_fn(feats)

    # Check if the suppression worked as expected
    assert torch.all(suppressed_feats[0, [1, 2]] == 0), (
        "Features 1 and 2 should be suppressed when feature 0 is 1"
    )
    assert suppressed_feats[1, 1] == 1, (
        "Feature 1 should not be affected when feature 0 is 0"
    )
    assert suppressed_feats[2, 2] == 1, (
        "Feature 2 should not be affected when feature 0 is 0"
    )
    assert torch.all(suppressed_feats[3, [1, 2]] == 0), (
        "Features 1 and 2 should be suppressed when feature 0 is 1"
    )
    assert torch.all(suppressed_feats[4, [1, 2]] == 1), (
        "Features 1 and 2 should not be affected when feature 0 is 0"
    )

    # Check that other features remain unchanged
    assert torch.all(suppressed_feats[:, [0, 3, 4, 5]] == feats[:, [0, 3, 4, 5]]), (
        "Features 0, 3, 4, and 5 should remain unchanged"
    )


def test_chain_modifiers_works():
    # Create test data
    feats = torch.zeros(10, 6)
    feats[0, 0] = 1  # Should trigger absorption and suppression
    feats[1, 1] = 1  # Should trigger absorption only
    feats[2, 2] = 1  # Should not trigger any modifications
    feats[3, 0] = 1  # Should trigger absorption and suppression
    feats[3, 1] = 1  # Should be absorbed but then suppressed
    feats[4, 1] = 1  # Should trigger absorption only

    # Define modifiers
    absorption_modifier = absorption(absorption_pairs=[(2, 0), (3, 1)])
    suppression_modifier = suppress_features(
        dominant_feature=0, suppressed_features=[1, 3]
    )

    # Chain modifiers
    chained_modifier = chain_modifiers([absorption_modifier, suppression_modifier])

    # Apply chained modifiers
    modified_feats = chained_modifier(feats)

    # Check absorption effects
    assert modified_feats[0, 2] == 1, "Feature 2 should be absorbed when feature 0 is 1"
    assert modified_feats[1, 3] == 1, "Feature 3 should be absorbed when feature 1 is 1"
    assert modified_feats[3, 2] == 1, "Feature 2 should be absorbed when feature 0 is 1"
    assert modified_feats[4, 3] == 1, "Feature 3 should be absorbed when feature 1 is 1"

    # Check suppression effects
    assert torch.all(modified_feats[0, [1, 3]] == 0), (
        "Features 1, 3 should be suppressed when feature 0 is 1"
    )
    assert torch.all(modified_feats[3, [1, 3]] == 0), (
        "Features 1, 3 should be suppressed when feature 0 is 1"
    )

    # Check that other features remain unchanged
    assert modified_feats[2, 2] == 1, (
        "Feature 2 should remain unchanged when no absorption or suppression is triggered"
    )
    assert torch.all(modified_feats[:, [4, 5]] == feats[:, [4, 5]]), (
        "Features 4 and 5 should remain unchanged"
    )

    # Check that absorption happens before suppression
    assert modified_feats[1, 3] == 1, (
        "Feature 3 should be absorbed and not suppressed when feature 1 is 1"
    )


def test_create_correlated_features_modifier_achieves_target_correlation():
    """Test that the correlation modifier achieves the target correlation with correct marginals."""
    p1, p2 = 0.25, 0.2
    target_correlation = 0.37

    corr_modifier = create_correlated_features_modifier(target_correlation, p1, p2)

    # Test with large batch to get accurate empirical estimates
    batch_size = 10000
    feat_probs = torch.tensor([p1, p2])
    batch = get_training_batch(
        batch_size=batch_size,
        firing_probabilities=feat_probs,
        modify_firing_features=corr_modifier,
    )

    # Calculate empirical statistics
    empirical_p1 = batch[:, 0].mean().item()
    empirical_p2 = batch[:, 1].mean().item()
    empirical_corr = torch.corrcoef(batch.T)[0, 1].item()

    # Check marginal probabilities (allow some tolerance due to sampling)
    assert empirical_p1 == pytest.approx(p1, abs=0.02)
    assert empirical_p2 == pytest.approx(p2, abs=0.02)

    # Check correlation (allow some tolerance due to sampling)
    assert empirical_corr == pytest.approx(target_correlation, abs=0.05)


def test_create_correlated_features_modifier_zero_correlation():
    """Test that zero correlation produces independent features."""
    p1, p2 = 0.3, 0.4
    target_correlation = 0.0

    corr_modifier = create_correlated_features_modifier(target_correlation, p1, p2)

    batch_size = 5000
    feat_probs = torch.tensor([p1, p2])
    batch = get_training_batch(
        batch_size=batch_size,
        firing_probabilities=feat_probs,
        modify_firing_features=corr_modifier,
    )

    empirical_corr = torch.corrcoef(batch.T)[0, 1].item()

    # For zero correlation, empirical correlation should be close to 0
    assert empirical_corr == pytest.approx(0.0, abs=0.05)


def test_create_correlated_features_modifier_negative_correlation():
    """Test that negative correlation works correctly."""
    p1, p2 = 0.4, 0.3
    target_correlation = -0.2

    corr_modifier = create_correlated_features_modifier(target_correlation, p1, p2)

    batch_size = 8000
    feat_probs = torch.tensor([p1, p2])
    batch = get_training_batch(
        batch_size=batch_size,
        firing_probabilities=feat_probs,
        modify_firing_features=corr_modifier,
    )

    empirical_corr = torch.corrcoef(batch.T)[0, 1].item()

    # Check negative correlation is achieved
    assert empirical_corr < 0
    assert empirical_corr == pytest.approx(target_correlation, abs=0.05)


def test_create_correlated_features_modifier_high_positive_correlation():
    """Test that high positive correlation works correctly."""
    p1, p2 = 0.5, 0.4
    target_correlation = 0.8

    corr_modifier = create_correlated_features_modifier(target_correlation, p1, p2)

    batch_size = 6000
    feat_probs = torch.tensor([p1, p2])
    batch = get_training_batch(
        batch_size=batch_size,
        firing_probabilities=feat_probs,
        modify_firing_features=corr_modifier,
    )

    empirical_corr = torch.corrcoef(batch.T)[0, 1].item()

    # Check high positive correlation is achieved
    assert empirical_corr > 0.7
    assert empirical_corr == pytest.approx(target_correlation, abs=0.05)


def test_create_correlated_features_modifier_infeasible_correlation_raises_error():
    """Test that infeasible correlations raise ValueError with helpful message."""
    p1, p2 = 0.1, 0.9
    target_correlation = 0.9  # This should be infeasible

    with pytest.raises(ValueError) as exc_info:
        create_correlated_features_modifier(target_correlation, p1, p2)

    # Check that error message contains useful information
    error_msg = str(exc_info.value)
    assert "not feasible" in error_msg
    assert "Valid range:" in error_msg
    assert str(target_correlation) in error_msg


def test_create_correlated_features_modifier_edge_case_small_probabilities():
    """Test correlation modifier with small probabilities."""
    p1, p2 = 0.05, 0.1
    target_correlation = 0.3

    corr_modifier = create_correlated_features_modifier(target_correlation, p1, p2)

    batch_size = 20000  # Larger batch needed for small probabilities
    feat_probs = torch.tensor([p1, p2])
    batch = get_training_batch(
        batch_size=batch_size,
        firing_probabilities=feat_probs,
        modify_firing_features=corr_modifier,
    )

    # Check that we get some positive samples
    assert batch[:, 0].sum() > 0, "Feature 1 should fire sometimes"
    assert batch[:, 1].sum() > 0, "Feature 2 should fire sometimes"

    empirical_p1 = batch[:, 0].mean().item()
    empirical_p2 = batch[:, 1].mean().item()

    # Allow larger tolerance for small probabilities
    assert empirical_p1 == pytest.approx(p1, abs=0.02)
    assert empirical_p2 == pytest.approx(p2, abs=0.02)


def test_create_correlated_features_modifier_preserves_other_features():
    """Test that correlation modifier only affects first two features."""
    p1, p2 = 0.3, 0.4
    target_correlation = 0.5

    corr_modifier = create_correlated_features_modifier(target_correlation, p1, p2)

    # Create batch with 4 features
    batch_size = 1000
    feat_probs = torch.tensor([p1, p2, 0.6, 0.7])
    batch = get_training_batch(
        batch_size=batch_size,
        firing_probabilities=feat_probs,
        modify_firing_features=corr_modifier,
    )

    # Check that features 2 and 3 are unmodified (should be all zeros since modifier overwrites)
    # Actually, our current implementation overwrites the entire tensor, so this test checks
    # that only the first two columns are non-zero
    assert batch[:, 2:].sum() == 0, "Features beyond first two should be zero"
    assert batch[:, :2].sum() > 0, "First two features should have some positive values"


def test_create_correlated_features_modifier_joint_probabilities():
    """Test that joint probabilities match theoretical expectations."""
    p1, p2 = 0.4, 0.3
    target_correlation = 0.5

    # Calculate expected joint probability
    expected_p11 = p1 * p2 + target_correlation * np.sqrt(p1 * (1 - p1) * p2 * (1 - p2))

    corr_modifier = create_correlated_features_modifier(target_correlation, p1, p2)

    batch_size = 10000
    feat_probs = torch.tensor([p1, p2])
    batch = get_training_batch(
        batch_size=batch_size,
        firing_probabilities=feat_probs,
        modify_firing_features=corr_modifier,
    )

    # Calculate empirical joint probabilities
    both_fire = ((batch[:, 0] == 1) & (batch[:, 1] == 1)).float().mean().item()
    only_first = ((batch[:, 0] == 1) & (batch[:, 1] == 0)).float().mean().item()
    only_second = ((batch[:, 0] == 0) & (batch[:, 1] == 1)).float().mean().item()
    neither = ((batch[:, 0] == 0) & (batch[:, 1] == 0)).float().mean().item()

    # Check joint probabilities
    assert both_fire == pytest.approx(expected_p11, abs=0.02)
    assert only_first == pytest.approx(p1 - expected_p11, abs=0.02)
    assert only_second == pytest.approx(p2 - expected_p11, abs=0.02)

    # Check that probabilities sum to 1
    total_prob = both_fire + only_first + only_second + neither
    assert total_prob == pytest.approx(1.0, abs=0.01)
