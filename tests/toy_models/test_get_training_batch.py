import torch

from hedging_paper.toy_models.get_training_batch import (
    AbsorptionPair,
    absorption,
    chain_modifiers,
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
