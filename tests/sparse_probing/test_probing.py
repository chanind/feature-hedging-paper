"""Tests for sparse probing utility functions."""

import torch
from sklearn.metrics import roc_auc_score

from hedging_paper.sparse_probing.probing import (
    ProbeEvaluationResults,
    evaluate_probe,
    get_latents_by_activation_difference,
    rescale_latent_activations,
    train_and_evaluate_k_sparse_probes,
    train_binary_probe,
)


def test_get_latents_by_activation_difference_with_tensor_labels():
    # Create test data
    activations = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # True sample
            [0.5, 1.5, 2.5, 3.5],  # True sample
            [0.1, 0.2, 0.3, 0.4],  # False sample
            [0.2, 0.3, 0.4, 0.5],  # False sample
        ]
    )

    labels = torch.tensor([True, True, False, False])

    # Expected differences: [0.6, 1.5, 2.4, 3.3]
    # Expected sorting (descending): [3, 2, 1, 0]

    sorted_indices = get_latents_by_activation_difference(activations, labels)
    assert sorted_indices == [3, 2, 1, 0]

    # Test ascending order
    sorted_indices = get_latents_by_activation_difference(
        activations, labels, descending=False
    )
    assert sorted_indices == [0, 1, 2, 3]


def test_rescale_latent_activations():
    # Create test data with some zeros and non-zeros
    latent_activations = torch.tensor(
        [
            [0.0, 2.0, 0.0, 4.0],
            [1.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 2.0, 6.0],
            [2.0, 4.0, 0.0, 0.0],
        ]
    )

    # Calculate expected results manually
    # For each column, compute mean of non-zero values:
    # Col 0: (1.0 + 2.0) / 2 = 1.5
    # Col 1: (2.0 + 4.0) / 2 = 3.0
    # Col 2: (3.0 + 2.0) / 2 = 2.5
    # Col 3: (4.0 + 6.0) / 2 = 5.0

    # Expected output should divide each value by its column's mean non-zero value
    expected = torch.tensor(
        [
            [0.0, 2.0 / 3.0, 0.0, 4.0 / 5.0],
            [1.0 / 1.5, 0.0, 3.0 / 2.5, 0.0],
            [0.0, 0.0, 2.0 / 2.5, 6.0 / 5.0],
            [2.0 / 1.5, 4.0 / 3.0, 0.0, 0.0],
        ]
    )

    # Apply the function
    result = rescale_latent_activations(latent_activations)

    # Check results
    assert torch.allclose(result, expected, rtol=1e-5)


def test_rescale_latent_activations_with_zeros():
    # Test with all zeros in a column
    latent_activations_with_zeros = torch.tensor(
        [
            [1.0, 0.0, 3.0],
            [2.0, 0.0, 4.0],
            [3.0, 0.0, 5.0],
        ]
    )

    # The second column has all zeros, so the mean should be clamped to 1.0
    # to avoid division by zero
    expected_with_zeros = torch.tensor(
        [
            [1.0 / 2.0, 0.0, 3.0 / 4.0],
            [2.0 / 2.0, 0.0, 4.0 / 4.0],
            [3.0 / 2.0, 0.0, 5.0 / 4.0],
        ]
    )

    result_with_zeros = rescale_latent_activations(latent_activations_with_zeros)
    assert torch.allclose(result_with_zeros, expected_with_zeros, rtol=1e-5)


def test_get_latents_by_activation_difference_with_list_labels():
    # Create test data
    activations = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # True sample
            [0.5, 1.5, 2.5, 3.5],  # True sample
            [0.1, 0.2, 0.3, 0.4],  # False sample
            [0.2, 0.3, 0.4, 0.5],  # False sample
        ]
    )

    labels = [True, True, False, False]

    sorted_indices = get_latents_by_activation_difference(activations, labels)
    assert sorted_indices == [3, 2, 1, 0]


def test_get_latents_by_activation_difference_with_int_labels():
    # Create test data
    activations = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # 1 sample
            [0.5, 1.5, 2.5, 3.5],  # 1 sample
            [0.1, 0.2, 0.3, 0.4],  # 0 sample
            [0.2, 0.3, 0.4, 0.5],  # 0 sample
        ]
    )

    labels = torch.tensor([1, 1, 0, 0])

    sorted_indices = get_latents_by_activation_difference(activations, labels)
    assert sorted_indices == [3, 2, 1, 0]


def test_train_binary_probe_perfect_separation():
    # Create perfectly separable data
    activations = torch.tensor(
        [
            [1.0, 0.1],  # Class 1
            [1.1, 0.2],  # Class 1
            [0.1, 1.0],  # Class 0
            [0.2, 1.1],  # Class 0
        ]
    )

    labels = torch.tensor([1, 1, 0, 0])

    # Train with a single C value to make test deterministic
    model = train_binary_probe(activations, labels, C_values=[1.0])

    # Check predictions
    acts_np = activations.cpu().numpy()
    predictions = model.predict(acts_np)

    assert (predictions == labels.cpu().numpy()).all()

    # Check coefficients - should put positive weight on first feature for class 1
    assert model.coef_[0, 0] > 0
    assert model.coef_[0, 1] < 0


def test_train_binary_probe_with_list_labels():
    activations = torch.tensor(
        [
            [1.0, 0.1],  # True sample
            [1.1, 0.2],  # True sample
            [0.1, 1.0],  # False sample
            [0.2, 1.1],  # False sample
        ]
    )

    labels = [True, True, False, False]

    # Train with a single C value to make test deterministic
    model = train_binary_probe(activations, labels, C_values=[1.0])

    # Check predictions
    acts_np = activations.cpu().numpy()
    predictions = model.predict(acts_np)
    expected = [1, 1, 0, 0]

    assert (predictions == expected).all()


def test_train_binary_probe_best_c_selection():
    # Create data where regularization matters
    # More features than samples to encourage overfitting
    torch.manual_seed(42)
    activations = torch.randn(10, 20)

    # Create labels with some noise
    true_weights = torch.zeros(20)
    true_weights[0] = 1.0  # Only first feature matters
    logits = activations @ true_weights
    probs = 1 / (1 + torch.exp(-logits))
    labels = (probs > 0.5).int()

    # Train with multiple C values
    c_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    best_model = train_binary_probe(activations, labels, C_values=c_values)

    # The best model should have a reasonable C value (not too high or low)
    # and should put most weight on the first feature
    assert abs(best_model.coef_[0, 0]) > abs(best_model.coef_[0, 1:]).mean()

    # Train individual models to verify best selection
    best_auc = -1
    for C in c_values:
        model = train_binary_probe(activations, labels, C_values=[C])
        y_pred = model.predict_proba(activations.numpy())[:, 1]
        auc = roc_auc_score(labels.numpy(), y_pred)

        if auc > best_auc:
            best_auc = auc

    # The selected model should achieve the best AUC
    y_pred_best = best_model.predict_proba(activations.numpy())[:, 1]
    best_model_auc = roc_auc_score(labels.numpy(), y_pred_best)

    assert abs(best_model_auc - best_auc) < 1e-6


def test_evaluate_probe():
    # Create simple test data with perfect separation
    activations = torch.tensor(
        [
            [1.0, 0.1],  # Class 1
            [1.1, 0.2],  # Class 1
            [0.1, 1.0],  # Class 0
            [0.2, 1.1],  # Class 0
        ]
    )

    labels = torch.tensor([1, 1, 0, 0])

    # Train a real probe on this data
    probe = train_binary_probe(activations, labels, C_values=[1.0])

    # Evaluate on the same data (should be perfect since data is separable)
    results = evaluate_probe(probe, activations, labels)

    # Check that results are as expected for perfect predictions
    assert isinstance(results, ProbeEvaluationResults)
    assert results.accuracy == 1.0
    assert results.precision == 1.0
    assert results.recall == 1.0
    assert results.f1 == 1.0
    assert results.auc == 1.0
    assert results.n_test == 4
    assert results.n_pos == 2
    assert results.n_neg == 2

    # Test with list labels
    list_labels = [1, 1, 0, 0]
    list_results = evaluate_probe(probe, activations, list_labels)
    assert list_results.accuracy == 1.0

    # Test with imperfect predictions by creating harder data
    # Create data where classes overlap
    noisy_activations = torch.tensor(
        [
            [0.6, 0.4],  # Class 1, but close to decision boundary
            [1.1, 0.2],  # Class 1, clear
            [0.4, 0.6],  # Class 0, but close to decision boundary
            [0.2, 1.1],  # Class 0, clear
        ]
    )

    noisy_labels = torch.tensor([1, 1, 0, 0])

    # Train on the original clean data
    probe = train_binary_probe(activations, labels, C_values=[1.0])

    # Evaluate on the noisy data
    noisy_results = evaluate_probe(probe, noisy_activations, noisy_labels)

    # Metrics should be less than perfect but still good
    assert 0.5 <= noisy_results.accuracy <= 1.0
    assert 0.5 <= noisy_results.precision <= 1.0
    assert 0.5 <= noisy_results.recall <= 1.0
    assert 0.5 <= noisy_results.f1 <= 1.0
    assert 0.5 <= noisy_results.auc <= 1.0


def test_train_and_evaluate_k_sparse_probes():
    # Create synthetic activations and labels
    torch.manual_seed(42)
    n_samples = 100

    # Create activations that have a pattern where the first 10 dimensions
    # are predictive of the label
    activations = torch.randn(n_samples, 37)

    # Create labels based on a pattern in the first few dimensions
    pattern = torch.zeros(37)
    pattern[:10] = torch.linspace(1.0, 0.1, 10)  # Decreasing importance

    # Generate labels with some noise
    logits = activations @ pattern
    probs = 1 / (1 + torch.exp(-logits))
    labels = (probs > 0.5).bool()

    # Test with a few k values
    k_values = [1, 5, 10, 20]

    # Run the function
    results = train_and_evaluate_k_sparse_probes(
        latent_activations=activations,
        labels=labels,
        k_values=k_values,
        train_ratio=0.8,
        random_seed=42,
    )

    # Check that we got results for each k value
    assert set(results.keys()) == set(k_values)

    # Check that each result has the expected structure
    for k, result in results.items():
        # All metrics should be between 0 and 1
        assert 0 <= result.eval_results.accuracy <= 1
        assert 0 <= result.eval_results.precision <= 1
        assert 0 <= result.eval_results.recall <= 1
        assert 0 <= result.eval_results.f1 <= 1
        assert 0 <= result.eval_results.auc <= 1

    # Performance should generally improve as k increases
    # (at least for the first few k values since we designed the data that way)
    assert (
        results[5].eval_results.accuracy >= results[1].eval_results.accuracy - 0.1
    )  # Allow some noise

    # Test with list labels
    list_labels = labels.tolist()
    list_results = train_and_evaluate_k_sparse_probes(
        latent_activations=activations,
        labels=list_labels,
        k_values=[1, 5],
        train_ratio=0.8,
        random_seed=42,
    )

    assert len(list_results) == 2
