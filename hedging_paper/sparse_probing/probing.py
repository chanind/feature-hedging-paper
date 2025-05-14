"""Utility functions for sparse probing."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from dataclasses_json import DataClassJsonMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor


def rescale_latent_activations(
    latent_activations: Tensor, max_samples: int | None = None
) -> Tensor:
    """
    Rescale latent acts to have mean firing magnitude of 1.0
    """
    activations = (
        latent_activations[:max_samples] if max_samples else latent_activations
    )
    mean_act_mags = activations.mean(dim=0) * (
        # converting to "bool" as a way to count non-zero activations with less memory
        activations.shape[0] / activations.bool().sum(dim=0).clamp(1)
    )
    return latent_activations / mean_act_mags.clamp(1)


def get_latents_by_activation_difference(
    latent_activations: Tensor,
    labels: list[bool] | Tensor,
    descending: bool = True,
) -> list[int]:
    """
    Find SAE latent indices sorted by mean activation difference between classes.

    Args:
        activations: Tensor of activations (sample x dim_model).
        labels: Binary labels (True/False) per sample.
        sae: Optional SAE model to transform activations to latents.
                If None, assumes activations are already latents.
        descending: Sort by descending difference (True) or ascending (False).

    Returns:
        List of indices sorted by mean activation difference.
    """
    # Convert labels to tensor if they're a list
    if isinstance(labels, list):
        labels = torch.tensor(labels, dtype=torch.bool)

    # Ensure labels are boolean
    if labels.dtype != torch.bool:
        labels = labels.bool()

    # Transform activations to latents if SAE is provided

    # Get positive and negative samples
    pos_samples = latent_activations[labels]
    neg_samples = latent_activations[~labels]

    # Calculate mean activation per latent for each class
    pos_means = pos_samples.mean(dim=0)
    neg_means = neg_samples.mean(dim=0)

    # Calculate differences
    differences = (pos_means - neg_means).abs()

    # Sort indices by difference
    sorted_indices = torch.argsort(differences, descending=descending).tolist()

    return sorted_indices


@torch.no_grad()
def train_binary_probe(
    train_activations: Tensor,
    train_labels: list[bool] | Tensor,
    max_iter: int = 1000,
    solver: Literal[
        "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
    ] = "lbfgs",
    # Use a tuple to ensure immutability
    C_values: Sequence[float] = tuple(np.logspace(5, -5, 10).tolist()),
) -> LogisticRegression:
    if isinstance(train_labels, list):
        train_labels = torch.tensor(train_labels, dtype=torch.bool)
    acts_np = train_activations.cpu().float().numpy()
    labels_np = train_labels.cpu().int().numpy()

    # Train probe
    best_auc = -1
    best_probe = None

    for C in C_values:
        # Create probe
        probe = LogisticRegression(
            solver=solver,
            penalty="l2",
            C=C,
            max_iter=max_iter,
            class_weight="balanced",
        )

        # Train probe
        probe.fit(acts_np, labels_np)

        # Evaluate on validation set
        y_pred_proba = probe.predict_proba(acts_np)[:, 1]
        auc = roc_auc_score(labels_np, y_pred_proba)

        if auc > best_auc:
            best_auc = auc
            best_probe = probe

    assert best_probe is not None
    return best_probe


@dataclass
class ProbeEvaluationResults(DataClassJsonMixin):
    """Results from probe training."""

    n_test: int
    n_pos: int
    n_neg: int
    auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass
class KSparseProbeResults(DataClassJsonMixin):
    """Results from training and evaluating sparse probes."""

    eval_results: ProbeEvaluationResults
    k: int
    feature_indices: list[int]


@torch.no_grad()
def evaluate_probe(
    probe: LogisticRegression,
    test_activations: Tensor,
    test_labels: list[bool | int] | Tensor,
) -> ProbeEvaluationResults:
    X_test = test_activations.cpu().float().numpy()
    if isinstance(test_labels, list):
        test_labels = torch.tensor(test_labels, dtype=torch.bool)
    y_test = test_labels.cpu().int().numpy()

    # Evaluate best model
    y_pred_proba = probe.predict_proba(X_test)[:, 1]
    y_pred = probe.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)  # type: ignore
    recall = recall_score(y_test, y_pred, zero_division=0)  # type: ignore
    f1 = f1_score(y_test, y_pred, zero_division=0)  # type: ignore
    n_test = len(y_test)
    n_pos = y_test.sum().item()
    n_neg = n_test - n_pos

    return ProbeEvaluationResults(
        n_test=n_test,
        n_pos=n_pos,
        n_neg=n_neg,
        auc=float(auc),
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


@torch.no_grad()
def train_and_evaluate_k_sparse_probes(
    latent_activations: Tensor,
    labels: list[bool | int] | Tensor,
    k_values: list[int],
    train_ratio: float = 0.8,
    max_iter: int = 1000,
    random_seed: int = 42,
    max_rescale_samples: int | None = None,
    rescale_before_mean_diff: bool = True,
    solver: Literal[
        "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
    ] = "lbfgs",
) -> dict[int, KSparseProbeResults]:
    """
    Train and evaluate probes using different numbers of top latents.

    Args:
        sae: The sparse autoencoder model
        activations: Tensor of activations (sample x dim_model)
        labels: Binary labels (True/False) per sample
        k_values: List of k values (number of top latents) to evaluate
        train_ratio: Ratio of data to use for training (remainder for testing)
        max_iter: Maximum iterations for logistic regression
        C_values: Regularization values to try for logistic regression
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary mapping k values to evaluation results
    """
    # Convert labels to tensor if they're a list
    if isinstance(labels, list):
        labels = torch.tensor(labels, dtype=torch.bool)

    # Ensure labels are boolean
    if labels.dtype != torch.bool:
        labels = labels.bool()

    # Split data into train and test sets
    num_samples = latent_activations.shape[0]
    num_train = int(train_ratio * num_samples)

    # Create random permutation for splitting
    generator = torch.Generator().manual_seed(random_seed)
    indices = torch.randperm(num_samples, generator=generator)

    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    train_latents = latent_activations[train_indices]
    train_labels = labels[train_indices]

    test_latents = latent_activations[test_indices].cpu()
    test_labels_subset = labels[test_indices].cpu()

    mean_diff_train_latents = train_latents
    if rescale_before_mean_diff:
        mean_diff_train_latents = rescale_latent_activations(
            mean_diff_train_latents, max_samples=max_rescale_samples
        )

    train_latents = train_latents.cpu()
    train_labels = train_labels.cpu()

    # Get latents sorted by activation difference
    sorted_latents = get_latents_by_activation_difference(
        latent_activations=mean_diff_train_latents,
        labels=train_labels,
        descending=True,
    )

    # Train and evaluate probes for each k
    results: dict[int, KSparseProbeResults] = {}

    for k in k_values:
        # Select top k latents
        top_k_latents = sorted_latents[:k]

        # Extract features using only top k latents
        train_features = train_latents[:, top_k_latents]
        test_features = test_latents[:, top_k_latents]

        # Train probe
        probe = train_binary_probe(
            train_activations=train_features,
            train_labels=train_labels,
            max_iter=max_iter,
            solver=solver,
        )

        # Evaluate probe
        eval_results = evaluate_probe(
            probe=probe,
            test_activations=test_features,
            test_labels=test_labels_subset,
        )

        results[k] = KSparseProbeResults(
            eval_results=eval_results,
            k=k,
            feature_indices=top_k_latents,
        )

    return results
