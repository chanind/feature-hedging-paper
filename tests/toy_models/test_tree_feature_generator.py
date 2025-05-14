import json
from pathlib import Path

import pytest
import torch

from hedging_paper.toy_models.tree_feature_generator import (
    TreeFeatureGenerator,
    _sample_is_active,
    validate_tree_dict,
)

TREE_JSON_FILE = Path(__file__).parent / "fixtures" / "tree.json"


def test_tree_dict():
    with open(TREE_JSON_FILE) as f:
        tree_dict = json.load(f)
    validate_tree_dict(tree_dict)


def test_sample_is_active_samples_according_to_prob():
    acts = _sample_is_active(
        0.25,
        batch_size=1000,
        parent_feats_mask=None,
        force_active_mask=None,
        force_inactive_mask=None,
    )
    assert acts.shape == (1000,)
    assert (acts.sum().item() / acts.shape[0]) == pytest.approx(0.25, abs=0.05)


def test_sample_is_active_sets_parent_acts_to_zero():
    parent_acts = _sample_is_active(
        0.25,
        batch_size=10_000,
        parent_feats_mask=None,
        force_active_mask=None,
        force_inactive_mask=None,
    )
    acts = _sample_is_active(
        0.25,
        batch_size=10_000,
        parent_feats_mask=parent_acts,
        force_active_mask=None,
        force_inactive_mask=None,
    )
    assert (parent_acts == 0).sum() > 10  # make sure we actually have some entries here
    assert (
        acts[parent_acts == 0].sum() == 0
    )  # acts should be 0 everywhere parent acts is 0
    assert (acts.sum().item() / (parent_acts > 0).sum()) == pytest.approx(
        0.25, abs=0.05
    )


def test_Tree_sample_respects_rules_defined_in_the_tree_json():
    batch_size = 50_000
    with open(TREE_JSON_FILE) as f:
        tree_dict = json.load(f)
    tree = TreeFeatureGenerator.from_json_dict(tree_dict)
    batch = tree.sample(batch_size)
    assert batch.shape == (batch_size, 20)

    # every feature should fire at least once
    assert (batch.sum(dim=0) == 0).sum() == 0

    # the last 8 features fire independently with prob 0.05
    for i in range(8):
        assert batch[:, -i - 1].sum().item() / batch_size == pytest.approx(
            0.05, abs=0.01
        )

    # feats 0, 4, and 8 all fire independently with prob 0.15
    for i in [0, 4, 8]:
        parent_batch = batch[:, i]
        assert parent_batch.sum().item() / batch_size == pytest.approx(0.15, abs=0.01)

        # they each have 3 child features
        for j in range(3):
            child_idx = i + j + 1
            child_batch = batch[:, child_idx]
            # child should never fire unless parent fires
            assert torch.all(child_batch[parent_batch == 0] == 0)
            # each child should fire 20% of the time that the parent fires
            assert child_batch.sum() / parent_batch.sum() == pytest.approx(
                0.2, abs=0.05
            )

        # each child should never fire if another child fires
        # this means all these children should sum to 1 across a row
        assert torch.all((batch[:, i + 1 : i + 4].sum(dim=-1) > 1) == 0)


def test_sample_is_active_respects_force_active_mask():
    batch_size = 10_000
    force_active_mask = torch.zeros(batch_size, dtype=torch.bool)
    force_active_mask[::2] = True  # Set every other sample to be forced active

    acts = _sample_is_active(
        0.25,
        batch_size=batch_size,
        parent_feats_mask=None,
        force_active_mask=force_active_mask,
        force_inactive_mask=None,
    )

    # Check forced active samples are all 1
    assert torch.all(acts[force_active_mask] == 1)

    # Check non-forced samples follow probability distribution
    non_forced = acts[~force_active_mask]
    assert non_forced.sum().item() / non_forced.shape[0] == pytest.approx(
        0.25, abs=0.05
    )


def test_sample_is_active_respects_force_inactive_mask():
    batch_size = 10_000
    force_inactive_mask = torch.zeros(batch_size, dtype=torch.bool)
    force_inactive_mask[::2] = True  # Set every other sample to be forced inactive

    acts = _sample_is_active(
        0.75,  # High probability to better test forced inactivity
        batch_size=batch_size,
        parent_feats_mask=None,
        force_active_mask=None,
        force_inactive_mask=force_inactive_mask,
    )

    # Check forced inactive samples are all 0
    assert torch.all(acts[force_inactive_mask] == 0)

    # Check non-forced samples follow probability distribution
    non_forced = acts[~force_inactive_mask]
    assert non_forced.sum().item() / non_forced.shape[0] == pytest.approx(
        0.75, abs=0.05
    )


def test_sample_is_active_force_masks_override_parent_mask():
    batch_size = 10_000
    parent_feats_mask = torch.ones(batch_size, dtype=torch.bool)
    force_active_mask = torch.zeros(batch_size, dtype=torch.bool)
    force_inactive_mask = torch.zeros(batch_size, dtype=torch.bool)

    # Set different regions for forced active/inactive
    force_active_mask[: batch_size // 3] = True
    force_inactive_mask[batch_size // 3 : batch_size // 2] = True

    acts = _sample_is_active(
        0.5,
        batch_size=batch_size,
        parent_feats_mask=parent_feats_mask,
        force_active_mask=force_active_mask,
        force_inactive_mask=force_inactive_mask,
    )

    # Check forced regions
    assert torch.all(acts[: batch_size // 3] == 1)  # force_active region
    assert torch.all(
        acts[batch_size // 3 : batch_size // 2] == 0
    )  # force_inactive region

    # Check remaining region follows probability with parent mask
    remaining = acts[batch_size // 2 :]
    assert remaining.sum().item() / remaining.shape[0] == pytest.approx(0.5, abs=0.05)


def test_tree_feature_generator_direct_construction():
    # Create a tree with non-mutually exclusive children
    # Structure:
    # Root (0.8)
    # ├── Child1 (0.7)
    # │   ├── GrandChild1 (0.6)
    # │   └── GrandChild2 (0.5)
    # └── Child2 (0.4)
    #     └── GrandChild3 (0.3)

    grandchild1 = TreeFeatureGenerator(active_prob=0.6)
    grandchild2 = TreeFeatureGenerator(active_prob=0.5)
    grandchild3 = TreeFeatureGenerator(active_prob=0.3)

    child1 = TreeFeatureGenerator(
        active_prob=0.7,
        children=[grandchild1, grandchild2],
        mutually_exclusive_children=False,
    )

    child2 = TreeFeatureGenerator(
        active_prob=0.4, children=[grandchild3], mutually_exclusive_children=False
    )

    root = TreeFeatureGenerator(
        active_prob=0.8, children=[child1, child2], mutually_exclusive_children=False
    )

    # Test basic properties
    assert root.n_features == 6  # All nodes are read out by default
    assert root.index == 0
    assert child1.index == 1
    assert grandchild1.index == 2
    assert grandchild2.index == 3
    assert child2.index == 4
    assert grandchild3.index == 5

    # Generate samples and test statistical properties
    batch_size = 10_000
    samples = root.sample(batch_size)

    # Test shape
    assert samples.shape == (batch_size, 6)

    # Test that all values are binary (0 or 1)
    assert torch.all(torch.logical_or(samples == 0, samples == 1))

    # Test root activation probability
    root_acts = samples[:, root.index]
    assert root_acts.mean().item() == pytest.approx(0.8, abs=0.05)

    # Test child1 activation probability (should be 0.7 * 0.8 since parent must be active)
    child1_acts = samples[:, child1.index]
    assert child1_acts.mean().item() == pytest.approx(0.7 * 0.8, abs=0.05)

    # Test child2 activation probability (should be 0.4 * 0.8)
    child2_acts = samples[:, child2.index]
    assert child2_acts.mean().item() == pytest.approx(0.4 * 0.8, abs=0.05)

    # Test grandchild activation probabilities
    gc1_acts = samples[:, grandchild1.index]
    assert gc1_acts.mean().item() == pytest.approx(0.6 * 0.7 * 0.8, abs=0.05)

    gc2_acts = samples[:, grandchild2.index]
    assert gc2_acts.mean().item() == pytest.approx(0.5 * 0.7 * 0.8, abs=0.05)

    gc3_acts = samples[:, grandchild3.index]
    assert gc3_acts.mean().item() == pytest.approx(0.3 * 0.4 * 0.8, abs=0.05)

    # Test conditional probabilities
    # When root is inactive, all children must be inactive
    root_inactive_mask = root_acts == 0
    assert torch.all(samples[root_inactive_mask, 1:] == 0)

    # When child1 is inactive, its grandchildren must be inactive
    child1_inactive_mask = child1_acts == 0
    assert torch.all(samples[child1_inactive_mask, grandchild1.index] == 0)
    assert torch.all(samples[child1_inactive_mask, grandchild2.index] == 0)

    # When child2 is inactive, grandchild3 must be inactive
    child2_inactive_mask = child2_acts == 0
    assert torch.all(samples[child2_inactive_mask, grandchild3.index] == 0)


def test_tree_feature_generator_magnitudes():
    # Create a simple tree with custom magnitude parameters
    root = TreeFeatureGenerator(active_prob=0.8, magnitude_mean=2.0, magnitude_std=0.5)

    # Generate samples
    batch_size = 10_000
    samples = root.sample(batch_size)

    # Test that inactive features are exactly 0
    inactive_mask = samples[:, root.index] == 0
    assert torch.all(samples[inactive_mask, root.index] == 0)

    # Test that active features have positive magnitudes
    active_mask = samples[:, root.index] > 0
    assert torch.all(samples[active_mask, root.index] > 0)

    # Test magnitude statistics
    active_magnitudes = samples[active_mask, root.index]
    assert active_magnitudes.mean().item() == pytest.approx(2.0, abs=0.1)
    assert active_magnitudes.std().item() == pytest.approx(0.5, abs=0.1)


def test_tree_feature_generator_zero_std():
    # Test when std=0, all active features should have exactly the mean magnitude
    root = TreeFeatureGenerator(active_prob=0.8, magnitude_mean=1.5, magnitude_std=0.0)

    batch_size = 1000
    samples = root.sample(batch_size)

    # All active features should have exactly magnitude_mean
    active_mask = samples[:, root.index] > 0
    assert torch.all(samples[active_mask, root.index] == 1.5)
