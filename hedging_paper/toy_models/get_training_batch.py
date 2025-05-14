from collections.abc import Iterable
from typing import Callable, NamedTuple

import torch

from hedging_paper.util import DEFAULT_DEVICE

######################################################################

# OPTIONS FOR modify_firing_features:


def mutually_exclusive_features(feats: torch.Tensor) -> torch.Tensor:
    # if feats 0 and feats 1,
    # features 0 and 1 are exlusive, features 2 and 3 are exclusive
    tiebreaks = torch.bernoulli(torch.ones(feats.shape[0], 2) * 0.5).to(feats.device)
    tiebreak_locs_0 = torch.nonzero(feats[:, 0] * feats[:, 1])
    tiebreak_locs_1 = torch.nonzero(feats[:, 2] * feats[:, 3])
    tiebreak_locs_2 = torch.nonzero(feats[:, 4] * feats[:, 5])

    feats[tiebreak_locs_0, 0] = 1 - tiebreaks[tiebreak_locs_0, 0]
    feats[tiebreak_locs_0, 1] = tiebreaks[tiebreak_locs_0, 0]
    feats[tiebreak_locs_1, 2] = 1 - tiebreaks[tiebreak_locs_1, 1]
    feats[tiebreak_locs_1, 3] = tiebreaks[tiebreak_locs_1, 1]
    feats[tiebreak_locs_2, 4] = 1 - tiebreaks[tiebreak_locs_2, 0]
    feats[tiebreak_locs_2, 5] = tiebreaks[tiebreak_locs_2, 0]

    feats[feats.sum(dim=-1) < 3, :] = 0
    return feats


def suppress_features(
    dominant_feature: int, suppressed_features: Iterable[int]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    If the dominant feature is 1, set all the suppressed features to 0
    """

    def suppress_features_cb(feats: torch.Tensor):
        for suppressed_feature in suppressed_features:
            feats[:, suppressed_feature] = torch.where(
                feats[:, dominant_feature] == 1,
                torch.tensor(0.0, device=feats.device),
                feats[:, suppressed_feature],
            )
        return feats

    return suppress_features_cb


class AbsorptionPair(NamedTuple):
    parent: int
    child: int


def absorption(
    absorption_pairs: Iterable[tuple[int, int]],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    absorption_pairs is a list of pairs (parent, child) where if the child is 1, the parent is set to 1
    """

    def absorption_cb(feats: torch.Tensor):
        for parent, child in absorption_pairs:
            feats[:, parent] = torch.where(
                feats[:, child] == 1,
                torch.tensor(1.0, device=feats.device),
                feats[:, parent],
            )
        return feats

    return absorption_cb


def chain_modifiers(
    modifiers: Iterable[Callable[[torch.Tensor], torch.Tensor]],
) -> Callable[[torch.Tensor], torch.Tensor]:
    def chain_modifiers_cb(feats: torch.Tensor):
        for modifier in modifiers:
            feats = modifier(feats)
        return feats

    return chain_modifiers_cb


######################################################################


def get_training_batch(
    batch_size: int,
    firing_probabilities: torch.Tensor,  # these are the independent probabilities of each feature firing
    std_firing_magnitudes: (
        torch.Tensor | None
    ) = None,  # If not provided, the stdev of magnitudes will be 0
    mean_firing_magnitudes: (
        torch.Tensor | None
    ) = None,  # If not provided, mean will be 1.0
    device: torch.device = DEFAULT_DEVICE,
    modify_firing_features: Callable[[torch.Tensor], torch.Tensor] | None = None,
):
    firing_features = torch.bernoulli(
        firing_probabilities.unsqueeze(0).expand(batch_size, -1).to(device)
    )
    if std_firing_magnitudes is None:
        std_firing_magnitudes = torch.zeros_like(firing_probabilities)
    if mean_firing_magnitudes is None:
        mean_firing_magnitudes = torch.ones_like(firing_probabilities)
    mean_firing_magnitudes = mean_firing_magnitudes.to(device)
    if modify_firing_features is not None:
        firing_features = modify_firing_features(firing_features)
    firing_features = firing_features.to(device)
    firing_magnitude_delta = torch.normal(
        torch.zeros_like(firing_probabilities)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .to(device),
        std_firing_magnitudes.unsqueeze(0).expand(batch_size, -1).to(device),
    )
    firing_magnitude_delta[firing_features == 0] = 0
    return (firing_features * mean_firing_magnitudes + firing_magnitude_delta).relu()
