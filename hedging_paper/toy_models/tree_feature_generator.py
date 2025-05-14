import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jsonschema
import torch

from hedging_paper.util import DEFAULT_DEVICE

TREE_SCHEMA_FILE = Path(__file__).parent / "tree.schema.json"


class TreeFeatureGenerator:
    """
    Hierarchical feature generator

    Based on Noa Nabeshima's Matryoshka SAEs: https://github.com/noanabeshima/matryoshka-saes/blob/main/toy_model.py
    """

    children: Sequence["TreeFeatureGenerator"]
    index: int | None
    next_index: int

    @classmethod
    def from_json_dict(cls, tree_dict: dict[str, Any]):
        validate_tree_dict(tree_dict)

        children = [
            TreeFeatureGenerator.from_json_dict(child_dict)
            for child_dict in tree_dict.get("children", [])
        ]
        return cls(
            active_prob=tree_dict["active_prob"],
            children=children,
            mutually_exclusive_children=tree_dict.get(
                "mutually_exclusive_children", False
            ),
            is_read_out=tree_dict.get("is_read_out", True),
            id=tree_dict.get("id", None),
            magnitude_mean=tree_dict.get("magnitude_mean", 1.0),
            magnitude_std=tree_dict.get("magnitude_std", 0.0),
        )

    def __init__(
        self,
        active_prob: float,
        children: Sequence["TreeFeatureGenerator"] | None = None,
        mutually_exclusive_children: bool = False,
        is_read_out: bool = True,
        id: str | None = None,
        magnitude_mean: float = 1.0,
        magnitude_std: float = 0.0,
    ):
        self.active_prob = active_prob
        self.children = children or []
        self.mutually_exclusive_children = mutually_exclusive_children
        self.is_read_out = is_read_out
        self.index = None
        self.id = id
        self.magnitude_mean = magnitude_mean
        self.magnitude_std = magnitude_std

        if self.mutually_exclusive_children:
            assert len(self.children) >= 2

        self._init_indices()

    def _init_indices(self, start_idx: int = 0) -> int:
        if self.is_read_out:
            self.index = start_idx
            start_idx += 1
        else:
            self.index = None

        for child in self.children:
            start_idx = child._init_indices(start_idx)

        return start_idx

    def __repr__(self, indent=0):
        s = " " * (indent * 2)
        s += (
            str(self.index) + " "
            if self.index is not False
            else " " * len(str(self.next_index)) + " "
        )
        s += "x" if self.mutually_exclusive_children else " "
        s += f" {self.active_prob}"

        for child in self.children:
            s += "\n" + child.__repr__(indent + 2)
        return s

    @property
    def n_features(self):
        count = 1 if self.is_read_out else 0
        for child in self.children:
            count += child.n_features
        return count

    @property
    def child_probs(self):
        return torch.tensor([child.active_prob for child in self.children])

    @torch.no_grad()
    def sample(
        self, batch_size: int, device: torch.device = DEFAULT_DEVICE
    ) -> torch.Tensor:
        batch = torch.zeros((batch_size, self.n_features))
        self._fill_batch(batch)
        return batch.to(device)

    def _fill_batch(
        self,
        batch: torch.Tensor,
        parent_feats_mask: torch.Tensor | None = None,
        force_active_mask: torch.Tensor | None = None,
        force_inactive_mask: torch.Tensor | None = None,
    ):
        batch_size = batch.shape[0]
        is_active = _sample_is_active(
            self.active_prob,
            batch_size=batch_size,
            parent_feats_mask=parent_feats_mask,
            force_active_mask=force_active_mask,
            force_inactive_mask=force_inactive_mask,
        )

        # append something if this is a readout
        if self.is_read_out:
            magnitudes = _sample_magnitudes(
                is_active, self.magnitude_mean, self.magnitude_std
            )
            batch[:, self.index] = magnitudes

        active_child = None
        if self.mutually_exclusive_children:
            active_child = torch.multinomial(
                self.child_probs.expand(batch_size, -1), 1
            ).squeeze(-1)

        for child_idx, child in enumerate(self.children):
            child_force_inactive = (
                None if active_child is None else active_child != child_idx
            )
            child_force_active = (
                None if active_child is None else active_child == child_idx
            )
            child._fill_batch(
                batch,
                parent_feats_mask=is_active,
                force_active_mask=child_force_active,
                force_inactive_mask=child_force_inactive,
            )


def _sample_is_active(
    active_prob,
    batch_size: int,
    parent_feats_mask: torch.Tensor | None,
    force_active_mask: torch.Tensor | None,
    force_inactive_mask: torch.Tensor | None,
):
    is_active = torch.bernoulli(torch.tensor(active_prob).expand(batch_size))
    if force_active_mask is not None:
        is_active[force_active_mask] = 1
    if force_inactive_mask is not None:
        is_active[force_inactive_mask] = 0
    if parent_feats_mask is not None:
        is_active[parent_feats_mask == 0] = 0
    return is_active


def _sample_magnitudes(
    is_active: torch.Tensor,
    magnitude_mean: float,
    magnitude_std: float,
) -> torch.Tensor:
    """Sample magnitudes for active features."""
    batch_size = is_active.shape[0]
    magnitudes = torch.zeros_like(is_active, dtype=torch.float32)

    if magnitude_std > 0:
        # Sample from normal distribution and take absolute value to ensure positive
        active_magnitudes = torch.abs(
            torch.normal(mean=magnitude_mean, std=magnitude_std, size=(batch_size,))
        )
    else:
        # If no std dev, just use the mean
        active_magnitudes = torch.full((batch_size,), magnitude_mean)

    magnitudes[is_active == 1] = active_magnitudes[is_active == 1]
    return magnitudes


def validate_tree_dict(instance):
    with open(TREE_SCHEMA_FILE) as f:
        schema = json.load(f)
    jsonschema.validate(instance, schema)
