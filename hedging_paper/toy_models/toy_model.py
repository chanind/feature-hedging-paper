from typing import Any

import torch
from transformer_lens.hook_points import HookedRootModule

from hedging_paper.toy_models.orthogonalize import orthogonalize


class ToyModel(HookedRootModule):
    def __init__(
        self,
        num_feats: int,
        hidden_dim: int,
        target_cos_sim: float = 0,
        bias: bool = False,
    ):
        super().__init__()
        self.embed = torch.nn.Linear(num_feats, hidden_dim, bias=bias)
        embeddings = orthogonalize(num_feats, hidden_dim, target_cos_sim=target_cos_sim)
        self.embed.weight.data = embeddings.T
        self.setup()

    def forward(self, x: torch.Tensor, **kwargs: Any):  # noqa: ARG002
        return self.embed(x)
