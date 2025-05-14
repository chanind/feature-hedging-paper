from dataclasses import fields
from typing import Any, TypeVar

import torch
from sae_lens import LanguageModelSAERunnerConfig, TrainingSAEConfig

T = TypeVar("T", bound=TrainingSAEConfig)


def get_extra_field_names(sub_config_cls: type[TrainingSAEConfig]) -> set[str]:
    return {f.name for f in fields(sub_config_cls)} - {
        f.name for f in fields(TrainingSAEConfig)
    }


def get_extra_field_items(sub_config: TrainingSAEConfig) -> dict[str, Any]:
    return {
        f: getattr(sub_config, f) for f in get_extra_field_names(sub_config.__class__)
    }


def from_sae_runner_config(
    runner_cfg: LanguageModelSAERunnerConfig, sub_config_class: type[T]
) -> T:
    base_cfg = TrainingSAEConfig.from_sae_runner_config(runner_cfg)
    sae_cfg = sub_config_class(**base_cfg.to_dict())
    for field_name in get_extra_field_names(sub_config_class):
        setattr(sae_cfg, field_name, getattr(runner_cfg, field_name))
    return sae_cfg


def running_update(
    running_val: torch.Tensor,
    new_val: torch.Tensor,
    alpha: float = 0.95,
    negative_means_uninitialized: bool = False,
) -> torch.Tensor:
    if negative_means_uninitialized and running_val < 0:
        return new_val.to(running_val.dtype)
    else:
        return alpha * running_val + (1 - alpha) * new_val.to(running_val.dtype)
