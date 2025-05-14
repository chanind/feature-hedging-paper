from typing import Any, TypedDict, TypeVar

from hedging_paper.saes.base_sae import BaseSAEConfig, BaseSAERunnerConfig


class BaseSAERunnerConfigDict(TypedDict, total=False):
    model_name: str
    hook_name: str
    hook_layer: int
    hook_head_index: int | None
    dataset_path: str
    dataset_trust_remote_code: bool
    is_dataset_tokenized: bool
    use_cached_activations: bool
    d_in: int
    l1_coefficient: float
    lp_norm: float
    lr: float
    train_batch_size_tokens: int
    context_size: int
    feature_sampling_window: int
    dead_feature_threshold: float
    dead_feature_window: int
    n_batches_in_buffer: int
    training_tokens: int
    store_batch_size_prompts: int
    normalize_sae_decoder: bool
    scale_sparsity_penalty_by_decoder_norm: bool
    init_encoder_as_decoder_transpose: bool
    log_to_wandb: bool
    wandb_project: str
    wandb_entity: str
    wandb_log_frequency: int
    device: str
    seed: int
    checkpoint_path: str
    dtype: str
    prepend_bos: bool


T = TypeVar("T", bound=BaseSAERunnerConfig)
K = TypeVar("K", bound=BaseSAEConfig)


def build_runner_cfg(
    runner_cfg_cls: type[T] = BaseSAERunnerConfig,
    **kwargs: Any,
) -> T:
    """
    Helper to create a mock instance of BaseSAERunnerConfig.
    """
    mock_config_dict: BaseSAERunnerConfigDict = {
        "model_name": "test",
        "hook_name": "blocks.0.hook_mlp_out",
        "hook_layer": 0,
        "hook_head_index": None,
        "dataset_path": "test",
        "dataset_trust_remote_code": True,
        "is_dataset_tokenized": False,
        "use_cached_activations": False,
        "normalize_sae_decoder": False,
        "scale_sparsity_penalty_by_decoder_norm": True,
        "init_encoder_as_decoder_transpose": True,
        "d_in": 64,
        "l1_coefficient": 2e-3,
        "lp_norm": 1,
        "lr": 2e-4,
        "train_batch_size_tokens": 4,
        "context_size": 6,
        "feature_sampling_window": 50,
        "dead_feature_threshold": 1e-7,
        "dead_feature_window": 1000,
        "n_batches_in_buffer": 2,
        "training_tokens": 1_000_000,
        "store_batch_size_prompts": 4,
        "log_to_wandb": False,
        "wandb_project": "test_project",
        "wandb_entity": "test_entity",
        "wandb_log_frequency": 10,
        "device": "cpu",
        "seed": 24,
        "dtype": "float32",
        "prepend_bos": True,
    }

    for key, value in kwargs.items():
        mock_config_dict[key] = value

    mock_config = runner_cfg_cls(**mock_config_dict)

    # reset checkpoint path (as we add an id to each each time)
    mock_config.checkpoint_path = (
        "test/checkpoints"
        if "checkpoint_path" not in kwargs
        else kwargs["checkpoint_path"]
    )

    return mock_config


def build_sae_cfg(
    runner_cfg_cls: type[T] = BaseSAERunnerConfig,
    sae_cfg_cls: type[K] = BaseSAEConfig,
    **kwargs: Any,
) -> K:
    runner_cfg = build_runner_cfg(runner_cfg_cls=runner_cfg_cls, **kwargs)
    return sae_cfg_cls.from_sae_runner_config(runner_cfg)  # type: ignore
