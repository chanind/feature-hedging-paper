import time
from pathlib import Path

import torch
from pytest import approx

from hedging_paper.saes.base_sae import BaseSAE, BaseSAEConfig, BaseSAERunnerConfig
from hedging_paper.saes.batch_topk_sae import BatchTopkSAE
from hedging_paper.train_sae import (
    _load_pretrained_weights,
    find_latest_checkpoint,
    hash_sae_cfg,
    train_sae,
)
from tests.helpers import build_runner_cfg, build_sae_cfg


def test_train_sae(tmp_path: Path):
    runner_cfg = build_runner_cfg(
        model_name="gpt2",
        hook_name="blocks.0.hook_resid_post",
        dataset_path="roneneldan/TinyStories",
        d_in=768,
        d_sae=4,
        training_tokens=100,
        store_batch_size_prompts=2,
        n_batches_in_buffer=2,
        checkpoint_path=str(tmp_path / "checkpoints"),
    )
    sae_cfg = BaseSAEConfig.from_sae_runner_config(runner_cfg)
    sae = BaseSAE(sae_cfg)

    original_W_dec_norm = sae.W_dec.norm()

    sae_stats = train_sae(
        sae=sae,
        cfg=runner_cfg,
    )

    with torch.no_grad():
        assert sae_stats.sparsity.shape == (4,)
        assert sae_stats.l0 > 0
        assert sae_stats.width == 4
        # make sure it actually changed
        assert sae.W_dec.norm() != approx(original_W_dec_norm, rel=1e-3)

    assert (tmp_path / "checkpoints" / "final_100").exists()
    assert (tmp_path / "checkpoints" / "final_100" / "cfg.json").exists()
    assert (tmp_path / "checkpoints" / "final_100" / "sae_weights.safetensors").exists()
    assert (tmp_path / "checkpoints" / "final_100" / "sparsity.safetensors").exists()


def test_train_sae_with_extension(tmp_path: Path):
    original_sae = BaseSAE(build_sae_cfg(d_sae=4, d_in=768))
    original_sae.save_model_final(tmp_path / "original_sae")

    runner_cfg = build_runner_cfg(
        model_name="gpt2",
        hook_name="blocks.0.hook_resid_post",
        dataset_path="roneneldan/TinyStories",
        d_in=768,
        d_sae=4,
        training_tokens=100,
        store_batch_size_prompts=2,
        n_batches_in_buffer=2,
        checkpoint_path=str(tmp_path / "checkpoints"),
        extend_sae_latents=2,
        extend_sae_path=str(tmp_path / "original_sae"),
    )
    sae_cfg = BaseSAEConfig.from_sae_runner_config(runner_cfg)
    sae = BaseSAE(sae_cfg)

    original_W_dec_norm = sae.W_dec.norm()

    sae_stats = train_sae(
        sae=sae,
        cfg=runner_cfg,
    )

    with torch.no_grad():
        assert sae_stats.sparsity.shape == (6,)
        assert sae_stats.l0 > 0
        assert sae_stats.width == 6
        # make sure it actually changed
        assert sae.W_dec.norm() != approx(original_W_dec_norm, rel=1e-3)

    assert (tmp_path / "checkpoints" / "final_100").exists()
    assert (tmp_path / "checkpoints" / "final_100" / "cfg.json").exists()
    assert (tmp_path / "checkpoints" / "final_100" / "sae_weights.safetensors").exists()
    assert (tmp_path / "checkpoints" / "final_100" / "sparsity.safetensors").exists()


def test_train_batch_topk_sae_disabling_bias(tmp_path: Path) -> None:
    runner_cfg = build_runner_cfg(
        BaseSAERunnerConfig,
        model_name="gpt2",
        hook_name="blocks.0.hook_resid_post",
        dataset_path="roneneldan/TinyStories",
        d_in=768,
        d_sae=4,
        training_tokens=1000,
        store_batch_size_prompts=2,
        n_batches_in_buffer=2,
        checkpoint_path=str(tmp_path / "checkpoints"),
        architecture="topk",
        activation_fn_kwargs={"k": 2},
        disable_enc_bias=True,
    )

    sae_cfg = BaseSAEConfig.from_sae_runner_config(runner_cfg)
    sae = BatchTopkSAE(sae_cfg)

    sae_stats = train_sae(
        sae=sae,  # Will be created inside train_sae based on config
        cfg=runner_cfg,
    )

    assert torch.allclose(sae.b_enc, torch.zeros_like(sae.b_enc))

    with torch.no_grad():
        assert sae_stats.sparsity.shape == (4,)
        assert sae_stats.l0 > 0
        assert sae_stats.width == 4

    assert (tmp_path / "checkpoints" / "final_1000").exists()
    assert (tmp_path / "checkpoints" / "final_1000" / "cfg.json").exists()
    assert (
        tmp_path / "checkpoints" / "final_1000" / "sae_weights.safetensors"
    ).exists()
    assert (tmp_path / "checkpoints" / "final_1000" / "sparsity.safetensors").exists()


def test_hash_sae_cfg() -> None:
    """Test that hashing SAE configs produces consistent results."""
    # Create two identical configs
    cfg1 = BaseSAERunnerConfig(
        model_name="gpt2",
        hook_name="blocks.0.hook_resid_post",
        d_sae=4,
        run_name="test_run_1",
        checkpoint_path="/path/to/checkpoints/1",
    )

    cfg2 = BaseSAERunnerConfig(
        model_name="gpt2",
        hook_name="blocks.0.hook_resid_post",
        d_sae=4,
        run_name="test_run_2",  # Different run name
        checkpoint_path="/path/to/checkpoints/2",  # Different checkpoint path
    )

    # Create a different config
    cfg3 = BaseSAERunnerConfig(
        model_name="gpt2",
        hook_name="blocks.0.hook_resid_post",
        d_sae=8,  # Different width
        run_name="test_run_3",
        checkpoint_path="/path/to/checkpoints/3",
    )

    # Test that identical configs (except for run_name and checkpoint_path) hash to the same value
    assert hash_sae_cfg(cfg1) == hash_sae_cfg(cfg2)

    # Test that different configs hash to different values
    assert hash_sae_cfg(cfg1) != hash_sae_cfg(cfg3)


def test_find_latest_checkpoint(tmp_path: Path) -> None:
    """Test that find_latest_checkpoint returns the most recent checkpoint."""
    checkpoints_path = tmp_path / "checkpoints"
    checkpoints_path.mkdir(parents=True, exist_ok=True)

    # Create some checkpoint files with different timestamps
    checkpoint_files = ["100", "200", "300"]

    # No checkpoints
    assert find_latest_checkpoint(checkpoints_path) is None

    # Create checkpoints with increasing timestamps
    for i, checkpoint in enumerate(checkpoint_files):
        checkpoint_file = checkpoints_path / checkpoint
        checkpoint_file.mkdir(parents=True, exist_ok=True)
        # Sleep to ensure different timestamps
        time.sleep(0.01)

    # The latest checkpoint should be the last one created
    assert find_latest_checkpoint(checkpoints_path) == str(
        checkpoints_path / checkpoint_files[-1]
    )


def test_load_pretrained_weights(tmp_path: Path) -> None:
    """Test that _load_pretrained_weights correctly loads weights from a saved BatchTopkSAE."""
    # Create a BatchTopkSAE
    cfg = build_sae_cfg(
        d_in=8,
        d_sae=10,
        architecture="topk",
        activation_fn_kwargs={"k": 2},
    )

    original_sae = BatchTopkSAE(cfg)

    # Set threshold to an arbitrary value
    arbitrary_threshold = 0.75
    original_sae.topk_threshold = torch.tensor(
        arbitrary_threshold, dtype=original_sae.topk_threshold.dtype
    )

    # Create test input and get feature activations
    x = torch.randn(4, 8)
    original_feature_acts = original_sae.encode(x)
    original_output = original_sae.decode(original_feature_acts)

    # Save the model as final (which converts to jumprelu)
    original_sae.save_model_final(tmp_path)

    # Create a new SAE with the same config
    new_sae = BatchTopkSAE(cfg)

    # Verify weights are different before loading
    assert not torch.allclose(new_sae.W_enc, original_sae.W_enc)
    assert not torch.allclose(new_sae.W_dec, original_sae.W_dec)

    # Load pretrained weights
    _load_pretrained_weights(
        sae=new_sae,
        pretrained_path=str(tmp_path),
        device="cpu",
    )

    # Verify weights are identical after loading
    torch.testing.assert_close(new_sae.W_enc, original_sae.W_enc)
    torch.testing.assert_close(new_sae.b_enc, original_sae.b_enc)
    torch.testing.assert_close(new_sae.W_dec, original_sae.W_dec)
    torch.testing.assert_close(new_sae.b_dec, original_sae.b_dec)

    # Verify behavior is identical
    new_feature_acts = new_sae.encode(x)
    new_output = new_sae.decode(new_feature_acts)

    torch.testing.assert_close(new_feature_acts, original_feature_acts)
    torch.testing.assert_close(new_output, original_output)

    # Verify the SAE was converted to jumprelu format
    # The original SAE should still be a topk SAE
    assert original_sae.cfg.architecture == "topk"

    # Check that threshold was properly loaded
    # Note: When loading from a saved model, the threshold is loaded as a tensor
    # with shape (d_sae,) where each element equals the original threshold
    assert "topk_threshold" in new_sae.state_dict()
    assert new_sae.topk_threshold == original_sae.topk_threshold
