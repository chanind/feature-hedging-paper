import pytest
import torch

from hedging_paper.saes.base_sae import BaseSAE, BaseSAEConfig
from tests.helpers import build_runner_cfg, build_sae_cfg


def test_base_sae_l2_reconstruction_loss():
    # Create configs with different reconstruction_loss values
    mse_config = build_sae_cfg(d_in=4, d_sae=2, reconstruction_loss="MSE")

    l2_config = build_sae_cfg(d_in=4, d_sae=2, reconstruction_loss="L2")

    # Create the SAEs
    mse_sae = BaseSAE(mse_config)
    l2_sae = BaseSAE(l2_config)

    # Create test data
    preds = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32
    )
    target = torch.tensor(
        [[0.0, 1.5, 2.5, 3.5], [1.0, 5.5, 6.5, 7.5]], dtype=torch.float32
    )

    # Get the losses
    mse_loss = mse_sae.mse_loss_fn(preds, target).sum(dim=-1).mean()
    l2_loss_result = l2_sae.mse_loss_fn(preds, target).sum(dim=-1).mean()

    # Expected results
    expected_mse = ((preds - target) ** 2).sum(dim=-1).mean()
    expected_l2 = (preds - target).norm(dim=-1, keepdim=True).sum(dim=-1).mean()

    # Test that MSE works correctly
    assert torch.allclose(mse_loss, expected_mse)

    # Test that L2 works correctly
    assert torch.allclose(l2_loss_result, expected_l2)

    # Verify they're different
    assert not torch.allclose(mse_loss, l2_loss_result)


def test_base_sae_config_pulls_in_lr_from_runner_config():
    runner_cfg = build_runner_cfg(lr=1e-4)
    sae_cfg = BaseSAEConfig.from_sae_runner_config(runner_cfg)
    assert sae_cfg.lr == 1e-4


@pytest.mark.parametrize(
    "architecture",
    ["standard", "topk"],
)
def test_base_sae_extend_sae(architecture: str):
    d_in = 4
    initial_d_sae = 8
    num_latents_to_add = 2
    expected_final_d_sae = initial_d_sae + num_latents_to_add

    cfg = build_sae_cfg(
        d_in=d_in,
        d_sae=initial_d_sae,
        architecture=architecture,
        activation_fn_kwargs={"k": 2},
    )
    sae = BaseSAE(cfg)

    # Store original weights
    original_W_enc = sae.W_enc.clone().detach()
    original_W_dec = sae.W_dec.clone().detach()
    original_b_enc = sae.b_enc.clone().detach()

    sae.extend_sae(num_latents_to_add)

    # Check shapes
    assert sae.W_enc.shape == (d_in, expected_final_d_sae)
    assert sae.W_dec.shape == (expected_final_d_sae, d_in)
    assert sae.b_enc.shape == (expected_final_d_sae,)

    # Check that original weights are preserved in the first part
    assert torch.allclose(sae.W_enc[:, :initial_d_sae], original_W_enc)
    assert torch.allclose(sae.W_dec[:initial_d_sae, :], original_W_dec)
    assert torch.allclose(sae.b_enc[:initial_d_sae], original_b_enc)

    # Check that new weights were added and are not zero (except b_enc)
    assert not torch.allclose(
        sae.W_enc[:, initial_d_sae:], torch.zeros_like(sae.W_enc[:, initial_d_sae:])
    )
    assert not torch.allclose(
        sae.W_dec[initial_d_sae:, :], torch.zeros_like(sae.W_dec[initial_d_sae:, :])
    )
    # New b_enc values should be zero
    assert torch.allclose(
        sae.b_enc[initial_d_sae:], torch.zeros_like(sae.b_enc[initial_d_sae:])
    )

    assert sae.encode(torch.randn(5, d_in)).shape == (5, expected_final_d_sae)
    assert sae.decode(torch.randn(5, expected_final_d_sae)).shape == (5, d_in)

    assert sae.W_enc.requires_grad
    assert sae.W_dec.requires_grad
    assert sae.b_enc.requires_grad

    assert sae.cfg.d_sae == expected_final_d_sae
