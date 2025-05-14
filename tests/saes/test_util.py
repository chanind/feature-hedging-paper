import torch

from hedging_paper.saes.util import running_update


def test_running_update_uninitialized():
    running_val = torch.tensor(-1.0, dtype=torch.float64)
    new_val = torch.tensor(5.0, dtype=torch.float32)

    result = running_update(
        running_val=running_val,
        new_val=new_val,
        alpha=0.99,
        negative_means_uninitialized=True,
    )

    # Should directly use the new value for initialization
    assert torch.isclose(result, new_val.to(torch.float64))
    assert result.dtype == running_val.dtype


def test_running_update_initialized():
    running_val = torch.tensor(10.0, dtype=torch.float64)
    new_val = torch.tensor(5.0, dtype=torch.float32)
    alpha = 0.95

    result = running_update(
        running_val=running_val,
        new_val=new_val,
        alpha=alpha,
        negative_means_uninitialized=True,
    )

    # Should use exponential moving average
    expected = alpha * running_val + (1 - alpha) * new_val.to(torch.float64)
    assert torch.isclose(result, expected)
    assert result.dtype == running_val.dtype


def test_running_update_default_negative_means_uninitialized():
    running_val = torch.tensor(-1.0, dtype=torch.float64)
    new_val = torch.tensor(5.0, dtype=torch.float32)

    # Without negative_means_uninitialized flag
    result = running_update(
        running_val=running_val,
        new_val=new_val,
        alpha=0.99,
        negative_means_uninitialized=False,
    )

    # Should use exponential moving average even with negative value
    expected = 0.99 * running_val + 0.01 * new_val.to(torch.float64)
    assert torch.isclose(result, expected)
