import torch

from hedging_paper.util import Tween, cos_sims


def test_cos_sims_calculates_cosine_similarity_correctly() -> None:
    # Create two test matrices with known cosine similarities
    mat1 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).T  # 2x3 matrix
    mat2 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).T  # 2x3 matrix

    result = cos_sims(mat1, mat2)

    # Expected cosine similarities:
    # [1.0, 0.0, 0.707]
    # [0.0, 1.0, 0.707]
    # [0.707, 0.707, 1.0]
    expected = torch.tensor([[1.0, 0.0, 0.707], [0.0, 1.0, 0.707], [0.707, 0.707, 1.0]])

    # Check shape
    assert result.shape == (3, 3)

    # Check values match expected within tolerance
    torch.testing.assert_close(result, expected, atol=1e-3, rtol=0)


def test_Tween_returns_start_value_before_start_step() -> None:
    tween = Tween(start=0.0, end=1.0, n_steps=10, start_step=5)
    assert tween(step=0) == 0.0
    assert tween(step=4) == 0.0


def test_Tween_returns_end_value_after_end_step() -> None:
    tween = Tween(start=0.0, end=1.0, n_steps=10, start_step=5)
    assert tween(step=16) == 1.0
    assert tween(step=20) == 1.0


def test_Tween_interpolates_linearly_during_steps() -> None:
    tween = Tween(start=0.0, end=1.0, n_steps=10, start_step=0)

    # Test midpoint
    assert tween(step=5) == 0.5

    # Test quarter points
    assert tween(step=2) == 0.2
    assert tween(step=7) == 0.7


def test_Tween_handles_negative_to_positive_range() -> None:
    tween = Tween(start=-1.0, end=1.0, n_steps=4, start_step=0)
    assert tween(step=0) == -1.0
    assert tween(step=2) == 0.0
    assert tween(step=4) == 1.0


def test_Tween_handles_non_zero_start_step() -> None:
    tween = Tween(start=0.0, end=1.0, n_steps=10, start_step=5)
    assert tween(step=3) == 0.0
    assert tween(step=10) == 0.5  # Halfway through the tween
    assert tween(step=15) == 1.0  # End of tween
