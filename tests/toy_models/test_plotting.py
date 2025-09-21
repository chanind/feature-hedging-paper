import matplotlib.pyplot as plt
import pytest
import torch

from hedging_paper.toy_models.plotting import (
    _add_row_rectangles,
    _find_best_index_reordering,
)


def test_add_row_rectangles_single_row_zero_based():
    """Test adding rectangle around a single row with 0-based indexing."""
    fig, ax = plt.subplots()

    # Test with single row
    _add_row_rectangles(ax, [0], n_latents=5, n_features=3, one_based_indexing=False)

    # Check that one patch was added
    assert len(ax.patches) == 1

    # Check rectangle properties (extends slightly on each side)
    rect = ax.patches[0]
    extension = 0.05
    assert abs(rect.get_x() - (-extension)) < 1e-10  # type: ignore  # Extended slightly to the left
    assert abs(rect.get_width() - (3 + 2 * extension)) < 1e-10  # type: ignore  # 3 features + small extension
    assert rect.get_height() == 1  # type: ignore

    plt.close(fig)


def test_add_row_rectangles_single_row_one_based():
    """Test adding rectangle around a single row with 1-based indexing."""
    fig, ax = plt.subplots()

    # Test with single row (1-based, so row 1 maps to 0-based row 0)
    _add_row_rectangles(ax, [1], n_latents=5, n_features=3, one_based_indexing=True)

    # Check that one patch was added
    assert len(ax.patches) == 1

    # Check rectangle properties (extends slightly on each side)
    rect = ax.patches[0]
    extension = 0.05
    assert abs(rect.get_x() - (-extension)) < 1e-10  # type: ignore  # Extended slightly to the left
    assert abs(rect.get_width() - (3 + 2 * extension)) < 1e-10  # type: ignore  # 3 features + small extension
    assert rect.get_height() == 1  # type: ignore

    plt.close(fig)


def test_add_row_rectangles_row_range():
    """Test adding rectangle around a range of rows."""
    fig, ax = plt.subplots()

    # Test with row range [1, 2] (0-based)
    _add_row_rectangles(
        ax, [[1, 2]], n_latents=5, n_features=3, one_based_indexing=False
    )

    # Check that one patch was added
    assert len(ax.patches) == 1

    # Check rectangle properties (extends slightly on each side)
    rect = ax.patches[0]
    extension = 0.05
    assert abs(rect.get_x() - (-extension)) < 1e-10  # type: ignore  # Extended slightly to the left
    assert abs(rect.get_width() - (3 + 2 * extension)) < 1e-10  # type: ignore  # 3 features + small extension
    assert rect.get_height() == 2  # type: ignore

    plt.close(fig)


def test_add_row_rectangles_multiple_ranges():
    """Test adding rectangles around multiple ranges."""
    fig, ax = plt.subplots()

    # Test with multiple ranges: single row 0 and range [2, 3]
    _add_row_rectangles(
        ax,
        [0, [2, 3]],  # type: ignore
        n_latents=5,
        n_features=3,
        one_based_indexing=False,
    )

    # Check that two patches were added
    assert len(ax.patches) == 2

    plt.close(fig)


def test_add_row_rectangles_invalid_row_index():
    """Test that invalid row indices raise appropriate errors."""
    fig, ax = plt.subplots()

    # Test with row index out of range
    with pytest.raises(ValueError, match="Row index .* out of range"):
        _add_row_rectangles(
            ax, [5], n_latents=5, n_features=3, one_based_indexing=False
        )

    # Test with negative row index
    with pytest.raises(ValueError, match="Row index .* out of range"):
        _add_row_rectangles(
            ax, [-1], n_latents=5, n_features=3, one_based_indexing=False
        )

    plt.close(fig)


def test_add_row_rectangles_empty_input():
    """Test that empty input doesn't add any rectangles."""
    fig, ax = plt.subplots()

    # Test with None
    _add_row_rectangles(
        ax,
        None,  # type: ignore
        n_latents=5,
        n_features=3,
        one_based_indexing=False,
    )
    assert len(ax.patches) == 0

    # Test with empty list
    _add_row_rectangles(ax, [], n_latents=5, n_features=3, one_based_indexing=False)
    assert len(ax.patches) == 0

    plt.close(fig)


def test_add_row_rectangles_invalid_input_type():
    """Test that invalid input types raise appropriate errors."""
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="Invalid highlight_rows item"):
        _add_row_rectangles(
            ax,
            ["invalid"],  # type: ignore
            n_latents=5,
            n_features=3,
            one_based_indexing=False,
        )

    plt.close(fig)


def test_find_best_index_reordering_square_tensor():
    """Test _find_best_index_reordering with square tensor."""
    # Create a 3x3 tensor where each row has highest similarity with different columns
    cos_sims = torch.tensor(
        [
            [0.1, 0.9, 0.2],  # Best match: column 1
            [0.8, 0.1, 0.1],  # Best match: column 0
            [0.1, 0.2, 0.7],  # Best match: column 2
        ]
    )

    score, sorted_indices = _find_best_index_reordering(cos_sims)

    # Expected order: [1, 0, 2] (sorted by best matching column)
    expected_indices = torch.tensor([1, 0, 2])
    assert torch.equal(sorted_indices, expected_indices)

    # Score should be the mean of diagonal elements after reordering
    reordered = cos_sims[sorted_indices]
    expected_score = torch.diagonal(reordered).mean().item()
    assert abs(score - expected_score) < 1e-6


def test_find_best_index_reordering_non_square_fewer_latents():
    """Test _find_best_index_reordering with fewer SAE latents than true features."""
    # Create a 2x4 tensor (2 SAE latents, 4 true features)
    cos_sims = torch.tensor(
        [
            [0.1, 0.2, 0.9, 0.1],  # Best match: column 2
            [0.8, 0.1, 0.1, 0.2],  # Best match: column 0
        ]
    )

    score, sorted_indices = _find_best_index_reordering(cos_sims)

    # Expected order: [1, 0] (sorted by best matching column: 0, 2)
    expected_indices = torch.tensor([1, 0])
    assert torch.equal(sorted_indices, expected_indices)

    # Score should be the mean of the first 2 diagonal elements after reordering
    reordered = cos_sims[sorted_indices]
    expected_score = torch.tensor([reordered[0, 0], reordered[1, 1]]).mean().item()
    assert abs(score - expected_score) < 1e-6


def test_find_best_index_reordering_non_square_more_latents():
    """Test _find_best_index_reordering with more SAE latents than true features."""
    # Create a 4x2 tensor (4 SAE latents, 2 true features)
    cos_sims = torch.tensor(
        [
            [0.9, 0.1],  # Best match: column 0
            [0.1, 0.8],  # Best match: column 1
            [0.7, 0.2],  # Best match: column 0
            [0.2, 0.6],  # Best match: column 1
        ]
    )

    score, sorted_indices = _find_best_index_reordering(cos_sims)

    # Should sort by best matching column (0, 0, 1, 1)
    # So latents matching column 0 come first, then those matching column 1
    best_matches = torch.argmax(torch.abs(cos_sims), dim=1)
    expected_indices = torch.argsort(best_matches)
    assert torch.equal(sorted_indices, expected_indices)

    # Score should be the mean of the first 2 diagonal elements (min of 4, 2)
    reordered = cos_sims[sorted_indices]
    expected_score = torch.tensor([reordered[0, 0], reordered[1, 1]]).mean().item()
    assert abs(score - expected_score) < 1e-6


def test_find_best_index_reordering_single_latent():
    """Test _find_best_index_reordering with single SAE latent."""
    # Create a 1x3 tensor (1 SAE latent, 3 true features)
    cos_sims = torch.tensor([[0.2, 0.8, 0.1]])  # Best match: column 1

    score, sorted_indices = _find_best_index_reordering(cos_sims)

    # Only one latent, so sorted_indices should be [0]
    expected_indices = torch.tensor([0])
    assert torch.equal(sorted_indices, expected_indices)

    # Score should be the first diagonal element
    assert abs(score - cos_sims[0, 0].item()) < 1e-6
