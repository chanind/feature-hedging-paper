import math
from pathlib import Path
from typing import Callable

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import torch
from matplotlib.axes import Axes
from plotly.subplots import make_subplots
from rich.jupyter import print as rprint
from rich.table import Table
from rich.text import Text
from sae_lens import TrainingSAE
from tueplots import axes, bundles

from hedging_paper.toy_models.toy_model import ToyModel
from hedging_paper.util import DEFAULT_DEVICE, cos_sims


def plot_latent_firing_histograms(
    sae: TrainingSAE,
    toy_model: ToyModel,
    activations_batch_generator: Callable[[int], torch.Tensor],
    num_sample_acts: int = 100_000,
    firing_threshold: float = 0.5,
):
    latent_acts = sae.encode(toy_model(activations_batch_generator(100_000)))
    B, D = latent_acts.shape

    # Calculate grid dimensions
    n_cols = min(5, D)
    n_rows = math.ceil(D / n_cols)

    # Create subplot grid
    fig = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=[f"Latent {i}" for i in range(D)]
    )

    # Create histograms
    for i in range(D):
        row = i // n_cols + 1
        col = i % n_cols + 1

        values = latent_acts[:, i].detach().cpu().float().numpy()
        values = values[values > firing_threshold]
        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=25,
                xbins=dict(start=0, end=values.max(), size=(values.max()) / 30),
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(
            range=[0, values.max()], row=row, col=col, title_text="Firing magnitude"
        )
        if col == 1:
            fig.update_yaxes(row=row, col=col, title_text="Count")

    suffix = f"({num_sample_acts} sample activations)" if n_cols > 1 else ""
    fig.update_layout(
        height=300 * n_rows,
        width=300 * n_cols,
        showlegend=False,
        title_text=f"SAE Latent firing distribution {suffix}",
    )
    fig.show()


def plot_latent_cos_sims(sae: TrainingSAE):
    latent_cos_sims = cos_sims(sae.W_dec.T, sae.W_dec.T)
    px.imshow(
        latent_cos_sims.detach().cpu().numpy(),
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="SAE latent cosine similarities",
        height=400,
        width=400,
    ).show()


def plot_sae_feat_cos_sims(
    sae: TrainingSAE,
    model: ToyModel,
    title_suffix: str,
    height: int = 400,
    width: int = 800,
    show_values: bool = False,  # New parameter to control showing values
):
    dec_cos_sims = (
        torch.round(cos_sims(sae.W_dec.T, model.embed.weight) * 100) / 100 + 0.0
    )
    enc_cos_sims = (
        torch.round(cos_sims(sae.W_enc, model.embed.weight) * 100) / 100 + 0.0
    )

    fig = make_subplots(rows=1, cols=2, subplot_titles=("SAE encoder", "SAE decoder"))
    hovertemplate = "True feature: %{x}<br>SAE Latent: %{y}<br>Cosine Similarity: %{z:.3f}<extra></extra>"

    # Create encoder heatmap trace with conditional text properties
    encoder_args = {
        "z": enc_cos_sims.detach().cpu().numpy(),
        "zmin": -1,
        "zmax": 1,
        "colorscale": "RdBu",
        "showscale": False,
        "hovertemplate": hovertemplate,
    }

    # Only add text-related properties if show_values is True
    if show_values:
        encoder_args["texttemplate"] = "%{z:.2f}"
        encoder_args["textfont"] = {"size": 10}

    fig.add_trace(go.Heatmap(**encoder_args), row=1, col=1)

    # Create decoder heatmap trace with conditional text properties
    decoder_args = {
        "z": dec_cos_sims.detach().cpu().numpy(),
        "zmin": -1,
        "zmax": 1,
        "colorscale": "RdBu",
        "colorbar": dict(title="cos sim", x=1.0, dtick=1, tickvals=[-1, 0, 1]),
        "hovertemplate": hovertemplate,
    }

    # Only add text-related properties if show_values is True
    if show_values:
        decoder_args["texttemplate"] = "%{z:.2f}"
        decoder_args["textfont"] = {"size": 10}

    fig.add_trace(go.Heatmap(**decoder_args), row=1, col=2)

    fig.update_layout(
        height=height,
        width=width,
        title_text=f"Cosine Similarity with True Features ({title_suffix})",
    )
    fig.update_xaxes(title_text="True feature", row=1, col=1, dtick=1)
    fig.update_xaxes(title_text="True feature", row=1, col=2, dtick=1)
    fig.update_yaxes(title_text="SAE Latent", row=1, col=1, dtick=1)
    fig.update_yaxes(title_text="SAE Latent", row=1, col=2, dtick=1)

    fig.show()


def plot_b_dec_feat_cos_sims(
    sae: TrainingSAE,
    model: ToyModel,
    title_suffix: str,
    height: int = 300,
    width: int = 800,
    show_values: bool = False,
) -> None:
    b_dec_cos_sims = cos_sims(sae.b_dec.unsqueeze(-1), model.embed.weight)
    fig = make_subplots(rows=1, cols=1)
    hovertemplate = "True feature: %{x}<br>Cosine Similarity: %{z:.3f}<extra></extra>"

    # Only set texttemplate when show_values is True
    texttemplate = "%{z:.2f}" if show_values else None

    # Create the heatmap trace with conditional text properties
    heatmap_args = {
        "z": b_dec_cos_sims.detach().cpu().numpy(),
        "zmin": -1,
        "zmax": 1,
        "colorscale": "RdBu",
        "colorbar": dict(title="cos sim", x=1.0, dtick=1, tickvals=[-1, 0, 1]),
        "hovertemplate": hovertemplate,
    }

    # Only add text-related properties if show_values is True
    if show_values:
        heatmap_args["texttemplate"] = texttemplate
        heatmap_args["textfont"] = {"size": 10}

    # Add decoder plot
    fig.add_trace(go.Heatmap(**heatmap_args), row=1, col=1)

    fig.update_layout(
        height=height,
        width=width,
        title_text=f"SAE b_dec cos sim with true features ({title_suffix})",
    )
    fig.update_xaxes(title_text="True feature", row=1, col=1, dtick=1)

    # Keep the y-axis label but hide the tick marks and tick labels
    fig.update_yaxes(
        title_text="SAE decoder bias",
        row=1,
        col=1,
        showticklabels=False,  # Hide tick labels (the numbers)
        showgrid=False,  # Hide grid lines
        zeroline=False,  # Hide the zero line
    )

    fig.show()


def print_sample_feats_and_acts(
    feats: torch.Tensor,
    sae: TrainingSAE,
    model: ToyModel,
    device: torch.device = DEFAULT_DEVICE,
):
    feat_mags = feats.float().to(device)
    latent_acts = sae.encode(model(feats.float().to(device)))

    table = Table(title="Sample feature values and corresponding SAE activations")

    # Add columns
    table.add_column("True features", justify="center")
    table.add_column("SAE Latent acts", justify="center")

    def style_row(row):
        text = Text()
        for val in row:
            style = "bold" if val > 1e-4 else "dim"
            text.append(f"{val:.2f}", style=style)
            text.append("  ")
        return text

    # Add rows
    for row1, row2 in zip(feat_mags, latent_acts):
        table.add_row(
            style_row(row1),
            style_row(row2),
        )
    rprint(table)


SEABORN_RC_CONTEXT = {
    **bundles.neurips2021(),
    **axes.lines(),
    # Use constrained_layout for automatic adjustments
    "figure.constrained_layout.use": True,
}


def _find_best_index_reordering(
    cos_sims: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """Find the best index reordering of the cos_sims tensor.

    Args:
        cos_sims: A tensor of cosine similarities between two sets of vectors.
                 Shape: (n_sae_latents, n_true_features)

    Returns:
        the score of the reordering, and the reordered tensor
    """
    best_feature_matches = torch.argmax(torch.abs(cos_sims), dim=1)
    # Sort SAE latents by their best matching true feature
    sorted_indices = torch.argsort(best_feature_matches)

    # Calculate score using the diagonal elements after reordering
    # We need to handle the case where cos_sims is not square
    reordered_cos_sims = cos_sims[sorted_indices]
    n_latents = reordered_cos_sims.shape[0]
    diagonal_indices = torch.arange(min(n_latents, reordered_cos_sims.shape[1]))
    score = reordered_cos_sims[diagonal_indices, diagonal_indices].mean().item()
    return score, sorted_indices


def _add_row_rectangles(
    ax: Axes,
    highlight_rows: list[int] | list[list[int]],
    n_latents: int,
    n_features: int,
    one_based_indexing: bool,
) -> None:
    """Add rectangles around specified rows in a heatmap.

    Args:
        ax: The matplotlib axes to draw on
        highlight_rows: Rows to highlight - can be individual rows or ranges
        n_latents: Number of latents (rows) in the heatmap
        n_features: Number of features (columns) in the heatmap
        one_based_indexing: Whether input uses 1-based indexing
    """

    # Convert highlight_rows to a consistent format: list of ranges
    ranges_to_highlight: list[list[int]] = []

    if not highlight_rows:
        return

    # Handle different input formats
    for item in highlight_rows:
        if isinstance(item, int):
            # Single row
            ranges_to_highlight.append([item])
        elif isinstance(item, list):
            # Range of rows
            ranges_to_highlight.append(item)
        else:
            raise ValueError(f"Invalid highlight_rows item: {item}")

    # Draw rectangles for each range
    for row_range in ranges_to_highlight:
        if not row_range:
            continue

        # Convert to 0-based indexing for internal calculations
        if one_based_indexing:
            zero_based_rows = [r - 1 for r in row_range]
        else:
            zero_based_rows = row_range.copy()

        # Validate row indices
        for row_idx in zero_based_rows:
            if row_idx < 0 or row_idx >= n_latents:
                raise ValueError(
                    f"Row index {row_idx} out of range [0, {n_latents - 1}]"
                )

        # Find the min and max rows to create a rectangle
        min_row = min(zero_based_rows)
        max_row = max(zero_based_rows)

        # Calculate rectangle coordinates
        # The heatmap cells are at positions [0, 1, 2, ..., n_latents-1]
        # Each cell spans from i to i+1 in the coordinate system
        # The y-axis will be inverted later, but we work in the original coordinate system
        y_bottom = min_row  # Bottom of rectangle (before inversion)
        y_height = max_row - min_row + 1  # Height of rectangle

        # Rectangle spans all columns and extends slightly outside on left and right
        # Use a small fraction of a cell width for the extension (~1-2 pixels visually)
        extension = 0.05  # Small extension in data coordinates
        x_left = -extension  # Extend slightly to the left
        x_width = n_features + (2 * extension)  # Extend slightly on each side

        # Create and add rectangle
        rect = patches.Rectangle(
            (x_left, y_bottom),
            x_width,
            y_height,
            linewidth=1,
            edgecolor="#333",
            facecolor="none",
        )
        ax.add_patch(rect)


def plot_sae_feat_cos_sims_seaborn(
    sae: TrainingSAE,
    model: ToyModel,
    title_suffix: str | None = None,
    title: str | None = None,
    height: float = 8,
    width: float = 16,
    show_values: bool = False,
    save_path: str | Path | None = None,
    one_based_indexing: bool = False,
    highlight_rows: list[int] | list[list[int]] | None = None,
    reorder_latents: bool | torch.Tensor = False,
) -> None:
    """Plot cosine similarities between SAE features and true features using seaborn.

    Args:
        sae: The trained SAE
        model: The toy model being analyzed
        title_suffix: Suffix to add to the plot title
        height: Figure height in inches
        width: Figure width in inches
        show_values: Whether to show the cosine similarity values on the heatmap
        save_path: Optional path to save the figure
        one_based_indexing: Whether to use 1-based indexing for axis labels
        highlight_rows: Rows to highlight with rectangles. Can be a single row (int),
            list of individual rows, or list of row ranges (each range as a list of ints).
            Uses the same indexing as one_based_indexing parameter.
    """
    dec_cos_sims = (
        torch.round(cos_sims(sae.W_dec.T, model.embed.weight) * 100) / 100 + 0.0
    )
    enc_cos_sims = (
        torch.round(cos_sims(sae.W_enc, model.embed.weight) * 100) / 100 + 0.0
    )

    # Apply feature reordering if requested
    if reorder_latents is not False:
        if isinstance(reorder_latents, bool):
            _, sorted_indices = _find_best_index_reordering(dec_cos_sims)
            dec_cos_sims = dec_cos_sims[sorted_indices]
            enc_cos_sims = enc_cos_sims[sorted_indices]
        else:
            dec_cos_sims = dec_cos_sims[reorder_latents]
            enc_cos_sims = enc_cos_sims[reorder_latents]

    # NOTE: We plot the original matrices, not flipped ones.
    # We will invert the y-axis later for correct visual orientation.

    plt.rcParams.update({"figure.dpi": 150})
    with plt.rc_context(SEABORN_RC_CONTEXT):
        # Create figure and subplots (no longer sharing y-axis)
        fig, axes = plt.subplots(1, 2, figsize=(width, height))
        ax1, ax2 = axes

        # Get dimensions for tick labels
        n_features = model.embed.weight.shape[1]
        n_latents = sae.W_enc.shape[1]

        # Create tick labels based on indexing preference
        raw_feature_ticks = (
            list(range(1, n_features + 1))
            if one_based_indexing
            else list(range(n_features))
        )
        raw_latent_ticks = (
            list(range(1, n_latents + 1))
            if one_based_indexing
            else list(range(n_latents))
        )

        # Convert to strings for matplotlib
        feature_ticks = [str(i) for i in raw_feature_ticks]
        latent_ticks = [str(i) for i in raw_latent_ticks]

        # Plot encoder heatmap with original data
        sns.heatmap(
            enc_cos_sims.detach().cpu().numpy(),  # Use original data
            ax=ax1,
            vmin=-1,
            vmax=1,
            cmap="RdBu",
            center=0,
            annot=show_values,
            fmt=".2f" if show_values else "",
            cbar=False,  # Colorbar handled separately
        )
        ax1.set_title("SAE encoder")
        ax1.set_xlabel("True feature")
        ax1.set_ylabel("SAE Latent")
        ax1.set_xticks([i + 0.5 for i in range(n_features)], feature_ticks)
        ax1.set_yticks(
            [i + 0.5 for i in range(n_latents)], latent_ticks
        )  # Use original labels

        # Plot decoder heatmap with original data
        sns.heatmap(
            dec_cos_sims.detach().cpu().numpy(),  # Use original data
            ax=ax2,
            vmin=-1,
            vmax=1,
            cmap="RdBu",
            center=0,
            annot=show_values,
            fmt=".2f" if show_values else "",
            cbar=True,  # Add colorbar here
            cbar_kws={
                "label": "cos sim",
                "ticks": [-1, 0, 1],
                "shrink": 0.75,
            },  # Adjust shrink as needed
        )
        ax2.set_title("SAE decoder")
        ax2.set_xlabel("True feature")
        ax2.set_ylabel("SAE Latent")  # Restore y-label as axes not shared
        ax2.set_xticks([i + 0.5 for i in range(n_features)], feature_ticks)
        ax2.set_yticks(
            [i + 0.5 for i in range(n_latents)], latent_ticks
        )  # Set y-ticks explicitly

        # Invert y-axis on both plots
        ax1.invert_yaxis()
        ax2.invert_yaxis()

        # Add rectangles to highlight rows if specified
        if highlight_rows is not None:
            _add_row_rectangles(
                ax1, highlight_rows, n_latents, n_features, one_based_indexing
            )
            _add_row_rectangles(
                ax2, highlight_rows, n_latents, n_features, one_based_indexing
            )
            # Extend x-axis limits to show the full rectangle (small extension on each side)
            extension = 0.05  # Match the extension used in rectangle drawing
            ax1.set_xlim(-extension, n_features + extension)
            ax2.set_xlim(-extension, n_features + extension)

        # Set the main title
        if title is None:
            title = (
                f"Cosine similarity with true features ({title_suffix})"
                if title_suffix
                else "Cosine similarity with true features"
            )
        fig.suptitle(title)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                save_path, bbox_inches="tight"
            )  # Use bbox_inches='tight' for saving
        plt.show()


def plot_b_dec_feat_cos_sims_seaborn(
    sae: TrainingSAE,
    model: ToyModel,
    title_suffix: str | None = None,
    height: float = 4,
    width: float = 12,
    show_values: bool = False,
    save_path: str | Path | None = None,
    one_based_indexing: bool = False,
) -> None:
    """Plot cosine similarities between SAE decoder bias and true features using seaborn.

    Args:
        sae: The trained SAE
        model: The toy model being analyzed
        title_suffix: Suffix to add to the plot title
        height: Figure height in inches
        width: Figure width in inches
        show_values: Whether to show the cosine similarity values on the heatmap
        save_path: Optional path to save the figure
        one_based_indexing: Whether to use 1-based indexing for axis labels
    """
    b_dec_cos_sims = cos_sims(sae.b_dec.unsqueeze(-1), model.embed.weight)

    plt.rcParams.update({"figure.dpi": 150})
    # Use the shared context with constrained_layout=True
    with plt.rc_context(SEABORN_RC_CONTEXT):
        # Create figure with single subplot
        fig, ax = plt.subplots(1, 1, figsize=(width, height))

        # Get dimensions for tick labels
        n_features = model.embed.weight.shape[1]

        # Create tick labels based on indexing preference
        raw_ticks = (
            list(range(1, n_features + 1))
            if one_based_indexing
            else list(range(n_features))
        )
        feature_ticks = [str(i) for i in raw_ticks]

        # Plot heatmap
        sns.heatmap(
            b_dec_cos_sims.detach().cpu().numpy(),
            ax=ax,
            vmin=-1,
            vmax=1,
            cmap="RdBu",
            center=0,
            annot=show_values,
            fmt=".2f" if show_values else "",
            cbar_kws={"label": "cos sim", "ticks": [-1, 0, 1]},
        )
        # ax.set_title("SAE decoder bias") # Subplot title usually not needed with suptitle
        ax.set_xlabel("True feature")
        ax.set_ylabel("SAE decoder bias")
        ax.set_xticks([i + 0.5 for i in range(n_features)], feature_ticks)

        # Hide y-axis tick labels since there's only one row
        ax.set_yticks([])

        # Set the main title (constrained_layout handles positioning)
        title = (
            f"SAE b_dec cos sim with true features ({title_suffix})"
            if title_suffix
            else "SAE b_dec cos sim with true features"
        )
        fig.suptitle(title)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            # Add bbox_inches='tight' to savefig
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()


def plot_2d_feature_space(
    sae: TrainingSAE,
    toy_model: ToyModel,
    title: str,
    angle_degrees: float = 0.0,
    height: float = 2.5,
    width: float = 2.5,
    save_path: str | Path | None = None,
    show_decoder_bias: bool = False,
    loc: str = "lower left",
    latex_legend: bool = False,
) -> None:
    """Plot toy model features and SAE latents on a 2D plane.

    Args:
        sae: The trained SAE
        toy_model: The toy model with 2 features
        title: Title for the plot
        angle_degrees: Angle in degrees that the first feature should be shown at relative to the y-axis
        height: Figure height in inches
        width: Figure width in inches
        save_path: Optional path to save the figure
        show_decoder_bias: Whether to show the SAE decoder bias as a point
    """
    sae.fold_W_dec_norm()
    if toy_model.embed.weight.shape[1] != 2:
        raise ValueError(
            f"This function requires exactly 2 true features, got {toy_model.embed.weight.shape[1]}"
        )

    # Get the true features from the toy model (shape: d_in x n_features)
    true_features_full = toy_model.embed.weight.detach().cpu().numpy()  # Shape: (50, 2)

    # Create 2D basis: first axis is Feature 0, second axis is orthogonal in the Feature 0-1 plane
    feature_0 = true_features_full[:, 0]  # First feature vector (50,)
    feature_1 = true_features_full[:, 1]  # Second feature vector (50,)

    # Normalize the first feature to create first basis vector
    basis_1 = feature_0 / np.linalg.norm(feature_0)

    # Create second basis vector orthogonal to first, in the plane spanned by both features
    # Use Gram-Schmidt orthogonalization
    feature_1_proj_on_basis_1 = np.dot(feature_1, basis_1) * basis_1
    basis_2_unnorm = feature_1 - feature_1_proj_on_basis_1
    basis_2 = basis_2_unnorm / np.linalg.norm(basis_2_unnorm)

    # Apply rotation to the basis vectors if angle is specified
    angle_rad = np.radians(angle_degrees)
    rotation_matrix_2d = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )

    # Rotate the basis vectors in 2D space, then extend back to full dimensionality
    # Swap order so Feature 0 is y-axis (second component) and orthogonal direction is x-axis
    basis_2d = np.column_stack(
        [basis_2, basis_1]
    )  # Shape: (50, 2) - [orthogonal, feature_0]
    rotated_basis_2d = basis_2d @ rotation_matrix_2d  # Rotate the basis vectors

    # Project true features onto the rotated 2D basis
    rotated_features = (true_features_full.T @ rotated_basis_2d).T  # Shape: (2, 2)

    # Project SAE decoder directions onto the rotated 2D basis
    sae_decoder_full = sae.W_dec.detach().cpu().numpy()  # Shape: (n_latents, 50)
    rotated_sae = (sae_decoder_full @ rotated_basis_2d).T  # Shape: (2, n_latents)

    # Project SAE decoder bias onto the rotated 2D basis if requested
    if show_decoder_bias:
        sae_bias_full = sae.b_dec.detach().cpu().numpy()  # Shape: (50,)
        rotated_bias = sae_bias_full @ rotated_basis_2d  # Shape: (2,)

    latent_color = "black"
    latent_alpha = 0.9

    bias_color = "grey"
    bias_alpha = 0.8

    plt.rcParams.update({"figure.dpi": 150})
    with plt.rc_context(SEABORN_RC_CONTEXT):
        fig, ax = plt.subplots(figsize=(width, height))

        color_map = plt.cm.get_cmap("viridis")
        # Plot true features as solid lines with dots
        colors = [color_map(0.95), color_map(0.75)]

        if latex_legend:
            feature_names = ["$f_1$", "$f_2$"]
        else:
            feature_names = ["Feature 1", "Feature 2"]

        for i in range(2):
            # Draw line from origin to feature endpoint
            ax.plot(
                [0, rotated_features[0, i]],
                [0, rotated_features[1, i]],
                color=colors[i],
                linewidth=2,
                label=feature_names[i],
            )
            # Add dot at the end
            ax.plot(
                rotated_features[0, i],
                rotated_features[1, i],
                "o",
                color=colors[i],
                markersize=8,
            )

        # Plot SAE latents as dotted lines with dots
        n_latents = sae.W_dec.shape[0]  # W_dec has shape (n_latents, d_in)
        multi_latent = n_latents > 1
        for i in range(n_latents):
            # Draw dotted line from origin to latent endpoint
            ax.plot(
                [0, rotated_sae[0, i]],
                [0, rotated_sae[1, i]],
                color=latent_color,
                linewidth=1,
                linestyle="--",
                alpha=latent_alpha,
                label=f"SAE Latent {i}" if multi_latent else "SAE Latent",
            )
            # Add dot at the end
            ax.plot(
                rotated_sae[0, i],
                rotated_sae[1, i],
                "o",
                color=latent_color,
                markersize=3,
                alpha=latent_alpha,
            )

        # Plot SAE decoder bias as dotted line if requested
        if show_decoder_bias:
            # Draw dotted line from origin to bias endpoint
            ax.plot(
                [0, rotated_bias[0]],  # type: ignore
                [0, rotated_bias[1]],  # type: ignore
                color=bias_color,
                linewidth=1,
                linestyle="--",
                alpha=bias_alpha,
                label="SAE $b_{dec}$" if latex_legend else "SAE Decoder Bias",
            )
            # Add dot at the end
            ax.plot(
                rotated_bias[0],  # type: ignore
                rotated_bias[1],  # type: ignore
                "o",
                color=bias_color,
                markersize=3,
                alpha=bias_alpha,
            )

        # Set equal aspect ratio and add grid
        ax.set_aspect("equal")

        # Set axis limits with some padding
        max_val = (
            max(np.max(np.abs(rotated_features)), np.max(np.abs(rotated_sae))) * 1.2
        )
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)

        # Set custom ticks to avoid 0.5, -0.5
        max_tick = int(np.floor(max_val))
        ticks = list(range(-max_tick, max_tick + 1))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)

        # Remove axis labels
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Add legend
        ax.legend(loc=loc)

        fig.suptitle(title)

        # Save if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")

        plt.show()
        plt.close(fig)  # Close the figure to prevent multiple plots


def plot_correlation_vs_cosine_similarity(
    data,
    title: str = "SAE Cosine Similarity vs Feature Correlation",
    height: float = 3,
    width: float = 4,
    save_path: str | Path | None = None,
) -> None:
    """Plot cosine similarity between SAE latents and true features vs correlation.

    Args:
        data: DataFrame with columns 'corr', 'f1_cos_sim', 'f2_cos_sim'
        title: Title for the plot
        height: Figure height in inches
        width: Figure width in inches
        save_path: Optional path to save the figure
    """
    plt.rcParams.update({"figure.dpi": 150})
    with plt.rc_context(SEABORN_RC_CONTEXT):
        fig, ax = plt.subplots(figsize=(width, height))

        # Add x and y axes
        ax.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)

        # Set up labels
        x_label = "Feature Correlation"

        # Plot Feature 2 cosine similarity only
        ax.plot(
            data["corr"],
            data["f2_cos_sim"],
            linewidth=1,
        )
        # Set labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel("Cosine Similarity")
        fig.suptitle(title)

        # No legend needed for single line

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

        # Set y-axis limits to show full range for Feature 2 only
        y_min = data["f2_cos_sim"].min()
        y_max = data["f2_cos_sim"].max()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        # Save if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")

        plt.show()
        plt.close(fig)
