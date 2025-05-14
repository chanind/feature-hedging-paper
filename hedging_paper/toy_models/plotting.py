import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import torch
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
    """
    dec_cos_sims = (
        torch.round(cos_sims(sae.W_dec.T, model.embed.weight) * 100) / 100 + 0.0
    )
    enc_cos_sims = (
        torch.round(cos_sims(sae.W_enc, model.embed.weight) * 100) / 100 + 0.0
    )

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
