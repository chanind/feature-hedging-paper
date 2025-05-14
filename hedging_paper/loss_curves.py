import re  # Import regular expression module
from pathlib import Path
from typing import Callable

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tueplots import axes, bundles

from hedging_paper.util import listify

# Define the seaborn context similar to plotting.py
# Ideally, this would be shared in a common utility module
SEABORN_RC_CONTEXT = {
    **bundles.neurips2021(),
    **axes.lines(),
    "figure.constrained_layout.use": True,
    # Ensure LaTeX is used if not already set by bundles
    "text.usetex": True,
}

# Add amsmath to the LaTeX preamble for \text command
# Ensure preamble updates happen *before* plotting
plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}"})

MSE_LOSS = lambda x, y: ((x - y) ** 2).sum()


@torch.no_grad()
def calc_loss_curve(
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    parent_only_prob: float,
    parent_and_child_prob: float,
    steps: int = 100,
    sparsity_coefficient: float = 0.0,
    sparsity_p: float = 1.0,
    rand_vecs: bool = False,
) -> list[float]:
    if rand_vecs:
        parent_vec = torch.randn(10).float()
        child_vec = torch.randn(10).float()
        child_vec = (
            child_vec - (child_vec @ parent_vec) * parent_vec / parent_vec.norm() ** 2
        )
        child_vec = child_vec / child_vec.norm()
        parent_vec = parent_vec / parent_vec.norm()
        parent_and_child_vec = parent_vec + child_vec
    else:
        parent_vec = torch.tensor([1, 0]).float()
        child_vec = torch.tensor([0, 1]).float()
        parent_and_child_vec = parent_vec + child_vec

    losses = []
    for i in range(steps):
        portion = i / (steps - 1)
        test_latent = parent_vec * (1 - portion) + child_vec * portion
        test_latent = test_latent / test_latent.norm()

        encode = lambda input_act: (input_act @ test_latent).relu()
        decode = lambda hidden_act: hidden_act * test_latent

        def calc_loss(input_act: torch.Tensor) -> torch.Tensor:
            hidden_act = encode(input_act)
            recons = decode(hidden_act)
            return loss(recons, input_act) + sparsity_coefficient * hidden_act.norm(
                p=sparsity_p
            )

        parent_loss = calc_loss(parent_vec)
        parent_and_child_loss = calc_loss(parent_and_child_vec)
        expected_loss = (
            parent_loss * parent_only_prob
            + parent_and_child_loss * parent_and_child_prob
        ).item()
        losses.append(expected_loss)
    return losses


def escape_latex_special_chars(text: str) -> str:
    """Escapes special LaTeX characters in a string."""
    # Order matters, especially for backslash
    chars = {
        "\\": r"\\textbackslash{}",  # Must be first
        "&": r"\&",
        "%": r"\%",
        # "$": r"\$",
        "#": r"\#",
        # "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "<": r"\textless{}",  # Use text commands for < and >
        ">": r"\textgreater{}",
    }
    regex = re.compile("|".join(re.escape(key) for key in chars.keys()))
    return regex.sub(lambda match: chars[match.group(0)], text)


def plot_loss_curve_seaborn(
    parent_only_prob: float = 0.15,
    parent_and_child_prob: float = 0.15,
    sparsity_coefficient: float | list[float] = 0.0,
    sparsity_p: float = 1.0,
    subtitle: str | None = None,
    rand_vecs: bool = False,
    steps: int = 10_000,
    width: float = 8,
    height: float = 6,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot loss curves for multiple sparsity coefficients using Seaborn.

    Args:
        parent_only_prob: Probability of parent-only case.
        parent_and_child_prob: Probability of parent-and-child case.
        sparsity_coefficient: Single value or list of coefficients for sparsity loss.
        sparsity_p: The p-norm used for sparsity loss.
        subtitle: Optional subtitle for the plot.
        rand_vecs: Whether to use random vectors instead of standard basis.
        steps: Number of interpolation steps.
        width: Figure width in inches.
        height: Figure height in inches.
        save_path: Optional path to save the figure.
    """
    sparsity_coefficients = listify(sparsity_coefficient)

    all_data = []
    for coeff in sparsity_coefficients:
        loss = calc_loss_curve(
            loss=MSE_LOSS,
            parent_only_prob=parent_only_prob,
            parent_and_child_prob=parent_and_child_prob,
            sparsity_coefficient=coeff,
            sparsity_p=sparsity_p,
            rand_vecs=rand_vecs,
            steps=steps,
        )

        # Create x-axis values
        x = np.linspace(0, 1, len(loss))

        # Store data for this coefficient
        df_coeff = pd.DataFrame(
            {
                "Interpolation": x,
                "Loss": loss,
                "Sparsity Coefficient": coeff,
            }
        )
        all_data.append(df_coeff)

    # Combine data from all coefficients
    df = pd.concat(all_data, ignore_index=True)

    sns.set_theme()
    # Update rcParams here too, just in case
    plt.rcParams.update({"figure.dpi": 150})
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}"})
    # Now use the rc_context which includes usetex=True
    with plt.rc_context(SEABORN_RC_CONTEXT):
        fig, ax = plt.subplots(figsize=(width, height))

        # Plot with Seaborn, using hue for different sparsity coefficients
        sns.lineplot(
            data=df,
            x="Interpolation",
            y="Loss",
            hue="Sparsity Coefficient",
            ax=ax,
        )

        # Get colors assigned by lineplot, checking handle type
        handles, labels = ax.get_legend_handles_labels()
        color_map = {}
        for handle, label in zip(handles, labels):
            if isinstance(handle, mlines.Line2D):
                color_map[label] = handle.get_color()
            # Add elif for other handle types if needed, or ignore

        # Plot minimum point for each curve, matching colors
        for coeff in sparsity_coefficients:
            subset = df[df["Sparsity Coefficient"] == coeff]
            # Use argmin on values and iloc for position-based indexing
            if not subset.empty:
                min_pos = subset["Loss"].argmin()  # type: ignore
                min_row = subset.iloc[min_pos]
                min_x = min_row["Interpolation"]
                min_y = min_row["Loss"]
                point_color = color_map.get(
                    str(coeff)
                )  # Get color corresponding to the coeff
                ax.scatter(
                    [min_x],
                    [min_y],
                    marker="o",
                    s=20,
                    color=point_color,  # Set the color explicitly
                    zorder=5,  # Ensure markers are on top
                )

        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Expected Loss")

        # Set axis limits
        ax.set_xlim(0, 1)
        # Adjust y-axis limit dynamically or set a reasonable minimum
        min_loss_overall = df["Loss"].min()
        max_loss_overall = df["Loss"].max()
        ax.set_ylim(
            bottom=max(
                0, min_loss_overall - 0.1 * (max_loss_overall - min_loss_overall)
            ),
            top=max_loss_overall + 0.1 * (max_loss_overall - min_loss_overall),
        )

        main_title = "Loss curves for single-latent SAE"
        if subtitle:
            # Escape special characters FIRST, then wrap in \text{...}
            escaped_subtitle = escape_latex_special_chars(subtitle)
            # Using \text requires amsmath package (added to preamble)
            fig.suptitle(main_title)
            ax.set_title(escaped_subtitle, fontsize="medium")
        else:
            fig.suptitle(main_title)

        # Improve legend title if needed - use handles/labels already fetched
        # Filter handles and labels to only include those corresponding to lineplot hues
        unique_coeffs_str = [str(c) for c in df["Sparsity Coefficient"].unique()]
        filtered_handles = []
        filtered_labels = []
        # Use explicit loop for potentially better linter analysis
        for h, label in zip(handles, labels):
            if label in unique_coeffs_str:
                filtered_handles.append(h)
                filtered_labels.append(label)

        # Check if we actually found handles corresponding to the coefficients
        if filtered_handles:
            ax.legend(
                filtered_handles,
                filtered_labels,
                title="L1 Coeff.",
                title_fontsize="small",
            )
        elif handles:  # Fallback if filtering failed, just show original legend
            ax.legend(
                title="L1 Coeff.", title_fontsize="small"
            )  # Show legend title even if empty?
        # If no handles at all, legend won't be created automatically

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")

        plt.show()
