"""Shared plotting style for every figure in the report.

Colourblind-safe only. Okabe-Ito for categorical series (habitats, context
modes, split labels), viridis for continuous scalars (density, correlation
magnitude, z-scores). Two accent colours, navy and steel blue, are reserved
for two-series contrasts (e.g. Method 1 vs Method 2, corrected vs
uncorrected) where only two colours are needed and Okabe-Ito's first two
would otherwise be used arbitrarily.

Every figure function in figures.py calls `apply_style()` once and
`save_figure()` at the end; nothing here is called directly by report.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # headless: no display in a report-generation script

# --------------------------------------------------------------------- colour

NAVY = "#1C2B4A"
STEEL_BLUE = "#2471A3"

# Okabe & Ito (2008), the standard colourblind-safe categorical palette.
OKABE_ITO = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

# Okabe-Ito provides only 8 colours, but this project routinely needs 10
# (ten habitats, ten CNs). Cycling the 8 silently paints category 9 the same
# black as category 1, which is a real legibility failure in a scatter map,
# not a cosmetic one. These two extras are drawn from Paul Tol's
# colourblind-safe qualitative set and chosen to stay separable from all
# eight above: indigo is far darker and more violet than Okabe-Ito blue,
# olive far darker and duller than Okabe-Ito yellow.
_EXTENDED = ["#332288", "#999933"]          # indigo, olive
CATEGORICAL = OKABE_ITO + _EXTENDED          # 10 distinguishable colours

VIRIDIS = matplotlib.colormaps["viridis"]

DPI = 300


def categorical_colors(n: int) -> list[str]:
    """Colourblind-safe categorical colours: Okabe-Ito, then two Tol extras.

    Beyond 10 categories the sequence necessarily cycles; a warning is
    logged rather than silently returning duplicate colours, because two
    categories sharing a colour is a figure that cannot be read.
    """
    if n > len(CATEGORICAL):
        import logging
        logging.getLogger(__name__).warning(
            "%d categories requested but only %d distinguishable colourblind-safe "
            "colours are defined; colours will repeat. Consider faceting instead.",
            n, len(CATEGORICAL))
    return [CATEGORICAL[i % len(CATEGORICAL)] for i in range(n)]


def continuous_cmap():
    return VIRIDIS


# --------------------------------------------------------------------- rcParams


def apply_style() -> None:
    """Set matplotlib rcParams. Idempotent, cheap, call at the top of every
    figure function so figures.py has no import-order dependency on report.py.
    """
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


# --------------------------------------------------------------------- helpers


def letter_panels(axes, x: float = -0.12, y: float = 1.05) -> None:
    """Label each axis A, B, C... in the top-left corner, bold."""
    axes_flat = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes_flat):
        letter = chr(ord("A") + i)
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="bottom", ha="right")


def save_figure(fig, name: str, figures_dir: str | Path) -> dict[str, str]:
    """Save a figure as 300 dpi PNG and vector PDF. Returns the two paths."""
    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    png_path = out / f"{name}.png"
    pdf_path = out / f"{name}.pdf"
    fig.savefig(png_path, dpi=DPI)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def placeholder_figure(
    fig_id: str, title: str, missing_file: str, unblocks: str,
    figsize: tuple[float, float] = (7.0, 4.0),
):
    """A clearly labelled MISSING-DATA panel, per the report's absolute rule:
    never present a simulated result as a finding, and never fabricate a real
    one. Kept numbered in sequence so the report stays complete even when a
    figure cannot be produced.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#FAFAFA")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#999999")
        spine.set_linestyle((0, (4, 3)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.62, f"{fig_id}  {title}", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#333333",
            transform=ax.transAxes, wrap=True)
    ax.text(0.5, 0.44, "DATA NOT PROVIDED", ha="center", va="center",
            fontsize=11, fontweight="bold", color=STEEL_BLUE,
            transform=ax.transAxes)
    ax.text(0.5, 0.30, f"needs: {missing_file}", ha="center", va="center",
            fontsize=9, color="#555555", transform=ax.transAxes, wrap=True)
    ax.text(0.5, 0.18, f"unblocks: {unblocks}", ha="center", va="center",
            fontsize=9, color="#555555", transform=ax.transAxes, wrap=True)
    return fig, ax


def source_caption(ax_or_fig, text: str, y: float = -0.16) -> None:
    """Every panel/figure must state REAL DATA (cohort, n) or SIMULATED
    (null fixture). This stamps that sentence under the figure, in
    figure-fraction coordinates (the default for `Figure.text`).
    """
    fig = ax_or_fig.figure if hasattr(ax_or_fig, "figure") else ax_or_fig
    fig.text(0.01, y, text, fontsize=8, color="#555555", style="italic")
