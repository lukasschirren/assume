# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""The one place for the look of the validation figures: journal sizes (single column 3.5 in,
double column 7.2 in), 8 pt type, no titles (the captions are written in LaTeX), one colour per
model in every figure (Okabe-Ito), seed bands in the model's colour made transparent, prices in
£/MWh, the calibration and headline windows shaded on time axes, and every figure saved as PDF and
PNG at 300 dpi.

The run figures of ``assume_gb.figures`` have their own style (``assume_gb/style.py``: 16:9, 11
pt, captions in a markdown file); the hues are the same.

Colour vision: the four model hues pass the dataviz palette checks (7 Oct 2026) except pink
against green under deuteranopia (Delta E 7.6, acceptable with a second cue). The grey of the
naive forecast and the pink of an extra model are indistinguishable under deuteranopia (Delta E
1.9), so the naive forecast is always dashed and an extra model dash-dotted, and every figure with
two or more series carries a legend.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from assume_gb.validation.contract import (
    ABM,
    CALIBRATION,
    COMPETITIVE,
    HEADLINE,
    LEAR,
    NAIVE,
    OBSERVED,
    ValidationData,
)

SINGLE, DOUBLE = 3.5, 7.2  # column widths, inches
FONT_SIZE = 8
DPI = 300
PRICE = "£/MWh"

EXTRA = "extra"
COLOURS = {
    OBSERVED: "#000000",
    NAIVE: "#999999",
    COMPETITIVE: "#0072B2",
    ABM: "#D55E00",
    LEAR: "#009E73",
    EXTRA: "#CC79A7",
}
LINESTYLES = {NAIVE: (0, (4, 2)), EXTRA: (0, (5, 1.5, 1, 1.5))}
LABELS = {
    OBSERVED: "Observed",
    NAIVE: "Naive",
    COMPETITIVE: "Competitive",
    ABM: "ABM",
    LEAR: "LEAR",
}
BAND_ALPHA = 0.25

INK = "#1A1A1A"  # text
MUTED = "#6B6A66"  # secondary text, reference marks
GRID = "#E1E0D9"  # hairline grid
WINDOW_SHADES = {CALIBRATION: "#D9D9D9", HEADLINE: "#F2F2F2"}
WINDOW_LABELS = {CALIBRATION: "Calibration", HEADLINE: "Headline (test)"}

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "legend.frameon": False,
    "axes.linewidth": 0.6,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,  # editable text in the PDF
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.dpi": 100,
}


# the font subsetting of the PDF backend reports every table it prunes at INFO
logging.getLogger("fontTools").setLevel(logging.WARNING)


def apply() -> None:
    """Sets matplotlib's rcParams to the style of these figures."""
    mpl.rcParams.update(RC)


def colour(name: str) -> str:
    """The colour of a series: its own for the known models, the extra pink for any other."""
    return COLOURS.get(name, COLOURS[EXTRA])


def linestyle(name: str) -> str | tuple:
    """Dashed for the naive forecast, dash-dotted for an extra model, solid otherwise."""
    return LINESTYLES.get(name, "-" if name in COLOURS else LINESTYLES[EXTRA])


def label(name: str) -> str:
    """The legend name of a series."""
    return LABELS.get(name, name)


def line_kwargs(name: str) -> dict:
    """The colour, line style, label and z-order of a series' line (the observed price on top)."""
    return {
        "color": colour(name),
        "linestyle": linestyle(name),
        "label": label(name),
        "zorder": 3 if name == OBSERVED else 2,
    }


def figure(
    width: float | str = "single",
    height: float | None = None,
    nrows: int = 1,
    ncols: int = 1,
    **kwargs,
) -> tuple[Figure, Axes]:
    """A figure of a column's width ("single", 3.5 in, "double", 7.2 in, or inches) and the given
    height (by default 0.7 of the width per row, capped at 9 in), with its axes, in this style."""
    apply()
    inches = {"single": SINGLE, "double": DOUBLE}.get(width, width)
    height = (
        height
        if height is not None
        else min(9.0, 0.7 * inches * nrows / max(ncols, 1) ** 0.5)
    )
    return plt.subplots(nrows, ncols, figsize=(inches, height), **kwargs)


def legend(ax: Axes, **kwargs) -> None:
    """A legend above the plot area, left-aligned, one row if it fits."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(0, 1.0),
            ncol=min(len(handles), 5),
            **kwargs,
        )


def shade_windows(
    ax: Axes, data: ValidationData, windows: tuple[str, ...] = (CALIBRATION, HEADLINE)
) -> None:
    """Shades the calibration and headline windows of ``data`` on a time axis, as contiguous spans
    of their periods; the headline window lighter. Only spans inside the x range are drawn."""
    index = data.observed.index
    for name in windows:
        mask = data.window(name) if name == HEADLINE or name in data.periods else None
        if mask is None or not mask.any():
            continue
        runs = pd.Series(mask, index=index)
        starts = index[mask & ~runs.shift(1, fill_value=False).to_numpy()]
        ends = index[mask & ~runs.shift(-1, fill_value=False).to_numpy()]
        for i, (start, end) in enumerate(zip(starts, ends)):
            ax.axvspan(
                start,
                end + data.resolution,
                color=WINDOW_SHADES[name],
                alpha=0.6,
                linewidth=0,
                zorder=0,
                label=WINDOW_LABELS[name] if i == 0 else None,
            )


def save(
    fig: Figure, stem: str | Path, formats: tuple[str, ...] = ("pdf", "png")
) -> list[Path]:
    """Writes ``fig`` as ``<stem>.pdf`` and ``<stem>.png`` (300 dpi) and closes it."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=DPI)
        paths.append(path)
    plt.close(fig)
    return paths
