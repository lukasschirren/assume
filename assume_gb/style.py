# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Presentation style of the result figures: plain academic figures. A figure holds axes, axis
labels with units, legends and short panel labels, nothing else: no title, no subtitle, no source
line, no value labels. What a figure shows, its message and its source are its caption, written to
``captions.md`` beside the figures (one section per figure, kept up to date when a figure is
redrawn). Fonts, colours and the 16:9 shape are those of the GB data repository's summary figures
(``scripts/fig_adam_cfd_ro/style.py`` there; copied, not imported, so that this folder runs on its
own). Arial only.

Colours: the same Okabe-Ito hues and greys. Each group of plant keeps one colour in every figure
(``GROUP_COLOURS``, checked for colour-blind separation of neighbours in stack order); greys are
context. In price figures the simulated price is navy, the observed price ink, the merit-order
model a dashed grey.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAVY = "#0072B2"
SKY = "#56B4E9"
WINE = "#882255"
GREEN = "#009E73"
VERMILLION = "#D55E00"
GOLD = "#CCA700"
PINK = "#CC79A7"
INK = "#1A1A1A"
BODY = "#52514E"
MUTED = "#8A8880"
LIGHT = "#BDBBB4"
FAINT = "#E1E0D9"
WHITE = "#FFFFFF"

GROUP_COLOURS = {
    "Wind": SKY,
    "Solar": GOLD,
    "Hydro": NAVY,
    "Biomass": GREEN,
    "Waste and other": LIGHT,
    "Gas": VERMILLION,
    "Coal and oil": BODY,
    "Imports": PINK,
    "Storage": WINE,
    "Unserved": INK,
    "Exports": MUTED,
    "Demand": INK,
}
SIMULATED, OBSERVED, MODEL = NAVY, INK, MUTED

W_IN, H_IN = 10.0, 5.625  # 16:9
CAPTIONS = "captions.md"


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "font.size": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.edgecolor": LIGHT,
            "axes.linewidth": 0.8,
            "axes.labelcolor": BODY,
            "xtick.color": LIGHT,
            "ytick.color": LIGHT,
            "xtick.labelcolor": BODY,
            "ytick.labelcolor": BODY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": FAINT,
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "legend.fontsize": 10.5,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def caption(message: str, description: str, source: str) -> str:
    """A figure caption: the message as a bold lead sentence, then what the figure shows, then the source."""

    def sentence(text: str) -> str:
        text = text.strip()
        return text if not text or text.endswith((".", "!", "?")) else text + "."

    parts = [
        f"**{sentence(message)}**" if message.strip() else "",
        sentence(description),
        sentence(source),
    ]
    return " ".join(p for p in parts if p)


def figure(
    message: str, description: str, source: str, w: float = W_IN, h: float = H_IN
):
    """A plain 16:9 figure. ``message``, ``description`` and ``source`` are not drawn: they are the
    caption, which ``save`` writes to ``captions.md``. Returns (fig, top, bottom): the figure
    fractions between which the plot area should sit, leaving room for a legend above it and the
    x axis below it."""
    setup()
    fig = plt.figure(figsize=(w, h))
    fig.caption_text = caption(message, description, source)
    top = 1 - 0.07 * H_IN / h
    bottom = 0.12 * H_IN / h
    return fig, top, bottom


def axes(fig, top, bottom, left=0.09, right=0.97):
    return fig.add_axes([left, bottom, right - left, top - bottom])


def two(fig, top, bottom, sharey=True, gap=0.09):
    """Two panels side by side."""
    width = (0.97 - 0.08 - gap) / 2
    a = fig.add_axes([0.08, bottom, width, top - bottom])
    b = fig.add_axes(
        [0.08 + width + gap, bottom, width, top - bottom], sharey=a if sharey else None
    )
    return a, b


def stacked(fig, top, bottom, ratios=(1, 1), hspace=0.04, left=0.09, right=0.97):
    """Panels one above the other on a shared x axis, heights in ``ratios``."""
    total = top - bottom - hspace * (len(ratios) - 1)
    out, y = [], top
    for i, r in enumerate(ratios):
        h = total * r / sum(ratios)
        ax = fig.add_axes(
            [left, y - h, right - left, h], sharex=out[0] if out else None
        )
        out.append(ax)
        y -= h + hspace
    for ax in out[:-1]:
        ax.tick_params(labelbottom=False)
    return out


def panel_label(ax, text):
    """A short heading at the top left of a panel, in body ink."""
    ax.text(
        0.0,
        1.02,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=BODY,
        fontweight="bold",
    )


def save(fig, out: Path, stem: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png", dpi=200)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    text = getattr(fig, "caption_text", "")
    if text:
        write_caption(out, stem, text)
    return out / f"{stem}.png"


def read_captions(path: Path) -> dict[str, str]:
    """The captions of ``captions.md``, by figure name."""
    if not path.exists():
        return {}
    entries: dict[str, str] = {}
    key, body = None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if key:
                entries[key] = "\n".join(body).strip()
            key, body = line[3:].strip(), []
        elif key:
            body.append(line)
    if key:
        entries[key] = "\n".join(body).strip()
    return entries


def write_caption(out: Path, stem: str, text: str) -> Path:
    """Puts the caption of figure ``stem`` into ``captions.md`` in ``out``, replacing an older one;
    the other figures' captions are kept, in the order of their names."""
    path = out / CAPTIONS
    entries = read_captions(path)
    entries[stem] = text
    lines = ["# Figure captions", ""]
    for key in sorted(entries):
        lines += [f"## {key}", "", entries[key], ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
