# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""The figures of the validation, in the style of ``style``: six core figures, one per criterion
they show, and five supplementary ones. Every plotting function returns (fig, ax), ``ax`` an array
where there are several panels, and draws no title: the captions belong in LaTeX. ``style.save``
writes PDF and PNG at 300 dpi.

    core           select_weeks + plot_weeks (1), plot_duration_curve (3), plot_error_heatmap (2),
                   plot_coefficients (5), plot_markups (6), plot_scorecard (all)
    supplementary  plot_scatter, plot_taylor, plot_profiles, plot_pit, plot_price_vs_residual_demand
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from assume_gb.validation import mechanisms, metrics, reference, style
from assume_gb.validation.contract import (
    ABM,
    COMPETITIVE,
    HEADLINE,
    NAIVE,
    OBSERVED,
    ValidationData,
    Windows,
    local_day,
)
from assume_gb.validation.scorecard import BY_KEY, Scorecard, number

WEEK_RULES = ("median_error", "worst_error", "windiest", "scarcest")
MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _names(
    data: ValidationData, models: list[str] | None, naive: bool = True
) -> list[str]:
    names = list(data.names if models is None else models)
    return names if naive else [name for name in names if name != NAIVE]


def _letter(ax: Axes, i: int) -> None:
    """A panel letter at the top left, outside the plot area: (a), (b), ..."""
    ax.text(
        -0.01,
        1.02,
        f"({chr(97 + i)})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=style.INK,
    )


def _series(data: ValidationData, name: str) -> pd.Series:
    return data.observed if name == OBSERVED else data.point(name)


# ------------------------------------------------------------------ 1 weeks
def select_weeks(
    data: ValidationData, model: str = ABM, window: str | Windows | None = HEADLINE
) -> pd.DataFrame:
    """The weeks of the overlay figure, chosen by rule among the complete local weeks (Monday to
    Sunday) of ``window``, each week once:

        median_error   the week whose MAE of ``model`` is nearest the median weekly MAE
        worst_error    the week with the largest MAE of ``model``
        windiest       the highest mean wind share (outturn wind over demand)
        scarcest       the lowest mean capacity margin (available capacity less residual demand),
                       or the highest mean observed price without the explanatory data

    One row per rule: the week's first day (local midnight), the rule, the value that chose it
    and a sentence for the caption. Criterion 1 [lagoForecastingDayaheadElectricity2021]."""
    sample = data.sample(window, [model])
    days = local_day(sample.index)
    week = days - pd.to_timedelta(days.dayofweek, unit="D")
    expected = {
        start: (
            (start + pd.Timedelta(days=7)).tz_localize(data.tz)
            - start.tz_localize(data.tz)
        )
        // data.resolution
        for start in week.unique()
    }
    counts = pd.Series(1, index=sample.index).groupby(week).sum()
    complete = [start for start, count in counts.items() if count == expected[start]]
    if not complete:
        raise ValueError("the window holds no complete week")
    error = (sample[model] - sample[OBSERVED]).abs().groupby(week).mean().loc[complete]
    candidates = {
        "median_error": (error - error.median()).abs().sort_values(),
        "worst_error": error.sort_values(ascending=False),
    }
    if data.exog is not None:
        exog = data.exog.loc[sample.index]
        share = (exog["wind"] / exog["demand"]).groupby(week).mean().loc[complete]
        candidates["windiest"] = share.sort_values(ascending=False)
        if "available_capacity" in exog:
            margin = (
                (exog["available_capacity"] - exog["residual_demand"])
                .groupby(week)
                .mean()
                .loc[complete]
            )
            candidates["scarcest"] = margin.sort_values()
    if "scarcest" not in candidates:
        candidates["scarcest"] = (
            sample[OBSERVED]
            .groupby(week)
            .mean()
            .loc[complete]
            .sort_values(ascending=False)
        )
    reasons = {
        "median_error": lambda w: (
            f"median weekly MAE of {style.label(model)}: {error[w]:.1f} {style.PRICE}"
        ),
        "worst_error": lambda w: (
            f"largest weekly MAE of {style.label(model)}: {error[w]:.1f} {style.PRICE}"
        ),
        "windiest": lambda w: (
            f"highest mean wind share: {100 * candidates['windiest'][w]:.0f}% of demand"
        ),
        "scarcest": lambda w: (
            f"lowest mean capacity margin: {candidates['scarcest'][w]:.1f} GW"
            if data.exog is not None and "available_capacity" in data.exog
            else f"highest mean observed price: {candidates['scarcest'][w]:.1f} {style.PRICE}"
        ),
    }
    rows, taken = [], set()
    for rule in WEEK_RULES:
        if rule not in candidates:
            continue
        start = next((w for w in candidates[rule].index if w not in taken), None)
        if start is None:
            continue
        taken.add(start)
        rows.append(
            {
                "week": start.tz_localize(data.tz),
                "rule": rule,
                "value": float(candidates[rule][start]),
                "reason": reasons[rule](start),
            }
        )
    return pd.DataFrame(rows)


def plot_weeks(
    data: ValidationData,
    selection: pd.DataFrame | None = None,
    models: list[str] | None = None,
    model: str = ABM,
    window: str | Windows | None = HEADLINE,
) -> tuple[Figure, np.ndarray]:
    """Criterion 1: the observed price (black) and the models over the weeks of ``selection``
    (``select_weeks`` by default), one panel per week, an ensemble as its seed mean with the band
    of its seeds (minimum to maximum); the calibration and headline windows shaded. GBP/MWh
    [lagoForecastingDayaheadElectricity2021]."""
    selection = select_weeks(data, model, window) if selection is None else selection
    names = _names(data, models, naive=False)
    fig, axes = style.figure(
        "double",
        height=1.55 * len(selection) + 0.4,
        nrows=len(selection),
        squeeze=False,
    )
    axes = axes[:, 0]
    for i, (ax, row) in enumerate(zip(axes, selection.itertuples())):
        start, end = row.week, row.week + pd.Timedelta(days=7)
        style.shade_windows(ax, data)
        for name in [*names, OBSERVED]:
            series = _series(data, name).loc[start : end - data.resolution]
            if name != OBSERVED and data.is_ensemble(name):
                members = data.members(name).loc[series.index]
                ax.fill_between(
                    series.index,
                    members.min(axis=1),
                    members.max(axis=1),
                    color=style.colour(name),
                    alpha=style.BAND_ALPHA,
                    linewidth=0,
                )
            ax.plot(series.index, series.to_numpy(), **style.line_kwargs(name))
        ax.set_xlim(start, end)
        ax.set_ylabel(style.PRICE)
        ax.xaxis.set_major_locator(mdates.DayLocator(tz=data.tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz=data.tz))
        _letter(ax, i)
    style.legend(axes[0])
    fig.tight_layout()
    return fig, axes


# ------------------------------------------------------------------ 3 duration curve
def plot_duration_curve(
    data: ValidationData,
    models: list[str] | None = None,
    window: str | Windows | None = HEADLINE,
    linthresh: float = 50.0,
) -> tuple[Figure, Axes]:
    """Criterion 3: the price duration curves, prices sorted from highest to lowest against the
    share of periods, of the observed price and of each model (an ensemble as the median of its
    seeds' curves with the band from their minimum to maximum), on a symmetric-log price axis that
    keeps both tails visible; the Wasserstein-1 distance of each model to the observed curve noted
    in the plot. GBP/MWh [nitschBacktestingAgentbasedModel2021]."""
    names = _names(data, models)
    sample = data.sample(window, names)
    n = len(sample)
    share = 100 * (np.arange(n) + 0.5) / n
    fig, ax = style.figure("single", height=2.8)
    notes = []
    for name in [*names, OBSERVED]:
        if name != OBSERVED and data.is_ensemble(name):
            curves = np.sort(data.ensemble(name, window, names).to_numpy(), axis=0)[
                ::-1
            ]
            ax.fill_between(
                share,
                curves.min(axis=1),
                curves.max(axis=1),
                color=style.colour(name),
                alpha=style.BAND_ALPHA,
                linewidth=0,
            )
            curve = np.median(curves, axis=1)
            w1 = metrics.wasserstein1(
                data.ensemble(name, window, names), sample[OBSERVED]
            )
        else:
            curve = np.sort(sample[name].to_numpy())[::-1]
            w1 = metrics.wasserstein1(sample[name], sample[OBSERVED])
        ax.plot(share, curve, **style.line_kwargs(name))
        if name != OBSERVED:
            notes.append(f"{style.label(name)} {w1:.1f}")
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of periods (%)")
    ax.set_ylabel(style.PRICE)
    note = "\n".join([f"W1 ({style.PRICE})", *notes])
    ax.text(
        0.03,
        0.04,
        note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=style.INK,
    )
    style.legend(ax)
    return fig, ax


# ------------------------------------------------------------------ 2 error heat map
def plot_error_heatmap(
    data: ValidationData,
    models: list[str] | None = None,
    window: str | Windows | None = HEADLINE,
) -> tuple[Figure, np.ndarray]:
    """Criterion 2: the mean error (model - observed) by local hour of the day and month, one panel
    per model, on a diverging scale (RdBu_r) centred at zero with symmetric limits shared by all
    panels. GBP/MWh [casarCanShadowPrices]."""
    names = _names(data, models, naive=False)
    sample = data.sample(window, names)
    index = sample.index
    tables = {}
    for name in names:
        error = sample[name] - sample[OBSERVED]
        tables[name] = error.groupby([index.hour, index.month]).mean().unstack()
    limit = max(float(np.nanmax(np.abs(t.to_numpy()))) for t in tables.values())
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit)
    fig, axes = style.figure(
        "double", height=2.6, ncols=len(names), squeeze=False, sharey=True
    )
    axes = axes[0]
    for i, (ax, name) in enumerate(zip(axes, names)):
        table = tables[name]
        image = ax.pcolormesh(
            np.arange(table.shape[1] + 1),
            np.arange(25),
            table.to_numpy(),
            cmap="RdBu_r",
            norm=norm,
            shading="flat",
        )
        ax.set_xticks(
            np.arange(table.shape[1]) + 0.5, [MONTHS[m - 1] for m in table.columns]
        )
        ax.set_yticks([0.5, 6.5, 12.5, 18.5, 23.5], ["0", "6", "12", "18", "23"])
        ax.set_xlabel(style.label(name))
        ax.grid(False)
        _letter(ax, i)
    axes[0].set_ylabel("Hour of the day (local)")
    bar = fig.colorbar(image, ax=list(axes), shrink=0.9)
    bar.set_label(f"Mean error ({style.PRICE})")
    return fig, axes


# ------------------------------------------------------------------ 5 coefficients
def causal_ranges(
    market: str,
    data: ValidationData | None = None,
    window: str | Windows | None = HEADLINE,
    band: bool = False,
    quantiles: tuple[float, float] = (0.05, 0.95),
) -> dict[str, tuple[float, float]]:
    """The range of the causal effect of wind and solar over the predicted penetration that occurs
    in ``window`` of ``data`` (between its ``quantiles``, on the reference definition of
    ``mechanisms.penetration``), GBP/MWh per GW: the range of the estimate, with ``band`` that of
    its 80% band too. Without data, or without the reference definition, over the whole curve;
    never extrapolated. The shaded band of ``plot_coefficients``
    [cacciarelliWeActuallyUnderstand2025]."""
    ranges = {}
    for technology in reference.TECHNOLOGIES:
        grid = reference.cate_curve(market, technology).index.to_numpy()
        if data is not None and data.exog is not None:
            try:
                share = mechanisms.penetration(
                    data.explanatory(window), technology
                ).dropna()
                grid = np.linspace(*np.quantile(share, quantiles), 101)
            except ValueError:
                pass  # no reference definition of penetration: the whole curve
        curve = reference.cate_at(grid, market, technology)
        values = curve[
            ["estimate", "ci80_low", "ci80_high"] if band else ["estimate"]
        ].to_numpy()
        if np.isfinite(values).any():
            ranges[technology] = (float(np.nanmin(values)), float(np.nanmax(values)))
    return ranges


def plot_coefficients(
    regressions: pd.DataFrame,
    ranges: dict[str, tuple[float, float]] | None = None,
    terms: tuple[str, ...] = mechanisms.TERMS,
) -> tuple[Figure, np.ndarray]:
    """Criterion 5: forest plot of the coefficients of the regression run on the observed price and
    on each model's (``mechanisms.compare_regressions``) with their 95% intervals, one row of panels
    per term (each its own unit) and one column per regime; where ``ranges`` gives one, the range
    of the causal estimate is shaded (``causal_ranges``; the solar one is indicative)
    [cacciarelliWeActuallyUnderstand2025] [pangalloDataDrivenEconomicAgentBased2024]."""
    regimes = list(dict.fromkeys(regressions["regime"]))
    models = list(dict.fromkeys(regressions["model"]))
    units = {
        "wind": "per GW",
        "solar": "per GW",
        "demand": "per GW",
        "gas": "per £/MWh",
        "carbon": "per £/t",
    }
    fig, axes = style.figure(
        "double",
        height=0.75 * len(terms) + 0.6,
        nrows=len(terms),
        ncols=len(regimes),
        squeeze=False,
        sharex="row",
    )
    for col, regime in enumerate(regimes):
        rows = regressions[regressions["regime"] == regime]
        for row, term in enumerate(terms):
            ax = axes[row, col]
            sub = rows[rows["term"] == term]
            if ranges and term in ranges:
                low, high = ranges[term]
                ax.axvspan(
                    low,
                    high,
                    color=style.MUTED,
                    alpha=0.18,
                    linewidth=0,
                    label="Causal estimate (range)",
                )
            if len(sub):
                first = sub.iloc[0]
                points = [
                    (OBSERVED, first.observed, first.observed_low, first.observed_high)
                ]
                points += [
                    (r.model, r.simulated, r.simulated_low, r.simulated_high)
                    for r in sub.itertuples()
                ]
                for k, (name, estimate, low, high) in enumerate(points):
                    y = -k
                    ax.errorbar(
                        estimate,
                        y,
                        xerr=[[estimate - low], [high - estimate]],
                        fmt="o",
                        markersize=3,
                        color=style.colour(name),
                        elinewidth=1,
                        capsize=0,
                        label=style.label(name),
                    )
            ax.set_yticks([])
            ax.set_ylim(-len(models) - 0.6, 0.6)
            ax.grid(axis="y", visible=False)
            if col == 0:
                ax.set_ylabel(
                    f"{term}\n{units.get(term, '')}",
                    rotation=0,
                    ha="right",
                    va="center",
                )
            if row == 0:
                heading = "Whole window" if regime == "all" else str(regime)
                ax.text(
                    0.0,
                    1.05,
                    heading,
                    transform=ax.transAxes,
                    color=style.INK,
                    va="bottom",
                )
    axes[-1, 0].set_xlabel(f"Coefficient ({style.PRICE} per unit)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(handles), 5),
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout()
    return fig, axes


# ------------------------------------------------------------------ 6 markups
def plot_markups(table: pd.DataFrame) -> tuple[Figure, Axes]:
    """Criterion 6: box plots of the markup over the competitive run by bin of scarcity
    (``mechanisms.markups``): the observed price's (observed - competitive) in black beside each
    model's (model - competitive); boxes from the 25th to the 75th percentile with the median,
    whiskers from the 5th to the 95th percentile. GBP/MWh [borensteinMeasuringMarketInefficiencies2002]
    [twomeyMonitoringMarketPower]."""
    bins = (
        table[["scarcity_low", "scarcity_high"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    series = [s for s in dict.fromkeys(table["series"]) if s != COMPETITIVE]
    width = 0.8 / len(series)
    fig, ax = style.figure("double", height=2.6)
    for j, name in enumerate(series):
        stats, positions = [], []
        for i, edges in bins.iterrows():
            row = table[
                (table["series"] == name)
                & (table["scarcity_low"] == edges.scarcity_low)
            ]
            if not len(row):
                continue
            r = row.iloc[0]
            stats.append(
                {
                    "med": r["median"],
                    "q1": r["q25"],
                    "q3": r["q75"],
                    "whislo": r["q05"],
                    "whishi": r["q95"],
                    "fliers": [],
                }
            )
            positions.append(i - 0.4 + width * (j + 0.5))
        colour = style.colour(name)
        ax.bxp(
            stats,
            positions=positions,
            widths=width * 0.8,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": colour, "alpha": 0.35, "edgecolor": colour},
            medianprops={"color": colour, "linewidth": 1.2},
            whiskerprops={"color": colour},
            capprops={"color": colour},
        )
        ax.plot(
            [],
            [],
            color=colour,
            linewidth=4,
            alpha=0.5,
            label="Observed − competitive"
            if name == OBSERVED
            else f"{style.label(name)} − competitive",
        )
    ax.axhline(0.0, color=style.MUTED, linewidth=0.6)
    edges = [
        f"{low:.2f} to {high:.2f}".replace("-", "−")
        for low, high in bins.itertuples(index=False)
    ]
    ax.set_xticks(range(len(bins)), edges)
    ax.set_xlabel("Scarcity: residual demand / available capacity")
    ax.set_ylabel(f"Markup ({style.PRICE})")
    style.legend(ax)
    return fig, ax


# ------------------------------------------------------------------ scorecard
def plot_scorecard(card: Scorecard) -> tuple[Figure, Axes]:
    """The scorecard as a heat map, metrics by series: each cell coloured by the model's skill
    against the naive forecast (blue better, red worse, clipped to [-1, 1]; grey where a skill does
    not apply, e.g. p-values and the observed column) and annotated with its value, the
    Diebold-Mariano stars of beating the naive forecast on the MAE row
    [lagoForecastingDayaheadElectricity2021]."""
    values, skills, stars = card.values, card.skill, card.stars
    rows, columns = list(values.index), list(values.columns)
    fig, ax = style.figure("double", height=0.19 * len(rows) + 0.9)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)
    cmap = LinearSegmentedColormap.from_list("skill", ["#B2182B", "#F7F7F7", "#2166AC"])
    shown = np.ma.masked_invalid(skills.to_numpy(dtype=float).clip(-1, 1))
    ax.set_facecolor("#F2F2F2")
    image = ax.pcolormesh(
        np.arange(len(columns) + 1),
        np.arange(len(rows) + 1),
        shown,
        cmap=cmap,
        norm=norm,
        edgecolors="white",
        linewidth=0.8,
    )
    for i, key in enumerate(rows):
        for j, name in enumerate(columns):
            value = values.loc[key, name]
            if not np.isfinite(value):
                continue
            strong = (
                np.isfinite(skills.loc[key, name]) and abs(skills.loc[key, name]) > 0.6
            )
            ax.text(
                j + 0.5,
                i + 0.5,
                number(key, value) + stars.loc[key, name],
                ha="center",
                va="center",
                fontsize=style.FONT_SIZE - 1.5,
                color="white" if strong else style.INK,
            )
    ax.set_xticks(np.arange(len(columns)) + 0.5, [style.label(c) for c in columns])
    ax.xaxis.tick_top()
    labels = [f"{BY_KEY[key].criterion} {BY_KEY[key].label}" for key in rows]
    ax.set_yticks(np.arange(len(rows)) + 0.5, labels)
    ax.set_ylim(len(rows), 0)
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    bar = fig.colorbar(image, ax=ax, shrink=0.5, aspect=25)
    bar.set_label("Skill against the naive forecast")
    return fig, ax


# ------------------------------------------------------------------ supplementary
def plot_scatter(
    data: ValidationData,
    models: list[str] | None = None,
    window: str | Windows | None = HEADLINE,
) -> tuple[Figure, np.ndarray]:
    """Supplementary: simulated against observed prices as a hexbin density, one panel per model,
    with the 45-degree line and the MAE, bias and Pearson r noted. GBP/MWh
    [maurerKnowYourTools2024]."""
    names = _names(data, models, naive=False)
    sample = data.sample(window, names)
    low, high = np.nanpercentile(sample.to_numpy(), [0.5, 99.5])
    fig, axes = style.figure(
        "double", height=2.6, ncols=len(names), squeeze=False, sharey=True
    )
    axes = axes[0]
    for i, (ax, name) in enumerate(zip(axes, names)):
        ax.hexbin(
            sample[OBSERVED],
            sample[name],
            gridsize=45,
            extent=(low, high, low, high),
            mincnt=1,
            cmap="Greys",
            bins="log",
            linewidths=0,
        )
        ax.plot([low, high], [low, high], color=style.colour(name), linewidth=0.8)
        note = (
            f"MAE {metrics.mae(sample[name], sample[OBSERVED]):.1f}\n"
            f"bias {metrics.bias(sample[name], sample[OBSERVED]):+.1f}\n"
            f"r {metrics.pearson_r(sample[name], sample[OBSERVED]):.2f}"
        )
        ax.text(
            0.04,
            0.96,
            note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=style.INK,
        )
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Observed ({style.PRICE})")
        ax.text(
            0.96,
            0.04,
            style.label(name),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=style.INK,
        )
        _letter(ax, i)
    axes[0].set_ylabel(f"Simulated ({style.PRICE})")
    return fig, axes


def plot_taylor(
    data: ValidationData,
    models: list[str] | None = None,
    window: str | Windows | None = HEADLINE,
) -> tuple[Figure, Axes]:
    """Supplementary: Taylor diagram, each model at the radius of its sigma ratio and the angle of
    its Pearson r (arccos r) against the observed price at (1, 0), with contours of the centred
    RMSE relative to the observed standard deviation, sqrt(1 + s^2 - 2 s r) [maurerKnowYourTools2024]
    [casarCanShadowPrices]."""
    names = _names(data, models)
    sample = data.sample(window, names)
    points = {
        name: (
            metrics.sigma_ratio(sample[name], sample[OBSERVED]),
            metrics.pearson_r(sample[name], sample[OBSERVED]),
        )
        for name in names
    }
    radius = max(1.5, 1.15 * max(s for s, _ in points.values()))
    fig, ax = style.figure("single", height=3.2, subplot_kw={"projection": "polar"})
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_rlim(0, radius)
    correlations = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
    ax.set_thetagrids(
        np.degrees(np.arccos(correlations)), [f"{r:g}" for r in correlations]
    )
    theta = np.linspace(0, np.pi / 2, 200)
    for level in (0.25, 0.5, 0.75, 1.0, 1.25):
        # centred RMSE contour around the observed point (1, 0): s^2 - 2 s cos(theta) + 1 = level^2
        disc = np.cos(theta) ** 2 - 1 + level**2
        s = np.where(disc >= 0, np.cos(theta) + np.sqrt(np.clip(disc, 0, None)), np.nan)
        ax.plot(theta, s, color=style.GRID, linewidth=0.8, zorder=1)
    ax.plot(
        0,
        1,
        marker="*",
        markersize=9,
        color=style.colour(OBSERVED),
        linestyle="none",
        label=style.label(OBSERVED),
        zorder=4,
    )
    for name, (sigma, r) in points.items():
        ax.plot(
            np.arccos(np.clip(r, -1, 1)),
            sigma,
            marker="o",
            markersize=5,
            color=style.colour(name),
            linestyle="none",
            label=style.label(name),
            zorder=3,
        )
    ax.set_xlabel("Standard deviation ratio", labelpad=14)
    ax.text(
        np.radians(45),
        radius * 1.25,
        "Pearson r",
        ha="center",
        va="center",
        color=style.INK,
        rotation=-45,
    )
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4)
    return fig, ax


def plot_profiles(
    data: ValidationData,
    models: list[str] | None = None,
    window: str | Windows | None = HEADLINE,
) -> tuple[Figure, np.ndarray]:
    """Supplementary: the mean price by local hour of the day and by weekday for the observed price
    and each model, with the 25th and 75th percentiles: a band for the observed price, thin lines
    for the models. GBP/MWh [casarCanShadowPrices]."""
    names = _names(data, models, naive=False)
    sample = data.sample(window, names)
    fig, axes = style.figure("double", height=2.4, ncols=2)
    keys = {
        "hour": (sample.index.hour, range(24), "Hour of the day (local)"),
        "weekday": (sample.index.dayofweek, range(7), ""),
    }
    for i, (ax, (by, (key, positions, xlabel))) in enumerate(zip(axes, keys.items())):
        for name in [*names, OBSERVED]:
            grouped = sample[name].groupby(key)
            low = grouped.quantile(0.25).reindex(positions)
            high = grouped.quantile(0.75).reindex(positions)
            if name == OBSERVED:
                ax.fill_between(
                    list(positions),
                    low,
                    high,
                    color=style.colour(name),
                    alpha=0.12,
                    linewidth=0,
                )
            else:
                for quartile in (low, high):
                    ax.plot(
                        list(positions),
                        quartile,
                        color=style.colour(name),
                        linestyle=style.linestyle(name),
                        linewidth=0.5,
                        alpha=0.8,
                    )
            ax.plot(
                list(positions),
                grouped.mean().reindex(positions),
                marker="o" if by == "weekday" else None,
                markersize=3,
                **style.line_kwargs(name),
            )
        if by == "weekday":
            ax.set_xticks(range(7), WEEKDAYS)
        else:
            ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlabel(xlabel)
        _letter(ax, i)
    axes[0].set_ylabel(style.PRICE)
    style.legend(axes[0])
    fig.tight_layout()
    return fig, axes


def plot_pit(
    data: ValidationData,
    model: str = ABM,
    window: str | Windows | None = HEADLINE,
    bins: int = 10,
) -> tuple[Figure, np.ndarray]:
    """Supplementary, criterion 8: the PIT histogram of the observed price within ``model``'s seed
    ensemble (density; a calibrated ensemble is flat at 1, a U shape too narrow) and the ensemble's
    CRPS by month. GBP/MWh for the CRPS [pinsonNonparametricProbabilisticForecasts2007]
    [nowotarskiRecentAdvancesElectricity2018]."""
    if not data.is_ensemble(model):
        raise ValueError(
            f"{model!r} is not an ensemble: PIT and CRPS need several seeds"
        )
    members = data.ensemble(model, window)
    observed = data.observed.loc[members.index]
    pit = metrics.pit_ensemble(members, observed)
    crps = metrics.crps_ensemble(members, observed)
    fig, axes = style.figure("double", height=2.3, ncols=2)
    colour = style.colour(model)
    axes[0].hist(
        pit,
        bins=np.linspace(0, 1, bins + 1),
        density=True,
        color=colour,
        alpha=0.6,
        edgecolor="white",
        linewidth=0.8,
    )
    axes[0].axhline(1.0, color=style.MUTED, linewidth=0.8, label="Calibrated")
    axes[0].set_xlabel("PIT")
    axes[0].set_ylabel("Density")
    monthly = crps.groupby(crps.index.strftime("%Y-%m")).mean()
    axes[1].bar(range(len(monthly)), monthly.to_numpy(), color=colour, width=0.6)
    axes[1].set_xticks(
        range(len(monthly)), [MONTHS[int(m[5:]) - 1] for m in monthly.index]
    )
    axes[1].set_ylabel(f"CRPS ({style.PRICE})")
    for i, ax in enumerate(axes):
        _letter(ax, i)
    style.legend(axes[0])
    fig.tight_layout()
    return fig, axes


def plot_price_vs_residual_demand(
    data: ValidationData,
    models: list[str] | None = None,
    window: str | Windows | None = HEADLINE,
    bins: int = 20,
) -> tuple[Figure, np.ndarray]:
    """Supplementary: the median price in quantile bins of residual demand (outturn, GW), with the
    band of its 25th to 75th percentile for the observed price, per regime (the whole window
    without regimes): the supply curve each price reveals [canalesEmpiricalEstimateElectricity2026]."""
    names = _names(data, models, naive=False)
    regimes = list(data.regimes) or [None]
    fig, axes = style.figure(
        "double", height=2.5, ncols=len(regimes), squeeze=False, sharey=True
    )
    axes = axes[0]
    for i, (ax, regime) in enumerate(zip(axes, regimes)):
        sample = data.sample(window, names, regime)
        demand = data.exog.loc[sample.index, "residual_demand"]
        cut = pd.qcut(demand, bins, duplicates="drop")
        centre = demand.groupby(cut, observed=True).median()
        for name in [*names, OBSERVED]:
            grouped = sample[name].groupby(cut, observed=True)
            if name == OBSERVED:
                ax.fill_between(
                    centre,
                    grouped.quantile(0.25),
                    grouped.quantile(0.75),
                    color=style.colour(name),
                    alpha=0.1,
                    linewidth=0,
                )
            ax.plot(
                centre,
                grouped.median(),
                marker="o",
                markersize=2.5,
                **style.line_kwargs(name),
            )
        ax.set_xlabel("Residual demand (GW)")
        if regime is not None:
            ax.text(
                0.0, 1.02, regime, transform=ax.transAxes, color=style.INK, va="bottom"
            )
        _letter(ax, i)
    axes[0].set_ylabel(style.PRICE)
    handles = [Line2D([], [], **style.line_kwargs(name)) for name in [OBSERVED, *names]]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout()
    return fig, axes
