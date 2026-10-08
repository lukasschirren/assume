# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The standard figures of one run of a GB scenario, written as PNG and PDF beside the numbers
they quote (``numbers.json``, ``numbers.csv``), their captions (``captions.md``) and the headline
numbers as a table (``00_summary.md``).

    python -m assume_gb figures <run folder> [--baseline <run folder>]
    python -m assume_gb run --year 2023 --case day_ahead --csv DIR --figures

The figures carry no text beyond axes, legends and panel labels. Each caption states the figure's
message, computed from the run so that it changes with the results, then what the figure shows and
the source. The set, and the literature each figure follows, is described in the README of this
folder ("Figures"). A figure that needs something the run does not have (observed prices, storage,
an intraday market, a baseline) is left out.

Comparison basis, as in ``compare``: the simulated day-ahead price against the N2EX hourly auction,
on hourly means of the periods the merit-order model admits. Times are UTC.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from assume_gb import compare, convert
from assume_gb import style as S
from assume_gb.results import EXPORTS, GROUPS, HOURS, Run

logger = logging.getLogger(__name__)
logging.getLogger("fontTools").setLevel(logging.WARNING)

OBSERVED = "n2ex_day_ahead"
OBSERVED_NAME = "N2EX"
QUARTERS = {1: "Jan to Mar", 2: "Apr to Jun", 3: "Jul to Sep", 4: "Oct to Dec"}
INTERCONNECTORS = "Interconnectors"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "navy", ["#F4F3EE", "#A9CCE6", S.NAVY, "#00385A"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "navy_wine", [S.NAVY, "#9DC3E2", "#F2F1EC", "#D9A3BE", S.WINE]
)
COLOURS = {**S.GROUP_COLOURS, INTERCONNECTORS: S.PINK}


# ================================================================== helpers
def gbp(x: float, digits: int = 0) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}£{abs(x):,.{digits}f}"


def pct(x: float, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f}%"


def lower_higher(x: float) -> str:
    return "lower" if x < 0 else "higher"


def n_units(run: Run) -> int:
    return len([u for u in run.position.columns if u != convert.DEMAND_UNIT])


def source(run: Run, observed: bool = True) -> str:
    text = (
        f"Simulation: ASSUME, scenario {run.scenario}, case {run.case}; {n_units(run)} units bidding in "
        f"uniform-price auctions for each half-hour (UTC)."
    )
    if observed and has_observed(run):
        text += f" Observed: {OBSERVED_NAME} hourly day-ahead auction (Nord Pool), compared on hourly means; periods with flagged inputs left out."
    return text


def has_observed(run: Run) -> bool:
    return OBSERVED in run.observed and run.observed[OBSERVED].notna().any()


def hourly_pair(run: Run) -> pd.DataFrame:
    """Simulated and observed day-ahead price on the hours made up of admissible periods."""
    sim = compare.hourly(run.day_ahead, run.admissible)
    obs = compare.hourly(run.observed[OBSERVED], run.admissible)
    return pd.concat({"simulated": sim, "observed": obs}, axis=1).dropna()


def daily(series: pd.Series) -> pd.Series:
    return series.resample("D").mean()


def date_axis(ax, monthly: bool = True):
    if monthly:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))


def legend_above(ax, handles, labels, ncol=None, **kw):
    """A legend in one row just above the plot area, left-aligned."""
    ax.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=ncol or len(labels),
        handlelength=1.4,
        columnspacing=1.2,
        borderaxespad=0.3,
        **kw,
    )


def line_handles(items):
    return [Line2D([], [], color=c, lw=lw, ls=ls) for c, lw, ls in items]


def patch_handles(groups):
    return [Patch(color=COLOURS[g]) for g in groups]


def price_lines(ax):
    legend_above(
        ax,
        line_handles([(S.OBSERVED, 1.1, "-"), (S.SIMULATED, 1.5, "-")]),
        [f"Observed ({OBSERVED_NAME})", "Simulated"],
    )


def colourbar(fig, mappable, cax=None, ax=None, label="", **kw):
    cb = fig.colorbar(mappable, cax=cax, ax=ax, **kw)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, colors=S.LIGHT, labelcolor=S.BODY)
    if label:
        cb.set_label(label, color=S.BODY, fontsize=10)
    return cb


def stack_area(ax, frame: pd.DataFrame):
    """Stacked areas: positive groups upwards from zero, negative ones (exports, charging) downwards."""
    x = frame.index
    up, down = np.zeros(len(frame)), np.zeros(len(frame))
    for group in frame.columns:
        values = frame[group].values / 1e3
        pos, neg = np.clip(values, 0, None), np.clip(values, None, 0)
        if pos.any():
            ax.fill_between(x, up, up + pos, color=COLOURS[group], lw=0, step="post")
            up = up + pos
        if neg.any():
            ax.fill_between(
                x, down, down + neg, color=COLOURS[group], lw=0, step="post"
            )
            down = down + neg


def present_groups(frame: pd.DataFrame) -> list[str]:
    return [g for g in frame.columns if frame[g].abs().sum() > 0]


# ================================================================== numbers
def price_stats(series: pd.Series) -> dict:
    s = series.dropna()
    return {
        "mean": s.mean(),
        "std": s.std(),
        "p05": s.quantile(0.05),
        "p50": s.median(),
        "p95": s.quantile(0.95),
        "min": s.min(),
        "max": s.max(),
        "negative_share": (s < 0).mean(),
        "daily_range": (s.resample("D").max() - s.resample("D").min()).mean(),
    }


def numbers(run: Run) -> dict:
    """The headline numbers of a run: what the figures quote, kept in ``numbers.json``."""
    out: dict = {
        "run": run.name,
        "scenario": run.scenario,
        "case": run.case,
        "markets": run.markets,
        "units": n_units(run),
        "periods": len(run.index),
        "start": str(run.index[0]),
        "end": str(run.index[-1]),
        "periods_left_out": int((~run.admissible).sum()),
    }
    for market_id in run.markets:
        out[f"price_{market_id}"] = price_stats(run.prices[market_id])
    if has_observed(run):
        pair = hourly_pair(run)
        fit = compare.difference(pair["simulated"], pair["observed"])
        fit["correlation"] = pair.corr().iloc[0, 1]
        fit["std_ratio"] = pair["simulated"].std() / pair["observed"].std()
        fit["slope_observed_on_simulated"] = float(
            np.polyfit(pair["simulated"], pair["observed"], 1)[0]
        )
        p99 = pair["observed"].quantile(0.99)
        fit["observed_p99"] = p99
        fit["hours_above_observed_p99"] = {
            "simulated": int((pair["simulated"] > p99).sum()),
            "observed": int((pair["observed"] > p99).sum()),
        }
        out["fit_hourly"] = fit
        out["price_observed_hourly"] = price_stats(pair["observed"])
        out["price_simulated_hourly"] = price_stats(pair["simulated"])
        model = run.reference.get("model_no_storage")
        if model is not None:
            mae_model = compare.difference(
                compare.hourly(model, run.admissible), pair["observed"]
            )["mae"]
            fit["mae_relative_to_merit_order_model"] = (
                fit["mae"] / mae_model if mae_model else np.nan
            )
    for name, series in run.reference.items():
        if name != "scenario_merit_order":
            out[f"vs_{name}"] = compare.difference(
                run.day_ahead, series[run.admissible]
            )
    # the reference merit order is that of the scenario's offers without storage and on the outturn: it applies
    # to the day-ahead case only; the order-book check below applies to every market
    out["vs_scenario_merit_order_max_abs"] = float(
        (run.day_ahead - run.reference["scenario_merit_order"]).abs().max()
    )
    out["merit_order_check"] = {m: merit_order_check(run, m) for m in run.markets}

    gen = run.generation
    twh = gen.clip(lower=0).sum() * HOURS / 1e6
    out["demand_served_twh"] = run.gb_demand.sum() * HOURS / 1e6
    out["energy_twh"] = twh.drop(EXPORTS).to_dict()
    out["exports_twh"] = -gen[EXPORTS].sum() * HOURS / 1e6
    out["imports_twh"] = gen["Imports"].sum() * HOURS / 1e6 if "Imports" in gen else 0.0
    out["share_of_supply"] = (twh.drop(EXPORTS) / twh.drop(EXPORTS).sum()).to_dict()
    out["consumer_cost_gbp_bn"] = (run.day_ahead * run.gb_demand).sum() * HOURS / 1e9
    out["demand_weighted_price"] = (
        run.day_ahead * run.gb_demand
    ).sum() / run.gb_demand.sum()
    setter = run.marginal(convert.MARKET_ID).group.replace(
        {"Imports": INTERCONNECTORS, EXPORTS: INTERCONNECTORS}
    )
    out["price_setter_share"] = setter.value_counts(normalize=True).to_dict()
    out["capture"] = run.capture_prices().to_dict(orient="index")
    if "Storage" in gen and gen["Storage"].abs().sum() > 0:
        out["storage"] = storage_numbers(run)
    if convert.INTRADAY_MARKET_ID in run.markets:
        spread = run.prices[convert.INTRADAY_MARKET_ID] - run.day_ahead
        out["intraday_spread"] = {
            "mean": spread.mean(),
            "mean_abs": spread.abs().mean(),
            "imbalance_mean_mw": run.imbalance.mean(),
            "imbalance_std_mw": run.imbalance.std(),
        }
    return out


def merit_order_check(run: Run, market_id: str) -> dict:
    """Does each clearing respect its own order book? An offer to sell below the price must be accepted in
    full and one above it rejected, a bid to buy above the price accepted in full and one below rejected;
    an order at the price may be partly accepted. Tolerance: the five significant digits of the CSV output."""
    price = run.prices[market_id].rename("clearing")
    # an order of no volume (the learning strategies write some) is on neither side
    book = run.orders[
        (run.orders.market_id == market_id) & (run.orders.volume != 0)
    ].join(price, on="start_time")
    tol = 1e-4 * book.clearing.abs() + 0.011
    share = book.accepted_volume / book.volume
    sale = book.volume > 0
    gap = np.where(
        sale, book.clearing - book.price, book.price - book.clearing
    )  # > 0: in the money
    wrong = np.where(
        gap > tol,
        (1 - share).clip(lower=0) > 1e-4,
        np.where(gap < -tol, share > 1e-4, False),
    )
    size = np.where(wrong, np.abs(gap), 0.0)
    per_period = pd.Series(size, index=book.start_time).groupby(level=0).max()
    return {
        "periods_violated": int((per_period > 0).sum()),
        "max_violation": float(per_period.max() if len(per_period) else 0.0),
    }


def _flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, key + "."))
        elif not isinstance(v, (list, tuple)):
            flat[key] = v
    return flat


def write_numbers(nums: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "numbers.json").write_text(
        json.dumps(nums, indent=1, default=float), encoding="utf-8"
    )
    pd.Series(_flatten(nums)).rename("value").rename_axis("number").to_csv(
        out / "numbers.csv"
    )


# ================================================================== 00 summary
def fig_summary(run: Run, nums: dict, out: Path):
    """The headline numbers as one table, written as Markdown (``00_summary.md``), not as a figure."""
    rows: list[tuple[str, str, str]] = []
    if "fit_hourly" in nums:
        fit, ps, po = (
            nums["fit_hourly"],
            nums["price_simulated_hourly"],
            nums["price_observed_hourly"],
        )
        rows += [
            (
                "Hourly day-ahead price, £ per MWh",
                "Simulated",
                f"Observed ({OBSERVED_NAME})",
            )
        ]
        for key, label in (
            ("mean", "Mean"),
            ("std", "Standard deviation"),
            ("p05", "5th percentile"),
            ("p50", "Median"),
            ("p95", "95th percentile"),
            ("max", "Highest"),
            ("min", "Lowest"),
            ("daily_range", "Mean daily range"),
        ):
            rows.append((label, f"{ps[key]:,.1f}", f"{po[key]:,.1f}"))
        rows.append(
            (
                "Hours below zero",
                pct(ps["negative_share"], 1),
                pct(po["negative_share"], 1),
            )
        )
        above = fit["hours_above_observed_p99"]
        rows.append(
            (
                f"Hours above {gbp(fit['observed_p99'])} (observed 99th pct.)",
                f"{above['simulated']:,}",
                f"{above['observed']:,}",
            )
        )
        rows += [("Fit of hourly prices", "", "")]
        rows += [
            ("R²", "", f"{fit['r2']:.3f}"),
            ("Correlation", "", f"{fit['correlation']:.3f}"),
            ("Bias, £ per MWh", "", f"{fit['bias']:.2f}"),
            ("Mean absolute error, £ per MWh", "", f"{fit['mae']:.2f}"),
            ("Root mean square error, £ per MWh", "", f"{fit['rmse']:.2f}"),
            (
                "Standard deviation, simulated over observed",
                "",
                f"{fit['std_ratio']:.2f}",
            ),
            (
                "Slope of observed on simulated",
                "",
                f"{fit['slope_observed_on_simulated']:.2f}",
            ),
        ]
    rows += [("Market", "", "")]
    share = nums["share_of_supply"]
    setter = nums["price_setter_share"]
    rows += [
        (
            "Demand served (net of nuclear and pumped storage), TWh",
            "",
            f"{nums['demand_served_twh']:,.1f}",
        ),
        (
            "Cost of that demand at the day-ahead price, £ billion",
            "",
            f"{nums['consumer_cost_gbp_bn']:,.2f}",
        ),
        (
            "Imports / exports, TWh",
            "",
            f"{nums['imports_twh']:,.1f} / {nums['exports_twh']:,.1f}",
        ),
        (
            "Share of the energy sold: gas / wind",
            "",
            f"{pct(share.get('Gas', 0))} / {pct(share.get('Wind', 0))}",
        ),
        (
            "Price set by gas / interconnectors",
            "",
            f"{pct(setter.get('Gas', 0))} / {pct(setter.get(INTERCONNECTORS, 0))}",
        ),
    ]
    cap = nums["capture"]
    for g in ("Wind", "Solar"):
        if g in cap:
            rows.append(
                (
                    f"{g}: price earned, £ per MWh / value factor",
                    "",
                    f"{cap[g]['capture_price']:.1f} / {cap[g]['value_factor']:.2f}",
                )
            )
    if "storage" in nums:
        st = nums["storage"]
        rows.append(
            (
                "Storage: full cycles / revenue net of charging, £ million",
                "",
                f"{st['cycles']:.0f} / {st['net_revenue_gbp_m']:,.1f}",
            )
        )
    for m, check in nums["merit_order_check"].items():
        rows.append(
            (
                f"{m}: half-hours with an order on the wrong side of the price",
                "",
                f"{check['periods_violated']:,} (largest £{check['max_violation']:.2f})",
            )
        )

    lines = [
        f"# {run.scenario}, {run.case}: headline numbers",
        "",
        f"{run.index[0]:%d %b %Y} to {run.index[-1]:%d %b %Y}; {nums['periods_left_out']} half-hours with "
        "flagged inputs left out of the comparison.",
        "",
    ]
    columns = ("Simulated", f"Observed ({OBSERVED_NAME})")
    for label, a, b in rows:
        if (a, b) == columns or (a, b) == ("", ""):
            # a new table for each section
            lines += ([""] if lines[-1] else []) + [
                f"| {label} | {a} | {b} |",
                "| --- | ---: | ---: |",
            ]
        else:
            lines.append(f"| {label} | {a} | {b} |")
    lines += ["", source(run), ""]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "00_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ================================================================== 01-06 price: does the model reproduce the market?
def fig_price_year(run: Run, nums: dict, out: Path):
    """The year in daily means, simulated against observed, with the monthly mean error below."""
    if not has_observed(run):
        return None
    fit = nums["fit_hourly"]
    sim, obs = (
        daily(run.day_ahead.where(run.admissible)),
        daily(run.observed[OBSERVED].where(run.admissible)),
    )
    pair = hourly_pair(run)
    monthly = (pair.simulated - pair.observed).groupby(pair.index.to_period("M")).mean()
    fig, top, bottom = S.figure(
        f"The simulated day-ahead price follows the observed one with R² {fit['r2']:.2f} on hourly prices and is "
        f"{gbp(abs(fit['bias']), 1)} per MWh {lower_higher(fit['bias'])} on average",
        f"Daily mean day-ahead price, simulated and observed ({OBSERVED_NAME}), £ per MWh; below, the monthly mean "
        f"difference (simulated less observed). Mean absolute error of hourly prices {gbp(fit['mae'], 1)}, "
        f"correlation {fit['correlation']:.2f}.",
        source(run),
    )
    ax, bx = S.stacked(fig, top - 0.04, bottom, ratios=(3, 1.1), hspace=0.05)
    ax.plot(obs.index, obs.values, color=S.OBSERVED, lw=1.0)
    ax.plot(sim.index, sim.values, color=S.SIMULATED, lw=1.4)
    ax.set_ylabel("£ per MWh")
    price_lines(ax)
    x = monthly.index.to_timestamp() + pd.Timedelta(days=14)
    bx.bar(
        x,
        monthly.values,
        width=20,
        color=[S.NAVY if v < 0 else S.WINE for v in monthly.values],
    )
    bx.axhline(0, color=S.LIGHT, lw=0.8)
    bx.set_ylabel("Difference")
    lim = max(abs(monthly).max() * 1.2, 1)
    bx.set_ylim(-lim, lim)
    date_axis(bx)
    ax.set_xlim(run.index[0], run.index[-1])
    return S.save(fig, out, "01_price_year")


def fig_duration(run: Run, nums: dict, out: Path):
    """Price duration curves on hourly means, the tails stretched (logit scale) as in Ward et al. (2019)."""
    if not has_observed(run):
        return None
    pair = hourly_pair(run)
    curves = {c: np.sort(pair[c].values)[::-1] for c in ("simulated", "observed")}
    n = len(pair)
    share = (np.arange(n) + 0.5) / n
    k = max(int(0.05 * n), 1)
    top_s, top_o = curves["simulated"][:k].mean(), curves["observed"][:k].mean()
    low_s, low_o = curves["simulated"][-k:].mean(), curves["observed"][-k:].mean()
    ps, po = nums["price_simulated_hourly"], nums["price_observed_hourly"]
    fig, top, bottom = S.figure(
        f"The dearest 5% of hours average {gbp(top_s)} per MWh simulated against {gbp(top_o)} observed, the "
        f"cheapest 5% {gbp(low_s)} against {gbp(low_o)}",
        f"Hourly day-ahead prices sorted from highest to lowest, simulated and observed ({OBSERVED_NAME}), £ per MWh; "
        f"the horizontal axis stretches both ends. Below zero in {pct(ps['negative_share'], 1)} of hours simulated, "
        f"{pct(po['negative_share'], 1)} observed; highest {gbp(ps['max'])} and {gbp(po['max'])}.",
        source(run),
    )
    ax = S.axes(fig, top - 0.04, bottom)
    ax.plot(share, curves["observed"], color=S.OBSERVED, lw=1.2)
    ax.plot(share, curves["simulated"], color=S.SIMULATED, lw=1.6)
    ax.set_xscale("logit")
    ticks = [0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99, 0.999]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{100 * t:g}%" for t in ticks])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(0.5 / n, 1 - 0.5 / n)
    ax.axhline(0, color=S.LIGHT, lw=0.8)
    ax.set_xlabel("Share of hours with a higher price")
    ax.set_ylabel("£ per MWh")
    price_lines(ax)
    return S.save(fig, out, "02_price_duration")


def fig_daily_shape(run: Run, nums: dict, out: Path):
    """Price by hour of the day in each quarter: mean, and the 10th and 90th percentiles."""
    if not has_observed(run):
        return None
    pair = hourly_pair(run)
    key = [pair.index.quarter, pair.index.hour]
    mean, p10, p90 = (
        pair.groupby(key).mean(),
        pair.groupby(key).quantile(0.1),
        pair.groupby(key).quantile(0.9),
    )
    quarters = [q for q in QUARTERS if q in mean.index.get_level_values(0)]
    err = (pair.simulated - pair.observed).groupby(pair.index.hour).mean()
    peak = err.idxmin() if abs(err.min()) >= abs(err.max()) else err.idxmax()
    fig, top, bottom = S.figure(
        f"The simulated price differs most from the observed one at {peak:02d}:00 UTC, by {gbp(err[peak], 1)} per "
        f"MWh over the year; in the median hour of the day the gap is {gbp(err.drop(peak).abs().median(), 1)}",
        f"Day-ahead price by hour of the day (UTC) in each quarter, simulated and observed ({OBSERVED_NAME}), "
        f"£ per MWh: mean (thick); 10th and 90th percentile of the hours (thin, simulated; shaded, observed)",
        source(run),
    )
    n = len(quarters)
    width = (0.97 - 0.08 - 0.025 * (n - 1)) / n
    axs = []
    for i, q in enumerate(quarters):
        ax = fig.add_axes(
            [0.08 + i * (width + 0.025), bottom, width, top - 0.04 - bottom],
            sharey=axs[0] if axs else None,
        )
        h = mean.loc[q].index
        ax.fill_between(
            h, p10.loc[q].observed, p90.loc[q].observed, color=S.FAINT, lw=0
        )
        ax.plot(h, mean.loc[q].observed, color=S.OBSERVED, lw=1.2)
        ax.plot(h, mean.loc[q].simulated, color=S.SIMULATED, lw=1.8)
        ax.plot(h, p10.loc[q].simulated, color=S.SIMULATED, lw=0.7)
        ax.plot(h, p90.loc[q].simulated, color=S.SIMULATED, lw=0.7)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlim(0, 23)
        ax.set_xlabel("Hour (UTC)")
        ax.text(
            0.03,
            0.97,
            QUARTERS[q],
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color=S.BODY,
            va="top",
        )
        if i:
            ax.tick_params(labelleft=False)
        axs.append(ax)
    axs[0].set_ylabel("£ per MWh")
    legend_above(
        axs[0],
        line_handles([(S.OBSERVED, 1.2, "-"), (S.SIMULATED, 1.8, "-")])
        + [Patch(color=S.FAINT)],
        [
            f"Observed ({OBSERVED_NAME})",
            "Simulated",
            "Observed 10th to 90th percentile",
        ],
    )
    return S.save(fig, out, "03_daily_shape")


def fig_fit(run: Run, nums: dict, out: Path):
    """Hour by hour (density) and month by month (mean and percentiles, as Ward et al. 2019)."""
    if not has_observed(run):
        return None
    pair = hourly_pair(run)
    fit = nums["fit_hourly"]
    err = pair.simulated - pair.observed
    month = pair.index.to_period("M")
    stats = {
        "Mean": pair.groupby(month).mean(),
        "10th percentile": pair.groupby(month).quantile(0.1),
        "90th percentile": pair.groupby(month).quantile(0.9),
    }
    rmse = {
        k: float(np.sqrt(((v.simulated - v.observed) ** 2).mean()))
        for k, v in stats.items()
    }
    worst_key = max(rmse, key=rmse.get)
    worst = worst_key.lower() if worst_key != "Mean" else "mean"
    fig, top, bottom = S.figure(
        f"In {pct((err.abs() <= 10).mean())} of hours the simulated price is within £10 per MWh of the observed one; "
        f"month by month the {worst} is missed most (RMSE {gbp(rmse[worst_key], 1)})",
        f"Left: hourly day-ahead price, simulated against observed ({OBSERVED_NAME}), £ per MWh; shade = number of "
        f"hours, line = equal prices; R² {fit['r2']:.2f}, mean absolute error {gbp(fit['mae'], 1)}, root mean square "
        f"error {gbp(fit['rmse'], 1)}, bias {gbp(fit['bias'], 1)} per MWh. Right: the mean and the 10th and 90th "
        f"percentile of the hourly prices of each month, simulated against observed; root mean square error "
        + ", ".join(f"{k.lower()} {gbp(v, 1)}" for k, v in rmse.items())
        + ".",
        source(run),
    )
    a, b = S.two(fig, top, bottom, sharey=False, gap=0.13)
    lo, hi = np.nanpercentile(pair.values, [0.2, 99.8])
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    hb = a.hexbin(
        pair.observed,
        pair.simulated,
        gridsize=60,
        extent=(lo, hi, lo, hi),
        cmap=SEQUENTIAL,
        mincnt=1,
        bins="log",
        linewidths=0,
    )
    a.plot([lo, hi], [lo, hi], color=S.INK, lw=0.8)
    a.set(
        xlim=(lo, hi),
        ylim=(lo, hi),
        xlabel="Observed, £ per MWh",
        ylabel="Simulated, £ per MWh",
    )
    a.grid(False)
    cb = colourbar(fig, hb, ax=a, fraction=0.04, pad=0.02)
    cb.ax.set_title("Hours", fontsize=9, color=S.BODY, loc="left")
    cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    styles = {
        "Mean": (S.NAVY, "o"),
        "10th percentile": (S.SKY, "v"),
        "90th percentile": (S.WINE, "^"),
    }
    allv = pd.concat([v.stack() for v in stats.values()])
    lo2, hi2 = allv.min(), allv.max()
    pad2 = 0.06 * (hi2 - lo2)
    for name, v in stats.items():
        colour, marker = styles[name]
        b.scatter(
            v.observed,
            v.simulated,
            s=36,
            color=colour,
            marker=marker,
            edgecolor="white",
            lw=0.8,
            label=name,
            zorder=3,
        )
    b.plot([lo2 - pad2, hi2 + pad2], [lo2 - pad2, hi2 + pad2], color=S.INK, lw=0.8)
    b.set(
        xlim=(lo2 - pad2, hi2 + pad2),
        ylim=(lo2 - pad2, hi2 + pad2),
        xlabel="Observed, £ per MWh",
        ylabel="Simulated, £ per MWh",
    )
    b.legend(loc="upper left", fontsize=9.5, handletextpad=0.3)
    b.grid(True, axis="both")
    return S.save(fig, out, "04_fit")


def fig_heatmap(run: Run, nums: dict, out: Path):
    """Day x hour map of the error: where in the year and the day the model misses."""
    if not has_observed(run):
        return None
    pair = hourly_pair(run)
    err = (pair.simulated - pair.observed).rename("e").to_frame()
    err["day"], err["hour"] = err.index.normalize(), err.index.hour
    grid = err.pivot_table(index="hour", columns="day", values="e").reindex(range(24))
    grid = grid.reindex(
        columns=pd.date_range(grid.columns[0], grid.columns[-1], freq="D")
    )
    lim = float(np.nanpercentile(np.abs(grid.values), 98))
    by_q = err.e.groupby(err.index.quarter).mean()
    worst_q = by_q.abs().idxmax()
    daily_abs = err.e.abs().groupby(err.day).mean().sort_values(ascending=False)
    worst_days = daily_abs.head(max(int(0.1 * len(daily_abs)), 1))
    share_worst = worst_days.sum() / daily_abs.sum()
    fig, top, bottom = S.figure(
        f"A tenth of the days carry {pct(share_worst)} of the absolute error; by quarter the mean error is largest "
        f"in {QUARTERS[worst_q]} ({gbp(by_q[worst_q], 1)} per MWh)",
        f"Hourly difference between the simulated and the observed ({OBSERVED_NAME}) day-ahead price, £ per MWh, "
        f"by day (across) and hour of the day (up, UTC); wine = simulated too high, navy = too low, blank = no data",
        source(run),
    )
    ax = S.axes(fig, top, bottom, right=0.9)
    days = mdates.date2num(grid.columns.to_pydatetime())
    im = ax.imshow(
        grid.values,
        aspect="auto",
        origin="lower",
        cmap=DIVERGING,
        norm=TwoSlopeNorm(0, -lim, lim),
        extent=(days[0], days[-1] + 1, 0, 24),
        interpolation="nearest",
    )
    ax.xaxis_date()
    date_axis(ax)
    ax.set_yticks([0, 6, 12, 18, 24])
    ax.set_ylabel("Hour of the day (UTC)")
    ax.grid(False)
    cax = fig.add_axes([0.92, bottom, 0.015, top - bottom])
    colourbar(fig, im, cax=cax, extend="both", label="£ per MWh")
    return S.save(fig, out, "05_error_map")


def _slope(x: pd.Series, y: pd.Series) -> float:
    lo, hi = x.quantile([0.1, 0.9])
    keep = (x >= lo) & (x <= hi)
    return float(np.polyfit(x[keep], y[keep], 1)[0])


def fig_residual_load(run: Run, nums: dict, out: Path):
    """Price against demand net of wind and solar: the supply curve each market reveals (Ward et al. 2019)."""
    if not has_observed(run):
        return None
    pair = hourly_pair(run)
    rl = compare.hourly(run.residual_load, run.admissible).reindex(pair.index) / 1e3
    slope = {c: _slope(rl, pair[c]) for c in ("simulated", "observed")}
    bins = np.arange(np.floor(rl.min()), np.ceil(rl.max()) + 1, 1.0)
    centre = (bins[:-1] + bins[1:]) / 2
    counts = rl.groupby(pd.cut(rl, bins), observed=False).size().values
    medians = {
        c: np.where(
            counts >= 10,
            pair[c].groupby(pd.cut(rl, bins), observed=False).median().values,
            np.nan,
        )
        for c in ("simulated", "observed")
    }
    fig, top, bottom = S.figure(
        f"Each GW of demand left after wind and solar adds {gbp(slope['simulated'], 1)} per MWh to the simulated "
        f"price and {gbp(slope['observed'], 1)} to the observed one",
        "Hourly day-ahead price against the demand the market serves less wind and solar output, GW (that demand is "
        "net of nuclear and pumped storage, which are not units); lines = median in bins of 1 GW with at least ten "
        "hours, both in each panel. Slopes fitted between the 10th and 90th percentile of that demand.",
        source(run),
    )
    a, b = S.two(fig, top - 0.03, bottom, sharey=True)
    lo, hi = np.nanpercentile(pair.values, [0.2, 99.8])
    for ax, col, colour, title in (
        (a, "simulated", S.SIMULATED, "Simulated"),
        (b, "observed", S.OBSERVED, f"Observed ({OBSERVED_NAME})"),
    ):
        ax.scatter(rl, pair[col], s=3, color=colour, alpha=0.10, lw=0, rasterized=True)
        other = "observed" if col == "simulated" else "simulated"
        ax.plot(
            centre,
            medians[other],
            color=S.SIMULATED if other == "simulated" else S.OBSERVED,
            lw=1.0,
        )
        ax.plot(centre, medians[col], color=colour, lw=2.2)
        ax.set_xlabel("Demand less wind and solar, GW")
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
        S.panel_label(ax, title)
    a.set_ylabel("£ per MWh")
    b.tick_params(labelleft=False)
    return S.save(fig, out, "06_price_vs_residual_load")


# ================================================================== 07-11 what the market did
def fig_generation(run: Run, nums: dict, out: Path):
    """Energy by group of plant per month, exports below zero."""
    gen = run.generation
    monthly = gen.groupby(gen.index.to_period("M")).sum() * HOURS / 1e6
    groups = present_groups(monthly)
    share = nums["share_of_supply"]
    ranked = sorted(((g, s) for g, s in share.items() if s > 0), key=lambda kv: -kv[1])[
        :3
    ]
    fig, top, bottom = S.figure(
        f"{ranked[0][0]} supplied {pct(ranked[0][1])} of the energy sold in the simulated market, "
        f"{ranked[1][0].lower()} {pct(ranked[1][1])} and {ranked[2][0].lower()} {pct(ranked[2][1])}; GB imported "
        f"{nums['imports_twh']:.1f} TWh and exported {nums['exports_twh']:.1f} TWh",
        f"Energy sold by each group of plant per month, TWh; exports below zero. The market serves "
        f"{nums['demand_served_twh']:.0f} TWh of GB demand net of nuclear and pumped storage, which are not units "
        f"(their output is taken off the demand).",
        source(run, observed=False),
    )
    ax = S.axes(fig, top - 0.05, bottom)
    x = np.arange(len(monthly))
    up, down = np.zeros(len(x)), np.zeros(len(x))
    for g in groups:
        v = monthly[g].values
        pos, neg = np.clip(v, 0, None), np.clip(v, None, 0)
        if pos.any():
            ax.bar(x, pos, 0.72, bottom=up, color=COLOURS[g], edgecolor="white", lw=0.6)
            up += pos
        if neg.any():
            ax.bar(
                x, neg, 0.72, bottom=down, color=COLOURS[g], edgecolor="white", lw=0.6
            )
            down += neg
    ax.axhline(0, color=S.BODY, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([p.strftime("%b") for p in monthly.index])
    ax.set_ylabel("TWh")
    legend_above(
        ax, patch_handles(groups), groups, ncol=min(len(groups), 6), fontsize=9.5
    )
    return S.save(fig, out, "07_generation_mix")


def _week(run: Run, which: str) -> pd.Timestamp:
    rl = run.residual_load.resample("D").mean()
    week = rl.rolling(7).mean().shift(-6).dropna()
    week = week[
        week.index + pd.Timedelta(days=7) <= run.index[-1] + pd.Timedelta(minutes=30)
    ]
    return week.idxmax() if which == "high" else week.idxmin()


def fig_weeks(run: Run, nums: dict, out: Path):
    """The week with the most and the week with the least demand left after wind and solar: price above,
    the stack of what was sold below (Electric Insights layout, no second axis)."""
    paths = []
    for which, reason in (
        ("high", "most demand left after wind and solar"),
        ("low", "least demand left after wind and solar"),
    ):
        start = _week(run, which)
        sl = slice(start, start + pd.Timedelta(days=7) - pd.Timedelta(minutes=30))
        gen = run.generation.loc[sl]
        groups = present_groups(gen)
        gen = gen[groups]
        demand = run.gb_demand.loc[sl]
        sim = run.day_ahead.loc[sl]
        wind = (
            gen.reindex(columns=["Wind"], fill_value=0).sum(axis=1).sum() / demand.sum()
        )
        title = f"Week from {start:%d %B}, {reason}: wind met {pct(wind)} of demand; the simulated price averaged {gbp(sim.mean())} per MWh"
        if has_observed(run):
            obs = run.observed[OBSERVED].loc[sl]
            title += f" against {gbp(obs.mean())} observed"
        fig, top, bottom = S.figure(
            title,
            "Above: day-ahead price, £ per MWh. Below: what each group of plant sold, GW, stacked; exports and "
            "storage charging below zero; line = GB demand the market serves (net of nuclear and pumped storage).",
            source(run),
        )
        ax, bx = S.stacked(fig, top - 0.05, bottom, ratios=(1, 2.2), hspace=0.05)
        if has_observed(run):
            ax.step(obs.index, obs.values, where="post", color=S.OBSERVED, lw=1.0)
        ax.step(sim.index, sim.values, where="post", color=S.SIMULATED, lw=1.4)
        ax.set_ylabel("£ per MWh")
        stack_area(bx, gen)
        bx.step(demand.index, demand.values / 1e3, where="post", color=S.INK, lw=1.0)
        bx.axhline(0, color=S.BODY, lw=0.6)
        bx.set_ylabel("GW")
        date_axis(bx, monthly=False)
        bx.set_xlim(sl.start, sl.stop + pd.Timedelta(minutes=30))
        handles = line_handles(
            [(S.OBSERVED, 1.0, "-"), (S.SIMULATED, 1.4, "-")]
        ) + patch_handles(groups)
        labels = [f"Observed ({OBSERVED_NAME})", "Simulated"] + groups
        if not has_observed(run):
            handles, labels = handles[1:], labels[1:]
        legend_above(ax, handles, labels, ncol=min(len(labels), 7), fontsize=9)
        paths.append(S.save(fig, out, f"08_week_{which}"))
    return paths


def fig_price_setter(run: Run, nums: dict, out: Path):
    """Which group of plant sets the price, by month and by hour of the day (Blume-Werry et al. 2021)."""
    setter = run.marginal(convert.MARKET_ID).group.replace(
        {"Imports": INTERCONNECTORS, EXPORTS: INTERCONNECTORS}
    )
    order = [
        g for g in [*GROUPS, INTERCONNECTORS, "Demand"] if g in set(setter.dropna())
    ]
    order = [g for g in order if g not in ("Imports",)]
    by_month = pd.crosstab(
        setter.index.to_period("M"), setter, normalize="index"
    ).reindex(columns=order, fill_value=0)
    by_hour = pd.crosstab(setter.index.hour, setter, normalize="index").reindex(
        columns=order, fill_value=0
    )
    share = nums["price_setter_share"]
    ranked = sorted(share.items(), key=lambda kv: -kv[1])
    second = (
        f", {ranked[1][0].lower()} in {pct(ranked[1][1])}" if len(ranked) > 1 else ""
    )
    fig, top, bottom = S.figure(
        f"{ranked[0][0]} set the simulated day-ahead price in {pct(ranked[0][1])} of half-hours{second}",
        "Share of half-hours in which each group of plant offers at the clearing price (the price-setting offer), "
        "by month and by hour of the day (UTC). An interconnector sets the price at the neighbouring market's price "
        "net of losses, on the import or the export side.",
        source(run, observed=False),
    )
    a, b = S.two(fig, top - 0.05, bottom, sharey=True, gap=0.06)
    for ax, table, labels in (
        (a, by_month, [p.strftime("%b") for p in by_month.index]),
        (b, by_hour, [f"{h}" for h in by_hour.index]),
    ):
        x = np.arange(len(table))
        base = np.zeros(len(x))
        for g in order:
            ax.bar(
                x,
                table[g].values * 100,
                0.8,
                bottom=base,
                color=COLOURS.get(g, S.LIGHT),
                edgecolor="white",
                lw=0.4,
            )
            base += table[g].values * 100
        ax.set_xticks(x if len(x) <= 12 else x[::3])
        ax.set_xticklabels(labels if len(x) <= 12 else labels[::3])
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.6, len(x) - 0.4)
    a.set_ylabel("Share of half-hours, %")
    b.set_xlabel("Hour of the day (UTC)")
    b.tick_params(labelleft=False)
    legend_above(a, patch_handles(order), order, ncol=min(len(order), 7), fontsize=9.5)
    return S.save(fig, out, "09_price_setter")


def _snapshots(run: Run) -> dict[str, pd.Timestamp]:
    price = run.day_ahead.dropna()
    median = (price - price.median()).abs().idxmin()
    return {"cheapest": price.idxmin(), "typical": median, "dearest": price.idxmax()}


def merit_order(
    ax, book: pd.DataFrame, price: float, observed: float | None, demand: float
):
    """Offers as blocks (width = MW, height = offer price) coloured by group, with the demand, the
    clearing point and the observed price."""
    sales = book[(book.volume > 0) & (book.unit_id != convert.UNSERVED_UNIT)]
    start = sales.cumulative - sales.volume
    for (_, row), x0 in zip(sales.iterrows(), start):
        ax.add_patch(
            Rectangle(
                (x0 / 1e3, 0),
                row.volume / 1e3,
                row.price,
                color=COLOURS.get(row.group, S.LIGHT),
                lw=0,
            )
        )
    ax.axvline(demand / 1e3, color=S.INK, lw=1.0)
    ax.plot(
        [demand / 1e3], [price], "o", color=S.INK, ms=6, mec="white", mew=1.2, zorder=5
    )
    if observed is not None and np.isfinite(observed):
        ax.axhline(observed, color=S.OBSERVED, lw=0.9, ls=(0, (3, 2)))
    ax.axhline(0, color=S.BODY, lw=0.6)
    ax.set_xlim(0, max(demand * 1.25, 1) / 1e3)
    return sales


def fig_supply_curves(run: Run, nums: dict, out: Path):
    """The offers of three half-hours (the cheapest, a typical and the dearest) as merit orders
    (Ward et al. 2019, Fig. 4; the exchanges' aggregated curves)."""
    snaps = _snapshots(run)
    marginal = run.marginal(convert.MARKET_ID)
    setters = {k: marginal.loc[t, "group"] for k, t in snaps.items()}
    fig, top, bottom = S.figure(
        f"{setters['typical']} set the price in a typical half-hour ({gbp(run.day_ahead[snaps['typical']])} per MWh), "
        f"{setters['dearest'].lower()} in the dearest ({gbp(run.day_ahead[snaps['dearest']])}) and "
        f"{setters['cheapest'].lower()} in the cheapest ({gbp(run.day_ahead[snaps['cheapest']])})",
        "Day-ahead offers of three half-hours sorted by price, from left to right "
        + ", ".join(f"the {k} ({t:%d %b %H:%M} UTC)" for k, t in snaps.items())
        + ": width = MW offered, height = offer price, £ per MWh; vertical line = demand (inelastic), dot = clearing "
        "price, dashed = observed price of that hour. Offers below zero are plants with a support payment per MWh.",
        source(run),
    )
    n = len(snaps)
    gap = 0.07  # each panel has its own price axis: room for its tick labels
    width = (0.97 - 0.08 - gap * (n - 1)) / n
    axs, seen = [], []
    for i, (name, t) in enumerate(snaps.items()):
        ax = fig.add_axes(
            [0.08 + i * (width + gap), bottom, width, top - 0.06 - bottom]
        )
        book = run.book(t)
        demand = -book.loc[book.unit_id == convert.DEMAND_UNIT, "volume"].sum()
        obs = run.observed[OBSERVED].get(t) if has_observed(run) else None
        sales = merit_order(ax, book, run.day_ahead[t], obs, demand)
        seen += [g for g in sales.group.unique() if g not in seen]
        visible = sales[sales.cumulative - sales.volume < demand * 1.25]
        ax.set_ylim(
            max(visible.price.min(), -320) * 1.05 - 10,
            max(run.day_ahead[t] * 1.8, visible.price.quantile(0.9), 60),
        )
        ax.set_xlabel("GW offered")
        ax.text(
            0.03,
            0.97,
            name.capitalize(),
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color=S.BODY,
            va="top",
        )
        axs.append(ax)
    axs[0].set_ylabel("£ per MWh")
    groups = [g for g in [*GROUPS, EXPORTS] if g in seen]
    legend_above(
        axs[0], patch_handles(groups), groups, ncol=min(len(groups), 6), fontsize=9.5
    )
    return S.save(fig, out, "10_supply_curves")


def fig_capture(run: Run, nums: dict, out: Path):
    """The price each group earned per MWh against the average price (value factor; Hirth 2013), and its revenue."""
    cap = run.capture_prices()
    cap = cap.drop(index=[g for g in ("Unserved",) if g in cap.index])
    base = run.day_ahead.mean()
    parts = [
        f"{g.lower()} {gbp(cap.loc[g, 'capture_price'])} ({cap.loc[g, 'value_factor']:.2f})"
        for g in ("Solar", "Gas")
        if g in cap.index
    ]
    lead = (
        (
            f"Wind earned {gbp(cap.loc['Wind', 'capture_price'])} per MWh, {cap.loc['Wind', 'value_factor']:.2f} times the average price of {gbp(base)}"
        )
        if "Wind" in cap.index
        else f"The average price was {gbp(base)}"
    )
    fig, top, bottom = S.figure(
        lead + ("; " + ", ".join(parts) if parts else ""),
        f"Left: market revenue per MWh sold by each group of plant, £ per MWh; line = time-weighted mean day-ahead "
        f"price ({gbp(base, 1)}). Value factor (the ratio of the two): "
        + ", ".join(f"{g.lower()} {vf:.2f}" for g, vf in cap.value_factor.items())
        + ". Right: market revenue, £ million. Support payments are not included; storage = revenue net of the cost "
        "of charging, per MWh discharged.",
        source(run, observed=False),
    )
    a, b = S.two(fig, top, bottom, sharey=False, gap=0.08)
    x = np.arange(len(cap))
    colours = [COLOURS[g] for g in cap.index]
    a.bar(x, cap.capture_price, 0.7, color=colours)
    a.axhline(base, color=S.INK, lw=1.0)
    a.set_ylabel("£ per MWh")
    b.bar(x, cap.revenue_gbp_m, 0.7, color=colours)
    b.set_ylabel("£ million")
    for ax in (a, b):
        ax.set_xticks(x)
        ax.set_xticklabels(
            [g.replace(" and ", "\nand ") for g in cap.index], fontsize=9, rotation=0
        )
    return S.save(fig, out, "11_capture_prices")


# ================================================================== 12-13 by case
def storage_numbers(run: Run) -> dict:
    unit = convert.STORAGE_UNIT
    pos = run.position[unit]
    spec = run.units.loc[unit]
    discharged = pos.clip(lower=0).sum() * HOURS
    revenue = float(run.revenue()[unit].sum())
    no_storage = run.reference.get("model_no_storage")
    out = {
        "power_mw": float(spec.max_power),
        "energy_mwh": float(spec.capacity),
        "discharged_mwh": discharged,
        "charged_mwh": -pos.clip(upper=0).sum() * HOURS,
        "cycles": discharged / float(spec.capacity),
        "net_revenue_gbp_m": revenue / 1e6,
        "revenue_per_mw_gbp_k": revenue / float(spec.max_power) / 1e3,
        "daily_range_with": price_stats(run.day_ahead)["daily_range"],
    }
    if no_storage is not None:
        out["daily_range_without"] = price_stats(no_storage)["daily_range"]
    return out


def fig_storage(run: Run, nums: dict, out: Path):
    """When the battery fleet charges and discharges, and what that does to the daily price shape."""
    if "storage" not in nums:
        return None
    st = nums["storage"]
    pos = run.position[convert.STORAGE_UNIT]
    hour = run.index.hour + run.index.minute / 60
    profile = pos.groupby(hour).mean()
    with_ = run.day_ahead.groupby(hour).mean()
    without = (
        run.reference["model_no_storage"].groupby(hour).mean()
        if "model_no_storage" in run.reference
        else None
    )
    monthly = (
        run.revenue()[convert.STORAGE_UNIT].groupby(run.index.to_period("M")).sum()
        / 1e6
    )
    narrow = ""
    if "daily_range_without" in st:
        narrow = f"; it narrows the mean daily price range from {gbp(st['daily_range_without'])} to {gbp(st['daily_range_with'])}"
    fig, top, bottom = S.figure(
        f"The battery fleet ({st['power_mw'] / 1e3:.1f} GW, {st['energy_mwh'] / 1e3:.1f} GWh) makes {st['cycles']:.0f} "
        f"full cycles and earns £{st['net_revenue_gbp_m']:,.0f} million net of charging{narrow}",
        "Left: mean day-ahead price by time of day with the batteries (simulated) and without them (the merit order "
        "without storage), £ per MWh; below, mean battery output, MW (charging below zero). Right: revenue net of "
        "charging per month, £ million.",
        source(run, observed=False),
    )
    ax, bx = S.stacked(
        fig, top, bottom, ratios=(1.2, 1), hspace=0.05, left=0.08, right=0.58
    )
    if without is not None:
        ax.plot(without.index, without.values, color=S.MODEL, lw=1.2, ls=(0, (3, 2)))
    ax.plot(with_.index, with_.values, color=S.SIMULATED, lw=1.6)
    ax.set_ylabel("£ per MWh")
    legend_above(
        ax,
        line_handles([(S.SIMULATED, 1.6, "-"), (S.MODEL, 1.2, (0, (3, 2)))]),
        ["With batteries", "Without"],
    )
    bx.bar(
        profile.index,
        profile.values,
        0.45,
        align="edge",
        color=[S.WINE if v >= 0 else "#C68BA8" for v in profile],
    )
    bx.axhline(0, color=S.BODY, lw=0.6)
    bx.set_ylabel("MW")
    bx.set_xlabel("Time of day (UTC)")
    bx.set_xticks([0, 6, 12, 18, 24])
    bx.set_xlim(0, 24)
    cx = fig.add_axes([0.68, bottom, 0.29, top - bottom])
    cx.bar(np.arange(len(monthly)), monthly.values, 0.7, color=S.WINE)
    cx.set_xticks(np.arange(len(monthly)))
    cx.set_xticklabels([p.strftime("%b")[0] for p in monthly.index])
    cx.set_ylabel("£ million")
    return S.save(fig, out, "12_storage")


INTRADAY_OBSERVED = {
    "apx_mid": ("Market index (APX)", S.OBSERVED),
    "system_price": ("System price", S.MUTED),
    "epex_ida1": ("EPEX IDA1", S.GREEN),
}


def fig_intraday(run: Run, nums: dict, out: Path):
    """What the forecast error is worth: the intraday less the day-ahead price against how long or short
    the system turned out, simulated and observed."""
    if convert.INTRADAY_MARKET_ID not in run.markets:
        return None
    imb = run.imbalance / 1e3
    spread = {"Simulated": run.prices[convert.INTRADAY_MARKET_ID] - run.day_ahead}
    if has_observed(run):
        for col, (name, _) in INTRADAY_OBSERVED.items():
            if col in run.observed and run.observed[col].notna().any():
                spread[name] = run.observed[col] - run.observed[OBSERVED]
    keep = run.admissible
    edges = np.array([-np.inf, -5, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 5, np.inf])
    labels = [
        "< -5",
        "-5 to -3",
        "-3 to -2",
        "-2 to -1",
        "-1 to -0.5",
        "±0.5",
        "0.5 to 1",
        "1 to 2",
        "2 to 3",
        "3 to 5",
        "> 5",
    ]
    cut = pd.cut(imb[keep], edges, labels=labels)
    means = pd.DataFrame(
        {k: v[keep].groupby(cut, observed=False).mean() for k, v in spread.items()}
    )
    counts = cut.value_counts().reindex(labels)
    long3, short3 = imb[keep] > 3, imb[keep] < -3
    sim = spread["Simulated"][keep]
    title = (
        f"Intraday less day-ahead price: {gbp(sim[long3].mean(), 1)} per MWh simulated when the system turns "
        f"out more than 3 GW long, {gbp(sim[short3].mean(), 1)} when short"
    )
    if "Market index (APX)" in spread:
        obs = spread["Market index (APX)"][keep]
        title += f"; observed market index {gbp(obs[long3].mean(), 1)} and {gbp(obs[short3].mean(), 1)}"
    fig, top, bottom = S.figure(
        title,
        "Mean price difference to the day-ahead auction, £ per MWh, by how much more wind and solar or less demand "
        "turned up than forecast at 09:00 the day before (GW; long > 0); simulated: intraday auction less day-ahead "
        f"auction; observed: each price less {OBSERVED_NAME}. Bars below: half-hours in each class.",
        source(run),
    )
    ax, bx = S.stacked(fig, top - 0.05, bottom, ratios=(3, 1), hspace=0.05)
    x = np.arange(len(labels))
    styles = {"Simulated": (S.SIMULATED, 2.0)} | {
        name: (c, 1.2) for name, c in INTRADAY_OBSERVED.values()
    }
    for name in means.columns:
        colour, lw = styles[name]
        ax.plot(x, means[name].values, color=colour, lw=lw, marker="o", ms=4)
    ax.axhline(0, color=S.BODY, lw=0.6)
    ax.set_ylabel("£ per MWh")
    legend_above(
        ax,
        line_handles([(styles[n][0], styles[n][1], "-") for n in means.columns]),
        list(means.columns),
    )
    bx.bar(x, counts.values, 0.7, color=S.LIGHT)
    bx.set_ylabel("Half-hours")
    bx.set_xticks(x)
    bx.set_xticklabels(labels, fontsize=9)
    bx.set_xlabel("System long (+) or short (-) against the day-ahead forecast, GW")
    return S.save(fig, out, "13_intraday")


# ================================================================== 14 against a baseline run
def fig_vs_baseline(run: Run, nums: dict, out: Path, base: Run | None = None):
    """The run against a baseline run of the same scenario (a counterfactual, or the competitive
    benchmark for strategic bidders): prices and revenue by group."""
    if base is None:
        return None
    diff = (run.day_ahead - base.day_ahead).dropna()
    rev = run.capture_prices().revenue_gbp_m.subtract(
        base.capture_prices().revenue_gbp_m, fill_value=0
    )
    rev = rev[rev.abs() > 0.05]
    rev = rev.reindex([g for g in [*GROUPS, EXPORTS] if g in rev.index])
    curves = {
        "This run": np.sort(run.day_ahead.dropna().values)[::-1],
        "Baseline": np.sort(base.day_ahead.dropna().values)[::-1],
    }
    cost = (
        nums["consumer_cost_gbp_bn"]
        - (base.day_ahead * base.gb_demand).sum() * HOURS / 1e9
    )
    fig, top, bottom = S.figure(
        f"Against the baseline ({base.case}), the day-ahead price is {gbp(abs(diff.mean()), 2)} per MWh "
        f"{lower_higher(diff.mean())} on average and the demand pays £{abs(cost) * 1e3:,.0f} million "
        f"{'less' if cost < 0 else 'more'}",
        f"Left: half-hourly day-ahead prices sorted from highest to lowest, this run ({run.case}) and the baseline "
        f"({base.case}), £ per MWh. Right: change in market revenue by group of plant, £ million ("
        + ", ".join(f"{g.lower()} {v:+,.0f}" for g, v in rev.items())
        + ").",
        source(run, observed=False),
    )
    a, b = S.two(fig, top - 0.04, bottom, sharey=False, gap=0.1)
    n = len(curves["This run"])
    share = (np.arange(n) + 0.5) / n
    a.plot(share, curves["Baseline"], color=S.MODEL, lw=1.2, ls=(0, (3, 2)))
    a.plot(share, curves["This run"], color=S.SIMULATED, lw=1.6)
    a.set_xscale("logit")
    ticks = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
    a.set_xticks(ticks)
    a.set_xticklabels([f"{100 * t:g}%" for t in ticks])
    a.xaxis.set_minor_locator(mticker.NullLocator())
    a.set_ylabel("£ per MWh")
    a.set_xlabel("Share of half-hours with a higher price")
    legend_above(
        a,
        line_handles([(S.SIMULATED, 1.6, "-"), (S.MODEL, 1.2, (0, (3, 2)))]),
        [f"This run ({run.case})", f"Baseline ({base.case})"],
    )
    y = np.arange(len(rev))
    b.barh(y, rev.values, 0.7, color=[COLOURS.get(g, S.LIGHT) for g in rev.index])
    b.set_yticks(y)
    b.set_yticklabels(rev.index)
    b.axvline(0, color=S.BODY, lw=0.6)
    b.grid(True, axis="x")
    b.grid(False, axis="y")
    b.set_xlabel("£ million")
    b.invert_yaxis()
    span = max(rev.max(), 0) - min(rev.min(), 0)
    b.set_xlim(min(rev.min(), 0) - 0.05 * span, max(rev.max(), 0) + 0.05 * span)
    return S.save(fig, out, "14_vs_baseline")


def offers(run: Run, market_id: str = convert.MARKET_ID) -> pd.DataFrame:
    """Each unit's offer price per period: the volume-weighted mean price of its offers to sell."""
    book = run.orders[(run.orders.market_id == market_id) & (run.orders.volume > 0)]
    book = book.assign(pv=book.price * book.volume)
    sums = book.groupby(["start_time", "unit_id"])[["pv", "volume"]].sum()
    return (sums.pv / sums.volume).unstack()


def fig_offers_vs_baseline(run: Run, nums: dict, out: Path, base: Run | None = None):
    """Mark-ups: each unit's offer against its own offer in the baseline run (for strategic or learning
    bidders against the competitive strategies), and the Lerner index of the price-setting offer."""
    if base is None:
        return None
    mine, theirs = offers(run), offers(base)
    units = [
        u
        for u in mine.columns.intersection(theirs.columns)
        if u not in (convert.UNSERVED_UNIT,)
    ]
    markup = (mine[units] - theirs[units].reindex(mine.index)).dropna(how="all")
    changed = [u for u in units if (markup[u].abs() > 0.01).mean() > 0.01]
    if not changed:
        return None
    ranked = markup[changed].abs().mean().sort_values(ascending=False).index[:12]
    mean_markup = markup[changed].stack().mean()
    setter = run.marginal(convert.MARKET_ID)
    base_offer = theirs.stack().rename("base")
    key = list(zip(setter.index, setter.unit_id))
    competitive = pd.Series(
        [base_offer.get(k, np.nan) for k in key], index=setter.index
    )
    lerner = ((run.day_ahead - competitive) / run.day_ahead).where(run.day_ahead > 1)
    side = "above" if mean_markup >= 0 else "below"
    fig, top, bottom = S.figure(
        f"{len(changed)} units offer {gbp(abs(mean_markup), 2)} per MWh {side} their offers in the baseline "
        f"({base.case}) on average; in the median half-hour the price is {pct(np.nanmedian(lerner), 1)} above the "
        f"price-setting unit's baseline offer",
        "Left: each unit's day-ahead offer less its offer in the baseline run, £ per MWh, per half-hour (box = middle "
        "half, line = median, whiskers = 5th to 95th percentile; the twelve units that differ most). Right: Lerner "
        "index of the price-setting unit, (price - its baseline offer) / price, share of half-hours.",
        source(run, observed=False),
    )
    a = fig.add_axes([0.25, bottom, 0.32, top - bottom])
    b = fig.add_axes([0.66, bottom, 0.31, top - bottom])
    data = [markup[u].dropna().values for u in ranked]
    bp = a.boxplot(
        data, vert=False, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.6
    )
    for patch, u in zip(bp["boxes"], ranked):
        patch.set(
            facecolor=COLOURS.get(run.units.group.get(u), S.LIGHT), edgecolor="white"
        )
    for part in ("medians",):
        for line in bp[part]:
            line.set(color=S.INK, lw=1.2)
    for part in ("whiskers", "caps"):
        for line in bp[part]:
            line.set(color=S.MUTED, lw=0.8)
    a.set_yticks(range(1, len(ranked) + 1))
    a.set_yticklabels(ranked, fontsize=8.5)
    a.invert_yaxis()
    a.axvline(0, color=S.BODY, lw=0.6)
    a.grid(True, axis="x")
    a.grid(False, axis="y")
    a.set_xlabel("Offer less baseline offer, £ per MWh")
    vals = lerner.dropna().round(6).clip(-0.5, 1)
    b.hist(
        vals,
        bins=30,
        range=(-0.5, 1.0),
        color=S.NAVY,
        weights=np.full(len(vals), 100 / max(len(vals), 1)),
    )
    b.set_xlim(-0.5, 1.0)
    b.axvline(0, color=S.BODY, lw=0.6)
    b.set_xlabel("Lerner index of the price-setting offer")
    b.set_ylabel("Share of half-hours, %")
    return S.save(fig, out, "15_offers_vs_baseline")


FIGURES = [
    fig_summary,
    fig_price_year,
    fig_duration,
    fig_daily_shape,
    fig_fit,
    fig_heatmap,
    fig_residual_load,
    fig_generation,
    fig_weeks,
    fig_price_setter,
    fig_supply_curves,
    fig_capture,
    fig_storage,
    fig_intraday,
]


# the number each figure's file name starts with, for ``--only``
NUMBERS = {
    "fig_summary": "00",
    "fig_price_year": "01",
    "fig_duration": "02",
    "fig_daily_shape": "03",
    "fig_fit": "04",
    "fig_heatmap": "05",
    "fig_residual_load": "06",
    "fig_generation": "07",
    "fig_weeks": "08",
    "fig_price_setter": "09",
    "fig_supply_curves": "10",
    "fig_capture": "11",
    "fig_storage": "12",
    "fig_intraday": "13",
    "fig_vs_baseline": "14",
    "fig_offers_vs_baseline": "15",
}


def make(
    run: Run, out: Path, only: list[str] | None = None, baseline: Run | None = None
) -> list[Path]:
    """Write every figure that applies to ``run`` into ``out``; a figure that fails is reported and
    skipped. ``only``: figure numbers ("01") or parts of the function names ("supply")."""
    nums = numbers(run)
    write_numbers(nums, out)
    written: list[Path] = []
    calls = [(fn, {}) for fn in FIGURES] + [
        (fn, {"base": baseline}) for fn in (fig_vs_baseline, fig_offers_vs_baseline)
    ]
    for fn, kw in calls:
        if only and not any(
            key == NUMBERS.get(fn.__name__) or key in fn.__name__ for key in only
        ):
            continue
        try:
            result = fn(run, nums, out, **kw)
        except Exception:  # one broken figure must not cost the others
            logger.error("figure %s failed:\n%s", fn.__name__, traceback.format_exc())
            continue
        if result is not None:
            written += result if isinstance(result, list) else [result]
    return written
