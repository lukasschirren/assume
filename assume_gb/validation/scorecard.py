# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""The scorecard: the eight criteria in one table, one row per metric and one column per model,
with each metric's direction and every model's skill against the naive forecast.

The headline window is scored in full. Any other window (the calibration window, the full year)
is scored without what belongs to the headline window only, the Diebold-Mariano tests, rMAE and
the skill against the naive forecast, and without the naive forecast and LEAR; it is labelled
in-sample where it overlaps the calibration window.

    card = scorecard(data)                        the headline window
    cards = scorecards(data)                      headline, calibration and year, as defined
    card.to_csv(path), card.to_latex(path)

The column "observed" holds what the observed price gives where that is the target or the
benchmark of a metric: its share of negative prices and its tail mean, the year-to-year stability
of its profiles, its own distance to the causal curve of the wind effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from assume_gb.validation import mechanisms, metrics, stats, value
from assume_gb.validation.contract import (
    CALIBRATION,
    COMPETITIVE,
    HEADLINE,
    LEAR,
    NAIVE,
    OBSERVED,
    YEAR,
    ValidationData,
)

LOWER, HIGHER, ONE, ZERO, TARGET = "lower", "higher", "one", "zero", "observed"
DIRECTIONS = {
    LOWER: "lower is better",
    HIGHER: "higher is better",
    ONE: "closer to 1 is better",
    ZERO: "closer to 0 is better",
    TARGET: "closer to the observed value is better",
}
SYMBOLS = {
    LOWER: r"$\downarrow$",
    HIGHER: r"$\uparrow$",
    ONE: r"$\to 1$",
    ZERO: r"$\to 0$",
    TARGET: r"$\to$ obs.",
}
CRITERIA = {
    1: "Accuracy",
    2: "Error structure",
    3: "Distribution",
    4: "Benchmarks",
    5: "Mechanisms",
    6: "Market power",
    7: "Value",
    8: "Seed spread",
}


@dataclass(frozen=True)
class Metric:
    """One row of the scorecard: its criterion, key, label and unit, the direction in which it is
    better, whether a skill against the naive forecast makes sense for it, whether it belongs to
    the headline window only, and the decimals it is shown with."""

    criterion: int
    key: str
    label: str
    unit: str
    direction: str
    skill: bool = True
    headline_only: bool = False
    digits: int = 2


# The naive forecast is the observed price a week earlier: it has the observed distribution, profiles
# and level almost by construction, so a skill against it means something only for the timing of
# prices and what follows from it (accuracy, correlation, capture prices, the storage decision,
# CRPS); elsewhere the metric is compared with its target alone.
METRICS = (
    Metric(1, "mae", "MAE", "£/MWh", LOWER),
    Metric(1, "rmse", "RMSE", "£/MWh", LOWER),
    Metric(1, "rmae", "rMAE", "", LOWER, skill=False, headline_only=True),
    Metric(2, "bias", "Bias", "£/MWh", ZERO, skill=False),
    Metric(2, "sigma_ratio", "Sigma ratio", "", ONE, skill=False),
    Metric(2, "pearson_r", "Pearson r", "", HIGHER),
    Metric(2, "profile_r_hour", "Hour-of-day profile r", "", HIGHER, skill=False),
    Metric(
        2,
        "profile_amplitude_hour",
        "Hour-of-day profile amplitude",
        "",
        ONE,
        skill=False,
    ),
    Metric(2, "profile_r_weekday", "Weekday profile r", "", HIGHER, skill=False),
    Metric(
        2,
        "profile_amplitude_weekday",
        "Weekday profile amplitude",
        "",
        ONE,
        skill=False,
    ),
    Metric(3, "w1", "Wasserstein-1", "£/MWh", LOWER, skill=False),
    Metric(
        3,
        "negative_share",
        "Share of negative prices",
        "",
        TARGET,
        digits=3,
        skill=False,
    ),
    Metric(3, "q01_error", "Error at the 1st percentile", "£/MWh", ZERO, skill=False),
    Metric(3, "q05_error", "Error at the 5th percentile", "£/MWh", ZERO, skill=False),
    Metric(3, "q95_error", "Error at the 95th percentile", "£/MWh", ZERO, skill=False),
    Metric(3, "q99_error", "Error at the 99th percentile", "£/MWh", ZERO, skill=False),
    Metric(
        3, "tail_mean", "Mean above the 95th percentile", "£/MWh", TARGET, skill=False
    ),
    Metric(
        4,
        "dm_naive",
        "DM p-value, better than naive",
        "",
        LOWER,
        skill=False,
        headline_only=True,
        digits=3,
    ),
    Metric(
        4,
        "dm_competitive",
        "DM p-value, better than competitive",
        "",
        LOWER,
        skill=False,
        headline_only=True,
        digits=3,
    ),
    Metric(
        5,
        "coefficients_consistent",
        "Coefficients with overlapping 95% intervals",
        "",
        HIGHER,
        skill=False,
    ),
    Metric(
        5,
        "coefficient_error",
        "Coefficient difference, relative",
        "",
        LOWER,
        skill=False,
    ),
    Metric(
        5,
        "wind_effect_error",
        "Wind effect against the causal curve",
        "£/MWh per GW",
        LOWER,
        skill=False,
    ),
    Metric(
        6,
        "markup_w1",
        "Markup Wasserstein-1, mean over scarcity bins",
        "£/MWh",
        LOWER,
        skill=False,
    ),
    Metric(
        6,
        "markup_peak",
        "Median markup error, scarcest bin",
        "£/MWh",
        ZERO,
        skill=False,
    ),
    Metric(7, "capture_wind", "Wind capture price error", "£/MWh", ZERO),
    Metric(7, "capture_solar", "Solar capture price error", "£/MWh", ZERO),
    Metric(
        7, "storage_revenue", "Storage revenue error, relative", "", ZERO, skill=False
    ),
    Metric(7, "decision_value", "Storage decision value", "", HIGHER),
    Metric(8, "crps", "CRPS", "£/MWh", LOWER),
)
BY_KEY = {metric.key: metric for metric in METRICS}
SEED_METRICS = {
    "mae": metrics.mae,
    "rmse": metrics.rmse,
    "bias": metrics.bias,
    "sigma_ratio": metrics.sigma_ratio,
    "pearson_r": metrics.pearson_r,
}


def skill(value: float, naive: float, direction: str, target: float = np.nan) -> float:
    """Skill of a metric value against the naive forecast's: 1 - loss / naive loss, the loss being
    the value (lower), its distance to 0, to 1 or to the observed ``target``; for a score whose
    best value is 1 (higher), (value - naive) / (1 - naive). 1 is perfect, 0 no better than the
    naive forecast, below 0 worse [lagoForecastingDayaheadElectricity2021]."""
    if direction == HIGHER:
        return (value - naive) / (1 - naive) if naive != 1 else np.nan
    losses = {
        LOWER: lambda v: v,
        ZERO: abs,
        ONE: lambda v: abs(v - 1),
        TARGET: lambda v: abs(v - target),
    }
    base = losses[direction](naive)
    return 1 - losses[direction](value) / base if base > 0 else np.nan


def significance(p_value: float) -> str:
    """Stars for a p-value: *** below 0.001, ** below 0.01, * below 0.05."""
    if not np.isfinite(p_value):
        return ""
    return (
        "***"
        if p_value < 0.001
        else "**"
        if p_value < 0.01
        else "*"
        if p_value < 0.05
        else ""
    )


def number(key: str, value: float) -> str:
    """A metric value as shown in the table and the figure: the metric's decimals, a p-value below
    0.001 as "<0.001", "--" for none."""
    if not np.isfinite(value):
        return "--"
    if key.startswith("dm_") and value < 0.001:
        return "<0.001"
    return f"{value:.{BY_KEY[key].digits}f}"


def _escape(text: str) -> str:
    for char, escaped in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ):
        text = text.replace(char, escaped)
    return text


@dataclass
class Scorecard:
    """The scorecard of one window: ``values`` (metric x series, the series being "observed" and
    the models), ``skill`` against the naive forecast (same shape; NaN where it does not apply),
    ``stars`` (the Diebold-Mariano significance of beating the naive forecast, on the MAE row),
    ``seeds`` (for an ensemble, the median and range of its single seeds' point metrics beside the
    value of its seed mean) and ``notes`` on what could not be computed."""

    window: str
    in_sample: bool
    values: pd.DataFrame
    skill: pd.DataFrame
    stars: pd.DataFrame
    seeds: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    @property
    def metrics(self) -> pd.DataFrame:
        """Criterion, label, unit and direction of every row."""
        rows = [BY_KEY[key] for key in self.values.index]
        return pd.DataFrame(
            {
                "criterion": [m.criterion for m in rows],
                "label": [m.label for m in rows],
                "unit": [m.unit for m in rows],
                "direction": [m.direction for m in rows],
            },
            index=self.values.index,
        )

    def long(self) -> pd.DataFrame:
        """One row per metric and series: window, in-sample flag, criterion, metric, label, unit,
        direction, series, value, skill and stars."""
        table = self.values.stack().dropna()
        frame = table.rename("value").rename_axis(["metric", "series"]).reset_index()
        frame["skill"] = [
            self.skill.loc[m, s] for m, s in zip(frame.metric, frame.series)
        ]
        frame["stars"] = [
            self.stars.loc[m, s] for m, s in zip(frame.metric, frame.series)
        ]
        meta = self.metrics
        for column in ("criterion", "label", "unit", "direction"):
            frame[column] = frame.metric.map(meta[column])
        frame.insert(0, "in_sample", self.in_sample)
        frame.insert(0, "window", self.window)
        columns = [
            "window",
            "in_sample",
            "criterion",
            "metric",
            "label",
            "unit",
            "direction",
        ]
        return frame[columns + ["series", "value", "skill", "stars"]]

    def to_csv(self, path: str | Path) -> Path:
        """Writes ``long`` to ``path``; the seed spread, if any, beside it as ``<stem>_seeds.csv``."""
        path = Path(path)
        self.long().to_csv(path, index=False)
        if len(self.seeds):
            self.seeds.to_csv(path.with_name(f"{path.stem}_seeds.csv"), index=False)
        return path

    def to_latex(self, path: str | Path | None = None) -> str:
        """The scorecard as a booktabs table (needs ``\\usepackage{booktabs}``): criterion, metric
        with its unit, direction, then one column per series; values with the decimals of their
        metric, the significance stars of the MAE row as a superscript, "--" where there is no
        value. The window and whether it is in-sample are noted in a comment line."""
        series = list(self.values.columns)
        lines = [
            f"% scorecard, window {self.window}, {'in-sample' if self.in_sample else 'out of sample'}",
            r"\begin{tabular}{llc" + "r" * len(series) + "}",
            r"\toprule",
            " & ".join(["Criterion", "Metric", "", *[_escape(s) for s in series]])
            + r" \\",
            r"\midrule",
        ]
        previous = None
        for key in self.values.index:
            metric = BY_KEY[key]
            if previous is not None and metric.criterion != previous:
                lines.append(r"\addlinespace")
            label = metric.label + (f" ({metric.unit})" if metric.unit else "")
            cells = [
                f"{metric.criterion} {CRITERIA[metric.criterion]}"
                if metric.criterion != previous
                else "",
                _escape(label),
                SYMBOLS[metric.direction],
            ]
            for name in series:
                text = number(key, self.values.loc[key, name])
                star = self.stars.loc[key, name]
                cells.append(text + (f"$^{{{star}}}$" if star else ""))
            lines.append(" & ".join(cells) + r" \\")
            previous = metric.criterion
        lines += [r"\bottomrule", r"\end{tabular}"]
        latex = "\n".join(lines) + "\n"
        if path is not None:
            Path(path).write_text(latex, encoding="utf-8")
        return latex


def scorecard(
    data: ValidationData,
    window: str = HEADLINE,
    models: list[str] | None = None,
    competitive: str = COMPETITIVE,
    market: str = "NordPool",
    battery: value.Battery = value.Battery(),
) -> Scorecard:
    """The scorecard of ``window`` (a name of ``data.periods``) for ``models`` (all by default),
    every metric on the same common sample. Point metrics use an ensemble's seed mean, the
    distribution metrics its pooled seeds and the CRPS its seeds; the CRPS of the naive forecast,
    a single run, is its MAE. ``market`` names the causal curve of the observed series ("NordPool"
    for N2EX, "APX" for EPEX) and ``battery`` the storage of the value criterion. What cannot be
    computed (no explanatory data, no competitive run, no reference penetration) is left empty
    and named in ``notes``."""
    headline = window == HEADLINE
    names = list(data.names if models is None else models)
    if not headline:
        names = [name for name in names if name not in (NAIVE, LEAR)]
    sample = data.sample(window, names)
    obs = sample[OBSERVED]
    values: dict[str, dict[str, float]] = {metric.key: {} for metric in METRICS}
    notes: list[str] = []
    ensembles = {
        name: data.ensemble(name, window, names)
        for name in names
        if data.is_ensemble(name)
    }

    def put(key: str, series: str, number: float) -> None:
        values[key][series] = float(number)

    # criteria 1 and 2 on the point forecasts
    for name in names:
        sim = sample[name]
        put("mae", name, metrics.mae(sim, obs))
        put("rmse", name, metrics.rmse(sim, obs))
        put("bias", name, metrics.bias(sim, obs))
        put("sigma_ratio", name, metrics.sigma_ratio(sim, obs))
        put("pearson_r", name, metrics.pearson_r(sim, obs))
        if headline and NAIVE in names:
            put("rmae", name, metrics.rmae(sim, obs, sample[NAIVE]))
        for by in ("hour", "weekday"):
            profile = metrics.profile_metrics(sim, obs, by)
            put(f"profile_r_{by}", name, profile["r"])
            put(f"profile_amplitude_{by}", name, profile["amplitude"])
    for by in ("hour", "weekday"):
        stability = metrics.profile_stability(data.observed, by)
        if len(stability):
            put(f"profile_r_{by}", OBSERVED, stability["r"].mean())
            put(f"profile_amplitude_{by}", OBSERVED, stability["amplitude"].mean())
        else:
            notes.append(
                f"no year-to-year benchmark of the {by} profile: fewer than two observed years"
            )

    # criterion 3: the distribution, an ensemble's seeds pooled
    for name in names:
        distribution = ensembles.get(name, sample[name])
        put("w1", name, metrics.wasserstein1(distribution, obs))
        put("negative_share", name, metrics.negative_share(distribution))
        errors = metrics.quantile_errors(distribution, obs)
        for q, error in errors.items():
            put(f"q{round(q * 100):02d}_error", name, error)
        put("tail_mean", name, metrics.tail_mean(distribution))
    put("negative_share", OBSERVED, metrics.negative_share(obs))
    put("tail_mean", OBSERVED, metrics.tail_mean(obs))

    # criterion 4, on the headline window only
    if headline:
        for name in names:
            if NAIVE in names and name != NAIVE:
                put("dm_naive", name, stats.dm_test(obs, sample[NAIVE], sample[name]))
            if competitive in names and name != competitive:
                put(
                    "dm_competitive",
                    name,
                    stats.dm_test(obs, sample[competitive], sample[name]),
                )

    # criteria 5 to 7 need the explanatory data
    if data.exog is None:
        notes.append("no explanatory data: criteria 5 to 7 left out")
    else:
        regressions = mechanisms.compare_regressions(data, window, names)
        if regressions.empty:
            notes.append("too few periods for the regressions of criterion 5")
        for name, rows in regressions[regressions["regime"] == "all"].groupby("model"):
            put("coefficients_consistent", name, rows["overlap"].mean())
            put(
                "coefficient_error",
                name,
                (rows["difference"].abs() / rows["observed"].abs()).mean(),
            )
        try:
            binned = mechanisms.compare_binned(data, "wind", market, window, names)
            for series, rows in binned.dropna(subset=["reference"]).groupby("series"):
                put(
                    "wind_effect_error",
                    series,
                    (rows["estimate"] - rows["reference"]).abs().mean(),
                )
        except ValueError as error:
            notes.append(f"wind effect against the causal curve left out: {error}")
        if competitive in names:
            table = mechanisms.markups(
                data, window, names, competitive, include_competitive=True
            )
            if table.empty:
                notes.append(
                    "no bins of scarcity: the scarcity measure does not vary in the window"
                )
            peak = table[
                table["scarcity_high"] == table["scarcity_high"].max()
            ].set_index("series")
            for series, rows in table[table["series"] != OBSERVED].groupby("series"):
                put("markup_w1", series, rows["w1"].mean())
                put(
                    "markup_peak",
                    series,
                    peak.loc[series, "median"] - peak.loc[OBSERVED, "median"],
                )
        else:
            notes.append(f"no competitive run ({competitive!r}): criterion 6 left out")
        generation = data.explanatory(window, names).reindex(columns=["wind", "solar"])
        captured = value.capture_prices(
            sample, generation.dropna(axis=1, how="all"), by="all"
        )
        for row in captured[captured["series"] != OBSERVED].itertuples():
            put(f"capture_{row.technology}", row.series, row.error)
        revenues = value.storage_revenue(sample, battery)
        if len(revenues):
            storage = value.storage_summary(revenues)
            for name in names:
                put("storage_revenue", name, storage.loc[name, "relative_error"])
                put("decision_value", name, storage.loc[name, "decision_value"])
        else:
            notes.append("no complete day in the window: the storage decision left out")

    # criterion 8, ensembles only; the naive forecast is a single run, whose CRPS is its MAE
    for name, members in ensembles.items():
        put("crps", name, metrics.crps_ensemble(members, obs).mean())
    if ensembles and NAIVE in names:
        put("crps", NAIVE, values["mae"][NAIVE])

    columns = [OBSERVED, *names]
    rows = [
        metric.key
        for metric in METRICS
        if values[metric.key] and (headline or not metric.headline_only)
    ]
    table = pd.DataFrame({key: values[key] for key in rows}).T.reindex(
        index=rows, columns=columns
    )
    skills = pd.DataFrame(np.nan, index=table.index, columns=columns)
    if headline and NAIVE in names:
        for key in rows:
            metric = BY_KEY[key]
            naive = table.loc[key, NAIVE]
            if metric.skill and np.isfinite(naive):
                for name in names:
                    skills.loc[key, name] = skill(
                        table.loc[key, name],
                        naive,
                        metric.direction,
                        table.loc[key, OBSERVED],
                    )
    stars = pd.DataFrame("", index=table.index, columns=columns)
    if "dm_naive" in table.index:
        for name in names:
            stars.loc["mae", name] = significance(table.loc["dm_naive", name])
    seeds = []
    for name, members in ensembles.items():
        for key, function in SEED_METRICS.items():
            single = [function(members[seed], obs) for seed in members.columns]
            seeds.append(
                {
                    "model": name,
                    "metric": key,
                    "seed_mean": table.loc[key, name],
                    "median": float(np.median(single)),
                    "min": float(np.min(single)),
                    "max": float(np.max(single)),
                    "seeds": len(single),
                }
            )
    return Scorecard(
        window=window,
        in_sample=data.in_sample(window),
        values=table,
        skill=skills,
        stars=stars,
        seeds=pd.DataFrame(
            seeds,
            columns=["model", "metric", "seed_mean", "median", "min", "max", "seeds"],
        ),
        notes=notes,
    )


def scorecards(data: ValidationData, **kwargs) -> dict[str, Scorecard]:
    """The scorecard of the headline window and, where defined, of the calibration window and the
    full year; ``kwargs`` go to ``scorecard``."""
    windows = [
        HEADLINE,
        *[name for name in (CALIBRATION, YEAR) if name in data.periods],
    ]
    return {window: scorecard(data, window, **kwargs) for window in windows}
