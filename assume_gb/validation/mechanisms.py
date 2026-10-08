# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Criteria 5 and 6 of the scorecard: are the mechanisms right, and is market power plausible?

Criterion 5 runs one reduced-form regression, identically, on the observed price and on each
simulated price, and compares the coefficients term by term and per regime
(``compare_regressions``). Each price is regressed on its own information set
(``ValidationData.information_set``): the observed price on the day-ahead forecasts it was formed
on, a model's price on what drove it (the outturn for the agent-based model). Regressing every
price on the outturn is the robustness variant; ``attenuation`` says how far that variant
understates an effect that works through the forecasts. ``binned_effects`` estimates the effect of
wind or solar per bin of predicted penetration, and ``compare_binned`` sets it beside the causal
curves of Cacciarelli et al. (``reference``), against which the regression is only a consistency
check [cacciarelliWeActuallyUnderstand2025] [pangalloDataDrivenEconomicAgentBased2024].

Criterion 6 compares the markup of each model over the competitive run with the markup of the
observed price over it, by bin of scarcity (``markups``)
[borensteinMeasuringMarketInefficiencies2002] [twomeyMonitoringMarketPower].

Standard errors are Newey-West (HAC) over one day of periods unless set otherwise. The rows of a
regression are the periods of its sample in time order; where the sample has gaps (a window of
several ranges, missing periods) the periods on either side are treated as adjacent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

from assume_gb.validation import metrics, reference
from assume_gb.validation.contract import (
    COMPETITIVE,
    FORECAST,
    HEADLINE,
    NAIVE,
    OBSERVED,
    OUTTURN,
    VINTAGED,
    ValidationData,
    Windows,
    infer_resolution,
    vintage,
)

DEFAULT_FORMULA = (
    "price ~ wind + solar + demand + gas + carbon + C(hour) + C(dow) + C(month)"
)
CONTROLS = "gas + carbon + C(hour) + C(dow) + C(month)"
TERMS = ("wind", "solar", "demand", "gas", "carbon")
OWN = "own"  # each price on its own information set
REGRESSION_COLUMNS = [
    "regime", "model", "term", "information_observed", "information_model", "observed", "observed_low",
    "observed_high", "simulated", "simulated_low", "simulated_high", "difference", "overlap", "n",
]  # fmt: skip
MARKUP_COLUMNS = [
    "scarcity_low", "scarcity_high", "series", "n", "mean", "median", "q05", "q25", "q75", "q95", "w1",
]  # fmt: skip
# bins of predicted penetration, %
BINS = {"wind": tuple(range(0, 101, 5)), "solar": tuple(range(0, 41, 2))}
# predicted penetration as the causal reference defines it (``penetration``)
REFERENCE, SCENARIO = "reference", "scenario"
REFERENCE_NUMERATOR = {"wind": "wind_tx_forecast", "solar": "solar_forecast"}
REFERENCE_LOADS = ("tsd_forecast", "nd_forecast")
# technologies whose reference curve is indicative only: the authors' solar forecast barely varies
# within a day, so their solar axis is not the half-hourly solar share (see data/reference)
INDICATIVE = ("solar",)


def _periods_per_day(index: pd.DatetimeIndex) -> int:
    return int(round(pd.Timedelta(days=1) / infer_resolution(index)))


def _regimes(data: ValidationData) -> list[str | None]:
    """None (the whole window) and every regime."""
    return [None, *data.regimes]


def design(
    price: pd.Series | None, exog: pd.DataFrame, information: str = FORECAST
) -> pd.DataFrame:
    """The regression frame of a price series: ``price`` (GBP/MWh), wind, solar and demand (GW)
    from the ``information`` set (the ``*_forecast`` columns for "forecast", the outturn for
    "outturn"), residual_demand (demand - wind - solar of that set), every other column of ``exog``
    as it is, and the local hour (0-23), weekday (``dow``, 0 = Monday) and month of each period."""
    frame = exog.drop(
        columns=[f"{name}_forecast" for name in VINTAGED], errors="ignore"
    )
    for name in VINTAGED:
        column = vintage(name, information)
        if column in exog:
            frame[name] = exog[column]
        elif name in exog:
            raise ValueError(
                f"exog has {name} but no {column}: the {information} set is incomplete"
            )
    if {"demand", "wind", "solar"} <= set(frame.columns):
        frame["residual_demand"] = frame["demand"] - frame["wind"] - frame["solar"]
    frame["hour"] = exog.index.hour
    frame["dow"] = exog.index.dayofweek
    frame["month"] = exog.index.month
    if price is not None:
        frame.insert(0, "price", price.reindex(exog.index))
    return frame


def regression(
    price: pd.Series,
    exog: pd.DataFrame,
    formula: str = DEFAULT_FORMULA,
    information: str = FORECAST,
    hac_lags: int | None = None,
) -> RegressionResultsWrapper:
    """OLS of ``formula`` on the frame of ``design`` with Newey-West (HAC) standard errors over
    ``hac_lags`` periods (one day of periods by default); periods with a missing value are left
    out. The default formula is the reduced form of criterion 5, price ~ wind + solar + demand +
    gas + carbon with hour, weekday and month fixed effects: coefficients in GBP/MWh per GW for
    wind, solar and demand, per GBP/MWh thermal for gas and per GBP/t for carbon
    [cacciarelliWeActuallyUnderstand2025] [fabraPassThroughEmissionsCosts2014]."""
    frame = design(price, exog, information)
    lags = hac_lags if hac_lags is not None else _periods_per_day(frame.index)
    return smf.ols(formula, frame).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def coefficients(
    result: RegressionResultsWrapper,
    terms: tuple[str, ...] = TERMS,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Estimate, standard error and (1 - ``alpha``) interval of each of ``terms`` present in a
    fitted regression, one row per term, with the number of periods ``n``."""
    present = [term for term in terms if term in result.params.index]
    interval = result.conf_int(alpha).loc[present]
    table = pd.DataFrame(
        {
            "estimate": result.params[present],
            "se": result.bse[present],
            "ci_low": interval[0],
            "ci_high": interval[1],
        }
    )
    return table.assign(n=int(result.nobs)).rename_axis("term")


def compare_regressions(
    data: ValidationData,
    window: str | Windows | None = HEADLINE,
    models: list[str] | None = None,
    formula: str = DEFAULT_FORMULA,
    information: str = OWN,
    hac_lags: int | None = None,
    terms: tuple[str, ...] = TERMS,
    min_periods: int = 500,
) -> pd.DataFrame:
    """Criterion 5: the regression ``formula`` run identically on the observed price and on each
    model's price (the point forecast of an ensemble) on the same periods, over the whole window
    ("all") and per regime with at least ``min_periods`` periods. ``information`` "own" regresses
    each price on its own information set (the observed price on the forecasts, a model on what
    drove it); "outturn" (or "forecast") regresses every price on that set, the robustness
    variant. One row per regime, model and term: the observed and simulated estimates with their
    95% intervals, their difference (observed - simulated) and whether the intervals overlap
    [cacciarelliWeActuallyUnderstand2025] [pangalloDataDrivenEconomicAgentBased2024]."""
    names = [
        name for name in (data.names if models is None else models) if name != NAIVE
    ]
    rows = []
    for regime in _regimes(data):
        sample = data.sample(window, names, regime)
        if len(sample) < min_periods:
            continue
        exog = data.exog.loc[sample.index]
        fits = {}
        for name in [OBSERVED, *names]:
            info = data.information_set(name) if information == OWN else information
            result = regression(sample[name], exog, formula, info, hac_lags)
            fits[name] = (info, coefficients(result, terms))
        observed_info, observed = fits[OBSERVED]
        for name in names:
            info, simulated = fits[name]
            for term, o in observed.iterrows():
                s = simulated.loc[term]
                rows.append(
                    {
                        "regime": regime or "all",
                        "model": name,
                        "term": term,
                        "information_observed": observed_info,
                        "information_model": info,
                        "observed": o.estimate,
                        "observed_low": o.ci_low,
                        "observed_high": o.ci_high,
                        "simulated": s.estimate,
                        "simulated_low": s.ci_low,
                        "simulated_high": s.ci_high,
                        "difference": o.estimate - s.estimate,
                        "overlap": bool(
                            o.ci_low <= s.ci_high and s.ci_low <= o.ci_high
                        ),
                        "n": int(o.n),
                    }
                )
    return pd.DataFrame(rows, columns=REGRESSION_COLUMNS)


def attenuation(
    data: ValidationData,
    window: str | Windows | None = HEADLINE,
    variables: tuple[str, ...] = VINTAGED,
    controls: str = CONTROLS,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """How far a coefficient on the outturn understates an effect that works through the
    day-ahead forecast, per variable over the whole window ("all") and per regime. With F~ and O~
    the residuals of the forecast and of the outturn on ``controls`` (Frisch-Waugh-Lovell; the
    other forecast variables are not partialled out):

        variance_ratio      Var(F~) / Var(O~): the attenuation factor when the forecast error is
                            uncorrelated with the forecast (an efficient forecast)
        regression_factor   Cov(F~, O~) / Var(O~): the factor without that assumption

    A price formed on the forecast and regressed on the outturn, as the observed price is in the
    outturn-for-both variant of ``compare_regressions``, has its coefficient scaled by the
    regression factor; below 1 is attenuation. On the common sample of ``window``."""
    rows = []
    for regime in _regimes(data):
        exog = data.explanatory(window, models, regime)
        frame = design(None, exog, OUTTURN)
        for name in variables:
            forecast = smf.ols(
                f"x ~ {controls}", frame.assign(x=exog[f"{name}_forecast"])
            ).fit()
            outturn = smf.ols(f"x ~ {controls}", frame.assign(x=exog[name])).fit()
            both = pd.concat(
                [forecast.resid, outturn.resid], axis=1, join="inner"
            ).to_numpy()
            covariance = np.cov(both, rowvar=False)
            rows.append(
                {
                    "regime": regime or "all",
                    "variable": name,
                    "var_forecast": covariance[0, 0],
                    "var_outturn": covariance[1, 1],
                    "variance_ratio": covariance[0, 0] / covariance[1, 1],
                    "regression_factor": covariance[0, 1] / covariance[1, 1],
                    "n": len(both),
                }
            )
    return pd.DataFrame(rows)


def reference_load(exog: pd.DataFrame) -> str | None:
    """The load the reference definition of penetration divides by: the transmission system
    demand forecast (``tsd_forecast``), else the national demand forecast (``nd_forecast``); None
    without either."""
    return next((column for column in REFERENCE_LOADS if column in exog), None)


def penetration(
    exog: pd.DataFrame, technology: str, definition: str = REFERENCE
) -> pd.Series:
    """Predicted penetration of ``technology``, %: forecast output over forecast load x 100, as
    forecast at 09:00 D-1.

        reference   the axis of the causal curves of Cacciarelli et al.: transmission-connected
                    wind (``wind_tx_forecast``; for solar the solar forecast) over the
                    transmission system demand forecast (``tsd_forecast``). Where there is none,
                    the national demand forecast (``nd_forecast``) stands in; it leaves out
                    station load, pumping and exports, so penetration comes out somewhat higher
        scenario    the scenario's own forecasts: wind with embedded wind over demand net of
                    nuclear and pumped storage; not the axis of the reference

    [cacciarelliWeActuallyUnderstand2025]"""
    if definition == SCENARIO:
        numerator, load = vintage(technology, FORECAST), vintage("demand", FORECAST)
    elif definition == REFERENCE:
        numerator, load = REFERENCE_NUMERATOR[technology], reference_load(exog)
        if load is None or numerator not in exog:
            raise ValueError(
                "exog lacks the reference definition of penetration (wind_tx_forecast and "
                "tsd_forecast or nd_forecast): write penetration_forecasts.csv with "
                "`python -m assume_gb penetration` where the GB data repository is, or pass "
                "definition='scenario'"
            )
    else:
        raise ValueError(
            f"definition must be {REFERENCE!r} or {SCENARIO!r}, not {definition!r}"
        )
    return (100 * exog[numerator] / exog[load]).rename(f"{technology}_penetration")


def _without(formula: str, term: str) -> tuple[str, str]:
    """The response and the right-hand side of ``formula`` without ``term``."""
    response, rhs = (side.strip() for side in formula.split("~"))
    kept = [part.strip() for part in rhs.split("+") if part.strip() != term]
    return response, " + ".join(kept)


def binned_effects(
    price: pd.Series,
    exog: pd.DataFrame,
    technology: str = "wind",
    information: str = FORECAST,
    bins: tuple[float, ...] | None = None,
    binning: pd.Series | None = None,
    formula: str = DEFAULT_FORMULA,
    hac_lags: int | None = None,
    min_periods: int = 200,
    definition: str = REFERENCE,
) -> pd.DataFrame:
    """The marginal effect of ``technology`` on the price in each bin of predicted penetration,
    GBP/MWh per GW: ``formula`` with the technology's term replaced by a level and a slope per
    bin, price ~ C(bin) + sum_b 1[bin = b] x + the other terms, with HAC errors, where x is the
    technology's output in the price's ``information`` set. The bins are on ``binning``
    (``penetration`` on ``definition`` by default) whatever the information set, so that observed
    and simulated prices are binned alike, and as the causal curves are
    [cacciarelliWeActuallyUnderstand2025]. One row per bin with at least ``min_periods`` periods:
    its edges (%, left edge included), the mean penetration and number of periods in it, the
    estimate, its standard error and 95% interval."""
    frame = design(price, exog, information)
    binning = penetration(exog, technology, definition) if binning is None else binning
    edges = BINS[technology] if bins is None else bins
    frame["penetration"] = binning.reindex(frame.index)
    frame = frame.dropna(
        subset=["price", "penetration", *[t for t in TERMS if t in frame]]
    )
    intervals = pd.cut(frame["penetration"], edges, right=False)
    frame["penetration_bin"] = intervals.cat.codes
    counts = frame["penetration_bin"].value_counts()
    kept = sorted(
        code for code, count in counts.items() if code >= 0 and count >= min_periods
    )
    frame = frame[frame["penetration_bin"].isin(kept)].copy()
    for code in kept:
        frame[f"slope_{code}"] = frame[technology] * (frame["penetration_bin"] == code)
    response, rest = _without(formula, technology)
    slopes = " + ".join(f"slope_{code}" for code in kept)
    lags = hac_lags if hac_lags is not None else _periods_per_day(frame.index)
    result = smf.ols(f"{response} ~ C(penetration_bin) + {slopes} + {rest}", frame).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    interval = result.conf_int(0.05)
    categories = intervals.cat.categories
    rows = []
    for code in kept:
        term = f"slope_{code}"
        in_bin = frame["penetration_bin"] == code
        rows.append(
            {
                "low": categories[code].left,
                "high": categories[code].right,
                "mean_penetration": frame.loc[in_bin, "penetration"].mean(),
                "n": int(in_bin.sum()),
                "estimate": result.params[term],
                "se": result.bse[term],
                "ci_low": interval.loc[term, 0],
                "ci_high": interval.loc[term, 1],
            }
        )
    return pd.DataFrame(rows)


def compare_binned(
    data: ValidationData,
    technology: str = "wind",
    market: str = "NordPool",
    window: str | Windows | None = HEADLINE,
    models: list[str] | None = None,
    information: str = OWN,
    regime: str | None = None,
    bins: tuple[float, ...] | None = None,
    binning: pd.Series | None = None,
    hac_lags: int | None = None,
    min_periods: int = 200,
    definition: str = REFERENCE,
) -> pd.DataFrame:
    """Criterion 5 against the causal curves: ``binned_effects`` of the observed price and of each
    model's price on the same periods and bins, each on its own information set (or all on
    ``information``), beside the reference curve of ``market`` (the market of the observed price
    series: "NordPool" for N2EX, "APX" for the APX/EPEX auction) at each bin's mean penetration,
    interpolated linearly and not extrapolated (``reference.cate_at``). One row per series and
    bin, GBP/MWh per GW; ``load`` names the load the penetration was divided by and
    ``indicative`` marks a reference curve to be read as indicative only (solar)
    [cacciarelliWeActuallyUnderstand2025]."""
    names = [
        name for name in (data.names if models is None else models) if name != NAIVE
    ]
    sample = data.sample(window, names, regime)
    exog = data.exog.loc[sample.index]
    if binning is not None:
        load = "given"
    elif definition == REFERENCE:
        load = reference_load(exog)
    else:
        load = vintage("demand", FORECAST)
    binning = penetration(exog, technology, definition) if binning is None else binning
    tables = []
    for name in [OBSERVED, *names]:
        info = data.information_set(name) if information == OWN else information
        table = binned_effects(
            sample[name],
            exog,
            technology,
            info,
            bins,
            binning,
            hac_lags=hac_lags,
            min_periods=min_periods,
        )
        tables.append(table.assign(series=name, information=info))
    table = pd.concat(tables, ignore_index=True)
    curve = reference.cate_at(table["mean_penetration"].to_numpy(), market, technology)
    return table.assign(
        reference=curve["estimate"].to_numpy(),
        reference_low=curve["ci80_low"].to_numpy(),
        reference_high=curve["ci80_high"].to_numpy(),
        load=load,
        indicative=technology in INDICATIVE,
    )


def scarcity(exog: pd.DataFrame) -> pd.Series:
    """Residual demand over available capacity (outturn, dimensionless): the default scarcity
    measure by which markups are binned [twomeyMonitoringMarketPower]."""
    return (exog["residual_demand"] / exog["available_capacity"]).rename("scarcity")


def _distribution(values: pd.Series | pd.DataFrame) -> dict[str, float]:
    v = np.asarray(values, dtype=float).ravel()
    v = v[~np.isnan(v)]
    q05, q25, q50, q75, q95 = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": len(v),
        "mean": v.mean(),
        "median": q50,
        "q05": q05,
        "q25": q25,
        "q75": q75,
        "q95": q95,
    }


def markups(
    data: ValidationData,
    window: str | Windows | None = HEADLINE,
    models: list[str] | None = None,
    competitive: str = COMPETITIVE,
    measure: pd.Series | None = None,
    bins: int | tuple[float, ...] = 5,
    include_competitive: bool = False,
) -> pd.DataFrame:
    """Criterion 6: the markup of each model's price over the competitive run (model -
    competitive) against the observed markup over it (observed - competitive, a Borenstein-style
    estimate of the observed markup), GBP/MWh, by bin of scarcity: ``measure`` (``scarcity`` of
    the exog by default) cut into ``bins`` quantile bins, or at the given edges. One row per bin
    and series: the scarcity edges, the number of prices (an ensemble's seeds pooled), mean,
    median, 5th, 25th, 75th and 95th percentile, and for a model the Wasserstein-1 distance
    of its markups to the observed ones in the bin. With ``include_competitive`` the competitive
    run is a series too, with a markup of zero: its distance is how far the observed markups are
    from none [borensteinMeasuringMarketInefficiencies2002] [twomeyMonitoringMarketPower]."""
    names = [
        n
        for n in (data.names if models is None else models)
        if n not in (NAIVE, competitive)
    ]
    index = data.common(window, [competitive, *names])
    base = data.point(competitive).loc[index]
    measure = scarcity(data.exog) if measure is None else measure
    values = measure.reindex(index)
    cut = (
        pd.qcut(values, bins, duplicates="drop")
        if isinstance(bins, int)
        else pd.cut(values, bins, include_lowest=True)
    )
    observed = data.observed.loc[index] - base
    rows = []
    for interval in cut.cat.categories:
        in_bin = (cut == interval).to_numpy()
        if not in_bin.any():
            continue
        edges = {"scarcity_low": interval.left, "scarcity_high": interval.right}
        rows.append(
            {
                **edges,
                "series": OBSERVED,
                **_distribution(observed[in_bin]),
                "w1": np.nan,
            }
        )
        if include_competitive:
            zero = pd.Series(0.0, index=observed.index[in_bin])
            w1 = metrics.wasserstein1(zero, observed[in_bin])
            rows.append(
                {**edges, "series": competitive, **_distribution(zero), "w1": w1}
            )
        for name in names:
            simulated = data.members(name).loc[index][in_bin].sub(base[in_bin], axis=0)
            w1 = metrics.wasserstein1(simulated, observed[in_bin])
            rows.append({**edges, "series": name, **_distribution(simulated), "w1": w1})
    return pd.DataFrame(rows, columns=MARKUP_COLUMNS)
