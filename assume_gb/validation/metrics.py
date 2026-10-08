# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Criteria 1, 2, 3 and 8 of the scorecard: how large the errors are, why they are large, whether
the distribution is right, and what the spread of the seeds says.

Every function takes prices in GBP/MWh, normally columns of ``ValidationData.sample`` (the common
sample of a window). Functions that compare two series use the periods in which both have a
value; the distribution functions take the values of each side unpaired, so a simulated side may
be the pooled seeds of an ensemble. Moments are population moments (divisor n), so that the MSE
decomposition adds up exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from assume_gb.validation.contract import local_day, period_of_day

QUANTILES = (0.01, 0.05, 0.95, 0.99)
CALENDAR = ("hour", "weekday", "month", "period")


def _paired(*series: pd.Series) -> list[np.ndarray]:
    """The values of the series on the periods in which all of them have one."""
    frame = pd.concat(series, axis=1, keys=range(len(series))).dropna()
    return [frame[k].to_numpy(dtype=float) for k in range(len(series))]


def _values(series: pd.Series | pd.DataFrame) -> np.ndarray:
    """All values of a series (or of every column of a frame), without the missing ones."""
    values = np.asarray(series, dtype=float).ravel()
    return values[~np.isnan(values)]


# ------------------------------------------------------------------ criterion 1: accuracy
def mae(sim: pd.Series, obs: pd.Series) -> float:
    """Mean absolute error, mean |s_t - o_t|, GBP/MWh: the headline accuracy metric
    [lagoForecastingDayaheadElectricity2021]."""
    s, o = _paired(sim, obs)
    return float(np.mean(np.abs(s - o)))


def rmse(sim: pd.Series, obs: pd.Series) -> float:
    """Root mean square error, sqrt(mean (s_t - o_t)^2), GBP/MWh: weighs spikes more than the MAE
    and makes the results comparable with the AMIRIS and ASSUME back-tests
    [maurerKnowYourTools2024]."""
    s, o = _paired(sim, obs)
    return float(np.sqrt(np.mean((s - o) ** 2)))


def rmae(sim: pd.Series, obs: pd.Series, naive: pd.Series) -> float:
    """Relative MAE: the MAE of ``sim`` over the MAE of the naive forecast on the same periods,
    dimensionless; below 1 beats the naive forecast. It makes errors comparable across years, and
    ranks models within one test period as the MAE does
    [lagoForecastingDayaheadElectricity2021]."""
    s, o, n = _paired(sim, obs, naive)
    return float(np.mean(np.abs(s - o)) / np.mean(np.abs(n - o)))


# ------------------------------------------------------------------ criterion 2: why errors are large
def bias(sim: pd.Series, obs: pd.Series) -> float:
    """Mean simulated minus mean observed price, GBP/MWh: the level part of the error
    [maurerKnowYourTools2024]."""
    s, o = _paired(sim, obs)
    return float(s.mean() - o.mean())


def sigma_ratio(sim: pd.Series, obs: pd.Series) -> float:
    """Standard deviation of the simulated over that of the observed price, dimensionless: the
    amplitude part of the error, below 1 when the model compresses price volatility
    [casarCanShadowPrices]."""
    s, o = _paired(sim, obs)
    return float(s.std() / o.std())


def pearson_r(sim: pd.Series, obs: pd.Series) -> float:
    """Pearson correlation of simulated and observed prices: the timing part of the error. On its
    own it hides errors in level and amplitude [maurerKnowYourTools2024]."""
    s, o = _paired(sim, obs)
    return float(np.corrcoef(s, o)[0, 1])


def mse_decomposition(sim: pd.Series, obs: pd.Series) -> dict[str, float]:
    """The mean squared error and its three parts, (GBP/MWh)^2:

        MSE = (mu_s - mu_o)^2 + (sigma_s - r sigma_o)^2 + sigma_o^2 (1 - r^2)
              "bias"            "amplitude"             "timing"

    with population moments, so the parts add up to the MSE exactly; bias, sigma ratio and r
    therefore determine the RMSE. A standard identity (the note's "Identities behind the
    overlaps"), used for criterion 2 [maurerKnowYourTools2024] [casarCanShadowPrices]."""
    s, o = _paired(sim, obs)
    r = np.corrcoef(s, o)[0, 1]
    return {
        "mse": float(np.mean((s - o) ** 2)),
        "bias": float((s.mean() - o.mean()) ** 2),
        "amplitude": float((s.std() - r * o.std()) ** 2),
        "timing": float(o.var() * (1 - r**2)),
    }


def _calendar(index: pd.DatetimeIndex, by: str) -> np.ndarray:
    if by == "hour":
        return np.asarray(index.hour)
    if by == "weekday":
        return np.asarray(index.dayofweek)
    if by == "month":
        return np.asarray(index.month)
    if by == "period":
        return period_of_day(index)
    raise ValueError(f"no profile by {by!r}: one of {CALENDAR}")


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else np.nan


def _correlation(a: pd.Series, b: pd.Series) -> float:
    """Pearson r of two profiles; NaN where one of them is flat."""
    if a.std(ddof=0) == 0 or b.std(ddof=0) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def profile(series: pd.Series, by: str = "hour") -> pd.Series:
    """Mean price by local hour of the day (0-23), weekday (0 = Monday), month (1-12) or
    wall-clock period of the day ("period"), GBP/MWh [casarCanShadowPrices]."""
    series = series.dropna()
    return series.groupby(_calendar(series.index, by)).mean().rename_axis(by)


def profile_metrics(
    sim: pd.Series, obs: pd.Series, by: str = "hour"
) -> dict[str, float]:
    """How well the mean profile of ``sim`` matches that of ``obs`` on the periods both have: the
    Pearson correlation of the two profiles ("r", do the peaks line up) and the ratio of their
    standard deviations across hours or weekdays ("amplitude", below 1 for a flattened profile).
    Casar et al. find the shadow prices of a cost-minimising energy system model about 7 times
    too flat by hour of day and 10 times by weekday [casarCanShadowPrices]."""
    both = pd.concat({"sim": sim, "obs": obs}, axis=1).dropna()
    s, o = profile(both["sim"], by), profile(both["obs"], by)
    return {"r": _correlation(s, o), "amplitude": _ratio(s.std(ddof=0), o.std(ddof=0))}


def profile_stability(
    observed: pd.Series, by: str = "hour", min_days: int = 300
) -> pd.DataFrame:
    """The benchmark for ``profile_metrics``: the correlation ("r") and the amplitude ratio (later
    over earlier year) of the observed profiles of consecutive calendar years, one row per pair.
    A model cannot be expected to match a profile more closely than one year matches the next
    [casarCanShadowPrices]. Years with prices on fewer than ``min_days`` days are left out."""
    observed = observed.dropna()
    years = np.asarray(observed.index.year)
    days = pd.Series(local_day(observed.index)).groupby(years).nunique()
    kept = [year for year in days.index if days[year] >= min_days]
    rows = []
    for earlier, later in zip(kept, kept[1:]):
        if later != earlier + 1:
            continue
        a = profile(observed[years == earlier], by)
        b = profile(observed[years == later], by)
        rows.append(
            {
                "years": f"{earlier}-{later}",
                "r": _correlation(a, b.reindex(a.index)),
                "amplitude": _ratio(b.std(ddof=0), a.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows, columns=["years", "r", "amplitude"]).set_index("years")


# ------------------------------------------------------------------ criterion 3: distribution and tails
def wasserstein1(sim: pd.Series | pd.DataFrame, obs: pd.Series) -> float:
    """Wasserstein-1 distance between the price distributions of ``sim`` and ``obs``, GBP/MWh.
    With as many prices on both sides it is the MAE between the two price duration curves,
    mean |s_(i) - o_(i)| over the sorted series; otherwise (e.g. pooled seeds)
    ``scipy.stats.wasserstein_distance``. A close match of duration curves can coexist with
    large hourly errors [nitschBacktestingAgentbasedModel2021]."""
    s, o = _values(sim), _values(obs)
    if len(s) == len(o):
        return float(np.mean(np.abs(np.sort(s) - np.sort(o))))
    return float(stats.wasserstein_distance(s, o))


def negative_share(series: pd.Series | pd.DataFrame) -> float:
    """Share of periods with a price below zero, 0 to 1. Energy system models tend to miss
    negative prices, which come from must-run conditions and strategic bidding
    [nitschBacktestingAgentbasedModel2021] [maurerKnowYourTools2024]."""
    return float(np.mean(_values(series) < 0))


def quantile_errors(
    sim: pd.Series | pd.DataFrame,
    obs: pd.Series,
    quantiles: tuple[float, ...] = QUANTILES,
) -> pd.Series:
    """Simulated minus observed price at each quantile of the two distributions, GBP/MWh: the
    error at the ends of the duration curve [nitschBacktestingAgentbasedModel2021]
    [bordignonCombiningDayaheadForecasts2013]."""
    q = np.asarray(quantiles, dtype=float)
    errors = np.quantile(_values(sim), q) - np.quantile(_values(obs), q)
    return pd.Series(errors, index=pd.Index(q, name="quantile"))


def tail_mean(series: pd.Series | pd.DataFrame, q: float = 0.95) -> float:
    """Mean price at or above the series' own ``q`` quantile, GBP/MWh: the upper-tail expected
    shortfall, which scores how high the spikes go [bordignonCombiningDayaheadForecasts2013]."""
    values = _values(series)
    return float(values[values >= np.quantile(values, q)].mean())


# ------------------------------------------------------------------ criterion 8: spread of the seeds
def _ensemble(
    members: pd.DataFrame, obs: pd.Series
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    both = members.join(obs.rename("__observed__"), how="inner").dropna()
    return (
        both.drop(columns="__observed__").to_numpy(float),
        both["__observed__"].to_numpy(float),
        both.index,
    )


def crps_ensemble(
    members: pd.DataFrame, obs: pd.Series, fair: bool = False
) -> pd.Series:
    """Continuous ranked probability score of an ensemble in each period, GBP/MWh, with the
    ensemble estimator

        CRPS = mean_i |x_i - y| - 1 / (2 m^2) sum_ij |x_i - x_j|

    over the m members x_i (the seeds, columns of ``members``) and the observation y. With
    ``fair`` the second term is divided by 2 m (m - 1) instead, which removes the bias of a small
    ensemble (it needs two members). For one member the CRPS is the absolute error, so its mean
    is on the scale of the MAE. A seed ensemble covers behavioural randomness only, not input
    uncertainty [pinsonNonparametricProbabilisticForecasts2007]
    [nowotarskiRecentAdvancesElectricity2018]."""
    x, y, index = _ensemble(members, obs)
    m = x.shape[1]
    if fair and m < 2:
        raise ValueError("the fair CRPS needs at least two members")
    spread = np.mean(np.abs(x - y[:, None]), axis=1)
    rank = np.arange(1, m + 1)
    pair_sum = 2 * np.sum(
        (2 * rank - m - 1) * np.sort(x, axis=1), axis=1
    )  # sum_ij |x_i - x_j|
    return pd.Series(
        spread - pair_sum / (2 * m * (m - 1) if fair else 2 * m * m),
        index=index,
        name="crps",
    )


def pit_ensemble(members: pd.DataFrame, obs: pd.Series, seed: int = 0) -> pd.Series:
    """Probability integral transform of the observation within the ensemble in each period, in
    [0, 1): (r + u) / (m + 1), where r is the number of the m members below the observation, ties
    broken at random, and u is uniform on [0, 1). Uniform for a calibrated ensemble; a U shape
    means too little spread [pinsonNonparametricProbabilisticForecasts2007]
    [nowotarskiRecentAdvancesElectricity2018]."""
    x, y, index = _ensemble(members, obs)
    rng = np.random.default_rng(seed)
    below = np.sum(x < y[:, None], axis=1)
    ties = np.sum(x == y[:, None], axis=1)
    rank = below + rng.integers(0, ties + 1)
    return pd.Series(
        (rank + rng.random(len(y))) / (x.shape[1] + 1), index=index, name="pit"
    )
