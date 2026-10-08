# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Criterion 4 of the scorecard: is a model better than simpler ones? One-sided Diebold-Mariano
tests of equal predictive accuracy, univariate (one test per period of the day) and multivariate
(one test on the daily loss), mirroring ``DM`` of epftoolbox
[lagoForecastingDayaheadElectricity2021], with an optional Newey-West correction for
autocorrelated loss differentials.

The convention is epftoolbox's: ``dm_test(obs, pred_1, pred_2)`` tests H0 "``pred_2`` is not more
accurate than ``pred_1``" against H1 "``pred_2`` is more accurate", so a small p-value says that
``pred_2`` is significantly better. In ``dm_matrix`` the entry in row A, column B is the p-value of
"B is more accurate than A".

Days and periods are local: the daily loss is the mean over the periods of the local delivery
day, so a day of 23 or 25 hours counts once, and the univariate test runs per wall-clock period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from assume_gb.validation.contract import local_day, period_of_day

VERSIONS = ("multivariate", "univariate")


def _loss(error: pd.Series, norm: int) -> pd.Series:
    if norm == 1:
        return error.abs()
    if norm == 2:
        return error**2
    raise ValueError(
        f"norm must be 1 (absolute errors) or 2 (squared errors), not {norm!r}"
    )


def loss_differential(
    obs: pd.Series, pred_1: pd.Series, pred_2: pd.Series, norm: int = 1
) -> pd.Series:
    """d_t = L(o_t - p1_t) - L(o_t - p2_t) on the periods in which all three series have a value,
    with L = |e| (``norm`` 1, GBP/MWh) or e^2 (``norm`` 2, (GBP/MWh)^2); positive where
    ``pred_2`` is the more accurate [lagoForecastingDayaheadElectricity2021]."""
    frame = pd.concat({"obs": obs, "p1": pred_1, "p2": pred_2}, axis=1).dropna()
    d = _loss(frame["obs"] - frame["p1"], norm) - _loss(
        frame["obs"] - frame["p2"], norm
    )
    return d.rename("d")


def daily_loss_differential(
    obs: pd.Series, pred_1: pd.Series, pred_2: pd.Series, norm: int = 1
) -> pd.Series:
    """The loss differential of the multivariate test, one value per local delivery day: the
    daily loss of ``pred_1`` less that of ``pred_2``, each the mean of the period losses of the
    day (the mean absolute error of the day for ``norm`` 1). A day of 23 or 25 hours counts once
    [lagoForecastingDayaheadElectricity2021]."""
    d = loss_differential(obs, pred_1, pred_2, norm)
    return d.groupby(local_day(d.index)).mean().rename_axis("day")


def long_run_variance(d: np.ndarray, lag: int = 0) -> float:
    """Newey-West estimate of the long-run variance of the series ``d``: its autocovariances
    gamma_k (divisor N) with Bartlett weights, gamma_0 + 2 sum_{k=1..lag} (1 - k / (lag + 1))
    gamma_k. With ``lag`` 0 it is the variance with divisor N that epftoolbox uses
    [lagoForecastingDayaheadElectricity2021]."""
    d = np.asarray(d, dtype=float)
    centred = d - d.mean()
    n = len(d)
    variance = centred @ centred / n
    for k in range(1, min(lag, n - 1) + 1):
        variance += 2 * (1 - k / (lag + 1)) * (centred[k:] @ centred[:-k]) / n
    return float(variance)


def dm_p_value(d: np.ndarray, lag: int = 0) -> float:
    """One-sided p-value 1 - Phi(DM) of the Diebold-Mariano statistic DM = mean(d) /
    sqrt(LRV(d) / N) of a loss differential series ``d`` (missing values dropped), small when
    ``d`` is positive on average; NaN when the two forecasts do not differ
    [lagoForecastingDayaheadElectricity2021]."""
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return np.nan
    variance = long_run_variance(d, lag)
    if variance <= 0:
        return np.nan
    return float(stats.norm.sf(d.mean() / np.sqrt(variance / len(d))))


def dm_test(
    obs: pd.Series,
    pred_1: pd.Series,
    pred_2: pd.Series,
    version: str = "multivariate",
    norm: int = 1,
    lag: int = 0,
) -> float | pd.Series:
    """One-sided Diebold-Mariano test of H0 "``pred_2`` is not more accurate than ``pred_1``"
    against H1 "``pred_2`` is more accurate": the p-value 1 - Phi(DM), with
    DM = mean(d) / sqrt(LRV(d) / N) over the loss differential d (see ``loss_differential``).

        multivariate   d per local delivery day (``daily_loss_differential``): one p-value
        univariate     d per day for each wall-clock period of the day: a p-value per period,
                       a Series indexed by period number; on the 25-hour day the two values of a
                       repeated period are averaged, on the 23-hour day the skipped ones are absent

    LRV is the Newey-West long-run variance over ``lag`` days (``long_run_variance``); with
    ``lag`` 0 this is epftoolbox's ``DM`` [lagoForecastingDayaheadElectricity2021]. One-sided DM
    tests on absolute errors are also used for agent-based price forecasts
    [fraunholzAdvancedPriceForecasting2021]."""
    if version == "multivariate":
        return dm_p_value(
            daily_loss_differential(obs, pred_1, pred_2, norm).to_numpy(), lag
        )
    if version == "univariate":
        d = loss_differential(obs, pred_1, pred_2, norm)
        table = d.groupby([local_day(d.index), period_of_day(d.index)]).mean().unstack()
        p_values = {
            period: dm_p_value(table[period].to_numpy(), lag)
            for period in table.columns
        }
        return pd.Series(p_values, name="p_value").rename_axis("period")
    raise ValueError(f"version must be one of {VERSIONS}, not {version!r}")


def _names(sample: pd.DataFrame, models: list[str] | None, observed: str) -> list[str]:
    return (
        list(models)
        if models is not None
        else [c for c in sample.columns if c != observed]
    )


def dm_matrix(
    sample: pd.DataFrame,
    models: list[str] | None = None,
    observed: str = "observed",
    norm: int = 1,
    lag: int = 0,
) -> pd.DataFrame:
    """p-values of the multivariate Diebold-Mariano test for every ordered pair of models in
    ``sample`` (the observed price and one column per model, as ``ValidationData.sample`` gives
    it): the entry in row A, column B is the p-value of "B is more accurate than A", so small
    values in a column mark the rows that model beats. The diagonal is NaN. The layout of
    epftoolbox's ``plot_multivariate_DM_test`` [lagoForecastingDayaheadElectricity2021]."""
    names = _names(sample, models, observed)
    matrix = pd.DataFrame(
        np.nan,
        index=pd.Index(names, name="model_1"),
        columns=pd.Index(names, name="model_2"),
    )
    for a in names:
        for b in names:
            if a != b:
                matrix.loc[a, b] = dm_test(
                    sample[observed], sample[a], sample[b], "multivariate", norm, lag
                )
    return matrix


def dm_by_period(
    sample: pd.DataFrame,
    models: list[str] | None = None,
    observed: str = "observed",
    norm: int = 1,
    lag: int = 0,
) -> pd.DataFrame:
    """p-values of the univariate Diebold-Mariano test for every ordered pair of models in
    ``sample``: one row per pair (``model_1``, ``model_2``: the p-value of "model_2 is more
    accurate than model_1"), one column per wall-clock period of the day
    [lagoForecastingDayaheadElectricity2021]."""
    names = _names(sample, models, observed)
    rows = {
        (a, b): dm_test(sample[observed], sample[a], sample[b], "univariate", norm, lag)
        for a in names
        for b in names
        if a != b
    }
    return pd.DataFrame(rows).T.rename_axis(["model_1", "model_2"])
