# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Criterion 4: the Diebold-Mariano tests reject for a clearly better model, give uniform p-values
for two equally good ones, keep clock-change days whole, and reduce to epftoolbox's ``DM`` at lag
0 (against a transcription of it always, against epftoolbox itself when it is installed)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from assume_gb.validation import stats as S
from assume_gb.validation.contract import TZ, to_resolution
from assume_gb.validation.tests import synthetic


def epftoolbox_dm(p_real, p_pred_1, p_pred_2, norm=1, version="univariate"):
    """``DM`` of epftoolbox (``epftoolbox/evaluation/_dm.py``), transcribed: arrays of shape
    (n_days, n_prices_day), one-sided, p-value of "p_pred_2 is more accurate than p_pred_1"."""
    errors_pred_1 = p_real - p_pred_1
    errors_pred_2 = p_real - p_pred_2
    if version == "univariate":
        if norm == 1:
            d = np.abs(errors_pred_1) - np.abs(errors_pred_2)
        if norm == 2:
            d = errors_pred_1**2 - errors_pred_2**2
        n = d.shape[0]
        dm_stat = np.mean(d, axis=0) / np.sqrt((1 / n) * np.var(d, ddof=0, axis=0))
    elif version == "multivariate":
        if norm == 1:
            d = np.mean(np.abs(errors_pred_1), axis=1) - np.mean(
                np.abs(errors_pred_2), axis=1
            )
        if norm == 2:
            d = np.mean(errors_pred_1**2, axis=1) - np.mean(errors_pred_2**2, axis=1)
        n = d.size
        dm_stat = np.mean(d) / np.sqrt((1 / n) * np.var(d, ddof=0))
    return 1 - stats.norm.cdf(dm_stat)


@pytest.fixture(scope="module")
def summer(observed):
    """Sixty whole days of hourly prices without a clock change, and two forecasts of them that
    differ a little in accuracy."""
    hourly = to_resolution(observed, pd.Timedelta(hours=1)).loc[
        "2023-06-01":"2023-07-30"
    ]
    rng = np.random.default_rng(5)
    p1 = hourly + rng.normal(0, 8, len(hourly))
    p2 = hourly + rng.normal(0, 7.8, len(hourly))
    return hourly, p1, p2


def _days(series: pd.Series) -> np.ndarray:
    return series.to_numpy().reshape(-1, 24)


@pytest.mark.parametrize("norm", [1, 2])
def test_lag_zero_is_epftoolbox_dm(summer, norm):
    obs, p1, p2 = summer
    univariate = S.dm_test(obs, p1, p2, "univariate", norm)
    assert univariate.index.tolist() == list(range(24))
    expected = epftoolbox_dm(_days(obs), _days(p1), _days(p2), norm, "univariate")
    np.testing.assert_allclose(univariate.to_numpy(), expected, rtol=1e-9, atol=1e-12)
    multivariate = S.dm_test(obs, p1, p2, "multivariate", norm)
    expected = epftoolbox_dm(_days(obs), _days(p1), _days(p2), norm, "multivariate")
    assert multivariate == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert (
        1e-4 < multivariate < 1 - 1e-4
    )  # not saturated, so the comparison means something


@pytest.mark.parametrize("version", ["univariate", "multivariate"])
def test_matches_epftoolbox_when_installed(summer, version):
    evaluation = pytest.importorskip("epftoolbox.evaluation")
    obs, p1, p2 = summer
    ours = S.dm_test(obs, p1, p2, version)
    theirs = evaluation.DM(_days(obs), _days(p1), _days(p2), norm=1, version=version)
    np.testing.assert_allclose(
        np.asarray(ours, dtype=float), theirs, rtol=1e-9, atol=1e-12
    )


def test_rejects_for_a_clearly_better_model(data):
    obs = data.sample().observed
    rng = np.random.default_rng(6)
    worse = obs + rng.normal(0, 10, len(obs))
    better = obs + rng.normal(0, 3, len(obs))
    assert S.dm_test(obs, worse, better) < 1e-6
    assert S.dm_test(obs, better, worse) > 1 - 1e-6
    assert (S.dm_test(obs, worse, better, "univariate") < 0.01).all()


def test_two_equally_good_models_give_uniform_p_values():
    index = pd.date_range("2023-05-01", "2023-08-09", freq="h", tz=TZ, inclusive="left")
    obs = pd.Series(0.0, index=index)
    rng = np.random.default_rng(7)
    multivariate, univariate = [], []
    for replication in range(300):
        p1 = obs + rng.normal(0, 5, len(obs))
        p2 = obs + rng.normal(0, 5, len(obs))
        multivariate.append(S.dm_test(obs, p1, p2))
        if replication < 40:
            univariate.extend(S.dm_test(obs, p1, p2, "univariate"))
    multivariate = np.array(multivariate)
    assert stats.kstest(multivariate, "uniform").pvalue > 0.01
    assert 0.02 < np.mean(multivariate < 0.05) < 0.09
    assert stats.kstest(univariate, "uniform").pvalue > 0.01


def test_clock_change_days_count_once_and_keep_their_periods():
    index = synthetic.index("2023-10-27", "2023-10-31")  # 29 October has 25 hours
    obs = pd.Series(0.0, index=index)
    worse = pd.Series(2.0, index=index)
    better = pd.Series(1.0, index=index)
    daily = S.daily_loss_differential(obs, worse, better)
    assert len(daily) == 4 and np.allclose(daily, 1.0)
    d = S.loss_differential(obs, worse, better, norm=2)
    assert np.allclose(d, 3.0)
    # periods follow the wall clock: the 25-hour day adds no period 48 or 49
    by_period = S.dm_test(obs, worse + np.arange(len(index)) % 3, better, "univariate")
    assert by_period.index.tolist() == list(range(48))


def test_newey_west_variance():
    rng = np.random.default_rng(8)
    white = rng.normal(0, 1, 2000)
    assert S.long_run_variance(white, 0) == pytest.approx(np.var(white))
    persistent = np.empty(2000)
    persistent[0] = 0.0
    for t in range(1, 2000):
        persistent[t] = 0.7 * persistent[t - 1] + rng.normal(0, 1)
    # the long-run variance of an AR(1) with phi 0.7 is about (1 + 0.7) / (1 - 0.7) times its variance
    ratio = S.long_run_variance(persistent, 30) / S.long_run_variance(persistent, 0)
    assert 3.5 < ratio < 7.5
    # with a positive mean, accounting for the persistence makes the test less sure of itself
    shifted = persistent + 0.2
    assert S.dm_p_value(shifted, 0) < S.dm_p_value(shifted, 30) < 0.5


def test_p_value_matrix_for_all_model_pairs(data):
    sample = data.sample(models=["naive", "shifted", "noisy"])
    matrix = S.dm_matrix(sample)
    assert (
        matrix.index.tolist()
        == matrix.columns.tolist()
        == ["naive", "shifted", "noisy"]
    )
    assert matrix.index.name == "model_1" and matrix.columns.name == "model_2"
    assert np.isnan(np.diag(matrix)).all()
    # noisy (MAE about 8) beats shifted (MAE 15): small in column noisy, row shifted
    assert (
        matrix.loc["shifted", "noisy"] < 0.01 and matrix.loc["noisy", "shifted"] > 0.99
    )
    by_period = S.dm_by_period(sample)
    assert len(by_period) == 6 and by_period.shape[1] == 48
    assert by_period.loc[("shifted", "noisy")].max() < 0.01
    with pytest.raises(ValueError):
        S.dm_test(sample.observed, sample.naive, sample.noisy, version="bivariate")
