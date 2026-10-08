# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Criteria 1, 2, 3 and 8 on the synthetic prices and the models with known distortions."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from assume_gb.validation import metrics as M
from assume_gb.validation.tests import synthetic

DISTORTED = ["shifted", "flattened", "lagged", "noisy"]


def test_the_mse_decomposition_adds_up_to_the_mse(data):
    sample = data.sample()
    for name in DISTORTED:
        parts = M.mse_decomposition(sample[name], sample.observed)
        assert parts["bias"] + parts["amplitude"] + parts["timing"] == pytest.approx(
            parts["mse"], rel=1e-12
        )
        assert np.sqrt(parts["mse"]) == pytest.approx(
            M.rmse(sample[name], sample.observed), rel=1e-12
        )
    # each distortion lands in its own part
    shifted = M.mse_decomposition(sample.shifted, sample.observed)
    assert (
        shifted["bias"] == pytest.approx(225.0)
        and shifted["amplitude"] + shifted["timing"] < 1e-9
    )
    flattened = M.mse_decomposition(sample.flattened, sample.observed)
    assert flattened["timing"] < 1e-9 and flattened["amplitude"] == pytest.approx(
        0.25 * sample.observed.var(ddof=0)
    )
    lagged = M.mse_decomposition(sample.lagged, sample.observed)
    assert (
        lagged["timing"] > 0.8 * lagged["mse"] and lagged["bias"] < 0.01 * lagged["mse"]
    )


def test_point_metrics_of_the_distorted_models(data):
    sample = data.sample()
    obs = sample.observed
    assert M.mae(sample.shifted, obs) == pytest.approx(15.0)
    assert M.rmse(sample.shifted, obs) == pytest.approx(15.0)
    assert M.bias(sample.shifted, obs) == pytest.approx(15.0)
    assert M.sigma_ratio(sample.flattened, obs) == pytest.approx(0.5)
    assert M.pearson_r(sample.flattened, obs) == pytest.approx(1.0)
    assert M.rmse(sample.noisy, obs) == pytest.approx(10.0, abs=0.2)
    assert M.pearson_r(sample.lagged, obs) < 0.99 and M.sigma_ratio(
        sample.lagged, obs
    ) == pytest.approx(1.0, abs=0.01)


def test_rmae_of_the_naive_model_is_one(data):
    sample = data.sample()
    assert M.rmae(sample.naive, sample.observed, sample.naive) == pytest.approx(1.0)
    expected = 15.0 / M.mae(sample.naive, sample.observed)
    assert M.rmae(sample.shifted, sample.observed, sample.naive) == pytest.approx(
        expected
    )


def test_wasserstein1_matches_scipy_and_the_mae_between_sorted_series(data):
    sample = data.sample()
    obs = sample.observed
    for name in DISTORTED:
        w1 = M.wasserstein1(sample[name], obs)
        assert w1 == pytest.approx(
            stats.wasserstein_distance(sample[name], obs), rel=1e-9
        )
        assert w1 == pytest.approx(
            np.mean(np.abs(np.sort(sample[name]) - np.sort(obs)))
        )
    assert M.wasserstein1(sample.shifted, obs) == pytest.approx(15.0)
    # the duration curve of a late model is right although its hourly errors are large
    assert M.wasserstein1(sample.lagged, obs) < 0.1 * M.mae(sample.lagged, obs)
    # pooled seeds: more prices on one side than the other
    seeds = synthetic.ensemble(obs, members=3)
    assert M.wasserstein1(seeds, obs) == pytest.approx(
        stats.wasserstein_distance(seeds.to_numpy().ravel(), obs)
    )


def test_tails_and_negative_prices(data):
    sample = data.sample()
    obs = sample.observed
    assert 0.005 < M.negative_share(obs) < 0.05
    assert M.negative_share(sample.shifted) < M.negative_share(obs)
    errors = M.quantile_errors(sample.shifted, obs)
    assert errors.index.tolist() == list(M.QUANTILES) and np.allclose(errors, 15.0)
    assert M.tail_mean(sample.shifted) - M.tail_mean(obs) == pytest.approx(15.0)
    assert M.tail_mean(obs) > obs.quantile(0.95)


def test_profile_metrics_detect_the_flattened_model(data):
    sample = data.sample()
    obs = sample.observed
    for by in ("hour", "weekday"):
        flat = M.profile_metrics(sample.flattened, obs, by)
        assert flat["amplitude"] == pytest.approx(0.5) and flat["r"] == pytest.approx(
            1.0
        )
        shift = M.profile_metrics(sample.shifted, obs, by)
        assert shift["amplitude"] == pytest.approx(1.0) and shift["r"] == pytest.approx(
            1.0
        )
    late = M.profile_metrics(sample.lagged, obs, "hour")
    assert 0.5 < late["r"] < 0.99
    assert M.profile(obs, "hour").index.tolist() == list(range(24))
    assert len(M.profile(obs, "period")) == 48
    with pytest.raises(ValueError):
        M.profile(obs, "season")


def test_profile_stability_compares_consecutive_years(observed):
    table = M.profile_stability(observed, "hour")
    assert table.index.tolist() == ["2022-2023"]
    assert table.loc["2022-2023", "r"] > 0.95 and table.loc[
        "2022-2023", "amplitude"
    ] == pytest.approx(1.0, abs=0.15)
    # a year with too few days is no benchmark
    assert M.profile_stability(observed.loc[:"2023-03-31"], "hour").empty


def test_crps_of_a_one_member_ensemble_is_the_mae(data):
    sample = data.sample()
    obs = sample.observed
    crps = M.crps_ensemble(sample[["noisy"]], obs)
    assert crps.mean() == pytest.approx(M.mae(sample.noisy, obs))
    with pytest.raises(ValueError):
        M.crps_ensemble(sample[["noisy"]], obs, fair=True)


def test_crps_of_an_ensemble_matches_its_definition():
    rng = np.random.default_rng(0)
    obs = pd.Series(rng.normal(50, 10, 200))
    members = pd.DataFrame(obs.to_numpy()[:, None] + rng.normal(0, 8, (200, 6)))
    x, y = members.to_numpy(), obs.to_numpy()
    pairs = np.abs(x[:, :, None] - x[:, None, :]).sum(axis=(1, 2))
    direct = np.abs(x - y[:, None]).mean(axis=1) - pairs / (2 * 6 * 6)
    direct_fair = np.abs(x - y[:, None]).mean(axis=1) - pairs / (2 * 6 * 5)
    np.testing.assert_allclose(M.crps_ensemble(members, obs), direct)
    np.testing.assert_allclose(M.crps_ensemble(members, obs, fair=True), direct_fair)
    # never more than the mean absolute error of the members
    assert M.crps_ensemble(members, obs).mean() < np.abs(x - y[:, None]).mean()


def test_pit_is_uniform_for_a_calibrated_ensemble_and_u_shaped_without_spread():
    rng = np.random.default_rng(1)
    obs = pd.Series(rng.normal(0, 1, 20000))
    calibrated = pd.DataFrame(rng.normal(0, 1, (20000, 9)))
    pit = M.pit_ensemble(calibrated, obs)
    assert pit.between(0, 1).all() and stats.kstest(pit, "uniform").pvalue > 0.01
    narrow = pd.DataFrame(rng.normal(0, 0.3, (20000, 9)))
    outer = M.pit_ensemble(narrow, obs)
    assert ((outer < 0.1) | (outer > 0.9)).mean() > 0.5
    # an observation equal to every member is placed at random
    ties = M.pit_ensemble(
        pd.DataFrame(np.zeros((20000, 4))), pd.Series(np.zeros(20000))
    )
    assert stats.kstest(ties, "uniform").pvalue > 0.01
