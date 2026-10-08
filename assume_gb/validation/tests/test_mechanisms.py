# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Criteria 5 and 6 on synthetic prices with known coefficients: the observed price is formed on
the forecasts, a simulated one on the outturn, so that each information set and the attenuation
of the outturn-for-both variant have a known answer."""

import numpy as np
import pandas as pd
import pytest

from assume_gb.validation import mechanisms as X
from assume_gb.validation.contract import ValidationData
from assume_gb.validation.tests import synthetic
from assume_gb.validation.tests.conftest import PERIODS

YEAR_2023 = slice("2023-01-01", "2023-12-31")


@pytest.fixture(scope="module")
def simulated(exog) -> pd.Series:
    """A price formed on the outturn with the same coefficients: a model with the right
    mechanisms."""
    return synthetic.prices(exog, "outturn", seed=1)


@pytest.fixture(scope="module")
def two(observed, simulated, exog) -> ValidationData:
    return ValidationData(
        observed,
        {"abm": simulated},
        exog,
        periods=dict(PERIODS),
        regimes={"first half": ("2023-01-01", "2023-06-30")},
    )


def test_the_regression_recovers_the_known_coefficients(observed, exog):
    result = X.regression(
        observed.loc[YEAR_2023], exog.loc[YEAR_2023], information="forecast"
    )
    table = X.coefficients(result)
    assert table.index.tolist() == list(X.TERMS) and (table.n == 17520).all()
    for term, true in synthetic.COEF.items():
        assert table.loc[term, "estimate"] == pytest.approx(
            true, abs=4 * table.loc[term, "se"]
        )
        assert (
            table.loc[term, "ci_low"]
            < table.loc[term, "estimate"]
            < table.loc[term, "ci_high"]
        )
    assert table.loc["wind", "estimate"] == pytest.approx(
        synthetic.COEF["wind"], rel=0.05
    )


def test_each_price_is_regressed_on_its_own_information_set(two):
    table = X.compare_regressions(two)
    assert set(table.regime) == {"all", "first half"} and set(table.model) == {"abm"}
    assert (table.information_observed == "forecast").all() and (
        table.information_model == "outturn"
    ).all()
    whole = table[table.regime == "all"].set_index("term")
    assert whole.overlap.all()
    for term, true in synthetic.COEF.items():
        assert whole.loc[term, "observed"] == pytest.approx(true, rel=0.1)
        assert whole.loc[term, "simulated"] == pytest.approx(true, rel=0.1)
    assert (whole.n == len(two.sample(models=["abm"]))).all()


def test_the_outturn_variant_is_attenuated_as_predicted(two):
    own = X.compare_regressions(two).query("regime == 'all'").set_index("term")
    outturn = (
        X.compare_regressions(two, information="outturn")
        .query("regime == 'all'")
        .set_index("term")
    )
    assert (outturn.information_observed == "outturn").all()
    factors = X.attenuation(two).query("regime == 'all'").set_index("variable")
    assert (
        set(factors.index) == {"wind", "solar", "demand"} and (factors.n > 16000).all()
    )
    wind = factors.loc["wind"]
    assert 0.5 < wind.variance_ratio < 1 and 0.5 < wind.regression_factor < 1
    # the observed price responds to the forecast: on the outturn its wind coefficient shrinks by
    # about the regression factor, while the simulated price, formed on the outturn, keeps it
    realised = outturn.loc["wind", "observed"] / own.loc["wind", "observed"]
    assert realised == pytest.approx(wind.regression_factor, abs=0.05)
    assert outturn.loc["wind", "simulated"] == pytest.approx(
        own.loc["wind", "simulated"]
    )
    assert not outturn.loc["wind", "overlap"]


def test_penetration_is_on_the_axis_of_the_reference(exog):
    x = exog.loc[YEAR_2023]
    on_reference = X.penetration(x, "wind")
    assert np.allclose(on_reference, 100 * x.wind_tx_forecast / x.tsd_forecast)
    # transmission wind over transmission demand sits below all wind over demand net of nuclear
    assert (on_reference < X.penetration(x, "wind", "scenario")).all()
    assert np.allclose(
        X.penetration(x, "solar"), 100 * x.solar_forecast / x.tsd_forecast
    )
    # without a transmission system demand forecast the national demand one stands in, by name
    national = x.drop(columns="tsd_forecast").assign(nd_forecast=x.demand_forecast)
    assert X.reference_load(national) == "nd_forecast"
    assert np.allclose(
        X.penetration(national, "wind"), 100 * x.wind_tx_forecast / x.demand_forecast
    )
    with pytest.raises(ValueError, match="assume_gb penetration"):
        X.penetration(x.drop(columns="tsd_forecast"), "wind")


def test_binned_effects_recover_a_slope_that_changes_with_penetration(exog):
    x = exog.loc[YEAR_2023]
    share = X.penetration(x, "wind")
    slope = np.select([share < 20, share < 40], [-6.0, -1.0], -4.0)  # a U-shaped effect
    base = synthetic.prices(x)
    price = base - synthetic.COEF["wind"] * x.wind_forecast + slope * x.wind_forecast
    table = X.binned_effects(price, x, "wind", bins=(0, 20, 40, 200))
    assert table[["low", "high"]].values.tolist() == [[0, 20], [20, 40], [40, 200]]
    assert table.estimate.to_numpy() == pytest.approx([-6.0, -1.0, -4.0], abs=0.4)
    assert (
        table.n.sum() == len(x)
        and (table.mean_penetration.between(table.low, table.high)).all()
    )
    # a bin with too few periods is left out
    sparse = X.binned_effects(
        price, x, "wind", bins=(0, 20, 40, 200), min_periods=10_000
    )
    assert len(sparse) < 3


def test_binned_effects_sit_beside_the_reference_curve(two):
    table = X.compare_binned(
        two, "wind", "NordPool", bins=(0, 10, 20, 30, 40, 50, 60, 200)
    )
    assert set(table.series) == {"observed", "abm"}
    assert set(table.information) == {"forecast", "outturn"}
    inside = table.mean_penetration.between(3, 50)
    assert (
        table.loc[inside, "reference"].notna().all()
        and table.loc[~inside, "reference"].isna().all()
    )
    first = table[inside].iloc[0]
    expected = X.reference.cate_at(first.mean_penetration, "NordPool", "wind").iloc[0]
    assert first.reference == pytest.approx(expected.estimate)
    assert first.reference_low <= first.reference <= first.reference_high
    assert (table.load == "tsd_forecast").all() and not table.indicative.any()
    solar = X.compare_binned(two, "solar", "NordPool", bins=(0, 2, 4, 6, 8, 100))
    assert solar.indicative.all() and set(solar.series) == {"observed", "abm"}


def test_markups_by_scarcity(observed, exog):
    rng = np.random.default_rng(3)
    competitive = (
        observed - 5.0
    )  # the observed price is 5 above the competitive run everywhere
    seeds = pd.DataFrame(
        competitive.to_numpy()[:, None] + 5.0 + rng.normal(0, 4.0, (len(observed), 3)),
        index=observed.index,
    )
    data = ValidationData(
        observed,
        {"competitive": competitive, "abm": seeds},
        exog,
        periods=dict(PERIODS),
    )
    table = X.markups(data)
    assert len(table) == 2 * 5 and set(table.series) == {"observed", "abm"}
    observed_rows = table[table.series == "observed"]
    assert np.allclose(observed_rows[["mean", "median", "q05", "q95"]], 5.0)
    assert observed_rows.n.sum() == len(data.common(models=["competitive", "abm"]))
    model_rows = table[table.series == "abm"]
    assert (model_rows.n == 3 * observed_rows.n.to_numpy()).all()  # seeds pooled
    assert model_rows["mean"].to_numpy() == pytest.approx(5.0, abs=0.2)
    # W1 of N(5, 4^2) against a point mass at 5 is E|N(0, 4^2)| = 4 sqrt(2 / pi)
    assert model_rows.w1.to_numpy() == pytest.approx(4 * np.sqrt(2 / np.pi), rel=0.05)
    # given edges instead of quantiles (scarcity is a ratio, about 0.3 on average here)
    edges = X.markups(data, bins=(-np.inf, 0.2, 0.4, np.inf))
    assert (
        edges.series == "observed"
    ).sum() == 3 and edges.scarcity_high.max() == np.inf
