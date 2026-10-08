# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Criterion 7: capture prices weigh every price series with the same generation, and the
storage programme earns what can be worked out by hand, keeps clock-change days whole and never
does better at the observed price than the observed price's own schedule."""

import numpy as np
import pandas as pd
import pytest

from assume_gb.validation import value as V
from assume_gb.validation.tests import synthetic

FLAT = V.Battery(power=1.0, energy=2.0, round_trip_efficiency=1.0, soc_day=0.0)


def test_capture_prices_weigh_every_price_with_the_same_generation():
    index = synthetic.index("2023-01-01", "2023-03-01")
    rng = np.random.default_rng(4)
    wind = pd.Series(rng.uniform(1, 20, len(index)), index=index)
    observed = 100 - 3 * wind  # cheap when windy
    sample = pd.DataFrame({"observed": observed, "flat": 70.0, "copy": observed})
    table = V.capture_prices(sample, wind.to_frame("wind"), by="month")
    january = table[(table.period == "2023-01")].set_index("series")
    jan = index.month == 1
    expected = (observed[jan] * wind[jan]).sum() / wind[jan].sum()
    assert january.loc["observed", "capture_price"] == pytest.approx(expected)
    assert (
        january.loc["observed", "value_factor"] < 1
    )  # wind earns less than the mean price
    assert january.loc["flat", "capture_price"] == pytest.approx(70.0)
    assert january.loc["flat", "value_factor"] == pytest.approx(1.0)
    assert january.loc["copy", "error"] == pytest.approx(0.0)
    assert january.loc["flat", "error"] == pytest.approx(70.0 - expected)
    yearly = V.capture_prices(sample, wind.to_frame("wind"), by="year")
    assert yearly.period.unique().tolist() == ["2023"] and set(yearly.by) == {"year"}
    with pytest.raises(ValueError):
        V.capture_prices(sample, wind.to_frame("wind"), by="week")


def test_arbitrage_earns_what_can_be_worked_out_by_hand():
    prices = np.r_[
        np.zeros(24), np.full(24, 100.0)
    ]  # cheap night, dear day, half-hours
    revenue, net = V.arbitrage(prices, 0.5, FLAT)
    assert revenue == pytest.approx(200.0)  # 2 MWh bought at 0, sold at 100
    assert net[:24].sum() * 0.5 == pytest.approx(-2.0) and net[
        24:
    ].sum() * 0.5 == pytest.approx(2.0)
    lossy = V.Battery(power=1.0, energy=2.0, round_trip_efficiency=0.81, soc_day=0.0)
    revenue, _ = V.arbitrage(prices, 0.5, lossy)
    assert revenue == pytest.approx(180.0)  # 2.22 MWh in, 2 MWh stored, 1.8 MWh out
    assert V.arbitrage(np.full(48, 55.0), 0.5, FLAT)[0] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        V.Battery(soc_min=0.6, soc_day=0.5)


def test_storage_keeps_clock_change_days_whole_and_skips_incomplete_ones():
    index = synthetic.index("2023-10-28", "2023-10-31")  # 29 October has 25 hours
    prices = pd.Series(np.tile([0.0, 100.0], len(index) // 2), index=index)
    sample = pd.DataFrame({"observed": prices, "model": prices})
    sample.iloc[5, 1] = np.nan  # one missing price makes 28 October incomplete
    table = V.storage_revenue(sample, FLAT)
    assert sorted(table.day.dt.day.unique()) == [29, 30]
    revenue = table.set_index(["day", "series"]).revenue
    assert revenue[(pd.Timestamp("2023-10-29"), "observed")] > 0


def test_no_schedule_earns_more_at_the_observed_price_than_its_own(observed):
    sample = pd.DataFrame(
        {
            "observed": observed,
            "noisy": synthetic.noisy(observed, sigma=20.0),
            "flattened": synthetic.flattened(observed),
        }
    ).loc["2023-06-01":"2023-06-30"]
    table = V.storage_revenue(sample)
    assert len(table) == 30 * 3
    days = table.pivot(index="day", columns="series", values="settled")
    assert (days.le(days["observed"] + 1e-6, axis=0)).all().all()
    summary = V.storage_summary(table)
    assert summary.loc["observed", "decision_value"] == pytest.approx(1.0)
    assert summary.loc["observed", "error"] == pytest.approx(0.0)
    # a schedule that chases noise trades on spreads that are not there and loses on the round
    # trip: the decision value can fall below zero
    assert summary.loc["noisy", "decision_value"] < 1
    # a flattened price promises less arbitrage, though its schedule is nearly the right one
    assert summary.loc["flattened", "relative_error"] < -0.3
    assert (
        summary.loc["flattened", "decision_value"]
        > summary.loc["noisy", "decision_value"]
    )
