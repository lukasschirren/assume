# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from assume.common.forecaster import UnitForecaster
from assume.strategies import bidding_strategies
from assume.strategies.storage_strategies import (
    StorageEnergyOptimizationScheduleStrategy,
)
from assume.units import Storage

start = datetime(2023, 7, 1)
# a cheap night, a dear evening and a flat day in between, hourly
prices = [10, 10, 10, 10, 40, 40, 40, 40, 40, 40, 90, 90, 40, 40, 40, 40]
products = [
    (start + timedelta(hours=h), start + timedelta(hours=h + 1), None)
    for h in range(len(prices))
]


def make_storage(**kwargs) -> Storage:
    index = pd.date_range(start, periods=len(prices) + 1, freq="h")
    forecaster = UnitForecaster(index, market_prices={"EOM": prices + [40]})
    params = dict(
        id="battery",
        unit_operator="test_operator",
        technology="battery",
        bidding_strategies={"EOM": StorageEnergyOptimizationScheduleStrategy()},
        forecaster=forecaster,
        max_power_charge=-100,
        max_power_discharge=100,
        capacity=200,
        efficiency_charge=1.0,
        efficiency_discharge=0.8,
        additional_cost_charge=2.0,
        additional_cost_discharge=1.0,
        initial_soc=0.0,
    )
    params.update(kwargs)
    return Storage(**params)


def bids_of(unit, mock_market_config):
    bids = unit.bidding_strategies["EOM"].calculate_bids(
        unit, mock_market_config, products
    )
    return {bid["start_time"].hour: (bid["volume"], bid["price"]) for bid in bids}


def test_strategy_is_registered():
    assert (
        bidding_strategies["storage_energy_optimization_schedule"]
        is StorageEnergyOptimizationScheduleStrategy
    )


def test_plan_charges_cheap_and_discharges_dear(mock_market_config):
    unit = make_storage()
    bids = bids_of(unit, mock_market_config)

    # 200 MWh are bought in the cheap hours and the 160 MWh they yield are sold in the two
    # dear hours (how they are split between equal prices is up to the solver); the day in
    # between is not worth a cycle (40 * 0.8 - 1 < 40 + 2)
    charged = {h: v for h, (v, p) in bids.items() if v < 0}
    discharged = {h: v for h, (v, p) in bids.items() if v > 0}
    assert set(charged) <= {0, 1, 2, 3} and sum(charged.values()) == pytest.approx(-200)
    assert set(discharged) == {10, 11} and sum(discharged.values()) == pytest.approx(
        160
    )
    assert all(v <= 100 + 1e-9 for v in discharged.values())

    # charging is bid at the maximum price, discharging at the break-even price of the stored energy
    assert all(
        p == mock_market_config.maximum_bid_price for h, (v, p) in bids.items() if v < 0
    )
    break_even = (10 + 2.0) / (1.0 * 0.8) + 1.0
    assert all(p == pytest.approx(break_even) for h, (v, p) in bids.items() if v > 0)
    assert all(bid[1] >= 0 for bid in bids.values())


def test_plan_starts_from_the_state_of_charge_and_ends_at_the_initial_one(
    mock_market_config,
):
    unit = make_storage(initial_soc=0.5)
    # the store is half full at the start, as the plan wants it at the end
    bids = bids_of(unit, mock_market_config)
    charged = sum(v for v, p in bids.values() if v < 0)
    discharged = sum(v for v, p in bids.values() if v > 0)
    assert discharged == pytest.approx(-charged * 0.8)

    # with an empty store the plan first has to buy what it ends with
    unit.outputs["soc"].at[start] = 0.0
    bids = bids_of(unit, mock_market_config)
    charged = sum(v for v, p in bids.values() if v < 0)
    discharged = sum(v for v, p in bids.values() if v > 0)
    assert -charged * 1.0 - discharged / 0.8 == pytest.approx(100)


def test_energy_already_sold_is_part_of_the_plan(mock_market_config):
    unit = make_storage(initial_soc=0.0)
    # 50 MW already sold in both dear hours on another market: they take 125 MWh of the 200 MWh
    # the store can hold, the remaining 75 MWh yield 60 MWh more, within the remaining 50 MW
    unit.outputs["energy"].at[products[10][0]] = 50
    unit.outputs["energy"].at[products[11][0]] = 50
    bids = bids_of(unit, mock_market_config)
    discharged = {h: v for h, (v, p) in bids.items() if v > 0}
    assert set(discharged) == {10, 11} and sum(discharged.values()) == pytest.approx(60)
    assert all(v <= 50 + 1e-9 for v in discharged.values())
    assert sum(v for v, p in bids.values() if v < 0) == pytest.approx(-200)


def test_a_step_charges_or_discharges_but_not_both(mock_market_config):
    # at negative prices charging and discharging at once would be paid for the round-trip loss
    unit = make_storage(additional_cost_charge=0.0, additional_cost_discharge=0.0)
    unit.forecaster.price["EOM"].data[:] = -60.0
    unit.forecaster.price["EOM"].data[12:14] = 90.0
    strategy = unit.bidding_strategies["EOM"]
    charge, discharge = strategy.plan(
        unit,
        unit.forecaster.price["EOM"].data[:-1],
        np.ones(len(prices)),
        np.zeros(len(prices)),
        0.0,
    )
    assert not ((charge > 1e-9) & (discharge > 1e-9)).any()
    # being paid to consume still makes cycling worthwhile, but step by step and within the store
    assert discharge.sum() == pytest.approx(charge.sum() * 0.8)
    # the dear hours get what a full store yields
    assert discharge[12] + discharge[13] == pytest.approx(160)


def test_no_bids_without_a_spread(mock_market_config):
    unit = make_storage()
    unit.forecaster.price["EOM"].data[:] = 40.0
    assert bids_of(unit, mock_market_config) == {}


def test_no_bids_on_a_price_forecast_with_missing_values(mock_market_config, caplog):
    unit = make_storage()
    unit.forecaster.price["EOM"].data[5] = np.nan
    with caplog.at_level("WARNING"):
        assert bids_of(unit, mock_market_config) == {}
    assert "missing values" in caplog.text


def test_reward_books_costs_and_profit(mock_market_config):
    unit = make_storage()
    orderbook = [
        {
            "start_time": products[0][0],
            "end_time": products[0][1],
            "only_hours": None,
            "price": 3000,
            "volume": -100,
            "accepted_price": 10,
            "accepted_volume": -100,
        }
    ]
    unit.set_dispatch_plan(mock_market_config, orderbook)
    unit.calculate_cashflow_and_reward(mock_market_config, orderbook)
    assert unit.outputs["energy_cashflow"].at[products[0][0]] == -1000
    assert unit.outputs["total_costs"].at[products[0][0]] == 100 * 2.0
    assert unit.outputs["profit"].at[products[0][0]] == -1000 - 200
    assert np.isclose(unit.outputs["soc"].at[products[0][1]], 0.5)
