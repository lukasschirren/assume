# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from assume.common.forecaster import PowerplantForecaster
from assume.strategies import EnergyHeuristicLookaheadStrategy, bidding_strategies
from assume.strategies.lookahead_strategies import plan_commitment, unit_state
from assume.units import PowerPlant

START = datetime(2023, 7, 1)


def make_plant(
    prices: list[float], freq: str = "h", min_up: int = 3, min_down: int = 2
):
    """A gas plant with marginal cost 40, 400 MW, minimum 200 MW and hot / warm / cold start
    costs of 4000 / 8000 / 12000 (hot up to 4 h off, warm up to 8 h), on a price forecast."""
    index = pd.date_range(START, periods=len(prices), freq=freq)
    forecaster = PowerplantForecaster(
        index,
        fuel_prices={"gas": 20, "co2": 0},
        market_prices={"EOM": pd.Series(prices, index=index)},
    )
    return PowerPlant(
        id="ccgt",
        unit_operator="operator",
        technology="CCGT",
        index=forecaster.index,
        max_power=400,
        min_power=200,
        efficiency=0.5,
        fuel_type="gas",
        emission_factor=0,
        bidding_strategies={},
        forecaster=forecaster,
        ramp_up=400,
        ramp_down=400,
        min_operating_time=min_up,
        min_down_time=min_down,
        hot_start_cost=10,
        warm_start_cost=20,
        cold_start_cost=30,
        downtime_hot_start=4,
        downtime_warm_start=8,
    )


def products(first: int, count: int, freq: timedelta = timedelta(hours=1)):
    return [
        (START + (first + k) * freq, START + (first + k + 1) * freq, None)
        for k in range(count)
    ]


def by_start(bids) -> dict:
    out: dict = {}
    for bid in bids:
        out.setdefault(bid["start_time"], []).append((bid["volume"], bid["price"]))
    return out


def test_registered():
    assert (
        bidding_strategies["powerplant_energy_heuristic_lookahead"]
        is EnergyHeuristicLookaheadStrategy
    )


def test_unit_state_counts_back_to_the_warm_threshold():
    plant = make_plant([50] * 48)
    # off since the start: counted back as far as the warm threshold allows (9 periods)
    assert unit_state(plant, 20) == (False, 9)
    plant.outputs["energy"].data[15:20] = 400
    assert unit_state(plant, 20) == (True, 5)
    assert unit_state(plant, 0)[0]  # before the index: running, as in the framework


def test_plan_starts_for_a_profitable_run_and_prices_the_start_into_it(
    mock_market_config,
):
    prices = [30] * 20 + [100] * 4 + [30] * 24
    plant = make_plant(prices)
    bids = by_start(
        EnergyHeuristicLookaheadStrategy().calculate_bids(
            plant, mock_market_config, products(12, 12)
        )
    )
    # off before the run: a start would have to pay a cold start (12000) over a minimum run of
    # three hours at full output (1200 MWh)
    for k in range(12, 20):
        assert bids[START + timedelta(hours=k)] == [(200, 50.0), (200, 50.0)]
    # the planned run of four hours at full output (1600 MWh) carries the start in every hour
    for k in range(20, 24):
        assert bids[START + timedelta(hours=k)] == [(200, 47.5), (200, 47.5)]


def test_committed_unit_rides_a_short_valley_below_cost(mock_market_config):
    prices = [100] * 14 + [35] * 2 + [100] * 32
    plant = make_plant(prices)
    plant.outputs["energy"].data[:12] = 400
    bids = by_start(
        EnergyHeuristicLookaheadStrategy().calculate_bids(
            plant, mock_market_config, products(12, 12)
        )
    )
    assert bids[START + timedelta(hours=12)] == [(200, 40.0), (200, 40.0)]
    # a loss of 2 x 5 x 200 = 2000 against a hot restart of 4000: the minimum output is offered
    # at marginal cost less the restart cost over the minimum output left in the valley
    assert bids[START + timedelta(hours=14)] == [(200, 30.0), (200, 40.0)]
    assert bids[START + timedelta(hours=15)] == [(200, 20.0), (200, 40.0)]
    assert bids[START + timedelta(hours=16)] == [(200, 40.0), (200, 40.0)]


def test_committed_unit_leaves_a_deep_valley(mock_market_config):
    prices = [100] * 14 + [0] * 8 + [100] * 26
    plant = make_plant(prices)
    plant.outputs["energy"].data[:12] = 400
    bids = by_start(
        EnergyHeuristicLookaheadStrategy().calculate_bids(
            plant, mock_market_config, products(12, 12)
        )
    )
    # a loss of 8 x 40 x 200 = 64000 against a warm restart of 8000: it shuts down at 14:00,
    # offered at marginal cost, cannot start at 15:00 (minimum down time), then offers a start
    assert bids[START + timedelta(hours=14)] == [(200, 40.0), (200, 40.0)]
    assert START + timedelta(hours=15) not in bids
    volume, price = bids[START + timedelta(hours=16)][0]
    assert volume == 200 and math.isclose(price, 40 + 4000 / 1200)
    # the run from 22:00 to the end of the window (26 hours at 400 MW) after 8 hours off
    volume, price = bids[START + timedelta(hours=22)][0]
    assert math.isclose(price, 40 + 8000 / (26 * 400))


def test_minimum_up_time_is_offered_at_the_floor(mock_market_config):
    prices = [10] * 48
    plant = make_plant(prices)
    plant.outputs["energy"].data[11] = 400  # started one hour before the first product
    bids = by_start(
        EnergyHeuristicLookaheadStrategy().calculate_bids(
            plant, mock_market_config, products(12, 4)
        )
    )
    assert bids[START + timedelta(hours=12)][0] == (200, -500.0)
    assert bids[START + timedelta(hours=13)][0] == (200, -500.0)
    # three hours run: free to stop, and nothing in merit within the window
    assert bids[START + timedelta(hours=14)][0] == (200, 40.0)


def test_half_hourly_products_spread_the_start_over_mwh(mock_market_config):
    prices = [30] * 40 + [100] * 8 + [30] * 48
    plant = make_plant(prices, freq="30min", min_up=6, min_down=4)
    half = timedelta(minutes=30)
    bids = by_start(
        EnergyHeuristicLookaheadStrategy().calculate_bids(
            plant, mock_market_config, products(24, 24, half)
        )
    )
    # four hours at 400 MW are 1600 MWh, eight half-hours; off since the start (cold, 12000)
    volume, price = bids[START + 40 * half][0]
    assert math.isclose(price, 40 + 12000 / 1600)


def test_plan_does_not_start_when_the_margin_misses_the_start_cost():
    price = np.array([30.0] * 4 + [41.0] * 2 + [30.0] * 6)
    plan = plan_commitment(
        price=price,
        cost=np.full(12, 40.0),
        qmax=np.full(12, 400.0),
        pmin=200.0,
        hours=1.0,
        state=(False, 10),
        min_up=3,
        min_down=2,
        start_cost=lambda op_time: 12000.0,
    )
    assert not plan.on.any()
    assert plan.merit[4] and plan.merit[5]
    assert (plan.block_of == -1).all()


@pytest.mark.parametrize("lookahead", ["6h", "24h"])
def test_lookahead_parameter(mock_market_config, lookahead):
    strategy = EnergyHeuristicLookaheadStrategy(lookahead=lookahead)
    assert strategy.lookahead == pd.Timedelta(lookahead)
    plant = make_plant([50] * 48)
    assert strategy.calculate_bids(plant, mock_market_config, products(0, 2))
