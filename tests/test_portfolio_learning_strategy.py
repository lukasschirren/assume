# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd

from assume.common.base import LearningConfig
from assume.common.forecaster import PowerplantForecaster, UnitsOperatorForecaster
from assume.common.market_objects import MarketConfig, MarketProduct

try:
    from assume.reinforcement_learning import Learning
    from assume.strategies.portfolio_learning_strategies import (
        PortfolioLearningStrategy,
    )
except ImportError:
    Learning = None
    PortfolioLearningStrategy = None

from assume.strategies.naive_strategies import EnergyNaiveStrategy
from assume.units.powerplant import PowerPlant

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)
MARKET_ID = "EOM"

# Expected observation dimension with defaults foresight=12, nbins=4:
# unique_obs_dim = nbins*2 + 2 = 10;  obs_dim = 3*foresight + unique_obs_dim = 46
EXPECTED_OBS_DIM = 46


@dataclass
class MockUnitsOperator:
    id: str
    units: dict
    forecaster: UnitsOperatorForecaster = None


@pytest.fixture
def portfolio_market_config():
    return MarketConfig(
        market_id=MARKET_ID,
        opening_hours=rr.rrule(rr.HOURLY, dtstart=start, until=end),
        opening_duration=rd(hours=1),
        market_mechanism="pay_as_clear",
        market_products=[MarketProduct(rd(hours=1), 1, rd(hours=1))],
        product_type="energy_eom",
    )


@pytest.fixture
def portfolio_units_operator():
    """
    MockUnitsOperator standing in for a UnitsOperator with 4 PowerPlant units,
    each having a distinct marginal cost (via fuel_price) to populate 4 cost bins.
    """
    index = pd.date_range(start, periods=48, freq="h")
    market_price_forecast = np.linspace(50, 150, 48)
    res_load_forecast = np.linspace(500, 1000, 48)
    units = {}
    for i, fuel_price in enumerate([10, 20, 30, 40]):
        ff = PowerplantForecaster(
            index,
            fuel_prices={"lignite": fuel_price, "co2": 10},
            residual_load={MARKET_ID: res_load_forecast},
            market_prices={MARKET_ID: market_price_forecast},
        )
        unit = PowerPlant(
            id=f"pp_{i}",
            unit_operator="test_portfolio_operator",
            technology="lignite",
            index=index,
            max_power=500 + i * 100,  # 500, 600, 700, 800 MW
            min_power=0,
            efficiency=0.5,
            additional_cost=5,
            bidding_strategies={MARKET_ID: EnergyNaiveStrategy()},
            fuel_type="lignite",
            emission_factor=0.5,
            forecaster=ff,
        )
        units[f"pp_{i}"] = unit

    # operator-level forecaster carrying the same market-wide price and residual
    # load the units see, so portfolio strategies can read them from the operator
    operator_forecaster = UnitsOperatorForecaster(
        index,
        residual_load={MARKET_ID: res_load_forecast},
        market_prices={MARKET_ID: market_price_forecast},
    )
    return MockUnitsOperator(
        id="test_portfolio_operator",
        units=units,
        forecaster=operator_forecaster,
    )


@pytest.fixture
def portfolio_strategy():
    """
    Create a PortfolioLearningStrategy together with an initialised Learning role.

    initialize_policy() must be called explicitly here because
    PortfolioLearningStrategy.get_actions() accesses self.actor.min_output /
    self.actor.max_output even in collect_initial_experience_mode, unlike
    EnergyLearningStrategy which avoids the actor in that phase.
    """
    learning_config = LearningConfig(
        algorithm="matd3",
        learning_mode=True,
        training_episodes=3,
    )
    lr = Learning(learning_config, start, end)
    strategy = PortfolioLearningStrategy(
        learning_role=lr,
        unit_id="test_portfolio_operator",
        nbins=4,
    )
    lr.initialize_policy()
    return strategy, lr


@pytest.mark.require_learning
def test_portfolio_observation_dimensions(portfolio_units_operator, portfolio_strategy):
    """Observation tensor length equals the expected concrete value and decomposes correctly."""
    strategy, _ = portfolio_strategy
    product_index = pd.date_range(start, periods=1, freq="h")

    obs = strategy.create_observation(
        portfolio_units_operator,
        MARKET_ID,
        product_index[0],
        product_index[0] + pd.Timedelta(hours=1),
    )

    assert len(obs) == EXPECTED_OBS_DIM
    assert strategy.obs_dim == EXPECTED_OBS_DIM
    assert (
        strategy.unique_obs_dim + strategy.foresight * strategy.num_timeseries_obs_dim
        == EXPECTED_OBS_DIM
    )


@pytest.mark.require_learning
def test_portfolio_observation_uses_operator_forecaster(
    portfolio_units_operator, portfolio_strategy
):
    """prepare_observations reads the operator-level forecaster: the price /
    residual load scaling bounds reflect the operator forecaster's values, not
    those of any individual unit (which here hold different forecasts)."""
    strategy, _ = portfolio_strategy

    # Give the operator forecaster distinct, known values so the bounds below can
    # only come from it (the units' forecasters span 50..150 / 500..1000).
    portfolio_units_operator.forecaster = UnitsOperatorForecaster(
        portfolio_units_operator.forecaster.index,
        market_prices={MARKET_ID: np.linspace(20, 80, 48)},
        residual_load={MARKET_ID: np.linspace(100, 400, 48)},
    )

    strategy.prepare_observations(portfolio_units_operator, MARKET_ID)

    assert strategy.min_price == pytest.approx(20)
    assert strategy.max_price == pytest.approx(80)
    assert strategy.min_res_load == pytest.approx(100)
    assert strategy.max_res_load == pytest.approx(400)


@pytest.mark.require_learning
def test_portfolio_calculate_bids(
    portfolio_units_operator, portfolio_market_config, portfolio_strategy
):
    """
    calculate_bids returns one flex bid per unit (min_power=0 → no inflexible bids)
    with valid prices, and caches observation + action in the learning role.
    """
    strategy, lr = portfolio_strategy
    n_units = len(portfolio_units_operator.units)

    product_index = pd.date_range(start, periods=1, freq="h")
    product_tuples = [(t, t + pd.Timedelta(hours=1), None) for t in product_index]

    bids = strategy.calculate_bids(
        portfolio_units_operator, portfolio_market_config, product_tuples
    )

    # With min_power=0 for all units, inflexible generation is 0 → only flex bids
    assert len(bids) == n_units

    # Observation and action must be cached for subsequent learning updates
    assert start in lr.all_obs
    assert "test_portfolio_operator" in lr.all_obs[start]
    assert start in lr.all_actions
    assert "test_portfolio_operator" in lr.all_actions[start]


@pytest.mark.require_learning
def test_portfolio_calculate_reward(
    portfolio_units_operator, portfolio_market_config, portfolio_strategy
):
    """Reward and profit are stored in the learning-role cache after clearing feedback."""
    strategy, lr = portfolio_strategy

    product_index = pd.date_range(start, periods=1, freq="h")
    product_tuples = [(t, t + pd.Timedelta(hours=1), None) for t in product_index]

    bids = strategy.calculate_bids(
        portfolio_units_operator, portfolio_market_config, product_tuples
    )
    # Simulate full acceptance at highest bid price
    accepted_price = max(order["price"] for order in bids)
    for order in bids:
        order["accepted_price"] = accepted_price
        order["accepted_volume"] = order["volume"]

    strategy.calculate_reward(
        portfolio_units_operator, portfolio_market_config, orderbook=bids
    )

    assert len(lr.all_rewards) == 1
    assert "test_portfolio_operator" in lr.all_rewards[start]
    assert "test_portfolio_operator" in lr.all_profits[start]


@pytest.mark.require_learning
def test_portfolio_observation_with_plants_dearer_than_any_price(
    portfolio_units_operator, portfolio_strategy
):
    """A plant can cost more than any forecasted price, and the residual load can be zero or
    negative: the observation stays finite and the bids are formed (the cost bins are scaled by
    the dearest plant, the inframarginal share by the installed capacity)."""
    strategy, _ = portfolio_strategy
    # the units cost about 30 to 90; a forecast below all of them, a residual load through zero
    portfolio_units_operator.forecaster = UnitsOperatorForecaster(
        portfolio_units_operator.forecaster.index,
        market_prices={MARKET_ID: np.linspace(5, 25, 48)},
        residual_load={MARKET_ID: np.linspace(-200, 200, 48)},
    )
    product_start = pd.date_range(start, periods=1, freq="h")[0]
    obs = strategy.create_observation(
        portfolio_units_operator,
        MARKET_ID,
        product_start,
        product_start + pd.Timedelta(hours=1),
    )
    assert bool(np.isfinite(obs.cpu().numpy()).all())
    assert strategy.max_cost > strategy.max_price
    bids = strategy.calculate_bids(
        portfolio_units_operator,
        MarketConfig(
            market_id=MARKET_ID,
            opening_hours=rr.rrule(rr.HOURLY, dtstart=start, until=end),
            opening_duration=rd(hours=1),
            market_mechanism="pay_as_clear",
            market_products=[MarketProduct(rd(hours=1), 1, rd(hours=1))],
            product_type="energy_eom",
        ),
        [(product_start, product_start + pd.Timedelta(hours=1), None)],
    )
    assert len(bids) == len(portfolio_units_operator.units)
    for bid in bids:
        unit = portfolio_units_operator.units[bid["unit_id"]]
        assert (
            bid["price"]
            >= unit.calculate_marginal_cost(product_start, unit.max_power) - 1e-9
        )


def make_units_operator(plants):
    """A MockUnitsOperator with one lignite plant per (fuel price, max power) in ``plants``."""
    index = pd.date_range(start, periods=48, freq="h")
    prices = np.linspace(50, 150, 48)
    res_load = np.linspace(500, 1000, 48)
    units = {}
    for i, (fuel_price, max_power) in enumerate(plants):
        units[f"pp_{i}"] = PowerPlant(
            id=f"pp_{i}",
            unit_operator="test_portfolio_operator",
            technology="lignite",
            index=index,
            max_power=max_power,
            min_power=0,
            efficiency=0.5,
            additional_cost=5,
            bidding_strategies={MARKET_ID: EnergyNaiveStrategy()},
            fuel_type="lignite",
            emission_factor=0.5,
            forecaster=PowerplantForecaster(
                index,
                fuel_prices={"lignite": fuel_price, "co2": 10},
                residual_load={MARKET_ID: res_load},
                market_prices={MARKET_ID: prices},
            ),
        )
    return MockUnitsOperator(
        id="test_portfolio_operator",
        units=units,
        forecaster=UnitsOperatorForecaster(
            index,
            residual_load={MARKET_ID: res_load},
            market_prices={MARKET_ID: prices},
        ),
    )


@pytest.mark.require_learning
def test_capacity_cost_bins():
    """Bins hold equal shares of capacity: many small cheap units do not take a bin of their
    own, units of equal cost share a bin, and a single cost fills the last bin."""
    from assume.strategies.portfolio_learning_strategies import capacity_cost_bins

    # eight wind farms at 3 and four gas plants (a GB owner in September 2023)
    costs = [3.0] * 8 + [73.0, 79.0, 83.0, 96.0]
    capacity = [300.0] * 8 + [1325.0, 1958.0, 1477.0, 613.0]
    index, bounds = capacity_cost_bins(costs, capacity, 2)
    assert index.tolist() == [0] * 9 + [1] * 3
    assert bounds.tolist() == [73.0, 96.0]

    index, bounds = capacity_cost_bins([5.0, 5.0, 5.0], [1.0, 2.0, 3.0], 2)
    assert index.tolist() == [1, 1, 1]
    assert bounds.tolist() == [5.0, 5.0]

    index, _ = capacity_cost_bins([10.0, 20.0, 30.0, 40.0], [100.0] * 4, 4)
    assert index.tolist() == [0, 1, 2, 3]


@pytest.mark.require_learning
def test_portfolio_cost_bins_by_capacity(portfolio_market_config):
    """With cost_bins="capacity" the two large dear plants of a portfolio with many small cheap
    ones are bid with different mark-ups; with the stock count quantiles they share one."""
    import torch as th

    # six small plants at one cost, two large dearer ones
    plants = [(10, 100)] * 6 + [(30, 1000), (40, 1000)]
    product_start = pd.date_range(start, periods=1, freq="h")[0]
    product_tuples = [(product_start, product_start + pd.Timedelta(hours=1), None)]

    bid_prices = {}
    for cost_bins in ("count", "capacity"):
        operator = make_units_operator(plants)
        lr = Learning(
            LearningConfig(algorithm="matd3", learning_mode=True, training_episodes=3),
            start,
            end,
        )
        strategy = PortfolioLearningStrategy(
            learning_role=lr,
            unit_id="test_portfolio_operator",
            nbins=2,
            max_markup=3,
            cost_bins=cost_bins,
        )
        lr.initialize_policy()
        # the cheaper bin at cost, the dearer one at three times the cost
        strategy.get_actions = lambda obs: (th.tensor([-1.0, 1.0]), th.zeros(2))
        bids = strategy.calculate_bids(
            operator, portfolio_market_config, product_tuples
        )
        bid_prices[cost_bins] = {
            bid["unit_id"]: bid["price"]
            / operator.units[bid["unit_id"]].calculate_marginal_cost(
                product_start, operator.units[bid["unit_id"]].max_power
            )
            for bid in bids
        }

    assert bid_prices["count"]["pp_6"] == pytest.approx(3.0)
    assert bid_prices["count"]["pp_7"] == pytest.approx(3.0)
    assert bid_prices["capacity"]["pp_0"] == pytest.approx(1.0)
    assert bid_prices["capacity"]["pp_6"] == pytest.approx(1.0)
    assert bid_prices["capacity"]["pp_7"] == pytest.approx(3.0)


@pytest.mark.require_learning
def test_portfolio_cost_bins_unknown():
    with pytest.raises(ValueError, match="cost_bins"):
        PortfolioLearningStrategy(
            learning_role=None, unit_id="test_portfolio_operator", cost_bins="volume"
        )


@pytest.mark.require_learning
def test_portfolio_capacity_bins_fewer_units_than_bins(portfolio_market_config):
    """With cost_bins="capacity" an operator of a single plant can bid with four bins: the plant
    takes the bin that holds its capacity, the others stay empty."""
    import torch as th

    operator = make_units_operator([(30, 1000)])
    lr = Learning(
        LearningConfig(algorithm="matd3", learning_mode=True, training_episodes=3),
        start,
        end,
    )
    strategy = PortfolioLearningStrategy(
        learning_role=lr,
        unit_id="test_portfolio_operator",
        nbins=4,
        max_markup=3,
        cost_bins="capacity",
    )
    lr.initialize_policy()
    strategy.get_actions = lambda obs: (th.tensor([-1.0, -1.0, 1.0, -1.0]), th.zeros(4))
    product_start = pd.date_range(start, periods=1, freq="h")[0]
    bids = strategy.calculate_bids(
        operator,
        portfolio_market_config,
        [(product_start, product_start + pd.Timedelta(hours=1), None)],
    )
    unit = operator.units["pp_0"]
    assert strategy.unit_bins == {"pp_0": 2}
    assert len(bids) == 1
    assert bids[0]["price"] == pytest.approx(
        3 * unit.calculate_marginal_cost(product_start, unit.max_power)
    )
