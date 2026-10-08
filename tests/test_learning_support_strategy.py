# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from assume.common.base import LearningConfig
from assume.common.forecaster import PowerplantForecaster
from assume.strategies import bidding_strategies
from assume.units.powerplant import PowerPlant

try:
    from assume.reinforcement_learning.learning_role import Learning
    from assume.strategies.learning_strategies import EnergyLearningStrategy
    from assume.strategies.learning_support_strategies import (
        EnergyLearningSupportStrategy,
    )
except ImportError:
    Learning = None

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)
MAX_BID_PRICE = 100.0


def learning_role():
    config = LearningConfig(
        train_freq="1h",
        algorithm="matd3",
        actor_architecture="mlp",
        learning_mode=True,
        evaluation_mode=False,
        training_episodes=3,
        episodes_collecting_initial_experience=1,
        continue_learning=False,
        trained_policies_save_path=None,
        max_bid_price=MAX_BID_PRICE,
    )
    return Learning(config, start=start, end=end)


def make_plant(strategy_class, role, **support) -> PowerPlant:
    index = pd.date_range(start, end, freq="h")
    forecaster = PowerplantForecaster(
        index,
        availability=1,
        fuel_prices={"reference": 110.0},
        market_prices={"EOM": np.linspace(40, 60, len(index))},
        residual_load={"EOM": np.linspace(500, 1000, len(index))},
    )
    strategy = strategy_class(unit_id="pp", learning_role=role)
    return PowerPlant(
        id="pp",
        unit_operator="test_operator",
        technology="wind_onshore",
        bidding_strategies={"EOM": strategy},
        forecaster=forecaster,
        max_power=100,
        additional_cost=3.0,
        **support,
    )


def accepted(price: float, volume: float) -> list[dict]:
    return [
        {
            "start_time": start,
            "end_time": start + timedelta(hours=1),
            "only_hours": None,
            "price": price,
            "volume": 100,
            "accepted_price": price,
            "accepted_volume": volume,
        }
    ]


@pytest.mark.require_learning
def test_strategy_is_registered():
    assert (
        bidding_strategies["powerplant_energy_learning_support"]
        is EnergyLearningSupportStrategy
    )


@pytest.mark.require_learning
def test_reward_includes_the_contract_payment(mock_market_config):
    role = learning_role()
    # the same plant with the stock learning strategy and with the support-aware one:
    # the premium of 90 per MWh is income only for the latter
    plain = make_plant(EnergyLearningStrategy, role)
    supported = make_plant(
        EnergyLearningSupportStrategy, role, support_scheme="premium", support_value=90
    )
    for unit in (plain, supported):
        unit.set_dispatch_plan(mock_market_config, accepted(50.0, 100))
        unit.calculate_cashflow_and_reward(mock_market_config, accepted(50.0, 100))
    assert plain.outputs["profit"].at[start] == pytest.approx((50 - 3) * 100)
    assert supported.outputs["profit"].at[start] == pytest.approx((50 + 90 - 3) * 100)
    # the market cashflow is the same, the contract is booked beside it
    assert supported.outputs["energy_cashflow"].at[start] == pytest.approx(5000)
    assert supported.outputs["support_cashflow"].at[start] == pytest.approx(9000)


@pytest.mark.require_learning
def test_without_a_contract_the_strategies_agree(mock_market_config):
    role = learning_role()
    plain = make_plant(EnergyLearningStrategy, role)
    supported = make_plant(EnergyLearningSupportStrategy, role)
    for unit in (plain, supported):
        unit.set_dispatch_plan(mock_market_config, accepted(50.0, 60))
        unit.calculate_cashflow_and_reward(mock_market_config, accepted(50.0, 60))
    assert plain.outputs["profit"].at[start] == supported.outputs["profit"].at[start]
    strategies = (plain.bidding_strategies["EOM"], supported.bidding_strategies["EOM"])
    observations = [
        s.get_individual_observations(u, start, start + timedelta(hours=1))
        for s, u in zip(strategies, (plain, supported))
    ]
    np.testing.assert_allclose(observations[0], observations[1])


@pytest.mark.require_learning
def test_agent_observes_the_price_down_to_which_generating_pays(mock_market_config):
    role = learning_role()
    cfd = make_plant(
        EnergyLearningSupportStrategy,
        role,
        support_scheme="cfd",
        support_value=130,
        support_reference="reference",
    )
    observation = cfd.bidding_strategies["EOM"].get_individual_observations(
        cfd, start, start + timedelta(hours=1)
    )
    # marginal cost 3 less the top-up of 130 - 110, scaled by the maximum bid price
    assert observation[-1] == pytest.approx((3 - 20) / MAX_BID_PRICE)
    assert observation[0] == 0.0  # nothing dispatched before the start


@pytest.mark.require_learning
def test_portfolio_strategy_bids_supported_plants_from_their_contract_floor(
    mock_market_config,
):
    from dataclasses import dataclass

    from assume.common.forecaster import UnitsOperatorForecaster
    from assume.strategies.learning_support_strategies import (
        PortfolioLearningSupportStrategy,
    )
    from assume.strategies.naive_strategies import EnergyNaiveStrategy

    @dataclass
    class Operator:
        id: str
        units: dict
        forecaster: UnitsOperatorForecaster

    index = pd.date_range(start, end, freq="h")
    role = learning_role()
    strategy = PortfolioLearningSupportStrategy(
        unit_id="owner", learning_role=role, nbins=2
    )
    role.initialize_policy()
    # a plant without a contract and one with a premium of 90, both at a marginal cost of 3
    units = {}
    for name, support in (
        ("plain", {}),
        ("ro", {"support_scheme": "premium", "support_value": 90}),
    ):
        forecaster = PowerplantForecaster(
            index,
            availability=1,
            market_prices={"EOM": 50.0},
            residual_load={"EOM": 500.0},
        )
        units[name] = PowerPlant(
            id=name,
            unit_operator="owner",
            technology="wind_onshore",
            bidding_strategies={"EOM": EnergyNaiveStrategy()},
            forecaster=forecaster,
            max_power=100,
            additional_cost=3.0,
            **support,
        )
    operator = Operator(
        "owner",
        units,
        UnitsOperatorForecaster(
            index, market_prices={"EOM": 50.0}, residual_load={"EOM": 500.0}
        ),
    )
    assert strategy.unit_floor(units["plain"], start, 3.0) == 3.0
    assert strategy.unit_floor(units["ro"], start, 3.0) == pytest.approx(-87.0)
    # the mark-up sits the same amount above the floor as above the marginal cost
    assert strategy.unit_bid(units["plain"], start, 3.0, 2.0) == pytest.approx(6.0)
    assert strategy.unit_bid(units["ro"], start, 3.0, 2.0) == pytest.approx(-84.0)
    # the income counts the contract
    assert strategy.unit_income(units["ro"], start, 10.0, 100) == pytest.approx(
        (10 + 90) * 100
    )
    assert strategy.unit_income(units["plain"], start, 10.0, 100) == pytest.approx(
        10 * 100
    )
    from dateutil import rrule as rr
    from dateutil.relativedelta import relativedelta as rd

    from assume.common.market_objects import MarketConfig, MarketProduct

    market_config = MarketConfig(
        market_id="EOM",
        opening_hours=rr.rrule(rr.HOURLY, dtstart=start, until=end),
        opening_duration=rd(hours=1),
        market_mechanism="pay_as_clear",
        market_products=[MarketProduct(rd(hours=1), 1, rd(hours=1))],
        product_type="energy",
    )
    strategy.prepare_observations(operator, "EOM")
    bids = strategy.calculate_bids(
        operator, market_config, [(start, start + timedelta(hours=1), None)]
    )
    prices = {bid["unit_id"]: bid["price"] for bid in bids if bid["volume"] > 0}
    assert prices["plain"] >= 3.0 - 1e-9
    assert -87.0 - 1e-9 <= prices["ro"] <= 3.0 + 1e-9
    assert prices["ro"] - (-87.0) == pytest.approx(prices["plain"] - 3.0)
