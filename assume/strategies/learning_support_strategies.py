# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from datetime import datetime

import numpy as np

from assume.common.base import SupportsMinMax
from assume.strategies.learning_strategies import EnergyLearningStrategy
from assume.strategies.portfolio_learning_strategies import PortfolioLearningStrategy
from assume.strategies.support_strategies import (
    EnergyNaiveSupportStrategy,
    support_bid_price,
)


def contract_floor(unit, start: datetime, marginal_cost: float) -> float:
    """
    The lowest price at which generating pays for ``unit`` in the time step starting at
    ``start``: its marginal cost less what its support contract pays per MWh, or the marginal
    cost itself without a contract.
    """
    if not getattr(unit, "support_scheme", ""):
        return marginal_cost
    reference = EnergyNaiveSupportStrategy.reference_price(unit)
    return float(
        support_bid_price(
            marginal_cost,
            unit.support_scheme,
            unit.support_value,
            unit.support_neg_price_rule,
            None if reference is None else reference.at[start],
        )
    )


def contract_income(unit, start: datetime, price: float, volume: float) -> float:
    """The market income of ``unit`` plus what its support contract pays, if it has one, less
    what a levy on its revenue takes, if there is one."""
    income = price * volume
    if getattr(unit, "support_scheme", ""):
        income += unit.support_payment(start, price, volume)
    if getattr(unit, "levy_rate", 0.0):
        income += unit.levy_payment(start, price, volume)
    return income


class EnergyLearningSupportStrategy(EnergyLearningStrategy):
    """
    The reinforcement learning strategy of ``EnergyLearningStrategy`` for a power plant with a
    support contract (``support_scheme`` on the unit, see ``powerplant_energy_naive_support``).

    Two things differ from the strategy without a contract. The income the reward is made of
    includes what the contract pays for the volume sold (``PowerPlant.support_payment``), so
    the agent is rewarded for what it earns in all and not for the market price alone. And the
    cost the agent observes, around which it explores at first, is the lowest price at which
    generating pays under the contract, the price the naive support strategy bids, instead of
    the marginal cost. A plant without a contract behaves exactly as under the parent strategy.

    Methods
    -------
    """

    def unit_income(
        self, unit: SupportsMinMax, start: datetime, price: float, volume: float
    ) -> float:
        """
        The income of the unit for ``volume`` MW sold at ``price`` in the time step starting at
        ``start``: the market price and the payment of the support contract.
        """
        return contract_income(unit, start, price, volume)

    def get_individual_observations(
        self, unit: SupportsMinMax, start: datetime, end: datetime
    ):
        """
        The unit-specific observations: the last dispatched volume, scaled by the maximum power,
        and the lowest price at which generating pays under the contract, scaled by the maximum
        bid price.
        """
        current_volume = unit.get_output_before(start)
        marginal_cost = unit.calculate_marginal_cost(start, current_volume)
        floor = contract_floor(unit, start, marginal_cost)
        return np.array([current_volume / unit.max_power, floor / self.max_bid_price])


class PortfolioLearningSupportStrategy(PortfolioLearningStrategy):
    """
    The portfolio learning strategy of ``PortfolioLearningStrategy`` for an operator whose plants
    may hold support contracts (``support_scheme`` on the unit).

    A plant's inflexible capacity is bid at the lowest price at which generating pays under its
    contract, its flexible capacity at that floor plus the mark-up the agent chooses, which is
    the same amount above the floor as the parent strategy puts above the marginal cost
    (marginal cost times the mark-up factor less one). The profit and the competitive profit the
    reward is made of include the contract's payments, and the competitive benchmark dispatches
    a plant whenever the forecast price is above its floor. A plant without a contract bids and
    earns exactly as under the parent strategy.

    Methods
    -------
    """

    def unit_floor(self, unit, start: datetime, marginal_cost: float) -> float:
        return contract_floor(unit, start, marginal_cost)

    def unit_bid(
        self, unit, start: datetime, marginal_cost: float, markup: float
    ) -> float:
        return contract_floor(unit, start, marginal_cost) + marginal_cost * (
            markup - 1.0
        )

    def unit_income(self, unit, start: datetime, price: float, volume: float) -> float:
        return contract_income(unit, start, price, volume)
