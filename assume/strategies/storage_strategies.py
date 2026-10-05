# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import timedelta

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition

from assume.common.base import MinMaxChargeStrategy, SupportsMinMaxCharge
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.common.utils import get_supported_solver_pyomo
from assume.strategies.flexable_storage import calculate_storage_reward

logger = logging.getLogger(__name__)

EPS = 1e-6
NO_TIME_LIMIT = 1e9


class StorageEnergyOptimizationScheduleStrategy(MinMaxChargeStrategy):
    """
    A storage strategy that plans its charging and discharging over the products of an auction
    by an optimisation on the price forecast, and bids that plan.

    The plan takes the price forecast of the market as given, so the unit is a price taker with
    perfect foresight over the horizon of the auction. It starts from the state of charge at the
    first product, ends at the initial state of charge of the unit, keeps within the power,
    capacity and efficiencies of the unit, charges or discharges in a step but not both, and
    includes what the unit has already sold or bought for the horizon on other markets. It
    maximises the revenue of discharging less the cost of charging, each including the
    additional cost of the unit. The plan is a linear programme unless it would charge and
    discharge in one step (which pays at negative prices); then that choice is made binary and
    solved within ``mip_time_limit`` seconds. If the price forecast has missing values within
    the horizon, the unit does not bid in that auction.

    Charging is bid at the maximum price of the market, so the energy the plan relies on is bought
    whatever the price turns out to be. Discharging is bid at the price that recovers the cost of
    the dearest energy the plan buys, so stored energy is never sold at a loss: (highest forecast
    price at which the plan charges + cost of charging) / (efficiency of charging x efficiency of
    discharging) + cost of discharging. If the plan does not charge, the energy already in the store
    is sold at any price above the cost of discharging.

    Methods
    -------
    """

    def __init__(self, *args, mip_time_limit: float = 10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.solver = SolverFactory(get_supported_solver_pyomo())
        self.mip_time_limit = mip_time_limit

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.

        Args:
            unit (SupportsMinMaxCharge): The unit that is dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): List of product tuples.

        Returns:
            Orderbook: Bids containing start_time, end_time, only_hours, price, volume.
        """
        start = product_tuples[0][0]
        starts = [product[0] for product in product_tuples]
        hours = np.array(
            [
                (product[1] - product[0]) / timedelta(hours=1)
                for product in product_tuples
            ]
        )
        price_forecast = unit.forecaster.price[market_config.market_id]
        prices = np.array([price_forecast.at[t] for t in starts], dtype=float)
        if not np.isfinite(prices).all():
            # a plan cannot be made on a missing price, and the solver does not return on one
            logger.warning(
                "unit %s: the price forecast of market %s has missing values from %s, "
                "it does not bid",
                unit.id,
                market_config.market_id,
                start,
            )
            return []
        committed = np.array([unit.outputs["energy"].at[t] for t in starts])
        soc_start = unit.outputs["soc"].at[start]

        charge, discharge = self.plan(unit, prices, hours, committed, soc_start)

        charging = charge > 1e-9
        if charging.any():
            break_even = (prices[charging].max() + unit.additional_cost_charge) / (
                unit.efficiency_charge * unit.efficiency_discharge
            )
            break_even += unit.additional_cost_discharge
        else:
            break_even = unit.additional_cost_discharge

        bids = []
        for product, charge_power, discharge_power in zip(
            product_tuples, charge, discharge
        ):
            volume = discharge_power - charge_power
            if volume > 1e-9:
                price = min(break_even, market_config.maximum_bid_price)
            elif volume < -1e-9:
                price = market_config.maximum_bid_price
            else:
                continue
            bids.append(
                {
                    "start_time": product[0],
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": price,
                    "volume": volume,
                    "node": unit.node,
                }
            )

        return bids

    def plan(
        self,
        unit: SupportsMinMaxCharge,
        prices: np.ndarray,
        hours: np.ndarray,
        committed: np.ndarray,
        soc_start: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Plans the additional charging and discharging power of the unit over the horizon.

        Args:
            unit (SupportsMinMaxCharge): The unit.
            prices (np.ndarray): The price forecast per product.
            hours (np.ndarray): The duration of each product in hours.
            committed (np.ndarray): The power already sold (positive) or bought (negative) per product.
            soc_start (float): The state of charge at the start of the horizon.

        Returns:
            tuple[np.ndarray, np.ndarray]: The charging and the discharging power per product, both positive, in MW.
        """
        steps = range(len(prices))
        committed_discharge = np.maximum(committed, 0.0)
        committed_charge = np.maximum(-committed, 0.0)
        max_charge = np.maximum(-unit.max_power_charge - committed_charge, 0.0)
        max_discharge = np.maximum(unit.max_power_discharge - committed_discharge, 0.0)
        min_energy = unit.min_soc * unit.capacity
        max_energy = unit.max_soc * unit.capacity
        energy_start = soc_start * unit.capacity
        energy_end = unit.initial_soc * unit.capacity

        model = pyo.ConcreteModel()
        model.charge = pyo.Var(
            steps, bounds=lambda m, t: (0.0, max_charge[t]), domain=pyo.NonNegativeReals
        )
        model.discharge = pyo.Var(
            steps,
            bounds=lambda m, t: (0.0, max_discharge[t]),
            domain=pyo.NonNegativeReals,
        )
        model.energy = pyo.Var(
            steps, bounds=(min(min_energy, energy_start), max(max_energy, energy_start))
        )
        # a step either charges or discharges: both at once would turn the round-trip loss
        # into a way of being paid for consumption whenever the price is negative. The plan is
        # first made with this choice relaxed, which is a linear programme; only if it then
        # charges and discharges in one step is the choice made binary.
        model.charging = pyo.Var(steps, domain=pyo.UnitInterval)
        model.charge_or_discharge = pyo.ConstraintList()
        for t in steps:
            model.charge_or_discharge.add(
                model.charge[t] <= max_charge[t] * model.charging[t]
            )
            model.charge_or_discharge.add(
                model.discharge[t] <= max_discharge[t] * (1 - model.charging[t])
            )

        def balance(m, t):
            previous = energy_start if t == 0 else m.energy[t - 1]
            stored = (
                (m.charge[t] + committed_charge[t]) * hours[t] * unit.efficiency_charge
            )
            taken = (m.discharge[t] + committed_discharge[t]) * hours[t]
            return m.energy[t] == previous + stored - taken / unit.efficiency_discharge

        model.balance = pyo.Constraint(steps, rule=balance)
        model.end = pyo.Constraint(expr=model.energy[len(prices) - 1] == energy_end)
        model.objective = pyo.Objective(
            expr=sum(
                hours[t]
                * (
                    (prices[t] - unit.additional_cost_discharge) * model.discharge[t]
                    - (prices[t] + unit.additional_cost_charge) * model.charge[t]
                )
                for t in steps
            ),
            sense=pyo.maximize,
        )

        if not self._solve(model):
            # the end state cannot be reached within the horizon: plan without it
            model.end.deactivate()
            if not self._solve(model):
                logger.warning(
                    "unit %s: no feasible storage plan, it does not bid", unit.id
                )
                return np.zeros(len(prices)), np.zeros(len(prices))

        charge = np.array([pyo.value(model.charge[t]) for t in steps])
        discharge = np.array([pyo.value(model.discharge[t]) for t in steps])
        if ((charge > EPS) & (discharge > EPS)).any():
            for t in steps:
                model.charging[t].domain = pyo.Binary
            if self._solve(model, time_limit=self.mip_time_limit):
                charge = np.array([pyo.value(model.charge[t]) for t in steps])
                discharge = np.array([pyo.value(model.discharge[t]) for t in steps])
            else:
                logger.warning(
                    "unit %s: the plan charges and discharges in one step and could not be "
                    "resolved in time, it is bid as it is",
                    unit.id,
                )
        return np.clip(charge, 0.0, None), np.clip(discharge, 0.0, None)

    def _solve(self, model: pyo.ConcreteModel, time_limit: float | None = None) -> bool:
        """
        Solves the model and returns whether a solution is loaded. A time limit is given for
        the binary programme; a feasible solution found within it counts.
        """
        options = {}
        if "highs" in getattr(self.solver, "name", ""):
            # the solver keeps its options from one call to the next, so the limit is always set
            options["time_limit"] = NO_TIME_LIMIT if time_limit is None else time_limit
        try:
            result = self.solver.solve(model, options=options)
        except RuntimeError:
            # the appsi interface raises when there is no feasible solution to load
            return False
        return result.solver.termination_condition in (
            TerminationCondition.optimal,
            TerminationCondition.maxTimeLimit,
        )

    def calculate_reward(
        self,
        unit: SupportsMinMaxCharge,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ):
        """
        Calculates the reward (costs and profit).

        Args:
            unit (SupportsMinMaxCharge): The unit to calculate reward for.
            marketconfig (MarketConfig): The market configuration.
            orderbook (Orderbook): The orderbook.
        """
        calculate_storage_reward(unit, marketconfig, orderbook)
