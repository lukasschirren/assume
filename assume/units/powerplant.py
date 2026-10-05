# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np

from assume.common.base import MinMaxStrategy, SupportsMinMax
from assume.common.exceptions import ValidationError
from assume.common.fast_pandas import FastSeries
from assume.common.forecaster import PowerplantForecaster
from assume.common.market_objects import Orderbook

logger = logging.getLogger(__name__)


class PowerPlant(SupportsMinMax):
    """
    A class for a power plant unit.

    Args:
        id (str): The ID of the storage unit.
        unit_operator (str): The operator of the unit.
        technology (str): The technology of the unit.
        bidding_strategies (dict[str, MinMaxChargeStrategy]): The bidding strategies of the unit.
        forecaster (Forecaster): A forecaster used to get key variables such as fuel or electricity prices.
        max_power (float): The maximum power output capacity of the power plant in MW.
        min_power (float, optional): The minimum power output capacity of the power plant in MW. Defaults to 0.0 MW.
        efficiency (float, optional): The efficiency of the power plant in converting fuel to electricity. Defaults to 1.0.
        additional_cost (float, optional): Additional costs associated with power generation, in EUR/MWh. Defaults to 0.
        partial_load_eff (bool, optional): Does the efficiency vary at part loads? Defaults to False.
        fuel_type (str, optional): The type of fuel used by the power plant for power generation. Defaults to "others".
        emission_factor (float, optional): The emission factor associated with the power plant's fuel type (tons of CO2 emissions per MWh). Defaults to 0.0.
        ramp_up (Union[float, None], optional): The ramp-up rate of the power plant, indicating how quickly it can increase power output. Defaults to None.
        ramp_down (Union[float, None], optional): The ramp-down rate of the power plant, indicating how quickly it can decrease power output. Defaults to None.
        hot_start_cost (float, optional): The cost of a hot start, where the power plant is restarted after a recent shutdown. Defaults to 0.
        warm_start_cost (float, optional): The cost of a warm start, where the power plant is restarted after a moderate downtime. Defaults to 0.
        cold_start_cost (float, optional): The cost of a cold start, where the power plant is restarted after a prolonged downtime. Defaults to 0.
        min_operating_time (float, optional): The minimum duration that the power plant must operate once started, in hours. Defaults to 0.
        min_down_time (float, optional): The minimum downtime required after a shutdown before the power plant can be restarted, in hours. Defaults to 0.
        downtime_hot_start (int, optional): The downtime required after a hot start before the power plant can be restarted, in hours. Defaults to 8.
        downtime_warm_start (int, optional): The downtime required after a warm start before the power plant can be restarted, in hours. Defaults to 48.
        max_heat_extraction (float, optional): The maximum amount of heat that the power plant can extract for external use, in some suitable unit. Defaults to 0.
        location (Tuple[float, float], optional): The geographical coordinates (latitude and longitude) of the power plant's location. Defaults to (0.0, 0.0).
        node (str, optional): The identifier of the electrical bus or network node to which the power plant is connected. Defaults to "node0".
        support_scheme (str, optional): The support contract of the power plant, read by support-aware bidding strategies: "premium" (a fixed payment per MWh generated) or "cfd" (a contract for differences). Defaults to "" (no contract).
        support_value (float, optional): The premium or the strike price of the support contract, per MWh. Defaults to 0.
        support_neg_price_rule (str, optional): When a "cfd" stops paying at negative prices: "any_hour" (in every negative period) or "six_hour" (only in a negative spell of six hours or more). Defaults to "".
        support_reference (str, optional): The price series (a column of the fuel prices) a "cfd" is settled against. Defaults to "", the market price the power plant itself earns.
        levy_rate (float, optional): The share of the receipts above ``levy_benchmark`` that a levy on generation revenue takes, such as the GB electricity generator levy. Defaults to 0 (no levy).
        levy_benchmark (float, optional): The price per MWh above which the levy is taken. Defaults to 0.
        **kwargs (dict, optional): Additional keyword arguments to be passed to the base class. Defaults to {}.
    """

    def __init__(
        self,
        id: str,
        unit_operator: str,
        technology: str,
        bidding_strategies: dict[str, MinMaxStrategy],
        forecaster: PowerplantForecaster,
        max_power: float,
        min_power: float = 0.0,
        efficiency: float = 1.0,
        additional_cost: float = 0.0,
        partial_load_eff: bool = False,
        fuel_type: str = "others",
        emission_factor: float = 0.0,
        ramp_up: float | None = None,
        ramp_down: float | None = None,
        hot_start_cost: float = 0,
        warm_start_cost: float = 0,
        cold_start_cost: float = 0,
        min_operating_time: int = 1,  # hours
        min_down_time: int = 1,  # hours
        downtime_hot_start: int = 0,  # hours
        downtime_warm_start: int = 0,  # hours
        max_heat_extraction: float = 0,
        location: tuple[float, float] = (0.0, 0.0),
        node: str = "node0",
        support_scheme: str = "",
        support_value: float = 0.0,
        support_neg_price_rule: str = "",
        support_reference: str = "",
        levy_rate: float = 0.0,
        levy_benchmark: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology=technology,
            bidding_strategies=bidding_strategies,
            forecaster=forecaster,
            node=node,
            location=location,
            **kwargs,
        )

        # the scenario loaders fill empty cells with 0, which is "no contract" here
        self.support_scheme = str(support_scheme).lower() if support_scheme else ""
        self.support_value = float(support_value or 0.0)
        self.support_neg_price_rule = (
            str(support_neg_price_rule) if support_neg_price_rule else ""
        )
        self.support_reference = str(support_reference) if support_reference else ""
        # a levy on the receipts above a benchmark price, such as the GB electricity generator
        # levy: the share taken, and the price per MWh above which it is taken
        self.levy_rate = float(levy_rate or 0.0)
        self.levy_benchmark = float(levy_benchmark or 0.0)
        if not 0.0 <= self.levy_rate < 1.0:
            raise ValidationError(
                message=f"{levy_rate=} must be in [0, 1) for unit {self.id}",
                id=self.id,
                field="levy_rate",
            )
        if self.support_scheme not in ("", "premium", "cfd"):
            raise ValidationError(
                message=f"{support_scheme=} must be 'premium', 'cfd' or empty for unit {self.id}",
                id=self.id,
                field="support_scheme",
            )

        if not isinstance(forecaster, PowerplantForecaster):
            raise ValueError(
                f"forecaster must be of type {PowerplantForecaster.__name__}"
            )

        if min_power < 0:
            raise ValidationError(
                message=f"{min_power=} must be >= 0 for unit {self.id}",
                id=self.id,
                field="min_power",
            )
        if max_power < 0:
            raise ValidationError(
                message=f"{max_power=} must be >= 0 for unit {self.id}",
                id=self.id,
                field="max_power",
            )
        if min_power > max_power:
            raise ValidationError(
                message=f"{min_power=} must be <= {max_power=} for unit {self.id}",
                id=self.id,
                field="min_power",
            )
        if efficiency < 0:
            raise ValidationError(
                message=f"{efficiency=} must be >= 0 for unit {self.id}",
                id=self.id,
                field="efficiency",
            )
        if emission_factor < 0:
            raise ValidationError(
                message=f"{emission_factor=} must be >= 0 for unit {self.id}",
                id=self.id,
                field="emission_factor",
            )
        if max_heat_extraction < 0:
            raise ValidationError(
                message=f"{max_heat_extraction=} must be >= 0 for unit {self.id}",
                id=self.id,
                field="max_heat_extraction",
            )
        self.max_power = max_power
        self.min_power = min_power
        self.efficiency = efficiency
        self.additional_cost = additional_cost
        self.partial_load_eff = partial_load_eff
        self.fuel_type = fuel_type
        self.emission_factor = emission_factor
        self.max_heat_extraction = max_heat_extraction
        self.hot_start_cost = hot_start_cost * max_power
        self.warm_start_cost = warm_start_cost * max_power
        self.cold_start_cost = cold_start_cost * max_power

        # check ramping enabled
        self.ramp_down = None if ramp_down == 0 else ramp_down
        self.ramp_up = None if ramp_up == 0 else ramp_up

        if min_operating_time < 0:
            raise ValidationError(
                message=f"{min_operating_time=} must be > 0 for unit {self.id}",
                id=self.id,
                field="min_operating_time",
            )
        self.min_operating_time = min_operating_time
        if min_down_time < 0:
            raise ValidationError(
                message=f"{min_down_time=} must be > 0 for unit {self.id}",
                id=self.id,
                field="min_down_time",
            )
        self.min_down_time = min_down_time
        self.downtime_hot_start = downtime_hot_start / (
            self.index.freq / timedelta(hours=1)
        )
        self.downtime_warm_start = downtime_warm_start / (
            self.index.freq / timedelta(hours=1)
        )

        self.marginal_cost = self.calc_simple_marginal_cost()

    def execute_current_dispatch(
        self,
        start: datetime,
        end: datetime,
    ) -> np.ndarray:
        """
        Executes the current dispatch of the unit based on the provided timestamps.

        The dispatch is only executed, if it is in the constraints given by the unit.
        Returns the volume of the unit within the given time range.

        Args:
            start (pandas.Timestamp): The start time of the dispatch.
            end (pandas.Timestamp): The end time of the dispatch.

        Returns:
            np.ndarray: The volume of the unit within the given time range.
        """
        start = max(start, self.index[0])

        max_power_values = self.forecaster.availability.loc[start:end] * self.max_power

        if (
            self.ramp_up is None
            and self.ramp_down is None
            and type(self).calculate_ramp is SupportsMinMax.calculate_ramp
        ):
            # Without ramp limits calculate_ramp returns the power it is given, so a time step
            # does not depend on the one before and all of them are limited at once.
            # fmin and fmax keep a value when it is compared with NaN, as min and max do.
            energy = self.outputs["energy"].loc[start:end]
            running = energy > 0
            energy[running] = np.fmax(
                np.fmin(energy[running], max_power_values[running]), self.min_power
            )
            return self.outputs["energy"].loc[start:end]

        for t, max_power in zip(self.index[start:end], max_power_values):
            current_power = self.outputs["energy"].at[t]
            previous_power = self.get_output_before(t)
            op_time = self.get_operation_time(t)

            current_power = self.calculate_ramp(op_time, previous_power, current_power)

            if current_power > 0:
                current_power = min(current_power, max_power)
                current_power = max(current_power, self.min_power)

            self.outputs["energy"].at[t] = current_power

        return self.outputs["energy"].loc[start:end]

    def calculate_generation_cost(
        self, start: datetime, end: datetime, product_type: str
    ) -> None:
        """
        Calculates the generation cost for a specific product type within the given time range.

        The marginal cost of a power plant is a series which does not depend on its power output,
        so the cost of all time steps is calculated at once. If the marginal cost is calculated
        in another way, each time step is calculated on its own as for any unit.

        Args:
            start (datetime.datetime): The start time for the calculation.
            end (datetime.datetime): The end time for the calculation.
            product_type (str): The type of product for which the generation cost is to be calculated.
        """
        marginal_cost_is_series = (
            self.marginal_cost is not None
            and len(self.marginal_cost) > 1
            and self.marginal_cost.index is self.index
            and getattr(self.calculate_marginal_cost, "__func__", None)
            is PowerPlant.calculate_marginal_cost
        )
        if not marginal_cost_is_series:
            return super().calculate_generation_cost(start, end, product_type)

        if start not in self.index:
            start = self.index[0]

        product_data = self.outputs[product_type].loc[start:end]
        marginal_costs = self.marginal_cost.loc[start:end]
        self.outputs[f"{product_type}_generation_costs"].loc[start:end] = np.abs(
            marginal_costs * product_data
        )

    def calc_simple_marginal_cost(
        self,
    ) -> FastSeries:
        """
        Calculates the marginal cost of the unit (simple method) and returns the marginal cost of the unit.

        Returns:
            pandas.Series: The marginal cost of the unit.
        """
        fuel_price = self.forecaster.get_price(self.fuel_type)
        co2_price = self.forecaster.get_price("co2")
        marginal_cost = (
            fuel_price / self.efficiency
            + co2_price * self.emission_factor / self.efficiency
            + self.additional_cost
        )

        return marginal_cost

    @lru_cache(maxsize=256)
    def calc_marginal_cost_with_partial_eff(
        self,
        power_output: float,
        timestep: datetime,
    ) -> float:
        """
        Calculates the marginal cost of the unit based on power output and timestamp, considering partial efficiency.
        Returns the marginal cost of the unit.

        Args:
            power_output (float): The power output of the unit.
            timestep (datetime.datetime): The timestamp of the unit.

        Returns:
            float: The marginal cost of the unit at the given timestamp.
        """
        fuel_price = self.forecaster.get_price(self.fuel_type).at[timestep]

        capacity_ratio = power_output / self.max_power

        if self.fuel_type in ["lignite", "hard coal"]:
            eta_loss = (
                0.095859 * (capacity_ratio**4)
                - 0.356010 * (capacity_ratio**3)
                + 0.532948 * (capacity_ratio**2)
                - 0.447059 * capacity_ratio
                + 0.174262
            )

        elif self.fuel_type == "combined cycle gas turbine":
            eta_loss = (
                0.178749 * (capacity_ratio**4)
                - 0.653192 * (capacity_ratio**3)
                + 0.964704 * (capacity_ratio**2)
                - 0.805845 * capacity_ratio
                + 0.315584
            )

        elif self.fuel_type == "open cycle gas turbine":
            eta_loss = (
                0.485049 * (capacity_ratio**4)
                - 1.540723 * (capacity_ratio**3)
                + 1.899607 * (capacity_ratio**2)
                - 1.251502 * capacity_ratio
                + 0.407569
            )

        else:
            eta_loss = 0

        efficiency = self.efficiency - eta_loss
        co2_price = self.forecaster.get_price("co2").at[timestep]

        marginal_cost = (
            fuel_price / efficiency
            + co2_price * self.emission_factor / efficiency
            + self.additional_cost
        )

        return marginal_cost

    def support_payment(self, start: datetime, price: float, volume: float) -> float:
        """
        Returns what the support contract of the plant pays for ``volume`` MW generated in the
        time step starting at ``start`` when the market price is ``price``, as a rate per hour
        like the cashflow. Nothing without a contract.

        - "premium": the premium per MWh, whatever the price.
        - "cfd" settled against the price the plant itself earns: the strike price less the
          market price, so that the plant earns the strike price in all. Under the "any_hour"
          rule nothing in a time step with a negative price; any other rule is taken as paying
          in every time step.
        - "cfd" settled against a reference price series (``support_reference``): the strike
          price less the reference price, nothing where the reference price is missing.

        Args:
            start (datetime.datetime): The start of the time step.
            price (float): The market price the plant earns in the time step.
            volume (float): The volume generated, in MW.

        Returns:
            float: The payment of the contract.
        """
        if not self.support_scheme or volume == 0:
            return 0.0
        if self.support_scheme == "premium":
            return self.support_value * volume
        if self.support_reference:
            if self.support_reference not in self.forecaster.fuel_prices:
                raise ValueError(
                    f"unit {self.id}: no price series '{self.support_reference}' for its support contract"
                )
            reference = self.forecaster.get_price(self.support_reference).at[start]
            if np.isnan(reference):
                return 0.0
            return (self.support_value - reference) * volume
        if self.support_neg_price_rule == "any_hour" and price < 0:
            return 0.0
        return (self.support_value - price) * volume

    def levy_payment(self, start: datetime, price: float, volume: float) -> float:
        """
        Returns what a levy on generation revenue takes from ``volume`` MW sold at ``price`` in
        the time step starting at ``start``, as a negative rate per hour: ``levy_rate`` times
        the receipts above ``levy_benchmark`` per MWh. Nothing below the benchmark, nothing
        without a levy, and nothing for a purchase (a negative volume).

        Such a levy is assessed on the realised average price of a company over a year in
        practice; taking it time step by time step is the marginal view of a plant that sells at
        the market price, and overstates what a company whose receipts stay below the benchmark
        on average would pay.
        """
        if self.levy_rate <= 0.0 or volume <= 0.0:
            return 0.0
        return -self.levy_rate * max(price - self.levy_benchmark, 0.0) * volume

    def calculate_cashflow(self, product_type: str, orderbook: Orderbook):
        """
        Calculates the cashflow for the given product type, and for energy the payments of the
        support contract of the plant (``outputs["support_cashflow"]``) and of a levy on its
        revenue (``outputs["levy_cashflow"]``, negative), which are not part of the market
        cashflow.

        Args:
            product_type: The product type.
            orderbook: The orderbook.
        """
        super().calculate_cashflow(product_type, orderbook)
        if product_type != "energy":
            return
        payments = []
        if self.support_scheme:
            payments.append(("support_cashflow", self.support_payment))
        if self.levy_rate > 0.0:
            payments.append(("levy_cashflow", self.levy_payment))
        if not payments:
            return
        for order in orderbook:
            start, end = order["start_time"], order["end_time"]
            start_idx, stop_idx, elapsed_intervals = self.index._get_idx_range(
                start, end
            )
            volume = order.get("accepted_volume", 0)
            price = order.get("accepted_price", 0)
            for name, payment_of in payments:
                if isinstance(volume, dict):
                    payment = np.array(
                        [
                            payment_of(
                                t,
                                price[t] if isinstance(price, dict) else price,
                                volume[t],
                            )
                            for t in volume
                        ]
                    )
                else:
                    payment = payment_of(start, price, volume)
                self.outputs[name].data[start_idx:stop_idx] += (
                    payment * elapsed_intervals
                )

    def calculate_min_max_power(
        self,
        start: datetime,
        end: datetime,
        product_type="energy",
        use_forecast: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the additional minimum and maximum power output.

        "Additional" means relative to the already-committed power, e.g. at other markets.
        Considers heat demand, positive capacity reserve (capacity_pos reduces headroom),
        and negative capacity reserve (capacity_neg raises the minimum obligation).

        Args:
            start (datetime): The start time of the dispatch.
            end (datetime): The end time of the dispatch (exclusive).
            product_type (str, optional): The product type. Defaults to "energy".
            use_forecast (bool, optional): Whether to use the availability as forecast ahead of
                time instead of the outturn, for a market that closes before the outturn is known.
                Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray]: The additional minimum and maximum power output of the unit.

        Note:
            The calculation does not include ramping constraints and can be used for arbitrary start times in the future.
            Ramping constraints can be applied afterwards via calculate_ramp().
        """
        # end includes the end of the last product, to get the last products' start time we deduct the frequency once
        end_excl = end - self.index.freq

        base_load = self.outputs["energy"].loc[start:end_excl]
        heat_demand = self.outputs["heat"].loc[start:end_excl]
        capacity_pos = self.outputs["capacity_pos"].loc[start:end_excl]
        capacity_neg = self.outputs["capacity_neg"].loc[start:end_excl]

        if use_forecast:
            availability = self.forecaster.availability_forecast.loc[start:end_excl]
        else:
            availability = self.forecaster.availability.loc[start:end_excl]
        available_power = availability * self.max_power

        # check if available power is larger than max_power and raise an error if so
        if (available_power > self.max_power).any():
            raise ValidationError(
                message=f"Available power is larger than max_power for unit {self.id} at time {start}.",
                id=self.id,
                field="availability",
            )

        # subtract positive reserve commitment and already-dispatched base_load from available power
        max_power = available_power - capacity_pos - base_load

        # warn if previous dispatch exceeded available power
        if (max_power < 0).any():
            logger.warning(
                f"Unit {self.id}: previous dispatch exceeded available power between {start} and {end}. Limiting additional power to 0."
            )
        # additional power can never be negative
        max_power = max_power.clip(min=0)

        # actual minimum is technical minimum + negative reserve obligation - already dispatched base_load
        min_power = self.min_power + capacity_neg - base_load
        # must also cover any heat demand not yet met by base_load
        # clip to 0: negative values mean base_load already covers heat demand fully
        heat_demand_still_needed = (heat_demand - base_load).clip(min=0)
        min_power = min_power.clip(min=heat_demand_still_needed)

        # if additional max_power is below the technical minimum, the unit cannot run at all
        infeasible = max_power < min_power
        min_power[infeasible] = 0
        max_power[infeasible] = 0

        return min_power, max_power

    def calculate_marginal_cost(self, start: datetime, power: float) -> float:
        """
        Calculates the marginal cost of the unit based on the provided start time and power output and returns it.
        Returns the marginal cost of the unit.

        Args:
            start (datetime.datetime): The start time of the dispatch.
            power (float): The power output of the unit.

        Returns:
            float: The marginal cost of the unit.
        """
        # if marginal costs already exists, return it
        if self.marginal_cost is not None:
            return (
                self.marginal_cost[start]
                if len(self.marginal_cost) > 1
                else self.marginal_cost
            )
        # if not, calculate it
        else:
            return self.calc_marginal_cost_with_partial_eff(
                power_output=power,
                timestep=start,
            )

    def as_dict(self) -> dict:
        """
        Returns the attributes of the unit as a dictionary, including specific attributes.

        Returns:
            dict: The attributes of the unit as a dictionary.
        """
        unit_dict = super().as_dict()
        unit_dict.update(
            {
                "max_power": self.max_power,
                "min_power": self.min_power,
                "emission_factor": self.emission_factor,
                "efficiency": self.efficiency,
                "fuel_type": self.fuel_type,
                "max_heat_extraction": self.max_heat_extraction,
                "hot_start_cost": self.hot_start_cost,
                "warm_start_cost": self.warm_start_cost,
                "cold_start_cost": self.cold_start_cost,
                "unit_type": "power_plant",
            }
        )

        return unit_dict
