# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""
Strategies for a sequence of markets in which the information of a unit changes: a market that
closes before the outturn of availability or demand is known is bid on the forecast, and a later
market trades the difference between the forecast and the outturn in both directions.
"""

import numpy as np

from assume.common.base import MinMaxStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.strategies.support_strategies import EnergyNaiveSupportStrategy


class EnergyNaiveForecastStrategy(EnergyNaiveSupportStrategy):
    """
    The naive strategy for a market that closes before the outturn is known: the volume follows
    the availability (power plant) or the demand (demand unit) as forecast ahead of time, read
    from the forecaster's availability_forecast or demand_forecast.

    The bid price is the marginal cost or, for a unit with a support contract, the lowest price at
    which generating pays under it. Without a forecast the unit's outturn is used, so the strategy
    then bids as the naive strategy does.

    Methods
    -------
    """

    def calculate_min_max_power(
        self, unit: SupportsMinMax, start, end
    ) -> tuple[np.ndarray, np.ndarray]:
        return unit.calculate_min_max_power(start, end, use_forecast=True)


class EnergyNaiveRebalanceStrategy(EnergyNaiveSupportStrategy):
    """
    A naive strategy for a power plant in a market that follows another one: it trades the
    difference between what it can deliver and what it has sold so far, in both directions.

    For each product the unit offers the power it can deliver beyond its position at the lowest
    price at which generating pays (its marginal cost or, with a support contract, less what the
    contract pays), and bids to buy back what it has sold down to its minimum power at that same
    price, so that it is replaced whenever power is cheaper than generating it. Power it has sold
    but cannot deliver is bought back at the maximum price of the market.

    When the market clears exactly at a unit's price, the unit is indifferent between its sell and
    its buy order and the clearing may accept both: a swap that leaves its position, cost and profit
    unchanged but counts in the traded volume. The net change of a unit's position is the sum of
    the accepted volumes of its orders.

    Methods
    -------
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.

        Args:
            unit (SupportsMinMax): The unit to be dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """
        supported = bool(getattr(unit, "support_scheme", ""))
        reference = self.reference_price(unit) if supported else None
        bids = []
        for product in product_tuples:
            start = product[0]
            available = unit.forecaster.availability.at[start] * unit.max_power
            sold = unit.outputs["energy"].at[start]
            price = unit.calculate_marginal_cost(start, sold)
            if supported:
                price = self.supported_price(
                    unit, market_config, price, start, reference
                )

            if available > sold:
                # more can be delivered than sold
                bids.append(self.order(unit, product, available - sold, price))
            elif sold > available:
                # what cannot be delivered has to be bought back
                bids.append(
                    self.order(
                        unit, product, available - sold, market_config.maximum_bid_price
                    )
                )
            # what is sold and can be delivered is bought back if that is cheaper than generating
            decrement = min(sold, available) - unit.min_power
            if decrement > 0:
                bids.append(self.order(unit, product, -decrement, price))

        return self.remove_empty_bids(bids)

    @staticmethod
    def order(
        unit: SupportsMinMax, product: Product, volume: float, price: float
    ) -> dict:
        return {
            "start_time": product[0],
            "end_time": product[1],
            "only_hours": product[2],
            "price": price,
            "volume": volume,
            "node": unit.node,
        }


class DemandEnergyNaiveRebalanceStrategy(MinMaxStrategy):
    """
    A naive strategy for a demand unit in a market that follows another one: it buys what it needs
    beyond what it has bought so far at the maximum price of the market, and sells what it has
    bought but does not need at the minimum price.

    Methods
    -------
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.

        Args:
            unit (SupportsMinMax): The unit to be dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """
        bids = []
        for product in product_tuples:
            start = product[0]
            # demand and what has been bought are negative: the difference is what is still needed
            volume = unit.forecaster.demand.at[start] - unit.outputs["energy"].at[start]
            if volume < 0:
                price = market_config.maximum_bid_price
            else:
                price = market_config.minimum_bid_price
            bids.append(
                EnergyNaiveRebalanceStrategy.order(unit, product, volume, price)
            )

        return self.remove_empty_bids(bids)
