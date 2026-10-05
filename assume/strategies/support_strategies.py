# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np

from assume.common.base import SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.strategies.naive_strategies import EnergyNaiveStrategy

PREMIUM = "premium"
CFD = "cfd"
ANY_HOUR = "any_hour"


def support_bid_price(
    marginal_cost: float | np.ndarray,
    scheme: str,
    value: float,
    neg_price_rule: str = "",
    reference_price: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """
    Returns the lowest market price at which generating still pays for a unit with a support contract.

    - "premium": a fixed payment per MWh on top of the market price, such as a certificate
      earned per MWh generated, a generation tariff or a fixed market premium. The unit earns
      the price plus the premium, so the bid is the marginal cost less the premium.
    - "cfd" without a reference price: the contract is settled against the price the unit
      itself earns and tops it up to the strike price. The top-up is at most the strike price,
      so at a negative price the unit earns the strike price plus that price, and the bid is
      the marginal cost less the strike price. Under the "any_hour" rule the contract pays
      nothing in a period with a negative price, while from a price of zero upwards the unit
      earns the strike price: the bid is zero, or the marginal cost if that is lower. Any other
      rule, such as a suspension only after six negative hours in a row ("six_hour"), is taken
      as paying in every period.
    - "cfd" with a reference price: the contract pays the strike price less a reference price
      that the unit's own output does not move, on top of the market price. The bid is the
      marginal cost less that difference, and the marginal cost where no reference price is
      given (NaN).

    Args:
        marginal_cost (float | np.ndarray): The marginal cost of the unit.
        scheme (str): The support scheme, "premium", "cfd" or "" for none.
        value (float): The premium or the strike price.
        neg_price_rule (str): The negative-price rule of a "cfd".
        reference_price (float | np.ndarray | None): The reference price of a "cfd".

    Returns:
        float | np.ndarray: The bid price, before the price limits of a market are applied.
    """
    if not scheme:
        return marginal_cost
    if scheme == PREMIUM:
        return marginal_cost - value
    if scheme == CFD:
        if reference_price is not None:
            reference = np.asarray(reference_price, dtype=float)
            return marginal_cost - np.where(np.isnan(reference), 0.0, value - reference)
        if neg_price_rule == ANY_HOUR:
            return np.minimum(marginal_cost, 0.0)
        return marginal_cost - value
    raise ValueError(f"unknown support scheme {scheme!r}")


class EnergyNaiveSupportStrategy(EnergyNaiveStrategy):
    """
    A naive strategy for a unit with a support contract: it bids the lowest price at which
    generating still pays, which is its marginal cost less what the contract pays per MWh.

    The volume and the marginal cost are those of the naive strategy. The contract is read from
    the unit (support_scheme, support_value, support_neg_price_rule, support_reference), and the
    bid price is kept within the price limits of the market. A unit without a contract bids
    exactly as with the naive strategy.

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
        bids = super().calculate_bids(unit, market_config, product_tuples, **kwargs)
        if not getattr(unit, "support_scheme", ""):
            return bids

        reference = self.reference_price(unit)
        for bid in bids:
            bid["price"] = self.supported_price(
                unit, market_config, bid["price"], bid["start_time"], reference
            )

        return bids

    @staticmethod
    def reference_price(unit: SupportsMinMax):
        """
        Returns the reference price series of the unit's contract, or None if the contract is
        settled against the price the unit itself earns.

        Args:
            unit (SupportsMinMax): The unit with the contract.

        Returns:
            FastSeries | None: The reference price series.
        """
        if not unit.support_reference:
            return None
        if unit.support_reference not in unit.forecaster.fuel_prices:
            raise ValueError(
                f"unit {unit.id}: no price series '{unit.support_reference}' for its support contract"
            )
        return unit.forecaster.get_price(unit.support_reference)

    @staticmethod
    def supported_price(
        unit: SupportsMinMax,
        market_config: MarketConfig,
        marginal_cost: float,
        start,
        reference,
    ) -> float:
        """
        Returns the lowest price at which generating pays under the unit's contract, within the
        price limits of the market.

        Args:
            unit (SupportsMinMax): The unit with the contract.
            market_config (MarketConfig): The market configuration.
            marginal_cost (float): The marginal cost of the unit at start.
            start (datetime.datetime): The start of the product.
            reference (FastSeries | None): The reference price series of the contract.

        Returns:
            float: The bid price.
        """
        price = float(
            support_bid_price(
                marginal_cost,
                unit.support_scheme,
                unit.support_value,
                unit.support_neg_price_rule,
                None if reference is None else reference.at[start],
            )
        )
        return min(
            max(price, market_config.minimum_bid_price),
            market_config.maximum_bid_price,
        )
