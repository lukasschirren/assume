# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""
A heuristic for a power plant with start-up costs and minimum up and down times that plans its
commitment on its price forecast and prices its start-up cost into the half-hours or hours of the
run it expects.
"""

from datetime import timedelta

import numpy as np

from assume.common.base import MinMaxStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.common.utils import parse_duration


class EnergyHeuristicLookaheadStrategy(MinMaxStrategy):
    """
    A power plant with start-up costs, a minimum stable output and minimum up and down times bids
    on a plan of its commitment made on its price forecast (``forecaster.price[market_id]``) over
    the auction's products and ``lookahead`` beyond them (default 24 hours).

    The plan, period by period from the unit's state before the first product: a unit that is
    off starts when the forecast price is at least its marginal cost and the forecast margin of
    the run, at least the minimum up time long, covers the start-up cost (hot, warm or cold by the
    time it has been off); a unit that is on stays on while the forecast price is at least its
    marginal cost, for its minimum up time, and through a period below its marginal cost when the
    loss at minimum output until the price returns is smaller than the cost of starting again.
    A running unit produces its maximum where the forecast is at least its marginal cost and its
    minimum elsewhere.

    The bids for each product, two per product (the minimum output and the rest):

    - committed (running before the first product and planned to keep running): everything at
      marginal cost, except the minimum output, which is offered at the market's lowest price
      within the minimum up time and, through a planned stretch below marginal cost, at marginal
      cost less the cost of starting again spread over the minimum output of that stretch;
    - not committed, in a run the plan starts during the auction: everything at marginal cost plus
      the start-up cost spread over the energy of the planned run (its maximum output where the
      forecast is at least the marginal cost, its minimum elsewhere), in every period of the run:
      a run that starts within a multi-product auction is not committed in any of its products;
    - not committed, outside a planned run: everything at marginal cost plus the start-up cost
      spread over a run of the minimum up time at the maximum output, so that a start that pays
      at a higher price than forecast is still offered; nothing within the minimum down time.

    Money and energy follow the product's duration: the start-up cost (currency) is spread over
    MWh, so at half-hourly products a period of maximum output counts max_power x 0.5 MWh.
    Unlike ``powerplant_energy_heuristic_flexable``, the strategy does not take a unit to be
    running in the later products of an auction because it bids in an earlier one, measures the
    time a unit has been off back to its warm-start threshold (so warm and cold starts are
    priced), and spreads the start-up cost over the run it expects instead of the minimum up time.
    Ramp limits are applied from the output before the first product and from the planned output
    after it.

    Methods
    -------
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lookahead = parse_duration(kwargs.get("lookahead", "24h"))

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Plans the unit's commitment on its price forecast and bids each product accordingly.

        Args:
            unit (SupportsMinMax): The unit to be dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The products of the auction.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """
        index = unit.index
        step = index.freq
        hours = step / timedelta(hours=1)
        first = product_tuples[0][0]
        i0 = index._get_idx_from_date(first)
        n = min(len(product_tuples) + int(self.lookahead / step), len(index) - i0)
        n = max(n, len(product_tuples))

        floor = market_config.minimum_bid_price
        cap = market_config.maximum_bid_price
        price = np.asarray(
            unit.forecaster.price[market_config.market_id].data[i0 : i0 + n],
            dtype=float,
        )
        cost = self.marginal_costs(unit, i0, n)
        qmax = np.asarray(
            unit.forecaster.availability.data[i0 : i0 + n], dtype=float
        ) * float(unit.max_power)
        n = min(n, len(price), len(cost), len(qmax))
        price, cost, qmax = price[:n], cost[:n], qmax[:n]
        pmin = float(unit.min_power)

        plan = plan_commitment(
            price=price,
            cost=cost,
            qmax=qmax,
            pmin=pmin,
            hours=hours,
            state=unit_state(unit, i0),
            min_up=int(unit.min_operating_time),
            min_down=int(unit.min_down_time),
            start_cost=unit.get_starting_costs,
        )

        min_power_values, max_power_values = unit.calculate_min_max_power(
            first, product_tuples[-1][1]
        )
        previous = unit.get_output_before(first)
        bids = []
        for k, (product, min_power, max_power) in enumerate(
            zip(product_tuples, min_power_values, max_power_values)
        ):
            if k >= n:
                break
            on_before = plan.on[k - 1] if k > 0 else plan.state_on
            if k > 0:
                previous = float(plan.output[k - 1])
            current = float(
                unit.outputs["energy"].data[i0 + k]
            )  # sold on earlier markets
            hi, lo = float(max_power), float(min_power)
            if unit.ramp_up is not None:
                hi = min(hi, previous + unit.ramp_up - current)
            if on_before and unit.ramp_down is not None:
                lo = max(lo, previous - unit.ramp_down - current)
            lo = min(max(lo, 0.0), hi)
            if hi <= 0:
                continue

            block = plan.block_of[k]
            mc = cost[k]
            if plan.committed(k):
                if 0 < plan.up_before(k) < unit.min_operating_time:
                    price_inflex = floor  # within the minimum up time: it runs anyway
                elif plan.on[k] and not plan.merit[k]:
                    price_inflex = mc - plan.restart_discount[k]
                else:
                    price_inflex = mc
                price_flex = mc
            elif block >= 0:
                b = plan.blocks[block]
                markup = b["start_cost"] / b["energy"] if b["energy"] > 0 else 0.0
                price_inflex = price_flex = mc + markup
            else:
                if plan.down[k] < unit.min_down_time and not on_before:
                    continue  # within the minimum down time: cannot start
                energy = hi * hours * max(int(unit.min_operating_time), 1)
                markup = plan.start_cost_off[k] / energy if energy > 0 else 0.0
                price_inflex = price_flex = mc + markup

            price_inflex = min(max(price_inflex, floor), cap)
            price_flex = min(max(price_flex, floor), cap)
            for volume, bid_price in ((lo, price_inflex), (hi - lo, price_flex)):
                bids.append(
                    {
                        "start_time": product[0],
                        "end_time": product[1],
                        "only_hours": product[2],
                        "price": bid_price,
                        "volume": volume,
                        "node": unit.node,
                    }
                )
        return self.remove_empty_bids(bids)

    @staticmethod
    def marginal_costs(unit: SupportsMinMax, i0: int, n: int) -> np.ndarray:
        """The unit's marginal cost at maximum output in the ``n`` periods from position ``i0``."""
        series = getattr(unit, "marginal_cost", None)
        if series is not None and hasattr(series, "data"):
            return np.asarray(series.data[i0 : i0 + n], dtype=float)
        if series is not None and np.ndim(series) == 0:
            return np.full(n, float(series))
        times = unit.index[i0 : i0 + n]
        return np.array(
            [unit.calculate_marginal_cost(t, unit.max_power) for t in times],
            dtype=float,
        )


class Plan:
    """The planned commitment over the look-ahead window (see ``plan_commitment``)."""

    def __init__(self, n: int, state_on: bool, state_count: int = 0):
        self.state_on = state_on
        self.state_count = state_count
        self.on = np.zeros(n, dtype=bool)
        self.merit = np.zeros(n, dtype=bool)
        self.output = np.zeros(n)
        self.up = np.zeros(n, dtype=int)  # periods on, this one included
        self.down = np.zeros(n, dtype=int)  # periods off before this one
        self.block_of = np.full(n, -1, dtype=int)
        self.restart_discount = np.zeros(n)
        self.start_cost_off = np.zeros(n)
        self.blocks: list[dict] = []

    def committed(self, k: int) -> bool:
        """Whether the unit is in, or has just left, the run it was in before the first product:
        it is running for certain then, and does not have to start."""
        b = self.block_of[k]
        if b >= 0:
            return self.blocks[b]["committed"]
        if k == 0:
            return self.state_on
        previous = self.block_of[k - 1]
        return previous >= 0 and self.blocks[previous]["committed"]

    def up_before(self, k: int) -> int:
        """Periods the unit has been running before period ``k`` (0 when it is off)."""
        if k == 0:
            return self.state_count if self.state_on else 0
        return int(self.up[k - 1]) if self.on[k - 1] else 0


def unit_state(unit: SupportsMinMax, i0: int) -> tuple[bool, int]:
    """Whether the unit is on in the period before position ``i0`` and for how many periods it
    has been in that state, counted back far enough to tell a warm or cold start from a hot one.
    Before the first period of the index the unit counts as running."""
    if i0 <= 0:
        return True, max(int(unit.min_operating_time), 1)
    warm = getattr(unit, "downtime_warm_start", 0) or 0
    lookback = int(
        max(unit.min_operating_time, unit.min_down_time, np.ceil(warm) + 1, 1)
    )
    history = np.asarray(
        unit.outputs["energy"].data[max(0, i0 - lookback) : i0], dtype=float
    )
    if len(history) == 0:
        return True, max(int(unit.min_operating_time), 1)
    running = history[::-1] > 0
    changes = np.flatnonzero(running != running[0])
    count = int(changes[0]) if len(changes) else len(running)
    return bool(running[0]), count


def plan_commitment(
    price: np.ndarray,
    cost: np.ndarray,
    qmax: np.ndarray,
    pmin: float,
    hours: float,
    state: tuple[bool, int],
    min_up: int,
    min_down: int,
    start_cost,
) -> Plan:
    """
    The commitment a unit plans on its price forecast, period by period (see
    ``EnergyHeuristicLookaheadStrategy``). ``start_cost(-d)`` is the start-up cost after ``d``
    periods off; ``qmax`` the available power per period; a period in which less than the minimum
    output is available is off.
    """
    n = len(price)
    on, count = state
    plan = Plan(n, on, count)
    available = (qmax > 0) & (qmax >= pmin)
    merit = (price >= cost) & available
    plan.merit = merit
    positions = np.arange(n)
    # the next period in merit from each period on (n if none) and the end of a merit run
    nxt = np.minimum.accumulate(np.where(merit, positions, n)[::-1])[::-1]
    run_end = np.minimum.accumulate(np.where(~merit, positions, n)[::-1])[::-1]
    output = np.where(merit, qmax, np.minimum(pmin, qmax))
    gain = (price - cost) * output * hours
    cum = np.concatenate([[0.0], np.cumsum(gain)])
    cum_energy = np.concatenate([[0.0], np.cumsum(output * hours)])

    up = count if on else 0
    down = 0 if on else count
    block = -1
    if on:
        plan.blocks.append({"start": 0, "end": n, "start_cost": 0.0, "committed": True})
        block = 0
    for t in range(n):
        plan.down[t] = down
        if on:
            stay = available[t] and (up < min_up or merit[t])
            if available[t] and not stay:
                j = int(nxt[t])
                if j < n:
                    loss = -(cum[j] - cum[t])
                    restart = start_cost(-(j - t))
                    stay = loss <= restart
                    if stay:
                        plan.restart_discount[t] = restart / (
                            max(pmin, 1e-9) * hours * (j - t)
                        )
            if stay:
                up += 1
                plan.on[t] = True
                plan.up[t] = up
                plan.block_of[t] = block
                plan.output[t] = output[t]
                continue
            plan.blocks[block]["end"] = t
            on, up, down = False, 0, 0
            plan.down[t] = 0
        # off in period t unless it starts now
        plan.start_cost_off[t] = start_cost(-max(down, 1))
        if available[t] and merit[t] and down >= min_down:
            end = min(max(int(run_end[t]), t + max(min_up, 1)), n)
            sc = start_cost(-max(down, 1))
            if cum[end] - cum[t] >= sc:
                on, up = True, 1
                plan.blocks.append(
                    {"start": t, "end": n, "start_cost": sc, "committed": False}
                )
                block = len(plan.blocks) - 1
                plan.on[t] = True
                plan.up[t] = up
                plan.block_of[t] = block
                plan.output[t] = output[t]
                continue
        down += 1
    for b in plan.blocks:
        b["energy"] = float(cum_energy[b["end"]] - cum_energy[b["start"]])
    return plan
