# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Criterion 7 of the scorecard: does the error matter for decisions? A more accurate price need
not lead to better decisions [nitkaCombiningPredictiveDistributions2023], so a model is also scored
by what its prices do to a decision taken on them [fritzLearningFixOptimisationAware2026]. Two
decisions, each evaluated on the observed price and on every model's price with the same weights
and parameters:

    capture prices   the generation-weighted price of wind and solar per local month or year, the
                     price a supported or merchant generator is paid (``capture_prices``)
    storage          a battery that arbitrages each local delivery day with perfect foresight of a
                     price: its revenue on that price, and the revenue of that schedule settled at
                     the observed price, the decision value of the model (``storage_revenue``,
                     ``storage_summary``)

The functions take the frame of ``ValidationData.sample``: the observed price and one column per
model, in GBP/MWh. Money is GBP, energy MWh: the length of a period is taken into account here,
unlike in the framework's own cash flows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from assume_gb.validation.contract import OBSERVED, infer_resolution, local_day

PERIODS = {
    "year": "%Y",
    "month": "%Y-%m",
    "all": None,
}  # None: the whole sample at once


def capture_prices(
    sample: pd.DataFrame,
    generation: pd.DataFrame,
    by: str = "month",
    observed: str = OBSERVED,
) -> pd.DataFrame:
    """The capture price of each technology (a column of ``generation``, in GW) under each price
    series (a column of ``sample``) per local ``by`` period ("month", "year", or "all" for the
    whole sample), GBP/MWh: the
    generation-weighted price sum_t p_t g_t / sum_t g_t. The same generation weighs every price
    series, so that the difference to the observed capture price ("error") is the price error a
    generator is paid. Beside it the time-weighted mean price and the value factor, capture price
    over mean price. One row per technology, period and series
    [nitkaCombiningPredictiveDistributions2023]."""
    if by not in PERIODS:
        raise ValueError(f"by must be one of {tuple(PERIODS)}, not {by!r}")
    labels = (
        pd.Index(["all"] * len(sample))
        if PERIODS[by] is None
        else sample.index.strftime(PERIODS[by])
    )
    mean = sample.groupby(labels).mean()
    tables = []
    for technology in generation.columns:
        weight = generation[technology].reindex(sample.index)
        captured = sample.mul(weight, axis=0).groupby(labels).sum()
        price = captured.div(weight.groupby(labels).sum(), axis=0)
        table = pd.DataFrame(
            {
                "capture_price": price.stack(),
                "mean_price": mean.stack(),
            }
        ).rename_axis(["period", "series"])
        table["value_factor"] = table["capture_price"] / table["mean_price"]
        reference = (
            price[observed].reindex(table.index.get_level_values("period")).to_numpy()
        )
        table["error"] = table["capture_price"].to_numpy() - reference
        tables.append(table.reset_index().assign(technology=technology, by=by))
    columns = [
        "by",
        "period",
        "technology",
        "series",
        "capture_price",
        "mean_price",
        "value_factor",
        "error",
    ]
    return pd.concat(tables, ignore_index=True)[columns]


@dataclass(frozen=True)
class Battery:
    """A battery for the arbitrage decision: ``power`` (MW, charging and discharging), ``energy``
    (MWh), ``round_trip_efficiency`` (split evenly between charging and discharging), the limits
    of the state of charge and its value at the start and the end of every day, as shares of the
    energy. By default one MW for two hours, 85% round trip, half full at midnight."""

    power: float = 1.0
    energy: float = 2.0
    round_trip_efficiency: float = 0.85
    soc_min: float = 0.0
    soc_max: float = 1.0
    soc_day: float = 0.5

    def __post_init__(self) -> None:
        if self.power <= 0 or self.energy <= 0:
            raise ValueError("power and energy must be positive")
        if not 0 < self.round_trip_efficiency <= 1:
            raise ValueError("the round-trip efficiency must be in (0, 1]")
        if not 0 <= self.soc_min <= self.soc_day <= self.soc_max <= 1:
            raise ValueError("need 0 <= soc_min <= soc_day <= soc_max <= 1")


def arbitrage(
    prices: np.ndarray, hours: float, battery: Battery = Battery()
) -> tuple[float, np.ndarray]:
    """The schedule of ``battery`` that earns most from ``prices`` (GBP/MWh, consecutive periods of
    ``hours`` each) with perfect foresight, as a linear programme: charge c_t and discharge d_t
    (MW) with c_t + d_t at most the power, the state of charge s_t = s_(t-1) + h (eta c_t - d_t /
    eta) within its limits, eta the square root of the round-trip efficiency, and s back at its
    starting value after the last period. Returns the revenue sum_t p_t (d_t - c_t) h (GBP) and the
    net discharge d_t - c_t of every period (MW). The coupling of charge and discharge is the
    usual linear stand-in for "not both at once" [fritzLearningFixOptimisationAware2026]."""
    p = np.asarray(prices, dtype=float)
    n = len(p)
    eta = np.sqrt(battery.round_trip_efficiency)
    start = battery.soc_day * battery.energy
    identity = sparse.identity(n, format="csr")
    previous = sparse.eye(n, k=-1, format="csr")
    # variables: charge (n), discharge (n), state of charge at the end of each period (n)
    cost = np.concatenate([p * hours, -p * hours, np.zeros(n)])
    dynamics = sparse.hstack(
        [-eta * hours * identity, hours / eta * identity, identity - previous]
    )
    initial = np.zeros(n)
    initial[0] = start
    coupling = sparse.hstack([identity, identity, sparse.csr_matrix((n, n))])
    limits = (battery.soc_min * battery.energy, battery.soc_max * battery.energy)
    bounds = [(0.0, battery.power)] * (2 * n) + [limits] * (n - 1) + [(start, start)]
    result = linprog(
        cost,
        A_ub=coupling,
        b_ub=np.full(n, battery.power),
        A_eq=dynamics,
        b_eq=initial,
        bounds=bounds,
        method="highs",
    )
    if result.status != 0:
        raise RuntimeError(f"the arbitrage programme failed: {result.message}")
    return float(-result.fun), result.x[n : 2 * n] - result.x[:n]


def storage_revenue(
    sample: pd.DataFrame, battery: Battery = Battery(), observed: str = OBSERVED
) -> pd.DataFrame:
    """Criterion 7, storage: for every complete local delivery day (23, 24 or 25 hours, no price
    missing) and price series in ``sample``, the revenue of the battery's best schedule on that
    price with perfect foresight ("revenue", GBP) and the revenue of the same schedule settled at
    the observed price ("settled", GBP): what a storage operator who trusted the model would earn.
    One row per day and series [nitkaCombiningPredictiveDistributions2023]
    [fritzLearningFixOptimisationAware2026]."""
    resolution = infer_resolution(sample.index)
    hours = resolution / pd.Timedelta(hours=1)
    days = local_day(sample.index)
    tz = sample.index.tz
    rows = []
    for day, block in sample.groupby(days):
        length = (day + pd.Timedelta(days=1)).tz_localize(tz) - day.tz_localize(tz)
        if len(block) != length // resolution or block.isna().any().any():
            continue
        settle = block[observed].to_numpy()
        for name in block.columns:
            revenue, net = arbitrage(block[name].to_numpy(), hours, battery)
            settled = float(np.sum(settle * net) * hours)
            rows.append(
                {"day": day, "series": name, "revenue": revenue, "settled": settled}
            )
    return pd.DataFrame(rows, columns=["day", "series", "revenue", "settled"])


def storage_summary(revenues: pd.DataFrame, observed: str = OBSERVED) -> pd.DataFrame:
    """Per price series over the days of ``storage_revenue``: the days, the total revenue on its
    own price (GBP), its error against the observed total (GBP, and as a share), the mean absolute
    error and the correlation of the daily revenues against the observed ones, the total of its
    schedules settled at the observed price (GBP) and the decision value, that total over the
    observed perfect-foresight revenue: 1 for the observed price itself, below 1 for any other, and
    below 0 for a price whose schedules lose money at the observed price
    [fritzLearningFixOptimisationAware2026]."""
    revenue = revenues.pivot(index="day", columns="series", values="revenue")
    settled = revenues.pivot(index="day", columns="series", values="settled")
    base = revenue[observed]
    rows = {
        name: {
            "days": len(revenue),
            "revenue": revenue[name].sum(),
            "error": revenue[name].sum() - base.sum(),
            "relative_error": revenue[name].sum() / base.sum() - 1,
            "daily_mae": (revenue[name] - base).abs().mean(),
            "daily_r": revenue[name].corr(base),
            "settled": settled[name].sum(),
            "decision_value": settled[name].sum() / base.sum(),
        }
        for name in revenue.columns
    }
    return pd.DataFrame(rows).T.rename_axis("series")
