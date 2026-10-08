# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One run of a GB scenario, read back for the figures: the framework's CSV output of the run
(``--csv DIR`` writes it to ``DIR/<scenario>_<case>``) and the scenario folder it was run from.

Nothing here is model logic; it is bookkeeping on the order book. Positions are taken from
``market_orders`` (accepted volume, summed over the markets), not from ``unit_dispatch``, which
ends at the close of the last auction. The CSV output holds five significant digits; the prices
are read at full precision from ``clearing_prices.csv`` when the run wrote one (``run --csv``
does), otherwise from ``market_meta``.

Three devices of the scenarios shape what the order book means (``convert``):

    demand          the GB demand less nuclear and pumped-storage output, plus the export capacity
                    of the interconnectors; nuclear and pumped storage are not units
    export units    offer the export capacity back at the neighbouring price: what an export unit
                    sells is export that does not happen, so export = offered - sold
    unserved demand a unit offering 100 GW at the price cap

so the market's own demand is not the GB demand. ``gb_demand`` is the demand position less the
export capacity, and ``generation`` (one column per group, exports negative) adds up to it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from assume_gb import convert, paths

HOURS = 0.5  # length of a period; money in the output is a rate per hour (price x MW)

# group of every technology, in stack order (the bottom of a generation chart first)
GROUPS = {
    "Wind": ["Wind"],
    "Solar": ["Solar"],
    "Hydro": ["Hydro"],
    "Biomass": ["Biomass", "Solid Biomass", "BIOMASS_MSG"],
    "Waste and other": ["Waste", "Other", "Geothermal", "EMB_MUSTRUN"],
    "Gas": ["CCGT", "OCGT", "CCGT_MSG", "EMB_FLEX"],
    "Coal and oil": ["Hard Coal", "Oil"],
    "Imports": ["IC_IMPORT"],
    "Storage": ["battery"],
    "Unserved": ["unserved_demand"],
}
EXPORTS = "Exports"
DEMAND = "Demand"
VRE = ("Wind", "Solar")
TECH_GROUP = {tech: group for group, techs in GROUPS.items() for tech in techs}
TECH_GROUP["IC_EXPORT"] = EXPORTS


def group_of(technology: str) -> str:
    return TECH_GROUP.get(technology, "Waste and other")


def _read(
    folder: Path,
    table: str,
    columns: list[str] | None = None,
    dates: tuple[str, ...] = (),
) -> pd.DataFrame | None:
    """A table of the framework's CSV output; a parquet copy is kept beside the CSV, as the
    order book of a year is several hundred MB."""
    csv, cache = folder / f"{table}.csv", folder / f"{table}.parquet"
    if not csv.exists():
        return None
    if cache.exists() and cache.stat().st_mtime >= csv.stat().st_mtime:
        frame = pd.read_parquet(cache, columns=columns)
    else:
        frame = pd.read_csv(csv, engine="pyarrow")
        for column in dates:
            frame[column] = pd.to_datetime(frame[column])
        frame.to_parquet(cache, index=False)
        frame = frame[columns] if columns else frame
    return frame


@dataclass
class Run:
    """The output of one simulation and the scenario folder behind it."""

    folder: Path
    scenario_dir: Path
    name: str
    scenario: str
    case: str
    prices: pd.DataFrame  # clearing price per market, index the start of the period
    orders: (
        pd.DataFrame
    )  # start_time, market_id, unit_id, price, volume, accepted_volume
    units: pd.DataFrame  # one row per unit: technology, group, kind, max_power, ...
    reference: pd.DataFrame
    admissible: pd.Series
    observed: pd.DataFrame
    meta: dict = field(default_factory=dict)
    soc: pd.Series | None = None

    # ------------------------------------------------------------------ what was cleared
    @property
    def markets(self) -> list[str]:
        return list(self.prices.columns)

    @property
    def day_ahead(self) -> pd.Series:
        return self.prices[convert.MARKET_ID]

    @property
    def year(self) -> int:
        return int(
            self.meta.get("source", {}).get(
                "year", self.prices.index[len(self.prices) // 2].year
            )
        )

    @cached_property
    def index(self) -> pd.DatetimeIndex:
        return self.prices.index

    def _pivot(
        self, frame: pd.DataFrame, values: str = "accepted_volume"
    ) -> pd.DataFrame:
        table = (
            frame.groupby(["start_time", "unit_id"], sort=False)[values]
            .sum()
            .unstack(fill_value=0.0)
        )
        return table.reindex(index=self.index, fill_value=0.0)

    @cached_property
    def position(self) -> pd.DataFrame:
        """MW each unit holds after all markets of the run (sold > 0, bought < 0)."""
        return self._pivot(self.orders)

    def market_position(self, market_id: str) -> pd.DataFrame:
        return self._pivot(self.orders[self.orders.market_id == market_id])

    @cached_property
    def offered(self) -> pd.DataFrame:
        """MW each unit offered to sell in the day-ahead market: what it expected to have."""
        da = self.orders[
            (self.orders.market_id == convert.MARKET_ID) & (self.orders.volume > 0)
        ]
        return self._pivot(da, "volume")

    def _units_of(self, *groups: str) -> list[str]:
        return [u for u in self.position.columns if self.units.group.get(u) in groups]

    @cached_property
    def export_capacity(self) -> pd.Series:
        return self.offered.reindex(
            columns=self._units_of(EXPORTS), fill_value=0.0
        ).sum(axis=1)

    @cached_property
    def exports(self) -> pd.Series:
        """MW exported: the export capacity the export units did not sell back."""
        units = self._units_of(EXPORTS)
        return (
            self.offered.reindex(columns=units, fill_value=0.0) - self.position[units]
        ).sum(axis=1)

    @cached_property
    def gb_demand(self) -> pd.Series:
        """MW of demand the market serves in GB: the demand unit's position less the export
        capacity (GB demand less nuclear and pumped-storage output)."""
        return -self.position[convert.DEMAND_UNIT] - self.export_capacity

    @cached_property
    def generation(self) -> pd.DataFrame:
        """MW per group in stack order, exports negative, storage net (charging negative)."""
        groups = self.units.group.reindex(self.position.columns)
        table = self.position.T.groupby(groups).sum().T
        table = table.drop(columns=[c for c in (EXPORTS, DEMAND) if c in table])
        table[EXPORTS] = -self.exports
        order = [
            g
            for g in GROUPS
            if g in table and (g != "Unserved" or table[g].abs().sum() > 0)
        ]
        return table[order + [EXPORTS]]

    @cached_property
    def residual_load(self) -> pd.Series:
        """GB demand less wind and solar output, MW."""
        return self.gb_demand - self.generation.reindex(
            columns=list(VRE), fill_value=0.0
        ).sum(axis=1)

    # ------------------------------------------------------------------ who sets the price
    def marginal(self, market_id: str = convert.MARKET_ID) -> pd.DataFrame:
        """The order that sets the price of every period: an order offered at the clearing price,
        preferring one that is partly accepted, then a sale. Columns ``unit_id``, ``group``
        (``Demand`` where the demand unit sets it) and ``partial``."""
        price = self.prices[market_id].rename("clearing")
        book = self.orders[self.orders.market_id == market_id].join(
            price, on="start_time"
        )
        tolerance = (
            1e-4 * book["clearing"].abs() + 0.011
        )  # the CSV output's five digits
        book = book[(book["price"] - book["clearing"]).abs() <= tolerance].copy()
        book["partial"] = (book.accepted_volume.abs() > 1e-6) & (
            book.accepted_volume.abs() < book.volume.abs() - 1e-6
        )
        book["sale"] = book.volume > 0
        book = book.sort_values(
            ["start_time", "partial", "sale"], ascending=[True, False, False]
        )
        chosen = book.drop_duplicates("start_time").set_index("start_time")
        # a unit buying back in a later market sets the price as its own technology
        chosen["group"] = chosen.unit_id.map(self.units.group).fillna(DEMAND)
        return chosen.reindex(self.index)[["unit_id", "group", "partial", "price"]]

    # ------------------------------------------------------------------ money
    def revenue(self) -> pd.DataFrame:
        """GBP each unit receives from the markets per period (bought energy counts negative)."""
        total = pd.DataFrame(0.0, index=self.index, columns=self.position.columns)
        for market_id in self.markets:
            total = total.add(
                self.market_position(market_id).mul(self.prices[market_id], axis=0)
                * HOURS,
                fill_value=0.0,
            )
        return total

    def capture_prices(self) -> pd.DataFrame:
        """Per group over the run: energy (TWh), market revenue (GBP m), the price earned per MWh
        and that price against the time-weighted mean day-ahead price (value factor)."""
        groups = self.units.group
        revenue = (
            self.revenue()
            .T.groupby(groups.reindex(self.position.columns))
            .sum()
            .T.sum()
        )
        sold = (
            self.position.clip(lower=0)
            .T.groupby(groups.reindex(self.position.columns))
            .sum()
            .T.sum()
            * HOURS
        )
        table = pd.DataFrame({"twh": sold / 1e6, "revenue_gbp_m": revenue / 1e6})
        table = table.reindex(
            [g for g in GROUPS if g in table.index and table.loc[g, "twh"] > 0]
        )
        table["capture_price"] = table.revenue_gbp_m * 1e6 / (table.twh * 1e6)
        table["value_factor"] = table.capture_price / self.day_ahead.mean()
        return table

    # ------------------------------------------------------------------ intraday
    @cached_property
    def imbalance(self) -> pd.Series:
        """MW by which the system turned out long at the intraday auction: what wind, solar and
        demand traded there (more wind or less demand than forecast > 0)."""
        if convert.INTRADAY_MARKET_ID not in self.markets:
            return pd.Series(np.nan, index=self.index)
        units = self._units_of(*VRE) + [convert.DEMAND_UNIT]
        return (
            self.market_position(convert.INTRADAY_MARKET_ID)
            .reindex(columns=units, fill_value=0.0)
            .sum(axis=1)
        )

    # ------------------------------------------------------------------ order book of one period
    def book(
        self, start: pd.Timestamp, market_id: str = convert.MARKET_ID
    ) -> pd.DataFrame:
        """The orders of one period of one market, sales by price ascending then bids by price
        descending, with each order's group and the cumulative MW of its side."""
        book = self.orders[
            (self.orders.start_time == start) & (self.orders.market_id == market_id)
        ].copy()
        book["group"] = book.unit_id.map(self.units.group).fillna(DEMAND)
        book.loc[book.unit_id == convert.DEMAND_UNIT, "group"] = DEMAND
        sales = book[book.volume > 0].sort_values("price")
        bids = book[book.volume < 0].sort_values("price", ascending=False)
        sales["cumulative"] = sales.volume.cumsum()
        bids["cumulative"] = -bids.volume.cumsum()
        return pd.concat([sales, bids])


def _scenario_and_case(name: str) -> tuple[str, str]:
    """Scenario and study case of a simulation id ``<scenario>_<case>``. A scenario folder that
    begins the name decides (the longest, so that a variant of a case is found whatever it is
    called); without one, the names of the built-in cases."""
    root = paths.scenarios_dir()
    folders = (
        sorted((p.name for p in root.iterdir() if p.is_dir()), key=len, reverse=True)
        if root.is_dir()
        else []
    )
    for scenario in folders:
        if name.startswith(scenario + "_"):
            return scenario, name[len(scenario) + 1 :]
    for case in sorted(_known_cases(), key=len, reverse=True):
        if name.endswith("_" + case):
            return name[: -len(case) - 1], case
    scenario, _, case = name.partition("_day_ahead")
    return scenario, "day_ahead" + case


def _known_cases() -> list[str]:
    return [
        convert.CASE,
        convert.WEEK_CASE,
        convert.STORAGE_CASE,
        convert.STORAGE_WEEK_CASE,
        convert.INTRADAY_CASE,
        convert.INTRADAY_WEEK_CASE,
        convert.LEARNING_CASE,
        convert.LEARNING_WEEK_CASE,
        convert.LEARNING_YEAR_CASE,
        convert.LEARNING_CFD_CASE,
        convert.LEARNING_CFD_WEEK_CASE,
        convert.LEARNING_CFD_YEAR_CASE,
    ]


def load(folder: Path | str, scenario_dir: Path | str | None = None) -> Run:
    """Read the run whose framework output is in ``folder`` (``<csv dir>/<scenario>_<case>``)."""
    folder = Path(folder)
    meta_table = _read(folder, "market_meta", dates=("product_start", "time"))
    if meta_table is None:
        raise FileNotFoundError(
            f"no market_meta.csv in {folder}: is it the CSV output of a run?"
        )
    name = str(meta_table["simulation"].iloc[0])
    scenario, case = _scenario_and_case(name)
    scenario_dir = (
        Path(scenario_dir) if scenario_dir else paths.scenarios_dir() / scenario
    )

    full = folder / "clearing_prices.csv"
    if full.exists():
        prices = pd.read_csv(full, index_col=0, parse_dates=True)
    else:
        meta_table["product_start"] = pd.to_datetime(meta_table["product_start"])
        prices = meta_table.pivot_table(
            index="product_start", columns="market_id", values="price", aggfunc="first"
        )
    prices = prices.sort_index().rename_axis("datetime")
    prices.columns.name = None

    orders = _read(
        folder,
        "market_orders",
        ["start_time", "market_id", "unit_id", "price", "volume", "accepted_volume"],
        dates=("start_time", "end_time"),
    )
    orders["start_time"] = pd.to_datetime(orders["start_time"])
    orders = orders[orders.start_time.isin(prices.index)]

    units = pd.read_csv(scenario_dir / "powerplant_units.csv").set_index("name")
    if (scenario_dir / "storage_units.csv").exists():
        storage = pd.read_csv(scenario_dir / "storage_units.csv").set_index("name")
        storage["kind"] = "storage"
        units = pd.concat(
            [units, storage.rename(columns={"max_power_discharge": "max_power"})]
        )
    units.loc[convert.DEMAND_UNIT, ["technology", "kind"]] = ["demand", "demand"]
    units["group"] = units.technology.map(group_of)
    units.loc[convert.DEMAND_UNIT, "group"] = DEMAND

    reference = pd.read_csv(
        scenario_dir / "reference_prices.csv", index_col=0, parse_dates=True
    )
    admissible = (
        reference.pop("admissible").astype(bool).reindex(prices.index, fill_value=False)
    )
    reference = reference.reindex(prices.index)
    observed = pd.DataFrame(index=prices.index)
    if (scenario_dir / "observed_prices.csv").exists():
        observed = pd.read_csv(
            scenario_dir / "observed_prices.csv", index_col=0, parse_dates=True
        ).reindex(prices.index)
    meta = (
        json.loads((scenario_dir / "scenario_meta.json").read_text(encoding="utf-8"))
        if (scenario_dir / "scenario_meta.json").exists()
        else {}
    )

    run = Run(
        folder,
        scenario_dir,
        name,
        scenario,
        case,
        prices,
        orders,
        units,
        reference,
        admissible,
        observed,
        meta,
    )
    dispatch = _read(folder, "unit_dispatch", dates=("time",))
    if dispatch is not None and "soc" in dispatch:
        soc = dispatch[dispatch.unit == convert.STORAGE_UNIT].set_index("time")["soc"]
        run.soc = soc[~soc.index.duplicated(keep="last")].reindex(prices.index)
    return run
