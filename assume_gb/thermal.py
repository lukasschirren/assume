# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Non-convex thermal supply for the GB scenarios, from the GB defaults of
`inputs/thermal_parameters.csv` (`gb_default = yes`; sources and reasoning in
`inputs/thermal_parameters.md`).

    python -m assume_gb thermal --years 2023 2025

writes, per scenario folder, `powerplant_units_nc.csv` and `powerplant_units_nc_avail.csv` and the
study cases `day_ahead_nc` and `day_ahead_nc_avail` (the `day_ahead` case with those units).

What changes, for the merchant CCGT, OCGT and coal units (oil, supported biomass and the rest keep
their offers):

- `min_power` = the minimum stable level x `max_power` (CCGT 0.468, OCGT 0.869, coal 0.46).
- `min_operating_time`, `min_down_time` in time steps of the scenario (ASSUME counts them so): the
  hours rounded up, at least one step (CCGT and coal 6 h = 12 half-hours, OCGT one step).
- `ramp_up`, `ramp_down` in MW per step: the rate per minute x 30 x `max_power`, at least the
  minimum stable level (a unit can always reach it in one step) and at most `max_power`. The
  ramps must be set: ASSUME enforces the minimum up and down times only for a unit with ramp limits.
- `hot_start_cost`, `warm_start_cost`, `cold_start_cost` in GBP per MW (ASSUME multiplies by
  `max_power`): the ENTSO-E wear cost in EUR at the year's average GBP/EUR rate (no deflator: the
  source states no price year), plus the start fuel at the year's mean fuel and carbon price. Gas
  start fuel is NCV and the scenarios' gas price GCV, so it is divided by 0.900; coal stays NCV
  with its NCV price. Carbon on start fuel: 0.1846 t/MWh GCV gas, 0.3344 t/MWh NCV coal (DESNZ).
  OCGT start fuel (Leigh Fisher, heating value not stated) is taken as GCV, any start type.
- `downtime_hot_start`, `downtime_warm_start` (hours, converted by ASSUME): the hot and warm
  thresholds.
- `bidding_DA` = `powerplant_energy_heuristic_flexable`: the inflexible part at the minimum stable
  level priced with a start-up markup when off, with a restart discount when on; the rest at
  marginal cost.
- `forecast_algorithms.price` = `price_naive_forecast`, so that the units hold the price forecast
  the strategy reads (the merit-order forecast of `forecasts_df.csv`).
- The unit `CCGT_MSG` is removed: the merit-order model's stand-in for the CCGTs' minimum stable
  generation (2.25 GW in 2023, always offered at -15 GBP/MWh, on top of the CCGT capacity). The
  units' own minimum stable level replaces it.
- `_nc_avail` also scales `max_power` by the Capacity Market de-rating factor of the delivery year
  that starts in the scenario year (CCGT 0.913 for 2023/24): the scenarios hold the CCGTs at full
  capacity in every half-hour.

Efficiencies and emission factors stay those of the merit-order model's stack. A merged class of
small plants (`agg_CCGT_*`) commits as one unit, which makes it lumpier than the plants it stands for.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from assume_gb import paths

TABLE = paths.TOOL_ROOT / "inputs" / "thermal_parameters.csv"
TECHNOLOGIES = {"CCGT": "ccgt", "OCGT": "ocgt", "Hard Coal": "coal"}
STRATEGY = "powerplant_energy_heuristic_flexable"
MSG_UNIT = "CCGT_MSG"
# GBP per EUR, ECB annual averages of the reference rate; 2025 provisional
GBP_PER_EUR = {
    2018: 0.8847,
    2019: 0.8778,
    2020: 0.8897,
    2021: 0.8596,
    2022: 0.8528,
    2023: 0.8698,
    2024: 0.8466,
    2025: 0.85,
}
NCV_PER_GCV_GAS = 0.900
CO2_PER_MWH_FUEL = {
    "ccgt": 0.1846,
    "ocgt": 0.1846,
    "coal": 0.3344,
}  # t/MWh on the basis of the fuel price
FUEL_COLUMN = {"ccgt": "natural gas", "ocgt": "natural gas", "coal": "hard coal"}


def defaults(tech: str, year: int) -> dict[str, float]:
    """The GB default of every parameter of ``tech``: the general row, or for availability the
    delivery year that starts in ``year`` (the nearest earlier one where it is missing)."""
    t = pd.read_csv(TABLE, dtype={"year": str})
    g = t[(t.gb_default == "yes") & (t.technology == tech)]
    out = {
        r.parameter: float(r.value)
        for r in g[g.year.isna() & g.commissioned_from.isna()].itertuples()
    }
    avail = g[g.parameter == "availability"].dropna(subset=["year"])
    if len(avail):
        avail = avail.assign(start=avail.year.str[:4].astype(int)).sort_values("start")
        earlier = avail[avail.start <= year]
        out["availability"] = float((earlier if len(earlier) else avail).iloc[-1].value)
    return out


def start_costs(
    tech: str, p: dict[str, float], year: int, fuel: float, co2: float
) -> dict[str, float]:
    """Hot, warm and cold start cost in GBP per MW of capacity."""
    fx = GBP_PER_EUR[year]
    per_mwh_fuel = fuel + co2 * CO2_PER_MWH_FUEL[tech]
    out = {}
    for kind in ("hot", "warm", "cold"):
        wear = p.get(f"start_up_cost_{kind}", 0.0) * fx
        if tech == "ocgt":
            fuel_mwh = p.get(
                "start_up_fuel", 0.0
            )  # Leigh Fisher, any start, taken as GCV
        else:
            fuel_mwh = p.get(f"start_up_fuel_{kind}", 0.0)
            if FUEL_COLUMN[tech] == "natural gas":
                fuel_mwh /= NCV_PER_GCV_GAS
        out[f"{kind}_start_cost"] = wear + fuel_mwh * per_mwh_fuel
    return out


def unit_table(
    year: int, availability: bool, source: str = "powerplant_units.csv"
) -> pd.DataFrame:
    """A units file of the scenario (``source``) with the non-convex parameters on its merchant
    thermal units. In a learning units file the portfolio agents' units keep their
    ``unit_operator``: their operator's strategy bids them; the others bid with the flexable one."""
    folder = paths.scenarios_dir() / paths.scenario_name(year)
    units = pd.read_csv(folder / source, float_precision="round_trip")
    prices = pd.read_csv(folder / "fuel_prices_df.csv", index_col=0, parse_dates=True)
    prices = prices[prices.index.year == year].mean()
    step_hours = 0.5
    units = units[units["name"] != MSG_UNIT].copy()
    for col in ("ramp_up", "ramp_down", "min_operating_time", "min_down_time", "hot_start_cost", "warm_start_cost", "cold_start_cost",
                "downtime_hot_start", "downtime_warm_start"):  # fmt: skip
        if col not in units:
            units[col] = 0.0
    for technology, tech in TECHNOLOGIES.items():
        rows = (units["technology"] == technology) & (units["kind"] == "thermal")
        if not rows.any():
            continue
        p = defaults(tech, year)
        if availability and "availability" in p:
            units.loc[rows, "max_power"] *= p["availability"]
        cap = units.loc[rows, "max_power"]
        msg = p["min_stable_generation"]
        units.loc[rows, "min_power"] = msg * cap
        units.loc[rows, "ramp_up"] = min(max(p["ramp_up_rate"] * 30, msg), 1.0) * cap
        units.loc[rows, "ramp_down"] = (
            min(max(p["ramp_down_rate"] * 30, msg), 1.0) * cap
        )
        units.loc[rows, "min_operating_time"] = max(
            1, math.ceil(p["min_up_time"] / step_hours - 1e-9)
        )
        units.loc[rows, "min_down_time"] = max(
            1, math.ceil(p["min_down_time"] / step_hours - 1e-9)
        )
        units.loc[rows, "downtime_hot_start"] = p["hot_start_max_offline"]
        units.loc[rows, "downtime_warm_start"] = p["warm_start_max_offline"]
        for k, v in start_costs(
            tech, p, year, float(prices[FUEL_COLUMN[tech]]), float(prices["co2"])
        ).items():
            units.loc[rows, k] = v
        units.loc[rows, "bidding_DA"] = STRATEGY
    return units


def write(year: int) -> pd.DataFrame:
    """Writes both units files and both study cases of ``year``; returns the parameters per
    technology for the record."""
    import argparse

    from assume_gb.__main__ import variant

    folder = paths.scenarios_dir() / paths.scenario_name(year)
    summary = []
    for suffix, availability in (("nc", False), ("nc_avail", True)):
        units = unit_table(year, availability)
        units.set_index("name").to_csv(folder / f"powerplant_units_{suffix}.csv")
        # the flexable strategy reads each unit's price forecast: price_naive_forecast puts the
        # merit-order forecast of forecasts_df.csv (price_DA) on the units; price_keep_given does not
        settings = [
            f"powerplant_units=powerplant_units_{suffix}.csv",
            "forecast_algorithms.price=price_naive_forecast",
        ]
        variant(
            argparse.Namespace(
                name=None,
                year=year,
                source="day_ahead",
                case=f"day_ahead_{suffix}",
                set=settings,
            )
        )
        changed = units[units["bidding_DA"] == STRATEGY]
        s = changed.groupby("technology").agg(
            units=("name", "size"), mw=("max_power", "sum"), min_power_share=("min_power", "sum"),
            min_up_steps=("min_operating_time", "first"), min_down_steps=("min_down_time", "first"),
            hot_gbp_per_mw=("hot_start_cost", "first"), warm_gbp_per_mw=("warm_start_cost", "first"), cold_gbp_per_mw=("cold_start_cost", "first"),
        )  # fmt: skip
        s["min_power_share"] /= s["mw"]
        summary.append(s.assign(year=year, case=f"day_ahead_{suffix}").reset_index())
    return pd.concat(summary)


def write_learning(
    year: int, source: str = "powerplant_units_learning.csv", suffix: str = "nc_avail"
) -> str:
    """The non-convex version (with the availability de-rating) of a learning units file, beside
    it as ``<source stem>_<suffix>.csv``; returns its name. The learning cases name it with
    ``powerplant_units``."""
    folder = paths.scenarios_dir() / paths.scenario_name(year)
    name = f"{Path(source).stem}_{suffix}.csv"
    unit_table(year, availability=suffix.endswith("avail"), source=source).set_index(
        "name"
    ).to_csv(folder / name)
    return name
