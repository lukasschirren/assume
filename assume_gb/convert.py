# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Offer stack -> ASSUME scenario folder.

The folder is the framework's ordinary CSV scenario (``assume.scenario.loader_csv``) and uses
only what the framework ships. What comes from the merit-order model is the fleet, what each
column of it can deliver in each period, the demand, and the numbers of its offer rules. What is
decided here is how each kind of stack column is said in ASSUME's terms:

    thermal           fuel, efficiency and emission factor; the start-up premium is an additional
                      cost per MWh. Bids with ``powerplant_energy_naive``.
    RO, FiT           ``support_scheme: premium``. The premium is band x ROC value, or the
                      generation tariff. Bids with ``powerplant_energy_naive_support``.
    CfD               ``support_scheme: cfd`` with the strike and the negative-price rule; a
                      baseload CfD names its reference price series (``support_reference``).
                      Under the any-hour rule the framework's strategy offers zero where the
                      merit-order model offers the avoidable cost: the one deliberate difference
                      (``model_offers``).
    must-run blocks   an additional cost equal to the block's offer (the cost a plant avoids by
                      staying on is negative). Energy from waste under a CfD offers the floor.
    merchant wind     an additional cost equal to the measured offer, zero
      and solar
    interconnector    a plant whose "fuel" is its own price series, the neighbouring market's
                      price net of losses. The export side is a step that backs off above that
                      price while the full export capacity sits in demand: the merit-order
                      model's own device, which a supply-side price rule clears correctly.
    anything else     an additional cost equal to its avoidable cost
    demand            one inelastic unit bidding the price cap
    unserved demand   a plant offering at the cap, so a period the fleet cannot cover clears at
                      the cap as it does in the merit-order model

``scenario_offers`` computes the offer every unit will make in the simulation, with the
framework's own marginal-cost formula and support arithmetic, so it can be compared with the
merit-order model's offers before anything is simulated (``check``).

The day-ahead auction is held once a day for the next day's periods, so the simulation starts
one day before the first delivery day; that lead-in day repeats the first day's inputs and
delivers nothing. Columns that offer under identical terms are merged into one unit unless
``merge=False``: the prices are the same and the scenario is five times smaller. The config
holds study cases on the same files, each for the whole stack and for its first week of delivery
(``*_week``): the day-ahead auction alone (``day_ahead``); with the merit-order model's battery
fleet as a storage unit that bids an optimised plan (``day_ahead_storage``, ``write_storage``);
and with the day-ahead auction bid on the day-ahead forecasts and an intraday auction that trades
the deviations (``day_ahead_intraday``, ``write_intraday``, given the forecast errors); and the
learning cases (``write_learning``), in which the units of the chosen technologies learn their bids
by reinforcement learning: ``learning`` trains them on the first weeks of the year, ``learning_year``
runs the whole year with the policies that training saved. The framework's learning strategies bid
the first product of an auction only, so these cases hold one auction per half-hour for that
half-hour, opening at the same lead as the day-ahead auction (``rolling_market_config``): with every
other unit bidding as before, the prices are the same as the day-ahead case's.
"""

import copy
import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from assume.strategies.support_strategies import (
    ANY_HOUR,
    CFD,
    PREMIUM,
    support_bid_price,
)
from assume_gb import forecasts
from assume_gb.stack import OfferStack, merit_order_price

MARKET_ID = "DA"
INTRADAY_MARKET_ID = "ID"
CASE = "day_ahead"
WEEK_CASE = "day_ahead_week"
STORAGE_CASE = "day_ahead_storage"
STORAGE_WEEK_CASE = "day_ahead_storage_week"
STORAGE_TRADERS_CASE = (
    "day_ahead_storage_traders"  # the battery fleet as one unit per trading company
)
STORAGE_TRADERS_WEEK_CASE = "day_ahead_storage_traders_week"
STORAGE_TRADERS_FILE = "storage_units_traders.csv"
INTRADAY_CASE = "day_ahead_intraday"
INTRADAY_WEEK_CASE = "day_ahead_intraday_week"
NAIVE = "powerplant_energy_naive"
SUPPORT = "powerplant_energy_naive_support"
FORECAST = "powerplant_energy_naive_forecast"
REBALANCE = "powerplant_energy_naive_rebalance"
DEMAND_FORECAST = "demand_energy_naive_forecast"
DEMAND_REBALANCE = "demand_energy_naive_rebalance"
STORAGE_UNIT = "battery"
STORAGE_STRATEGY = "storage_energy_optimization_schedule"
LEARNING_CASE = "learning"
LEARNING_WEEK_CASE = "learning_week"
LEARNING_YEAR_CASE = "learning_year"
# the portfolio learning cases whose portfolios hold the CfD units too (write_cfd_variant), beside the ones above
CFD_VARIANT = "_cfd"
LEARNING_CFD_CASE = "learning_cfd"
LEARNING_CFD_WEEK_CASE = "learning_cfd_week"
LEARNING_CFD_YEAR_CASE = "learning_cfd_year"
AVAILABILITY_CFD_FILE = "availability_df_cfd.csv.gz"
LEARNING = "powerplant_energy_learning"
LEARNING_SUPPORT = "powerplant_energy_learning_support"
PORTFOLIO_LEARNING = "portfolio_learning_support"  # the framework's portfolio_learning, aware of support contracts
LEARNING_TECHS = ("CCGT",)
DEMAND_UNIT = "demand"
UNSERVED_UNIT = "unserved_demand"
UNSERVED_MW = 100_000.0
NO_FUEL = "others"
GAS, COAL, OIL, CARBON, REFERENCE = "natural gas", "hard coal", "oil", "co2", "bmrp"
GAS_TECHS = ("CCGT", "OCGT", "EMB_FLEX")
COAL_TECHS = ("Hard Coal",)
INTERCONNECTOR_TECHS = ("IC_IMPORT", "IC_EXPORT")

# the licence header the repository's own config files carry (REUSE); the data files get theirs
# as ``<file>.license`` sidecars, which name the data's licence and are not written here
SPDX_HEADER = "# SPDX-FileCopyrightText: ASSUME Developers\n#\n# SPDX-License-Identifier: MIT\n\n"

SUPPORT_COLUMNS = [
    "support_scheme",
    "support_value",
    "support_neg_price_rule",
    "support_reference",
]
OFFER_COLUMNS = [
    "technology",
    "fuel_type",
    "emission_factor",
    "efficiency",
    "additional_cost",
    *SUPPORT_COLUMNS,
]


def _clean(value) -> str:
    return "" if pd.isna(value) else str(value)


def classify(tech: str, regime: str, cfd_class: str, c: dict) -> str:
    """The kind of offer a stack column makes, from its technology, support regime and CfD
    class. The order is the reverse of the order in which the merit-order model's
    ``bid_prices`` overwrites its bid vector, so the last write there is the first match here."""
    if tech in INTERCONNECTOR_TECHS:
        return "interconnector"
    if tech in c["thermal_techs"] or tech == "EMB_FLEX":
        return "thermal"
    if regime == "RO" and tech not in c["ro_mustrun_techs"]:
        return "ro"
    if cfd_class == "" and regime not in ("RO", "FiT") and tech in c["vre_techs"]:
        return "merchant_vre"
    if tech in must_run_offers(c):
        return "must_run"
    if regime == "FiT":
        return "fit"
    if cfd_class == "base" and c["baseload_cfd_enabled"]:
        return "cfd_baseload"
    if cfd_class == "int":
        return "cfd_intermittent"
    return "avoidable"


def must_run_offers(c: dict) -> dict[str, float]:
    return {
        "Nuclear": c["nuclear_bid"],
        "CCGT_MSG": c["ccgt_msg_bid"],
        "BIOMASS_MSG": c["biomass_msg_bid"],
        "EMB_MUSTRUN": c["embedded_mustrun_bid"],
    }


def unit_table(stack: OfferStack) -> pd.DataFrame:
    """One row per stack column: the ASSUME power-plant parameters that make it offer what the
    merit-order model offers."""
    c = stack.meta["constants"]
    floor = float(stack.meta["floor"])
    roc_value = float(stack.meta["roc_value_gbp_per_roc"])
    discount = float(stack.meta["clear_kwargs"].get("vre_discount", 0.0))
    rows = []
    for j, r in enumerate(stack.agents.itertuples(index=False)):
        tech, regime = str(r.tech), _clean(r.support_regime)
        cfd_class, neg_rule = _clean(r.cfd_class), _clean(r.neg_price_rule)
        kind = classify(tech, regime, cfd_class, c)
        avoidable = float(c["avoidable_gbp_mwh"].get(tech, 0.0))
        strike = (
            0.0 if pd.isna(r.support_value_gbp_mwh) else float(r.support_value_gbp_mwh)
        )
        row = {
            "name": str(r.unit_id),
            "technology": tech,
            "fuel_type": NO_FUEL,
            "emission_factor": 0.0,
            "efficiency": 1.0,
            "additional_cost": avoidable,
            "support_scheme": "",
            "support_value": 0.0,
            "support_neg_price_rule": "",
            "support_reference": "",
            "support_regime": regime,
            "kind": kind,
            "max_power": float(stack.mw[:, j].max()),
            "stack_columns": str(j),
        }
        if kind == "thermal":
            premium = (
                0.0
                if pd.isna(r.startup_premium_gbp_mwh)
                else float(r.startup_premium_gbp_mwh)
            )
            row["fuel_type"] = (
                GAS if tech in GAS_TECHS else COAL if tech in COAL_TECHS else OIL
            )
            usable = pd.notna(r.efficiency) and r.efficiency > 0
            row["efficiency"] = (
                float(r.efficiency) if usable else float(c["efficiency_fallback"])
            )
            row["emission_factor"] = (
                0.0 if pd.isna(r.emissions_factor) else float(r.emissions_factor)
            )
            row["additional_cost"] = premium + float(r.caprec_mu_gbp_mwh)
        elif kind == "interconnector":
            row["fuel_type"], row["additional_cost"] = str(r.unit_id), 0.0
        elif kind == "must_run":
            row["additional_cost"] = float(must_run_offers(c)[tech])
        elif kind == "merchant_vre":
            row["additional_cost"] = float(c["merchant_vre_bid"]) - discount
        elif kind == "ro":
            band = 0.0 if pd.isna(r.band) else float(r.band)
            row.update(support_scheme=PREMIUM, support_value=band * roc_value)
        elif kind == "fit":
            tariff = c["fit_tariff_gbp_mwh"].get(tech, c["fit_tariff_default_gbp_mwh"])
            row.update(support_scheme=PREMIUM, support_value=float(tariff))
        elif kind == "cfd_intermittent":
            if c["cfd_bid_basis"] == "floor" and not (
                c["ar_split_enabled"] and neg_rule == "any_hour"
            ):
                row["additional_cost"] = floor
            else:
                rule = neg_rule if c["ar_split_enabled"] else ""
                row.update(
                    support_scheme=CFD,
                    support_value=strike,
                    support_neg_price_rule=rule,
                )
        elif kind == "cfd_baseload":
            if c["efw_cfd_mustrun"] and tech == "Waste":
                row["additional_cost"] = floor
            else:
                row.update(
                    support_scheme=CFD,
                    support_value=strike,
                    support_reference=REFERENCE,
                )
        row[f"bidding_{MARKET_ID}"] = SUPPORT if row["support_scheme"] else NAIVE
        rows.append(row)
    return pd.DataFrame(rows)


def price_series(stack: OfferStack, units: pd.DataFrame) -> pd.DataFrame:
    """The price series the units' costs and offers are built from, one column each: the fuels,
    carbon, the baseload reference price and, per interconnector step, its neighbour price."""
    n = len(stack.index)
    out = {
        GAS: stack.series["gas_gbp_mwh_th"],
        COAL: stack.series["coal_gbp_mwh_th"],
        OIL: np.full(n, float(stack.meta["constants"]["oil_gbp_mwh_th"])),
        CARBON: stack.series["carbon_gbp_t"],
        REFERENCE: stack.series.get("bmrp", np.full(n, np.nan)),
    }
    for r in units[units["kind"] == "interconnector"].itertuples(index=False):
        out[r.name] = stack.bids[:, int(r.stack_columns)]
    return pd.DataFrame(out, index=stack.index)


def scenario_offers(stack: OfferStack, units: pd.DataFrame | None = None) -> np.ndarray:
    """The offer every stack column makes in the simulation, (periods, columns): the framework's
    power-plant marginal cost and, for a supported unit, its support arithmetic within the price
    limits. Comparable with ``stack.bids`` element by element."""
    units = unit_table(stack) if units is None else units
    prices = price_series(stack, units)
    floor, cap = float(stack.meta["floor"]), float(stack.meta["cap"])
    carbon = prices[CARBON].to_numpy()
    out = np.empty(stack.bids.shape)
    for j, unit in enumerate(units.itertuples(index=False)):
        fuel = prices[unit.fuel_type].to_numpy() if unit.fuel_type in prices else 0.0
        offer = (
            fuel + carbon * unit.emission_factor
        ) / unit.efficiency + unit.additional_cost
        if unit.support_scheme:
            reference = (
                prices[unit.support_reference].to_numpy()
                if unit.support_reference
                else None
            )
            offer = support_bid_price(
                offer,
                unit.support_scheme,
                unit.support_value,
                unit.support_neg_price_rule,
                reference,
            )
            offer = np.clip(offer, floor, cap)
        out[:, j] = offer
    return out


def any_hour_columns(units: pd.DataFrame) -> np.ndarray:
    """Columns under a CfD that pays nothing in any negative-price period: the one kind of offer
    where the scenario deliberately differs from the merit-order model."""
    return (
        (units["support_scheme"] == CFD) & (units["support_neg_price_rule"] == ANY_HOUR)
    ).to_numpy()


def model_offers(stack: OfferStack, units: pd.DataFrame | None = None) -> np.ndarray:
    """The merit-order model's offers with that one difference applied to them.

    The model offers such a unit at its avoidable cost. The contract pays again from a price of
    zero upwards, so the lowest price at which the unit wants to run is zero (or its avoidable
    cost if that is negative), and that is what the framework's strategy bids. Everywhere else
    this is ``stack.bids`` unchanged."""
    units = unit_table(stack) if units is None else units
    out = stack.bids.copy()
    columns = any_hour_columns(units)
    out[:, columns] = np.minimum(out[:, columns], 0.0)
    return out


def check_offers(stack: OfferStack) -> pd.DataFrame:
    """Per kind of offer: columns, capacity and the largest absolute difference between the
    scenario's offer and the merit-order model's, GBP/MWh. The last two rows are all columns
    against ``model_offers``, which must be zero, and the columns under the any-hour rule against
    the model's own offer, which is the size of the deliberate difference."""
    units = unit_table(stack)
    floor, cap = float(stack.meta["floor"]), float(stack.meta["cap"])
    offers = np.clip(scenario_offers(stack, units), floor, cap)
    deviation = np.abs(offers - np.clip(model_offers(stack, units), floor, cap))
    units["max_abs_dev"] = deviation.max(axis=0)
    out = units.groupby("kind").agg(
        columns=("name", "count"),
        max_gw=("max_power", lambda x: x.sum() / 1e3),
        max_abs_dev=("max_abs_dev", "max"),
    )
    out.loc["all"] = [len(units), units["max_power"].sum() / 1e3, deviation.max()]
    columns = any_hour_columns(units)
    raw = np.abs(offers[:, columns] - np.clip(stack.bids[:, columns], floor, cap))
    out.loc["any_hour_rule_vs_model"] = [
        columns.sum(),
        units.loc[columns, "max_power"].sum() / 1e3,
        raw.max() if raw.size else 0.0,
    ]
    return out


def _merge(
    units: pd.DataFrame, *arrays: np.ndarray
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Merge columns that offer under identical terms, summing the given (periods, columns) arrays
    over the merged columns. Interconnector steps each have their own price series, so they never
    merge; nor do the named wind farms (``FARM_``), which keep their identity for their owners'
    portfolios. The capacity of a merged unit is the largest sum of the first array (the MW offered)."""
    key = (
        units[OFFER_COLUMNS + ["support_regime", "kind"]]
        .astype(str)
        .agg("|".join, axis=1)
    )
    named = (units["kind"] == "interconnector") | units["name"].astype(
        str
    ).str.startswith("FARM_")
    key = key.where(~named, units["name"])
    rows, merged, counter = [], [[] for _ in arrays], {}
    for members in (g.index.to_numpy() for _, g in units.groupby(key, sort=False)):
        first = units.loc[members[0]].copy()
        if len(members) > 1:
            base = f"{first['kind']}_{first['technology']}".replace(" ", "_").lower()
            counter[base] = counter.get(base, 0) + 1
            first["name"] = f"{base}_{counter[base]:03d}"
            first["stack_columns"] = ";".join(units.loc[members, "stack_columns"])
        totals = [array[:, members].sum(axis=1) for array in arrays]
        first["max_power"] = float(totals[0].max())
        rows.append(first)
        for column, total in zip(merged, totals):
            column.append(total)
    return pd.DataFrame(rows).reset_index(drop=True), [
        np.column_stack(c) for c in merged
    ]


def _pad(values: np.ndarray, lead: int) -> np.ndarray:
    """The lead-in day in front (a repeat of the first day) and one closing step behind."""
    values = np.asarray(values)
    return np.concatenate([values[:lead], values, values[-1:]])


def market_config(
    stack: OfferStack,
    start: pd.Timestamp,
    gate: str,
    gate_minutes: int,
    market_id: str = MARKET_ID,
    mechanism: str = "pay_as_clear",
) -> dict:
    """One auction a day, opening at ``gate`` on the day before delivery, for all periods of the
    delivery day. A ``complex_clearing`` market prices from the duals of its optimisation, which
    is right when demand bids set the price too, as in a market that trades deviations."""
    first_delivery = pd.Timedelta("1D") - pd.Timedelta(gate)
    config = {
        "operator": "EOM_operator",
        "product_type": "energy",
        "start_date": (start + pd.Timedelta(gate)).strftime("%Y-%m-%d %H:%M"),
        "products": [
            {
                "duration": f"{int(stack.freq / pd.Timedelta('1min'))}min",
                "count": int(pd.Timedelta("1D") / stack.freq),
                "first_delivery": f"{int(first_delivery / pd.Timedelta('1min'))}min",
            }
        ],
        "opening_frequency": "24h",
        "opening_duration": f"{gate_minutes}min",
        "volume_unit": "MW",
        "maximum_bid_volume": 1_000_000,
        "maximum_bid_price": float(stack.meta["cap"]),
        "minimum_bid_price": float(stack.meta["floor"]),
        "price_unit": "GBP/MWh",
        "market_mechanism": mechanism,
    }
    if mechanism == "complex_clearing":
        config["param_dict"] = {
            "solver_name": "highs",
            "pricing_mechanism": "pay_as_clear",
        }
        config["additional_fields"] = ["bid_type"]
    return {market_id: config}


def rolling_market_config(
    stack: OfferStack, start: pd.Timestamp, gate: str, gate_minutes: int
) -> dict:
    """The day-ahead market as one auction per period for that period, each opening at the lead
    of the day-ahead auction (``gate`` on the day before delivery for the first period of a day),
    for the learning cases: the framework's learning strategies bid the first product of an
    auction only. With no inter-temporal constraint on any unit the prices are those of the
    day-ahead auction."""
    config = market_config(stack, start, gate, gate_minutes)[MARKET_ID]
    config["products"] = [dict(config["products"][0], count=1)]
    config["opening_frequency"] = f"{int(stack.freq / pd.Timedelta('1min'))}min"
    return {MARKET_ID: config}


def portfolio_params(nbins: int, max_markup: float) -> dict:
    """The parameters of the framework's portfolio learning strategy (``bidding_strategy_params``
    of a study case, handed to every strategy of the case): an agent per owner bids each of its
    plants at marginal cost times a mark-up it chooses per cost bin, between 1 and ``max_markup``;
    an owner needs at least ``nbins`` plants."""
    return {
        "nbins": int(nbins),
        "min_markup": 1.0,
        "max_markup": float(max_markup),
        "foresight": 12,
        "steps": 1,
    }


def learning_config(
    max_bid_price: float,
    training_episodes: int,
    initial_episodes: int,
    validation_interval: int,
    load_path: str | None = None,
) -> dict:
    """The framework's learning configuration for the GB cases. Training (``load_path`` None): the
    actors act on prices up to ``max_bid_price`` and train on a day of experience at a time (ten
    gradient steps per day: a step costs about 45 ms per learning unit on a CPU, so 37 units and
    twenty steps made a four-week episode take a quarter of an hour); the
    policies go to ``learned_strategies/<scenario>_<case>`` in the scenario folder, the best ones
    by validation reward to its ``avg_reward_eval_policies``. With ``load_path`` the saved policies
    bid without noise and nothing is trained."""
    return {
        "learning_mode": load_path is None,
        "continue_learning": False,
        "trained_policies_save_path": None,
        "trained_policies_load_path": load_path,
        "max_bid_price": float(max_bid_price),
        "algorithm": "matd3",
        "actor_architecture": "mlp",
        "learning_rate": 0.001,
        "training_episodes": int(training_episodes),
        "episodes_collecting_initial_experience": int(initial_episodes),
        "validation_episodes_interval": int(validation_interval),
        "train_freq": "24h",
        "gradient_steps": 10,
        "batch_size": 128,
        "gamma": 0.99,
        "device": "cpu",
        "action_noise_schedule": "linear",
        "noise_sigma": 0.1,
        "noise_scale": 1,
        "noise_dt": 1,
    }


def default_max_bid_price(merit_order: np.ndarray, cap: float) -> float:
    """The highest price the learning actors can bid: the highest merit-order price of the year
    below the cap, rounded up to the next hundred."""
    prices = merit_order[np.isfinite(merit_order) & (merit_order < cap)]
    return float(np.ceil(prices.max() / 100.0) * 100.0) if prices.size else cap


def is_learning_case(folder: Path, case: str) -> bool:
    """Whether study case ``case`` of the scenario folder trains learning agents."""
    config = yaml.safe_load((Path(folder) / "config.yaml").read_text(encoding="utf-8"))
    return bool(config.get(case, {}).get("learning_config", {}).get("learning_mode"))


def build_scenario(
    stack: OfferStack,
    out_dir: Path,
    *,
    merge: bool = True,
    gate: str = "09:00:00",
    gate_minutes: int = 20,
    forecast_errors: pd.DataFrame | None = None,
    intraday_gate: str = "17:30:00",
    observed: pd.DataFrame | None = None,
    learning_techs: tuple[str, ...] = LEARNING_TECHS,
    learning_count: int | None = None,
    learning_pick: str = "largest",
    learning_owners: pd.Series | None = None,
    learning_agent_level: str | None = None,
    nbins: int = 2,
    min_portfolio_mw: float = 500.0,
    max_markup: float = 2.0,
    learning_weeks: int = 4,
    max_bid_price: float | None = None,
    levy: tuple[float, float] | None = None,
    battery_traders: pd.Series | None = None,
) -> Path:
    """Write the scenario folder for ``stack`` and return its path.

    Args:
        merge: merge columns that offer under identical terms into one unit.
        gate: time of day the day-ahead auction opens on the day before delivery.
        gate_minutes: how long an auction stays open.
        forecast_errors: the day-ahead forecast errors (``forecasts.errors``); with them the folder
            also holds the intraday cases, in which the day-ahead auction is bid on the forecasts
            and an intraday auction, opening at ``intraday_gate`` on the day before delivery,
            trades the deviations.
        observed: further observed prices for ``observed_prices.csv`` (``observed.prices``).
        learning_techs: the technologies whose units learn their bids in the learning cases; none
            for no learning cases.
        learning_count: only so many of those units learn (all if None): the largest, or with
            ``learning_pick="cheapest"`` the ones with the lowest marginal cost at the year's mean
            prices, which are nearest the margin.
        learning_owners: with ``learning_pick="portfolio"``, the agent of every unit that has
            one (``owners.portfolio_agents``): an agent with at least ``nbins`` units of the
            technologies and ``min_portfolio_mw`` of their capacity becomes one learning agent that
            bids its whole portfolio (``portfolio_learning_support``), with mark-ups on marginal
            cost up to ``max_markup``; the units themselves bid as in the day-ahead case.
        learning_agent_level: who the agents of ``learning_owners`` are (``ownership.AGENT_LEVELS``:
            "trader", the company that bids the unit, or "operator", the one that runs it),
            recorded in ``scenario_meta.json``.
        learning_weeks: the length of the training period, from the first delivery day.
        max_bid_price: the highest price a learning unit can bid; by default the highest
            merit-order price of the year below the cap, rounded up to the next hundred.
        levy: (rate, benchmark price) of a levy on generation revenue, put on the in-scope units
            (``egl.unit_parameters``): the framework books it per period beside the market
            cashflow and the learning rewards see it. None for no levy.
        battery_traders: the share of the battery fleet each company trades (``ownership.battery_traders``,
            from the BM register); with it the folder also holds the case ``day_ahead_storage_traders``, in
            which the fleet is one storage unit per trading company (``storage_units_traders.csv``,
            ``unit_operator`` = the company). The units bid as the single fleet does, on the same forecast,
            so the case is the agent set for trader-level behaviour, not a different market. None for none.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    freq = stack.freq
    lead = int(pd.Timedelta("1D") / freq)
    start, end = stack.index[0] - pd.Timedelta("1D"), stack.index[-1] + freq
    index = pd.date_range(start, end, freq=freq, name="datetime")
    cap = float(stack.meta["cap"])

    units = unit_table(stack)
    prices = price_series(stack, units)
    keep = (units["max_power"] > 0).to_numpy()
    units = units[keep].reset_index(drop=True)
    arrays = [stack.mw[:, keep]]
    if forecast_errors is not None:
        forecast_mw, forecast_demand = forecasts.apply(stack, forecast_errors)
        arrays.append(forecast_mw[:, keep])
    if merge:
        units, arrays = _merge(units, *arrays)
    mw = arrays[0]
    units["min_power"] = 0.0
    units["unit_operator"] = (
        (units["kind"] + "_" + units["technology"]).str.replace(" ", "_").str.lower()
    )

    unserved = {c: "" for c in units.columns}
    unserved.update(
        name=UNSERVED_UNIT, technology=UNSERVED_UNIT, fuel_type=NO_FUEL, emission_factor=0.0,
        efficiency=1.0, additional_cost=cap, support_value=0.0, kind=UNSERVED_UNIT,
        max_power=UNSERVED_MW, min_power=0.0, unit_operator=UNSERVED_UNIT,
    )  # fmt: skip
    unserved[f"bidding_{MARKET_ID}"] = NAIVE
    # which stack columns each unit stands for, for attributing a merged class to its plants;
    # the model's exports carry each column's capacity, a stack without it gets the largest offer
    if "capacity_mw" in stack.agents.columns:
        capacity = stack.agents["capacity_mw"].to_numpy(dtype=float)
    else:
        capacity = stack.mw.max(axis=0)
    columns = pd.DataFrame(
        {
            "name": units["name"],
            "stack_columns": units["stack_columns"],
            "stack_unit_ids": [
                ";".join(
                    stack.agents["unit_id"]
                    .astype(str)
                    .iloc[[int(j) for j in str(cols).split(";")]]
                )
                for cols in units["stack_columns"]
            ],
            "column_capacity_mw": [
                ";".join(f"{capacity[int(j)]:.3f}" for j in str(cols).split(";"))
                for cols in units["stack_columns"]
            ],
        }
    )
    columns.set_index("name").to_csv(out_dir / "unit_columns.csv")
    table = pd.concat([units, pd.DataFrame([unserved])], ignore_index=True).drop(
        columns="stack_columns"
    )
    if levy is not None:
        from assume_gb import egl  # the levy module reads this one

        table = egl.unit_parameters(
            table.set_index("name"), levy[0], levy[1]
        ).reset_index()
    table.set_index("name").to_csv(out_dir / "powerplant_units.csv")

    share = mw / units["max_power"].to_numpy()
    varies = ~np.isclose(share, 1.0, rtol=0.0, atol=1e-12).all(axis=0)
    availability = pd.DataFrame(
        _pad(share[:, varies], lead),
        index=index,
        columns=units["name"].to_numpy()[varies],
    )
    availability.clip(0.0, 1.0).to_csv(
        out_dir / "availability_df.csv.gz", float_format="%.10g"
    )

    # Prices and demand are written at full precision: an offer is a function of the prices, and
    # a rounded one moves the clearing price by the rounding. Availability is the large file and
    # only moves a price where demand sits exactly on the edge of an offer.
    pd.DataFrame(
        _pad(prices.to_numpy(), lead), index=index, columns=prices.columns
    ).to_csv(out_dir / "fuel_prices_df.csv")
    demand = np.clip(
        stack.demand, 0.0, None
    )  # a corrupt-demand period can be negative; ASSUME takes |x|
    pd.DataFrame({DEMAND_UNIT: _pad(demand, lead)}, index=index).to_csv(
        out_dir / "demand_df.csv"
    )
    demand_unit = {
        "name": DEMAND_UNIT, "technology": "inflex_demand", f"bidding_{MARKET_ID}": "demand_energy_naive",
        "max_power": 1_000_000, "min_power": 0, "unit_operator": DEMAND_UNIT, "price": cap,
    }  # fmt: skip
    pd.DataFrame([demand_unit]).set_index("name").to_csv(out_dir / "demand_units.csv")

    merit_order = scenario_merit_order(stack)
    write_reference(stack, out_dir, merit_order, observed)
    # the price forecast of the battery and of the learning agents: the merit order of the
    # scenario's own offers without storage, in every period (a plan or an observation cannot be
    # made on a missing price)
    pd.DataFrame({f"price_{MARKET_ID}": _pad(merit_order, lead)}, index=index).to_csv(
        out_dir / "forecasts_df.csv"
    )
    fleet = write_storage(stack, out_dir)
    traders_written = (
        bool(fleet)
        and battery_traders is not None
        and write_storage_traders(out_dir, battery_traders)
    )
    if forecast_errors is not None:
        write_intraday(
            out_dir,
            table,
            demand_unit,
            units,
            mw,
            arrays[1],
            forecast_demand,
            lead,
            index,
        )
    learning_units, learning_operators = (
        write_learning(
            out_dir,
            table,
            learning_techs,
            learning_count,
            learning_pick,
            prices.mean(),
            learning_owners,
            nbins,
            min_portfolio_mw,
        )
        if learning_techs
        else ([], [])
    )
    if max_bid_price is None:
        max_bid_price = default_max_bid_price(merit_order, cap)

    def study_case(
        case_end: pd.Timestamp,
        storage: bool = False,
        intraday: bool = False,
        learning: dict | None = None,
    ) -> dict:
        case = {
            "start_date": start.strftime("%Y-%m-%d %H:%M"),
            "end_date": case_end.strftime("%Y-%m-%d %H:%M"),
            "time_step": f"{int(freq / pd.Timedelta('1min'))}min",
            "save_frequency_hours": None,
            "availability_df": "availability_df.csv.gz",
            "markets_config": market_config(stack, start, gate, gate_minutes),
        }
        if learning is not None:
            # one auction per period (see rolling_market_config); the agents observe the price
            # forecast given in forecasts_df and the residual load the framework derives from the
            # demand and the wind and solar units
            case["markets_config"] = rolling_market_config(
                stack, start, gate, gate_minutes
            )
            case["powerplant_units"] = "powerplant_units_learning.csv"
            case["storage_units"] = None
            if learning_operators:
                case["unit_operators"] = "unit_operators_learning.csv"
                case["bidding_strategy_params"] = portfolio_params(nbins, max_markup)
            case["forecast_algorithms"] = {
                "price": "price_naive_forecast",
                "residual_load": "residual_load_naive_forecast",
            }
            case["learning_config"] = learning
        elif storage:
            # The battery bids on the price forecast in forecasts_df; the framework takes a given
            # forecast before it calculates its own.
            case["forecast_algorithms"] = {
                "price": "price_naive_forecast",
                "residual_load": "residual_load_keep_given",
            }
        else:
            # The framework's default price forecast is a merit order of marginal costs. No
            # strategy of these cases reads it and it ignores the support contracts.
            case["storage_units"] = None
            case["forecasts_df"] = None
            case["forecast_algorithms"] = {
                "price": "price_keep_given",
                "residual_load": "residual_load_keep_given",
            }
        if learning is None:
            case["unit_operators"] = None
        if intraday:
            case["powerplant_units"] = "powerplant_units_intraday.csv"
            case["demand_units"] = "demand_units_intraday.csv"
            case["availability_forecast_df"] = "availability_forecast_df.csv.gz"
            case["demand_forecast_df"] = "demand_forecast_df.csv"
            case["markets_config"].update(
                market_config(
                    stack,
                    start,
                    intraday_gate,
                    gate_minutes,
                    INTRADAY_MARKET_ID,
                    "complex_clearing",
                )
            )
        else:
            case["availability_forecast_df"] = None
            case["demand_forecast_df"] = None
        return case

    # The short cases deliver the first week only: seconds to run, and what the tests simulate.
    week_end = min(stack.index[0] + pd.Timedelta("7D"), end)
    config = {CASE: study_case(end), WEEK_CASE: study_case(week_end)}
    if fleet:
        config[STORAGE_CASE] = study_case(end, storage=True)
        config[STORAGE_WEEK_CASE] = study_case(week_end, storage=True)
    if traders_written:
        config[STORAGE_TRADERS_CASE] = {
            **study_case(end, storage=True),
            "storage_units": STORAGE_TRADERS_FILE,
        }
        config[STORAGE_TRADERS_WEEK_CASE] = {
            **study_case(week_end, storage=True),
            "storage_units": STORAGE_TRADERS_FILE,
        }
    if forecast_errors is not None:
        config[INTRADAY_CASE] = study_case(end, intraday=True)
        config[INTRADAY_WEEK_CASE] = study_case(week_end, intraday=True)
    if learning_units:
        # training on the first weeks (the week case: a few episodes, to try the set-up), then the
        # year with the best policies of the training case
        training_end = min(stack.index[0] + pd.Timedelta(weeks=learning_weeks), end)
        policies = f"learned_strategies/{out_dir.name}_{LEARNING_CASE}/avg_reward_eval_policies"
        config[LEARNING_CASE] = study_case(
            training_end, learning=learning_config(max_bid_price, 50, 5, 5)
        )
        config[LEARNING_WEEK_CASE] = study_case(
            week_end, learning=learning_config(max_bid_price, 4, 2, 2)
        )
        config[LEARNING_YEAR_CASE] = study_case(
            end, learning=learning_config(max_bid_price, 50, 5, 5, load_path=policies)
        )
    (out_dir / "config.yaml").write_text(
        SPDX_HEADER + yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    all_units = unit_table(stack)
    any_hour = any_hour_columns(all_units)
    meta = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": {
            k: stack.meta.get(k) for k in ("year", "design", "version", "tag", "git")
        },
        "merged": merge,
        "units": int(len(table)),
        "stack_columns": int(stack.bids.shape[1]),
        # where the scenario's offers differ from the merit-order model's by design (model_offers)
        "any_hour_rule": {
            "stack_columns": int(any_hour.sum()),
            "max_mw": float(all_units.loc[any_hour, "max_power"].sum()),
        },
        "delivery_start": str(stack.index[0]),
        "delivery_end": str(end),
        "lead_in_steps": lead,
        "market": MARKET_ID,
        "cases": list(config),
        "battery_fleet": fleet,
        "learning": {
            "units": learning_units,
            "technologies": list(learning_techs),
            "count": learning_count,
            "pick": learning_pick,
            "operators": learning_operators,
            "agent_level": learning_agent_level if learning_operators else None,
            "min_portfolio_mw": float(min_portfolio_mw) if learning_operators else None,
            "portfolio": portfolio_params(nbins, max_markup)
            if learning_operators
            else None,
            "max_bid_price": float(max_bid_price),
            "training_weeks": learning_weeks,
        }
        if learning_units
        else None,
        "roc_value_gbp_per_roc": stack.meta["roc_value_gbp_per_roc"],
        "levy": None if levy is None else {"rate": levy[0], "benchmark": levy[1]},
        "forecast_errors": None
        if forecast_errors is None
        else forecasts.summary(forecast_errors.reindex(stack.index)),
    }
    (out_dir / "scenario_meta.json").write_text(
        json.dumps(meta, indent=1), encoding="utf-8"
    )
    return out_dir


def write_intraday(
    out_dir: Path,
    table: pd.DataFrame,
    demand_unit: dict,
    units: pd.DataFrame,
    mw: np.ndarray,
    forecast_mw: np.ndarray,
    forecast_demand: np.ndarray,
    lead: int,
    index: pd.DatetimeIndex,
) -> None:
    """The files of the intraday cases: the units with a day-ahead strategy that bids the forecast
    and an intraday strategy that trades the deviation, and the forecasts themselves.

    The forecast of a unit is written where it differs from the outturn (wind and solar); every
    other unit is as well known the day before as on the day."""
    intraday = table.copy()
    intraday[f"bidding_{MARKET_ID}"] = FORECAST
    intraday[f"bidding_{INTRADAY_MARKET_ID}"] = REBALANCE
    intraday.set_index("name").to_csv(out_dir / "powerplant_units_intraday.csv")
    demand = dict(
        demand_unit,
        **{
            f"bidding_{MARKET_ID}": DEMAND_FORECAST,
            f"bidding_{INTRADAY_MARKET_ID}": DEMAND_REBALANCE,
        },
    )
    pd.DataFrame([demand]).set_index("name").to_csv(
        out_dir / "demand_units_intraday.csv"
    )

    differs = (np.abs(forecast_mw - mw) > 1e-9).any(axis=0)
    share = np.clip(
        forecast_mw[:, differs] / units["max_power"].to_numpy()[differs], 0.0, 1.0
    )
    pd.DataFrame(
        _pad(share, lead), index=index, columns=units["name"].to_numpy()[differs]
    ).to_csv(out_dir / "availability_forecast_df.csv.gz", float_format="%.10g")
    pd.DataFrame(
        {DEMAND_UNIT: _pad(np.clip(forecast_demand, 0.0, None), lead)}, index=index
    ).to_csv(out_dir / "demand_forecast_df.csv")


def scenario_merit_order(stack: OfferStack) -> np.ndarray:
    """The merit order of the offers the scenario's units will make, in every period of the stack."""
    floor, cap = float(stack.meta["floor"]), float(stack.meta["cap"])
    offers = np.clip(scenario_offers(stack), floor, cap)
    return merit_order_price(
        offers, stack.mw, np.clip(stack.demand, 0.0, None), floor, cap
    )


def write_reference(
    stack: OfferStack,
    out_dir: Path,
    merit_order: np.ndarray,
    observed: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The prices a run is compared with, on the delivery periods.

    ``scenario_merit_order`` is ``merit_order``, the merit order of the offers the scenario's units
    will make (``scenario_merit_order``): what the framework must reproduce to rounding, left empty
    where there is no demand to clear. ``model_no_storage`` and ``model_with_battery`` are
    the merit-order model's own prices; the first equals ``scenario_merit_order`` unless the stack
    has columns under the any-hour rule (``model_offers``). The observed day-ahead auction price
    and the further observed prices given in ``observed`` (``observed.prices``) go in a
    file of their own, which the folder's ``.gitignore`` keeps out of version control, as some of
    them are licensed exchange data."""
    index = stack.index.rename("datetime")
    # a period without demand (corrupt input, flagged by the model) clears nothing: no price to compare
    reference = {
        "scenario_merit_order": np.where(stack.demand > 0, merit_order, np.nan)
    }
    if "price_no_storage" in stack.series:
        reference["model_no_storage"] = stack.series["price_no_storage"]
    if "price" in stack.series:
        reference["model_with_battery"] = stack.series["price"]
    reference["admissible"] = ~np.logical_or.reduce(
        list(stack.flags.values()) or [np.zeros(len(index), bool)]
    )
    reference = pd.DataFrame(reference, index=index)
    reference.to_csv(out_dir / "reference_prices.csv")
    observed_prices = (
        pd.DataFrame(index=index) if observed is None else observed.reindex(index)
    )
    if "da_price" in stack.series and np.isfinite(stack.series["da_price"]).any():
        venue = str(stack.meta.get("venue", "day_ahead"))
        observed_prices.insert(0, f"{venue}_day_ahead", stack.series["da_price"])
    if not observed_prices.empty:
        observed_prices.to_csv(out_dir / "observed_prices.csv")
        ignore = "# licensed exchange data: keep out of version control\nobserved_prices.csv\n"
        (out_dir / ".gitignore").write_text(SPDX_HEADER + ignore, encoding="utf-8")
    return reference


def write_learning(
    out_dir: Path,
    table: pd.DataFrame,
    learning_techs: tuple[str, ...],
    learning_count: int | None = None,
    learning_pick: str = "largest",
    mean_prices: pd.Series | None = None,
    owners: pd.Series | None = None,
    nbins: int = 2,
    min_portfolio_mw: float = 0.0,
    variant: str = "",
) -> tuple[list[str], list[str]]:
    """The units of the learning cases. With ``learning_pick`` "largest" or "cheapest", the units
    of the chosen technologies learn their bids one by one (a unit with a support contract with
    the strategy that knows the contract), or ``learning_count`` of them, the largest or the
    cheapest (by marginal cost at the year's mean fuel and carbon prices, ``mean_prices``: the
    plants nearest the margin, which set prices most often). With "portfolio", every owner in
    ``owners`` with at least ``nbins`` units of the technologies becomes one learning agent that
    bids its whole portfolio (``unit_operators_learning.csv``), and its units keep their day-ahead
    strategies. Every other unit bids as in the day-ahead case. ``variant`` names the files of
    other learning cases (``learning_files``). Returns the names of the learning units and of the
    learning operators."""
    plants_file, operators_file = learning_files(variant)
    learning = table.copy()
    learns = learning["technology"].isin(learning_techs) & (
        learning["name"] != UNSERVED_UNIT
    )
    if learning_pick == "portfolio":
        if owners is None:
            raise ValueError("learning_pick 'portfolio' needs the owners of the units")
        owner = learning["name"].map(owners).where(learns)
        counts = owner.value_counts()
        capacity = learning["max_power"].groupby(owner).sum()
        owner = owner.where(
            (owner.map(counts) >= nbins) & (owner.map(capacity) >= min_portfolio_mw)
        )
        learns = owner.notna()
        learning.loc[learns, "unit_operator"] = owner[learns]
        operators = sorted(owner.dropna().unique())
        pd.DataFrame(
            {"name": operators, f"bidding_{MARKET_ID}": PORTFOLIO_LEARNING}
        ).set_index("name").to_csv(out_dir / operators_file)
        learning.set_index("name").to_csv(out_dir / plants_file)
        return learning.loc[learns, "name"].tolist(), operators
    if learning_count is not None:
        if learning_pick == "largest":
            rank = learning.loc[learns, "max_power"].sort_values(ascending=False)
        elif learning_pick == "cheapest":
            candidates = learning[learns]
            fuel = (
                candidates["fuel_type"]
                .map(lambda f: mean_prices.get(f, 0.0))
                .astype(float)
            )
            cost = (
                fuel + mean_prices.get("co2", 0.0) * candidates["emission_factor"]
            ) / candidates["efficiency"] + candidates["additional_cost"]
            rank = cost.sort_values()
        else:
            raise ValueError(
                f"learning_pick must be 'largest' or 'cheapest', not {learning_pick!r}"
            )
        learns = learns & learning.index.isin(rank.index[:learning_count])
    supported = learning["support_scheme"].fillna("").astype(str).str.len() > 0
    learning.loc[learns & ~supported, f"bidding_{MARKET_ID}"] = LEARNING
    learning.loc[learns & supported, f"bidding_{MARKET_ID}"] = LEARNING_SUPPORT
    learning.set_index("name").to_csv(out_dir / plants_file)
    return learning.loc[learns, "name"].tolist(), []


def learning_files(variant: str = "") -> tuple[str, str]:
    """The files of the learning cases, their units and their portfolio agents: ``powerplant_units_learning.csv`` and
    ``unit_operators_learning.csv``, or with ``variant`` (``CFD_VARIANT``) ``powerplant_units_learning_cfd.csv`` and
    ``unit_operators_learning_cfd.csv``."""
    return (
        f"powerplant_units_learning{variant}.csv",
        f"unit_operators_learning{variant}.csv",
    )


def rewrite_learning(
    out_dir: Path, owners: pd.Series, agent_level: str, min_portfolio_mw: float = 500.0
) -> tuple[list[str], list[str]]:
    """Give a built scenario's portfolio learning cases new agents: ``owners`` (unit name -> agent, from
    ``owners.portfolio_agents``) at ``agent_level``, with the folder's own technologies and bins and its portfolio
    threshold (``min_portfolio_mw`` for a folder that records none, as before 7 Oct 2026). Rewrites
    ``powerplant_units_learning.csv`` and ``unit_operators_learning.csv`` from the folder's ``powerplant_units.csv``,
    and the learning block of ``scenario_meta.json``; nothing else changes, since the study cases read the same files.
    Policies trained on the former agents no longer fit. Returns the learning units and agents."""
    out_dir = Path(out_dir)
    meta_path = out_dir / "scenario_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    learning = meta.get("learning") or {}
    if learning.get("pick") != "portfolio":
        raise ValueError(
            f"{out_dir.name} has no portfolio learning cases (build it with --learning-pick portfolio)"
        )
    threshold = (
        min_portfolio_mw
        if learning.get("min_portfolio_mw") is None
        else learning["min_portfolio_mw"]
    )
    # read exactly as written (the default parser can miss the last digit), so the other columns stay as they are
    table = pd.read_csv(out_dir / "powerplant_units.csv", float_precision="round_trip")
    with tempfile.TemporaryDirectory(
        dir=out_dir
    ) as tmp:  # the folder keeps its files unless every one is written
        units, operators = write_learning(
            Path(tmp),
            table,
            tuple(learning["technologies"]),
            learning["count"],
            "portfolio",
            None,
            owners,
            learning["portfolio"]["nbins"],
            threshold,
        )
        if not operators:
            raise ValueError(
                f"{out_dir.name}: no agent has a portfolio at {agent_level} level; the learning cases would have no learning units"
            )
        for name in learning_files():
            os.replace(Path(tmp) / name, out_dir / name)
    learning.update(units=units, operators=operators, agent_level=agent_level, min_portfolio_mw=float(threshold),
                    agents_written=time.strftime("%Y-%m-%d %H:%M:%S"))  # fmt: skip
    meta["learning"] = learning
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return units, operators


def split_cfd_units(
    table: pd.DataFrame, columns: pd.DataFrame, stack: OfferStack
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, list[str]]]:
    """``table`` (a folder's units) with every merged CfD unit split back into its stack columns: one unit per
    contract, named by its stack unit id, on the merged unit's terms and with its own capacity, as an unmerged build
    writes it. ``columns`` is the folder's ``unit_columns.csv`` (index: unit name). Returns the table, the
    availability share of every new unit whose availability varies (name -> array over the stack's periods) and the
    split (merged name -> its units). ValueError where the stack is not the one the folder was built from (the
    columns would not add up to the merged unit)."""
    ids = stack.agents["unit_id"].astype(str).to_numpy()
    rows, shares, split = [], {}, {}
    for _, row in table.iterrows():
        members = [
            int(j)
            for j in str(columns["stack_columns"].get(row["name"], "")).split(";")
            if j
        ]
        if not (str(row["kind"]).startswith("cfd") and len(members) > 1):
            rows.append(row)
            continue
        recorded = str(columns["stack_unit_ids"].get(row["name"], "")).split(";")
        if max(members) >= len(ids) or [ids[j] for j in members] != recorded:
            raise ValueError(
                f"{row['name']}: the stack's columns {members} are not {recorded}: not the stack the folder was built from"
            )
        total = stack.mw[:, members].sum(axis=1).max()
        if not np.isclose(total, float(row["max_power"]), rtol=1e-9, atol=1e-6):
            raise ValueError(
                f"{row['name']}: its stack columns add up to {total:.3f} MW, not {float(row['max_power']):.3f} MW: not the stack the folder was built from"
            )
        for j in members:
            unit = row.copy()
            unit["name"], unit["max_power"] = ids[j], float(stack.mw[:, j].max())
            rows.append(unit)
            share = stack.mw[:, j] / unit["max_power"]
            if not np.isclose(share, 1.0, rtol=0.0, atol=1e-12).all():
                shares[ids[j]] = share
        split[row["name"]] = [ids[j] for j in members]
    return pd.DataFrame(rows).reset_index(drop=True), shares, split


def cfd_cases(config: dict, scenario: str, availability: str | None) -> dict:
    """The learning cases of the CfD variant, from the folder's own learning cases: the same settings on the
    variant's units, agents and availability, and the year case on the variant's policies."""
    plants, operators = learning_files(CFD_VARIANT)
    cases = {}
    for source, case in (
        (LEARNING_CASE, LEARNING_CFD_CASE),
        (LEARNING_WEEK_CASE, LEARNING_CFD_WEEK_CASE),
        (LEARNING_YEAR_CASE, LEARNING_CFD_YEAR_CASE),
    ):
        if source not in config:
            continue
        # a copy, not shared settings: yaml would write a shared dict as an anchor in the source case
        cases[case] = copy.deepcopy(config[source])
        cases[case].update(powerplant_units=plants, unit_operators=operators)
        if availability:
            cases[case]["availability_df"] = availability
        settings = cases[case].get("learning_config") or {}
        if settings.get("trained_policies_load_path"):
            settings["trained_policies_load_path"] = (
                f"learned_strategies/{scenario}_{LEARNING_CFD_CASE}/avg_reward_eval_policies"
            )
    return cases


def write_cfd_variant(
    out_dir: Path,
    stack: OfferStack,
    owners: pd.Series,
    agent_level: str,
    min_portfolio_mw: float = 500.0,
) -> tuple[list[str], list[str]]:
    """Give a built scenario with portfolio learning cases the variant in which the CfD units belong to their
    companies' portfolios (the author, 7 Oct 2026: a portfolio that holds its CfD farms may trade differently),
    beside the cases in which they stay out and bid their contracts.

    The merged CfD units are split back into their contracts (``split_cfd_units``), since a merged unit can hold
    several companies' contracts (Walney Extension and Burbo Bank Extension, Orsted's, with Dudgeon, Equinor's).
    ``owners`` (unit name -> agent, ``owners.portfolio_agents`` with ``ownership.PORTFOLIO_KINDS_CFD``) then gives
    the portfolios, with the folder's technologies, bins and threshold (``min_portfolio_mw`` where it records none).
    Written: ``powerplant_units_learning_cfd.csv``, ``unit_operators_learning_cfd.csv``,
    ``availability_df_cfd.csv.gz`` (the folder's availability with the contracts' own), the cases ``learning_cfd``,
    ``learning_cfd_week`` and ``learning_cfd_year`` in ``config.yaml`` (copies of the learning cases on these files;
    the year case loads the policies of ``learning_cfd``) and ``learning.cfd_variant`` in ``scenario_meta.json``.
    The units' offers are unchanged, so without learning the prices are those of the other cases; nothing else in
    the folder changes. ``stack`` must be the stack the folder was built from. Returns the learning units and
    agents of the variant."""
    out_dir = Path(out_dir)
    meta_path = out_dir / "scenario_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    learning = meta.get("learning") or {}
    if learning.get("pick") != "portfolio":
        raise ValueError(
            f"{out_dir.name} has no portfolio learning cases (build it with --learning-pick portfolio)"
        )
    built_from, given = (meta.get("source") or {}).get("git"), stack.meta.get("git")
    if built_from and given and built_from != given:
        raise ValueError(
            f"{out_dir.name} was built from the stack export of commit {built_from}, this stack is of {given}"
        )
    threshold = (
        min_portfolio_mw
        if learning.get("min_portfolio_mw") is None
        else learning["min_portfolio_mw"]
    )
    table = pd.read_csv(out_dir / "powerplant_units.csv", float_precision="round_trip")
    columns = pd.read_csv(out_dir / "unit_columns.csv", index_col=0)
    table, shares, split = split_cfd_units(table, columns, stack)
    config_path = out_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    written = list(learning_files(CFD_VARIANT))
    with tempfile.TemporaryDirectory(
        dir=out_dir
    ) as tmp:  # the folder keeps its files unless every one is written
        units, operators = write_learning(
            Path(tmp), table, tuple(learning["technologies"]), learning["count"], "portfolio", None, owners,
            learning["portfolio"]["nbins"], threshold, variant=CFD_VARIANT,
        )  # fmt: skip
        if not operators:
            raise ValueError(
                f"{out_dir.name}: no agent has a portfolio at {agent_level} level with the CfD units"
            )
        if shares:
            availability = pd.read_csv(
                out_dir / "availability_df.csv.gz",
                index_col=0,
                float_precision="round_trip",
            )
            lead = (
                len(availability) - len(stack.index) - 1
            )  # _pad: the lead-in day in front, a closing step behind
            contracts = pd.DataFrame(
                {name: _pad(share, lead) for name, share in shares.items()},
                index=availability.index,
            )
            availability = pd.concat(
                [
                    availability.drop(
                        columns=[c for c in split if c in availability.columns]
                    ),
                    contracts.clip(0.0, 1.0),
                ],
                axis=1,
            )
            availability.to_csv(Path(tmp) / AVAILABILITY_CFD_FILE, float_format="%.10g")
            written.append(AVAILABILITY_CFD_FILE)
        for name in written:
            os.replace(Path(tmp) / name, out_dir / name)
    cases = cfd_cases(config, out_dir.name, AVAILABILITY_CFD_FILE if shares else None)
    config.update(cases)
    config_path.write_text(
        SPDX_HEADER + yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    learning["cfd_variant"] = {
        "units": units, "operators": operators, "agent_level": agent_level, "portfolio_kinds": ["plant", "farm", "cfd"],
        "min_portfolio_mw": float(threshold), "split": split, "cases": list(cases), "files": written,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }  # fmt: skip
    meta["learning"] = learning
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return units, operators


def write_storage(stack: OfferStack, out_dir: Path) -> dict | None:
    """The merit-order model's battery fleet as one storage unit.

    The model dispatches the fleet by an optimisation inside its clearing, knowing the day's whole
    stack. The framework's storage strategies bid on a price forecast instead, the one in
    ``forecasts_df.csv``: the merit order of the scenario's own offers without storage, the same
    information. The fleet's round-trip efficiency is booked on discharge, as the model does.
    Returns the fleet, or None if the stack has none."""
    fleet = stack.meta["constants"].get("battery_fleet")
    if not fleet or not fleet.get("power_mw"):
        return None
    unit = {
        "name": STORAGE_UNIT,
        "technology": "battery",
        f"bidding_{MARKET_ID}": STORAGE_STRATEGY,
        "max_power_charge": fleet["power_mw"],
        "max_power_discharge": fleet["power_mw"],
        "capacity": fleet["energy_mwh"],
        "efficiency_charge": 1.0,
        "efficiency_discharge": fleet["round_trip_efficiency"],
        "initial_soc": 0.0,
        "min_soc": 0.0,
        "max_soc": 1.0,
        # the model's cycling hurdle per MWh drawn; the framework books it as a cost, not in the bid
        "additional_cost_charge": float(
            stack.meta["constants"].get("storage_degradation_gbp_mwh", 0.0)
        ),
        "additional_cost_discharge": 0.0,
        "unit_operator": STORAGE_UNIT,
    }
    pd.DataFrame([unit]).set_index("name").to_csv(out_dir / "storage_units.csv")
    return dict(fleet)


def write_storage_traders(out_dir: Path, shares: pd.Series) -> bool:
    """The battery fleet of ``storage_units.csv`` as one unit per trading company: each unit is the fleet scaled
    by the company's share (``shares``: company -> share, summing to one), named ``battery_<company>``, with the
    company as its ``unit_operator``. Returns whether the file was written."""
    shares = shares[shares > 0].astype(float)
    if shares.empty:
        return False
    shares = shares / shares.sum()
    fleet = pd.read_csv(out_dir / "storage_units.csv").iloc[0]
    rows = []
    for company, share in shares.items():
        unit = fleet.copy()
        unit["name"] = f"{STORAGE_UNIT}_" + re.sub(
            r"[^A-Za-z0-9]+", "_", str(company)
        ).strip("_")
        for column in ("max_power_charge", "max_power_discharge", "capacity"):
            unit[column] = float(fleet[column]) * share
        unit["unit_operator"] = str(company)
        rows.append(unit)
    pd.DataFrame(rows).set_index("name").to_csv(out_dir / STORAGE_TRADERS_FILE)
    return True
