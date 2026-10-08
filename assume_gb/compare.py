# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run a GB scenario folder with the framework as it ships and compare its prices with the
reference prices in the folder.

Nothing here is model logic: the run is ``World`` + ``load_scenario_folder`` + ``run``, and the
comparison is arithmetic on price series. Three kinds of reference are in a scenario folder
(``reference_prices.csv``, written by ``convert.write_reference``):

    scenario_merit_order   the merit order of the scenario's own offers. The framework must
                           reproduce it to rounding: this checks the engine.
    model_no_storage       the merit-order model's price for the same stack without its battery.
    model_with_battery     that model's price with its battery fleet.

and, if present, the observed prices (``observed_prices.csv``, licensed, local only): the hourly
day-ahead auctions of both exchanges and their blend, compared on hourly means, and the
half-hourly auctions and indices, compared period by period.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from assume import World
from assume.markets.base_market import MarketRole
from assume.scenario.loader_csv import load_scenario_folder, run_learning
from assume_gb import convert, paths


def run_scenario(
    name: str,
    case: str = convert.CASE,
    inputs_path: Path | None = None,
    csv_path: str = "",
    database_uri: str = "",
) -> tuple[World, float]:
    """Simulate study case ``case`` of scenario folder ``name``; the world after the run and the
    seconds the simulation took. Nothing is written unless ``csv_path`` or ``database_uri`` is given.
    A learning case trains its agents first (``run_learning``: the training episodes, then the run
    with the trained policies, which is what the world holds afterwards) and needs a database, as
    the framework reads the validation rewards back from it."""
    world = World(
        database_uri=database_uri, export_csv_path=csv_path, log_level="WARNING"
    )
    inputs_path = (
        Path(inputs_path) if inputs_path is not None else paths.scenarios_dir()
    )
    load_scenario_folder(
        world, inputs_path=str(inputs_path), scenario=name, study_case=case
    )
    start = time.perf_counter()
    if world.learning_mode:
        if not database_uri:
            raise ValueError(
                f"{name}/{case} trains learning agents and needs a database (database_uri)"
            )
        run_learning(world)
    world.run()
    return world, time.perf_counter() - start


def clearing_prices(world: World) -> pd.DataFrame:
    """The clearing price of every product the markets of ``world`` have cleared, at full
    precision: one column per market, indexed by the start of the product."""
    results = pd.DataFrame(
        [
            r
            for op in world.market_operators.values()
            for role in op.roles
            if isinstance(role, MarketRole)
            for r in role.results
        ]
    )
    results["product_start"] = pd.to_datetime(results["product_start"])
    return results.pivot(
        index="product_start", columns="market_id", values="price"
    ).sort_index()


def hourly(price: pd.Series, admissible: pd.Series) -> pd.Series:
    """Hourly means of a price. An hour with a period that is not admissible is left out."""
    hour = price.index.floor("h")
    keep = (
        admissible.reindex(price.index, fill_value=False).groupby(hour).transform("all")
    )
    return price[keep].groupby(hour[keep]).mean()


def difference(price: pd.Series, reference: pd.Series) -> dict:
    """How far a price is from a reference price on the periods both have. ``r2`` is
    1 - SSE / SST, the statistic of the merit-order model's own fit table."""
    both = pd.concat({"price": price, "reference": reference}, axis=1).dropna()
    error = both["price"] - both["reference"]
    sst = ((both["reference"] - both["reference"].mean()) ** 2).sum()
    return {
        "periods": len(both),
        "bias": error.mean(),
        "mae": error.abs().mean(),
        "rmse": np.sqrt((error**2).mean()),
        "max_abs": error.abs().max(),
        "r2": 1 - (error**2).sum() / sst if sst > 0 else np.nan,
    }


# the observed prices each market is compared with. The simulated day-ahead market is one market
# for all demand: it is compared with both hourly day-ahead auctions and their volume-weighted
# blend on hourly means, and with the half-hourly auction period by period, as the intraday
# market is with its targets.
HOURLY_TARGETS = ["n2ex_day_ahead", "epex_day_ahead", "blend_day_ahead"]
TARGETS = {
    convert.MARKET_ID: ["epex_hh_day_ahead"],
    convert.INTRADAY_MARKET_ID: ["epex_ida1", "epex_ida2", "apx_mid", "system_price"],
}


def compare(prices: pd.DataFrame | pd.Series, folder: Path) -> pd.DataFrame:
    """One row per comparison of the simulated prices (one column per market) with the references
    in ``folder``.

    The day-ahead price is compared with the merit order of the scenario's own offers on every
    period, with the merit-order model's prices on the periods that model itself admits (it flags
    scarcity and corrupt demand input), and with the observed auction price on the hours made up
    of such periods. An intraday price is compared with the observed intraday and balancing prices
    on the admissible periods."""
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(convert.MARKET_ID)
    reference = pd.read_csv(
        folder / "reference_prices.csv", index_col=0, parse_dates=True
    )
    admissible = reference.pop("admissible").astype(bool)
    admitted = admissible[admissible].index
    observed = pd.DataFrame(index=reference.index)
    if (folder / "observed_prices.csv").exists():
        observed = pd.read_csv(
            folder / "observed_prices.csv", index_col=0, parse_dates=True
        )

    rows = {}
    if convert.MARKET_ID in prices:
        price = prices[convert.MARKET_ID]
        for name, series in reference.items():
            keep = slice(None) if name == "scenario_merit_order" else admitted
            rows[f"simulated vs {name}"] = difference(price, series.loc[keep])
        models = {
            "simulated": price,
            **{k: v for k, v in reference.items() if k.startswith("model")},
        }
        for target in HOURLY_TARGETS:
            if target in observed:
                for name, series in models.items():
                    rows[f"{name} vs {target}, hourly"] = difference(
                        hourly(series, admissible), hourly(observed[target], admissible)
                    )
    for market_id in prices.columns:
        for target in TARGETS.get(market_id, []):
            if target in observed:
                rows[f"simulated {market_id} vs {target}"] = difference(
                    prices[market_id], observed.loc[admitted, target]
                )
        if convert.MARKET_ID in prices and market_id != convert.MARKET_ID:
            rows[f"simulated {market_id} vs simulated {convert.MARKET_ID}"] = (
                difference(prices[market_id], prices[convert.MARKET_ID])
            )
    return pd.DataFrame(rows).T.astype({"periods": int})
