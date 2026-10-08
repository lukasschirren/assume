# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""The local GB data tool on a small synthetic offer stack (three days of half-hours, one column
for every kind of offer, invented numbers): the scenario it writes must make every unit offer
what the stack offers, and must clear, in the framework, at the merit order of that stack.

    python -m pytest assume_gb
"""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from assume import World
from assume.markets.base_market import MarketRole
from assume.scenario.loader_csv import load_scenario_folder
from assume_gb import convert
from assume_gb.stack import OfferStack, merit_order_price

META = {
    "year": 2023, "design": "status_quo", "version": "test", "tag": "test", "git": "none", "venue": "n2ex",
    "floor": -500.0, "cap": 3000.0, "roc_value_gbp_per_roc": 60.0, "clear_kwargs": {"vre_discount": 0.0},
    "constants": {
        "avoidable_gbp_mwh": {"Wind": 3.0, "Solar": 1.0, "Biomass": 68.0, "Solid Biomass": 68.0, "Waste": 5.0},
        "fit_tariff_gbp_mwh": {"Solar": 63.8, "Wind": 42.5}, "fit_tariff_default_gbp_mwh": 60.0,
        "nuclear_bid": -200.0, "ccgt_msg_bid": -15.0, "biomass_msg_bid": -10.0, "embedded_mustrun_bid": -5.0,
        "merchant_vre_bid": 0.0, "cfd_bid_basis": "minus_strike", "ar_split_enabled": True,
        "baseload_cfd_enabled": True, "efw_cfd_mustrun": True,
        "thermal_techs": ["CCGT", "OCGT", "Hard Coal", "Oil"], "vre_techs": ["Wind", "Solar"],
        "ro_mustrun_techs": ["BIOMASS_MSG"], "efficiency_fallback": 0.35, "oil_gbp_mwh_th": 80.0,
        "battery_fleet": {"power_mw": 300.0, "energy_mwh": 600.0, "round_trip_efficiency": 0.85},
        "storage_degradation_gbp_mwh": 10.0,
    },
}  # fmt: skip

# unit_id, tech, regime, cfd class, neg rule, band, strike, efficiency, emissions, premium, MW, kind
COLUMNS = [
    ("ro_wind", "Wind", "RO", "", "", 1.0, np.nan, np.nan, 0.0, np.nan, 5000.0, "ro"),
    ("ro_wind_other_zone", "Wind", "RO", "", "", 1.0, np.nan, np.nan, 0.0, np.nan, 1000.0, "ro"),
    ("ro_wind_offshore", "Wind", "RO", "", "", 2.0, np.nan, np.nan, 0.0, np.nan, 3000.0, "ro"),
    ("ro_biomass", "Solid Biomass", "RO", "", "", 1.5, np.nan, 0.38, 0.0, np.nan, 800.0, "ro"),
    ("cfd_wind_six", "Wind", "CfD", "int", "six_hour", np.nan, 100.0, np.nan, 0.0, np.nan, 2500.0, "cfd_intermittent"),
    ("cfd_wind_any", "Wind", "CfD", "int", "any_hour", np.nan, 80.0, np.nan, 0.0, np.nan, 900.0, "cfd_intermittent"),
    ("cfd_biomass", "Biomass", "CfD", "base", "six_hour", np.nan, 130.0, 0.38, 0.0, np.nan, 600.0, "cfd_baseload"),
    ("cfd_waste", "Waste", "CfD", "base", "six_hour", np.nan, 105.0, np.nan, 0.0, np.nan, 40.0, "cfd_baseload"),
    ("fit_solar", "Solar", "FiT", "", "", np.nan, np.nan, np.nan, 0.0, np.nan, 1500.0, "fit"),
    ("merchant_solar", "Solar", "merchant_vre", "", "", np.nan, np.nan, np.nan, 0.0, np.nan, 2000.0, "merchant_vre"),
    ("merchant_wind", "Wind", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, np.nan, 700.0, "merchant_vre"),
    ("nuclear", "Nuclear", "nuclear", "", "", np.nan, np.nan, np.nan, 0.0, np.nan, 4000.0, "must_run"),
    ("ccgt_msg", "CCGT_MSG", "merchant_thermal", "", "", np.nan, np.nan, np.nan, 0.0, np.nan, 2000.0, "must_run"),
    ("biomass_msg", "BIOMASS_MSG", "RO", "", "", np.nan, np.nan, np.nan, 0.0, np.nan, 150.0, "must_run"),
    ("embedded", "EMB_MUSTRUN", "embedded", "", "", np.nan, np.nan, np.nan, 0.0, np.nan, 900.0, "must_run"),
    ("ccgt_new", "CCGT", "merchant_thermal", "", "", np.nan, np.nan, 0.55, 0.184, 4.0, 7000.0, "thermal"),
    ("ccgt_old", "CCGT", "merchant_thermal", "", "", np.nan, np.nan, 0.47, 0.184, 9.0, 6000.0, "thermal"),
    ("ocgt", "OCGT", "merchant_thermal", "", "", np.nan, np.nan, np.nan, 0.184, 38.0, 1500.0, "thermal"),
    ("coal", "Hard Coal", "merchant_thermal", "", "", np.nan, np.nan, 0.36, 0.34, 12.0, 1800.0, "thermal"),
    ("oil", "Oil", "merchant_thermal", "", "", np.nan, np.nan, 0.35, 0.28, 20.0, 500.0, "thermal"),
    ("waste", "Waste", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, np.nan, 300.0, "avoidable"),
    ("IC_FR_IMPORT", "IC_IMPORT", "interconnector", "", "", np.nan, np.nan, np.nan, 0.0, np.nan, 3000.0, "interconnector"),
    ("IC_FR_EXPORT", "IC_EXPORT", "interconnector", "", "", np.nan, np.nan, np.nan, 0.0, np.nan, 3000.0, "interconnector"),
]  # fmt: skip


def make_stack(days: int = 3, seed: int = 7) -> OfferStack:
    """A consistent synthetic stack: its bids are written from the offer rules by hand here, not
    by the tool, so the tool's conversion is tested against an independent statement of them."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=48 * days, freq="30min")
    n = len(index)
    hour = (index.hour + index.minute / 60).to_numpy()
    names = [c[0] for c in COLUMNS]
    keys = ["unit_id", "tech", "support_regime", "cfd_class", "neg_price_rule", "band", "support_value_gbp_mwh",
            "efficiency", "emissions_factor", "startup_premium_gbp_mwh"]  # fmt: skip
    agents = pd.DataFrame({k: [c[i] for c in COLUMNS] for i, k in enumerate(keys)})
    agents["caprec_mu_gbp_mwh"] = 0.0
    capacity = np.array([c[10] for c in COLUMNS])
    mw = np.repeat(capacity[None, :], n, axis=0)
    wind = np.clip(
        0.5 + 0.45 * np.sin(np.arange(n) / 23.0) + rng.normal(0, 0.03, n), 0.0, 1.0
    )
    solar = np.clip(np.sin((hour - 6) / 12 * np.pi), 0.0, None)
    for j, tech in enumerate(agents["tech"]):
        if tech == "Wind":
            mw[:, j] = capacity[j] * wind
        elif tech == "Solar":
            mw[:, j] = capacity[j] * solar
    mw[:, names.index("IC_FR_EXPORT")] = np.where(hour < 12, 3000.0, 2000.0)

    gas = 35.0 + 5.0 * np.sin(np.arange(n) / 60.0)
    carbon, coal, oil = 70.0, 22.0, 80.0
    bmrp = np.where(np.arange(n) < n // 2, 110.0, 150.0)
    neighbour = 60.0 + 50.0 * np.sin(np.arange(n) / 17.0)
    one = np.ones(n)
    by_hand = {
        "ro_wind": -(1.0 * 60.0 - 3.0) * one,
        "ro_wind_other_zone": -(1.0 * 60.0 - 3.0) * one,
        "ro_wind_offshore": -(2.0 * 60.0 - 3.0) * one,
        "ro_biomass": -(1.5 * 60.0 - 68.0) * one,
        "cfd_wind_six": -(100.0 - 3.0) * one,
        "cfd_wind_any": 3.0 * one,
        "cfd_biomass": 68.0 - (130.0 - bmrp),
        "cfd_waste": -500.0 * one,
        "fit_solar": (1.0 - 63.8) * one,
        "merchant_solar": 0.0 * one,
        "merchant_wind": 0.0 * one,
        "nuclear": -200.0 * one,
        "ccgt_msg": -15.0 * one,
        "biomass_msg": -10.0 * one,
        "embedded": -5.0 * one,
        "ccgt_new": (gas + carbon * 0.184) / 0.55 + 4.0,
        "ccgt_old": (gas + carbon * 0.184) / 0.47 + 9.0,
        "ocgt": (gas + carbon * 0.184) / 0.35 + 38.0,
        "coal": ((coal + carbon * 0.34) / 0.36 + 12.0) * one,
        "oil": ((oil + carbon * 0.28) / 0.35 + 20.0) * one,
        "waste": 5.0 * one,
        "IC_FR_IMPORT": neighbour / 0.98,
        "IC_FR_EXPORT": neighbour * 0.98,
    }
    bids = np.column_stack([by_hand[name] for name in names])
    demand = (
        20000.0 + 13000.0 * np.sin((hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 300, n)
    )
    demand = demand + mw[:, names.index("IC_FR_EXPORT")]
    series = {
        "gas_gbp_mwh_th": gas, "coal_gbp_mwh_th": np.full(n, coal), "carbon_gbp_t": np.full(n, carbon),
        "bmrp": bmrp, "demand": demand,
        "price_no_storage": merit_order_price(bids, mw, demand, -500.0),
        "da_price": np.full(n, 50.0),
    }  # fmt: skip
    return OfferStack(
        index=index, agents=agents, bids=bids, mw=mw, series=series, meta=META
    )


def simulate(inputs_path, name: str) -> tuple[pd.Series, pd.DataFrame]:
    """Clearing price and dispatch of a scenario folder, run with the framework as it ships."""
    world = World(log_level="WARNING")
    load_scenario_folder(
        world, inputs_path=str(inputs_path), scenario=name, study_case=convert.CASE
    )
    world.run()
    results = pd.DataFrame(
        [
            r
            for op in world.market_operators.values()
            for role in op.roles
            if isinstance(role, MarketRole)
            for r in role.results
        ]
    )
    price = results.set_index(pd.to_datetime(results["product_start"]))[
        "price"
    ].sort_index()
    index = next(iter(world.units.values())).index.as_datetimeindex()
    dispatch = pd.DataFrame(
        {u.id: u.outputs["energy"].data.copy() for u in world.units.values()},
        index=index,
    )
    return price, dispatch.loc[price.index[0] : price.index[-1]]


@pytest.fixture(scope="module")
def stack():
    return make_stack()


@pytest.fixture(scope="module")
def run(stack, tmp_path_factory):
    inputs = tmp_path_factory.mktemp("gb_inputs")
    convert.build_scenario(stack, inputs / "gb_test", merge=False)
    return simulate(inputs, "gb_test")


def test_every_kind_of_offer_is_classified_as_written(stack):
    units = convert.unit_table(stack)
    assert units["kind"].tolist() == [c[11] for c in COLUMNS]


ANY_HOUR = [c[0] for c in COLUMNS].index("cfd_wind_any")


def expected_offers(stack) -> np.ndarray:
    """The stack's offers as the scenario makes them: the model's, except that the unit under the
    any-hour rule offers zero, where its contract starts to pay, and not its avoidable cost."""
    offers = stack.bids.copy()
    offers[:, ANY_HOUR] = 0.0
    return offers


def test_scenario_offers_are_the_models_offers_but_for_the_any_hour_rule(stack):
    assert (stack.bids[:, ANY_HOUR] == 3.0).all()
    np.testing.assert_allclose(
        convert.scenario_offers(stack), expected_offers(stack), rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(
        convert.model_offers(stack), expected_offers(stack), rtol=0, atol=1e-9
    )
    table = convert.check_offers(stack)
    assert table.loc["all", "max_abs_dev"] < 1e-9
    assert table.loc["any_hour_rule_vs_model", ["columns", "max_abs_dev"]].tolist() == [
        1,
        3.0,
    ]


def test_simulated_prices_are_the_merit_order_of_the_stack(stack, run, tmp_path):
    price, _ = run
    expected = merit_order_price(expected_offers(stack), stack.mw, stack.demand, -500.0)
    assert price.index.equals(stack.index)
    np.testing.assert_allclose(price.to_numpy(), expected, rtol=0, atol=1e-9)
    # the test is only worth something if the price moves across the kinds of offer
    assert (
        expected.min() < -50
        and expected.max() > 100
        and len(np.unique(expected.round(6))) > 20
    )
    # and the reference file says the same, beside the model's own price, which the any-hour unit moves
    folder = convert.build_scenario(stack, tmp_path / "gb_reference")
    reference = pd.read_csv(
        folder / "reference_prices.csv", index_col=0, parse_dates=True
    )
    np.testing.assert_allclose(
        reference["scenario_merit_order"].to_numpy(), expected, rtol=0, atol=1e-9
    )
    moved = reference["scenario_merit_order"] != reference["model_no_storage"]
    assert moved.any() and set(reference.loc[moved, "model_no_storage"]) == {3.0}


def test_supply_meets_demand_and_the_rest_is_unserved_at_the_cap(stack, run):
    price, dispatch = run
    supply = dispatch.drop(columns=convert.DEMAND_UNIT).sum(axis=1)
    np.testing.assert_allclose(supply.to_numpy(), stack.demand, rtol=0, atol=1e-6)
    shortfall = np.clip(stack.demand - stack.mw.sum(axis=1), 0.0, None)
    assert (shortfall > 0).any() and (shortfall == 0).any()
    np.testing.assert_allclose(
        dispatch[convert.UNSERVED_UNIT].to_numpy(), shortfall, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        price.to_numpy()[shortfall > 0], 3000.0, rtol=0, atol=1e-9
    )


def test_merged_scenario_gives_the_same_prices_with_fewer_units(stack, run, tmp_path):
    convert.build_scenario(stack, tmp_path / "gb_merged", merge=True)
    price, dispatch = simulate(tmp_path, "gb_merged")
    np.testing.assert_allclose(price.to_numpy(), run[0].to_numpy(), rtol=0, atol=1e-9)
    # the two RO wind columns with the same band are one unit now, carrying both outputs
    assert len(dispatch.columns) == len(run[1].columns) - 1
    both = run[1][["ro_wind", "ro_wind_other_zone"]].sum(axis=1)
    np.testing.assert_allclose(
        dispatch["ro_wind_001"].to_numpy(), both.to_numpy(), rtol=0, atol=1e-6
    )


def test_battery_cycles_on_the_price_forecast_and_moves_prices(stack, run, tmp_path):
    convert.build_scenario(stack, tmp_path / "gb_storage")
    world = World(log_level="WARNING")
    load_scenario_folder(
        world,
        inputs_path=str(tmp_path),
        scenario="gb_storage",
        study_case=convert.STORAGE_CASE,
    )
    world.run()
    results = pd.DataFrame(
        [
            r
            for op in world.market_operators.values()
            for role in op.roles
            if isinstance(role, MarketRole)
            for r in role.results
        ]
    )
    price = results.set_index(pd.to_datetime(results["product_start"]))[
        "price"
    ].sort_index()
    battery = world.units[convert.STORAGE_UNIT]
    energy = pd.Series(
        battery.outputs["energy"].data, index=battery.index.as_datetimeindex()
    ).loc[price.index]
    fleet = META["constants"]["battery_fleet"]
    # what is sold is what was bought less the round-trip loss, within the fleet's power
    assert (
        energy.max() <= fleet["power_mw"] + 1e-6
        and energy.min() >= -fleet["power_mw"] - 1e-6
    )
    assert energy[energy > 0].sum() == pytest.approx(
        -energy[energy < 0].sum() * fleet["round_trip_efficiency"], rel=1e-6
    )
    assert energy[energy > 0].sum() > 0
    # the battery lifts cheap periods and lowers dear ones relative to the run without it
    # (this small stack has a wide thermal block at the top: discharging there moves no price)
    moved = price - run[0]
    up, down = moved[moved > 1e-6], moved[moved < -1e-6]
    assert len(up) > 0
    assert up.index.isin(energy[energy < 0].index).all()
    assert down.index.isin(energy[energy > 0].index).all()


def test_scenario_folder_uses_only_what_the_framework_ships(stack, tmp_path):
    folder = convert.build_scenario(stack, tmp_path / "gb_files")
    files = {p.name for p in folder.iterdir()}
    assert {"config.yaml", "powerplant_units.csv", "demand_units.csv", "demand_df.csv", "fuel_prices_df.csv",
            "availability_df.csv.gz", "reference_prices.csv", "observed_prices.csv", ".gitignore"} <= files  # fmt: skip
    assert "observed_prices.csv" in (folder / ".gitignore").read_text()
    cases = yaml.safe_load((folder / "config.yaml").read_text())
    # the short case is the same market on the same files, ending earlier (here the stack is shorter
    # than a week, so it ends where the full case does); the storage cases add the battery fleet
    assert list(cases) == [
        convert.CASE, convert.WEEK_CASE, convert.STORAGE_CASE, convert.STORAGE_WEEK_CASE,
        convert.LEARNING_CASE, convert.LEARNING_WEEK_CASE, convert.LEARNING_YEAR_CASE,
    ]  # fmt: skip
    short = dict(cases[convert.WEEK_CASE], end_date=cases[convert.CASE]["end_date"])
    assert short == cases[convert.CASE]
    assert (
        cases[convert.CASE]["storage_units"] is None
        and "storage_units" not in cases[convert.STORAGE_CASE]
    )
    battery = pd.read_csv(folder / "storage_units.csv", index_col=0).loc[
        convert.STORAGE_UNIT
    ]
    assert (
        battery["max_power_discharge"] == META["constants"]["battery_fleet"]["power_mw"]
    )
    assert (
        battery["efficiency_discharge"]
        == META["constants"]["battery_fleet"]["round_trip_efficiency"]
    )
    forecast = pd.read_csv(folder / "forecasts_df.csv", index_col=0, parse_dates=True)
    reference = pd.read_csv(
        folder / "reference_prices.csv", index_col=0, parse_dates=True
    )
    np.testing.assert_allclose(
        forecast.loc[reference.index, f"price_{convert.MARKET_ID}"],
        reference["scenario_merit_order"],
    )
    assert forecast.notna().all().all()
    config = cases[convert.CASE]
    market = config["markets_config"][convert.MARKET_ID]
    # one auction a day for the next day's 48 half-hours, opened the day before the first delivery
    assert config["time_step"] == "30min" and market["products"][0]["count"] == 48
    assert pd.Timestamp(config["start_date"]) == stack.index[0] - pd.Timedelta("1D")
    assert "bidding_strategy_params" not in config
    units = pd.read_csv(folder / "powerplant_units.csv", index_col=0)
    assert set(units[f"bidding_{convert.MARKET_ID}"]) == {
        convert.NAIVE,
        convert.SUPPORT,
    }
    assert set(units["support_scheme"].dropna()) == {"premium", "cfd"}


def synthetic_errors(stack) -> pd.DataFrame:
    """Forecast errors by hand: wind over-forecast by a fifth on the first day, under-forecast by a
    tenth on the second, exact on the third; solar exact; demand under-forecast by 800 MW on the
    second day."""
    day = (stack.index - stack.index[0]).days.to_numpy()
    wind = np.select([day == 0, day == 1], [1.2, 0.9], 1.0)
    demand = np.where(day == 1, -800.0, 0.0)
    return pd.DataFrame(
        {"wind_ratio": wind, "solar_ratio": 1.0, "demand_error_mw": demand},
        index=stack.index,
    )


def test_intraday_case_bids_the_forecast_day_ahead_and_trades_the_deviation(
    stack, run, tmp_path
):
    errors = synthetic_errors(stack)
    folder = convert.build_scenario(
        stack, tmp_path / "gb_intraday", forecast_errors=errors
    )
    cases = yaml.safe_load((folder / "config.yaml").read_text())
    assert convert.INTRADAY_CASE in cases and convert.INTRADAY_WEEK_CASE in cases
    units = pd.read_csv(folder / "powerplant_units_intraday.csv", index_col=0)
    assert set(units[f"bidding_{convert.MARKET_ID}"]) == {convert.FORECAST}
    assert set(units[f"bidding_{convert.INTRADAY_MARKET_ID}"]) == {convert.REBALANCE}
    forecast = pd.read_csv(
        folder / "availability_forecast_df.csv.gz", index_col=0, parse_dates=True
    )
    # only the wind units differ from the outturn, by the ratio
    assert all(units.loc[c, "technology"] == "Wind" for c in forecast.columns)
    availability = pd.read_csv(
        folder / "availability_df.csv.gz", index_col=0, parse_dates=True
    )
    for column in forecast.columns:
        expected = np.minimum(
            availability.loc[stack.index, column] * errors["wind_ratio"], 1.0
        )
        np.testing.assert_allclose(
            forecast.loc[stack.index, column], expected, rtol=1e-6
        )

    world = World(log_level="WARNING", export_csv_path=str(tmp_path / "out"))
    load_scenario_folder(
        world,
        inputs_path=str(tmp_path),
        scenario="gb_intraday",
        study_case=convert.INTRADAY_CASE,
    )
    world.run()
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
    prices = results.pivot(
        index="product_start", columns="market_id", values="price"
    ).sort_index()
    assert prices.index.equals(stack.index)

    # the day-ahead price is the merit order of the forecast offers, the intraday market clears
    # every period, and after it every unit's position is what it can deliver: supply meets demand
    forecast_mw, forecast_demand = convert.forecasts.apply(stack, errors)
    forecast_mw = np.minimum(
        forecast_mw, stack.mw.max(axis=0)
    )  # a forecast above the unit's maximum power is capped
    expected_da = merit_order_price(
        expected_offers(stack), forecast_mw, forecast_demand, -500.0
    )
    np.testing.assert_allclose(
        prices[convert.MARKET_ID].to_numpy(), expected_da, rtol=0, atol=1e-6
    )
    assert prices[convert.INTRADAY_MARKET_ID].notna().all()
    index = next(iter(world.units.values())).index.as_datetimeindex()
    dispatch = pd.DataFrame(
        {u.id: u.outputs["energy"].data.copy() for u in world.units.values()},
        index=index,
    ).loc[stack.index]
    supply = dispatch.drop(columns=convert.DEMAND_UNIT).sum(axis=1)
    np.testing.assert_allclose(supply.to_numpy(), stack.demand, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        -dispatch[convert.DEMAND_UNIT].to_numpy(), stack.demand, rtol=0, atol=1e-6
    )
    # what is traded intraday: the net change of position per offer price. A unit at the margin sells
    # and buys back at one price, and so do units that offer at the same price among themselves; the
    # clearing may accept both sides, a swap that moves no price and no cost. The CSV output carries
    # five significant digits, so a net change below 1 MW is none.
    orders = pd.read_csv(
        tmp_path / "out" / f"gb_intraday_{convert.INTRADAY_CASE}" / "market_orders.csv",
        parse_dates=["start_time"],
    )
    orders = orders[orders["market_id"] == convert.INTRADAY_MARKET_ID]
    net = (
        orders.groupby(["start_time", "price"])["accepted_volume"]
        .sum()
        .abs()
        .groupby("start_time")
        .sum()
        / 2
    )
    # the third day has no forecast error: the two markets clear at one price and no position changes
    exact = errors.index[
        (errors["wind_ratio"] == 1.0) & (errors["demand_error_mw"] == 0.0)
    ]
    np.testing.assert_allclose(
        prices.loc[exact, convert.INTRADAY_MARKET_ID],
        prices.loc[exact, convert.MARKET_ID],
        rtol=0,
        atol=1e-6,
    )
    assert (net.reindex(exact).fillna(0.0) < 1.0).all()
    # the first day over-forecast wind: the system is short intraday, positions change wherever the
    # forecast differs from the outturn (not at the wind peak, where the cap removes the error), and
    # the price there is not below the day-ahead price
    short = errors.index[errors["wind_ratio"] > 1.0]
    deviation = pd.Series(
        np.abs(forecast_mw - stack.mw).sum(axis=1)
        + np.abs(forecast_demand - stack.demand),
        index=stack.index,
    )
    assert (deviation.loc[short] > 1.0).sum() > 40
    assert (net.reindex(short[deviation.loc[short] > 1.0]) > 1.0).all()
    assert (
        prices.loc[short, convert.INTRADAY_MARKET_ID]
        >= prices.loc[short, convert.MARKET_ID] - 1e-6
    ).all()
    assert (
        prices.loc[short, convert.INTRADAY_MARKET_ID]
        > prices.loc[short, convert.MARKET_ID] + 1e-6
    ).any()


def test_learning_case_holds_the_same_market_one_period_at_a_time(stack, run, tmp_path):
    folder = convert.build_scenario(
        stack, tmp_path / "gb_learning", learning_techs=("CCGT",), learning_weeks=1
    )
    # the largest so many units of the technologies can be picked
    only_largest = convert.build_scenario(
        stack,
        tmp_path / "gb_largest",
        learning_techs=("CCGT",),
        learning_count=1,
        learning_weeks=1,
    )
    largest = pd.read_csv(only_largest / "powerplant_units_learning.csv", index_col=0)
    assert largest.index[
        largest[f"bidding_{convert.MARKET_ID}"] == convert.LEARNING
    ].tolist() == ["ccgt_new"]
    cheapest = convert.build_scenario(
        stack,
        tmp_path / "gb_cheapest",
        learning_techs=("CCGT", "OCGT"),
        learning_count=2,
        learning_pick="cheapest",
        learning_weeks=1,
    )
    picked = pd.read_csv(cheapest / "powerplant_units_learning.csv", index_col=0)
    assert set(
        picked.index[picked[f"bidding_{convert.MARKET_ID}"] == convert.LEARNING]
    ) == {"ccgt_new", "ccgt_old"}
    # portfolio agents: an owner with enough plants bids them as one agent, the plants keep their strategies
    owners = pd.Series({"ccgt_new": "A", "ccgt_old": "A", "ocgt": "B", "coal": "A"})
    portfolio = convert.build_scenario(
        stack,
        tmp_path / "gb_portfolio",
        learning_techs=("CCGT", "OCGT"),
        learning_pick="portfolio",
        learning_owners=owners,
        nbins=2,
        learning_weeks=1,
    )
    plants = pd.read_csv(portfolio / "powerplant_units_learning.csv", index_col=0)
    operators = pd.read_csv(portfolio / "unit_operators_learning.csv", index_col=0)
    assert (
        operators.index.tolist() == ["A"]
        and operators.loc["A", f"bidding_{convert.MARKET_ID}"]
        == convert.PORTFOLIO_LEARNING
    )
    assert plants.loc[["ccgt_new", "ccgt_old"], "unit_operator"].tolist() == ["A", "A"]
    assert (
        plants.loc["ocgt", "unit_operator"] != "B"
        and plants.loc["coal", "unit_operator"] != "A"
    )  # too few plants; not a learning technology
    assert set(plants[f"bidding_{convert.MARKET_ID}"]) == {
        convert.NAIVE,
        convert.SUPPORT,
    }
    cases = yaml.safe_load((portfolio / "config.yaml").read_text())
    assert (
        cases[convert.LEARNING_CASE]["unit_operators"] == "unit_operators_learning.csv"
    )
    assert cases[convert.LEARNING_CASE]["bidding_strategy_params"]["nbins"] == 2
    assert cases[convert.CASE]["unit_operators"] is None
    assert json.loads((portfolio / "scenario_meta.json").read_text())["learning"][
        "operators"
    ] == ["A"]
    cases = yaml.safe_load((folder / "config.yaml").read_text())
    learning = cases[convert.LEARNING_CASE]
    market = learning["markets_config"][convert.MARKET_ID]
    # one product per auction, one auction per period, the same lead as the day-ahead auction
    assert market["products"] == [
        {"duration": "30min", "count": 1, "first_delivery": "900min"}
    ]
    assert market["opening_frequency"] == "30min"
    assert learning["learning_config"]["learning_mode"] is True
    assert learning["learning_config"][
        "max_bid_price"
    ] == convert.default_max_bid_price(convert.scenario_merit_order(stack), 3000.0)
    assert pd.Timestamp(learning["end_date"]) == min(
        stack.index[0] + pd.Timedelta(weeks=1), stack.index[-1] + stack.freq
    )
    year = cases[convert.LEARNING_YEAR_CASE]
    assert year["learning_config"]["learning_mode"] is False
    assert year["learning_config"]["trained_policies_load_path"].startswith(
        "learned_strategies/gb_learning_learning"
    )
    # the gas plants learn, every other unit bids as before
    units = pd.read_csv(folder / "powerplant_units_learning.csv", index_col=0)
    base = pd.read_csv(folder / "powerplant_units.csv", index_col=0)
    learns = units["technology"] == "CCGT"
    assert learns.sum() == 2 and set(
        units.loc[learns, f"bidding_{convert.MARKET_ID}"]
    ) == {convert.LEARNING}
    assert units.loc[~learns, f"bidding_{convert.MARKET_ID}"].equals(
        base.loc[~learns, f"bidding_{convert.MARKET_ID}"]
    )
    meta = json.loads((folder / "scenario_meta.json").read_text())
    assert meta["learning"]["units"] == units.index[learns].tolist()

    # with the day-ahead units in the rolling market the prices are those of the day-ahead auction
    rolling = dict(learning, powerplant_units="powerplant_units.csv")
    del rolling["learning_config"]
    cases["rolling"] = rolling
    (folder / "config.yaml").write_text(yaml.safe_dump(cases, sort_keys=False))
    world = World(log_level="WARNING")
    load_scenario_folder(
        world, inputs_path=str(tmp_path), scenario="gb_learning", study_case="rolling"
    )
    world.run()
    results = pd.DataFrame(
        [
            r
            for op in world.market_operators.values()
            for role in op.roles
            if isinstance(role, MarketRole)
            for r in role.results
        ]
    )
    price = results.set_index(pd.to_datetime(results["product_start"]))[
        "price"
    ].sort_index()
    expected = run[0].loc[price.index]
    assert len(price) == len(stack.index)
    np.testing.assert_allclose(price.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-9)
