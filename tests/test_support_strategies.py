# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from assume import World
from assume.common.exceptions import ValidationError
from assume.common.forecaster import PowerplantForecaster
from assume.scenario.loader_csv import load_scenario_folder
from assume.strategies import EnergyNaiveStrategy, bidding_strategies
from assume.strategies.support_strategies import (
    EnergyNaiveSupportStrategy,
    support_bid_price,
)
from assume.units import PowerPlant

start = datetime(2023, 7, 1)
products = [
    (start + timedelta(hours=h), start + timedelta(hours=h + 1), None) for h in range(4)
]


def make_plant(additional_cost=3.0, **support) -> PowerPlant:
    index = pd.date_range(start, periods=6, freq="h")
    forecaster = PowerplantForecaster(
        index,
        availability=1,
        fuel_prices={"reference": [110.0, 150.0, np.nan, 110.0, 110.0, 110.0]},
        market_prices={"EOM": 50},
    )
    return PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="wind_onshore",
        bidding_strategies={"EOM": EnergyNaiveSupportStrategy()},
        forecaster=forecaster,
        max_power=100,
        additional_cost=additional_cost,
        **support,
    )


def bid_prices(unit, mock_market_config, strategy=None):
    strategy = strategy or EnergyNaiveSupportStrategy()
    return [
        b["price"] for b in strategy.calculate_bids(unit, mock_market_config, products)
    ]


def test_strategy_is_registered():
    assert (
        bidding_strategies["powerplant_energy_naive_support"]
        is EnergyNaiveSupportStrategy
    )


def test_no_contract_bids_like_the_naive_strategy(mock_market_config):
    unit = make_plant()
    expected = EnergyNaiveStrategy().calculate_bids(unit, mock_market_config, products)
    assert (
        EnergyNaiveSupportStrategy().calculate_bids(unit, mock_market_config, products)
        == expected
    )
    assert bid_prices(unit, mock_market_config) == [3.0] * 4


def test_premium_is_bid_below_marginal_cost(mock_market_config):
    # 1.5 certificates per MWh worth 60 each, running cost 3: pays up to 87 to stay on
    unit = make_plant(support_scheme="premium", support_value=90.0)
    bids = EnergyNaiveSupportStrategy().calculate_bids(
        unit, mock_market_config, products
    )
    assert [b["price"] for b in bids] == [-87.0] * 4
    assert [b["volume"] for b in bids] == [100.0] * 4


def test_cfd_bids_minus_strike_unless_payment_stops_at_negative_prices(
    mock_market_config,
):
    six_hour = make_plant(
        support_scheme="cfd", support_value=100.0, support_neg_price_rule="six_hour"
    )
    assert bid_prices(six_hour, mock_market_config) == [-97.0] * 4
    # no payment below zero, the strike price from zero upwards: worth running from zero on
    any_hour = make_plant(
        support_scheme="cfd", support_value=100.0, support_neg_price_rule="any_hour"
    )
    assert bid_prices(any_hour, mock_market_config) == [0.0] * 4
    # a unit that is paid to run anyway keeps running below zero, down to its marginal cost
    paid_to_run = make_plant(
        additional_cost=-5.0,
        support_scheme="cfd",
        support_value=100.0,
        support_neg_price_rule="any_hour",
    )
    assert bid_prices(paid_to_run, mock_market_config) == [-5.0] * 4


def test_bid_is_kept_within_the_price_limits_of_the_market(mock_market_config):
    unit = make_plant(support_scheme="cfd", support_value=700.0)
    assert (
        bid_prices(unit, mock_market_config)
        == [mock_market_config.minimum_bid_price] * 4
    )


def test_cfd_against_a_reference_price_follows_it(mock_market_config):
    # strike 130, running cost 68: top-up 20 -> bids 48; reference above the strike -> pays back
    # 20, bids 88; no reference published -> marginal cost
    unit = make_plant(
        additional_cost=68.0,
        support_scheme="cfd",
        support_value=130.0,
        support_reference="reference",
    )
    assert bid_prices(unit, mock_market_config) == [48.0, 88.0, 68.0, 48.0]


def test_missing_reference_series_is_an_error(mock_market_config):
    unit = make_plant(
        support_scheme="cfd", support_value=130.0, support_reference="not_there"
    )
    with pytest.raises(ValueError, match="no price series"):
        bid_prices(unit, mock_market_config)


def test_scenario_cells_filled_with_zero_mean_no_contract(mock_market_config):
    # the csv loader fills empty cells with 0
    unit = make_plant(
        support_scheme=0, support_value=0, support_neg_price_rule=0, support_reference=0
    )
    assert unit.support_scheme == "" and unit.support_reference == ""
    assert bid_prices(unit, mock_market_config) == [3.0] * 4


def test_unknown_scheme_is_rejected():
    with pytest.raises(ValidationError):
        make_plant(support_scheme="lottery")
    with pytest.raises(ValueError):
        support_bid_price(3.0, "lottery", 1.0)


def test_support_bid_price_on_a_series():
    cost = np.array([68.0, 68.0, 68.0])
    reference = np.array([110.0, 150.0, np.nan])
    assert support_bid_price(
        cost, "cfd", 130.0, reference_price=reference
    ).tolist() == [48.0, 88.0, 68.0]
    assert support_bid_price(cost, "premium", 45.0).tolist() == [23.0] * 3
    assert support_bid_price(cost, "cfd", 130.0).tolist() == [-62.0] * 3
    assert support_bid_price(
        np.array([68.0, -5.0]), "cfd", 130.0, neg_price_rule="any_hour"
    ).tolist() == [0.0, -5.0]
    assert support_bid_price(cost, "", 45.0) is cost


SCENARIO_CONFIG = """
base:
  start_date: 2023-07-01 00:00
  end_date: 2023-07-01 11:00
  time_step: 1h
  save_frequency_hours: null
  markets_config:
    EOM:
      operator: EOM_operator
      product_type: energy
      products:
        - duration: 1h
          count: 1
          first_delivery: 1h
      opening_frequency: 1h
      opening_duration: 1h
      volume_unit: MWh
      maximum_bid_volume: 100000
      maximum_bid_price: 3000
      minimum_bid_price: -500
      price_unit: EUR/MWh
      market_mechanism: pay_as_clear
"""

# five plants of 100 MW each; the supported ones bid -97, -87, 0 and 68 - (130 - reference),
# the gas plant its marginal cost of 60
SCENARIO_UNITS = """name,technology,bidding_EOM,fuel_type,emission_factor,max_power,min_power,efficiency,additional_cost,unit_operator,support_scheme,support_value,support_neg_price_rule,support_reference
cfd_wind,wind_offshore,powerplant_energy_naive_support,others,0,100,0,1,3,renewables,cfd,100,six_hour,
ro_wind,wind_onshore,powerplant_energy_naive_support,others,0,100,0,1,3,renewables,premium,90,,
cfd_wind_any_hour,wind_offshore,powerplant_energy_naive_support,others,0,100,0,1,3,renewables,cfd,100,any_hour,
cfd_biomass,biomass,powerplant_energy_naive_support,others,0,100,0,1,68,biomass,cfd,130,,reference
gas,combined cycle gas turbine,powerplant_energy_naive,natural gas,0,100,0,0.5,0,gas,,,,
"""

SCENARIO_DEMAND_UNITS = """name,technology,bidding_EOM,max_power,min_power,unit_operator
demand_EOM,inflex_demand,demand_energy_naive,1000000,0,demand
"""

# per delivery hour: demand, reference price of the biomass contract, expected clearing price
SCENARIO_HOURS = [
    (50, 110, -97),
    (150, 110, -87),
    (250, 110, 0),
    (350, 110, 48),  # biomass tops up 20 and undercuts gas
    (450, 110, 60),
    (50, 150, -97),
    (150, 150, -87),
    (250, 150, 0),
    (350, 150, 60),  # biomass pays back 20 and comes after gas
    (450, 150, 88),
]


def write_scenario(folder):
    index = pd.date_range(
        "2023-07-01 00:00", "2023-07-01 11:00", freq="h", name="datetime"
    )
    # the first hour is before the first delivery, the last one after the last
    demand = [50] + [hour[0] for hour in SCENARIO_HOURS] + [50]
    reference = [110] + [hour[1] for hour in SCENARIO_HOURS] + [150]
    folder.mkdir(parents=True)
    (folder / "config.yaml").write_text(SCENARIO_CONFIG)
    (folder / "powerplant_units.csv").write_text(SCENARIO_UNITS)
    (folder / "demand_units.csv").write_text(SCENARIO_DEMAND_UNITS)
    pd.DataFrame({"demand_EOM": demand}, index=index).to_csv(folder / "demand_df.csv")
    pd.DataFrame(
        {"natural gas": 30, "co2": 0, "reference": reference}, index=index
    ).to_csv(folder / "fuel_prices_df.csv")


def test_csv_scenario_clears_at_the_bids_of_the_supported_units(tmp_path):
    write_scenario(tmp_path / "inputs" / "support")
    world = World(
        database_uri="", export_csv_path=str(tmp_path / "outputs"), log_level="WARNING"
    )
    load_scenario_folder(
        world,
        inputs_path=str(tmp_path / "inputs"),
        scenario="support",
        study_case="base",
    )
    world.run()

    market_meta = pd.read_csv(tmp_path / "outputs" / "support_base" / "market_meta.csv")
    price = market_meta.set_index(pd.to_datetime(market_meta["product_start"]))[
        "price"
    ].sort_index()
    expected = pd.Series(
        [float(hour[2]) for hour in SCENARIO_HOURS],
        index=pd.date_range("2023-07-01 01:00", periods=len(SCENARIO_HOURS), freq="h"),
    )
    np.testing.assert_allclose(
        price.loc[expected.index].to_numpy(), expected.to_numpy(), atol=1e-9
    )


def test_contract_payment_per_scheme():
    t = start
    # a premium is paid whatever the price
    assert make_plant(support_scheme="premium", support_value=90).support_payment(
        t, 50.0, 100
    ) == pytest.approx(9000)
    # a cfd tops the price up to the strike price, also from a negative price
    cfd = make_plant(
        support_scheme="cfd", support_value=100, support_neg_price_rule="six_hour"
    )
    assert cfd.support_payment(t, 50.0, 100) == pytest.approx(5000)
    assert cfd.support_payment(t, -20.0, 100) == pytest.approx(12000)
    # under the any-hour rule nothing is paid at a negative price
    any_hour = make_plant(
        support_scheme="cfd", support_value=100, support_neg_price_rule="any_hour"
    )
    assert any_hour.support_payment(t, 50.0, 100) == pytest.approx(5000)
    assert any_hour.support_payment(t, -20.0, 100) == 0.0
    # against a reference price the payment is the strike less that price, whatever the market price
    reference = make_plant(
        support_scheme="cfd", support_value=130, support_reference="reference"
    )
    assert reference.support_payment(t, 50.0, 100) == pytest.approx(2000)
    assert reference.support_payment(products[1][0], 50.0, 100) == pytest.approx(-2000)
    assert (
        reference.support_payment(products[2][0], 50.0, 100) == 0.0
    )  # no reference price
    # nothing without a contract, nothing for no volume
    assert make_plant().support_payment(t, 50.0, 100) == 0.0
    assert cfd.support_payment(t, 50.0, 0) == 0.0


def test_levy_takes_a_share_of_the_receipts_above_the_benchmark(mock_market_config):
    unit = make_plant(
        support_scheme="premium", support_value=90, levy_rate=0.45, levy_benchmark=75
    )
    t = start
    assert unit.levy_payment(t, 100.0, 100) == pytest.approx(-0.45 * 25 * 100)
    assert unit.levy_payment(t, 60.0, 100) == 0.0
    assert unit.levy_payment(t, 100.0, -50) == 0.0
    assert make_plant().levy_payment(t, 100.0, 100) == 0.0
    orderbook = [
        {
            "start_time": products[0][0],
            "end_time": products[0][1],
            "only_hours": None,
            "price": -87,
            "volume": 100,
            "accepted_price": 100.0,
            "accepted_volume": 100,
        }
    ]
    unit.set_dispatch_plan(mock_market_config, orderbook)
    unit.calculate_cashflow_and_reward(mock_market_config, orderbook)
    assert unit.outputs["energy_cashflow"].at[products[0][0]] == pytest.approx(10000)
    assert unit.outputs["support_cashflow"].at[products[0][0]] == pytest.approx(9000)
    assert unit.outputs["levy_cashflow"].at[products[0][0]] == pytest.approx(-1125)
    with pytest.raises(ValidationError):
        make_plant(levy_rate=1.5)


def test_cashflow_books_the_contract_payment_beside_the_market_cashflow(
    mock_market_config,
):
    unit = make_plant(support_scheme="premium", support_value=90)
    orderbook = [
        {
            "start_time": products[0][0],
            "end_time": products[0][1],
            "only_hours": None,
            "price": -87,
            "volume": 100,
            "accepted_price": 50.0,
            "accepted_volume": 100,
        }
    ]
    unit.set_dispatch_plan(mock_market_config, orderbook)
    unit.calculate_cashflow_and_reward(mock_market_config, orderbook)
    assert unit.outputs["energy_cashflow"].at[products[0][0]] == pytest.approx(5000)
    assert unit.outputs["support_cashflow"].at[products[0][0]] == pytest.approx(9000)
    assert unit.outputs["support_cashflow"].at[products[1][0]] == 0.0

    plain = make_plant()
    plain.set_dispatch_plan(mock_market_config, orderbook)
    plain.calculate_cashflow_and_reward(mock_market_config, orderbook)
    assert plain.outputs["energy_cashflow"].at[products[0][0]] == pytest.approx(5000)
    assert plain.outputs["support_cashflow"].at[products[0][0]] == 0.0
