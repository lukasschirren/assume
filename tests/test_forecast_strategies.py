# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine

from assume import World
from assume.common.forecaster import DemandForecaster, PowerplantForecaster
from assume.scenario.loader_csv import load_scenario_folder
from assume.strategies import bidding_strategies
from assume.strategies.forecast_strategies import (
    DemandEnergyNaiveRebalanceStrategy,
    EnergyNaiveForecastStrategy,
    EnergyNaiveRebalanceStrategy,
)
from assume.units import Demand, PowerPlant

start = datetime(2023, 7, 1)
products = [
    (start + timedelta(hours=h), start + timedelta(hours=h + 1), None) for h in range(4)
]


def make_plant(strategy, availability, availability_forecast=None, **support):
    index = pd.date_range(start, periods=5, freq="h")
    forecaster = PowerplantForecaster(
        index,
        availability=availability,
        availability_forecast=availability_forecast,
        fuel_prices={"others": 0, "co2": 0},
        market_prices={"EOM": 50},
    )
    return PowerPlant(
        id="plant",
        unit_operator="operator",
        technology="wind",
        bidding_strategies={"EOM": strategy},
        forecaster=forecaster,
        max_power=100,
        additional_cost=3.0,
        **support,
    )


def orders(unit, mock_market_config):
    bids = unit.bidding_strategies["EOM"].calculate_bids(
        unit, mock_market_config, products
    )
    return sorted(
        (b["start_time"].hour, round(b["volume"], 9), b["price"]) for b in bids
    )


def test_strategies_are_registered():
    assert (
        bidding_strategies["powerplant_energy_naive_forecast"]
        is EnergyNaiveForecastStrategy
    )
    assert (
        bidding_strategies["demand_energy_naive_forecast"]
        is EnergyNaiveForecastStrategy
    )
    assert (
        bidding_strategies["powerplant_energy_naive_rebalance"]
        is EnergyNaiveRebalanceStrategy
    )
    assert (
        bidding_strategies["demand_energy_naive_rebalance"]
        is DemandEnergyNaiveRebalanceStrategy
    )


def test_forecast_strategy_bids_the_forecast_not_the_outturn(mock_market_config):
    unit = make_plant(
        EnergyNaiveForecastStrategy(),
        availability=[1, 1, 1, 1, 1],
        availability_forecast=[0.5, 0.5, 1, 0, 1],
    )
    assert orders(unit, mock_market_config) == [
        (0, 50, 3.0),
        (1, 50, 3.0),
        (2, 100, 3.0),
    ]
    # the outturn is what the plant can deliver
    assert unit.forecaster.availability.data.tolist() == [1, 1, 1, 1, 1]

    # without a forecast the forecast is the outturn
    unit = make_plant(EnergyNaiveForecastStrategy(), availability=[1, 0.5, 1, 1, 1])
    assert orders(unit, mock_market_config) == [
        (0, 100, 3.0),
        (1, 50, 3.0),
        (2, 100, 3.0),
        (3, 100, 3.0),
    ]

    # a support contract sets the price as in the support strategy
    unit = make_plant(
        EnergyNaiveForecastStrategy(),
        availability=[1, 1, 1, 1, 1],
        availability_forecast=[0.5, 0.5, 1, 0, 1],
        support_scheme="premium",
        support_value=90.0,
    )
    assert orders(unit, mock_market_config) == [
        (0, 50, -87.0),
        (1, 50, -87.0),
        (2, 100, -87.0),
    ]


def test_rebalance_strategy_trades_the_difference_in_both_directions(
    mock_market_config,
):
    unit = make_plant(
        EnergyNaiveRebalanceStrategy(), availability=[0.4, 1.0, 0.7, 1.0, 1.0]
    )
    for product, sold in zip(products, [60, 60, 70, 0]):
        unit.outputs["energy"].at[product[0]] = sold
    cap = mock_market_config.maximum_bid_price
    assert orders(unit, mock_market_config) == [
        (
            0,
            -40,
            3.0,
        ),  # buys back what it can deliver if that is cheaper than generating
        (0, -20, cap),  # buys back what it cannot deliver whatever the price
        (1, -60, 3.0),
        (1, 40, 3.0),  # offers what it can deliver beyond its position
        (2, -70, 3.0),
        (3, 100, 3.0),
    ]


def test_rebalance_strategy_respects_minimum_power_and_support(mock_market_config):
    unit = make_plant(
        EnergyNaiveRebalanceStrategy(),
        availability=[1.0, 1.0, 1.0, 1.0, 1.0],
        support_scheme="premium",
        support_value=90.0,
    )
    unit.min_power = 30
    for product, sold in zip(products, [20, 50, 0, 100]):
        unit.outputs["energy"].at[product[0]] = sold
    assert orders(unit, mock_market_config) == [
        (0, 80, -87.0),  # sold less than the minimum: nothing to buy back
        (1, -20, -87.0),
        (1, 50, -87.0),
        (2, 100, -87.0),
        (3, -70, -87.0),
    ]


def test_demand_rebalance_buys_what_it_needs_and_sells_what_it_does_not(
    mock_market_config,
):
    index = pd.date_range(start, periods=5, freq="h")
    forecaster = DemandForecaster(
        index, demand=[-120, -80, -100, -100, -100], demand_forecast=[-100] * 5
    )
    unit = Demand(
        id="demand",
        unit_operator="operator",
        technology="demand",
        bidding_strategies={"EOM": DemandEnergyNaiveRebalanceStrategy()},
        forecaster=forecaster,
        max_power=-1000,
        min_power=0,
    )
    for product in products:
        unit.outputs["energy"].at[product[0]] = -100
    cap, floor = (
        mock_market_config.maximum_bid_price,
        mock_market_config.minimum_bid_price,
    )
    assert orders(unit, mock_market_config) == [(0, -20, cap), (1, 20, floor)]

    # the forecast strategy bids the forecast demand
    unit.bidding_strategies["EOM"] = EnergyNaiveForecastStrategy()
    for product in products:
        unit.outputs["energy"].at[product[0]] = 0
    assert [v for h, v, p in orders(unit, mock_market_config)] == [-100] * 4


SCENARIO_CONFIG = """
base:
  start_date: 2023-07-01 00:00
  end_date: 2023-07-01 07:00
  time_step: 1h
  save_frequency_hours: null
  markets_config:
    DA:
      operator: market_operator
      product_type: energy
      start_date: 2023-07-01 00:00
      products:
        - duration: 1h
          count: 4
          first_delivery: 2h
      opening_frequency: 24h
      opening_duration: 30min
      volume_unit: MWh
      maximum_bid_volume: 100000
      maximum_bid_price: 3000
      minimum_bid_price: -500
      price_unit: EUR/MWh
      market_mechanism: pay_as_clear
    ID:
      operator: market_operator
      product_type: energy
      start_date: 2023-07-01 01:00
      products:
        - duration: 1h
          count: 4
          first_delivery: 1h
      opening_frequency: 24h
      opening_duration: 30min
      volume_unit: MWh
      maximum_bid_volume: 100000
      maximum_bid_price: 3000
      minimum_bid_price: -500
      price_unit: EUR/MWh
      market_mechanism: complex_clearing
      param_dict:
        solver_name: highs
        pricing_mechanism: pay_as_clear
      additional_fields:
        - bid_type
"""

SCENARIO_UNITS = """name,technology,bidding_DA,bidding_ID,fuel_type,emission_factor,max_power,min_power,efficiency,additional_cost,unit_operator
wind,wind_onshore,powerplant_energy_naive_forecast,powerplant_energy_naive_rebalance,others,0,100,0,1,3,renewables
gas,combined cycle gas turbine,powerplant_energy_naive_forecast,powerplant_energy_naive_rebalance,natural gas,0,200,0,0.5,0,gas
"""

SCENARIO_DEMAND_UNITS = """name,technology,bidding_DA,bidding_ID,max_power,min_power,unit_operator
demand,inflex_demand,demand_energy_naive_forecast,demand_energy_naive_rebalance,1000000,0,demand
"""


def write_scenario(folder, wind_outturn, demand_outturn):
    index = pd.date_range(
        "2023-07-01 00:00", "2023-07-01 07:00", freq="h", name="datetime"
    )
    folder.mkdir(parents=True)
    (folder / "config.yaml").write_text(SCENARIO_CONFIG)
    (folder / "powerplant_units.csv").write_text(SCENARIO_UNITS)
    (folder / "demand_units.csv").write_text(SCENARIO_DEMAND_UNITS)
    pd.DataFrame({"natural gas": 25, "co2": 0}, index=index).to_csv(
        folder / "fuel_prices_df.csv"
    )
    pd.DataFrame({"wind": wind_outturn}, index=index).to_csv(
        folder / "availability_df.csv"
    )
    pd.DataFrame({"wind": 1.0}, index=index).to_csv(
        folder / "availability_forecast_df.csv"
    )
    pd.DataFrame({"demand": demand_outturn}, index=index).to_csv(
        folder / "demand_df.csv"
    )
    pd.DataFrame({"demand": 150}, index=index).to_csv(folder / "demand_forecast_df.csv")


def run_scenario(tmp_path, wind_outturn, demand_outturn):
    write_scenario(tmp_path / "inputs" / "sequence", wind_outturn, demand_outturn)
    database_uri = f"sqlite:///{(tmp_path / 'sequence.db').as_posix()}"
    world = World(database_uri=database_uri, log_level="WARNING")
    load_scenario_folder(
        world,
        inputs_path=str(tmp_path / "inputs"),
        scenario="sequence",
        study_case="base",
    )
    world.run()
    engine = create_engine(database_uri)
    with engine.begin() as conn:
        meta = pd.read_sql(
            "SELECT market_id, product_start, price FROM market_meta", conn
        )
    engine.dispose()
    prices = meta.pivot(index="product_start", columns="market_id", values="price")
    delivery = pd.date_range("2023-07-01 02:00", periods=4, freq="h")
    positions = {
        unit_id: [unit.outputs["energy"].at[t] for t in delivery]
        for unit_id, unit in world.units.items()
    }
    return prices, positions


def test_wind_short_of_its_forecast_buys_back_in_the_second_market(tmp_path):
    # day ahead: wind sells its forecast of 100 MW at 3, gas covers the other 50 MW of the
    # forecast demand of 150 MW at its marginal cost of 50. Then 40 MW of wind do not turn up.
    prices, positions = run_scenario(tmp_path, wind_outturn=0.6, demand_outturn=150)
    assert prices["DA"].tolist() == pytest.approx([50.0] * 4)
    # in the second market wind buys the 40 MW back, which gas provides at 50
    assert prices["ID"].tolist() == pytest.approx([50.0] * 4)
    assert positions["wind"] == pytest.approx([60.0] * 4)
    assert positions["gas"] == pytest.approx([90.0] * 4)
    assert positions["demand"] == pytest.approx([-150.0] * 4)


def test_long_system_sells_back_at_the_price_of_the_displaced_plant(tmp_path):
    # the demand turns out 30 MW lower than forecast and wind as forecast: demand sells 30 MW
    # back at any price and gas, which generates at 50, is the one to buy it
    prices, positions = run_scenario(tmp_path, wind_outturn=1.0, demand_outturn=120)
    assert prices["DA"].tolist() == pytest.approx([50.0] * 4)
    assert prices["ID"].tolist() == pytest.approx([50.0] * 4)
    assert positions["wind"] == pytest.approx([100.0] * 4)
    assert positions["gas"] == pytest.approx([20.0] * 4)
    assert positions["demand"] == pytest.approx([-120.0] * 4)
    # what every unit delivers is what it holds, so the balance closes without any clipping
    assert np.isclose(sum(p[0] for p in positions.values()), 0.0)
