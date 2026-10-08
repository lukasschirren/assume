# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The adapters read the folders of ``assume_gb`` into the data contract: on a small stand-in
for a scenario folder and saved runs (two weeks of January 2023 and the week before), and, where
it is built, on the real 2023 folder."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from assume_gb import paths
from assume_gb.validation import adapters as A
from assume_gb.validation.contract import TZ

DELIVERY = pd.date_range(
    "2023-01-01", "2023-01-15", freq="30min", inclusive="left", name="datetime"
)
FOLDER_INDEX = pd.date_range(
    "2022-12-31", "2023-01-15", freq="30min", name="datetime"
)  # lead-in day, closing row


def _write(frame: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """Scenario folders for 2022 (observed prices of the last week only) and 2023, and saved runs."""
    inputs, results = tmp_path / "inputs", tmp_path / "results"
    hours = np.arange(len(DELIVERY)) // 2
    observed = pd.DataFrame(
        {
            "n2ex_day_ahead": 50.0 + hours % 24,
            "epex_hh_day_ahead": 50.0 + np.arange(len(DELIVERY)) % 7,
        },
        index=DELIVERY,
    )
    _write(observed, inputs / "gb_2023" / "observed_prices.csv")
    reference = pd.DataFrame(
        {"scenario_merit_order": observed.n2ex_day_ahead - 5.0, "admissible": True},
        index=DELIVERY,
    )
    reference.loc["2023-01-02 10:30", "admissible"] = False
    _write(reference, inputs / "gb_2023" / "reference_prices.csv")
    before = pd.date_range(
        "2022-12-25", "2023-01-01", freq="30min", inclusive="left", name="datetime"
    )
    _write(
        pd.DataFrame({"n2ex_day_ahead": 40.0, "epex_hh_day_ahead": 40.0}, index=before),
        inputs / "gb_2022" / "observed_prices.csv",
    )
    _write(
        pd.DataFrame({"scenario_merit_order": 35.0, "admissible": True}, index=before),
        inputs / "gb_2022" / "reference_prices.csv",
    )

    folder = inputs / "gb_2023"
    units = pd.DataFrame(
        {
            "name": ["w1", "s1", "g1", "x1", "u1"],
            "technology": ["Wind", "Solar", "CCGT", "IC_EXPORT", "unserved_demand"],
            "max_power": [1000.0, 500.0, 2000.0, 800.0, 100000.0],
        }
    )
    units.to_csv(folder / "powerplant_units.csv", index=False)
    pd.DataFrame({"w1": 0.5, "s1": 0.2, "x1": 0.75}, index=FOLDER_INDEX).to_csv(
        folder / "availability_df.csv.gz"
    )
    pd.DataFrame({"w1": 0.6, "s1": 0.25}, index=FOLDER_INDEX).to_csv(
        folder / "availability_forecast_df.csv.gz"
    )
    pd.DataFrame({"demand": 30000.0}, index=FOLDER_INDEX).to_csv(
        folder / "demand_df.csv"
    )
    pd.DataFrame({"demand": 31000.0}, index=FOLDER_INDEX).to_csv(
        folder / "demand_forecast_df.csv"
    )
    pd.DataFrame({"natural gas": 35.0, "co2": 70.0}, index=FOLDER_INDEX).to_csv(
        folder / "fuel_prices_df.csv"
    )
    pd.DataFrame(
        {"wind_tx_forecast_mw": 400.0, "nd_forecast_mw": 29000.0}, index=FOLDER_INDEX
    ).to_csv(folder / "penetration_forecasts.csv")
    meta = {
        "delivery_start": "2023-01-01 00:00:00",
        "delivery_end": "2023-01-15 00:00:00",
        "battery_fleet": {
            "power_mw": 100.0,
            "energy_mwh": 200.0,
            "round_trip_efficiency": 0.9,
        },
    }
    (folder / "scenario_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    cases = {
        "day_ahead": {"start_date": "2022-12-31 00:00", "end_date": "2023-01-15 00:00"},
        "learning_test_s0": {
            "start_date": "2023-01-04 00:00",
            "end_date": "2023-01-08 00:00",
        },
        "learning_test_s0_year": {
            "start_date": "2022-12-31 00:00",
            "end_date": "2023-01-15 00:00",
        },
    }
    (folder / "config.yaml").write_text(yaml.safe_dump(cases), encoding="utf-8")

    _write(
        pd.DataFrame({"DA": observed.n2ex_day_ahead - 4.0}, index=DELIVERY),
        results / "prices" / "gb_2023_day_ahead.csv",
    )
    for seed in (0, 1):
        _write(
            pd.DataFrame({"DA": observed.n2ex_day_ahead + seed}, index=DELIVERY),
            results / "prices" / f"gb_2023_learning_test_s{seed}_year.csv",
        )
    _write(
        pd.DataFrame({"DA": observed.n2ex_day_ahead}, index=DELIVERY),
        results / "prices" / "gb_2023_learning_test_s0_year_last.csv",
    )
    training = DELIVERY[(DELIVERY >= "2023-01-05") & (DELIVERY < "2023-01-08")]
    _write(
        pd.DataFrame({"DA": 60.0}, index=training),
        results / "prices" / "gb_2023_learning_test_s0.csv",
    )
    monkeypatch.setattr(paths, "scenarios_dir", lambda: inputs)
    monkeypatch.setattr(paths, "TOOL_ROOT", tmp_path)
    return tmp_path


def test_observed_prices_are_local_hourly_with_the_week_before(layout):
    price = A.observed(2023)
    assert str(price.index.tz) == TZ and price.index[0] == pd.Timestamp(
        "2022-12-25", tz=TZ
    )
    assert (
        len(price) == 21 * 24
        and (price.index[1:] - price.index[:-1] == pd.Timedelta(hours=1)).all()
    )
    assert (
        price[pd.Timestamp("2023-01-01 05:00", tz=TZ)] == 55.0
        and (price.loc[:"2022-12-31"] == 40.0).all()
    )
    assert np.isnan(
        price[pd.Timestamp("2023-01-02 10:00", tz=TZ)]
    )  # an inadmissible half-hour empties its hour
    half_hourly = A.observed(2023, "epex_hh_day_ahead")
    assert len(half_hourly) == 21 * 48 and A.market("epex_hh_day_ahead") == "APX"
    with pytest.raises(ValueError):
        A.observed(2023, "system_price")


def test_runs_competitive_and_seeds(layout):
    competitive = A.competitive(2023)
    assert (
        len(competitive) == len(DELIVERY) and competitive.iloc[0] == 46.0
    )  # the saved run
    (layout / "results" / "prices" / "gb_2023_day_ahead.csv").unlink()
    assert (
        A.competitive(2023).iloc[0] == 45.0
    )  # the merit order of the scenario's offers
    seeds = A.abm(2023, "learning_test")
    assert seeds.columns.tolist() == [0, 1] and (seeds[1] - seeds[0] == 1.0).all()
    assert A.abm(2023, "learning_test", policies="last").columns.tolist() == [0]
    with pytest.raises(FileNotFoundError):
        A.abm(2023, "learning_test", seeds=[0, 5])
    with pytest.raises(FileNotFoundError):
        A.run(2023, "day_ahead_storage")


def test_exog_from_the_scenario_inputs(layout):
    x = A.exog(2023)
    assert str(x.index.tz) == TZ and len(x) == len(
        DELIVERY
    )  # without the lead-in day and closing row
    first = x.iloc[0]
    assert first.wind == pytest.approx(0.5) and first.solar == pytest.approx(0.1)
    assert first.demand == pytest.approx(
        30.0 - 0.6
    )  # less the export capacity, 0.75 x 800 MW
    # the export unit has no forecast column: its forecast availability is its outturn's
    assert first.demand_forecast == pytest.approx(31.0 - 0.6)
    assert first.wind_forecast == pytest.approx(
        0.6
    ) and first.solar_forecast == pytest.approx(0.125)
    assert first.residual_demand == pytest.approx(29.4 - 0.5 - 0.1)
    assert first.available_capacity == pytest.approx(2.0)  # the CCGT only
    assert first.gas == 35.0 and first.carbon == 70.0
    assert first.wind_tx_forecast == pytest.approx(
        0.4
    ) and first.nd_forecast == pytest.approx(29.0)


def test_windows_battery_and_the_whole_contract(layout):
    windows = A.periods(2023, "learning_test")
    start, end = windows["calibration"]
    assert start == pd.Timestamp("2023-01-05 00:00", tz=TZ) and end == pd.Timestamp(
        "2023-01-07 23:30", tz=TZ
    )
    battery = A.battery(2023)
    assert (battery.power, battery.energy, battery.round_trip_efficiency) == (
        100.0,
        200.0,
        0.9,
    )
    data = A.load(2023, "learning_test")
    assert data.names == ["naive", "competitive", "abm"] and data.is_ensemble("abm")
    assert data.resolution == pd.Timedelta(hours=1)
    assert data.in_sample("year") and not data.in_sample("headline")
    sample = data.sample()
    assert sample.index.min() == pd.Timestamp("2023-01-01", tz=TZ)
    assert not sample.index.isin(data.sample("calibration").index).any()
    with pytest.raises(ValueError):
        A.load([2022, 2023], "learning_test")


def test_the_training_window_without_its_case_in_the_config(layout):
    """A config rewritten without the arm's cases: the training run's saved prices give the window."""
    path = layout / "inputs" / "gb_2023" / "config.yaml"
    path.write_text(
        yaml.safe_dump({"day_ahead": {"start_date": "2022-12-31 00:00"}}),
        encoding="utf-8",
    )
    start, end = A.periods(2023, "learning_test")["calibration"]
    assert start == pd.Timestamp("2023-01-05 00:00", tz=TZ)
    assert end == pd.Timestamp("2023-01-07 23:30", tz=TZ)
    with pytest.raises(KeyError):
        A.periods(2023, "learning_other")


REAL = paths.scenarios_dir() / "gb_2023"
REAL_ARM = "learning_novdec_own_e100"


@pytest.mark.skipif(
    not (REAL / "observed_prices.csv").exists()
    or not list(paths.results_dir("prices").glob(f"gb_2023_{REAL_ARM}_s*_year.csv")),
    reason="the 2023 folder with observed prices and the arm's year runs is not here",
)
def test_the_real_2023_folder():
    data = A.load(2023, REAL_ARM)
    assert data.resolution == pd.Timedelta(hours=1) and data.is_ensemble("abm")
    start, end = data.periods["calibration"]
    assert start == pd.Timestamp("2023-11-06", tz=TZ)
    sample = data.sample()
    assert 8000 < len(sample) < 8760 - 27 * 24
    x = data.exog
    assert (
        7 < x["wind"].mean() < 12
        and 20 < x["demand"].mean() < 32
        and 30 < x["gas"].mean() < 45
    )
