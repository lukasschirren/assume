# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""The generator levy settled on a small invented run: scope, grouping, benchmark, allowance,
threshold and the forward-sale shelter."""

import numpy as np
import pandas as pd
import pytest

from assume_gb import egl

HALF_HOURS = 17520


def write_run(folder, units, prices_by_unit):
    """A year of orders: every unit sells 100 MW in every half-hour at its own price series."""
    index = pd.date_range("2023-01-01", periods=HALF_HOURS, freq="30min")
    rows = []
    for unit, price in prices_by_unit.items():
        rows.append(
            pd.DataFrame(
                {
                    "unit_id": unit,
                    "start_time": index,
                    "accepted_volume": 100.0,
                    "accepted_price": price,
                    "market_id": "DA",
                }
            )
        )
    folder.mkdir(parents=True)
    pd.concat(rows).to_csv(folder / "market_orders.csv", index=False)


def test_levy_on_realised_receipts_above_the_benchmark(tmp_path):
    units = pd.DataFrame(
        {
            "technology": ["Wind", "Wind", "Wind", "CCGT", "Solar"],
            "support_scheme": ["premium", "premium", "cfd", "", "premium"],
        },
        index=["wind_a", "wind_b", "wind_cfd", "gas", "small_solar"],
    )
    # 100 MW all year = 876 GWh per unit; wind_a and wind_b belong to one group
    write_run(
        tmp_path / "gb_test_day_ahead",
        units,
        {
            "wind_a": 100.0,
            "wind_b": 60.0,
            "wind_cfd": 100.0,
            "gas": 100.0,
            "small_solar": 100.0,
        },
    )
    owner = pd.Series({"wind_a": "Group", "wind_b": "Group"})
    table = egl.settle(
        tmp_path / "gb_test_day_ahead", units, 2023, owner, benchmark=75.0
    )
    # the group's realised price is the mean of 100 and 60: 80, five above the benchmark
    group = table.loc["Group"]
    assert group["generation_gwh"] == 2 * 876
    assert group["realised_price"] == 80.0
    assert group["excess_gbp_m"] == 5 * 2 * 876e3 / 1e6
    # 8.76 million of excess receipts: below the allowance, so nothing is paid; with a smaller
    # allowance the levy is the rate on the rest
    assert group["levy_gbp_m"] == 0.0
    lower = egl.settle(
        tmp_path / "gb_test_day_ahead",
        units,
        2023,
        owner,
        benchmark=75.0,
        allowance=1e6,
    )
    assert lower.loc["Group", "levy_gbp_m"] == 0.45 * (group["excess_gbp_m"] - 1)
    # a CfD unit and a gas plant are out of scope
    assert "wind_cfd" not in table.index and "gas" not in table.index
    # a lone unit is a group of its own, and pays on its own excess
    solar = table.loc["small_solar"]
    assert solar["levy_gbp_m"] == 0.45 * (25 * 876e3 / 1e6 - 10)
    # a group below the threshold pays nothing
    small = egl.settle(
        tmp_path / "gb_test_day_ahead",
        units,
        2023,
        owner,
        benchmark=75.0,
        threshold_gwh=2000.0,
    )
    assert (
        small.loc["Group", "levy_gbp_m"] == 0.0
        and small.loc["Group", "in_scope"] == np.False_
    )
    # output sold forward at the benchmark shelters the receipts: a fully hedged group pays nothing
    hedged = egl.settle(
        tmp_path / "gb_test_day_ahead",
        units,
        2023,
        owner,
        benchmark=75.0,
        hedge_share=1.0,
    )
    assert (
        hedged.loc["Group", "realised_price"] == 75.0
        and hedged.loc["Group", "levy_gbp_m"] == 0.0
    )
    half = egl.settle(
        tmp_path / "gb_test_day_ahead",
        units,
        2023,
        owner,
        benchmark=75.0,
        hedge_share=0.5,
    )
    assert half.loc["Group", "excess_gbp_m"] == 0.5 * group["excess_gbp_m"]


def test_rate_rises_from_july_2026():
    assert (
        egl.rate_for(2025) == 0.45
        and egl.rate_for(2027) == 0.55
        and egl.rate_for(2026) == 0.5
    )


def test_unit_parameters_mark_the_in_scope_units():
    units = pd.DataFrame(
        {
            "technology": ["Wind", "CCGT", "Biomass"],
            "support_scheme": ["premium", "", "cfd"],
        },
        index=["w", "g", "b"],
    )
    out = egl.unit_parameters(units, 0.45, 75.0)
    assert out["levy_rate"].tolist() == [0.45, 0.0, 0.0] and out[
        "levy_benchmark"
    ].tolist() == [75.0, 0.0, 0.0]


def test_observed_prices_nuclear_and_company_shares(tmp_path):
    units = pd.DataFrame(
        {
            "technology": ["Solid Biomass", "Wind"],
            "support_scheme": ["premium", "premium"],
        },
        index=["biomass_class", "wind_a"],
    )
    write_run(
        tmp_path / "gb_test_day_ahead", units, {"biomass_class": 100.0, "wind_a": 100.0}
    )
    index = pd.date_range("2023-01-01", periods=HALF_HOURS, freq="30min")
    observed = pd.Series(120.0, index=index)
    nuclear = pd.Series(1000.0, index=index)
    # the biomass class holds a named company's columns for a third of its capacity
    shares = {"biomass_class": {"Drax": 1 / 3}}
    table = egl.settle(
        tmp_path / "gb_test_day_ahead",
        units,
        2023,
        None,
        benchmark=75.0,
        observed_prices=observed,
        nuclear=nuclear,
        shares=shares,
    )
    assert (
        table.loc["wind_a", "realised_price"] == 120.0
    )  # the observed price, not the run's 100
    assert (
        table.loc[egl.NUCLEAR_GROUP, "generation_gwh"] == 1000 * HALF_HOURS * 0.5 / 1e3
    )
    assert table.loc["Drax", "generation_gwh"] == pytest.approx(876 / 3)
    assert table.loc["biomass_class", "generation_gwh"] == pytest.approx(876 * 2 / 3)
    assert table.attrs["prices"] == "observed" and table.attrs["nuclear"] is True


def test_implied_realised_price_inverts_the_levy(tmp_path, monkeypatch):
    units = pd.DataFrame(
        {"technology": ["Solid Biomass"], "support_scheme": ["premium"]},
        index=["drax_unit"],
    )
    write_run(tmp_path / "gb_test_day_ahead", units, {"drax_unit": 100.0})
    owner = pd.Series({"drax_unit": "Drax"})
    table = egl.settle(
        tmp_path / "gb_test_day_ahead", units, 2023, owner, benchmark=75.0
    )
    # 876 GWh at 100: excess 21.9 m, levy 0.45 x 11.9 m; a disclosure of exactly that levy implies
    # the price the settlement used, a larger one a higher realised price
    levy = table.loc["Drax", "levy_gbp_m"]
    companies = pd.DataFrame(
        {
            "company": ["Drax", "EDF"],
            "calendar_year": [2023, 2023],
            "value_gbp_m": [levy, 200.0],
        }
    )
    monkeypatch.setattr(egl, "outturn", lambda: {"companies": companies})
    implied = egl.implied_prices(table)
    assert implied.index.tolist() == ["Drax"]  # EDF is not in this settlement
    assert implied.loc["Drax", "implied_realised_gbp_mwh"] == pytest.approx(100.0)
    assert implied.loc["Drax", "settled_at_gbp_mwh"] == pytest.approx(100.0)
    companies.loc[0, "value_gbp_m"] = 2 * levy
    assert egl.implied_prices(table).loc[
        "Drax", "implied_realised_gbp_mwh"
    ] == pytest.approx(75 + (2 * levy * 1e6 / 0.45 + 10e6) / 876e3)
