# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Ownership: the rules that turn register names into groups, dated sales and joint ventures into owners and an
operator, merged units into shares, and the battery fleet into one unit per trader; then the 2023 scenario
against what the registers say (skipped without the GB data repository)."""

import numpy as np
import pandas as pd
import pytest

from assume_gb import convert, ownership, paths

# The company rules (names, groups, dated owners and operators) moved to gb-power-data on 6 Oct 2026:
# their tests run where it is importable and are skipped elsewhere (ownership.md).
needs_gb_power_data = pytest.mark.skipif(
    not ownership.available(),
    reason="needs gb-power-data, where the company rules live",
)


@needs_gb_power_data
def test_names_reduce_to_a_matching_key_and_a_readable_label():
    co = ownership.gpd().co
    assert co.norm("Keadby Generation Limited") == co.norm("KEADBY GENERATION LTD")
    assert co.norm("Coryton Energy Company, Ltd.") == "CORYTON ENERGY"
    assert co.norm(None) == "" and co.norm(float("nan")) == ""
    assert co.clean("Seabank Power Limited") == "Seabank Power"


@needs_gb_power_data
def test_group_order_alias_then_name_then_cm_parent_then_itself():
    # an alias wins over the name rules: Keadby Generation names no group, the alias says SSE
    assert ownership.group_of("Keadby Generation Limited") == ("SSE", "alias")
    # a group's name in the company's name
    assert ownership.group_of("RWE Renewables UK Robin Rigg") == ("RWE", "name")
    assert ownership.group_of("Npower Commercial Gas Limited") == (
        "E.ON",
        "name",
    )  # npower's supply business
    assert ownership.group_of("npower renewables limited") == (
        "RWE",
        "name",
    )  # its renewables went with RWE
    # the parent in the CM register, resolved the same way
    assert ownership.group_of("Foo Storage Limited", {"FOO STORAGE": "SSE plc"}) == (
        "SSE",
        "cm_parent",
    )
    # Elexon cuts names at 30 characters: the start of exactly one alias takes it
    assert ownership.group_of("Triton Knoll Offshore Wind")[0] == "RWE"
    # otherwise the company is a group of its own, under its readable name
    assert ownership.group_of("Lakeside Energy Storage Ltd") == (
        "Lakeside Energy Storage",
        "self",
    )
    assert ownership.group_of("") == ("", "")


@needs_gb_power_data
def test_a_dated_sale_changes_owner_and_operator():
    before, _, basis = ownership.resolve(
        "EDF Energy", pd.Timestamp("2020-12-31"), "West_Burton"
    )
    after, operator, _ = ownership.resolve(
        "UK Transition Limited", pd.Timestamp("2023-12-31"), "West_Burton"
    )
    assert basis == "curated"
    assert before == [{"group": "EDF", "share": 1.0}]
    assert after == [{"group": "EIG", "share": 1.0}] and operator == "EIG"


@needs_gb_power_data
def test_a_joint_venture_has_owners_with_shares_and_an_operating_company():
    owners, operator, _ = ownership.resolve(
        "Energy Capital Partners", pd.Timestamp("2023-12-31"), "Saltend"
    )
    assert sorted((o["group"], o["share"]) for o in owners) == [
        ("Equinor", 0.5),
        ("SSE", 0.5),
    ]
    assert operator == "Triton Power"  # the operating company is not an owner
    owners, operator, _ = ownership.resolve(
        "Energy Capital Partners", pd.Timestamp("2021-12-31"), "Saltend"
    )
    assert [o["group"] for o in owners] == [
        "Energy Capital Partners"
    ] and operator == "Triton Power"


@needs_gb_power_data
def test_shares_no_source_gives_stay_unknown_and_known_shares_leave_the_rest_unresolved():
    # Fallago Rig: Hermes GPE holds a majority and EDF the rest, but no source gives the shares
    owners, operator, _ = ownership.resolve(
        "Fallago Rig Windfarm Limited", pd.Timestamp("2023-12-31")
    )
    assert {o["group"] for o in owners} == {"EDF", "Hermes GPE"} and all(
        np.isnan(o["share"]) for o in owners
    )
    assert operator == "EDF"
    # Lincs before May 2022: only Orsted's quarter is known
    owners, operator, _ = ownership.resolve(
        "Lincs Wind Farm Ltd", pd.Timestamp("2021-12-31")
    )
    assert owners == [
        {"group": "Orsted", "share": 0.25},
        {"group": ownership.UNRESOLVED, "share": 0.75},
    ]
    assert operator == "Orsted"


@needs_gb_power_data
def test_shares_of_a_verified_joint_venture_add_up_and_change_with_its_sales():
    def shares(company, day):
        owners, operator, _ = ownership.resolve(company, pd.Timestamp(day))
        return {o["group"]: round(o["share"], 3) for o in owners}, operator

    assert shares("Seabank Power Limited", "2023-12-31") == (
        {"SSE": 0.5, "CKI": 0.25, "Power Assets": 0.25},
        "SSE",
    )
    # London Array: E.ON's 30% went to RWE with the asset swap of 30 Sep 2019, Orsted's 25% to Schroders Greencoat in 2023
    assert shares("London Array Ltd", "2018-12-31") == (
        {"E.ON": 0.3, "Orsted": 0.25, "CDPQ": 0.25, "Masdar": 0.2},
        "E.ON",
    )
    assert shares("London Array Ltd", "2024-12-31") == (
        {"RWE": 0.3, "Schroders Greencoat": 0.25, "CDPQ": 0.25, "Masdar": 0.2}, "RWE")  # fmt: skip
    for company in (
        "Walney (UK) Offshore Windfarms Ltd.",
        "Rampion Offshore Wind Ltd",
        "Gwynt y Mor Offshore wind farm limited",
    ):
        for day in ("2018-12-31", "2019-12-31", "2021-12-31", "2025-12-31"):
            assert sum(shares(company, day)[0].values()) == pytest.approx(1.0), (
                company,
                day,
            )


@needs_gb_power_data
def test_a_company_without_a_record_owns_and_runs_its_unit():
    owners, operator, basis = ownership.resolve(
        "Uniper UK Limited", pd.Timestamp("2023-12-31")
    )
    assert (
        owners == [{"group": "Uniper", "share": 1.0}]
        and operator == "Uniper"
        and basis == "alias"
    )


def test_a_merged_unit_takes_its_stack_units_shares_by_capacity():
    roles = pd.DataFrame(
        [
            {"unit": "a", "role": "owner", "group": "X", "share": 1.0},
            {"unit": "b", "role": "owner", "group": "Y", "share": 0.5},
            {"unit": "b", "role": "owner", "group": "Z", "share": 0.5},
            {"unit": "a", "role": "operator", "group": "X", "share": 1.0},
            {"unit": "b", "role": "operator", "group": "Y", "share": 1.0},
            {
                "unit": "c",
                "role": "owner",
                "group": "P",
                "share": np.nan,
            },  # shares unknown: split equally
            {"unit": "c", "role": "owner", "group": "Q", "share": np.nan},
        ]
    )
    columns = pd.DataFrame(
        {
            "stack_unit_ids": ["a;b;agg_x", "c"],
            "column_capacity_mw": ["300;100;100", "50"],
        },
        index=["merged", "jv"],
    )
    view = ownership.scenario_view(roles, columns).set_index(["unit", "role", "group"])[
        "share"
    ]
    assert view[("merged", "owner", "X")] == pytest.approx(0.6)
    assert view[("merged", "owner", "Y")] == pytest.approx(0.1) and view[
        ("merged", "owner", "Z")
    ] == pytest.approx(0.1)
    assert view[("merged", "owner", ownership.UNRESOLVED)] == pytest.approx(
        0.2
    )  # the aggregate column
    assert view[("merged", "operator", "X")] == pytest.approx(0.6) and view[
        ("merged", "operator", "Y")
    ] == pytest.approx(0.2)
    assert view[("jv", "owner", "P")] == pytest.approx(0.5) and view[
        ("jv", "owner", "Q")
    ] == pytest.approx(0.5)
    assert view[("jv", "trader", ownership.UNRESOLVED)] == pytest.approx(1.0)


def test_the_scenario_carries_its_single_operators(tmp_path):
    assert ownership.operators_from_view(tmp_path) is None  # no unit_owners.csv
    pd.DataFrame(
        [
            {"unit": "Pembroke", "role": "operator", "group": "RWE", "share": 1.0},
            {"unit": "Pembroke", "role": "owner", "group": "RWE", "share": 1.0},
            {
                "unit": "merged",
                "role": "operator",
                "group": "Orsted",
                "share": 0.7,
            },  # a class of several: no single one
            {"unit": "merged", "role": "operator", "group": "Equinor", "share": 0.3},
            {
                "unit": "ro_wind_001",
                "role": "operator",
                "group": ownership.UNRESOLVED,
                "share": 1.0,
            },
        ]
    ).to_csv(tmp_path / "unit_owners.csv", index=False)
    assert ownership.operators_from_view(tmp_path).to_dict() == {"Pembroke": "RWE"}


def test_the_battery_fleet_splits_into_one_unit_per_trader(tmp_path):
    fleet = {
        "name": convert.STORAGE_UNIT, "technology": "battery", "bidding_DA": convert.STORAGE_STRATEGY,
        "max_power_charge": 1000.0, "max_power_discharge": 1000.0, "capacity": 2000.0, "efficiency_charge": 1.0,
        "efficiency_discharge": 0.85, "initial_soc": 0.0, "min_soc": 0.0, "max_soc": 1.0,
        "additional_cost_charge": 10.0, "additional_cost_discharge": 0.0, "unit_operator": convert.STORAGE_UNIT,
    }  # fmt: skip
    pd.DataFrame([fleet]).set_index("name").to_csv(tmp_path / "storage_units.csv")
    assert convert.write_storage_traders(
        tmp_path, pd.Series({"Zenobe": 3.0, "Tesla": 1.0, "nobody": 0.0})
    )
    units = pd.read_csv(tmp_path / convert.STORAGE_TRADERS_FILE).set_index("name")
    assert list(units.index) == ["battery_Zenobe", "battery_Tesla"]
    assert units["max_power_discharge"].sum() == pytest.approx(1000.0) and units[
        "capacity"
    ].sum() == pytest.approx(2000.0)
    assert units.loc["battery_Zenobe", "max_power_charge"] == pytest.approx(750.0)
    assert list(units["unit_operator"]) == ["Zenobe", "Tesla"]
    assert (units["efficiency_discharge"] == 0.85).all()
    assert not convert.write_storage_traders(tmp_path, pd.Series(dtype=float))


def _gb_files_present() -> bool:
    try:
        root = paths.gb_repo()
    except FileNotFoundError:
        return False
    needed = [
        root / "data" / "processed" / "bm_unit_register_2023.csv",
        root / "data" / "results" / "comb_stack_agents_2023_status_quo_register.csv",
        root / ownership.DUKES_FILE,
    ]
    return all(p.exists() for p in needed) and any(
        (root / "data" / "raw" / "neso" / "cm_register").glob("cmu_*.csv")
    )


@pytest.mark.skipif(
    not _gb_files_present(),
    reason="needs the GB data repository's registers and the 2023 stack",
)
def test_2023_owners_operators_and_traders_from_the_registers():
    from assume_gb.stack import stack_files

    agents = pd.read_csv(stack_files(2023, "status_quo", "register")["agents"])
    roles = ownership.unit_roles(2023, agents)
    operator = (
        roles[roles["role"] == "operator"]
        .drop_duplicates("unit")
        .set_index("unit")["group"]
    )
    trader = (
        roles[roles["role"] == "trader"]
        .drop_duplicates("unit")
        .set_index("unit")["group"]
    )
    expected = {
        "Pembroke": "RWE", "Keadby": "SSE", "Grain": "Uniper", "Immingham": "VPI", "South_Humber_Bank": "EPUKi",
        "Spalding": "InterGen", "Saltend": "Triton Power", "West_Burton": "EIG", "Seabank": "SSE",
        "FARM_Greater_Gabbard_Offshore_Windfarm_b2": "SSE", "FARM_London_Array_Windfarm_b2": "RWE",
        "INV-HOR-001": "Orsted", "AR2-TKN-303": "RWE",
    }  # fmt: skip
    assert {u: operator.get(u) for u in expected} == expected
    # traders from today's BM register: Keadby Generation is SSE's (CM register), a joint venture's own company
    # trades for its operator, and Severn is traded under Centrica since its sale
    assert (
        trader["Keadby"] == "SSE"
        and trader["FARM_London_Array_Windfarm_b2"] == "RWE"
        and trader["Severn"] == "Centrica"
    )
    # every named farm has an operator; the owners' shares of a unit add up to one where they are known
    farms = agents.loc[agents["unit_id"].str.startswith("FARM_"), "unit_id"]
    assert set(farms) <= set(operator.index)
    shares = (
        roles[(roles["role"] == "owner") & roles["share"].notna()]
        .groupby("unit")["share"]
        .sum()
    )
    assert np.allclose(
        shares[shares.index.isin(roles.loc[roles["share"].isna(), "unit"]) == False],
        1.0,
    )  # noqa: E712
    battery = ownership.battery_roles(2023)
    for role in ("trader", "owner"):
        assert battery.loc[battery["role"] == role, "share"].sum() == pytest.approx(1.0)
