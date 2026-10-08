# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Who owns, runs and trades each unit of the GB scenarios, year by year.

Why (6 Oct 2026, the author): the portfolio agents of the learning cases and the generator levy need each unit's
owner, and the work on market structure needs to keep apart who owns a unit and who trades it. This module answers
that for the stack's named units and the battery fleet, and writes a table beside every scenario
(``unit_owners.csv``) that the framework does not read. ``ownership.md`` lays out the approach.

The registers, the curated tables and the rules live in gb-power-data, the public GB data package
(``paths.gb_power_data()``: the sibling folder ``gb-power-data`` or ``GB_POWER_DATA_REPO``), since the evening of
6 Oct 2026; this module read its own copies before.

- ``gbpowerdata.common.companies``: a company's group (``data/reference/company_groups.csv``, the group's name in
  the company's, the registered address, the parent in the Capacity Market register) and the dated owners with
  shares and the operator of joint ventures, sales and stations (``data/reference/company_owners.csv``, every row
  with its sources, listed in ``docs/ownership_references.md``).
- ``gbpowerdata.build.build_ownership``: the registers (DESNZ's DUKES 5.11 by edition, Ofgem's RO register, LCCC's
  CfD register, NESO's Capacity Market register, Elexon's BM units by year) with their links to stations, farms
  and BM units, and the built tables ``data/processed/ownership_{year}.csv`` (one row per register record, role
  and group) and ``ownership_bmu_{year}.csv`` (one row per BM unit active in the year, role and group).

What stays here belongs to the model: which station, farm or contract a stack unit is
(``ownership_data/curated/plants.csv`` names the station of every named plant in gb-power-data's
``dukes_bmu_map.csv``; a farm unit is its station in the farm map; a CfD unit is its contract), the split of
merged scenario units, and the battery fleet as one unit. gb-power-data is imported on first use, so the tool
runs without it (the HPC reads each scenario's ``unit_owners.csv``); its ``config.local.yaml``, which points its
raw folder at the data, is used where present and ``GBPOWERDATA_CONFIG`` is not set.

Three roles per unit and year, at the end of the year (the current year: today):

- ``owner``: who holds the equity, with shares where a source gives them (a joint venture has several owners; a
  share no source gives is left blank, and the part the known shares do not cover is ``unresolved``).
- ``operator``: the company that runs the unit for its owners. A single owner is its own operator; a joint
  venture's is the one ``company_owners.csv`` marks. The generator levy groups units by it (``owners.owners``).
  For the battery fleet the operator is the trader (the decision agreed with the author on 6 Oct 2026).
- ``trader``: the BM lead party of the unit's BM units in the year, from gb-power-data's BM-unit table: a
  documented lead party (``bmu_lead_parties.csv``), else the register snapshot of the year, else today's register
  as a proxy, replaced by the unit's operator in the year where a dated source shows that the unit changed hands
  since (Damhead Creek and Rye House were Drax's until January 2021; VPI trades them today). A joint venture's own company
  trades for the venture's operator; a project company that trades its own farm is placed in its parent's group
  by Companies House (Kilbraur Wind Energy in Renantis).

The portfolio agents of the learning cases are the traders (``bidders``; the author, 7 Oct 2026: the model
evaluates market power, which is exercised by whoever bids a unit, not by whoever owns it). The lead party submits
the unit's notifications and trades its output, and under an offtake contract it also bears the price: Eneco
trades Fred Olsen's Crystal Rig, Statkraft RWE's An Suidhe. A unit whose BM units were not active in the year,
whose trader is not resolved, or whose lead party resolves to itself (taken as the unit's own project company) is
bid by its operator (``bidders_from_roles``). ``--agent-level operator`` keeps the operators of before.

A plant's company is DESNZ's for its station at the end of the year (the edition before for a plant that closed in
the year), a farm's the holder of its RO stations, a CfD unit's the counterparty; each is resolved at the end of the
year, the station's dated rows first, then the company's. Where that company has no dated rows but gb-power-data
gives the unit's BM units the dated rows of the joint venture that trades them, those give the owners (Gunfleet
Sands, whose RO holder Orsted Burbo (UK) also holds Burbo Bank).

Output: ``ownership_data/built/unit_roles_{year}.csv`` (one row per model unit, role and group) and
``groups_{year}.csv`` (each group's owned, operated and traded MW); ``<scenario folder>/unit_owners.csv``, the
same per scenario unit, with merged units split by the capacity of their stack columns.

    python -m assume_gb ownership --years 2018 2025      (``build`` writes the scenario's table as well)

after, in gb-power-data, ``python -m gbpowerdata.build.bm_units`` and ``python -m gbpowerdata.build.build_ownership``
for the same years.
"""

import importlib.util
import os
import re
import sys
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from assume_gb import paths

DATA = paths.TOOL_ROOT / "ownership_data"
CURATED = DATA / "curated"
BUILT = DATA / "built"
BATTERY_UNIT = "battery"  # convert.STORAGE_UNIT, not imported to keep this module free of the converter
UNRESOLVED = "unresolved"  # gbpowerdata.common.companies.UNRESOLVED
ROLES = ("owner", "operator", "trader")
AGENT_LEVELS = ("trader", "operator")  # who the portfolio agents are: the default first
# the units a portfolio holds: named plants and wind farms; with the CfD units in the variant that puts them there
PORTFOLIO_KINDS = ("plant", "farm")
PORTFOLIO_KINDS_CFD = ("plant", "farm", "cfd")
CONFIG_ENV = "GBPOWERDATA_CONFIG"
LOCAL_CONFIG = "config.local.yaml"
# how gb-power-data found a BM unit's trader in the year (its "register" column)
TRADER_SOURCES = {
    "bm": "Elexon BM register (the year's lead party)",
    "bm_proxy": "Elexon BM register (today's lead party, a proxy for the year)",
    "operator_at_date": "the unit's operator in the year (it changed hands since the register's date)",
    "curated": "documented lead party (gb-power-data bmu_lead_parties.csv)",
}


# ---------------------------------------------------------------------------- gb-power-data
@cache
def gpd() -> SimpleNamespace:
    """gb-power-data's modules: ``co`` (gbpowerdata.common.companies), ``build`` (gbpowerdata.build.build_ownership),
    ``processed`` (gbpowerdata.processed) and ``config`` (gbpowerdata.config). An installed gbpowerdata, else the
    one in ``paths.gb_power_data()``; FileNotFoundError where there is neither (as on the HPC)."""
    spec = importlib.util.find_spec("gbpowerdata")
    if spec is None:
        root = paths.gb_power_data()
        sys.path.insert(0, str(root / "src"))
    else:
        root = (
            Path(spec.origin).resolve().parents[2]
        )  # src/gbpowerdata/__init__.py -> the repository
    # gbpowerdata reads its config path once, on import
    if CONFIG_ENV not in os.environ and (root / LOCAL_CONFIG).exists():
        os.environ[CONFIG_ENV] = LOCAL_CONFIG
    from gbpowerdata import config, processed
    from gbpowerdata.build import build_ownership
    from gbpowerdata.common import companies

    return SimpleNamespace(
        co=companies,
        build=build_ownership,
        processed=processed,
        config=config,
        root=root,
    )


def available() -> bool:
    """Whether gb-power-data can be imported here."""
    try:
        gpd()
    except FileNotFoundError:
        return False
    return True


def group_of(name, cm_parents: dict | None = None) -> tuple[str, str]:
    """(group, basis) of a company name (``gbpowerdata.common.companies.group_of``)."""
    return gpd().co.group_of(name, cm_parents)


def resolve(
    company,
    when: pd.Timestamp,
    unit: str = "",
    cm_parents: dict | None = None,
    address=None,
) -> tuple[list[dict], str, str]:
    """The owners (group, share), the operator and the basis of a unit held by ``company`` at ``when``
    (``gbpowerdata.common.companies.resolve``); a named plant (``unit``) takes its station's dated rows first."""
    station = plants()["station"].get(unit, "") if unit else ""
    return gpd().co.resolve(
        company, when, station=station, cm_parents=cm_parents, address=address
    )


# ---------------------------------------------------------------------------- the model's units
@cache
def plants() -> pd.DataFrame:
    """``curated/plants.csv``: the model's named plants and their stations in gb-power-data's ``dukes_bmu_map.csv``."""
    return pd.read_csv(
        CURATED / "plants.csv", dtype=str, keep_default_na=False
    ).set_index("model_unit")


def farm_label(unit: str, label: str | None = None) -> str:
    """A named farm's station in the farm map: the stack's name of the unit without its note ("London Array
    Windfarm (RO farm, observed FPN)"), or, without a name, its id without prefix, band and tags."""
    if label:
        return re.sub(r"\(.*?\)", "", str(label)).strip()
    text = str(unit).replace("FARM_", "", 1)
    text = re.sub(r"_RO_[0-9.]+_[A-Z]", "", text)
    text = re.sub(r"_b[0-9.]+$", "", text)
    return text.replace("_", " ").strip()


def _farm_key(station: str) -> str:
    # the ids lose the commas of a station of several farms ("Clyde Central Windfarm, Clyde North Windfarm, ...")
    return re.sub(r"\s+", " ", re.sub(r"[,_]", " ", str(station))).strip().lower()


@cache
def farm_links() -> dict[str, dict]:
    """gb-power-data's farm links (``build_ownership.farm_ro_links``) by station: bmu_ids, ro_ids, holder (the RO
    accreditation holder, "" where the farm matches no RO station) and address."""
    return {
        _farm_key(r["station"]): r
        for r in gpd().build.farm_ro_links().to_dict("records")
    }


def unit_kind(unit: str, regime) -> str:
    if unit.startswith("FARM_"):
        return "farm"
    if unit in plants().index:
        return "plant"
    if str(regime) == "CfD":
        return "cfd"
    if str(regime) == "interconnector":
        return "interconnector"
    return "class"


@cache
def bmu_traders(year: int) -> pd.DataFrame:
    """The trader of every BM unit active in ``year`` (gb-power-data's ``ownership_bmu_{year}.csv``)."""
    try:
        return gpd().processed.read_ownership_bmu(year, role="trader")
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{error}: run `python -m gbpowerdata.build.bm_units` and `python -m gbpowerdata.build.build_ownership` in gb-power-data"
        ) from error


@cache
def bmu_owner_rows(year: int) -> pd.DataFrame:
    """The owner and operator rows of every BM unit active in ``year`` (gb-power-data's ``ownership_bmu_{year}.csv``)."""
    table = gpd().processed.read_ownership_bmu(year)
    return table[table["role"].isin(["owner", "operator"])]


def _venture(bmus: list[str], year: int) -> tuple[list[dict], str, str] | None:
    """(owners, operator, company) where gb-power-data gives a unit's BM units the dated rows of the joint venture
    that trades them (register "bm", basis "curated": Gunfleet Sands, whose RO holder holds other farms too), from
    the largest such unit; None otherwise."""
    rows = bmu_owner_rows(year)
    rows = rows[
        rows["unit_id"].isin(bmus)
        & (rows["register"] == "bm")
        & (rows["basis"] == "curated")
    ]
    if rows.empty:
        return None
    rows = rows[
        rows["unit_id"]
        == rows.sort_values("capacity_mw", ascending=False)["unit_id"].iloc[0]
    ]
    owners = rows[rows["role"] == "owner"]
    operator = rows.loc[rows["role"] == "operator", "group"]
    return ([{"group": g, "share": s} for g, s in zip(owners["group"], owners["share"])],
            operator.iloc[0] if len(operator) else "", str(rows["company"].iloc[0]))  # fmt: skip


def _trader(bmus: list[str], year: int) -> dict | None:
    """The trader of a unit's BM units in ``year``: the group that trades most of their capacity, with its lead
    party, basis and source; None where none of them was active."""
    if not bmus:
        return None
    rows = bmu_traders(year)
    rows = rows[rows["unit_id"].isin(bmus)]
    if rows.empty:
        return None
    group = rows.groupby("group")["capacity_mw"].sum().idxmax()
    first = (
        rows[rows["group"] == group].sort_values("capacity_mw", ascending=False).iloc[0]
    )
    return {"group": group, "company": first["company"] if pd.notna(first["company"]) else "", "basis": first["basis"],
            "source": TRADER_SOURCES.get(first["register"], str(first["register"]))}  # fmt: skip


def unit_roles(year: int, agents: pd.DataFrame) -> pd.DataFrame:
    """One row per stack unit, role and group for ``year``: owners with shares, the operator, the trader.
    ``agents`` is the stack's table (``OfferStack.agents``: unit_id, name, support_regime, capacity_mw)."""
    g = gpd()
    when = g.build.as_of(year)
    cm = g.build.cm_parents()
    cfd = g.build.cfd_register().set_index("record_id")
    cfd_bmus = g.build.cfd_bmus(year)
    stations = g.co.dukes_bmu_map()
    links = farm_links()
    rows, unmatched = [], []
    for unit, label, regime, capacity in zip(
        agents["unit_id"].astype(str),
        agents["name"],
        agents["support_regime"],
        agents["capacity_mw"],
    ):
        kind = unit_kind(unit, regime)
        if kind in ("class", "interconnector"):
            continue
        company, address, station, where, bmus = "", None, "", "", []
        if kind == "plant":
            station = plants().loc[unit, "station"]
            company, record = g.build.dukes_company(station, year)
            where, bmus = (
                (f"DESNZ DUKES 5.11: {record}" if record else ""),
                list(stations.loc[station, "bmus"]),
            )
        elif kind == "farm":
            link = links.get(
                _farm_key(farm_label(unit, label if pd.notna(label) else None))
            )
            if link is None:
                unmatched.append(unit)
            else:
                company, address, bmus = (
                    link["holder"],
                    link["address"],
                    list(link["bmu_ids"]),
                )
                where = (
                    f"Ofgem RO register: {';'.join(link['ro_ids'])}" if company else ""
                )
        elif kind == "cfd" and unit in cfd.index:
            company, address, where = (
                cfd.loc[unit, "company"],
                cfd.loc[unit, "address"],
                "LCCC CfD register",
            )
            bmus = cfd_bmus.get(unit, [])
        owners, operator, basis = (
            g.co.resolve(company, when, station=station, cm_parents=cm, address=address)
            if (company or station)
            else ([], "", "")
        )
        if basis not in ("curated", "") and bmus and (venture := _venture(bmus, year)):
            # the register names a company without dated rows; the venture that trades the unit has them
            owners, operator, venture_company = venture
            basis, where = (
                "curated",
                f"gb-power-data BM-unit table: dated rows of {venture_company}",
            )
        trader = _trader(bmus, year)
        common = {"year": year, "unit": unit, "kind": kind, "capacity_mw": float(capacity), "company": g.co.clean(company) if company else "",
                  "source": where, "bmu_ids": ";".join(bmus)}  # fmt: skip
        for o in owners:
            rows.append(
                {
                    **common,
                    "role": "owner",
                    "group": o["group"],
                    "share": o["share"],
                    "basis": basis,
                }
            )
        if operator:
            rows.append(
                {
                    **common,
                    "role": "operator",
                    "group": operator,
                    "share": 1.0,
                    "basis": basis,
                }
            )
        if trader:
            rows.append({**common, "role": "trader", "group": trader["group"], "share": 1.0, "basis": trader["basis"],
                         "company": g.co.clean(trader["company"]) if trader["company"] else "", "source": trader["source"]})  # fmt: skip
    if unmatched:
        print(
            f"ownership {year}: farm units without a station in the farm map: {', '.join(unmatched)}"
        )
    roles = pd.DataFrame(
        rows,
        columns=[
            "year",
            "unit",
            "kind",
            "role",
            "group",
            "share",
            "capacity_mw",
            "company",
            "basis",
            "source",
            "bmu_ids",
        ],
    )
    # each group in gb-power-data's one spelling (its tables give the traders), so an operator resolved here and a
    # trader read from them name one company alike ("Renewable And Grid Services", "Renewable and Grid Services")
    return g.build.one_spelling(roles)


def battery_roles(year: int) -> pd.DataFrame:
    """The battery fleet split by trader (gb-power-data's BM-unit table: the battery units active in ``year`` by
    their trader's group) and by owner (its register table: the battery units of NESO's Capacity Market register
    with an agreement in force in the delivery year from October ``year``, at their connection capacity, by owner
    group; a unit whose owners' shares no source gives is split equally). The operator is the trader."""
    g = gpd()
    traders = bmu_traders(year)
    traders = traders[traders["tech_class"] == "battery"]
    traded = traders.groupby("group")["capacity_mw"].sum()
    owners = g.processed.read_ownership(year, register="cm", role="owner")
    owners = owners[owners["tech_class"] == "battery"]
    share = owners["share"].astype(float)
    unknown = share.isna().groupby(owners["record_id"]).transform("all")
    count = owners.groupby("record_id")["group"].transform("size")
    share = share.where(~unknown, 1.0 / count).fillna(0.0)
    owned = (
        (pd.to_numeric(owners["capacity_mw"], errors="coerce") * share)
        .groupby(owners["group"])
        .sum()
    )
    rows = []
    for role, series, source in (
        (
            "trader",
            traded,
            "gb-power-data BM-unit table (traders of the active battery units)",
        ),
        ("operator", traded, "the trader (agreed default for batteries)"),
        (
            "owner",
            owned,
            f"NESO CM register via gb-power-data (owners of the battery units with an agreement in force in delivery year {year}/{year + 1})",
        ),
    ):
        total = float(series.sum())
        for group, mw in series.sort_values(ascending=False).items():
            if mw > 0 and total > 0:
                rows.append({"year": year, "unit": BATTERY_UNIT, "kind": "battery", "role": role, "group": group, "share": mw / total,
                             "capacity_mw": float(mw), "company": "", "basis": "register", "source": source, "bmu_ids": ""})  # fmt: skip
    return pd.DataFrame(rows)


def battery_traders(year: int) -> pd.Series:
    """The share of the battery fleet each company trades in ``year`` (gb-power-data's active battery units by
    trader's group), for ``convert.build_scenario(battery_traders=...)``."""
    trader = battery_roles(year)
    trader = trader[trader["role"] == "trader"]
    return trader.set_index("group")["share"]


def _stack_units(
    unit_names: list[str], unit_labels: dict[str, str] | None
) -> pd.DataFrame:
    """Units given by name as stack units (``unit_roles``'s ``agents``), with names from ``unit_labels``."""
    return pd.DataFrame(
        {
            "unit_id": unit_names,
            "name": [(unit_labels or {}).get(u) for u in unit_names],
            "support_regime": "",
            "capacity_mw": 0.0,
        }
    )


def operators(year: int, unit_names: list[str], unit_labels: dict[str, str] | None = None, agents: pd.DataFrame | None = None,
              kinds: tuple[str, ...] = PORTFOLIO_KINDS) -> pd.Series:  # fmt: skip
    """The operator group of every named unit of ``kinds`` (the company that runs it for its owners), by unit name.
    Without ``agents``, the units are taken as stack units with names from ``unit_labels``. Units without an
    operator are left out."""
    roles = unit_roles(
        year, _stack_units(unit_names, unit_labels) if agents is None else agents
    )
    op = roles[
        (roles["role"] == "operator")
        & roles["kind"].isin(kinds)
        & roles["unit"].isin(unit_names)
    ]
    return op.drop_duplicates("unit").set_index("unit")["group"].rename("owner")


def bidders_from_roles(
    roles: pd.DataFrame, unit_names: list[str], kinds: tuple[str, ...] = PORTFOLIO_KINDS
) -> pd.Series:
    """The bidding agent of every named unit of ``kinds`` in ``roles`` (``unit_roles``), by unit name: its trader,
    else its operator. The operator bids where none of the unit's BM units was active in the year, where its trader
    is not resolved, and where the lead party resolves to itself (basis "self": no rule, source or Companies House
    parent places it), which is then taken as the unit's own project company (Kilbraur and Millennium Wind Energy,
    Renantis' farms, whose Companies House entries name only a lender from December 2022). Units with neither are
    left out.

    CfD units are not of the default kinds: their revenue does not move with the price, so they add nothing to a
    portfolio's reason to raise it, and they bid their contract as in the day-ahead case. ``PORTFOLIO_KINDS_CFD``
    puts them in their companies' portfolios (the cases ``learning_cfd*``, ``convert.write_cfd_variant``)."""
    named = roles[
        roles["kind"].isin(kinds)
        & roles["unit"].isin(unit_names)
        & (roles["group"] != UNRESOLVED)
    ]
    named = named[~((named["role"] == "trader") & (named["basis"] == "self"))]

    def first(role: str) -> pd.Series:
        return (
            named[named["role"] == role]
            .drop_duplicates("unit")
            .set_index("unit")["group"]
        )

    return first("trader").combine_first(first("operator")).rename("owner")


def bidders(year: int, unit_names: list[str], unit_labels: dict[str, str] | None = None, agents: pd.DataFrame | None = None,
            kinds: tuple[str, ...] = PORTFOLIO_KINDS) -> pd.Series:  # fmt: skip
    """The bidding agent of every named unit of ``kinds`` in ``year`` (the portfolio agent of the learning cases), by
    unit name: the group that trades its BM units, else its operator (``bidders_from_roles``). ``unit_labels`` and
    ``agents`` as in ``operators``."""
    roles = unit_roles(
        year, _stack_units(unit_names, unit_labels) if agents is None else agents
    )
    return bidders_from_roles(roles, unit_names, kinds)


def operators_from_view(scenario_dir) -> pd.Series | None:
    """The operator of every scenario unit that has exactly one (``unit_owners.csv``: role operator, share one,
    not ``unresolved``), by unit name; None without the file. The scenario carries its operators this way, so the
    levy groups units by operator where gb-power-data is not reachable (the HPC): a joint venture's whole levy goes
    to its operator, not to its owners by their shares."""
    path = Path(scenario_dir) / "unit_owners.csv"
    if not path.exists():
        return None
    view = pd.read_csv(path)
    single = view[
        (view["role"] == "operator")
        & (view["group"] != UNRESOLVED)
        & (view["share"] >= 1.0 - 1e-6)
    ]
    return single.drop_duplicates("unit").set_index("unit")["group"].rename("owner")


def scenario_view(roles: pd.DataFrame, columns: pd.DataFrame) -> pd.DataFrame:
    """The roles per scenario unit: ``columns`` is the scenario's ``unit_columns.csv`` (index: unit name;
    ``stack_unit_ids`` and ``column_capacity_mw``, ";"-joined). A merged unit's shares are its stack units' shares
    weighted by their capacity; a unit or part of one with no owner of record is ``unresolved``."""
    rows = []
    by_unit = {u: g for u, g in roles.groupby("unit")}
    for name, row in columns.iterrows():
        ids = str(row["stack_unit_ids"]).split(";")
        caps = np.array([float(c) for c in str(row["column_capacity_mw"]).split(";")])
        weights = (
            caps / caps.sum() if caps.sum() > 0 else np.full(len(ids), 1.0 / len(ids))
        )
        for role in ROLES:
            share: dict[str, float] = {}
            for member, w in zip(ids, weights):
                part = by_unit.get(member)
                part = part[part["role"] == role] if part is not None else None
                if part is None or part.empty:
                    share[UNRESOLVED] = share.get(UNRESOLVED, 0.0) + w
                    continue
                s = part["share"].astype(float)
                s = s.fillna(1.0 / len(s)) if s.isna().all() else s.fillna(0.0)
                for group, value in zip(part["group"], s):
                    share[group] = share.get(group, 0.0) + w * value
            for group, value in sorted(share.items(), key=lambda kv: -kv[1]):
                if value > 1e-9:
                    rows.append(
                        {
                            "unit": name,
                            "role": role,
                            "group": group,
                            "share": round(value, 6),
                        }
                    )
    return pd.DataFrame(rows, columns=["unit", "role", "group", "share"])


def group_summary(roles: pd.DataFrame) -> pd.DataFrame:
    """Each group's MW as owner (equity-weighted where shares are known), operator and trader. A battery row's
    capacity is already the group's own MW (battery_roles), so it is not weighted by its share again."""
    r = roles.copy()
    share = r["share"].fillna(0.0).astype(float)
    r["mw"] = r["capacity_mw"].where(
        r["unit"].eq(BATTERY_UNIT), r["capacity_mw"] * share
    )
    table = (
        r.pivot_table(index="group", columns="role", values="mw", aggfunc="sum")
        .reindex(columns=list(ROLES))
        .fillna(0.0)
    )
    return table.add_suffix("_mw").sort_values("operator_mw", ascending=False)


def write(
    year: int, agents: pd.DataFrame, scenario_dir: Path | None = None
) -> pd.DataFrame:
    """Build and write the year's roles (``ownership_data/built``) and, with ``scenario_dir``, the scenario's
    ``unit_owners.csv``. Returns the roles."""
    roles = pd.concat(
        [unit_roles(year, agents), battery_roles(year)], ignore_index=True
    )
    BUILT.mkdir(parents=True, exist_ok=True)
    roles.to_csv(BUILT / f"unit_roles_{year}.csv", index=False)
    group_summary(roles).round(1).to_csv(BUILT / f"groups_{year}.csv")
    if scenario_dir is not None and (Path(scenario_dir) / "unit_columns.csv").exists():
        columns = pd.read_csv(Path(scenario_dir) / "unit_columns.csv", index_col=0)
        view = scenario_view(roles[roles["unit"] != BATTERY_UNIT], columns)
        battery = roles[roles["unit"] == BATTERY_UNIT][
            ["unit", "role", "group", "share"]
        ]
        if (Path(scenario_dir) / "storage_units.csv").exists():
            view = pd.concat([view, battery.round({"share": 6})], ignore_index=True)
        view.to_csv(Path(scenario_dir) / "unit_owners.csv", index=False)
    return roles
