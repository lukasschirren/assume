# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Who owns and who bids the named plants of a scenario, for the portfolio learning cases and the generator levy.

Since 6 Oct 2026 this is a thin layer over ``ownership.py``, which takes owner, operator and trader of every unit
from gb-power-data (the public GB data package: its registers, its curated tables with a source for every row,
and its BM-unit table by year). The hand-written tables that lived here moved there: the plants' DUKES sites
(``ALIASES`` -> gb-power-data's ``dukes_bmu_map.csv``, with the stations' BM units; ``plants.csv`` here names
each model plant's station), the company groups (``GROUPS``, and ``WIND_GROUPS`` for the RO wind farms' holders
-> ``company_groups.csv``) and the joint ventures noted in comments (-> ``company_owners.csv``, with shares,
operator and dates). ``owners`` keeps its signature and returns each unit's OPERATOR, the company that runs the
unit for its owners, by which the generator levy groups units. ``bidders`` returns the company that TRADES each
unit, the portfolio agent of the learning cases since 7 Oct 2026 (``ownership.bidders``: the model evaluates
market power, which whoever bids a unit exercises). The owners' equity shares are in ``ownership.unit_roles`` and
the scenario's ``unit_owners.csv``.

Differences against the tables of 5 Oct 2026 (checked for 2018-2025 on the register stacks, nothing dropped):
West Burton B is EIG's from 31 Aug 2021, not EDF's; Saltend and Deeside are Triton Power's (the joint venture
of SSE and Equinor since 1 Sep 2022, Energy Capital Partners before), not "ECP"; the plants of the older DUKES
editions are found (their site names differ from the 2024 edition the aliases were written for); company
names are always reduced to groups ("Drax Power Ltd" -> Drax, "EPPDII Ltd" -> EPUKi); after the check of the
joint ventures on 6 Oct 2026, Marchwood is SSE's to run (all its output goes to SSE) and in 2018 Robin Rigg,
Humber Gateway and Rampion are E.ON's (its renewables went to RWE at the end of 30 Sep 2019), while London Array
was run by its own company until RWE took over operations at the start of 2023 (The Crown Estate's tables).
"""

import pandas as pd

from assume_gb import ownership


def owners(year: int, unit_names: list[str], unit_labels: dict[str, str] | None = None, agents: pd.DataFrame | None = None,
           kinds: tuple[str, ...] = ownership.PORTFOLIO_KINDS) -> pd.Series:  # fmt: skip
    """The operator group of every named unit (the gas and coal plants of DESNZ's list for the end of ``year``
    and the RO wind farms of Ofgem's register; the CfD units too with ``ownership.PORTFOLIO_KINDS_CFD``), by unit
    name: the company that runs the unit for its owners. Units without a match are left out."""
    return ownership.operators(
        year, unit_names, unit_labels, agents=agents, kinds=kinds
    )


def bidders(year: int, unit_names: list[str], unit_labels: dict[str, str] | None = None, agents: pd.DataFrame | None = None,
            kinds: tuple[str, ...] = ownership.PORTFOLIO_KINDS) -> pd.Series:  # fmt: skip
    """The bidding agent of every named unit (the same units as ``owners``), by unit name: the company group that
    trades its BM units in ``year``, else its operator. Units without a match are left out."""
    return ownership.bidders(year, unit_names, unit_labels, agents=agents, kinds=kinds)


def portfolio_agents(level: str, year: int, unit_names: list[str], unit_labels: dict[str, str] | None = None, agents: pd.DataFrame | None = None,
                     kinds: tuple[str, ...] = ownership.PORTFOLIO_KINDS) -> pd.Series:  # fmt: skip
    """The portfolio agent of every named unit of ``kinds`` at ``level`` (``ownership.AGENT_LEVELS``): ``bidders``
    for "trader", ``owners`` for "operator"."""
    if level not in ownership.AGENT_LEVELS:
        raise ValueError(
            f"agent level must be one of {ownership.AGENT_LEVELS}, not {level!r}"
        )
    return (bidders if level == "trader" else owners)(
        year, unit_names, unit_labels, agents=agents, kinds=kinds
    )


def portfolios(owner_by_unit: pd.Series, min_units: int) -> dict[str, list[str]]:
    """The owners with at least ``min_units`` units and their units, largest portfolios first."""
    groups = owner_by_unit.groupby(owner_by_unit).apply(lambda s: s.index.tolist())
    groups = {
        owner: units for owner, units in groups.items() if len(units) >= min_units
    }
    return dict(sorted(groups.items(), key=lambda kv: -len(kv[1])))
