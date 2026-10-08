# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""The Electricity Generator Levy settled on a run, the way the levy is assessed, and checked
against what it raised.

The levy (Finance Act 2023, Part 2; 1 January 2023 to 31 March 2028) takes a share of the
receipts of low-carbon generation above a benchmark price: nuclear, wind, solar, hydro, biomass
and energy from waste are in scope, output under a CfD is not, nor are gas, coal, oil, pumped
storage and batteries. The rate is 45 percent (55 percent from 1 July 2026), the benchmark
GBP 75/MWh indexed to CPI from April 2024, the first GBP 10 million of excess receipts of a group
in a year are exempt, and a group generating 50 GWh or less a year is out of scope. It is
assessed on a group's realised receipts over the whole year, not period by period, so output
sold forward at a lower price shelters the group: a hedged generator pays nothing while spot
prices spike.

``settle`` applies exactly that to the framework's output of a run: the accepted volumes of every
in-scope unit from ``market_orders`` at the run's prices or at the observed day-ahead price,
grouped by owner (``owners.owners`` for the named plants; a merged class of small plants is a
group of its own, which overstates the levy on it, with the share of a named company's columns
in it attributed to that company through ``unit_columns.csv``), nuclear added from its metered
output (it is not a unit of the scenarios: the model nets it off demand), over the calendar year,
with an optional share of output sold forward at a contract price. ``validation`` puts the
settlement beside what the levy raised: HMRC's receipts, the forecast at introduction and the
OBR's outturns, and what EDF (nuclear) and Drax (RO biomass) disclosed, all read from the GB data
repository. The framework's own per-period levy (``PowerPlant.levy_payment``, the ``levy_rate``
and ``levy_benchmark`` parameters) is the marginal view a learning agent needs; this is the
settlement.
"""

from pathlib import Path

import pandas as pd

from assume_gb import owners, paths

IN_SCOPE = (
    "Wind",
    "Solar",
    "Hydro",
    "Biomass",
    "Solid Biomass",
    "Waste",
    "BIOMASS_MSG",
    "EMB_MUSTRUN",
    "Nuclear",
)
RATE = 0.45
RATE_FROM_JULY_2026 = 0.55
BENCHMARK_2023 = 75.0
ALLOWANCE_GBP = 10e6
THRESHOLD_GWH = 50.0
HOURS = 0.5  # length of a period in hours
NUCLEAR_GROUP = "EDF nuclear"
# companies whose plants sit inside merged classes, found by their stack unit ids
COLUMN_COMPANIES = {"Drax": "drax"}
ESPENI_NUCLEAR = "ELEC_POWER_ELEX_NUCLEAR[MW](float32)"
ESPENI_TIME = "ELEC_elex_startTime[utc](datetime)"


def rate_for(year: int) -> float:
    """The levy rate of a calendar year: 45 percent, 55 percent from 1 July 2026 (half of 2026)."""
    if year < 2026:
        return RATE
    if year == 2026:
        return (RATE + RATE_FROM_JULY_2026) / 2.0
    return RATE_FROM_JULY_2026


def in_scope(units: pd.DataFrame) -> pd.Series:
    """Which units the levy applies to: in-scope technologies without a CfD."""
    cfd = units["support_scheme"].fillna("").astype(str).str.lower() == "cfd"
    return units["technology"].isin(IN_SCOPE) & ~cfd


def nuclear_output(year: int) -> pd.Series:
    """Metered nuclear output, MW per half-hour, from the ESPENI snapshot in the GB data repository."""
    table = pd.read_csv(
        paths.gb_repo() / "data" / "raw" / "espeni.csv",
        usecols=[ESPENI_TIME, ESPENI_NUCLEAR],
    )
    index = pd.DatetimeIndex(
        pd.to_datetime(table.pop(ESPENI_TIME), utc=True)
    ).tz_localize(None)
    series = pd.Series(table[ESPENI_NUCLEAR].to_numpy(dtype=float), index=index)
    return series[series.index.year == year]


def column_shares(scenario_dir: Path | str) -> dict[str, dict[str, float]]:
    """For every unit that holds a named company's stack columns (``COLUMN_COMPANIES``), the
    company and the share of the unit's capacity those columns make up, from ``unit_columns.csv``."""
    path = Path(scenario_dir) / "unit_columns.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path, index_col=0)
    shares = {}
    for unit, row in table.iterrows():
        ids = str(row["stack_unit_ids"]).split(";")
        caps = [float(c) for c in str(row["column_capacity_mw"]).split(";")]
        total = sum(caps) or 1.0
        for company, key in COLUMN_COMPANIES.items():
            share = sum(c for i, c in zip(ids, caps) if key in i.lower()) / total
            if share > 0:
                shares.setdefault(unit, {})[company] = share
    return shares


def settle(
    run_folder: Path | str,
    units: pd.DataFrame,
    year: int,
    owner_by_unit: pd.Series | None = None,
    benchmark: float = BENCHMARK_2023,
    rate: float | None = None,
    allowance: float = ALLOWANCE_GBP,
    threshold_gwh: float = THRESHOLD_GWH,
    hedge_share: float = 0.0,
    hedge_price: float | None = None,
    observed_prices: pd.Series | None = None,
    nuclear: pd.Series | None = None,
    shares: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """The levy of every group for the calendar ``year`` of the run whose CSV output is in
    ``run_folder``: generation, receipts, the realised average price, the excess over the
    benchmark and the levy, one row per group, the groups that pay first.

    ``observed_prices`` (a half-hourly series) replaces the run's prices as what the output
    earns; ``nuclear`` (MW per half-hour) adds the nuclear fleet as one group; ``shares``
    (``column_shares``) hands a company its share of a merged class; ``hedge_share`` of every
    group's output is taken as sold forward at ``hedge_price`` (the benchmark by default).
    ``units`` is the scenario's units table (index: unit name); ``owner_by_unit`` maps unit
    names to groups."""
    rate = rate_for(year) if rate is None else rate
    hedge_price = benchmark if hedge_price is None else hedge_price
    orders = pd.read_csv(
        Path(run_folder) / "market_orders.csv",
        usecols=["unit_id", "start_time", "accepted_volume", "accepted_price"],
        parse_dates=["start_time"],
    )
    orders = orders[
        (orders["start_time"].dt.year == year) & (orders["accepted_volume"] > 0)
    ]
    scope = units.index[in_scope(units)]
    orders = orders[orders["unit_id"].isin(scope)].copy()
    if observed_prices is not None:
        orders["accepted_price"] = observed_prices.reindex(
            orders["start_time"]
        ).to_numpy()
        orders = orders[orders["accepted_price"].notna()]
    if nuclear is not None:
        price = (
            observed_prices if observed_prices is not None else _run_prices(run_folder)
        )
        nuc = pd.DataFrame(
            {
                "unit_id": NUCLEAR_GROUP,
                "start_time": nuclear.index,
                "accepted_volume": nuclear.to_numpy(),
            }
        )
        nuc["accepted_price"] = price.reindex(nuc["start_time"]).to_numpy()
        nuc = nuc[
            (nuc["accepted_volume"] > 0)
            & nuc["accepted_price"].notna()
            & (nuc["start_time"].dt.year == year)
        ]
        orders = pd.concat([orders, nuc], ignore_index=True)
    group = (
        orders["unit_id"].map(owner_by_unit)
        if owner_by_unit is not None
        else pd.Series(index=orders.index, dtype=object)
    )
    orders["group"] = group.fillna(orders["unit_id"])
    orders["weight"] = 1.0
    if shares:
        pieces = []
        for unit, companies in shares.items():
            mine = orders["unit_id"] == unit
            if not mine.any():
                continue
            rest = 1.0 - sum(companies.values())
            orders.loc[mine, "weight"] = rest
            for company, share in companies.items():
                piece = orders[mine].copy()
                piece["group"] = company
                piece["weight"] = share
                pieces.append(piece)
        orders = pd.concat([orders[orders["weight"] > 0], *pieces], ignore_index=True)
    orders["mwh"] = orders["accepted_volume"] * HOURS * orders["weight"]
    orders["receipts"] = orders["mwh"] * (
        (1 - hedge_share) * orders["accepted_price"] + hedge_share * hedge_price
    )
    table = orders.groupby("group").agg(
        generation_gwh=("mwh", lambda x: x.sum() / 1e3),
        receipts_gbp_m=("receipts", lambda x: x.sum() / 1e6),
    )
    table["units"] = orders.groupby("group")["unit_id"].nunique()
    table["realised_price"] = (
        table["receipts_gbp_m"] * 1e6 / (table["generation_gwh"] * 1e3)
    )
    table["excess_gbp_m"] = (
        (table["realised_price"] - benchmark).clip(lower=0.0)
        * table["generation_gwh"]
        * 1e3
    ) / 1e6
    table["in_scope"] = table["generation_gwh"] > threshold_gwh
    table["levy_gbp_m"] = (
        rate * (table["excess_gbp_m"] - allowance / 1e6).clip(lower=0.0)
    ).where(table["in_scope"], 0.0)
    table.attrs.update(
        {
            "year": year,
            "benchmark": benchmark,
            "rate": rate,
            "allowance": allowance,
            "hedge_share": hedge_share,
            "prices": "observed" if observed_prices is not None else "simulated",
            "nuclear": nuclear is not None,
        }
    )
    return table.sort_values("levy_gbp_m", ascending=False)


def _run_prices(run_folder: Path | str) -> pd.Series:
    path = Path(run_folder) / "clearing_prices.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    meta = pd.read_csv(
        Path(run_folder) / "market_meta.csv", parse_dates=["product_start"]
    )
    return meta.set_index("product_start")["price"].sort_index()


def scenario_units(run_folder: Path | str, scenario_dir: Path | str) -> pd.DataFrame:
    """The units of the scenario a run was made from (its learning file where the run used it)."""
    config_case = Path(run_folder).name
    folder = Path(scenario_dir)
    file = (
        "powerplant_units_learning.csv"
        if "learning" in config_case
        and (folder / "powerplant_units_learning.csv").exists()
        else "powerplant_units.csv"
    )
    return pd.read_csv(folder / file, index_col=0)


def summary(table: pd.DataFrame) -> str:
    paying = table[table["levy_gbp_m"] > 0]
    total = table["levy_gbp_m"].sum()
    excess = table.loc[table["in_scope"], "excess_gbp_m"].sum()
    a = table.attrs
    return (
        f"{a['year']}, {a['prices']} prices{', with nuclear' if a['nuclear'] else ''}: benchmark {a['benchmark']:.2f} GBP/MWh, "
        f"rate {a['rate']:.0%}, allowance {a['allowance'] / 1e6:.0f} m, hedged share {a['hedge_share']:.0%}; "
        f"{int(table['in_scope'].sum())} groups above {THRESHOLD_GWH:.0f} GWh with {excess:,.0f} m of excess receipts, "
        f"{len(paying)} pay, levy {total:,.0f} m GBP"
    )


def outturn() -> dict[str, pd.DataFrame]:
    """What the levy raised and was expected to raise, from the GB data repository: HMRC's cash
    receipts by month and fiscal year (``egl_receipts_hmrc.csv``), the Autumn Statement 2022
    costing and the OBR's later rows (``egl_forecasts.csv``), and the companies' disclosures
    (``egl_company_disclosures.csv``)."""
    data = paths.gb_repo() / "data"
    receipts = pd.read_csv(data / "processed" / "egl_receipts_hmrc.csv")
    receipts = receipts[receipts["reported"].astype(str).str.lower() == "true"]
    forecasts = pd.read_csv(
        data / "raw" / "manual" / "obr_egl_forecasts" / "egl_forecasts.csv"
    )
    companies = pd.read_csv(
        data
        / "raw"
        / "manual"
        / "company_egl_disclosures"
        / "egl_company_disclosures.csv"
    )
    return {"receipts": receipts, "forecasts": forecasts, "companies": companies}


def _fiscal(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


def _accrued(forecasts: pd.DataFrame, fiscal: str) -> float:
    """The OBR's latest accrued outturn of a fiscal year, NaN while there is none."""
    rows = forecasts[
        (forecasts["basis"] == "outturn_accrued") & (forecasts["fiscal_year"] == fiscal)
    ]
    return (
        float(rows.sort_values("published")["value_gbp_m"].iloc[-1])
        if not rows.empty
        else float("nan")
    )


def _disclosed(companies: pd.DataFrame, year: int) -> dict[str, float]:
    out = {}
    for _, r in companies[companies["calendar_year"] == year].iterrows():
        key = NUCLEAR_GROUP if r["company"] == "EDF" else r["company"]
        out[key] = float(r["value_gbp_m"])
    return out


def validation(settlements: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One table of the levy for a calendar year: the settlements given (name -> ``settle`` table)
    in total and for EDF nuclear and Drax, beside what the levy raised. The OBR's accrued outturn
    is the liability of the period and the nearest match to a calendar-year settlement (the
    calendar year taken as a quarter of one fiscal year and three quarters of the next); HMRC's
    cash comes in quarterly instalments and lags; the Autumn Statement 2022 costing is what was
    expected at introduction; the companies' disclosures are for their calendar years."""
    year = next(iter(settlements.values())).attrs["year"]
    data = outturn()
    rows = {}
    for name, table in settlements.items():
        rows[name] = {
            "total": table["levy_gbp_m"].sum(),
            "EDF nuclear": table["levy_gbp_m"].get(NUCLEAR_GROUP, float("nan")),
            "Drax": table["levy_gbp_m"].get("Drax", float("nan")),
        }
    fiscal = _fiscal(year)
    forecasts = data["forecasts"]
    before, during = _accrued(forecasts, _fiscal(year - 1)), _accrued(forecasts, fiscal)
    rows[f"OBR accrued outturn, calendar {year}"] = {
        "total": 0.25 * before + 0.75 * during
    }
    rows[f"OBR accrued outturn, fiscal {fiscal}"] = {"total": during}
    receipts = data["receipts"]
    months = receipts[receipts["period_type"] == "month"]
    cash_calendar = months.loc[
        months["period"].str[:4] == str(year), "value_gbp_m"
    ].sum()
    cash_fiscal = receipts.loc[
        (receipts["period_type"] == "fiscal_year") & (receipts["period"] == fiscal),
        "value_gbp_m",
    ].sum()
    rows[f"HMRC cash, calendar {year}"] = {"total": cash_calendar}
    rows[f"HMRC cash, fiscal {fiscal}"] = {"total": cash_fiscal}
    at_introduction = forecasts[
        (forecasts["publication"].str.contains("Autumn Statement 2022"))
        & (forecasts["fiscal_year"] == fiscal)
    ]
    if not at_introduction.empty:
        rows[f"Autumn Statement 2022 costing, {fiscal}"] = {
            "total": float(at_introduction["value_gbp_m"].iloc[0])
        }
    disclosed = _disclosed(data["companies"], year)
    if disclosed:
        rows[f"disclosed by the companies, {year}"] = {
            ("EDF nuclear" if k == NUCLEAR_GROUP else k): v
            for k, v in disclosed.items()
        }
    return pd.DataFrame(rows).T


def implied_prices(table: pd.DataFrame) -> pd.DataFrame:
    """The realised price each company's disclosed levy implies for the generation the settlement
    attributes to it, benchmark + (levy / rate + allowance) / generation, beside the price the
    settlement used: the gap between the two is what forward sales did to that company's
    receipts in the year."""
    a = table.attrs
    disclosed = _disclosed(outturn()["companies"], a["year"])
    rows = {}
    for company, levy in disclosed.items():
        if company not in table.index or table.loc[company, "generation_gwh"] <= 0:
            continue
        generation = table.loc[company, "generation_gwh"] * 1e3
        excess = (levy * 1e6 / a["rate"] + a["allowance"]) if levy > 0 else float("nan")
        rows["EDF nuclear" if company == NUCLEAR_GROUP else company] = {
            "generation_gwh": table.loc[company, "generation_gwh"],
            "settled_at_gbp_mwh": table.loc[company, "realised_price"],
            "disclosed_levy_gbp_m": levy,
            "implied_realised_gbp_mwh": a["benchmark"] + excess / generation,
        }
    return pd.DataFrame(rows).T


def unit_parameters(units: pd.DataFrame, rate: float, benchmark: float) -> pd.DataFrame:
    """``levy_rate`` and ``levy_benchmark`` for the in-scope units of a units table (the
    framework's per-period levy, for the learning rewards), zero for the rest."""
    out = units.copy()
    scope = in_scope(out)
    out["levy_rate"] = 0.0
    out["levy_benchmark"] = 0.0
    out.loc[scope, "levy_rate"] = rate
    out.loc[scope, "levy_benchmark"] = benchmark
    return out


_ = owners  # the owner map is the grouping the settlement needs; the command passes it in
