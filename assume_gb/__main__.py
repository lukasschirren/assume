# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Command line of the local GB data tool.

    python -m assume_gb build --year 2023     offer stack, forecasts, observed prices -> assume_gb/inputs/gb_2023
    python -m assume_gb check --year 2023     the scenario's offers against the model's own
    python -m assume_gb run   --year 2023     simulate a study case and compare its prices
    python -m assume_gb figures RUN_FOLDER    the standard figures and the viewer of a run
    python -m assume_gb learning --db URI --simulation NAME    the figures of a learning run
    python -m assume_gb ownership --years 2018 2025    owner, operator and trader of every unit (ownership.md)
    python -m assume_gb agents --years 2018 2025       the portfolio learning agents of built folders, rewritten
                       [--portfolio-cfd]                 (+ cases learning_cfd*: the CfD units in the portfolios)
    python -m assume_gb penetration --years 2018 2025  the forecasts that define predicted penetration (validation)
    python -m assume_gb validate --year 2023 --arm learning_novdec_own_e100    scorecards, tables, figures (validation/README.md)
    python -m assume_gb tidy [--apply]                 results written in the old flat layout into the tidy one

The offer stack is written first, once per year, in the GB data repository
(``python -m meritorder.stack_export --years 2023 2023``); ``GB_DATA_REPO`` points at that
repository if it is not the sibling folder. ``--no-merge`` keeps one unit per stack column;
``--no-intraday`` builds without the forecasts and the intraday cases. ``run --case`` picks a
study case (``day_ahead``, ``day_ahead_storage``, ``day_ahead_intraday``, each also as ``*_week``
for the first week only), ``--years FROM TO`` runs several years with a summary, ``--save``
saves the simulated clearing prices at full precision into the results folder
(``results/prices/<run>.csv``; ``--out DIR`` instead writes ``DIR/<run>_prices.csv``) and ``--csv
DIR`` also writes the framework's CSV output there (``DIR/<scenario>_<case>``, with the clearing
prices at full precision beside it); ``--figures`` then draws the run's figures into
``results/figures/<run>``. Every path into the results folder comes from ``paths.results_dir``
(``results/README.md``); ``tidy`` moves results written in the old flat layout into it.

A learning case (``learning``, ``learning_week``, ``learning_year``) trains its agents first and
writes to a database (``--db``, by default ``results/db/<run>.db``), which the
``learning`` command reads. ``variant`` derives a study case from another with settings changed
(``--set seed=1 --set learning_config.noise_sigma=0.2``), one per seed or sweep point, so that
every run has its own simulation id, database and policies. ``figures RUN_FOLDER`` reads the framework's CSV output of a run and writes PNG and PDF figures, the
numbers they quote and an interactive viewer (``viewer.html``) to ``results/figures/<run>``;
``--baseline RUN_FOLDER`` adds the comparison with another run, ``--only 01,10`` draws some figures,
``--no-viewer`` skips the viewer. The viewer and the figures hold the licensed observed prices: keep
them local. ``learning`` draws the learning curves of a learning run from its database; several
``--simulation`` names are taken as repetitions with different seeds.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from assume_gb import (
    compare,
    convert,
    egl,
    forecasts,
    observed,
    owners,
    ownership,
    paths,
)
from assume_gb.stack import load_stack, stack_files


def build(args) -> int:
    stack = load_stack(args.year, args.design, args.tag)
    errors = None if args.no_intraday else forecasts.errors(args.year, stack.index)
    observed_prices = observed.prices(args.year, stack.index)
    learning_owners = None
    if args.learning_pick == "portfolio":
        agents = stack.agents
        labels = dict(zip(agents["unit_id"].astype(str), agents["name"].astype(str)))
        learning_owners = owners.portfolio_agents(
            args.agent_level,
            args.year,
            convert.unit_table(stack)["name"].tolist(),
            labels,
            agents=agents,
        )
    battery_traders = (
        ownership.battery_traders(args.year) if args.battery_traders else None
    )
    folder = convert.build_scenario(
        stack,
        paths.scenarios_dir() / (args.name or paths.scenario_name(args.year)),
        merge=args.merge,
        forecast_errors=errors,
        observed=observed_prices,
        learning_techs=tuple(args.learning_techs),
        learning_count=args.learning_units,
        learning_pick=args.learning_pick,
        learning_owners=learning_owners,
        learning_agent_level=args.agent_level
        if args.learning_pick == "portfolio"
        else None,
        nbins=args.nbins,
        min_portfolio_mw=args.min_portfolio_mw,
        levy=(args.egl_rate, args.egl_benchmark) if args.egl_rate else None,
        max_markup=args.max_markup,
        learning_weeks=args.learning_weeks,
        max_bid_price=args.max_bid_price,
        battery_traders=battery_traders,
    )
    if errors is not None:
        print(
            f"penetration forecasts: {forecasts.write_penetration(args.year, folder)}"
        )
    try:
        roles = ownership.write(args.year, stack.agents, folder)
        print(
            f"ownership: {folder / 'unit_owners.csv'} ({roles['unit'].nunique()} units with an owner, operator or trader of record)"
        )
    except FileNotFoundError as error:
        print(f"ownership not written ({error})")
    if (
        args.portfolio_cfd
        and args.learning_pick == "portfolio"
        and learning_owners is not None
    ):
        with_cfd = owners.portfolio_agents(
            args.agent_level,
            args.year,
            convert.unit_table(stack)["name"].tolist(),
            labels,
            agents=agents,
            kinds=ownership.PORTFOLIO_KINDS_CFD,
        )
        cfd_units, cfd_operators = convert.write_cfd_variant(
            folder, stack, with_cfd, args.agent_level, args.min_portfolio_mw
        )
        print(
            f"learning portfolios with the CfD units (cases {convert.LEARNING_CFD_CASE}*): {len(cfd_units)} units, {cfd_operators}"
        )
    units = pd.read_csv(folder / "powerplant_units.csv")
    meta = json.loads((folder / "scenario_meta.json").read_text(encoding="utf-8"))
    print(
        f"{folder}: {len(units)} units from {stack.bids.shape[1]} stack columns, {len(stack.index)} periods, cases {meta['cases']}"
    )
    if meta["forecast_errors"]:
        print(
            "forecast errors:",
            {k: round(v, 3) for k, v in meta["forecast_errors"].items()},
        )
    print("observed prices:", list(observed_prices.columns))
    if meta["learning"]:
        print(
            f"learning units: {len(meta['learning']['units'])} ({', '.join(meta['learning']['technologies'])}), "
            f"max bid price {meta['learning']['max_bid_price']:.0f}, training weeks {meta['learning']['training_weeks']}"
        )
        if meta["learning"]["operators"]:
            print(
                f"learning portfolios ({meta['learning']['agent_level']} level): {meta['learning']['operators']}, {meta['learning']['portfolio']}"
            )
    return 0


def check(args) -> int:
    table = convert.check_offers(load_stack(args.year, args.design, args.tag))
    print(table.round(6).to_string())
    return 0 if table.loc["all", "max_abs_dev"] < 1e-6 else 1


def run(args) -> int:
    years = range(args.years[0], args.years[1] + 1) if args.years else [args.year]
    tables = {}
    for year in years:
        name = args.name or paths.scenario_name(year)
        db = args.db
        if not db and convert.is_learning_case(paths.scenarios_dir() / name, args.case):
            database = paths.database_file(paths.run_name(year, args.case, args.name))
            database.parent.mkdir(parents=True, exist_ok=True)
            db = f"sqlite:///{database.as_posix()}"
            print(f"learning run, database {db}", flush=True)
        world, seconds = compare.run_scenario(
            name, args.case, csv_path=args.csv, database_uri=db
        )
        prices = compare.clearing_prices(world)
        if args.csv:  # beside the framework's output, whose CSV files keep five digits
            prices.rename_axis("datetime").to_csv(
                Path(args.csv) / f"{name}_{args.case}" / "clearing_prices.csv"
            )
        if args.save:
            saved = paths.prices_file(
                paths.run_name(year, args.case, args.name), args.save
            )
            saved.parent.mkdir(parents=True, exist_ok=True)
            prices.rename_axis("datetime").to_csv(saved)
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            prices.rename_axis("datetime").to_csv(
                out / f"{name}_{args.case}_prices.csv"
            )
        tables[name] = compare.compare(prices, paths.scenarios_dir() / name)
        if args.figures:
            figures(
                argparse.Namespace(
                    folder=str(Path(args.csv) / f"{name}_{args.case}"),
                    baseline="",
                    only="",
                    no_viewer=False,
                    out="",
                )
            )
        print(f"simulated {name}/{args.case} in {seconds:.0f} s", flush=True)
        print(tables[name].to_string(float_format="{:.4g}".format), flush=True)
    if len(tables) > 1:
        print("\nall years, one column per comparison")
        for column in ("max_abs", "mae", "bias", "r2"):
            summary = pd.DataFrame(
                {name: table[column] for name, table in tables.items()}
            ).T
            print(f"\n{column}\n" + summary.to_string(float_format="{:.4g}".format))
    return 0


def figures(args) -> int:
    from assume_gb import figures as figs
    from assume_gb import results, viewer

    run = results.load(args.folder)
    base = results.load(args.baseline) if args.baseline else None
    out = (
        Path(args.out)
        if args.out
        else paths.results_dir("figures") / Path(args.folder).name
    )
    only = [k.strip() for k in args.only.split(",") if k.strip()] or None
    written = figs.make(run, out, only=only, baseline=base)
    if not args.no_viewer and not only:
        written.append(viewer.write(run, out))
    print(f"{run.name}: {len(written)} files in {out}")
    for path in written:
        print("  " + path.name)
    return 0


def learning(args) -> int:
    from assume_gb import learning_figures

    out = (
        Path(args.out)
        if args.out
        else paths.results_dir("learning") / "_".join(args.simulation)
    )
    written = learning_figures.make(args.db, args.simulation, out)
    print(f"{len(written)} figures in {out}")
    for path in written:
        print("  " + path.name)
    return 0


def levy(args) -> int:
    """The electricity generator levy settled on a run, by group."""
    from assume_gb import results

    folder = Path(args.folder)
    scenario, _ = results._scenario_and_case(folder.name)
    scenario_dir = paths.scenarios_dir() / scenario
    units = egl.scenario_units(folder, scenario_dir)
    year = args.year or int(
        json.loads((scenario_dir / "scenario_meta.json").read_text(encoding="utf-8"))[
            "source"
        ]["year"]
    )
    owner_by_unit = None
    if not args.no_owners:
        # the scenario's own unit_owners.csv first (no GB data repository needed, as on the HPC)
        owner_by_unit = ownership.operators_from_view(scenario_dir)
        if owner_by_unit is None:
            try:
                owner_by_unit = owners.owners(year, units.index.tolist(), None)
            except FileNotFoundError as error:
                print(
                    f"owners not available ({error}); every unit is a group of its own"
                )
    observed = None
    if args.observed_prices or args.validate:
        observed = pd.read_csv(
            scenario_dir / "observed_prices.csv", index_col=0, parse_dates=True
        )["n2ex_day_ahead"]
    nuclear = egl.nuclear_output(year) if (args.nuclear or args.validate) else None
    shares = egl.column_shares(scenario_dir)
    settings = dict(
        benchmark=args.benchmark,
        rate=args.rate,
        allowance=args.allowance,
        threshold_gwh=args.threshold_gwh,
        hedge_price=args.hedge_price,
        shares=shares,
    )
    table = egl.settle(
        folder,
        units,
        year,
        owner_by_unit,
        hedge_share=args.hedge_share,
        observed_prices=observed if args.observed_prices else None,
        nuclear=nuclear,
        **settings,
    )
    print(egl.summary(table))
    print(
        table[table["generation_gwh"] > 0]
        .head(args.rows)
        .to_string(float_format="{:,.2f}".format)
    )
    out = (
        paths.results_dir("figures") / folder.name / "egl.csv"
    )  # beside the run's figures
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out)
    print(f"written {out}")
    if args.validate:
        # the settlement under the run's and the observed prices, with nuclear, unhedged and with half
        # the output sold forward, beside what the levy raised
        settlements = {
            "settled, simulated prices": table
            if not args.observed_prices and not args.hedge_share
            else egl.settle(
                folder, units, year, owner_by_unit, nuclear=nuclear, **settings
            ),
            "settled, observed prices": egl.settle(
                folder,
                units,
                year,
                owner_by_unit,
                observed_prices=observed,
                nuclear=nuclear,
                **settings,
            ),
            "settled, observed prices, half sold forward": egl.settle(
                folder,
                units,
                year,
                owner_by_unit,
                observed_prices=observed,
                nuclear=nuclear,
                hedge_share=0.5,
                **settings,
            ),
        }
        print("")
        print("levy in GBP million against what it raised:")
        print(
            egl.validation(settlements).to_string(
                float_format="{:,.0f}".format, na_rep=""
            )
        )
        implied = egl.implied_prices(settlements["settled, observed prices"])
        if not implied.empty:
            print("")
            print(
                "realised price per MWh implied by each company's disclosed levy, beside the observed price it was settled at:"
            )
            print(implied.to_string(float_format="{:,.1f}".format))
    return 0


def ownership_command(args) -> int:
    """Owner, operator and trader of every unit by year (``ownership.py``), written to ownership_data/built and,
    where the scenario folder exists, to its ``unit_owners.csv``. Nothing else in the folder changes."""
    years = range(args.years[0], args.years[1] + 1) if args.years else [args.year]
    for year in years:
        agents = pd.read_csv(stack_files(year, args.design, args.tag)["agents"])
        folder = paths.scenarios_dir() / (args.name or paths.scenario_name(year))
        roles = ownership.write(year, agents, folder if folder.exists() else None)
        named = roles[roles["kind"].isin(["plant", "farm", "cfd"])]
        owned = named[
            (named["role"] == "owner") & (named["group"] != ownership.UNRESOLVED)
        ]
        operated = (
            named[named["role"] == "operator"]
            .groupby("group")["capacity_mw"]
            .sum()
            .sort_values(ascending=False)
        )
        traded = named[named["role"] == "trader"]
        third = traded.merge(
            named[named["role"] == "operator"][["unit", "group"]],
            on="unit",
            suffixes=("", "_operator"),
        )
        third = third[third["group"] != third["group_operator"]]
        print(
            f"{year}: {owned['unit'].nunique()} named units with an owner of record "
            f"({owned.groupby('kind')['unit'].nunique().to_dict()}), "
            f"{operated.size} operators; largest {', '.join(f'{g} {mw / 1e3:.1f} GW' for g, mw in operated.head(6).items())}; "
            f"{third['unit'].nunique()} units traded by another company than their operator"
        )
        if folder.exists():
            print(f"  -> {folder / 'unit_owners.csv'}")
    print(f"  -> {ownership.BUILT}")
    return 0


def agents_command(args) -> int:
    """The portfolio learning agents of built scenarios at ``--agent-level`` (``convert.rewrite_learning``): the
    learning files and the learning block of ``scenario_meta.json`` are rewritten, nothing else in the folder. With
    ``--portfolio-cfd``, or where a folder has them already, also the cases in which the CfD units belong to the
    portfolios (``convert.write_cfd_variant``: their own files, and the three cases added to ``config.yaml``)."""
    years = range(args.years[0], args.years[1] + 1) if args.years else [args.year]
    for year in years:
        folder = paths.scenarios_dir() / (args.name or paths.scenario_name(year))
        meta = json.loads((folder / "scenario_meta.json").read_text(encoding="utf-8"))
        before = dict.fromkeys((meta.get("learning") or {}).get("operators") or [])
        agents = pd.read_csv(
            stack_files(year, meta["source"]["design"], meta["source"]["tag"])["agents"]
        )
        names = pd.read_csv(folder / "powerplant_units.csv")["name"].tolist()
        agent_by_unit = owners.portfolio_agents(
            args.agent_level, year, names, agents=agents
        )
        units, operators = convert.rewrite_learning(
            folder, agent_by_unit, args.agent_level, args.min_portfolio_mw
        )
        print(
            f"{folder.name}: {len(units)} learning units in {len(operators)} portfolios at {args.agent_level} level: {', '.join(operators)}"
            + (
                f"; new {sorted(set(operators) - set(before))}, gone {sorted(set(before) - set(operators))}"
                if before
                else ""
            )
        )
        if args.portfolio_cfd or (meta.get("learning") or {}).get("cfd_variant"):
            stack = load_stack(year, meta["source"]["design"], meta["source"]["tag"])
            with_cfd = owners.portfolio_agents(args.agent_level, year, stack.agents["unit_id"].astype(str).tolist(), agents=stack.agents,
                                               kinds=ownership.PORTFOLIO_KINDS_CFD)  # fmt: skip
            cfd_units, cfd_operators = convert.write_cfd_variant(
                folder, stack, with_cfd, args.agent_level, args.min_portfolio_mw
            )
            print(
                f"  with the CfD units (cases {convert.LEARNING_CFD_CASE}*): {len(cfd_units)} learning units in {len(cfd_operators)} portfolios: {', '.join(cfd_operators)}"
            )
        if (folder / "learned_strategies").exists():
            print(
                f"  the policies in {folder / 'learned_strategies'} were trained on other agents: train again"
            )
    return 0


def variant(args) -> int:
    """Derive a study case from another with some settings changed, for sweeps and seeds."""
    import copy

    import yaml

    folder = paths.scenarios_dir() / (args.name or paths.scenario_name(args.year))
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    if args.source not in config:
        raise SystemExit(
            f"{folder.name} has no study case {args.source!r}; it has {list(config)}"
        )
    case = copy.deepcopy(config[args.source])
    for setting in args.set:
        key, _, value = setting.partition("=")
        if not value:
            raise SystemExit(f"--set needs key=value, not {setting!r}")
        target = case
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = yaml.safe_load(value)
    config[args.case] = case
    (folder / "config.yaml").write_text(
        convert.SPDX_HEADER + yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    print(
        f"{folder.name}: study case {args.case} from {args.source} with {args.set or 'no changes'}"
    )
    return 0


def penetration(args) -> int:
    """Writes only ``penetration_forecasts.csv`` into existing scenario folders: the file to copy to
    a machine without the GB data repository."""
    years = range(args.years[0], args.years[1] + 1) if args.years else [args.year]
    for year in years:
        path = forecasts.write_penetration(
            year, paths.scenarios_dir() / (args.name or paths.scenario_name(year))
        )
        table = pd.read_csv(path, index_col=0)
        load = (
            "transmission system demand"
            if "tsd_forecast_mw" in table
            else "national demand (the forecast state has no TSD forecast)"
        )
        print(
            f"{path}: transmission wind and {load}, {int(table.notna().all(axis=1).sum())} periods"
        )
    return 0


def validate(args) -> int:
    """The validation against observed prices (``assume_gb/validation/README.md``): of a year, with
    a learning arm's seeds as the agent-based model, or of several years with the competitive run
    only; scorecards, tables and figures into one folder."""
    from assume_gb.validation import adapters, report

    years = list(range(args.years[0], args.years[1] + 1)) if args.years else [args.year]
    runs = dict(item.split("=", 1) for item in args.run)
    regimes = {}
    for item in args.regime:
        name, span = item.split("=", 1)
        regimes[name] = tuple(span.split(":", 1))
    data = adapters.load(
        years, args.arm, args.venue, args.policies, args.seeds, runs, regimes
    )
    span = paths.scenario_name(years[0]) + (f"_{years[-1]}" if len(years) > 1 else "")
    name = (
        f"{span}_{args.arm}{'_last' if args.policies == 'last' else ''}"
        if args.arm
        else f"{span}_competitive"
    )
    out = Path(args.out) if args.out else paths.results_dir("validation") / name
    written = report.write(
        data,
        out,
        market=adapters.market(args.venue),
        battery=adapters.battery(years[0]),
        draw=not args.no_figures,
    )
    notes = (out / "notes.txt").read_text(encoding="utf-8").strip()
    print(f"{out}: {len(written)} files" + (f"; left out:\n{notes}" if notes else ""))
    return 0


def tidy_command(args) -> int:
    """Moves results written in the old flat layout into the tidy one (``tidy``)."""
    from assume_gb import tidy

    moves = tidy.plan(args.root or None)
    if not args.apply:
        for source, target in moves:
            print(f"{source} -> {target}")
        print(f"{len(moves)} files would move; --apply moves them")
        return 0
    moved, skipped = tidy.apply(moves, args.root or None)
    for source, target in skipped:
        print(f"left in place, {target} exists: {source}")
    print(f"{len(moved)} files moved, {len(skipped)} left in place")
    return 0


def thermal_command(args) -> int:
    """Units files and study cases with non-convex thermal supply (``thermal``, thermal.py)."""
    from assume_gb import thermal

    table = pd.concat(
        [thermal.write(y) for y in range(args.years[0], args.years[-1] + 1)]
    )
    print(table.round(3).to_string(index=False))
    return 0


def curves_command(args) -> int:
    """The exchanges' bid curves of each year, resampled, with their reconstruction check (``curves``)."""
    from assume_gb import exchange_curves

    check = exchange_curves.build(list(range(args.years[0], args.years[-1] + 1)))
    print(check.to_string(index=False, float_format="{:.3f}".format))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m assume_gb",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    helps = {
        "build": "write the scenario folder",
        "check": "compare the scenario's offers",
        "run": "simulate and compare prices",
    }
    p = sub.add_parser("figures", help="the standard figures and the viewer of a run")
    p.add_argument(
        "folder",
        help="the framework's CSV output of a run: <csv dir>/<scenario>_<case>",
    )
    p.add_argument(
        "--baseline", default="", help="the CSV output of a run to compare with"
    )
    p.add_argument(
        "--out",
        default="",
        help="folder for the figures (default: assume_gb/results/figures/<run>)",
    )
    p.add_argument(
        "--only", default="", help="comma-separated parts of figure names, e.g. 01,10"
    )
    p.add_argument(
        "--no-viewer", action="store_true", help="without the interactive viewer"
    )
    p = sub.add_parser(
        "egl", help="the electricity generator levy settled on a run, by group"
    )
    p.add_argument(
        "folder",
        help="the framework's CSV output of a run: <csv dir>/<scenario>_<case>",
    )
    p.add_argument(
        "--year",
        type=int,
        default=None,
        help="calendar year to settle (default: the scenario's year)",
    )
    p.add_argument(
        "--benchmark",
        type=float,
        default=egl.BENCHMARK_2023,
        help="benchmark price, GBP/MWh (75 in 2023, CPI-indexed from April 2024)",
    )
    p.add_argument(
        "--rate",
        type=float,
        default=None,
        help="levy rate (default 0.45; 0.55 from July 2026)",
    )
    p.add_argument(
        "--allowance",
        type=float,
        default=egl.ALLOWANCE_GBP,
        help="exempt excess receipts per group and year, GBP (default 10 million)",
    )
    p.add_argument(
        "--threshold-gwh",
        type=float,
        default=egl.THRESHOLD_GWH,
        help="groups generating no more than this are out of scope (default 50 GWh)",
    )
    p.add_argument(
        "--hedge-share",
        type=float,
        default=0.0,
        help="share of output sold forward at --hedge-price instead of the spot price (default 0)",
    )
    p.add_argument(
        "--hedge-price",
        type=float,
        default=None,
        help="the forward price of the hedged share (default: the benchmark)",
    )
    p.add_argument("--rows", type=int, default=25, help="groups to print")
    p.add_argument(
        "--no-owners", action="store_true", help="every unit as a group of its own"
    )
    p.add_argument(
        "--observed-prices",
        action="store_true",
        help="settle at the observed N2EX day-ahead price instead of the run's",
    )
    p.add_argument(
        "--nuclear",
        action="store_true",
        help="add the nuclear fleet from its metered output (ESPENI), which is not a unit",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="also settle at observed prices with nuclear and compare with HMRC receipts, the costing and company disclosures",
    )
    p = sub.add_parser(
        "variant",
        help="a study case derived from another with settings changed (seeds, sweeps)",
    )
    p.add_argument("--year", type=int, default=2023)
    p.add_argument(
        "--name", default="", help="scenario folder name (default gb_<year>)"
    )
    p.add_argument(
        "--source",
        default=convert.LEARNING_CASE,
        help=f"the study case to derive from (default {convert.LEARNING_CASE})",
    )
    p.add_argument(
        "--case", required=True, help="name of the new study case, e.g. learning_s1"
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="a setting to change, dotted for nesting, e.g. seed=1, learning_config.noise_sigma=0.2",
    )
    p = sub.add_parser(
        "ownership",
        help="owner, operator and trader of every unit (ownership.md), per year",
    )
    p.add_argument("--year", type=int, default=2023)
    p.add_argument(
        "--years", nargs=2, type=int, metavar=("FROM", "TO"), help="several years"
    )
    p.add_argument(
        "--name", default="", help="scenario folder name (default gb_<year>)"
    )
    p.add_argument("--design", default="status_quo")
    p.add_argument(
        "--tag",
        default="register",
        help="tag of the stack export whose units are attributed (default register)",
    )
    p = sub.add_parser(
        "agents",
        help="the portfolio learning agents of built scenarios, rewritten at another level (ownership.md)",
    )
    p.add_argument("--year", type=int, default=2023)
    p.add_argument(
        "--years", nargs=2, type=int, metavar=("FROM", "TO"), help="several years"
    )
    p.add_argument(
        "--name", default="", help="scenario folder name (default gb_<year>)"
    )
    p.add_argument("--agent-level", choices=ownership.AGENT_LEVELS, default=ownership.AGENT_LEVELS[0],
                   help="the company that trades each unit (default) or the one that runs it")  # fmt: skip
    p.add_argument(
        "--min-portfolio-mw",
        type=float,
        default=500.0,
        help="the least capacity an agent needs, where the folder records none (default 500 MW)",
    )
    p.add_argument(
        "--portfolio-cfd", action="store_true",
        help=f"also the cases {convert.LEARNING_CFD_CASE}* in which the CfD units belong to their companies' portfolios (rewritten anyway where a folder has them)",
    )  # fmt: skip
    p = sub.add_parser(
        "penetration",
        help="the forecasts that define predicted penetration, into existing scenario folders",
    )
    p.add_argument("--year", type=int, default=2023)
    p.add_argument(
        "--years", nargs=2, type=int, metavar=("FROM", "TO"), help="several years"
    )
    p.add_argument(
        "--name", default="", help="scenario folder name (default gb_<year>)"
    )
    p = sub.add_parser(
        "validate",
        help="scorecards, tables and figures of the validation against observed prices",
    )
    p.add_argument("--year", type=int, default=2023)
    p.add_argument(
        "--years",
        nargs=2,
        type=int,
        metavar=("FROM", "TO"),
        help="several years, the competitive run only (no --arm)",
    )
    p.add_argument(
        "--arm",
        default=None,
        help="a learning arm, e.g. learning_novdec_own_e100: its year runs, one per seed",
    )
    p.add_argument(
        "--policies",
        choices=("best", "last"),
        default="best",
        help="year runs with the best evaluation's policies or the last",
    )
    p.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="only these seeds (default: all found)",
    )
    p.add_argument(
        "--venue", default="n2ex_day_ahead", choices=("n2ex_day_ahead", "epex_day_ahead", "blend_day_ahead", "epex_hh_day_ahead"),
        help="the observed price (default N2EX, hourly)",
    )  # fmt: skip
    p.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=CASE",
        help="a further saved run as a model, e.g. storage=day_ahead_storage",
    )
    p.add_argument(
        "--regime",
        action="append",
        default=[],
        metavar="NAME=START:END",
        help="a regime of the mechanism checks, e.g. crisis=2021-07-01:2022-12-31",
    )
    p.add_argument(
        "--out",
        default="",
        help="folder (default: assume_gb/results/validation/<name>)",
    )
    p.add_argument(
        "--no-figures", action="store_true", help="scorecards and tables only"
    )
    p = sub.add_parser(
        "thermal",
        help="units and cases with non-convex thermal supply (thermal.py, inputs/thermal_parameters.csv)",
    )
    p.add_argument(
        "--years", type=int, nargs="+", default=[2023], help="first and last year"
    )
    p = sub.add_parser(
        "curves",
        help="the N2EX and EPEX bid curves of each year on one price grid (exchange_curves.py)",
    )
    p.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2023],
        help="first and last year (2020 onwards)",
    )
    p = sub.add_parser(
        "tidy",
        help="move results written in the old flat layout into the tidy one (results/README.md)",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="move the files (default: list what would move)",
    )
    p.add_argument(
        "--root", default="", help="another results folder (default: assume_gb/results)"
    )
    p = sub.add_parser(
        "learning", help="the figures of a learning run, from its database"
    )
    p.add_argument(
        "--db",
        nargs="+",
        required=True,
        help="database URI(s) of the learning run(s), e.g. sqlite:///path.db; several are read together",
    )
    p.add_argument(
        "--simulation",
        nargs="+",
        required=True,
        help="simulation id(s); several = repetitions (seeds)",
    )
    p.add_argument(
        "--out",
        default="",
        help="folder for the figures (default: assume_gb/results/learning/<names>)",
    )
    for command, help_text in helps.items():
        p = sub.add_parser(command, help=help_text)
        p.add_argument("--year", type=int, default=2023)
        p.add_argument(
            "--name", default="", help="scenario folder name (default gb_<year>)"
        )
        if command in ("build", "check"):
            p.add_argument("--design", default="status_quo")
            p.add_argument(
                "--tag",
                default="srmc",
                help="tag of the stack export (the merit-order model's version)",
            )
        if command == "build":
            p.add_argument(
                "--no-merge",
                dest="merge",
                action="store_false",
                help="one unit per stack column",
            )
            p.add_argument(
                "--no-intraday",
                action="store_true",
                help="without the forecasts and the intraday cases",
            )
            p.add_argument(
                "--learning-techs", nargs="*", default=list(convert.LEARNING_TECHS), metavar="TECH",
                help="technologies whose units learn their bids in the learning cases (default CCGT; none for no learning cases)",
            )  # fmt: skip
            p.add_argument(
                "--learning-units",
                type=int,
                default=None,
                metavar="N",
                help="only N of those units learn (default: all)",
            )
            p.add_argument(
                "--learning-pick",
                choices=("largest", "cheapest", "portfolio"),
                default="largest",
                help="which units: the N largest, the N cheapest (nearest the margin), or one agent per company (--agent-level) with at least --nbins plants",
            )
            p.add_argument(
                "--agent-level", choices=ownership.AGENT_LEVELS, default=ownership.AGENT_LEVELS[0],
                help="portfolio agents: the company that trades each unit, its BM lead party's group (default), or the one that runs it",
            )  # fmt: skip
            p.add_argument(
                "--portfolio-cfd", action="store_true",
                help=f"portfolio agents: also the cases {convert.LEARNING_CFD_CASE}* in which the CfD units belong to their companies' portfolios "
                     "(default: they stay out and bid their contracts)",
            )  # fmt: skip
            p.add_argument(
                "--nbins",
                type=int,
                default=2,
                help="portfolio agents: cost bins per agent, and the least plants an owner needs (default 2)",
            )
            p.add_argument(
                "--max-markup",
                type=float,
                default=2.0,
                help="portfolio agents: highest mark-up on marginal cost, as a factor (default 2)",
            )
            p.add_argument(
                "--min-portfolio-mw",
                type=float,
                default=500.0,
                help="portfolio agents: the least capacity an owner needs (default 500 MW)",
            )
            p.add_argument(
                "--egl-rate",
                type=float,
                default=0.0,
                help="put the generator levy on the in-scope units at this rate (default off)",
            )
            p.add_argument(
                "--egl-benchmark",
                type=float,
                default=egl.BENCHMARK_2023,
                help="the levy's benchmark price, GBP/MWh (default 75)",
            )
            p.add_argument(
                "--learning-weeks",
                type=int,
                default=4,
                help="length of the training period in weeks (default 4)",
            )
            p.add_argument(
                "--max-bid-price",
                type=float,
                default=None,
                help="highest price a learning unit can bid (default: from the merit order)",
            )
            p.add_argument(
                "--battery-traders", action="store_true",
                help="also the case day_ahead_storage_traders: the battery fleet as one unit per trading company (BM register)",
            )  # fmt: skip
        if command == "run":
            p.add_argument(
                "--years",
                nargs=2,
                type=int,
                metavar=("FROM", "TO"),
                help="several years, with a summary",
            )
            p.add_argument(
                "--case", default=convert.CASE, help="study case (see above)"
            )
            p.add_argument(
                "--csv",
                default="",
                help="folder for the framework's CSV output (default: none)",
            )
            p.add_argument(
                "--save", nargs="?", const=str(paths.results_dir()), default="", metavar="RESULTS",
                help="save the clearing prices into the results folder, prices/<run>.csv (default folder: assume_gb/results)",
            )  # fmt: skip
            p.add_argument(
                "--out",
                default="",
                help="a folder for the clearing prices as <run>_prices.csv, the old flat layout (default: none)",
            )
            p.add_argument(
                "--figures",
                action="store_true",
                help="draw the run's figures (needs --csv)",
            )
            p.add_argument(
                "--db",
                default="",
                help="database URI for the run (default for a learning case: assume_gb/results/db/<run>.db)",
            )
    args = parser.parse_args(argv)
    if args.command == "run" and args.figures and not args.csv:
        parser.error(
            "--figures needs --csv: the figures are drawn from the framework's output"
        )
    commands = {
        "build": build, "check": check, "run": run, "figures": figures, "learning": learning, "variant": variant, "egl": levy,
        "ownership": ownership_command, "agents": agents_command, "penetration": penetration, "validate": validate, "tidy": tidy_command, "curves": curves_command, "thermal": thermal_command,
    }  # fmt: skip
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
