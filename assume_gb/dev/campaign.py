# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Summary of a learning campaign: one row per training run (how its owners did against the
competitive benchmark, how far its policies move the price in the training window and over the
year), and per arm of the campaign (the runs that differ only by seed) the mark-ups each owner
settled on.

    python assume_gb/dev/campaign.py learning_sep_s0 learning_sep_s1 learning_sep_s2
    python assume_gb/dev/campaign.py 'learning_sep4*'          # a glob over the result databases

A run is read from its learning database ``results/db/gb_<year>_<case>.db`` (the evaluation
episodes of ``rl_params``) and its price files ``results/prices/gb_<year>_<case>.csv``,
``..._year.csv`` (the year with the policies of the best evaluation episode) and
``..._year_last.csv`` (with those at the end of training). Money in the framework is price
x MW per half-hour step, an hourly rate: the GBP figures here are halved. The competitive profit
is what the owner's plants earn bidding their floors at the merit-order price.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from assume_gb import compare, paths  # noqa: E402


def evaluations(db: Path) -> pd.DataFrame:
    """The evaluation episodes of a run: profit, competitive profit, reward and actions per owner
    and period."""
    with sqlite3.connect(db) as con:
        columns = [r[1] for r in con.execute("pragma table_info(rl_params)")]
        actions = sorted(c for c in columns if c.startswith("actions_"))
        query = f"select episode, unit, profit, regret, reward, {', '.join(actions)} from rl_params where evaluation_mode = 1"
        return pd.read_sql(query, con).rename(columns={"regret": "competitive"})


def fit(prices: Path, folder: Path) -> dict:
    """Price over the merit order, and R2 and MAE against N2EX, of a saved price file."""
    if not prices.exists():
        return {}
    table = compare.compare(
        pd.read_csv(prices, index_col=0, parse_dates=True)["DA"], folder
    )
    out = {"lift": table.loc["simulated vs scenario_merit_order", "bias"]}
    if "simulated vs n2ex_day_ahead, hourly" in table.index:
        out["r2"] = table.loc["simulated vs n2ex_day_ahead, hourly", "r2"]
        out["mae"] = table.loc["simulated vs n2ex_day_ahead, hourly", "mae"]
    return out


def summarise(cases: list[str], year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = paths.scenarios_dir() / paths.scenario_name(year)
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    runs, owners = [], []
    for case in cases:
        name = f"gb_{year}_{case}"
        e = evaluations(paths.database_file(name))
        per_episode = e.groupby(["episode", "unit"])[
            ["profit", "competitive", "reward"]
        ].sum()
        totals = per_episode.groupby("episode").sum()
        last, best = totals.index.max(), totals["reward"].idxmax()
        final = per_episode.loc[last]
        row = {
            "case": case,
            "best_ep": best,
            "evals": len(totals),
            "above": int((final["reward"] > 0).sum()),
            "owners": len(final),
            "ratio": final["profit"].sum() / final["competitive"].sum(),
            "excess_gbp_m": (final["profit"].sum() - final["competitive"].sum())
            / 2
            / 1e6,
            "reward_first": totals["reward"].iloc[0],
            "reward_last": totals.loc[last, "reward"],
            # the learning can flip between states late: the mean of the last three evaluations
            "reward_last3": totals["reward"].iloc[-3:].mean(),
            "ratio_last3": totals["profit"].iloc[-3:].sum()
            / totals["competitive"].iloc[-3:].sum(),
        }
        for label, suffix in (
            ("window", ""),
            ("year", "_year"),
            ("year_last", "_year_last"),
        ):
            row.update(
                {
                    f"{label}_{k}": v
                    for k, v in fit(
                        paths.prices_file(f"{name}{suffix}"), folder
                    ).items()
                }
            )
        row["trajectory"] = " ".join(
            f"{r:.0f}" for r in totals["reward"]
        )  # reward per evaluation
        runs.append(row)

        # a case no longer in config.yaml is read with the base learning case's mark-up bounds
        params = config.get(case, config.get("learning", {})).get(
            "bidding_strategy_params", {}
        )
        low, high = params.get("min_markup", 1.0), params.get("max_markup", 3.0)
        actions = [c for c in e.columns if c.startswith("actions_")]
        for unit, g in e[e["episode"] == last].groupby("unit"):
            entry = {
                "arm": case.rsplit("_s", 1)[0],
                "case": case,
                "unit": unit,
                "ratio": final.loc[unit, "profit"] / final.loc[unit, "competitive"],
            }
            for a in actions:
                k = a.split("_")[1]
                entry[f"m{k}"] = (low + (g[a] + 1) / 2 * (high - low)).mean()
                entry[f"cap{k}"] = (g[a] > 0.95).mean()
            owners.append(entry)
    return pd.DataFrame(runs).set_index("case"), pd.DataFrame(owners)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "cases", nargs="+", help="case names, or globs over the result databases"
    )
    parser.add_argument("--year", type=int, default=2023)
    args = parser.parse_args()
    cases = []
    for pattern in args.cases:
        found = sorted(
            p.stem.removeprefix(f"gb_{args.year}_")
            for p in paths.results_dir("db").glob(f"gb_{args.year}_{pattern}.db")
        )
        cases += found or [pattern]
    runs, owners = summarise(cases, args.year)
    with pd.option_context(
        "display.width",
        250,
        "display.max_columns",
        40,
        "display.float_format",
        "{:.3f}".format,
    ):
        print(runs.to_string())
        numbers = [c for c in owners.columns if c not in ("arm", "case", "unit")]
        for arm, g in owners.groupby("arm", sort=False):
            print(
                f"\n{arm}: mean over {g['case'].nunique()} seeds (m = mark-up per action, cap = share at the cap)"
            )
            print(g.groupby("unit")[numbers].mean().to_string())
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
