# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The portfolio agents of a scenario folder from its own ``unit_owners.csv``, without the GB
data repository (on the HPC): the files of a learning set, ``powerplant_units_learning_<set>.csv``
and ``unit_operators_learning_<set>.csv``, beside the folder's other files. A study case uses them
with ``--set powerplant_units=... --set unit_operators=...``.

    python assume_gb/dev/learning_sets.py --year 2023 own                        # as build would
    python assume_gb/dev/learning_sets.py --year 2023 fringe --techs CCGT "Hard Coal" Wind --least-units 1
    python assume_gb/dev/learning_sets.py --year 2023 --check                     # the 5 Oct files

The agents are chosen as ``build`` chooses them (``convert.write_learning``, ``learning_pick``
"portfolio"): the operator of every named plant and wind farm (``ownership.operators_from_view``
restricted to ``ownership.unit_kind`` "plant" and "farm", as ``ownership.operators``) with at least
``--least-units`` units of ``--techs`` and ``--min-mw`` of their capacity becomes one agent; its
units keep their day-ahead strategies. ``--check`` rebuilds the folder's existing learning files
from their own operators and compares them byte for byte (up to line ends).
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from assume_gb import convert, ownership, paths  # noqa: E402


def learning_owners(folder: Path, names: list[str]) -> pd.Series:
    """The operator of every named plant and wind farm of the folder, by unit name."""
    operators = ownership.operators_from_view(folder)
    if operators is None:
        raise SystemExit(f"{folder} has no unit_owners.csv")
    kinds = pd.Series({n: ownership.unit_kind(n, "") for n in names})
    return operators[operators.index.isin(kinds.index[kinds.isin(["plant", "farm"])])]


def write_set(
    folder: Path,
    label: str,
    owners: pd.Series,
    techs: list[str],
    least_units: int,
    min_mw: float,
) -> list[str]:
    table = pd.read_csv(
        folder / "powerplant_units.csv", float_precision="round_trip"
    )  # exact floats
    with tempfile.TemporaryDirectory() as tmp:
        units, agents = convert.write_learning(
            Path(tmp),
            table,
            tuple(techs),
            learning_pick="portfolio",
            owners=owners,
            nbins=least_units,
            min_portfolio_mw=min_mw,
        )
        suffix = f"_{label}" if label else ""
        for name in ("powerplant_units_learning", "unit_operators_learning"):
            shutil.copyfile(Path(tmp) / f"{name}.csv", folder / f"{name}{suffix}.csv")
    learning = table[table["name"].isin(units)].assign(
        agent=lambda t: t["name"].map(owners)
    )
    summary = learning.pivot_table(
        index="agent",
        columns="technology",
        values="max_power",
        aggfunc=["count", "sum"],
        fill_value=0,
    )
    print(
        f"{folder.name}: learning set {label or '(default files)'}: {len(agents)} agents, {len(units)} units"
    )
    print(summary.round(0).to_string())
    return agents


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "label", nargs="?", default="", help="the set's name, the suffix of its files"
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--techs", nargs="+", default=list(convert.LEARNING_TECHS))
    parser.add_argument(
        "--least-units",
        type=int,
        default=2,
        help="the least units of the techs an agent needs (build: --nbins)",
    )
    parser.add_argument(
        "--min-mw",
        type=float,
        default=500.0,
        help="the least capacity of the techs an agent needs",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild the existing learning files from their own operators",
    )
    args = parser.parse_args()
    folder = paths.scenarios_dir() / paths.scenario_name(args.year)
    names = pd.read_csv(folder / "powerplant_units.csv")["name"].tolist()
    if args.check:
        existing = pd.read_csv(folder / "powerplant_units_learning.csv")
        agents = set(pd.read_csv(folder / "unit_operators_learning.csv")["name"])
        owners = existing.set_index("name")["unit_operator"]
        owners = owners[owners.isin(agents)]
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("powerplant_units.csv",):
                shutil.copyfile(folder / name, Path(tmp) / name)
            write_set(Path(tmp), "", owners, args.techs, args.least_units, args.min_mw)
            same = all(  # the laptop (Windows) writes CRLF line ends, the HPC LF
                (Path(tmp) / f).read_bytes().replace(b"\r\n", b"\n")
                == (folder / f).read_bytes().replace(b"\r\n", b"\n")
                for f in (
                    "powerplant_units_learning.csv",
                    "unit_operators_learning.csv",
                )
            )
        print(
            "identical to the folder's files"
            if same
            else "DIFFERENT from the folder's files"
        )
        return 0 if same else 1
    write_set(
        folder,
        args.label,
        learning_owners(folder, names),
        args.techs,
        args.least_units,
        args.min_mw,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
