# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Regression harness for changes to the framework's bookkeeping: run the example scenarios (and
the first GB week) into sqlite databases, which keep full precision, under a tag, and compare two
tags table by table. Use it before and after a change that must leave results unchanged:

    git stash                                            # the code before the change
    python assume_gb/dev/regress.py run before "" 4      # tag, name filter, parallel jobs
    git stash pop
    python assume_gb/dev/regress.py run after "" 4
    python assume_gb/dev/regress.py compare before after

Every case runs in a process of its own with ``PYTHONHASHSEED=0``: without a fixed hash seed two
runs of the same code already differ, because the order in which the agents act, and with it the
random choice among equal-priced offers, follows the iteration order of sets. The learning case
(``example_02a/tiny``) differs from run to run regardless, and
``example_01c/dam_with_complex_opt_clearing`` fails in the stock code (``KeyError:
'min_acceptance_ratio'``). The databases land in ``assume_gb/dev/regress_out/<tag>/``.

    python regress.py one <tag> <inputs> <scenario> <case>      (used internally, one process per case)
"""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
OUT = HERE / "regress_out"
REPO = HERE.parents[1]
EXAMPLES = "examples/inputs"
SHORT = HERE / "regress_inputs"

CASES = [
    (EXAMPLES, "example_01a", "base"),
    (EXAMPLES, "example_01a", "dam"),
    (EXAMPLES, "example_01a", "base_with_exchanges"),
    (EXAMPLES, "example_01a", "dam_with_complex_clearing"),
    (EXAMPLES, "example_01b", "base"),
    (EXAMPLES, "example_01c", "eom_only"),
    (EXAMPLES, "example_01c", "eom_and_crm"),
    (EXAMPLES, "example_01c", "dam_with_complex_opt_clearing"),
    (EXAMPLES, "example_01d", "base"),
    (EXAMPLES, "example_01d", "zonal_case"),
    (EXAMPLES, "example_01d", "nodal_case"),
    (EXAMPLES, "example_01f", "eom_case"),
    (EXAMPLES, "example_01f", "ltm_case"),
    (EXAMPLES, "example_01h", "eom"),
    (EXAMPLES, "example_02a", "tiny"),
    (EXAMPLES, "example_04a", "base"),
    (str(SHORT), "example_03", "base_case_2019"),
    (str(SHORT), "example_03", "eom_crm_case_2019"),
    (str(SHORT), "example_03", "dam_case_2019"),
    (str(SHORT), "example_03", "base_case_2019_with_DSM"),
    ("assume_gb/inputs", "gb_2023", "day_ahead_week"),
]

# shortened copies of the long example: (case, new end date)
SHORTEN = {
    "base_case_2019": "2019-01-21 00:00",
    "eom_crm_case_2019": "2019-01-11 00:00",
    "dam_case_2019": "2019-01-06 00:00",
}


def prepare_short_inputs():
    target = SHORT / "example_03"
    if target.exists():
        return
    shutil.copytree(REPO / EXAMPLES / "example_03", target)
    config = yaml.safe_load((target / "config.yaml").read_text())
    for case, end in SHORTEN.items():
        config[case]["end_date"] = end
    (target / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def one(tag, inputs, scenario, case):
    import os

    os.environ["NON_INTERACTIVE"] = "1"
    from assume import World
    from assume.scenario.loader_csv import load_scenario_folder, run_learning

    db = OUT / tag / f"{scenario}__{case}.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()
    world = World(database_uri=f"sqlite:///{db.as_posix()}", log_level="ERROR")
    load_scenario_folder(world, inputs_path=inputs, scenario=scenario, study_case=case)
    if world.learning_mode:
        run_learning(world)
    start = time.perf_counter()
    world.run()
    seconds = time.perf_counter() - start
    (db.with_suffix(".json")).write_text(json.dumps({"seconds": seconds}))
    print(f"{scenario}/{case}: {seconds:.1f}s")


def run_case(tag, inputs, scenario, case):
    t0 = time.perf_counter()
    # A fixed hash seed makes the order of sets, and with it the order in which the agents act,
    # the same in every run. Without it two runs of the same code already differ (which of
    # several equal-priced offers is accepted depends on the order the bids arrive in).
    env = dict(os.environ, PYTHONHASHSEED="0")
    result = subprocess.run(
        [sys.executable, str(Path(__file__)), "one", tag, inputs, scenario, case],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=env,
    )
    lines = [
        line
        for line in (result.stdout + result.stderr).splitlines()
        if line.startswith(f"{scenario}/")
    ]
    status = (
        lines[-1]
        if lines
        else "FAILED: " + (result.stderr.strip().splitlines() or ["?"])[-1][:300]
    )
    return f"[{time.perf_counter() - t0:6.1f}s wall] {status}"


def run(tag, name_filter="", jobs=1):
    from concurrent.futures import ThreadPoolExecutor

    prepare_short_inputs()
    cases = [c for c in CASES if not name_filter or name_filter in f"{c[1]}/{c[2]}"]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for status in pool.map(lambda c: run_case(tag, *c), cases):
            print(status, flush=True)


def tables(db):
    with sqlite3.connect(db) as con:
        names = [
            r[0]
            for r in con.execute("select name from sqlite_master where type='table'")
        ]
        return {name: pd.read_sql(f'select * from "{name}"', con) for name in names}


def normalise(df):
    """Rows in a canonical order: agents write in a different order from run to run. A numeric
    ``index`` column is only the row counter of a write and is dropped."""
    if "index" in df.columns and pd.api.types.is_numeric_dtype(df["index"]):
        df = df.drop(columns="index")
    text = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in df.columns if c not in text]
    keyed = pd.concat([df[text].astype(str), df[numeric]], axis=1)
    return df.loc[keyed.sort_values(text + numeric, kind="stable").index].reset_index(
        drop=True
    )


def compare(tag_a, tag_b):
    worst = 0.0
    for db_a in sorted((OUT / tag_a).glob("*.db")):
        db_b = OUT / tag_b / db_a.name
        if (
            not db_b.exists()
            or not db_a.with_suffix(".json").exists()
            or not db_b.with_suffix(".json").exists()
        ):
            print(f"{db_a.stem}: not completed in both")
            continue
        a, b = tables(db_a), tables(db_b)
        ta = json.loads(db_a.with_suffix(".json").read_text())["seconds"]
        tb = json.loads(db_b.with_suffix(".json").read_text())["seconds"]
        notes = []
        for name in sorted(set(a) | set(b)):
            if name not in a or name not in b:
                notes.append(f"{name}: only in one")
                continue
            x, y = normalise(a[name]), normalise(b[name])
            if x.shape != y.shape or list(x.columns) != list(y.columns):
                notes.append(f"{name}: shape {x.shape} vs {y.shape}")
                continue
            if x.empty:
                continue
            numeric = [
                c
                for c in x.columns
                if pd.api.types.is_numeric_dtype(x[c])
                and pd.api.types.is_numeric_dtype(y[c])
            ]
            other = [c for c in x.columns if c not in numeric]
            if other and not x[other].astype(str).equals(y[other].astype(str)):
                bad = [
                    c for c in other if not x[c].astype(str).equals(y[c].astype(str))
                ]
                notes.append(f"{name}: text columns differ {bad}")
            if numeric:
                xa, ya = (
                    x[numeric].to_numpy(dtype=float),
                    y[numeric].to_numpy(dtype=float),
                )
                both_nan = np.isnan(xa) & np.isnan(ya)
                diff = np.where(both_nan, 0.0, np.abs(xa - ya))
                if np.isnan(diff).any():
                    notes.append(f"{name}: NaN in one only")
                d = float(np.nanmax(diff)) if diff.size else 0.0
                worst = max(worst, d)
                if d > 0:
                    col = numeric[int(np.nanargmax(np.nanmax(diff, axis=0)))]
                    notes.append(f"{name}: max |diff| {d:.3g} (column {col})")
        rows = sum(len(t) for t in a.values())
        print(
            f"{db_a.stem:55s} {ta:7.1f}s -> {tb:7.1f}s  x{ta / tb:4.2f}  rows {rows:8d}  "
            + ("IDENTICAL" if not notes else "; ".join(notes))
        )
    print(f"largest numeric difference over all tables: {worst:.3g}")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "one":
        one(*sys.argv[2:6])
    elif command == "run":
        run(
            sys.argv[2],
            sys.argv[3] if len(sys.argv) > 3 else "",
            int(sys.argv[4]) if len(sys.argv) > 4 else 1,
        )
    elif command == "compare":
        compare(sys.argv[2], sys.argv[3])
