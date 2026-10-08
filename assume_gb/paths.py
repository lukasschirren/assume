# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Where the tool reads from and writes to.

The GB data repository is the environment variable ``GB_DATA_REPO`` or, by default, the sibling
folder of this repository. Only this tool knows that location; the scenario folders it writes do
not.

The scenario folders are written to ``assume_gb/inputs``, inside this local folder and therefore
outside version control. ``examples/inputs`` is the framework's place for the example scenarios it
publishes; a scenario that is to be published is moved there, with a licence file beside every
data file.
"""

import os
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOL_ROOT.parent
GB_REPO_ENV = "GB_DATA_REPO"
GB_REPO_DEFAULT_NAME = "Agent-Based Electricity Pricing GB"


def gb_repo() -> Path:
    """Root of the GB data repository."""
    env = os.environ.get(GB_REPO_ENV)
    path = Path(env) if env else REPO_ROOT.parent / GB_REPO_DEFAULT_NAME
    if not path.is_dir():
        raise FileNotFoundError(
            f"GB data repository not found at {path}. Set {GB_REPO_ENV} to its root."
        )
    return path


def gb_results_dir() -> Path:
    """Folder the merit-order model writes its results and stack exports to."""
    return gb_repo() / "data" / "results"


GB_POWER_DATA_ENV = "GB_POWER_DATA_REPO"
GB_POWER_DATA_DEFAULT_NAME = "gb-power-data"


def gb_power_data() -> Path:
    """Root of gb-power-data, the public GB data package that holds the ownership registers, the
    curated tables and the company rules (``ownership``): the environment variable
    ``GB_POWER_DATA_REPO`` or, by default, the sibling folder ``gb-power-data``. FileNotFoundError
    where it is not (as on the HPC, where each scenario's ``unit_owners.csv`` stands in)."""
    env = os.environ.get(GB_POWER_DATA_ENV)
    path = Path(env) if env else REPO_ROOT.parent / GB_POWER_DATA_DEFAULT_NAME
    if not path.is_dir():
        raise FileNotFoundError(
            f"gb-power-data not found at {path}. Set {GB_POWER_DATA_ENV} to its root."
        )
    return path


def scenarios_dir() -> Path:
    """Folder that holds the GB scenario folders: the ``inputs_path`` to give the framework."""
    return TOOL_ROOT / "inputs"


def scenario_name(year: int) -> str:
    return f"gb_{year}"


# The results of the runs (results/README.md): the raw outputs of a run in a folder per kind, the
# outputs of the analysis commands in a folder named after the command that writes them. Every path
# into the results is made here; the PBS scripts in hpc/ mirror these names.
RAW = ("prices", "db", "csv", "logs")
DERIVED = ("figures", "learning", "validation")


def results_dir(part: str | None = None, root: Path | str | None = None) -> Path:
    """The results folder (``assume_gb/results``, or ``root``), or one of its parts: the raw
    outputs ``prices`` (clearing prices), ``db`` (learning databases), ``csv`` (the framework's
    output) and ``logs``, and the outputs of the commands ``figures``, ``learning`` and
    ``validation``."""
    base = Path(root) if root is not None else TOOL_ROOT / "results"
    if part is None:
        return base
    if part not in RAW + DERIVED:
        raise ValueError(f"no results part {part!r}: one of {RAW + DERIVED}")
    return base / part


def run_name(year: int, case: str, name: str = "") -> str:
    """The name of a run, its scenario folder and study case: ``gb_<year>_<case>``."""
    return f"{name or scenario_name(year)}_{case}"


def prices_file(run: str, root: Path | str | None = None) -> Path:
    """The saved clearing prices of a run: ``results/prices/<run>.csv``."""
    return results_dir("prices", root) / f"{run}.csv"


def database_file(run: str, root: Path | str | None = None) -> Path:
    """The database of a learning run: ``results/db/<run>.db``."""
    return results_dir("db", root) / f"{run}.db"
