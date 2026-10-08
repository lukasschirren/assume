# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Moves a results folder from the old flat layout into the tidy one (``results/README.md``,
``paths.results_dir``), without overwriting anything:

    <run>_prices.csv          ->  prices/<run>.csv
    <run>.db                  ->  db/<run>.db
    hpc/<file>                ->  logs/<file>        hpc is then a link to logs
    csv/<run>/figures/<file>  ->  figures/<run>/<file>

It is safe to run again: jobs submitted before the change still write the old layout when they
finish, and a copy of the results elsewhere (``--root``) can be tidied the same way. A file whose
destination exists is left where it is and reported. Every move is appended to
``logs/tidy_moves.csv`` (when, from, to), from which it can be traced or undone.

    python -m assume_gb tidy             what would move
    python -m assume_gb tidy --apply     move
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

from assume_gb import paths

OLD_LOGS = "hpc"
MOVES_LOG = "tidy_moves.csv"


def plan(root: Path | str | None = None) -> list[tuple[Path, Path]]:
    """The moves that bring the results folder ``root`` into the tidy layout: (from, to) pairs."""
    root = paths.results_dir(root=root)
    moves: list[tuple[Path, Path]] = []
    for path in sorted(root.glob("*_prices.csv")):
        moves.append(
            (path, paths.prices_file(path.name.removesuffix("_prices.csv"), root))
        )
    for path in sorted(root.glob("*.db")):
        moves.append((path, paths.database_file(path.stem, root)))
    old_logs = root / OLD_LOGS
    if old_logs.is_dir() and not old_logs.is_symlink():
        for path in sorted(p for p in old_logs.rglob("*") if p.is_file()):
            moves.append(
                (path, paths.results_dir("logs", root) / path.relative_to(old_logs))
            )
    for figures in sorted(root.glob("csv/*/figures")):
        run = figures.parent.name
        for path in sorted(p for p in figures.rglob("*") if p.is_file()):
            moves.append(
                (
                    path,
                    paths.results_dir("figures", root)
                    / run
                    / path.relative_to(figures),
                )
            )
    return moves


def apply(
    moves: list[tuple[Path, Path]], root: Path | str | None = None
) -> tuple[list, list]:
    """Moves each file of ``moves`` unless its destination exists; records the moves in
    ``logs/tidy_moves.csv``, removes the folders it emptied and replaces the old log folder with a
    link to the new one. Returns (moved, skipped)."""
    root = paths.results_dir(root=root)
    moved, skipped = [], []
    for source, target in moves:
        if target.exists():
            skipped.append((source, target))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, target)
        moved.append((source, target))
    if moved:
        log = paths.results_dir("logs", root) / MOVES_LOG
        log.parent.mkdir(parents=True, exist_ok=True)
        new = not log.exists()
        stamp = datetime.now().isoformat(timespec="seconds")
        with log.open("a", encoding="utf-8") as file:
            if new:
                file.write("when,from,to\n")
            for source, target in moved:
                file.write(
                    f"{stamp},{source.relative_to(root)},{target.relative_to(root)}\n"
                )
    for figures in root.glob("csv/*/figures"):
        _remove_empty(figures)
    old_logs = root / OLD_LOGS
    if old_logs.is_dir() and not old_logs.is_symlink():
        _remove_empty(old_logs)
        if not old_logs.exists():
            paths.results_dir("logs", root).mkdir(parents=True, exist_ok=True)
            os.symlink(
                "logs", old_logs
            )  # PBS output and live logs of jobs submitted before
    return moved, skipped


def _remove_empty(folder: Path) -> None:
    """Removes ``folder`` and its sub-folders if they hold no file."""
    for sub in sorted(
        (p for p in folder.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        if not any(sub.iterdir()):
            sub.rmdir()
    if folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()
