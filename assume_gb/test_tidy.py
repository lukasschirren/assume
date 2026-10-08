# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""The results layout (``paths.results_dir``) and ``tidy``, which moves a results folder from the
old flat layout into it: every kind of file to its place, nothing overwritten, every move logged,
the old log folder left as a link, and a second run moving only what arrived in the old layout
since.

    python -m pytest assume_gb
"""

import pytest

from assume_gb import paths, tidy


def _old_layout(root):
    (root / "csv" / "gb_2023_day_ahead" / "figures").mkdir(parents=True)
    (root / "hpc").mkdir()
    (root / "figures" / "gb_2023_day_ahead").mkdir(parents=True)
    files = {
        "gb_2023_day_ahead_prices.csv": "prices",
        "gb_2023_learning_sep_s0_year_last_prices.csv": "year",
        "gb_2023_learning_sep_s0.db": "db",
        "hpc/4275808.pbs-7.OU": "pbs",
        "hpc/gb_2023_learning_sep_s0.log": "log",
        "csv/gb_2023_day_ahead/market_meta.csv": "raw",
        "csv/gb_2023_day_ahead/figures/01_price_year.png": "png",
        "csv/gb_2023_day_ahead/figures/captions.md": "new captions",
        "figures/gb_2023_day_ahead/captions.md": "old captions",
        "README.md": "readme",
    }
    for name, text in files.items():
        (root / name).write_text(text)


def test_the_parts_of_the_results_folder(tmp_path):
    assert paths.results_dir(root=tmp_path) == tmp_path
    assert (
        paths.prices_file("gb_2023_day_ahead", tmp_path)
        == tmp_path / "prices" / "gb_2023_day_ahead.csv"
    )
    assert (
        paths.database_file("gb_2023_learning_s0", tmp_path)
        == tmp_path / "db" / "gb_2023_learning_s0.db"
    )
    assert paths.run_name(2023, "day_ahead") == "gb_2023_day_ahead"
    with pytest.raises(ValueError):
        paths.results_dir("hpc")


def test_tidy_moves_the_old_layout_without_overwriting(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    _old_layout(root)
    moves = tidy.plan(root)
    assert (
        len(moves) == 7 and (root / "gb_2023_day_ahead_prices.csv").exists()
    )  # a plan moves nothing
    moved, skipped = tidy.apply(moves, root)
    assert len(moved) == 6 and [target.name for _, target in skipped] == ["captions.md"]
    assert (root / "prices" / "gb_2023_day_ahead.csv").read_text() == "prices"
    assert (
        root / "prices" / "gb_2023_learning_sep_s0_year_last.csv"
    ).read_text() == "year"
    assert (root / "db" / "gb_2023_learning_sep_s0.db").read_text() == "db"
    assert (root / "logs" / "gb_2023_learning_sep_s0.log").read_text() == "log"
    assert (
        root / "figures" / "gb_2023_day_ahead" / "01_price_year.png"
    ).read_text() == "png"
    # the existing destination is kept, the new file left where it was
    assert (
        root / "figures" / "gb_2023_day_ahead" / "captions.md"
    ).read_text() == "old captions"
    assert (root / "csv" / "gb_2023_day_ahead" / "figures" / "captions.md").exists()
    assert (root / "csv" / "gb_2023_day_ahead" / "market_meta.csv").exists() and (
        root / "README.md"
    ).exists()
    # the old log folder is a link to the new one: jobs submitted before still write their logs there
    assert (root / "hpc").is_symlink() and (
        root / "hpc" / "4275808.pbs-7.OU"
    ).read_text() == "pbs"
    log = (root / "logs" / tidy.MOVES_LOG).read_text().splitlines()
    assert log[0] == "when,from,to" and len(log) == 7
    assert log[1].endswith("gb_2023_day_ahead_prices.csv,prices/gb_2023_day_ahead.csv")


def test_tidy_runs_again_on_what_arrived_since(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    _old_layout(root)
    tidy.apply(tidy.plan(root), root)
    # a job submitted before the change finishes: its prices and database in the old places
    (root / "gb_2023_learning_sep_s1_prices.csv").write_text("late")
    (root / "gb_2023_learning_sep_s1.db").write_text("late db")
    (root / "hpc" / "gb_2023_learning_sep_s1.log").write_text(
        "late log"
    )  # through the link
    moves = tidy.plan(root)
    assert [target.name for _, target in moves] == [
        "gb_2023_learning_sep_s1.csv",
        "gb_2023_learning_sep_s1.db",
        "captions.md",
    ]
    moved, skipped = tidy.apply(moves, root)
    assert len(moved) == 2 and len(skipped) == 1
    assert (root / "logs" / "gb_2023_learning_sep_s1.log").read_text() == "late log"
    assert len((root / "logs" / tidy.MOVES_LOG).read_text().splitlines()) == 9
