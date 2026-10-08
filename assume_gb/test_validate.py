# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""The validation's command, ``python -m assume_gb validate``, end to end on a stand-in for a
scenario folder and its saved runs (the fixture of the adapters' tests).

    python -m pytest assume_gb
"""

from assume_gb import __main__ as cli

# the stand-in scenario folders
from assume_gb.validation.tests.test_adapters import layout  # noqa: F401


def test_validate_writes_the_scorecards_and_tables(layout, tmp_path, capsys):  # noqa: F811
    out = tmp_path / "out"
    argv = [
        "validate",
        "--year",
        "2023",
        "--arm",
        "learning_test",
        "--no-figures",
        "--out",
        str(out),
    ]
    assert cli.main(argv) == 0
    for name in (
        "scorecard_headline.csv",
        "scorecard_headline.tex",
        "scorecard_calibration.csv",
        "scorecard_year.csv",
        "storage.csv",
        "notes.txt",
    ):
        assert (out / name).exists(), name
    printed = capsys.readouterr().out
    assert (
        str(out) in printed and "left out" in printed
    )  # two weeks: no complete week, no regression
