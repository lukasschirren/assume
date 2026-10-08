# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The report writes every scorecard, table and figure into one folder, and names what it leaves
out instead of stopping."""

from assume_gb.validation import report

TABLES = (
    "scorecard_headline.csv",
    "scorecard_headline.tex",
    "scorecard_headline_seeds.csv",
    "scorecard_calibration.csv",
    "scorecard_year.csv",
    "weeks.csv",
    "regressions.csv",
    "regressions_outturn.csv",
    "attenuation.csv",
    "binned_wind.csv",
    "binned_solar.csv",
    "markups.csv",
    "capture_prices.csv",
    "storage_daily.csv",
    "storage.csv",
    "notes.txt",
)
FIGURES = (
    "fig1_weeks",
    "fig2_duration",
    "fig3_error_heatmap",
    "fig4_coefficients",
    "fig5_markups",
    "fig6_scorecard",
    "figS1_scatter",
    "figS2_taylor",
    "figS3_profiles",
    "figS4_pit",
    "figS5_price_vs_residual_demand",
)


def test_the_report_writes_every_piece(ensemble_data, tmp_path):
    written = {path.name for path in report.write(ensemble_data, tmp_path)}
    assert set(TABLES) <= written
    for stem in FIGURES:
        assert {f"{stem}.pdf", f"{stem}.png"} <= written
        assert (tmp_path / f"{stem}.png").stat().st_size > 0
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == ""


def test_what_cannot_be_made_is_noted(ensemble_data, tmp_path):
    without = ensemble_data.exog.drop(columns=["wind_tx_forecast", "tsd_forecast"])
    data = type(ensemble_data)(
        ensemble_data.observed,
        {"competitive": ensemble_data.models["competitive"]},
        without,
        periods=dict(ensemble_data.periods),
    )
    written = {path.name for path in report.write(data, tmp_path, draw=False)}
    assert "binned_wind.csv" not in written and "scorecard_headline.csv" in written
    notes = (tmp_path / "notes.txt").read_text(encoding="utf-8")
    assert "binned wind effects" in notes and "assume_gb penetration" in notes
