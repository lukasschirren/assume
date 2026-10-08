# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The figures render with the Agg backend (conftest) and keep their conventions: fixed model
colours, a symmetric diverging scale centred at zero, weeks chosen by rule with their reasons,
PDF and PNG at 300 dpi."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from PIL import Image

from assume_gb.validation import figures as F
from assume_gb.validation import mechanisms, scorecard, style
from assume_gb.validation.contract import ValidationData
from assume_gb.validation.tests import synthetic


@pytest.fixture(scope="module")
def models(observed, exog) -> ValidationData:
    competitive = synthetic.prices(exog, "outturn", seed=1)
    rng = np.random.default_rng(11)
    seeds = pd.DataFrame(
        competitive.to_numpy()[:, None] + 3.0 + rng.normal(0, 5, (len(competitive), 3)),
        index=competitive.index,
    )
    return ValidationData(
        observed,
        {"competitive": competitive, "abm": seeds},
        exog,
        periods={
            "calibration": ("2023-05-01", "2023-05-31"),
            "headline": ("2023-06-01", "2023-08-31"),
        },
        regimes={
            "June": ("2023-06-01", "2023-06-30"),
            "July and August": ("2023-07-01", "2023-08-31"),
        },
    )


def _check(result, panels: int | None = None) -> Figure:
    fig, ax = result
    assert isinstance(fig, Figure)
    if panels is not None:
        assert np.size(ax) == panels
    fig.canvas.draw()  # render with Agg
    return fig


def test_weeks_are_chosen_by_rule_and_drawn(models):
    selection = F.select_weeks(models)
    assert selection.rule.tolist() == list(F.WEEK_RULES) and selection.week.is_unique
    assert (selection.week.dt.dayofweek == 0).all() and selection.week.dt.tz is not None
    assert (selection.week >= pd.Timestamp("2023-06-01", tz=models.tz)).all()
    assert selection.reason.str.len().gt(10).all()
    assert "MAE of ABM" in selection.reason.iloc[0]
    _check(F.plot_weeks(models, selection), panels=4)


def test_duration_curve_keeps_the_model_colours(models):
    fig = _check(F.plot_duration_curve(models))
    colours = {line.get_label(): line.get_color() for line in fig.axes[0].get_lines()}
    assert (
        colours["ABM"] == style.COLOURS["abm"]
        and colours["Observed"] == style.COLOURS["observed"]
    )
    assert fig.axes[0].get_yscale() == "symlog"
    assert any("W1" in text.get_text() for text in fig.axes[0].texts)


def test_error_heatmap_is_centred_and_symmetric(models):
    fig = _check(F.plot_error_heatmap(models), panels=2)
    norm = fig.axes[0].collections[0].norm
    assert norm.vcenter == 0 and norm.vmin == -norm.vmax


def test_mechanism_market_power_and_scorecard_figures(models):
    regressions = mechanisms.compare_regressions(models)
    ranges = F.causal_ranges("NordPool", models)
    whole = F.causal_ranges("NordPool")
    # over the penetration of the window only, so within the range of the whole curve
    assert (
        whole["wind"][0]
        <= ranges["wind"][0]
        < ranges["wind"][1]
        <= whole["wind"][1]
        < 0
    )
    assert F.causal_ranges("NordPool", band=True)["wind"][0] < whole["wind"][0]
    _check(F.plot_coefficients(regressions, ranges), panels=len(mechanisms.TERMS) * 3)
    _check(F.plot_markups(mechanisms.markups(models, include_competitive=True)))
    _check(F.plot_scorecard(scorecard.scorecard(models)))


def test_supplementary_figures(models):
    _check(F.plot_scatter(models), panels=2)
    _check(F.plot_taylor(models))
    _check(F.plot_profiles(models), panels=2)
    _check(F.plot_pit(models, "abm"), panels=2)
    _check(F.plot_price_vs_residual_demand(models), panels=2)
    with pytest.raises(ValueError):
        F.plot_pit(models, "competitive")


def test_save_writes_pdf_and_png_at_300_dpi(models, tmp_path):
    fig, _ = F.plot_duration_curve(models)
    paths = style.save(fig, tmp_path / "duration")
    assert [p.suffix for p in paths] == [".pdf", ".png"] and all(
        p.stat().st_size > 0 for p in paths
    )
    width, _ = Image.open(paths[1]).size
    assert 0.8 * style.SINGLE * style.DPI < width < 1.2 * style.SINGLE * style.DPI
