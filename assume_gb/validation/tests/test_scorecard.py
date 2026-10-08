# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The scorecard: every criterion on the headline window, the benchmark comparisons there only,
skills against the naive forecast, the seed spread of an ensemble, and the exports."""

import numpy as np
import pandas as pd
import pytest

from assume_gb.validation import scorecard as C
from assume_gb.validation.contract import ValidationData
from assume_gb.validation.tests import synthetic


@pytest.fixture(scope="module")
def three_models(observed, exog) -> ValidationData:
    """A competitive run formed on the outturn, an agent-based ensemble 3 GBP/MWh above it with
    noise, and the naive forecast; tuning in May, the headline window June to August."""
    competitive = synthetic.prices(exog, "outturn", seed=1)
    rng = np.random.default_rng(9)
    noise = rng.normal(0.0, 5.0, (len(competitive), 3))
    seeds = pd.DataFrame(
        competitive.to_numpy()[:, None] + 3.0 + noise, index=competitive.index
    )
    return ValidationData(
        observed,
        {"competitive": competitive, "abm": seeds},
        exog,
        periods={
            "calibration": ("2023-05-01", "2023-05-31"),
            "headline": ("2023-06-01", "2023-08-31"),
            "year": ("2023-05-01", "2023-08-31"),
        },
    )


@pytest.fixture(scope="module")
def cards(three_models):
    return C.scorecards(three_models, battery=C.value.Battery(power=1.0, energy=2.0))


def test_the_headline_card_holds_every_criterion(cards, three_models):
    card = cards["headline"]
    assert not card.in_sample and card.notes == []
    assert card.values.columns.tolist() == ["observed", "naive", "competitive", "abm"]
    assert set(card.metrics.criterion) == set(range(1, 9))
    assert set(card.metrics.direction) <= set(C.DIRECTIONS)
    values, skill = card.values, card.skill
    assert values.loc["rmae", "naive"] == pytest.approx(1.0)
    assert (skill["naive"].dropna() == 0).all()
    # the competitive run, formed on the outturn with the right coefficients, beats the naive
    # forecast by far, and significantly
    assert (
        skill.loc["mae", "competitive"] > 0.5
        and card.stars.loc["mae", "competitive"] == "***"
    )
    assert values.loc["dm_naive", "competitive"] < 0.001
    assert np.isnan(skill.loc["rmae", "competitive"]) and np.isnan(
        skill.loc["dm_naive", "abm"]
    )
    # the ensemble sits 3 above the competitive run: its bias and markups say so
    assert values.loc["bias", "abm"] - values.loc[
        "bias", "competitive"
    ] == pytest.approx(3.0, abs=0.1)
    assert values.loc["markup_peak", "abm"] == pytest.approx(3.0, abs=1.5)
    assert values.loc["markup_peak", "competitive"] == pytest.approx(0.0, abs=1.5)
    assert values.loc["crps", "naive"] == pytest.approx(values.loc["mae", "naive"])
    assert np.isnan(values.loc["crps", "competitive"]) and values.loc["crps", "abm"] > 0
    # benchmarks and targets sit in the observed column
    assert values.loc["profile_r_hour", "observed"] > 0.9
    assert values.loc["negative_share", "observed"] == pytest.approx(
        C.metrics.negative_share(three_models.sample("headline").observed)
    )
    assert np.isfinite(values.loc["wind_effect_error", "observed"])
    assert values.loc["coefficients_consistent", "competitive"] >= 0.8


def test_other_windows_leave_out_the_benchmark_comparisons(cards):
    assert list(cards) == ["headline", "calibration", "year"]
    for name in ("calibration", "year"):
        card = cards[name]
        assert card.in_sample
        assert "naive" not in card.values.columns
        assert not {"rmae", "dm_naive", "dm_competitive"} & set(card.values.index)
        assert card.skill.isna().all().all() and (card.stars == "").all().all()


def test_the_seed_spread_beside_the_seed_mean(cards):
    seeds = cards["headline"].seeds.set_index("metric")
    assert set(seeds.model) == {"abm"} and (seeds.seeds == 3).all()
    assert (seeds["min"] <= seeds["median"]).all() and (
        seeds["median"] <= seeds["max"]
    ).all()
    # averaging the seeds smooths their noise: the seed mean has a lower MAE than any one seed
    assert seeds.loc["mae", "seed_mean"] < seeds.loc["mae", "min"]


def test_exports(cards, tmp_path):
    card = cards["headline"]
    path = card.to_csv(tmp_path / "scorecard.csv")
    table = pd.read_csv(path)
    assert {
        "window",
        "in_sample",
        "criterion",
        "metric",
        "direction",
        "series",
        "value",
        "skill",
    } <= set(table.columns)
    assert len(table) == card.values.notna().sum().sum()
    assert (tmp_path / "scorecard_seeds.csv").exists()
    latex = card.to_latex(tmp_path / "scorecard.tex")
    assert (tmp_path / "scorecard.tex").read_text(encoding="utf-8") == latex
    for part in (
        r"\toprule",
        r"\midrule",
        r"\bottomrule",
        r"$^{***}$",
        "competitive",
        r"$\downarrow$",
    ):
        assert part in latex
    assert latex.count(r"\\") == len(card.values) + 1  # header and one line per metric
    assert "out of sample" in latex and "in-sample" in cards["year"].to_latex()


def test_skill_follows_the_direction_of_the_metric():
    assert C.skill(5.0, 10.0, C.LOWER) == pytest.approx(0.5)
    assert C.skill(-2.0, 4.0, C.ZERO) == pytest.approx(0.5)
    assert C.skill(0.9, 0.6, C.ONE) == pytest.approx(0.75)
    assert C.skill(0.8, 0.6, C.HIGHER) == pytest.approx(0.5)
    assert C.skill(0.02, 0.0, C.TARGET, target=0.01) == pytest.approx(0.0)
    assert C.significance(0.0004) == "***" and C.significance(0.2) == ""
