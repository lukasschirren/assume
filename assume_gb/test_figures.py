# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The figures of a run: the reading of the framework's output, the figures and the viewer, on the
first week of a built scenario (two cases: storage, and day-ahead with intraday), and the learning
figures on a synthetic learning database. Skipped for runs when no scenario folder is built.

    python -m pytest assume_gb
"""

import base64
import json
import zlib

import numpy as np
import pandas as pd
import pytest

from assume_gb import (
    compare,
    convert,
    figures,
    learning_figures,
    paths,
    results,
    viewer,
)
from assume_gb import style as S

BUILT = sorted(
    p.name for p in paths.scenarios_dir().glob("gb_*") if (p / "config.yaml").exists()
)
needs_scenario = pytest.mark.skipif(
    not BUILT, reason="no scenario folder is built (python -m assume_gb build)"
)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The CSV output of two one-week runs of the latest built year, with the full-precision prices
    beside it as ``run --csv`` writes them."""
    out = tmp_path_factory.mktemp("runs")
    name = BUILT[-1]
    loaded = {}
    for case in (convert.STORAGE_WEEK_CASE, convert.INTRADAY_WEEK_CASE):
        world, _ = compare.run_scenario(name, case, csv_path=str(out))
        compare.clearing_prices(world).rename_axis("datetime").to_csv(
            out / f"{name}_{case}" / "clearing_prices.csv"
        )
        loaded[case] = results.load(out / f"{name}_{case}")
    return loaded


@needs_scenario
def test_generation_adds_up_to_the_demand_served(runs):
    for run in runs.values():
        # five significant digits in the CSV output: 0.5 MW on 40 GW
        assert (run.generation.sum(axis=1) - run.gb_demand).abs().max() < 1.0
        assert (run.exports >= -1.0).all() and (
            run.exports <= run.export_capacity + 1.0
        ).all()


@needs_scenario
def test_every_clearing_respects_its_order_book(runs):
    for run in runs.values():
        for market_id in run.markets:
            assert figures.merit_order_check(run, market_id)["periods_violated"] == 0
    # and the check sees a violation: an offer below the price that is rejected
    run = runs[convert.STORAGE_WEEK_CASE]
    orders = run.orders.copy()
    cheap = orders.index[
        (orders.market_id == convert.MARKET_ID)
        & (orders.volume > 0)
        & (orders.price < -100)
    ][0]
    run.orders.loc[cheap, "accepted_volume"] = 0.0
    try:
        assert (
            figures.merit_order_check(run, convert.MARKET_ID)["periods_violated"] == 1
        )
    finally:
        run.orders = orders


@needs_scenario
def test_every_period_has_a_price_setting_order(runs):
    for run in runs.values():
        for market_id in run.markets:
            marginal = run.marginal(market_id)
            assert marginal.unit_id.notna().all()
            np.testing.assert_allclose(
                marginal.price,
                run.prices[market_id],
                atol=0.02 + 1e-4 * run.prices[market_id].abs().max(),
            )


@needs_scenario
def test_figures_and_viewer_are_written(runs, tmp_path):
    storage, intraday = (
        runs[convert.STORAGE_WEEK_CASE],
        runs[convert.INTRADAY_WEEK_CASE],
    )
    written = {p.stem for p in figures.make(storage, tmp_path / "storage")}
    assert {
        "00_summary",
        "02_price_duration",
        "07_generation_mix",
        "10_supply_curves",
        "12_storage",
    } <= written
    written = {
        p.stem for p in figures.make(intraday, tmp_path / "intraday", baseline=storage)
    }
    assert {"13_intraday", "14_vs_baseline"} <= written and "12_storage" not in written
    # the headline numbers are a table, every figure has its caption beside it
    assert (tmp_path / "intraday" / "00_summary.md").exists()
    captions = S.read_captions(tmp_path / "intraday" / S.CAPTIONS)
    assert set(captions) == {p.stem for p in (tmp_path / "intraday").glob("*.png")}
    assert all(text.startswith("**") for text in captions.values())
    nums = json.loads(
        (tmp_path / "intraday" / "numbers.json").read_text(encoding="utf-8")
    )
    assert set(nums["merit_order_check"]) == {
        convert.MARKET_ID,
        convert.INTRADAY_MARKET_ID,
    }
    page = viewer.write(intraday, tmp_path / "intraday")
    assert page.stat().st_size > 4e6  # Plotly.js is inlined


@needs_scenario
def test_viewer_unpacks_to_the_order_book(runs):
    run = runs[convert.INTRADAY_WEEK_CASE]
    data = viewer.payload(run)
    units = [u["id"] for u in data["units"]]
    n = len(run.index)
    for market_id in run.markets:
        book = data["books"][market_id]
        price = np.frombuffer(
            zlib.decompress(base64.b64decode(book["price"])), "<i2"
        ).reshape(-1, n)
        volume = np.frombuffer(
            zlib.decompress(base64.b64decode(book["volume"])), "<i2"
        ).reshape(-1, n)
        t = 100
        packed = sorted(
            (units[u], p / 10, v * scale)
            for u, p, v, scale in zip(
                book["slots"], price[:, t], volume[:, t], book["vscale"]
            )
            if p != viewer.MISSING and v != 0
        )
        orders = run.orders[
            (run.orders.market_id == market_id)
            & (run.orders.start_time == run.index[t])
            & (run.orders.volume != 0)
        ]
        original = sorted(zip(orders.unit_id, orders.price, orders.volume))
        assert [u for u, *_ in packed] == [u for u, *_ in original]
        np.testing.assert_allclose(
            [p for _, p, _ in packed], [p for _, p, _ in original], atol=0.051
        )
        np.testing.assert_allclose(
            [v for *_, v in packed], [v for *_, v in original], rtol=1e-4, atol=0.5
        )


def _learning_db(path, seeds=2, episodes=6, units=("pp_1", "pp_2")):
    from sqlalchemy import create_engine

    rng = np.random.default_rng(0)
    times = pd.date_range("2023-01-01", periods=48, freq="30min")
    params, grad, meta = [], [], []
    for s in range(seeds):
        sim = f"learning_seed{s}"
        for ep in range(1, episodes + 1):
            meta.append(
                {
                    "simulation": sim,
                    "episode": ep,
                    "eval_episode": 1,
                    "evaluation_mode": 0,
                    "learning_mode": 1,
                }
            )
            for mode in (0, 1) if ep % 2 == 0 else (0,):
                if mode:
                    meta.append(
                        {
                            "simulation": sim,
                            "episode": ep,
                            "eval_episode": ep // 2,
                            "evaluation_mode": 1,
                            "learning_mode": 1,
                        }
                    )
                for u in units:
                    for t in times:
                        params.append(
                            {
                                "simulation": sim,
                                "datetime": t,
                                "unit": u,
                                "episode": ep // 2 if mode else ep,
                                "evaluation_mode": mode,
                                "reward": ep + rng.normal(),
                                "profit": 1000.0 * ep,
                                "regret": 0.0,
                                "actions_0": rng.uniform(),
                                "actions_1": rng.uniform(),
                                "exploration_noise_0": 0.0
                                if mode
                                else rng.normal(0, 0.3 / ep),
                                "exploration_noise_1": 0.0
                                if mode
                                else rng.normal(0, 0.3 / ep),
                            }
                        )
            if ep > 2:
                for step in range(10):
                    for u in units:
                        grad.append(
                            {
                                "simulation": sim,
                                "episode": ep,
                                "evaluation_mode": 0,
                                "unit": u,
                                "step": (ep - 3) * 10 + step,
                                "critic_loss": 1 / (1 + step + ep),
                                "actor_loss": -1.0,
                                "critic_total_grad_norm": 1.0,
                                "actor_total_grad_norm": 0.5,
                                "critic_max_grad_norm": 1.0,
                                "actor_max_grad_norm": 0.5,
                                "learning_rate": 0.001,
                            }
                        )
    engine = create_engine(f"sqlite:///{path.as_posix()}")
    pd.DataFrame(params).to_sql("rl_params", engine, index=False)
    pd.DataFrame(grad).to_sql("rl_grad_params", engine, index=False)
    pd.DataFrame(meta).to_sql("rl_meta", engine, index=False)
    return f"sqlite:///{path.as_posix()}", [f"learning_seed{s}" for s in range(seeds)]


def test_learning_figures_on_a_synthetic_database(tmp_path):
    uri, sims = _learning_db(tmp_path / "learning.db")
    data = learning_figures.load(uri, sims)
    hours = learning_figures.step_hours(data["params"])
    assert hours == 0.5
    # profit per episode: 1000 x episode per step and unit, an hourly rate over 48 half-hours and two units
    profit = learning_figures.per_episode(
        data["params"], "profit", evaluation=False, hours=hours
    )
    np.testing.assert_allclose(profit.loc[3].to_numpy(), 1000.0 * 3 * 48 * 2 * 0.5)
    stats = learning_figures.interval(profit)
    assert (stats["lo"] <= stats["mean"]).all() and (stats["mean"] <= stats["hi"]).all()
    written = {p.stem for p in learning_figures.make(uri, sims, tmp_path / "figures")}
    assert written == {"20_learning_curves", "21_training_diagnostics", "22_actions"}
    assert set(S.read_captions(tmp_path / "figures" / S.CAPTIONS)) == written


def test_a_redrawn_figure_replaces_its_caption(tmp_path):
    """The figures carry no text; their captions go to captions.md, one section per figure, and a
    figure drawn again replaces its own caption only."""
    for stem, message in (
        ("02_b", "Second"),
        ("01_a", "First"),
        ("02_b", "Second, redrawn"),
    ):
        fig, top, bottom = S.figure(message, "What it shows", "Source of the data.")
        assert not fig.texts and top > 0.9 and bottom < 0.15
        S.axes(fig, top, bottom).plot([0, 1], [0, 1])
        S.save(fig, tmp_path, stem)
    captions = S.read_captions(tmp_path / S.CAPTIONS)
    assert captions == {
        "01_a": "**First.** What it shows. Source of the data.",
        "02_b": "**Second, redrawn.** What it shows. Source of the data.",
    }
    assert (tmp_path / S.CAPTIONS).read_text(encoding="utf-8").index("## 01_a") < (
        tmp_path / S.CAPTIONS
    ).read_text(encoding="utf-8").index("## 02_b")
