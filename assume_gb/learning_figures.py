# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The standard figures of a learning run: how the agents learned, read from the database the run
wrote (``rl_params``, ``rl_grad_params``, ``rl_meta``; learning runs need a database).

    python -m assume_gb learning --db sqlite:///path.db --simulation NAME [NAME ...] [--out DIR]

Several simulations are taken as repetitions of one experiment with different seeds: the curves are
then their mean with a 95% bootstrap interval, as the reporting guides for reinforcement learning
ask (Henderson et al. 2018; Agarwal et al. 2021). With one simulation there is no interval, and the
captions say so. The figures carry no text beyond axes and legends; their captions go to
``captions.md`` beside them. What the learned bids do to the market is in the run figures against a baseline run
with the competitive strategies (``python -m assume_gb figures <run> --baseline <run>``, figures 14
and 15).

Money in ``rl_params`` is price x MW per step, an hourly rate: profits are multiplied by the length
of a step in hours here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from assume_gb import style as S
from assume_gb.figures import legend_above, line_handles

BOOTSTRAP = 2000


def _load_one(db_uri: str, simulations: list[str]) -> dict[str, pd.DataFrame]:
    from sqlalchemy import create_engine, inspect

    engine = create_engine(db_uri)
    tables = set(inspect(engine).get_table_names())
    if "rl_params" not in tables:
        raise ValueError(
            f"{db_uri} has no rl_params table: is it the database of a learning run?"
        )
    names = ", ".join(f"'{s}'" for s in simulations)
    out = {
        "params": pd.read_sql(
            f"select * from rl_params where simulation in ({names})",
            engine,
            parse_dates=["datetime"],
        )
    }
    out["grad"] = (
        pd.read_sql(
            f"select * from rl_grad_params where simulation in ({names})", engine
        )
        if "rl_grad_params" in tables
        else pd.DataFrame()
    )
    out["meta"] = (
        pd.read_sql(f"select * from rl_meta where simulation in ({names})", engine)
        if "rl_meta" in tables
        else pd.DataFrame()
    )
    return out


def load(db_uri: str | list[str], simulations: list[str]) -> dict[str, pd.DataFrame]:
    """The learning tables of the simulations, from one database or from several (runs on a
    cluster write a database each)."""
    uris = [db_uri] if isinstance(db_uri, str) else list(db_uri)
    parts = [_load_one(uri, simulations) for uri in uris]
    out = {
        key: pd.concat([part[key] for part in parts], ignore_index=True)
        for key in ("params", "grad", "meta")
    }
    missing = set(simulations) - set(out["params"].simulation)
    if missing:
        raise ValueError(
            f"no learning output for {sorted(missing)} in {', '.join(uris)}"
        )
    return out


def step_hours(params: pd.DataFrame) -> float:
    times = params.datetime.drop_duplicates().sort_values()
    return (
        float(times.diff().median() / pd.Timedelta(hours=1)) if len(times) > 1 else 1.0
    )


def interval(table: pd.DataFrame, rng=None) -> pd.DataFrame:
    """Mean over the columns (repetitions) of each row with a 95% bootstrap interval."""
    rng = rng or np.random.default_rng(0)
    values = table.to_numpy(dtype=float)
    mean = np.nanmean(values, axis=1)
    if values.shape[1] < 2:
        return pd.DataFrame(
            {"mean": mean, "lo": np.nan, "hi": np.nan}, index=table.index
        )
    picks = rng.integers(0, values.shape[1], size=(BOOTSTRAP, values.shape[1]))
    boot = np.nanmean(values[:, picks], axis=2)
    return pd.DataFrame(
        {
            "mean": mean,
            "lo": np.percentile(boot, 2.5, axis=1),
            "hi": np.percentile(boot, 97.5, axis=1),
        },
        index=table.index,
    )


def per_episode(
    params: pd.DataFrame, value: str, evaluation: bool, hours: float = 1.0
) -> pd.DataFrame:
    """Sum of ``value`` over an episode, summed over the units: one column per simulation."""
    p = params[params.evaluation_mode.astype(int) == int(evaluation)]
    table = p.groupby(["episode", "simulation"])[value].sum().unstack()
    return table * (hours if value == "profit" else 1.0)


def eval_positions(meta: pd.DataFrame, episodes: pd.Index) -> pd.Index:
    """The training episode after which each evaluation episode ran."""
    if meta.empty or "eval_episode" not in meta:
        return episodes
    m = (
        meta[meta.evaluation_mode.astype(int) == 1]
        .drop_duplicates("eval_episode")
        .set_index("eval_episode")
        .episode
    )
    return pd.Index([m.get(e, e) for e in episodes])


def first_learning_episode(grad: pd.DataFrame) -> int | None:
    return int(grad.episode.min()) if not grad.empty and "episode" in grad else None


def _curve(ax, x, stats: pd.DataFrame, colour, label=None, marker=None):
    if stats["lo"].notna().any():
        ax.fill_between(x, stats["lo"], stats["hi"], color=colour, alpha=0.18, lw=0)
    ax.plot(x, stats["mean"], color=colour, lw=1.8, marker=marker, ms=5, label=label)


def fig_learning_curves(data: dict, out: Path, label: str):
    params, meta, grad = data["params"], data["meta"], data["grad"]
    hours = step_hours(params)
    n_sims = params.simulation.nunique()
    n_units = params.unit.nunique()
    train_r, eval_r = (
        per_episode(params, "reward", False),
        per_episode(params, "reward", True),
    )
    train_p, eval_p = (
        per_episode(params, "profit", False, hours),
        per_episode(params, "profit", True, hours),
    )
    ex = eval_positions(meta, eval_r.index)
    er = interval(eval_r)
    rise = (er["mean"].iloc[-1] - er["mean"].iloc[0]) if len(er) > 1 else np.nan
    repeat = (
        f"mean of {n_sims} runs with a 95% bootstrap interval"
        if n_sims > 1
        else "one run, so no interval: repeat with several seeds before reading differences"
    )
    trend = "rose" if rise > 0 else "fell"
    agents = "the learning agent" if n_units == 1 else f"the {n_units} learning agents"
    title = (
        f"Over {len(train_r)} training episodes the evaluation reward of {agents} {trend} from {er['mean'].iloc[0]:.1f} to {er['mean'].iloc[-1]:.1f}"
        if len(er) > 1
        else f"{len(train_r)} training episodes of {n_units} learning agents"
    )
    fig, top, bottom = S.figure(
        title,
        f"Reward (as the bidding strategy defines it) and profit (£, market revenue less cost) per episode, summed over "
        f"the learning units: training episodes (line) and evaluation episodes without exploration (dots, placed after "
        f"the training episode they followed); {repeat}. Shaded: episodes that only collect experience.",
        f"Learning output (rl_params, rl_meta, rl_grad_params) of {label}.",
        h=6.4,
    )
    axs = S.stacked(fig, top - 0.03, bottom, ratios=(1, 1), hspace=0.09)
    first = first_learning_episode(grad)
    for ax, train, ev, ylabel in (
        (axs[0], train_r, eval_r, "Reward"),
        (axs[1], train_p / 1e6, eval_p / 1e6, "Profit, £ million"),
    ):
        if first and first > 1:
            ax.axvspan(0.5, first - 0.5, color=S.FAINT, lw=0)
        _curve(ax, train.index, interval(train), S.LIGHT if n_sims == 1 else S.MUTED)
        if n_sims == 1:
            ax.plot(train.index, train.iloc[:, 0], color=S.MUTED, lw=1.2)
        _curve(ax, ex, interval(ev), S.NAVY, marker="o")
        ax.set_ylabel(ylabel)
        ax.axhline(0, color=S.LIGHT, lw=0.8)
    axs[1].tick_params(labelbottom=True)
    axs[1].set_xlabel("Training episode")
    axs[0].set_xlim(0.5, max(train_r.index.max(), max(ex)) + 0.5)
    legend_above(
        axs[0],
        line_handles([(S.MUTED, 1.2, "-"), (S.NAVY, 1.8, "-")]),
        ["Training", "Evaluation"],
    )
    return S.save(fig, out, "20_learning_curves")


def fig_diagnostics(data: dict, out: Path, label: str):
    grad, params = data["grad"], data["params"]
    noise_cols = [c for c in params.columns if c.startswith("exploration_noise_")]
    train = params[params.evaluation_mode.astype(int) == 0]
    noise = (
        train.groupby("episode")[noise_cols].std().mean(axis=1)
        if noise_cols
        else pd.Series(dtype=float)
    )
    fig, top, bottom = S.figure(
        "Training diagnostics: the critics' loss should settle and the exploration noise shrink as the agents learn",
        "Left: critic and actor loss per gradient step, mean over the learning units (log scale). Middle: total gradient "
        "norm of the critics and actors. Right: spread (standard deviation) of the exploration noise added to the "
        "actions in each training episode.",
        f"Learning output (rl_grad_params, rl_params) of {label}.",
    )
    width = (0.97 - 0.08 - 2 * 0.07) / 3
    axs = [
        fig.add_axes([0.08 + i * (width + 0.07), bottom, width, top - 0.04 - bottom])
        for i in range(3)
    ]
    if not grad.empty:
        g = grad.groupby("step")[
            [
                "critic_loss",
                "actor_loss",
                "critic_total_grad_norm",
                "actor_total_grad_norm",
            ]
        ].mean()
        roll = max(len(g) // 50, 1)
        for col, colour in (("critic_loss", S.NAVY), ("actor_loss", S.WINE)):
            series = g[col].abs().dropna()
            axs[0].plot(
                series.index,
                series.rolling(roll, min_periods=1).mean(),
                color=colour,
                lw=1.4,
            )
        axs[0].set_yscale("log")
        for col, colour in (
            ("critic_total_grad_norm", S.NAVY),
            ("actor_total_grad_norm", S.WINE),
        ):
            series = g[col].dropna()
            axs[1].plot(
                series.index,
                series.rolling(roll, min_periods=1).mean(),
                color=colour,
                lw=1.4,
            )
        legend_above(
            axs[0],
            line_handles([(S.NAVY, 1.4, "-"), (S.WINE, 1.4, "-")]),
            ["Critic", "Actor (absolute)"],
        )
    for ax in axs[:2]:
        ax.set_xlabel("Gradient step")
    axs[0].set_ylabel("Loss")
    axs[1].set_ylabel("Gradient norm")
    if not noise.empty:
        axs[2].plot(noise.index, noise.values, color=S.BODY, lw=1.4, marker="o", ms=4)
    axs[2].set_xlabel("Training episode")
    axs[2].set_ylabel("Noise, standard deviation")
    return S.save(fig, out, "21_training_diagnostics")


def fig_actions(data: dict, out: Path, label: str):
    params = data["params"]
    action_cols = sorted(c for c in params.columns if c.startswith("actions_"))
    ev = params[params.evaluation_mode.astype(int) == 1]
    if ev.empty or not action_cols:
        return None
    first, last = ev.episode.min(), ev.episode.max()
    hour = ev.datetime.dt.hour + ev.datetime.dt.minute / 60
    prof = ev.groupby([ev.episode, hour])[action_cols].mean()
    spread = (
        ev.groupby([ev.episode, hour, ev.unit])[action_cols]
        .mean()
        .groupby(level=[0, 1])
        .agg(["min", "max"])
    )
    change = {
        c: prof.loc[last, c].mean() - prof.loc[first, c].mean() for c in action_cols
    }
    fig, top, bottom = S.figure(
        "Between the first and the last evaluation the agents' actions moved by "
        + ", ".join(
            f"{v:+.2f} ({c.replace('actions_', 'action ')})" for c, v in change.items()
        )
        + " on average",
        "Mean action of the learning units by time of day in the first and in the last evaluation episode (the "
        "strategy's own scale, before it is turned into a bid price); shaded: lowest to highest unit.",
        f"Learning output (rl_params) of {label}.",
    )
    n = len(action_cols)
    width = (0.97 - 0.08 - 0.07 * (n - 1)) / n
    for i, col in enumerate(action_cols):
        ax = fig.add_axes(
            [0.08 + i * (width + 0.07), bottom, width, top - 0.05 - bottom]
        )
        for ep, colour in ((first, S.LIGHT), (last, S.NAVY)):
            m = prof.loc[ep, col]
            lo, hi = spread.loc[ep, (col, "min")], spread.loc[ep, (col, "max")]
            ax.fill_between(m.index, lo, hi, color=colour, alpha=0.2, lw=0)
            ax.plot(m.index, m.values, color=colour if ep == last else S.MUTED, lw=1.8)
        ax.set_xlabel("Time of day")
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_ylabel(col.replace("actions_", "Action "))
        if i == 0:
            legend_above(
                ax,
                line_handles([(S.MUTED, 1.8, "-"), (S.NAVY, 1.8, "-")]),
                [f"First evaluation ({first})", f"Last evaluation ({last})"],
                fontsize=9.5,
            )
    return S.save(fig, out, "22_actions")


def make(db_uri: str | list[str], simulations: list[str], out: Path) -> list[Path]:
    data = load(db_uri, simulations)
    label = ", ".join(simulations)
    written = []
    for fn in (fig_learning_curves, fig_diagnostics, fig_actions):
        path = fn(data, out, label)
        if path is not None:
            written.append(path)
    return written
