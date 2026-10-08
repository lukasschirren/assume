# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""The validation written out: for one ``ValidationData``, the scorecard of every window, the
tables behind the figures, the core and supplementary figures and a note of what could not be
computed, into one folder.

    scorecard_<window>.csv / .tex / _seeds.csv   per window (headline, calibration, year)
    weeks.csv, regressions.csv, regressions_outturn.csv, attenuation.csv,
    binned_wind.csv, binned_solar.csv, markups.csv, capture_prices.csv, storage_daily.csv,
    storage.csv                                  the tables
    fig1_weeks ... fig6_scorecard, figS1_scatter ... figS5_price_vs_residual_demand
                                                 the figures, PDF and PNG
    notes.txt                                    what was left out, and why

A table or figure that cannot be made (no complete week, no reference definition of
penetration, no competitive run, no ensemble) is left out and named in ``notes.txt``; the rest is
still written.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from assume_gb.validation import figures, mechanisms, scorecard, style, value
from assume_gb.validation.contract import (
    ABM,
    COMPETITIVE,
    HEADLINE,
    OUTTURN,
    ValidationData,
)


def write(
    data: ValidationData,
    out: str | Path,
    market: str = "NordPool",
    battery: value.Battery = value.Battery(),
    model: str = ABM,
    draw: bool = True,
) -> list[Path]:
    """Writes the validation of ``data`` into the folder ``out``: the scorecards of every window,
    the tables and, with ``draw``, the figures, with ``model`` (the agent-based model, or the
    competitive run without one) as the model the weeks are chosen by. ``market`` is the curve of
    the causal reference that matches the observed series and ``battery`` the storage of the
    value criterion. Returns the paths written; ``notes.txt`` names what was left out."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    notes: list[str] = []
    model = model if model in data.names else COMPETITIVE

    def attempt(label: str, make: Callable[[], list[Path] | Path]) -> None:
        try:
            result = make()
            written.extend(result if isinstance(result, list) else [result])
        except Exception as error:  # a report goes on without the piece it cannot make
            notes.append(f"{label}: {type(error).__name__}: {error}")

    def table(frame: pd.DataFrame, name: str, index: bool = False) -> Path:
        path = out / f"{name}.csv"
        frame.to_csv(path, index=index)
        return path

    cards = scorecard.scorecards(data, market=market, battery=battery)
    for window, card in cards.items():
        written.append(card.to_csv(out / f"scorecard_{window}.csv"))
        written.append(out / f"scorecard_{window}.tex")
        card.to_latex(written[-1])
        if len(card.seeds):
            written.append(out / f"scorecard_{window}_seeds.csv")
        notes += [f"scorecard {window}: {note}" for note in card.notes]

    attempt("weeks", lambda: table(figures.select_weeks(data, model), "weeks"))
    attempt(
        "regressions",
        lambda: table(mechanisms.compare_regressions(data), "regressions"),
    )
    attempt(
        "regressions on the outturn",
        lambda: table(
            mechanisms.compare_regressions(data, information=OUTTURN),
            "regressions_outturn",
        ),
    )
    attempt("attenuation", lambda: table(mechanisms.attenuation(data), "attenuation"))
    for technology in ("wind", "solar"):
        attempt(
            f"binned {technology} effects",
            lambda technology=technology: table(
                mechanisms.compare_binned(data, technology, market),
                f"binned_{technology}",
            ),
        )
    attempt(
        "markups",
        lambda: table(mechanisms.markups(data, include_competitive=True), "markups"),
    )

    def capture() -> Path:
        generation = (
            data.explanatory()
            .reindex(columns=["wind", "solar"])
            .dropna(axis=1, how="all")
        )
        return table(
            value.capture_prices(data.sample(), generation, "month"), "capture_prices"
        )

    def storage() -> list[Path]:
        daily = value.storage_revenue(data.sample(), battery)
        return [
            table(daily, "storage_daily"),
            table(value.storage_summary(daily), "storage", index=True),
        ]

    attempt("capture prices", capture)
    attempt("storage", storage)

    if draw:
        attempt(
            "fig1_weeks",
            lambda: style.save(
                figures.plot_weeks(data, model=model)[0], out / "fig1_weeks"
            ),
        )
        attempt(
            "fig2_duration",
            lambda: style.save(
                figures.plot_duration_curve(data)[0], out / "fig2_duration"
            ),
        )
        attempt(
            "fig3_error_heatmap",
            lambda: style.save(
                figures.plot_error_heatmap(data)[0], out / "fig3_error_heatmap"
            ),
        )
        attempt(
            "fig4_coefficients",
            lambda: style.save(
                figures.plot_coefficients(
                    mechanisms.compare_regressions(data),
                    figures.causal_ranges(market, data),
                )[0],
                out / "fig4_coefficients",
            ),
        )
        attempt(
            "fig5_markups",
            lambda: style.save(
                figures.plot_markups(
                    mechanisms.markups(data, include_competitive=True)
                )[0],
                out / "fig5_markups",
            ),
        )
        attempt(
            "fig6_scorecard",
            lambda: style.save(
                figures.plot_scorecard(cards[HEADLINE])[0], out / "fig6_scorecard"
            ),
        )
        attempt(
            "figS1_scatter",
            lambda: style.save(figures.plot_scatter(data)[0], out / "figS1_scatter"),
        )
        attempt(
            "figS2_taylor",
            lambda: style.save(figures.plot_taylor(data)[0], out / "figS2_taylor"),
        )
        attempt(
            "figS3_profiles",
            lambda: style.save(figures.plot_profiles(data)[0], out / "figS3_profiles"),
        )
        if data.is_ensemble(model):
            attempt(
                "figS4_pit",
                lambda: style.save(figures.plot_pit(data, model)[0], out / "figS4_pit"),
            )
        else:
            notes.append(f"figS4_pit: {model!r} has one run, no seed ensemble")
        attempt(
            "figS5_price_vs_residual_demand",
            lambda: style.save(
                figures.plot_price_vs_residual_demand(data)[0],
                out / "figS5_price_vs_residual_demand",
            ),
        )

    path = out / "notes.txt"
    path.write_text("\n".join(notes) + ("\n" if notes else ""), encoding="utf-8")
    written.append(path)
    return written
