# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Reference estimates from the literature, read from the package's own ``data/reference`` folder
(see its README for provenance): the yardsticks the mechanism checks of criterion 5 are compared
with. This is the only file the core of the package reads, and it travels with the package.

    cacciarelli_cate_digitised.csv   the causal effect of the day-ahead wind and solar forecasts on
                                     GB day-ahead prices by predicted penetration, with its 80%
                                     band, for the APX and the Nord Pool auction
                                     [cacciarelliWeActuallyUnderstand2025]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_DIR = Path(__file__).resolve().parent / "data" / "reference"
CACCIARELLI = REFERENCE_DIR / "cacciarelli_cate_digitised.csv"

MARKETS = ("APX", "NordPool")
TECHNOLOGIES = ("wind", "solar")
CURVE = {
    "cate_gbp_per_mwh_per_gw": "estimate",
    "ci80_low": "ci80_low",
    "ci80_high": "ci80_high",
}


def cate_curve(market: str, technology: str, path: Path = CACCIARELLI) -> pd.DataFrame:
    """The causal effect of +1 GW of day-ahead ``technology`` forecast on the day-ahead price of
    ``market`` ("APX" for the APX/EPEX half-hourly auction, "NordPool" for N2EX), in GBP/MWh per
    GW, as traced from Figs. 4 and 5 of the published paper: indexed by predicted penetration
    (forecast / estimated load x 100, %), columns ``estimate``, ``ci80_low`` and ``ci80_high``
    (the 80% band; NaN where the figure hides it). Approximate: about +/-0.05 GBP/MWh on the
    estimate and +/-0.1 on the band. Causal estimates from local partially linear double machine
    learning, against which a regression is only a consistency check
    [cacciarelliWeActuallyUnderstand2025]."""
    if market not in MARKETS:
        raise ValueError(f"market must be one of {MARKETS}, not {market!r}")
    if technology not in TECHNOLOGIES:
        raise ValueError(
            f"technology must be one of {TECHNOLOGIES}, not {technology!r}"
        )
    table = pd.read_csv(path)
    curve = table[(table["market"] == market) & (table["technology"] == technology)]
    curve = curve.set_index("predicted_penetration_pct").sort_index()[list(CURVE)]
    return curve.rename(columns=CURVE).rename_axis("penetration")


def cate_at(
    penetration: float | np.ndarray | pd.Series,
    market: str,
    technology: str,
    path: Path = CACCIARELLI,
) -> pd.DataFrame:
    """The reference curve of ``cate_curve`` at the given predicted penetrations (%), interpolated
    linearly between its points and NaN outside the range of each column (no extrapolation), in
    GBP/MWh per GW [cacciarelliWeActuallyUnderstand2025]."""
    curve = cate_curve(market, technology, path)
    x = np.atleast_1d(np.asarray(penetration, dtype=float))
    out = {}
    for column in curve.columns:
        points = curve[column].dropna()
        out[column] = np.interp(
            x, points.index, points.to_numpy(), left=np.nan, right=np.nan
        )
    return pd.DataFrame(out, index=pd.Index(x, name="penetration"))
