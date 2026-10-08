# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Observed prices from the GB data repository, for the scenario's ``observed_prices.csv``.

    n2ex_day_ahead      the N2EX hourly day-ahead auction (Nord Pool, licensed; in the stack export)
    epex_day_ahead      the EPEX SPOT GB hourly day-ahead auction (licensed), from 2020
    blend_day_ahead     the two hourly auctions weighted by their volumes
    epex_hh_day_ahead   the EPEX SPOT GB half-hourly day-ahead auction, 15:30 D-1 (licensed), from 2020
    epex_ida1        EPEX SPOT GB intraday auction 1, 17:30 D-1, all 48 half-hours (licensed)
    epex_ida2        EPEX SPOT GB intraday auction 2, 08:00 D, half-hours from 12:00 (licensed)
    apx_mid          Elexon market index price, APX: the within-day traded price (public)
    system_price     Elexon cash-out price (public)

The file is kept out of version control as a whole because of the licensed columns.
"""

import pandas as pd

from assume_gb import paths


def _utc_naive(series: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(series, utc=True)).tz_localize(None)


def _series(
    path, time_column: str, value_column: str, index: pd.DatetimeIndex
) -> pd.Series | None:
    if not path.exists():
        return None
    table = pd.read_csv(path, usecols=[time_column, value_column])
    table.index = _utc_naive(table.pop(time_column))
    return table[~table.index.duplicated()][value_column].reindex(index)


def prices(year: int, index: pd.DatetimeIndex) -> pd.DataFrame:
    """The observed prices of the year on ``index`` beside the N2EX day-ahead price: the other
    day-ahead auctions, the intraday auctions, the market index and the system price; a missing
    source is left out. The simulated day-ahead market is one market for all demand, so it is
    compared with both hourly auctions and their blend."""
    repo = paths.gb_repo() / "data"
    epex = repo / "raw" / "epex" / "great-britain"
    venues = repo / "processed" / f"dayahead_price_gb_{year}.csv"
    candidates = {
        "epex_day_ahead": _series(
            venues, "startTime", "epex_hourly_gbp_per_mwh", index
        ),
        "blend_day_ahead": _series(venues, "startTime", "blend_gbp_per_mwh", index),
        "epex_hh_day_ahead": _series(
            repo / "processed" / f"dayahead_price_epex_hh_{year}.csv",
            "startTime",
            "price_gbp_per_mwh",
            index,
        ),
        "epex_ida1": _series(
            epex / "epex_gb_intraday_ida1_prices_2020_2025.csv",
            "deliveryStart",
            "price_gbp_mwh",
            index,
        ),
        "epex_ida2": _series(
            epex / "epex_gb_intraday_ida2_prices_2020_2025.csv",
            "deliveryStart",
            "price_gbp_mwh",
            index,
        ),
        "apx_mid": _series(
            repo / "processed" / f"mid_target_{year}.csv",
            "startTime",
            "price_gbp_per_mwh",
            index,
        ),
        "system_price": _series(
            repo / "processed" / f"system_prices_{year}.csv",
            "startTime",
            "price_gbp_per_mwh",
            index,
        ),
    }
    return pd.DataFrame(
        {k: v for k, v in candidates.items() if v is not None and v.notna().any()},
        index=index,
    )
