# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The aggregated bid curves of the GB day-ahead auctions, hour by hour, beside the model's.

Two hourly auctions trade the GB day ahead: N2EX (Nord Pool) and EPEX's GB hourly auction. Both
publish their aggregated supply and demand curves (licensed: `assume_gb/bid_curves/`, local only).
This module reads them, checks that they reproduce each auction's published price, and writes
them resampled onto one price grid per year:

    python -m assume_gb curves --years 2023 2025      -> bid_curves/derived/curves_<year>.npz

**What an exchange curve is.** It is the net position of the portfolios that trade on that
exchange, not the physical merit order: about a third of GB demand clears on N2EX (13.7 GW of
44.7 GW at 17:00 on 27 Nov 2023), the rest is supplied by its owners' own plants, under contracts
or on the other exchange. Much of the "demand" curve is make-or-buy: a company bids to buy back
what it would otherwise generate itself. The supply and demand curves therefore cannot be laid
over the model's offers and GB demand. Their difference can: the excess supply
ES(p) = S(p) - D(p) + shift is unchanged by volume a company supplies to itself or holds in a
fixed contract (it leaves both sides), a make-or-buy bid enters it where the company's plant
would, its zero is the clearing price, and its slope there is the price response to a shift of
demand. That is the curve the viewer sets beside the model's, whose excess supply is its offers
less its (fixed) demand bid.

**What clears the auctions.**
- EPEX GB hourly: the aggregated curves alone (step curves; a step is two points at one price).
  Hour h of a day is the CET hour [h-1, h) ("3B" the repeated hour in October).
- N2EX: the aggregated curves of the hourly orders (piecewise linear), plus the accepted block
  orders (sell less buy, from the same file), plus the net import of the auction's coupled
  interconnector (North Sea Link to NO2 since October 2021, from `flows/by_day`). With all three
  the curves reproduce the published price to about £0.02/MWh.

The check is written with the curves (`<venue>_price` reconstructed, `<venue>_published`), and
`curves` prints its error per year: read it before using a year.

Formats read: EPEX's daily CSV inside the yearly zip (2020 onwards; 2018-2019 come as nested
zips in another layout and are not read yet); Nord Pool's daily JSON in `nordpool.tar.xz`.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
import zipfile

import numpy as np
import pandas as pd

from assume_gb import paths

ROOT = paths.TOOL_ROOT / "bid_curves"
DERIVED = ROOT / "derived"
NORDPOOL = ROOT / "nordpool.tar.xz"
EPEX_ZIP = (
    ROOT
    / "epex"
    / "great-britain"
    / "aggregated_curves_hourly"
    / "auction_aggregated_curves_great-britain_{year}.zip"
)
VENUES = ("n2ex", "epex")

# the price grid of the resampled curves: fine where prices usually are, coarse in the tails
GRID = np.unique(
    np.concatenate(
        [
            [-500, -400, -300, -200, -150],
            np.arange(-100, -20, 5),
            np.arange(-20, 300, 1),
            np.arange(300, 500, 5),
            np.arange(500, 1000, 25),
            np.arange(1000, 3001, 250),
        ]
    ).astype(float)
)


def hours(year: int) -> pd.DatetimeIndex:
    """The UTC hours (naive, as the scenarios' index) that start in the delivery year, with the
    evening before it, so every half-hour of a scenario's year has its hour."""
    return pd.date_range(f"{year - 1}-12-31 22:00", f"{year}-12-31 23:00", freq="h")


def clearing(supply_p, supply_v, demand_p, demand_v, shift: float = 0.0) -> float:
    """The price at which supply plus ``shift`` meets demand, from the curves' own points (linear
    between them); nan if the curves never cross."""
    p = np.union1d(supply_p, demand_p)
    es = np.interp(p, supply_p, supply_v) - np.interp(p, demand_p, demand_v) + shift
    k = int(np.argmax(es >= 0))
    if es[k] < 0:
        return np.nan
    if k == 0:
        return float(p[0])
    # between the two points that bracket the crossing, search on a grid of 1p: right for the
    # piecewise linear N2EX curves and for EPEX's steps (a step is two points at one price)
    fine = np.arange(p[k - 1], p[k] + 0.005, 0.01)
    es = (
        np.interp(fine, supply_p, supply_v)
        - np.interp(fine, demand_p, demand_v)
        + shift
    )
    return float(fine[int(np.argmax(es >= 0))])


def _cumulate(price: np.ndarray, volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A curve as points sorted by price, volumes as given (supply rises, demand falls)."""
    order = np.argsort(price, kind="stable")
    return price[order].astype(float), volume[order].astype(float)


def read_epex(year: int) -> dict:
    """EPEX GB hourly auction: per UTC hour, the supply and demand curves (points)."""
    out = {}
    with zipfile.ZipFile(str(EPEX_ZIP).format(year=year)) as z:
        for name in sorted(z.namelist()):
            if not name.endswith(".csv"):
                continue
            c = pd.read_csv(io.BytesIO(z.read(name)), skiprows=1)
            for (date, label), g in c.groupby(["Date", "Hour"], sort=False):
                label = str(label)
                h = int(re.match(r"\d+", label).group())
                # hour h is the CET wall-clock hour [h-1, h); "3B" the second of the repeated hour
                wall = pd.to_datetime(date, dayfirst=True) + pd.Timedelta(hours=h - 1)
                start = (
                    wall.tz_localize("Europe/Paris", ambiguous=not label.endswith("B"))
                    .tz_convert("UTC")
                    .tz_localize(None)
                )
                s, d = (
                    g[g["Sale/Purchase"] == "Sell"],
                    g[g["Sale/Purchase"] == "Purchase"],
                )
                out[start] = (
                    *_cumulate(s.Price.to_numpy(), s.Volume.to_numpy()),
                    *_cumulate(d.Price.to_numpy(), d.Volume.to_numpy()),
                    0.0,
                )
    return out


def read_n2ex(years: list[int]) -> dict[int, dict]:
    """N2EX: per UTC hour, the curves (points) and the shift (accepted blocks and coupled flows),
    for each of ``years``, in one pass over the archive (curves come before flows in it)."""
    curves: dict[pd.Timestamp, list] = {}
    flows: dict[pd.Timestamp, float] = {}
    wanted = re.compile(r"nordpool/(curves|flows/by_day)/((\d{4})-\d{2}-\d{2})\.json$")
    days = {
        f"{y - 1}-12-31" for y in years
    }  # the first hours of a year are in the day file before
    with tarfile.open(NORDPOOL, "r|xz") as tar:
        for member in tar:
            m = wanted.match(member.name)
            if not m or (int(m.group(3)) not in years and m.group(2) not in days):
                continue
            data = json.load(tar.extractfile(member))
            if m.group(1) == "curves":
                blocks: dict[str, float] = {}
                for b in data.get("blockOrders", []):
                    sign = 1.0 if b["side"] == "Sell" else -1.0
                    for iv in b["intervals"]:
                        blocks[iv["deliveryStart"]] = blocks.get(
                            iv["deliveryStart"], 0.0
                        ) + sign * (iv.get("acceptedVolume") or 0.0)
                for op in data.get("orderPositions", []):
                    t = (
                        pd.Timestamp(op["deliveryStart"])
                        .tz_convert("UTC")
                        .tz_localize(None)
                    )
                    s, d = (
                        pd.DataFrame(op["supplyCurve"]),
                        pd.DataFrame(op["demandCurve"]),
                    )
                    if s.empty or d.empty:
                        continue
                    curves[t] = [
                        *_cumulate(s.price.to_numpy(), s.volume.to_numpy()),
                        *_cumulate(d.price.to_numpy(), d.volume.to_numpy()),
                        blocks.get(op["deliveryStart"], 0.0),
                    ]
            else:
                for area in data:
                    for f in area.get("flows", []):
                        t = (
                            pd.Timestamp(f["deliveryStart"])
                            .tz_convert("UTC")
                            .tz_localize(None)
                        )
                        flows[t] = flows.get(t, 0.0) + float(
                            f.get("totalNetPosition") or 0.0
                        )
    for t, c in curves.items():
        c[4] += flows.get(t, 0.0)
    out: dict[int, dict] = {y: {} for y in years}
    for y in years:
        span = hours(y)
        out[y] = {t: tuple(c) for t, c in curves.items() if span[0] <= t <= span[-1]}
    return out


def resample(points: dict, index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Curves on ``GRID`` (MW) and the exact clearing price, per hour of ``index`` (nan where the
    auction has no curves)."""
    n = len(index)
    supply = np.full((n, len(GRID)), np.nan, dtype="float32")
    demand = np.full((n, len(GRID)), np.nan, dtype="float32")
    shift = np.zeros(n, dtype="float32")
    price = np.full(n, np.nan)
    pos = {t: i for i, t in enumerate(index)}
    for t, (sp, sv, dp, dv, sh) in points.items():
        i = pos.get(t)
        if i is None:
            continue
        supply[i] = np.interp(GRID, sp, sv)
        demand[i] = np.interp(GRID, dp, dv)
        shift[i] = sh
        price[i] = clearing(sp, sv, dp, dv, sh)
    return {"supply": supply, "demand": demand, "shift": shift, "price": price}


def published(year: int, index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """The published hourly prices of both auctions on ``index`` (from the scenario's
    ``observed_prices.csv``, half-hourly there: the hour's first half-hour)."""
    obs = pd.read_csv(
        paths.scenarios_dir() / paths.scenario_name(year) / "observed_prices.csv",
        index_col=0,
        parse_dates=True,
    )
    return {
        "n2ex": obs["n2ex_day_ahead"].reindex(index).to_numpy(),
        "epex": obs["epex_day_ahead"].reindex(index).to_numpy(),
    }


def build(years: list[int]) -> pd.DataFrame:
    """Writes ``bid_curves/derived/curves_<year>.npz`` for each year; returns the reconstruction
    check per year and venue (hours with curves, mean and largest absolute error, £/MWh)."""
    DERIVED.mkdir(parents=True, exist_ok=True)
    n2ex = read_n2ex(years)
    rows = []
    for y in years:
        index = hours(y)
        arrays = {
            "hours": index.values.astype("datetime64[ns]").astype("int64"),
            "grid": GRID,
        }
        pub = published(y, index)
        for venue, points in (("n2ex", n2ex[y]), ("epex", read_epex(y))):
            r = resample(points, index)
            for k, v in r.items():
                arrays[f"{venue}_{k}"] = v
            arrays[f"{venue}_published"] = pub[venue]
            err = np.abs(r["price"] - pub[venue])
            rows.append(
                {
                    "year": y,
                    "venue": venue,
                    "hours": int(np.isfinite(r["price"]).sum()),
                    "mean_abs_error": float(np.nanmean(err)),
                    "max_abs_error": float(np.nanmax(err)),
                    "share_within_0.10": float(np.nanmean(err <= 0.10)),
                }
            )
        np.savez_compressed(DERIVED / f"curves_{y}.npz", **arrays)
    return pd.DataFrame(rows)


def load(year: int) -> dict[str, np.ndarray] | None:
    """The resampled curves of a year, or None where they have not been built."""
    path = DERIVED / f"curves_{year}.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
