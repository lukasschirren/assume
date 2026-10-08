# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Synthetic data for the tests: half-hourly prices on a Europe/London index (so the clock changes
are in it), made from synthetic wind, solar, demand, gas and carbon with the known coefficients
``COEF`` plus noise, with daily and weekly seasonality (through demand and solar), price spikes
and some negative prices (high wind at low demand); and "models" with known distortions of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from assume_gb.validation.contract import FORECAST, TZ, vintage

# GBP/MWh per GW (wind, solar, demand), per GBP/MWh thermal (gas), per GBP/t (carbon)
COEF = {"wind": -3.0, "solar": -2.0, "demand": 3.5, "gas": 1.8, "carbon": 0.4}
INTERCEPT = -100.0


def index(
    start: str = "2022-01-01", end: str = "2024-01-01", freq: str = "30min"
) -> pd.DatetimeIndex:
    """Periods from local midnight of ``start`` up to ``end`` (excluded), in Europe/London."""
    return pd.date_range(start, end, freq=freq, tz=TZ, inclusive="left")


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    shocks = rng.normal(0.0, sigma, n)
    out = np.empty(n)
    out[0] = shocks[0] / np.sqrt(1 - phi**2)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + shocks[t]
    return out


def _daily_walk(
    idx: pd.DatetimeIndex, level: float, step: float, rng: np.random.Generator
) -> np.ndarray:
    """A price that changes once a local day: a mean-reverting walk around ``level``."""
    days = idx.tz_localize(None).normalize()
    unique = days.unique()
    walk = level + _ar1(len(unique), 0.98, step, rng)
    return pd.Series(walk, index=unique).reindex(days).to_numpy()


def _errors(n: int, sd: float, rng: np.random.Generator) -> np.ndarray:
    """Persistent forecast errors: an AR(1) with stationary standard deviation ``sd``."""
    return _ar1(n, 0.9, sd * np.sqrt(1 - 0.9**2), rng)


def exog(idx: pd.DatetimeIndex | None = None, seed: int = 0) -> pd.DataFrame:
    """Wind, solar and demand in two information sets: the day-ahead forecast (``*_forecast``)
    and the outturn, the forecast with an error whose mean given the forecast is zero (an
    efficient forecast: a multiplicative error of about 15% for wind and solar, so that they stay
    positive, and an additive one of 0.6 GW for demand); residual demand of the outturn and
    available capacity (GW), gas (GBP/MWh thermal) and carbon (GBP/t); and the forecasts on the
    definitions of the causal reference, transmission-connected wind and transmission system
    demand (GW), in every period of ``idx``."""
    idx = index() if idx is None else idx
    rng = np.random.default_rng(seed)
    n = len(idx)
    hour = np.asarray(idx.hour + idx.minute / 60)
    season = np.cos(
        2 * np.pi * (np.asarray(idx.dayofyear) - 15) / 365.25
    )  # 1 in mid-January
    weekend = np.asarray(idx.dayofweek >= 5)

    wind = 1 + 21 / (1 + np.exp(-(_ar1(n, 0.995, 0.1, rng) + 0.5 * season)))
    daylight = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) ** 1.5
    cloud = np.clip(0.65 + _ar1(n, 0.98, 0.04, rng), 0, 1)
    solar = 10 * daylight * (0.55 - 0.45 * season) * cloud
    demand = (
        27
        + 5 * season
        + 4 * np.exp(-(((hour - 18) / 2.5) ** 2))
        + 2.5 * np.exp(-(((hour - 9.5) / 3) ** 2))
        - 4 * np.exp(-(((hour - 3.5) / 3) ** 2))
        - 2.5 * weekend
        + _ar1(n, 0.95, 0.3, rng)
    )
    log_sd = 0.15
    frame = pd.DataFrame(
        {
            "gas": _daily_walk(idx, 35.0, 2.0, rng),
            "carbon": _daily_walk(idx, 70.0, 2.5, rng),
            "available_capacity": 48 + _ar1(n, 0.99, 0.1, rng),
            "wind_forecast": wind,
            "solar_forecast": solar,
            "demand_forecast": demand,
        },
        index=idx,
    )
    # the errors are drawn last, so that the forecasts and prices keep their random paths
    frame.insert(0, "wind", wind * np.exp(_errors(n, log_sd, rng) - log_sd**2 / 2))
    frame.insert(1, "solar", solar * np.exp(_errors(n, log_sd, rng) - log_sd**2 / 2))
    frame.insert(2, "demand", demand + _errors(n, 0.6, rng))
    frame.insert(3, "residual_demand", frame.demand - frame.wind - frame.solar)
    # the definitions of the causal reference's penetration: transmission-connected wind (part of
    # all wind) over transmission system demand (above the demand net of nuclear)
    frame["wind_tx_forecast"] = 0.7 * frame["wind_forecast"]
    frame["tsd_forecast"] = frame["demand_forecast"] + 1.5
    return frame


def prices(
    x: pd.DataFrame,
    information: str = FORECAST,
    seed: int = 0,
    spike_share: float = 0.003,
) -> pd.Series:
    """Prices in GBP/MWh formed on ``information`` (the forecasts, as the market's are, or the
    outturn, as the agent-based model's are): ``INTERCEPT`` + sum of ``COEF`` x exog + AR(1)
    noise, with positive spikes in a share ``spike_share`` of the periods, independent of the
    exog."""
    rng = np.random.default_rng(seed + 100)
    columns = {name: vintage(name, information) for name in COEF}
    price = INTERCEPT + sum(
        coef * x[columns[name]].to_numpy() for name, coef in COEF.items()
    )
    price = price + _ar1(len(x), 0.8, 4.0, rng)
    spikes = rng.random(len(x)) < spike_share
    price = price + spikes * rng.uniform(80, 300, len(x))
    return pd.Series(
        price,
        index=x.index,
        name="observed" if information == FORECAST else "simulated",
    )


# ---------------------------------------------------------------- models with known distortions
def shifted(observed: pd.Series, delta: float = 15.0) -> pd.Series:
    """Every price ``delta`` higher: a pure level error."""
    return (observed + delta).rename("shifted")


def flattened(observed: pd.Series, factor: float = 0.5) -> pd.Series:
    """Every deviation from the mean scaled by ``factor``: too little amplitude, perfect timing."""
    return (observed.mean() + factor * (observed - observed.mean())).rename("flattened")


def lagged(observed: pd.Series, periods: int = 4) -> pd.Series:
    """The price ``periods`` periods late: a timing error (NaN at the start)."""
    return observed.shift(periods).rename("lagged")


def noisy(observed: pd.Series, sigma: float = 10.0, seed: int = 1) -> pd.Series:
    """The price plus white noise of standard deviation ``sigma``."""
    rng = np.random.default_rng(seed)
    return (observed + rng.normal(0.0, sigma, len(observed))).rename("noisy")


def ensemble(
    observed: pd.Series, members: int = 5, sigma: float = 10.0, seed: int = 2
) -> pd.DataFrame:
    """``members`` noisy copies of the price, one column per seed."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, (len(observed), members))
    return pd.DataFrame(
        observed.to_numpy()[:, None] + noise,
        index=observed.index,
        columns=range(members),
    )
