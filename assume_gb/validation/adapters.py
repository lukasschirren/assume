# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""The one module that knows the folders of ``assume_gb``: it reads the scenario folders
(``assume_gb/inputs/gb_<year>``) and the saved runs (``assume_gb/results``) into the data contract
of the validation package. Every file there is on a naive UTC index, half-hourly; it is localised
here, UTC to Europe/London.

    observed      ``observed_prices.csv`` (licensed): N2EX, EPEX hourly and their blend (hourly,
                  written on both half-hours of the hour) or the EPEX half-hourly auction
    competitive   ``results/prices/gb_<year>_day_ahead.csv`` where the run is saved, otherwise
                  ``scenario_merit_order`` of ``reference_prices.csv``: the same price to 1e-13
    abm           ``results/prices/gb_<year>_<arm>_s<seed>_year.csv`` (the policies of the best
                  evaluation) or ``..._year_last.csv`` (those at the end of training)
    exog          the scenario's inputs: wind and solar as availability x max_power of their units
                  (the outturn and the 09:00 D-1 forecast), the demand less the interconnectors'
                  export capacity, fuel and carbon prices, and ``penetration_forecasts.csv``
    periods       the training window of the arm's first seed in ``config.yaml``
    battery       the battery fleet of ``scenario_meta.json``

Periods the merit-order model flags as inadmissible (``admissible`` in ``reference_prices.csv``,
e.g. corrupt demand in 2018 and 2020) are left out of the observed price.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import yaml

from assume_gb import paths
from assume_gb.validation.contract import (
    ABM,
    CALIBRATION,
    COMPETITIVE,
    HEADLINE,
    TZ,
    YEAR,
    ValidationData,
)
from assume_gb.validation.value import Battery

HOURLY_VENUES = ("n2ex_day_ahead", "epex_day_ahead", "blend_day_ahead")
HALF_HOURLY_VENUES = ("epex_hh_day_ahead",)
# the curve of the causal reference that belongs to a venue (reference.MARKETS); the blend is
# mostly N2EX
MARKETS = {
    "n2ex_day_ahead": "NordPool",
    "blend_day_ahead": "NordPool",
    "epex_day_ahead": "APX",
    "epex_hh_day_ahead": "APX",
}
WIND, SOLAR, EXPORT, UNSERVED = "Wind", "Solar", "IC_EXPORT", "unserved_demand"
DEMAND_UNIT = "demand"
PERIOD = pd.Timedelta(minutes=30)


def folder(year: int) -> Path:
    """The scenario folder of ``year``."""
    return paths.scenarios_dir() / paths.scenario_name(year)


def _prices_dir() -> Path:
    return paths.results_dir("prices")


def _years(years: int | Iterable[int]) -> list[int]:
    return sorted({years} if isinstance(years, int) else set(years))


def _local(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """A frame on a naive UTC index, on the Europe/London clock."""
    return frame.set_axis(
        pd.DatetimeIndex(frame.index).tz_localize("UTC").tz_convert(TZ)
    )


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True, **kwargs)


def _bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Start and end (excluded) of the delivery of ``year``'s folder, naive UTC: as its
    ``scenario_meta.json`` records them, the calendar year without one."""
    meta = folder(year) / "scenario_meta.json"
    if meta.exists():
        record = json.loads(meta.read_text(encoding="utf-8"))
        if "delivery_start" in record:
            return pd.Timestamp(record["delivery_start"]), pd.Timestamp(
                record["delivery_end"]
            )
    return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")


def _delivery(frame: pd.DataFrame | pd.Series, year: int) -> pd.DataFrame | pd.Series:
    """The delivery periods of ``year``'s folder, without its lead-in day and closing row."""
    start, end = _bounds(year)
    return frame[(frame.index >= start) & (frame.index < end)]


def admissible(years: int | Iterable[int]) -> pd.Series:
    """Whether the merit-order model admits each half-hour of ``years`` (``reference_prices.csv``)."""
    flags = [
        _delivery(_read(folder(y) / "reference_prices.csv")["admissible"], y)
        for y in _years(years)
    ]
    return _local(pd.concat(flags).astype(bool))


def market(venue: str) -> str:
    """The market of the causal reference that matches an observed venue."""
    return MARKETS[venue]


def observed(
    years: int | Iterable[int],
    venue: str = "n2ex_day_ahead",
    lead_in: pd.Timedelta = pd.Timedelta(days=7),
) -> pd.Series:
    """The observed day-ahead price of ``venue`` in ``years``, GBP/MWh, Europe/London, preceded
    by ``lead_in`` from the previous year's folder where there is one (the naive forecast needs
    the week before). An hourly venue comes back hourly (the price of the hour, written on both
    of its half-hours in the folder), the half-hourly auction half-hourly. A period with an
    inadmissible half-hour is NaN."""
    if venue not in HOURLY_VENUES + HALF_HOURLY_VENUES:
        raise ValueError(
            f"venue must be one of {HOURLY_VENUES + HALF_HOURLY_VENUES}, not {venue!r}"
        )
    years = _years(years)
    parts = []
    for year in [years[0] - 1, *years]:
        path = folder(year) / "observed_prices.csv"
        if not path.exists():
            if year == years[0] - 1:
                continue
            raise FileNotFoundError(f"{path} is missing (licensed data, local only)")
        table = _delivery(_read(path), year)
        if venue not in table:
            raise KeyError(f"{path} has no {venue}")
        price = table[venue].where(
            _delivery(
                _read(folder(year) / "reference_prices.csv")["admissible"], year
            ).astype(bool)
        )
        parts.append(price)
    price = _local(pd.concat(parts).sort_index())
    start = pd.Timestamp(f"{years[0]}-01-01", tz=TZ) - lead_in
    price = price[price.index >= start].rename("observed")
    if venue in HOURLY_VENUES:
        hour = price.index.tz_convert("UTC").floor("h")
        both = price.groupby(hour).agg(["first", "count", "max", "min"])
        hourly = both["first"].where(
            (both["count"] == 2) & (both["max"] == both["min"])
        )
        price = hourly.set_axis(hourly.index.tz_convert(TZ)).rename("observed")
    return price


def _run(path: Path) -> pd.Series:
    return _local(_read(path)["DA"]).astype(float)


def competitive(years: int | Iterable[int]) -> pd.Series:
    """The competitive run, the case ``day_ahead`` (every unit bids its marginal cost net of its
    support payment), GBP/MWh, half-hourly: the saved run where there is one, otherwise the merit
    order of the scenario's own offers, which that run reproduces to 1e-13."""
    parts = []
    for year in _years(years):
        saved = paths.prices_file(paths.run_name(year, "day_ahead"))
        if saved.exists():
            parts.append(_run(saved))
        else:
            merit = _delivery(
                _read(folder(year) / "reference_prices.csv")["scenario_merit_order"],
                year,
            )
            parts.append(_local(merit).astype(float))
    return pd.concat(parts).sort_index().rename(COMPETITIVE)


def run(year: int, case: str) -> pd.Series:
    """Any other saved run of ``year`` (``results/prices/gb_<year>_<case>.csv``, its day-ahead
    price), GBP/MWh, half-hourly: e.g. ``day_ahead_storage``."""
    path = paths.prices_file(paths.run_name(year, case))
    if not path.exists():
        raise FileNotFoundError(f"no saved run {path}")
    return _run(path).rename(case)


def abm(
    year: int, arm: str, policies: str = "best", seeds: Iterable[int] | None = None
) -> pd.DataFrame:
    """The delivery year run with the trained policies of a learning ``arm`` (e.g.
    ``learning_novdec_own_e100``), one column per seed, GBP/MWh, half-hourly: with the policies of
    the best evaluation episode (``policies="best"``) or those at the end of training ("last")."""
    if policies not in ("best", "last"):
        raise ValueError(f"policies must be 'best' or 'last', not {policies!r}")
    suffix = "_year.csv" if policies == "best" else "_year_last.csv"
    prefix = f"{paths.scenario_name(year)}_{arm}_s"
    pattern = re.compile(re.escape(prefix) + r"(\d+)" + re.escape(suffix) + "$")
    found = {
        int(m.group(1)): path
        for path in _prices_dir().glob(f"{prefix}*{suffix}")
        if (m := pattern.match(path.name))
    }
    wanted = sorted(found) if seeds is None else sorted(seeds)
    missing = [seed for seed in wanted if seed not in found]
    if not wanted or missing:
        raise FileNotFoundError(
            f"no year runs of {prefix}<seed>{suffix} for seeds {missing or 'any'} in {_prices_dir()}"
        )
    return pd.DataFrame({seed: _run(found[seed]) for seed in wanted})


def _unit_mw(
    units: pd.DataFrame, availability: pd.DataFrame, names: list[str]
) -> pd.Series:
    """MW of ``names``: availability (1 where a unit has no column) x max_power, summed."""
    share = pd.DataFrame(1.0, index=availability.index, columns=names)
    share.update(
        availability.reindex(columns=[n for n in names if n in availability.columns])
    )
    return share.mul(units.loc[names, "max_power"], axis=1).sum(axis=1)


def exog(years: int | Iterable[int]) -> pd.DataFrame:
    """The explanatory series of ``years`` from the scenario folders, half-hourly, Europe/London:

    wind, solar                  outturn, GW: availability x max_power of the Wind and Solar
                                 units (the metered outturn, embedded wind included)
    wind_forecast, solar_forecast  the same with the 09:00 D-1 availability forecast, GW
    demand, demand_forecast      the demand unit's demand less the export capacity of the
                                 interconnectors: GB demand net of nuclear and pumped
                                 storage, GW
    residual_demand              demand - wind - solar, GW
    available_capacity           availability x max_power of every other unit but the export
                                 units and unserved demand, imports included, GW
    gas, carbon                  natural gas (GBP/MWh thermal) and carbon (GBP/t) prices
    wind_tx_forecast, tsd_forecast or nd_forecast
                                 from ``penetration_forecasts.csv`` where the folder has it:
                                 transmission-connected wind and transmission system (or
                                 national) demand as forecast at 09:00 D-1, GW
    """
    frames = []
    for year in _years(years):
        path = folder(year)
        units = pd.read_csv(path / "powerplant_units.csv").set_index("name")
        technology = units["technology"]
        names = {
            WIND: list(units.index[technology == WIND]),
            SOLAR: list(units.index[technology == SOLAR]),
            EXPORT: list(units.index[technology == EXPORT]),
        }
        others = list(units.index[~technology.isin([WIND, SOLAR, EXPORT, UNSERVED])])
        outturn = _read(path / "availability_df.csv.gz")
        # a unit without a forecast column has no forecast error (as in the framework's loader)
        forecast = _read(path / "availability_forecast_df.csv.gz").combine_first(
            outturn
        )
        demand = _read(path / "demand_df.csv")[DEMAND_UNIT]
        demand_forecast = _read(path / "demand_forecast_df.csv")[DEMAND_UNIT]
        export = _unit_mw(units, outturn, names[EXPORT])
        export_forecast = _unit_mw(units, forecast, names[EXPORT])
        fuel = _read(path / "fuel_prices_df.csv")
        table = pd.DataFrame(
            {
                "wind": _unit_mw(units, outturn, names[WIND]),
                "solar": _unit_mw(units, outturn, names[SOLAR]),
                "demand": demand - export,
                "wind_forecast": _unit_mw(units, forecast, names[WIND]),
                "solar_forecast": _unit_mw(units, forecast, names[SOLAR]),
                "demand_forecast": demand_forecast - export_forecast,
                "available_capacity": _unit_mw(units, outturn, others),
            }
        )
        table.insert(
            3, "residual_demand", table["demand"] - table["wind"] - table["solar"]
        )
        table = table / 1000.0  # MW -> GW
        table["gas"] = fuel["natural gas"]
        table["carbon"] = fuel["co2"]
        extra = path / "penetration_forecasts.csv"
        if extra.exists():
            penetration = _read(extra) / 1000.0
            table = table.join(
                penetration.rename(columns=lambda c: c.removesuffix("_mw"))
            )
        frames.append(_delivery(table, year))
    return _local(pd.concat(frames).sort_index())


def _training_window(year: int, arm: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """First and last delivery period (UTC) of the arm's training: from ``config.yaml`` (the case
    of its first seed, or the arm itself; the lead-in day delivers nothing), or, where the config
    no longer holds the case, from the training run's saved prices, which span the window."""
    config = yaml.safe_load((folder(year) / "config.yaml").read_text(encoding="utf-8"))
    cases = sorted(
        case for case in config if re.fullmatch(re.escape(arm) + r"(_s\d+)?", case)
    )
    if cases:
        case = config[cases[0]]
        start = pd.Timestamp(case["start_date"], tz="UTC") + pd.Timedelta(days=1)
        return start, pd.Timestamp(case["end_date"], tz="UTC") - PERIOD
    prefix = f"{paths.scenario_name(year)}_{arm}"
    pattern = re.compile(re.escape(prefix) + r"(_s\d+)?\.csv")
    saved = sorted(
        path
        for path in _prices_dir().glob(f"{prefix}*.csv")
        if pattern.fullmatch(path.name)
    )
    if not saved:
        return None
    index = pd.read_csv(saved[0], usecols=[0], index_col=0, parse_dates=True).index
    return index[0].tz_localize("UTC"), index[-1].tz_localize("UTC")


def periods(year: int, arm: str | None = None) -> dict[str, tuple]:
    """The windows of ``year``: the delivery year as ``year`` and ``headline`` (the contract takes
    the calibration window out of the headline), and for a learning ``arm`` its training window
    as ``calibration``: from ``config.yaml``, or from the training run's saved prices
    (``results/prices/gb_<year>_<arm>_s<seed>.csv``) where the config no longer holds the case."""
    windows = {
        HEADLINE: (f"{year}-01-01", f"{year}-12-31"),
        YEAR: (f"{year}-01-01", f"{year}-12-31"),
    }
    if arm is None:
        return windows
    training = _training_window(year, arm)
    if training is None:
        raise KeyError(
            f"no training window of {arm!r}: no study case {arm} or {arm}_s<seed> in "
            f"{folder(year) / 'config.yaml'} and no {paths.scenario_name(year)}_{arm}_s<seed>.csv in {_prices_dir()}"
        )
    windows[CALIBRATION] = tuple(stamp.tz_convert(TZ) for stamp in training)
    return windows


def battery(year: int) -> Battery:
    """The scenario's battery fleet (``scenario_meta.json``) as the battery of the value criterion."""
    fleet = json.loads(
        (folder(year) / "scenario_meta.json").read_text(encoding="utf-8")
    )["battery_fleet"]
    return Battery(
        power=fleet["power_mw"],
        energy=fleet["energy_mwh"],
        round_trip_efficiency=fleet["round_trip_efficiency"],
    )


def load(
    years: int | Iterable[int],
    arm: str | None = None,
    venue: str = "n2ex_day_ahead",
    policies: str = "best",
    seeds: Iterable[int] | None = None,
    runs: dict[str, str] | None = None,
    regimes: dict[str, tuple] | None = None,
) -> ValidationData:
    """The data contract for ``years``: the observed price of ``venue``, the naive forecast, the
    competitive run, the agent-based ``arm`` (one year only) with its seeds as "abm", further
    saved runs (name -> case, e.g. {"storage": "day_ahead_storage"}), the explanatory series, the
    windows (``periods``) and ``regimes``. The observed price starts a year early where the
    previous folder has it: the naive forecast needs the week before, the year-to-year benchmark
    of the profiles the year before. The models are driven by the outturn, the observed price by
    the forecasts (the contract's defaults)."""
    years = _years(years)
    if arm is not None and len(years) > 1:
        raise ValueError(
            "an agent-based arm belongs to one year: load the years one at a time"
        )
    models: dict[str, pd.Series | pd.DataFrame] = {COMPETITIVE: competitive(years)}
    if arm is not None:
        models[ABM] = abm(years[0], arm, policies, seeds)
    for name, case in (runs or {}).items():
        models[name] = pd.concat([run(year, case) for year in years]).rename(name)
    windows = (
        periods(years[0], arm)
        if len(years) == 1
        else {
            HEADLINE: (f"{years[0]}-01-01", f"{years[-1]}-12-31"),
            YEAR: (f"{years[0]}-01-01", f"{years[-1]}-12-31"),
        }
    )
    price = observed(years, venue, lead_in=pd.Timedelta(days=366))
    return ValidationData(
        price, models, exog(years), periods=windows, regimes=dict(regimes or {})
    )
