# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""The data contract of the validation package and the calendar every criterion shares.

    observed   pd.Series of prices in GBP/MWh on a tz-aware DatetimeIndex. Its resolution (hourly
               or half-hourly) is inferred from the index. A clock-change day has 23 or 25 hours
               (46 or 50 half-hours): nothing here assumes 24 periods a day.
    models     name -> pd.Series (a deterministic run) or pd.DataFrame (one column per seed), at
               any resolution: a finer model is averaged to the observed periods, a coarser one
               repeated over them. "naive", the observed price one week earlier, is added.
    exog       pd.DataFrame of explanatory series, aligned in the same way. Wind, solar and demand
               (GW) come in two information sets: the outturn as ``wind``, ``solar``, ``demand``
               and the day-ahead forecast (the 09:00 D-1 vintage) as ``wind_forecast``,
               ``solar_forecast``, ``demand_forecast``; further columns such as residual_demand,
               available_capacity (GW), gas (GBP/MWh thermal) and carbon (GBP/t) once. For the
               penetration axis of the causal reference: ``wind_tx_forecast`` (transmission-
               connected wind) and ``tsd_forecast`` (transmission system demand) or, without it,
               ``nd_forecast`` (national demand), all GW as forecast at 09:00 D-1.
    information
               name -> "forecast" or "outturn": the information set a price series was formed on.
               The observed price is formed on the forecasts, a model by default on the outturn
               (the agent-based model is driven by it); the mechanism checks use each price's own.
    periods    name -> window:
                 "headline"     the window every criterion uses unless told otherwise: data never
                                used for tuning. The calibration window is always taken out of it,
                                and without a headline window it is every period but those.
                 "calibration"  the data used for tuning (e.g. the training window of the agents),
                                scored separately and shaded in the figures.
                 "year"         the full delivery year, reported as a secondary result, in-sample
                                when it overlaps the calibration window (``in_sample``).
               The Diebold-Mariano tests and the comparisons with the naive forecast and LEAR are
               made on the headline window only.
    regimes    name -> window, optional: the mechanism checks run per regime.

A window is a (start, end) pair applied as pandas label slicing, both ends included (a date string
includes its whole day), or a list of such pairs. A time without a zone is read in the zone of the
data, Europe/London unless set otherwise.

Every model is scored on the same periods, the common sample: the periods of the window in which
the observed price and every model (every seed of an ensemble) have a value. Ratios such as rMAE
and the Diebold-Mariano tests then compare like with like.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TZ = "Europe/London"

OBSERVED = "observed"
NAIVE = "naive"
COMPETITIVE = "competitive"
ABM = "abm"
LEAR = "lear"

HEADLINE = "headline"
CALIBRATION = "calibration"
YEAR = "year"

FORECAST = "forecast"
OUTTURN = "outturn"
INFORMATION_SETS = (FORECAST, OUTTURN)
# carried in both information sets: <name> is the outturn, <name>_forecast the day-ahead forecast
VINTAGED = ("wind", "solar", "demand")


def vintage(name: str, information: str) -> str:
    """The exog column of variable ``name`` in an information set: ``name`` for the outturn,
    ``name_forecast`` for the day-ahead forecast; other variables have one column."""
    if information not in INFORMATION_SETS:
        raise ValueError(
            f"information set must be one of {INFORMATION_SETS}, not {information!r}"
        )
    return f"{name}_forecast" if information == FORECAST and name in VINTAGED else name


Bound = str | dt.date | dt.datetime | pd.Timestamp
Window = tuple[Bound, Bound]
Windows = Window | list[Window]
Model = pd.Series | pd.DataFrame

HOUR = pd.Timedelta(hours=1)


def check_index(frame: Model, name: str) -> None:
    """Raises unless ``frame`` has a tz-aware, unique and increasing DatetimeIndex."""
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(
            f"{name}: the index must be a DatetimeIndex, not {type(index).__name__}"
        )
    if index.tz is None:
        raise ValueError(
            f"{name}: the index must be tz-aware, e.g. index.tz_localize('UTC').tz_convert('{TZ}')"
        )
    if not index.is_unique:
        raise ValueError(f"{name}: the index has duplicate timestamps")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name}: the index must be sorted")


def infer_resolution(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Length of one period: the most frequent step between consecutive timestamps. Steps are
    absolute time, so clock changes and gaps do not affect it."""
    if len(index) < 2:
        raise ValueError("at least two timestamps are needed to infer the resolution")
    step = pd.Series(index[1:] - index[:-1]).mode().iloc[0]
    if step <= pd.Timedelta(0):
        raise ValueError("the index must increase")
    return pd.Timedelta(step)


def _floor(index: pd.DatetimeIndex, resolution: pd.Timedelta) -> pd.DatetimeIndex:
    """Start of the period of length ``resolution`` that holds each timestamp. Floored in UTC:
    for periods that divide an hour, in a zone whose offsets are whole hours, these are the local
    periods, and no local time is missing or repeated on the way."""
    if HOUR % resolution != pd.Timedelta(0):
        raise ValueError(f"periods of {resolution} do not divide an hour")
    return index.tz_convert("UTC").floor(resolution).tz_convert(index.tz)


def to_resolution(frame: Model, resolution: pd.Timedelta) -> Model:
    """``frame`` averaged into periods of length ``resolution``, each labelled by its start. A
    period lacking a value for any of its sub-periods is NaN, so an hour with a missing half-hour
    is left out, as ``assume_gb.compare.hourly`` does."""
    own = infer_resolution(frame.index)
    if resolution == own:
        return frame
    if resolution < own or resolution % own != pd.Timedelta(0):
        raise ValueError(
            f"cannot average periods of {own} into periods of {resolution}"
        )
    grouped = frame.groupby(_floor(frame.index, resolution))
    return grouped.mean().where(grouped.count() == resolution // own)


def align(frame: Model, index: pd.DatetimeIndex) -> Model:
    """``frame`` on the periods of ``index``: averaged where ``frame`` is finer (see
    ``to_resolution``), repeated over the sub-periods where it is coarser, NaN where it has no
    value."""
    frame = frame.set_axis(frame.index.tz_convert(index.tz))
    own, target = infer_resolution(frame.index), infer_resolution(index)
    if own < target:
        return to_resolution(frame, target).reindex(index)
    if own > target:
        if own % target != pd.Timedelta(0):
            raise ValueError(f"cannot repeat periods of {own} over periods of {target}")
        return frame.reindex(_floor(index, own)).set_axis(index)
    return frame.reindex(index)


def naive_week_ago(observed: pd.Series) -> pd.Series:
    """The naive forecast: the observed price of the same local delivery period one week earlier,
    GBP/MWh. That is 168 hours earlier, or 167 or 169 across a clock change, so that the period
    keeps its wall-clock time; where that time does not exist or is repeated a week earlier, the
    price 168 hours earlier. NaN for the first week. It is the denominator of rMAE
    [lagoForecastingDayaheadElectricity2021], whose naive forecast differs on Tuesday to Friday:
    there it is the price of the day before."""
    index = observed.index
    wall = index.tz_localize(None) - pd.Timedelta(days=7)
    source = wall.tz_localize(index.tz, ambiguous="NaT", nonexistent="NaT")
    same_clock = observed.reindex(source).to_numpy()
    absolute = observed.reindex(index - pd.Timedelta(days=7)).to_numpy()
    return pd.Series(
        np.where(source.isna(), absolute, same_clock), index=index, name=NAIVE
    )


def local_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Local delivery day of each period, as midnight without a zone; a clock-change day keeps
    its 23 or 25 hours."""
    return index.tz_localize(None).normalize()


def period_of_day(
    index: pd.DatetimeIndex, resolution: pd.Timedelta | None = None
) -> np.ndarray:
    """Wall-clock period of the local day, 0 for the period that starts at midnight (at
    half-hourly resolution: hour x 2 + minute // 30). On the 25-hour day the periods of the
    repeated hour share their numbers; on the 23-hour day those of the skipped hour are absent."""
    resolution = resolution if resolution is not None else infer_resolution(index)
    minutes = np.asarray(index.hour * 60 + index.minute)
    return minutes // (resolution // pd.Timedelta(minutes=1))


def _bound(value: Bound, tz) -> str | pd.Timestamp:
    if isinstance(value, str):
        return value  # pandas reads it in the zone of the index; a date string spans its day
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return value.isoformat()
    stamp = pd.Timestamp(value)
    return stamp.tz_localize(tz) if stamp.tz is None else stamp.tz_convert(tz)


def window_mask(index: pd.DatetimeIndex, window: Windows | None) -> np.ndarray:
    """Boolean mask of the periods of ``index`` in ``window``; every period for None."""
    if window is None:
        return np.ones(len(index), dtype=bool)
    ranges = window if isinstance(window, list) else [window]
    positions = pd.Series(np.arange(len(index)), index=index)
    mask = np.zeros(len(index), dtype=bool)
    for start, end in ranges:
        mask[
            positions.loc[_bound(start, index.tz) : _bound(end, index.tz)].to_numpy()
        ] = True
    return mask


@dataclass
class ValidationData:
    """Observed prices, the models to validate and what explains them, on one clock: the data
    contract of every criterion. On construction the indexes are checked and converted to ``tz``,
    the resolution is inferred from ``observed``, every model and ``exog`` are aligned to the
    observed periods, and "naive" is added unless it is given or ``add_naive`` is False. An
    ensemble enters the point metrics as the mean of its seeds (``point_forecast="median"`` for
    the median). ``information`` names the information set of a price series where the default
    (forecast for the observed price, outturn for a model) does not hold."""

    observed: pd.Series
    models: dict[str, Model]
    exog: pd.DataFrame | None = None
    periods: dict[str, Windows] = field(default_factory=dict)
    regimes: dict[str, Windows] = field(default_factory=dict)
    information: dict[str, str] = field(default_factory=dict)
    tz: str = TZ
    add_naive: bool = True
    point_forecast: str = "mean"
    resolution: pd.Timedelta = field(init=False)

    def __post_init__(self) -> None:
        if self.point_forecast not in ("mean", "median"):
            raise ValueError(
                f"point_forecast must be 'mean' or 'median', not {self.point_forecast!r}"
            )
        for name, information in self.information.items():
            if information not in INFORMATION_SETS:
                raise ValueError(
                    f"{name}: information set must be one of {INFORMATION_SETS}"
                )
        check_index(self.observed, OBSERVED)
        observed = self.observed.set_axis(self.observed.index.tz_convert(self.tz))
        self.observed = observed.astype(float).rename(OBSERVED)
        self.resolution = infer_resolution(self.observed.index)
        models: dict[str, Model] = {}
        if self.add_naive and NAIVE not in self.models:
            models[NAIVE] = naive_week_ago(self.observed)
        for name, model in self.models.items():
            check_index(model, name)
            models[name] = align(model.astype(float), self.observed.index)
        self.models = models
        if self.exog is not None:
            check_index(self.exog, "exog")
            self.exog = align(self.exog.astype(float), self.observed.index)

    @property
    def names(self) -> list[str]:
        return list(self.models)

    def is_ensemble(self, name: str) -> bool:
        """True for a model given as several seeds."""
        model = self.models[name]
        return isinstance(model, pd.DataFrame) and model.shape[1] > 1

    def members(self, name: str) -> pd.DataFrame:
        """The seeds of model ``name``, one column each; a single run is one column."""
        model = self.models[name]
        return model if isinstance(model, pd.DataFrame) else model.to_frame(name)

    def point(self, name: str) -> pd.Series:
        """Model ``name`` as one price series: a single run as it is, an ensemble as the mean (or
        median) of its seeds, NaN where a seed has no value."""
        members = self.members(name)
        if members.shape[1] == 1:
            return members.iloc[:, 0].rename(name)
        if self.point_forecast == "median":
            return members.median(axis=1, skipna=False).rename(name)
        return members.mean(axis=1, skipna=False).rename(name)

    def information_set(self, name: str) -> str:
        """The information set price series ``name`` was formed on: as given in ``information``,
        otherwise "forecast" for the observed price and "outturn" for a model."""
        return self.information.get(name, FORECAST if name == OBSERVED else OUTTURN)

    def window(self, window: str | Windows | None = HEADLINE) -> np.ndarray:
        """Mask of the observed periods in ``window``: the name of one of ``periods`` or
        ``regimes``, a window, or None for every period. The headline window never contains a
        period of the calibration window, and without a headline window it is every other
        period."""
        index = self.observed.index
        if not isinstance(window, str):
            return window_mask(index, window)
        if window in self.periods:
            mask = window_mask(index, self.periods[window])
        elif window in self.regimes:
            mask = window_mask(index, self.regimes[window])
        elif window == HEADLINE:
            mask = window_mask(index, None)
        else:
            raise KeyError(
                f"no window {window!r}: periods {list(self.periods)}, regimes {list(self.regimes)}"
            )
        if window == HEADLINE and CALIBRATION in self.periods:
            mask &= ~window_mask(index, self.periods[CALIBRATION])
        return mask

    def in_sample(self, window: str | Windows | None = HEADLINE) -> bool:
        """True when ``window`` overlaps the calibration window, i.e. holds data used for
        tuning; its scores are then in-sample."""
        if CALIBRATION not in self.periods:
            return False
        calibration = window_mask(self.observed.index, self.periods[CALIBRATION])
        return bool((self.window(window) & calibration).any())

    def common(
        self,
        window: str | Windows | None = HEADLINE,
        models: list[str] | None = None,
        regime: str | None = None,
    ) -> pd.DatetimeIndex:
        """The common sample: the periods of ``window`` (and of ``regime``, if given) in which the
        observed price and every model in ``models`` (all by default; every seed of an ensemble)
        have a value."""
        keep = self.window(window) & self.observed.notna().to_numpy()
        if regime is not None:
            keep &= self.window(regime)
        for name in self.names if models is None else models:
            keep &= self.members(name).notna().all(axis=1).to_numpy()
        return self.observed.index[keep]

    def sample(
        self,
        window: str | Windows | None = HEADLINE,
        models: list[str] | None = None,
        regime: str | None = None,
    ) -> pd.DataFrame:
        """The observed price (first column) and the point forecast of each model in ``models``
        (all by default) on the common sample, GBP/MWh."""
        names = self.names if models is None else list(models)
        columns = [self.observed] + [self.point(name) for name in names]
        return pd.concat(columns, axis=1).loc[self.common(window, names, regime)]

    def ensemble(
        self,
        name: str,
        window: str | Windows | None = HEADLINE,
        models: list[str] | None = None,
        regime: str | None = None,
    ) -> pd.DataFrame:
        """The seeds of model ``name`` on the same common sample as ``sample``, GBP/MWh."""
        return self.members(name).loc[self.common(window, models, regime)]

    def explanatory(
        self,
        window: str | Windows | None = HEADLINE,
        models: list[str] | None = None,
        regime: str | None = None,
    ) -> pd.DataFrame:
        """``exog`` on the same common sample as ``sample``."""
        if self.exog is None:
            raise ValueError("no explanatory series (exog) were given")
        return self.exog.loc[self.common(window, models, regime)]
