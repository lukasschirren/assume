# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""The data contract: resolution, the local calendar across clock changes, alignment of models to
the observed periods, the naive forecast, windows and the common sample."""

import numpy as np
import pandas as pd
import pytest

from assume_gb.validation.contract import (
    NAIVE,
    TZ,
    ValidationData,
    align,
    infer_resolution,
    local_day,
    naive_week_ago,
    period_of_day,
    to_resolution,
    vintage,
    window_mask,
)
from assume_gb.validation.tests import synthetic

HALF_HOUR, HOUR = pd.Timedelta(minutes=30), pd.Timedelta(hours=1)


def at(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=TZ)


def test_resolution_is_inferred_and_clock_change_days_keep_their_length():
    idx = synthetic.index("2023-03-20", "2023-11-05")
    assert infer_resolution(idx) == HALF_HOUR
    assert infer_resolution(idx[::2]) == HOUR
    per_day = pd.Series(1, index=idx).groupby(local_day(idx)).size()
    assert per_day[pd.Timestamp("2023-03-26")] == 46
    assert per_day[pd.Timestamp("2023-10-29")] == 50
    assert per_day[pd.Timestamp("2023-06-01")] == 48


def test_period_of_day_follows_the_wall_clock_on_clock_change_days():
    autumn = synthetic.index("2023-10-29", "2023-10-30")
    assert len(autumn) == 50
    assert period_of_day(autumn)[:7].tolist() == [
        0,
        1,
        2,
        3,
        2,
        3,
        4,
    ]  # 01:00-02:00 twice
    spring = synthetic.index("2023-03-26", "2023-03-27")
    assert len(spring) == 46
    assert period_of_day(spring)[:3].tolist() == [0, 1, 4]  # 01:00-02:00 skipped
    assert period_of_day(spring[::2], HOUR)[:3].tolist() == [0, 2, 3]


def test_half_hours_are_averaged_into_hours_and_an_incomplete_hour_is_dropped():
    half_hours = pd.Series(
        np.arange(8.0), index=synthetic.index("2023-06-01 00:00", "2023-06-01 04:00")
    )
    half_hours.iloc[3] = np.nan
    hourly = align(half_hours, half_hours.index[::2])
    assert hourly.iloc[[0, 2, 3]].tolist() == [0.5, 4.5, 6.5]
    assert np.isnan(hourly.iloc[1])
    # the 25-hour day becomes 25 hours, the repeated hour averaged on its own
    autumn = pd.Series(
        np.arange(50.0), index=synthetic.index("2023-10-29", "2023-10-30")
    )
    hours = to_resolution(autumn, HOUR)
    assert len(hours) == 25 and hours.iloc[1:3].tolist() == [2.5, 4.5]


def test_an_hourly_model_is_repeated_over_half_hours():
    hourly = pd.Series(
        [1.0, 2.0], index=pd.date_range("2023-06-01", periods=2, freq="h", tz=TZ)
    )
    half_hours = synthetic.index("2023-06-01 00:00", "2023-06-01 02:00")
    assert align(hourly, half_hours).tolist() == [1.0, 1.0, 2.0, 2.0]


def test_naive_is_the_same_wall_clock_period_one_week_earlier(observed):
    naive = naive_week_ago(observed)
    t = at("2023-06-15 18:00")
    assert naive[t] == observed[t - pd.Timedelta(days=7)]
    # a week after the spring clock change the same wall-clock time is 167 hours back
    t, source = at("2023-03-28 12:00"), at("2023-03-21 12:00")
    assert t - source == pd.Timedelta(hours=167) and naive[t] == observed[source]
    # a week after the skipped hour of 26 March there is no such time: 168 hours back
    t = at("2023-04-02 01:30")
    assert naive[t] == observed[t - pd.Timedelta(days=7)]
    # only the first week has no naive forecast
    assert naive.iloc[: 7 * 48].isna().all() and naive.iloc[7 * 48 :].notna().all()


def test_windows_include_whole_days_read_local_times_and_can_be_unions():
    idx = synthetic.index("2023-01-01", "2023-01-10")
    assert window_mask(idx, ("2023-01-02", "2023-01-03")).sum() == 96
    assert (
        window_mask(
            idx, [("2023-01-02", "2023-01-02"), ("2023-01-05", "2023-01-05")]
        ).sum()
        == 96
    )
    local = window_mask(
        idx, (pd.Timestamp("2023-01-02 00:00"), pd.Timestamp("2023-01-02 00:30"))
    )
    assert idx[local].tolist() == [at("2023-01-02 00:00"), at("2023-01-02 00:30")]
    assert window_mask(idx, None).all()


def test_the_contract_requires_a_tz_aware_index(observed):
    naive_clock = observed.set_axis(observed.index.tz_localize(None))
    with pytest.raises(ValueError, match="tz-aware"):
        ValidationData(naive_clock, {})
    with pytest.raises(ValueError, match="tz-aware"):
        ValidationData(observed, {"model": naive_clock})


def test_models_are_aligned_and_scored_on_one_common_sample(observed):
    hourly_observed = to_resolution(observed, HOUR)
    seeds = synthetic.ensemble(observed, members=3)
    seeds.loc[at("2023-02-01 10:30"), 1] = np.nan  # one seed misses one half-hour
    data = ValidationData(
        hourly_observed,
        {"lagged": synthetic.lagged(observed), "seeds": seeds},
        periods={"headline": ("2023-01-01", "2023-12-31")},
    )
    assert data.resolution == HOUR and data.names == [NAIVE, "lagged", "seeds"]
    assert data.is_ensemble("seeds") and not data.is_ensemble("lagged")
    sample = data.sample()
    assert sample.columns.tolist() == ["observed", NAIVE, "lagged", "seeds"]
    assert sample.notna().all().all()
    assert sample.index[0] == at("2023-01-01 00:00") and sample.index[-1] == at(
        "2023-12-31 23:00"
    )
    # the hour with the missing seed is out of the sample of every model
    assert at("2023-02-01 10:00") not in sample.index and len(sample) == 8760 - 1
    # an ensemble enters as the mean of its seeds, each averaged over the hour
    t = at("2023-07-01 18:00")
    hour_of_seeds = seeds.loc[t : t + pd.Timedelta(minutes=30)].mean()
    assert sample.loc[t, "seeds"] == pytest.approx(hour_of_seeds.mean())
    assert data.ensemble("seeds").index.equals(sample.index)
    with pytest.raises(KeyError):
        data.sample("winter")


def test_the_headline_window_never_holds_calibration_data(observed):
    calibration = ("2023-11-06", "2023-12-03")
    data = ValidationData(
        observed,
        {"model": synthetic.noisy(observed)},
        periods={"calibration": calibration, "headline": ("2023-01-01", "2023-12-31")},
        regimes={
            "winter": [("2023-01-01", "2023-02-28"), ("2023-12-01", "2023-12-31")]
        },
    )
    headline = data.sample()
    assert len(headline) == (365 - 28) * 48  # the calibration days are taken out
    assert not headline.index.isin(data.sample(calibration).index).any()
    assert data.sample(calibration).index[0] == at("2023-11-06 00:00")
    winter = data.sample(regime="winter")  # the headline periods of the regime
    assert len(winter) == (31 + 28 + 28) * 48
    # without a headline window: every period but the calibration days (and the naive's first week)
    data = ValidationData(
        observed,
        {"model": synthetic.noisy(observed)},
        periods={"calibration": calibration},
    )
    assert len(data.sample()) == len(observed) - 7 * 48 - 28 * 48
    data = ValidationData(observed, {"model": synthetic.noisy(observed)})
    assert len(data.sample()) == len(observed) - 7 * 48


def test_a_window_overlapping_the_calibration_window_is_in_sample(data):
    assert data.in_sample("year") and data.in_sample("calibration")
    assert not data.in_sample("headline") and not data.in_sample(
        ("2023-01-01", "2023-06-30")
    )


def test_information_sets(observed, exog):
    data = ValidationData(observed, {"abm": synthetic.noisy(observed)}, exog)
    assert (
        data.information_set("observed") == "forecast"
        and data.information_set("abm") == "outturn"
    )
    data = ValidationData(
        observed,
        {"lear": synthetic.noisy(observed)},
        exog,
        information={"lear": "forecast"},
    )
    assert data.information_set("lear") == "forecast"
    assert (
        vintage("wind", "forecast") == "wind_forecast"
        and vintage("wind", "outturn") == "wind"
    )
    assert vintage("gas", "forecast") == "gas"
    with pytest.raises(ValueError):
        ValidationData(observed, {}, information={"observed": "nowcast"})
