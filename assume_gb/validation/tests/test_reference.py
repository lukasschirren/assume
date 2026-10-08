# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The reference curves of Cacciarelli et al.: every market and technology is there, values are
interpolated linearly and never extrapolated, and the hidden part of the solar band stays empty."""

import numpy as np
import pytest

from assume_gb.validation import reference as R


def test_every_curve_is_there_on_increasing_penetration():
    for market in R.MARKETS:
        for technology in R.TECHNOLOGIES:
            curve = R.cate_curve(market, technology)
            assert curve.columns.tolist() == ["estimate", "ci80_low", "ci80_high"]
            assert curve.index.is_monotonic_increasing and len(curve) >= 24
            band = curve.dropna()
            assert (band.ci80_low <= band.estimate).all()
            assert (band.estimate <= band.ci80_high).all()
    wind = R.cate_curve("NordPool", "wind")
    assert wind.loc[3.0, "estimate"] == -8.35
    assert wind.index.min() == 3.0 and wind.index.max() == 50.0


def test_values_are_interpolated_linearly_and_never_extrapolated():
    at = R.cate_at([3.0, 3.5, 10.0, 11.25, 2.0, 55.0], "APX", "wind")
    assert at.loc[3.0, "estimate"] == pytest.approx(-6.78)
    assert at.loc[3.5, "estimate"] == pytest.approx((-6.78 - 6.00) / 2)
    assert at.loc[11.25, "ci80_high"] == pytest.approx((-2.06 - 1.43) / 2)
    assert at.loc[[2.0, 55.0]].isna().all().all()


def test_the_hidden_part_of_the_solar_band_stays_empty():
    at = R.cate_at([7.0, 7.5, 8.0], "APX", "solar")
    assert at.loc[7.0, "ci80_low"] == pytest.approx(-2.15)
    assert np.isnan(at.loc[7.5, "ci80_low"]) and np.isnan(at.loc[8.0, "ci80_low"])
    assert at.loc[8.0, "estimate"] == pytest.approx(-0.84)
    assert at.loc[7.5, "ci80_high"] == pytest.approx(-0.10)


def test_unknown_markets_and_technologies_are_refused():
    with pytest.raises(ValueError):
        R.cate_curve("EPEX", "wind")
    with pytest.raises(ValueError):
        R.cate_curve("APX", "nuclear")
