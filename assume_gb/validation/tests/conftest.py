# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fixtures of the validation tests: two years of synthetic half-hourly data (2022 and 2023, local
time). As in the GB runs, the calibration window (four weeks of training, 6 November to 3 December
2023) lies inside the delivery year 2023; the headline window is 2023 without it, and 2022 is the
lead-in for the naive forecast and the profile benchmark.

    python -m pytest assume_gb/validation
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from assume_gb.validation.contract import ValidationData  # noqa: E402
from assume_gb.validation.tests import synthetic  # noqa: E402

PERIODS = {
    "calibration": ("2023-11-06", "2023-12-03"),
    "headline": ("2023-01-01", "2023-12-31"),
    "year": ("2023-01-01", "2023-12-31"),
}


@pytest.fixture(scope="session")
def exog() -> pd.DataFrame:
    return synthetic.exog()


@pytest.fixture(scope="session")
def observed(exog) -> pd.Series:
    return synthetic.prices(exog)


@pytest.fixture
def data(observed, exog) -> ValidationData:
    """The observed prices with the four distorted models, the exog and the windows."""
    models = {
        "shifted": synthetic.shifted(observed),
        "flattened": synthetic.flattened(observed),
        "lagged": synthetic.lagged(observed),
        "noisy": synthetic.noisy(observed),
    }
    return ValidationData(observed, models, exog, periods=dict(PERIODS))


@pytest.fixture(scope="session")
def ensemble_data(observed, exog) -> ValidationData:
    """A competitive run formed on the outturn and an agent-based ensemble of three seeds 3
    GBP/MWh above it, the naive forecast; tuning in May, the headline window June to August, two
    regimes."""
    competitive = synthetic.prices(exog, "outturn", seed=1)
    noise = np.random.default_rng(11).normal(0.0, 5.0, (len(competitive), 3))
    seeds = pd.DataFrame(
        competitive.to_numpy()[:, None] + 3.0 + noise, index=competitive.index
    )
    return ValidationData(
        observed,
        {"competitive": competitive, "abm": seeds},
        exog,
        periods={
            "calibration": ("2023-05-01", "2023-05-31"),
            "headline": ("2023-06-01", "2023-08-31"),
            "year": ("2023-05-01", "2023-08-31"),
        },
        regimes={
            "June": ("2023-06-01", "2023-06-30"),
            "July and August": ("2023-07-01", "2023-08-31"),
        },
    )
