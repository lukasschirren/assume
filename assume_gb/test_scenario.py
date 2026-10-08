# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The built GB scenario folders, simulated with the framework as it ships: the first week of
each must clear at the merit order of its own offers. Skipped for a year that is not built.

    python -m pytest assume_gb
"""

import numpy as np
import pandas as pd
import pytest

from assume_gb import compare, convert, paths

BUILT = sorted(
    p.name for p in paths.scenarios_dir().glob("gb_*") if (p / "config.yaml").exists()
)


@pytest.mark.skipif(
    not BUILT, reason="no scenario folder is built (python -m assume_gb build)"
)
@pytest.mark.parametrize("name", BUILT)
def test_first_week_clears_at_the_merit_order_of_the_scenario(name):
    world, _ = compare.run_scenario(name, convert.WEEK_CASE)
    price = compare.clearing_prices(world)[convert.MARKET_ID]
    reference = pd.read_csv(
        paths.scenarios_dir() / name / "reference_prices.csv",
        index_col=0,
        parse_dates=True,
    )

    # one auction per day for the 48 half-hours of the next day
    assert price.index.equals(reference.index[: 7 * 48])
    expected = reference.loc[price.index, "scenario_merit_order"].to_numpy()
    cleared = np.isfinite(expected)  # a period without demand has no price to compare
    np.testing.assert_allclose(
        price.to_numpy()[cleared], expected[cleared], rtol=0, atol=1e-9
    )
    # a week has prices set by different kinds of unit, or it checks little
    assert len(np.unique(expected[cleared].round(6))) > 20
