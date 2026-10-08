# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""The forecasts that define predicted penetration (``penetration_forecasts.csv``), written from a
stand-in for the GB data repository: on the scenario folder's own index, with the transmission
system demand forecast where the forecast state has one and the national demand forecast, under
its own name, where it does not.

    python -m pytest assume_gb
"""

import pandas as pd
import pytest

from assume_gb import __main__ as cli
from assume_gb import forecasts, paths


def _repo(root, with_tsd: bool):
    processed = root / "repo" / "data" / "processed"
    processed.mkdir(parents=True)
    times = pd.date_range("2023-01-01", periods=96, freq="30min", tz="UTC")
    state = pd.DataFrame(
        {
            "startTime": times.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wind_tx_fc_mw": 5000.0,
            "emb_wind_fc_mw": 2000.0,
            "emb_solar_fc_mw": 0.0,
            "dgws_solar_fc_mw": 0.0,
            "demand_fc_mw": 25000.0,
        }
    )
    if with_tsd:
        state[forecasts.TSD_FORECAST] = 27500.0
    state.to_csv(processed / "forecast_state_2023.csv", index=False)
    return root / "repo"


def _folder(root):
    """A scenario folder's demand file: the lead-in day, the delivery day, the closing row."""
    folder = root / "inputs" / "gb_2023"
    folder.mkdir(parents=True)
    index = pd.date_range("2022-12-31", "2023-01-02", freq="30min", name="datetime")
    pd.DataFrame({"demand": 30000.0}, index=index).to_csv(folder / "demand_df.csv")
    return folder


@pytest.mark.parametrize("with_tsd", [True, False])
def test_penetration_forecasts_are_written_on_the_folders_index(
    tmp_path, monkeypatch, with_tsd
):
    monkeypatch.setenv(paths.GB_REPO_ENV, str(_repo(tmp_path, with_tsd)))
    folder = _folder(tmp_path)
    table = pd.read_csv(
        forecasts.write_penetration(2023, folder), index_col=0, parse_dates=True
    )
    assert table.index.equals(
        pd.read_csv(folder / "demand_df.csv", index_col=0, parse_dates=True).index
    )
    load = "tsd_forecast_mw" if with_tsd else "nd_forecast_mw"
    assert table.columns.tolist() == ["wind_tx_forecast_mw", load]
    day = table.loc["2023-01-01"]
    assert (day["wind_tx_forecast_mw"] == 5000.0).all()
    assert (day[load] == (27500.0 if with_tsd else 25000.0)).all()
    assert (
        table.loc["2022-12-31"].isna().all().all()
    )  # the lead-in day delivers nothing


def test_the_command_writes_only_that_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(paths.GB_REPO_ENV, str(_repo(tmp_path, with_tsd=False)))
    folder = _folder(tmp_path)
    monkeypatch.setattr(paths, "scenarios_dir", lambda: tmp_path / "inputs")
    assert cli.main(["penetration", "--year", "2023"]) == 0
    assert sorted(p.name for p in folder.iterdir()) == [
        "demand_df.csv",
        forecasts.PENETRATION_FILE,
    ]
    assert "national demand" in capsys.readouterr().out
