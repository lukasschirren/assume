# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Day-ahead forecast errors from the GB data repository, for the intraday stage.

The merit-order model clears the outturn. The real day-ahead auction clears what was known the day
before: ``processed/forecast_state_{year}.csv`` in the GB data repository holds the information set
at 09:00 UTC on D-1 (Elexon WINDFOR for transmission-metered wind, the NESO embedded wind and solar
forecasts with Elexon DGWS as the proxy where they are missing, and the NDF national demand
forecast or, before mid-2021, NESO's cardinal-point forecast). The outturns on the same basis are
Elexon FUELHH wind and the NESO embedded estimates, both read from the ESPENI snapshot, and the
Elexon INDO national demand outturn (``raw/indo_*.csv``).

Per half-hour this module forms

    wind ratio     (transmission wind forecast + embedded wind forecast) / (their outturns)
    solar ratio    solar forecast / embedded solar outturn
    demand error   national demand forecast - national demand outturn, MW

and ``apply`` scales every wind column of a stack by the wind ratio and every solar column by the
solar ratio (the share of capacity is then clipped to [0, 1]) and adds the demand error to the
stack's demand. Where a forecast or an outturn is missing, or the outturn is below
``MIN_OUTTURN_MW``, the ratio is 1 and the error 0: no forecast error in that half-hour. The errors
are national; the forecast of a single unit carries the national error of its technology, so the
units keep their relative output.

Two things the wind ratio carries that are not forecast errors: curtailment, as the outturn is
metered after the balancing mechanism has turned wind down while the forecast is of what the wind
would give (the mean ratio is above 1 in every year), and the embedded forecasts' coverage, which
differs from the NESO estimates' before mid-2019 when DGWS stands in for them.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from assume_gb import paths
from assume_gb.stack import OfferStack

WIND, SOLAR = "Wind", "Solar"
MIN_OUTTURN_MW = 200.0
ESPENI_COLUMNS = {
    "ELEC_elex_startTime[utc](datetime)": "time",
    "ELEC_POWER_ELEX_WIND[MW](float32)": "wind_tx_out",
    "ELEC_POWER_NGEM_EMBEDDED_WIND_GENERATION[MW](float32)": "wind_emb_out",
    "ELEC_POWER_NGEM_EMBEDDED_SOLAR_GENERATION[MW](float32)": "solar_out",
}


def _utc_naive(series: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(series, utc=True)).tz_localize(None)


def forecast_state(year: int) -> pd.DataFrame:
    """The D-1 forecasts of the year on a naive UTC index."""
    table = pd.read_csv(
        paths.gb_repo() / "data" / "processed" / f"forecast_state_{year}.csv"
    )
    table.index = _utc_naive(table.pop("startTime"))
    return table


def outturns(year: int) -> pd.DataFrame:
    """Transmission wind, embedded wind and solar (ESPENI) and national demand (INDO) outturns."""
    raw = paths.gb_repo() / "data" / "raw"
    espeni = pd.read_csv(raw / "espeni.csv", usecols=list(ESPENI_COLUMNS)).rename(
        columns=ESPENI_COLUMNS
    )
    espeni.index = _utc_naive(espeni.pop("time"))
    espeni = espeni[espeni.index.year == year]
    indo_files = sorted(raw.glob("indo_*.csv"))
    if not indo_files:
        raise FileNotFoundError(
            f"no INDO file in {raw}: python -m gbabm.pull.elexon_pull indo"
        )
    indo = pd.read_csv(indo_files[-1], usecols=["startTime", "demand"])
    indo.index = _utc_naive(indo.pop("startTime"))
    indo = indo[~indo.index.duplicated()].sort_index()
    espeni["demand_out"] = indo["demand"].reindex(espeni.index)
    return espeni


def errors(year: int, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Wind ratio, solar ratio and demand error (MW) on ``index``: 1, 1 and 0 where unknown."""
    fc = forecast_state(year).reindex(index)
    out = outturns(year).reindex(index)
    wind_fc = fc["wind_tx_fc_mw"] + fc["emb_wind_fc_mw"]
    wind_out = out["wind_tx_out"] + out["wind_emb_out"]
    # without an embedded wind forecast the transmission wind forecast stands for all wind
    no_embedded = fc["emb_wind_fc_mw"].isna() & fc["wind_tx_fc_mw"].notna()
    wind_fc[no_embedded] = fc.loc[no_embedded, "wind_tx_fc_mw"]
    wind_out[no_embedded] = out.loc[no_embedded, "wind_tx_out"]
    solar_fc = fc["emb_solar_fc_mw"].fillna(fc["dgws_solar_fc_mw"])
    solar_out = out["solar_out"]
    demand_error = fc["demand_fc_mw"] - out["demand_out"]

    def ratio(forecast, outturn):
        # a ratio of two small numbers (dawn and dusk for solar) says nothing about the forecast
        valid = forecast.notna() & outturn.notna() & (outturn > MIN_OUTTURN_MW)
        return pd.Series(
            np.where(valid, forecast / outturn.where(valid, 1.0), 1.0), index=index
        )

    return pd.DataFrame(
        {
            "wind_ratio": ratio(wind_fc, wind_out),
            "solar_ratio": ratio(solar_fc, solar_out),
            "demand_error_mw": demand_error.fillna(0.0),
        }
    )


# Predicted penetration as Cacciarelli et al. define it (the causal reference of
# assume_gb.validation): transmission-connected wind over transmission system demand, both as
# forecast at 09:00 D-1. The forecast state holds the wind forecast (WINDFOR); the transmission
# system demand forecast is read from TSD_FORECAST where the file has that column, and the national
# demand forecast stands in for it otherwise, under its own name.
TSD_FORECAST = "tsd_fc_mw"
PENETRATION_FILE = "penetration_forecasts.csv"


def penetration_forecasts(year: int, index: pd.DatetimeIndex) -> pd.DataFrame:
    """The D-1 forecasts that define predicted penetration as the validation's causal reference
    does, MW on ``index``: ``wind_tx_forecast_mw`` (transmission-connected wind, WINDFOR) and
    ``tsd_forecast_mw`` (transmission system demand) where the forecast state has it, otherwise
    ``nd_forecast_mw`` (national demand, which leaves out station load, pumping and exports)."""
    fc = forecast_state(year).reindex(index)
    table = pd.DataFrame({"wind_tx_forecast_mw": fc["wind_tx_fc_mw"]}, index=index)
    if TSD_FORECAST in fc:
        table["tsd_forecast_mw"] = fc[TSD_FORECAST]
    else:
        table["nd_forecast_mw"] = fc["demand_fc_mw"]
    return table


def write_penetration(year: int, folder: Path) -> Path:
    """Writes ``penetration_forecasts.csv`` into the scenario folder ``folder`` on the folder's own
    index (that of ``demand_df.csv``), and nothing else, so that the file alone can be copied to a
    machine without the GB data repository. The framework does not read it; the validation
    package does."""
    index = pd.read_csv(
        folder / "demand_df.csv", usecols=[0], index_col=0, parse_dates=True
    ).index
    table = penetration_forecasts(year, pd.DatetimeIndex(index, name="datetime"))
    path = folder / PENETRATION_FILE
    table.to_csv(path, float_format="%.6g")
    return path


def apply(stack: OfferStack, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The forecast MW of every stack column and the forecast demand, (periods, columns) and (periods,)."""
    table = table.reindex(stack.index)
    tech = stack.agents["tech"].to_numpy()
    factor = np.ones(stack.mw.shape)
    factor[:, tech == WIND] = table["wind_ratio"].to_numpy()[:, None]
    factor[:, tech == SOLAR] = table["solar_ratio"].to_numpy()[:, None]
    demand = np.clip(stack.demand + table["demand_error_mw"].to_numpy(), 0.0, None)
    return stack.mw * factor, demand


def summary(table: pd.DataFrame) -> dict:
    """How large the forecast errors are, for the scenario's record."""
    return {
        "wind_ratio_mean": float(table["wind_ratio"].mean()),
        "wind_ratio_std": float(table["wind_ratio"].std()),
        "solar_ratio_mean": float(table["solar_ratio"].mean()),
        "demand_error_mean_mw": float(table["demand_error_mw"].mean()),
        "demand_error_mae_mw": float(table["demand_error_mw"].abs().mean()),
        "periods_with_wind_forecast": int((table["wind_ratio"] != 1.0).sum()),
        "periods_with_demand_forecast": int((table["demand_error_mw"] != 0.0).sum()),
    }
