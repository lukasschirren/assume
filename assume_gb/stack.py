# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Reader for the merit-order model's exported offer stack.

``python -m meritorder.stack_export`` in the GB data repository writes, per year, the stack that
model clears: one column per offer (an agent, a must-run block, an interconnector step) with its
price and MW in every half-hour, and the demand they are cleared against. Those three files are
the inputs of a GB scenario here, read in place:

    comb_stack_{year}_{design}_{tag}.npz           arrays
    comb_stack_agents_{year}_{design}_{tag}.csv    one row per stack column
    comb_stack_meta_{year}_{design}_{tag}.json     constants, battery fleet, flags

``OfferStack`` can also be built by hand, which is how the tests make small synthetic ones.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from assume_gb import paths

SERIES = (
    "demand",
    "demand_gross",
    "nuclear_exog",
    "ps_exog",
    "ic_export_mw",
    "bmrp",
    "gas_gbp_mwh_th",
    "coal_gbp_mwh_th",
    "carbon_gbp_t",
    "price_no_storage",
    "price",
    "storage_charge_mw",
    "storage_discharge_mw",
    "da_price",
)
FLAGS = ("demand_bad", "scarce")


@dataclass
class OfferStack:
    """An offer stack: ``bids`` and ``mw`` are (periods, columns); ``agents`` has one row per
    column, in column order; ``series`` holds the per-period arrays named in ``SERIES``."""

    index: pd.DatetimeIndex
    agents: pd.DataFrame
    bids: np.ndarray
    mw: np.ndarray
    series: dict[str, np.ndarray]
    meta: dict
    flags: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        n_periods, n_columns = self.bids.shape
        if self.mw.shape != self.bids.shape:
            raise ValueError(
                f"bids {self.bids.shape} and mw {self.mw.shape} differ in shape"
            )
        if len(self.index) != n_periods:
            raise ValueError(f"{len(self.index)} timestamps for {n_periods} periods")
        if len(self.agents) != n_columns:
            raise ValueError(
                f"{len(self.agents)} agent rows for {n_columns} stack columns"
            )
        if self.index.tz is not None:
            raise ValueError("index must be timezone-naive (UTC wall time)")
        if not self.agents["unit_id"].is_unique:
            raise ValueError("unit_id must be unique")

    @property
    def freq(self) -> pd.Timedelta:
        return pd.Timedelta(self.index[1] - self.index[0])

    @property
    def demand(self) -> np.ndarray:
        return self.series["demand"]


def stack_files(
    year: int,
    design: str = "status_quo",
    tag: str = "srmc",
    results_dir: Path | None = None,
) -> dict[str, Path]:
    root = Path(results_dir) if results_dir is not None else paths.gb_results_dir()
    stem = f"{year}_{design}_{tag}"
    return {
        "arrays": root / f"comb_stack_{stem}.npz",
        "agents": root / f"comb_stack_agents_{stem}.csv",
        "meta": root / f"comb_stack_meta_{stem}.json",
    }


def load_stack(
    year: int,
    design: str = "status_quo",
    tag: str = "srmc",
    results_dir: Path | None = None,
) -> OfferStack:
    """Read one exported stack. Timestamps become timezone-naive UTC, which is what ASSUME's
    index is."""
    files = stack_files(year, design, tag, results_dir)
    missing = [p.name for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"stack export not found: {missing} in {files['arrays'].parent}. Write it in the GB "
            f"data repository with: python -m meritorder.stack_export --years {year} {year}"
        )
    with np.load(files["arrays"]) as z:
        index = pd.DatetimeIndex(pd.to_datetime(z["start_time"], utc=True)).tz_localize(
            None
        )
        bids, mw = z["bids"], z["mw"]
        series = {k: np.asarray(z[k], dtype=float) for k in SERIES if k in z.files}
        flags = {k: np.asarray(z[k], dtype=bool) for k in FLAGS if k in z.files}
    agents = pd.read_csv(files["agents"])
    meta = json.loads(files["meta"].read_text(encoding="utf-8"))
    return OfferStack(
        index=index,
        agents=agents,
        bids=bids,
        mw=mw,
        series=series,
        meta=meta,
        flags=flags,
    )


def merit_order_price(
    bids: np.ndarray,
    mw: np.ndarray,
    demand: np.ndarray,
    floor: float,
    cap: float = 3000.0,
) -> np.ndarray:
    """The merit order on a stack: per period, the price of the first offer, in price order,
    whose cumulative MW reaches the demand, clipped to the floor and the cap; the cap where the
    whole stack falls short. The arithmetic of the merit-order model's own clearing."""
    order = np.argsort(bids, axis=1, kind="stable")
    sorted_bids = np.take_along_axis(bids, order, axis=1)
    cumulative = np.cumsum(np.take_along_axis(mw, order, axis=1), axis=1)
    reached = (cumulative < demand[:, None]).sum(axis=1)
    last = sorted_bids.shape[1] - 1
    price = np.take_along_axis(
        sorted_bids, np.clip(reached, 0, last)[:, None], axis=1
    ).ravel()
    return np.where(reached > last, cap, np.clip(price, floor, cap))
