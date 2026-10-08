# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""An interactive page for one run: pick any half-hour and see its order book as a merit order,
the day around it and the year, beside the observed prices.

    python -m assume_gb figures <run folder>       writes viewer.html beside the static figures

The page is one self-contained HTML file (Plotly.js inlined, no network needed). It holds the
licensed observed prices of the scenario folder, so it stays local like them: do not publish it.

Where the exchanges' bid curves of the year have been built (`python -m assume_gb curves`,
`exchange_curves.py`), the page also sets the model's excess supply (offers less bids, at each
price) beside that of N2EX, of EPEX's GB hourly auction and of the two together, hour by hour,
or shows the exchanges' raw supply and demand curves. Only the excess supply is comparable: the
exchanges carry the portfolios' net positions, not the physical merit order (see that module).

The order book of every period is packed per order slot (a unit's first, second, ... order of a
period, by price) into three arrays over the periods, compressed with zlib and unpacked in the
browser: the offer price in tenths of £ (int16), the volume as a share of the slot's largest
volume (int16) and the accepted share of the volume (uint8, 1/250 steps). A year of the day-ahead
market is a few MB that way.
"""

from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from assume_gb import convert
from assume_gb import style as S
from assume_gb.figures import COLOURS, OBSERVED, OBSERVED_NAME, has_observed
from assume_gb.results import EXPORTS, GROUPS, VRE, Run

MISSING = -32768
OBSERVED_SERIES = {
    OBSERVED: f"Observed ({OBSERVED_NAME})",
    "epex_hh_day_ahead": "EPEX half-hourly auction",
    "apx_mid": "Market index (APX)",
    "epex_ida1": "EPEX IDA1",
    "system_price": "System price",
}


def _pack(array: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(array.tobytes(), 9)).decode("ascii")


def _round(series: pd.Series, digits: int = 2) -> list:
    values = np.round(series.to_numpy(dtype=float), digits)
    return [
        None if not np.isfinite(v) else (int(v) if digits == 0 else float(v))
        for v in values
    ]


def _support(row: pd.Series) -> str:
    scheme = row.get("support_scheme")
    if not isinstance(scheme, str) or scheme in ("", "0", "nan"):
        return ""
    value = row.get("support_value", 0)
    if scheme == "cfd":
        ref = row.get("support_reference")
        ref = ref if isinstance(ref, str) and ref not in ("0", "nan") else "own price"
        return f"CfD, strike £{float(value):.2f} against {ref}"
    return f"{scheme} £{float(value):.2f} per MWh"


def _book(run: Run, market_id: str, unit_index: dict[str, int]) -> dict:
    book = run.orders[run.orders.market_id == market_id].copy()
    book = book[book.volume != 0]
    book = book.sort_values(["start_time", "unit_id", "price"])
    book["k"] = book.groupby(["start_time", "unit_id"]).cumcount()
    slots = (
        book[["unit_id", "k"]]
        .drop_duplicates()
        .sort_values(["unit_id", "k"])
        .reset_index(drop=True)
    )
    slot_of = {(u, k): i for i, (u, k) in enumerate(zip(slots.unit_id, slots.k))}
    s = np.array([slot_of[(u, k)] for u, k in zip(book.unit_id, book.k)])
    t = run.index.get_indexer(book.start_time)
    n, m = len(run.index), len(slots)
    price = np.full((m, n), MISSING, dtype="<i2")
    price[s, t] = np.clip(np.round(book.price.to_numpy() * 10), -32767, 32767)
    vscale = np.zeros(m)
    np.maximum.at(vscale, s, book.volume.abs().to_numpy())
    vscale[vscale == 0] = 1.0
    volume = np.zeros((m, n), dtype="<i2")
    volume[s, t] = np.round(book.volume.to_numpy() / vscale[s] * 32000)
    acc = np.zeros((m, n), dtype="u1")
    share = np.where(book.volume != 0, book.accepted_volume / book.volume, 0.0)
    acc[s, t] = np.round(np.clip(share, 0, 1) * 250)
    return {
        "slots": [unit_index[u] for u in slots.unit_id],
        "vscale": (vscale / 32000).tolist(),
        "price": _pack(price),
        "volume": _pack(volume),
        "acc": _pack(acc),
    }


def _curves(run: Run) -> dict | None:
    """The exchanges' curves of the run's year, packed: volumes in 10 MW steps, delta-encoded
    along the price grid (int16), per venue; the shift, reconstructed and published price per hour."""
    from assume_gb import exchange_curves

    year = int(run.scenario.rsplit("_", 1)[-1])
    c = exchange_curves.load(year)
    if c is None:
        return None
    out = {
        "start": pd.Timestamp(int(c["hours"][0])).isoformat(),
        "n": int(len(c["hours"])),
        "grid": c["grid"].tolist(),
        "venues": {},
    }
    for venue in exchange_curves.VENUES:
        packed = {}
        for side in ("supply", "demand"):
            v = np.round(c[f"{venue}_{side}"] / 10.0)
            missing = ~np.isfinite(v).all(axis=1)
            v = np.nan_to_num(v)
            d = np.diff(v, axis=1, prepend=0).astype("<i2")
            d[missing, 0] = MISSING
            packed[side] = _pack(d)
        packed["shift"] = [int(x) for x in np.round(c[f"{venue}_shift"])]
        packed["price"] = _round(pd.Series(c[f"{venue}_price"]))
        packed["published"] = _round(pd.Series(c[f"{venue}_published"]))
        out["venues"][venue] = packed
    return out


def payload(run: Run) -> dict:
    unit_ids = sorted(set(run.orders.unit_id))
    unit_index = {u: i for i, u in enumerate(unit_ids)}
    units = []
    for u in unit_ids:
        row = run.units.loc[u] if u in run.units.index else pd.Series(dtype=object)
        group = (
            "Demand"
            if u == convert.DEMAND_UNIT
            else str(row.get("group", "Waste and other"))
        )
        units.append(
            {
                "id": u,
                "group": group,
                "technology": str(row.get("technology", "")),
                "kind": str(row.get("kind", "")),
                "support": _support(row),
            }
        )

    series = {f"sim_{m}": _round(run.prices[m]) for m in run.markets}
    intraday = convert.INTRADAY_MARKET_ID in run.markets
    if has_observed(run):
        wanted = [OBSERVED, "epex_hh_day_ahead"] + (
            ["apx_mid", "epex_ida1", "system_price"] if intraday else []
        )
        for col in wanted:
            if col in run.observed and run.observed[col].notna().any():
                series[col] = _round(run.observed[col])
    if "storage" in run.case:
        for name in ("model_no_storage", "model_with_battery"):
            if name in run.reference:
                series[name] = _round(run.reference[name])

    gen = run.generation
    vre = gen.reindex(columns=list(VRE), fill_value=0).sum(axis=1)
    extra = {
        "demand": _round(run.gb_demand, 0),
        "residual": _round(run.residual_load, 0),
        "vre": _round(vre, 0),
        "imports": _round(gen.get("Imports", pd.Series(0.0, index=run.index)), 0),
        "exports": _round(-gen[EXPORTS], 0),
    }
    if convert.INTRADAY_MARKET_ID in run.markets:
        extra["imbalance"] = _round(run.imbalance, 0)
        da = run.market_position(convert.MARKET_ID)
        vre_units = [u for u in da.columns if run.units.group.get(u) in VRE]
        extra["vre_day_ahead"] = _round(da[vre_units].sum(axis=1), 0)
        extra["demand_day_ahead"] = _round(
            -da[convert.DEMAND_UNIT] - run.export_capacity, 0
        )

    marginal = {}
    for m in run.markets:
        mg = run.marginal(m)
        marginal[m] = [
            unit_index.get(u, -1) if isinstance(u, str) else -1 for u in mg.unit_id
        ]

    sim = run.day_ahead
    jumps = {
        "Dearest half-hour": int(np.nanargmax(sim.values)),
        "Cheapest half-hour": int(np.nanargmin(sim.values)),
        "Most demand left after wind and solar": int(
            np.nanargmax(run.residual_load.values)
        ),
        "Least demand left after wind and solar": int(
            np.nanargmin(run.residual_load.values)
        ),
    }
    if has_observed(run):
        err = (sim - run.observed[OBSERVED]).where(run.admissible)
        if err.notna().any():
            jumps["Largest miss, simulated below observed"] = int(
                np.nanargmin(err.values)
            )
            jumps["Largest miss, simulated above observed"] = int(
                np.nanargmax(err.values)
            )
    if convert.INTRADAY_MARKET_ID in run.markets:
        spread = run.prices[convert.INTRADAY_MARKET_ID] - sim
        jumps["Largest intraday rise"] = int(np.nanargmax(spread.values))
        jumps["Largest intraday fall"] = int(np.nanargmin(spread.values))

    daily = pd.DataFrame({"sim": sim})
    if has_observed(run):
        daily["obs"] = run.observed[OBSERVED]
    daily = daily.resample("D").mean()

    step = int((run.index[1] - run.index[0]).total_seconds() // 60)
    return {
        "meta": {
            "name": run.name,
            "scenario": run.scenario,
            "case": run.case,
            "start": run.index[0].isoformat(),
            "n": len(run.index),
            "step_minutes": step,
            "markets": run.markets,
            "observed_name": OBSERVED_NAME,
            "groups": [*GROUPS, EXPORTS, "Demand"],
            "colours": COLOURS,
            "series_names": {
                **{f"sim_{m}": f"Simulated {m}" for m in run.markets},
                **OBSERVED_SERIES,
                "model_no_storage": "Merit order without storage",
                "model_with_battery": "Merit-order model with battery",
            },
        },
        "series": series,
        "extra": extra,
        "units": units,
        "marginal": marginal,
        "jumps": jumps,
        "daily": {
            "day": [d.strftime("%Y-%m-%d") for d in daily.index],
            "sim": _round(daily["sim"]),
            "obs": _round(daily["obs"]) if "obs" in daily else None,
        },
        "generation": {
            g: _round(gen[g], 0) for g in gen.columns if gen[g].abs().sum() > 0
        },
        "books": {m: _book(run, m, unit_index) for m in run.markets},
        "curves": _curves(run),
    }


def write(run: Run, out: Path) -> Path:
    from plotly.offline import get_plotlyjs

    data = json.dumps(payload(run), separators=(",", ":")).replace("</", "<\\/")
    title = f"{run.scenario} · {run.case}: market viewer"
    html = (
        TEMPLATE.replace("__TITLE__", title)
        .replace("__DATA__", data)
        .replace("__PLOTLY__", get_plotlyjs())
        .replace("__INK__", S.INK)
        .replace("__BODY__", S.BODY)
        .replace("__MUTED__", S.MUTED)
        .replace("__LIGHT__", S.LIGHT)
        .replace("__FAINT__", S.FAINT)
        .replace("__NAVY__", S.NAVY)
        .replace("__WINE__", S.WINE)
    )
    out.mkdir(parents=True, exist_ok=True)
    path = out / "viewer.html"
    path.write_text(html, encoding="utf-8")
    return path


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --ink:__INK__; --body:__BODY__; --muted:__MUTED__; --light:__LIGHT__; --faint:__FAINT__; --bg:#FFFFFF; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font-family: Arial, Helvetica, sans-serif; }
header { padding:18px 24px 6px; }
h1 { font-size:20px; margin:0 0 4px; }
.sub { color:var(--body); font-size:13px; margin:0; max-width:1100px; }
.controls { display:flex; flex-wrap:wrap; gap:8px; align-items:center; padding:10px 24px; border-bottom:1px solid var(--faint);
  position:sticky; top:0; background:var(--bg); z-index:10; }
.controls button, .controls select, .controls input { font:inherit; font-size:13px; padding:4px 8px; border:1px solid var(--light);
  background:#fff; border-radius:4px; color:var(--ink); }
.controls button { cursor:pointer; } .controls button:hover { background:#F4F3EE; }
.controls label { font-size:13px; color:var(--body); }
.hint { color:var(--muted); font-size:12px; margin-left:auto; }
.tiles { display:grid; grid-template-columns:repeat(auto-fill, minmax(165px, 1fr)); gap:4px 16px; padding:12px 24px 4px; }
.tile { border-top:2px solid var(--faint); padding:6px 0; }
.tile .v { font-size:19px; font-weight:bold; } .tile .l { font-size:12px; color:var(--body); }
.grid { display:grid; grid-template-columns:minmax(0, 1.3fr) minmax(0, 1fr); gap:20px; padding:8px 24px; }
@media (max-width: 960px) { .grid { grid-template-columns:minmax(0, 1fr); } .hint { display:none; } }
.card h2 { font-size:14px; margin:10px 0 2px; } .card p { margin:0 0 4px; font-size:12px; color:var(--body); }
.tablewrap { max-height:330px; overflow:auto; border-top:1px solid var(--faint); margin-top:6px; }
table { border-collapse:collapse; width:100%; font-size:12px; }
th { position:sticky; top:0; background:var(--bg); color:var(--body); font-weight:bold; }
th, td { text-align:right; padding:3px 6px; border-bottom:1px solid var(--faint); font-variant-numeric:tabular-nums; white-space:nowrap; }
th:nth-child(-n+2), td:nth-child(-n+2) { text-align:left; }
tr.marginal td { font-weight:bold; background:#F4F3EE; }
.key { display:inline-block; width:9px; height:9px; margin-right:6px; vertical-align:middle; }
footer { color:var(--muted); font-size:12px; padding:10px 24px 28px; max-width:1100px; }
#loading { padding:24px; color:var(--body); }
</style>
</head>
<body>
<header><h1 id="title"></h1><p class="sub" id="subtitle"></p></header>
<div class="controls">
  <button id="prevDay" title="previous day (key: down)">&#9664;&#9664; day</button>
  <button id="prev" title="previous half-hour (key: left)">&#9664; half-hour</button>
  <input type="date" id="date" aria-label="day">
  <select id="time" aria-label="time of day (UTC)"></select>
  <button id="next" title="next half-hour (key: right)">half-hour &#9654;</button>
  <button id="nextDay" title="next day (key: up)">day &#9654;&#9654;</button>
  <label>Order book <select id="market"></select></label>
  <label>Jump to <select id="jump"><option value="">choose</option></select></label>
  <span class="hint">&larr; &rarr; half-hour &middot; &uarr; &darr; day &middot; click a chart to pick a time &middot; times UTC</span>
</div>
<div id="loading">Unpacking the order book&hellip;</div>
<div class="tiles" id="tiles"></div>
<div class="grid">
  <div class="card">
    <h2>The year</h2><p>Daily mean day-ahead price, £ per MWh; click a day</p><div id="year" style="height:150px"></div>
    <h2 id="dayTitle">The day</h2><p>Price, £ per MWh; below, what each group of plant sold, GW (exports and charging below zero); line = GB demand the market serves</p>
    <div id="dayPrice" style="height:250px"></div><div id="dayStack" style="height:300px"></div>
  </div>
  <div class="card">
    <h2 id="bookTitle">Offers</h2><p id="bookNote"></p><div id="book" style="height:480px"></div>
    <div id="curvesCard" style="display:none">
      <h2 id="curvesTitle">The exchanges' curves</h2>
      <p id="curvesNote"></p>
      <label style="font-size:12px;color:var(--body)">Show <select id="curvesMode"><option value="es">excess supply: model and exchanges</option><option value="raw">the exchanges' supply and demand</option></select></label>
      <div id="curves" style="height:420px"></div>
    </div>
    <div class="tablewrap"><table id="bookTable"></table></div>
  </div>
</div>
<footer id="footer"></footer>
<script type="application/json" id="data">__DATA__</script>
<script>__PLOTLY__</script>
<script>
(async function () {
  const D = JSON.parse(document.getElementById('data').textContent);
  const M = D.meta, n = M.n, perDay = Math.round(1440 / M.step_minutes);
  const t0 = Date.parse(M.start + 'Z'), stepMs = M.step_minutes * 60000;
  const C = M.colours, INK = '__INK__', BODY = '__BODY__', MUTED = '__MUTED__', LIGHT = '__LIGHT__', FAINT = '__FAINT__';
  const NAVY = '__NAVY__', WINE = '__WINE__';
  const font = {family: 'Arial, Helvetica, sans-serif', size: 12, color: BODY};
  const config = {displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d']};
  const stamp = i => new Date(t0 + i * stepMs).toISOString().slice(0, 16).replace('T', ' ');
  const gbp = (v, d = 2) => v == null || !isFinite(v) ? 'n/a' : (v < 0 ? '-£' : '£') + Math.abs(v).toLocaleString('en-GB', {minimumFractionDigits: d, maximumFractionDigits: d});
  const gw = v => v == null || !isFinite(v) ? 'n/a' : (v / 1000).toLocaleString('en-GB', {maximumFractionDigits: 1}) + ' GW';
  const el = id => document.getElementById(id);

  document.title = M.scenario + ' · ' + M.case + ': market viewer';
  el('title').textContent = M.scenario + ' · ' + M.case + ': the market in any half-hour';
  el('subtitle').textContent = 'Simulated with ASSUME: ' + M.markets.join(' and ') + ' auctions, uniform price, one product per half-hour; ' +
    'observed prices from the scenario folder (licensed exchange data: keep this file local).';
  el('footer').textContent = 'The market\'s demand is GB demand less nuclear and pumped-storage output (not units) plus the export ' +
    'capacity of the interconnectors, which export units offer back at the neighbouring price; an export unit that sells is export that ' +
    'does not happen. Offers below zero are plants with a support payment per MWh. Offer prices are stored to £0.1, volumes to 1/32000 of ' +
    'each unit\'s largest offer.';

  async function inflate(b64) {
    const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    const stream = new Blob([bin]).stream().pipeThrough(new DecompressionStream('deflate'));
    return await new Response(stream).arrayBuffer();
  }
  const books = {};
  for (const [m, b] of Object.entries(D.books)) {
    books[m] = {slots: b.slots, vscale: b.vscale, price: new Int16Array(await inflate(b.price)),
                volume: new Int16Array(await inflate(b.volume)), acc: new Uint8Array(await inflate(b.acc))};
  }
  let X = null;
  if (D.curves) {
    const c = D.curves, g = c.grid.length;
    X = {t0: Date.parse(c.start + 'Z'), n: c.n, grid: c.grid, venues: {}};
    for (const [v, p] of Object.entries(c.venues)) {
      const side = {};
      for (const s of ['supply', 'demand']) {
        const d = new Int16Array(await inflate(p[s])), out = new Float32Array(d.length);
        for (let h = 0; h < c.n; h++) {
          if (d[h * g] === -32768) { out[h * g] = NaN; continue; }
          let acc = 0; for (let k = 0; k < g; k++) { acc += d[h * g + k]; out[h * g + k] = acc * 10; }
        }
        side[s] = out;
      }
      X.venues[v] = {...side, shift: p.shift, price: p.price, published: p.published};
    }
    el('curvesCard').style.display = '';
    el('curvesMode').onchange = () => drawCurves();
  }
  el('loading').remove();

  function ordersAt(m, t) {
    const b = books[m], out = [];
    for (let s = 0; s < b.slots.length; s++) {
      const k = s * n + t, p = b.price[k];
      if (p === -32768) continue;
      const v = b.volume[k] * b.vscale[s];
      if (v === 0) continue;
      out.push({unit: b.slots[s], price: p / 10, volume: v, accepted: b.acc[k] / 250 * v});
    }
    return out;
  }

  let cur = D.jumps['Dearest half-hour'] || 0, market = M.markets[0];
  const dayVisible = {};
  const hash = new URLSearchParams(location.hash.slice(1));
  if (hash.has('t')) { const t = Date.parse(hash.get('t').replace(' ', 'T') + 'Z'); if (isFinite(t)) cur = Math.max(0, Math.min(n - 1, Math.round((t - t0) / stepMs))); }
  if (hash.has('m') && M.markets.includes(hash.get('m'))) market = hash.get('m');
  const groups = M.groups;

  // controls
  for (let i = 0; i < perDay; i++) {
    const o = document.createElement('option'); o.value = i; o.textContent = stamp(i).slice(11); el('time').appendChild(o);
  }
  for (const m of M.markets) { const o = document.createElement('option'); o.value = m; o.textContent = m; el('market').appendChild(o); }
  for (const [name, i] of Object.entries(D.jumps)) { const o = document.createElement('option'); o.value = i; o.textContent = name; el('jump').appendChild(o); }
  el('date').min = stamp(0).slice(0, 10); el('date').max = stamp(n - 1).slice(0, 10);
  const go = i => { cur = Math.max(0, Math.min(n - 1, i)); render(); };
  el('prev').onclick = () => go(cur - 1); el('next').onclick = () => go(cur + 1);
  el('prevDay').onclick = () => go(cur - perDay); el('nextDay').onclick = () => go(cur + perDay);
  el('time').onchange = e => go(Math.floor(cur / perDay) * perDay + Number(e.target.value));
  el('date').onchange = e => { const d = Date.parse(e.target.value + 'T00:00Z'); if (isFinite(d)) go(Math.round((d - t0) / stepMs) + cur % perDay); };
  el('market').onchange = e => { market = e.target.value; render(); };
  el('jump').onchange = e => { if (e.target.value !== '') go(Number(e.target.value)); e.target.value = ''; };
  document.addEventListener('keydown', e => {
    if (['INPUT', 'SELECT'].includes(document.activeElement.tagName)) return;
    const k = {ArrowLeft: -1, ArrowRight: 1, ArrowDown: -perDay, ArrowUp: perDay}[e.key];
    if (k) { e.preventDefault(); go(cur + k); }
  });

  // the year
  const yearTraces = [];
  if (D.daily.obs) yearTraces.push({x: D.daily.day, y: D.daily.obs, name: 'Observed (' + M.observed_name + ')', type: 'scatter', mode: 'lines', line: {color: INK, width: 1}});
  yearTraces.push({x: D.daily.day, y: D.daily.sim, name: 'Simulated', type: 'scatter', mode: 'lines', line: {color: NAVY, width: 1.5}});
  const baseLayout = h => ({font, margin: {l: 52, r: 12, t: 8, b: 28}, height: h, paper_bgcolor: '#fff', plot_bgcolor: '#fff',
    hovermode: 'x unified', showlegend: false, xaxis: {gridcolor: FAINT, linecolor: LIGHT, zeroline: false},
    yaxis: {gridcolor: FAINT, linecolor: LIGHT, zerolinecolor: LIGHT}});
  Plotly.newPlot('year', yearTraces, {...baseLayout(150), yaxis: {...baseLayout(150).yaxis, title: {text: '£/MWh'}}}, config);
  el('year').on('plotly_click', ev => { const d = Date.parse(ev.points[0].x.slice(0, 10) + 'T00:00Z'); go(Math.round((d - t0) / stepMs) + cur % perDay); });

  function dayRange() { const s = Math.floor(cur / perDay) * perDay; return [s, Math.min(n, s + perDay)]; }
  function xs(a, b) { const out = []; for (let i = a; i < b; i++) out.push(stamp(i)); return out; }
  function marker(i, top) { return {type: 'line', xref: 'x', yref: 'paper', x0: stamp(i), x1: stamp(i), y0: 0, y1: top, line: {color: INK, width: 1, dash: 'dot'}}; }

  function drawDay() {
    const [a, b] = dayRange(), x = xs(a, b);
    el('dayTitle').textContent = 'The day: ' + stamp(a).slice(0, 10);
    const styles = {sim_DA: [NAVY, 2, 'solid'], sim_ID: [WINE, 2, 'solid'], n2ex_day_ahead: [INK, 1.2, 'solid'],
      epex_hh_day_ahead: [MUTED, 1, 'dot'], apx_mid: [MUTED, 1.2, 'solid'], epex_ida1: [C['Biomass'], 1, 'solid'],
      system_price: [LIGHT, 1, 'solid'], model_no_storage: [MUTED, 1.2, 'dash'], model_with_battery: [MUTED, 1.2, 'dashdot']};
    const hidden = {epex_ida1: true, system_price: true, epex_hh_day_ahead: true};
    const traces = Object.entries(D.series).filter(([k]) => styles[k]).map(([k, v]) => ({x, y: v.slice(a, b), name: M.series_names[k] || k,
      type: 'scatter', mode: 'lines', line: {color: styles[k][0], width: styles[k][1], dash: styles[k][2], shape: 'hv'},
      visible: dayVisible[k] ?? (hidden[k] ? 'legendonly' : true), hovertemplate: '%{y:.2f}'}));
    const bl = baseLayout(250);
    Plotly.react('dayPrice', traces, {...bl, showlegend: true, legend: {orientation: 'h', y: 1.02, yanchor: 'bottom', x: 0, font: {size: 11}},
      margin: {l: 52, r: 12, t: 54, b: 22}, xaxis: {...bl.xaxis, tickformat: '%H:%M'}, yaxis: {...bl.yaxis, title: {text: '£/MWh'}},
      shapes: [marker(cur, 1)], uirevision: 'day'}, config);
    const stack = [];
    for (const g of groups) {
      const v = D.generation[g]; if (!v) continue;
      const pos = v.slice(a, b).map(z => z > 0 ? z / 1000 : 0), neg = v.slice(a, b).map(z => z < 0 ? z / 1000 : 0);
      if (pos.some(z => z > 0)) stack.push({x, y: pos, name: g, stackgroup: 'up', type: 'scatter', mode: 'none', fillcolor: C[g], line: {shape: 'hv'}, hovertemplate: '%{y:.1f} GW'});
      if (neg.some(z => z < 0)) stack.push({x, y: neg, name: g + ' ', stackgroup: 'down', type: 'scatter', mode: 'none', fillcolor: C[g], line: {shape: 'hv'}, hovertemplate: '%{y:.1f} GW', showlegend: false});
    }
    stack.push({x, y: D.extra.demand.slice(a, b).map(z => z / 1000), name: 'Demand', type: 'scatter', mode: 'lines', line: {color: INK, width: 1.2, shape: 'hv'}, hovertemplate: '%{y:.1f} GW'});
    const sl = baseLayout(300);
    Plotly.react('dayStack', stack, {...sl, showlegend: true, legend: {orientation: 'h', y: -0.12, x: 0, font: {size: 11}},
      margin: {l: 52, r: 12, t: 8, b: 60}, xaxis: {...sl.xaxis, tickformat: '%H:%M'}, yaxis: {...sl.yaxis, title: {text: 'GW'}},
      shapes: [marker(cur, 1)]}, config);
  }

  function drawBook() {
    const os = ordersAt(market, cur), price = D.series['sim_' + market][cur];
    const marginal = D.marginal[market][cur];
    const sales = os.filter(o => o.volume > 0 && D.units[o.unit].id !== 'unserved_demand').sort((p, q) => p.price - q.price);
    const bids = os.filter(o => o.volume < 0).sort((p, q) => q.price - p.price);
    const byGroup = {}; let cum = 0, sold = 0;
    for (const o of sales) {
      const u = D.units[o.unit], g = u.group;
      const t = byGroup[g] || (byGroup[g] = {x: [], y: [], w: [], cd: [], line: []});
      t.x.push((cum + o.volume / 2) / 1000); t.w.push(o.volume / 1000); t.y.push(o.price);
      t.cd.push([u.id, u.technology, o.volume, o.accepted, u.support || '']);
      t.line.push(o.unit === marginal ? 2 : 0);
      cum += o.volume; sold += o.accepted;
    }
    const traces = groups.filter(g => byGroup[g]).map(g => ({type: 'bar', name: g, x: byGroup[g].x, y: byGroup[g].y, width: byGroup[g].w,
      marker: {color: C[g] || LIGHT, line: {color: INK, width: byGroup[g].line}}, customdata: byGroup[g].cd,
      hovertemplate: '<b>%{customdata[0]}</b> (' + g + ', %{customdata[1]})<br>offer £%{y:.2f}/MWh<br>%{customdata[2]:,.0f} MW offered, %{customdata[3]:,.0f} MW sold<br>%{customdata[4]}<extra></extra>'}));
    // bids as a falling step line
    const bx = [], by = [], bt = []; let bc = 0;
    for (const o of bids) { const u = D.units[o.unit]; bx.push(bc / 1000); by.push(o.price); bt.push(u.id); bc += -o.volume; bx.push(bc / 1000); by.push(o.price); bt.push(u.id); }
    if (bids.length) { bx.push(bc / 1000); by.push(-500); bt.push(''); }
    if (bids.length) traces.push({type: 'scatter', mode: 'lines', name: 'Bids to buy', x: bx, y: by, text: bt, line: {color: INK, width: 1.5},
      hovertemplate: '<b>%{text}</b><br>bid £%{y:.2f}/MWh<extra></extra>'});
    const obsKey = market === 'DA' ? 'n2ex_day_ahead' : 'apx_mid';
    const obs = D.series[obsKey] ? D.series[obsKey][cur] : null;
    const shapes = [];
    if (obs != null) shapes.push({type: 'line', xref: 'paper', x0: 0, x1: 1, y0: obs, y1: obs, line: {color: INK, width: 1, dash: 'dash'}});
    traces.push({type: 'scatter', mode: 'markers', name: 'Clearing', x: [sold / 1000], y: [price], marker: {color: INK, size: 10, line: {color: '#fff', width: 2}},
      hovertemplate: 'clearing price £%{y:.2f}/MWh<br>%{x:.1f} GW sold<extra></extra>'});
    const visible = sales.filter(o => true);
    let lo = Math.min(0, ...visible.slice(0, 400).map(o => o.price)), hi = Math.max(60, (price || 0) * 1.8, obs || 0);
    lo = Math.max(lo, -350) - 15;
    const xmax = Math.max(sold, bc, 1) * 1.25 / 1000;
    Plotly.react('book', traces, {font, height: 480, margin: {l: 56, r: 12, t: 8, b: 76}, bargap: 0, barmode: 'overlay', paper_bgcolor: '#fff', plot_bgcolor: '#fff',
      showlegend: true, legend: {orientation: 'h', y: -0.16, x: 0, font: {size: 11}}, hovermode: 'closest', shapes,
      xaxis: {title: {text: 'GW'}, range: [0, xmax], gridcolor: FAINT, linecolor: LIGHT, zeroline: false},
      yaxis: {title: {text: '£ per MWh'}, range: [lo, hi * 1.05], gridcolor: FAINT, linecolor: LIGHT, zerolinecolor: BODY}}, config);
    const mu = marginal >= 0 ? D.units[marginal] : null;
    el('bookTitle').textContent = market + ' order book, ' + stamp(cur);
    el('bookNote').textContent = 'Offers to sell sorted by price (width = MW, height = offer price), bids to buy as a line; dot = clearing; dashed = ' +
      (obs != null ? (market === 'DA' ? 'observed ' + M.observed_name : 'observed market index') + ' price' : 'no observed price') +
      '. Price set by ' + (mu ? mu.id + ' (' + mu.group + ')' : 'n/a') + ', outlined.';
    // table view of the same book
    const table = el('bookTable'); table.textContent = '';
    const head = table.insertRow(); for (const h of ['Unit', 'Group', 'Offer, £/MWh', 'Offered, MW', 'Sold, MW']) { const th = document.createElement('th'); th.textContent = h; head.appendChild(th); }
    for (const o of [...sales, ...bids]) {
      const u = D.units[o.unit], r = table.insertRow(); if (o.unit === marginal) r.className = 'marginal';
      const c0 = r.insertCell(); c0.textContent = u.id;
      const c1 = r.insertCell(); const key = document.createElement('span'); key.className = 'key'; key.style.background = C[u.group] || LIGHT; c1.appendChild(key); c1.appendChild(document.createTextNode(u.group));
      r.insertCell().textContent = o.price.toFixed(1);
      r.insertCell().textContent = Math.round(o.volume).toLocaleString('en-GB');
      r.insertCell().textContent = Math.round(o.accepted).toLocaleString('en-GB');
    }
  }

  const VENUE = {n2ex: ['N2EX', INK, 'solid'], epex: ['EPEX GB hourly', MUTED, 'solid']};
  function drawCurves() {
    if (!X) return;
    const g = X.grid.length, h = Math.floor((t0 + cur * stepMs - X.t0) / 3600000);
    const mode = el('curvesMode').value, traces = [], prices = [];
    const hourOk = h >= 0 && h < X.n;
    const curve = (v, s) => Array.from(X.venues[v][s].subarray(h * g, (h + 1) * g));
    const live = hourOk ? Object.keys(X.venues).filter(v => !isNaN(X.venues[v].supply[h * g])) : [];
    if (mode === 'es') {
      // the model: offers to sell at or below p less bids to buy at or above p, in GW
      const os = ordersAt(market, cur);
      const es = X.grid.map(p => os.reduce((a, o) => a + (o.volume > 0 ? (o.price <= p ? o.volume : 0) : (o.price >= p ? o.volume : 0)), 0) / 1000);
      traces.push({x: es, y: X.grid, name: 'Model (' + market + ')', type: 'scatter', mode: 'lines', line: {color: NAVY, width: 2, shape: 'vh'},
        hovertemplate: 'model: %{x:.1f} GW at £%{y:.0f}<extra></extra>'});
      let sum = null;
      for (const v of live) {
        const sup = curve(v, 'supply'), dem = curve(v, 'demand'), sh = X.venues[v].shift[h];
        const e = sup.map((s, k) => (s - dem[k] + sh) / 1000);
        sum = sum ? sum.map((a, k) => a + e[k]) : e;
        traces.push({x: e, y: X.grid, name: VENUE[v][0], type: 'scatter', mode: 'lines', line: {color: VENUE[v][1], width: 1.5},
          hovertemplate: VENUE[v][0] + ': %{x:.1f} GW at £%{y:.0f}<extra></extra>'});
        if (X.venues[v].price[h] != null) prices.push([VENUE[v][0], X.venues[v].price[h], X.venues[v].published[h]]);
      }
      if (live.length > 1) traces.push({x: sum, y: X.grid, name: 'N2EX and EPEX together', type: 'scatter', mode: 'lines', line: {color: WINE, width: 1.5, dash: 'dash'},
        hovertemplate: 'together: %{x:.1f} GW at £%{y:.0f}<extra></extra>'});
    } else {
      for (const v of live) {
        for (const [s, dash] of [['supply', 'solid'], ['demand', 'dot']]) {
          const vol = curve(v, s).map((q, k) => (q + (s === 'supply' ? X.venues[v].shift[h] : 0)) / 1000);
          traces.push({x: vol, y: X.grid, name: VENUE[v][0] + ' ' + s, type: 'scatter', mode: 'lines', line: {color: VENUE[v][1], width: 1.5, dash},
            hovertemplate: VENUE[v][0] + ' ' + s + ': %{x:.1f} GW at £%{y:.0f}<extra></extra>'});
        }
        if (X.venues[v].price[h] != null) prices.push([VENUE[v][0], X.venues[v].price[h], X.venues[v].published[h]]);
      }
    }
    const sim = D.series['sim_' + market][cur];
    const all = [sim, ...prices.map(p => p[1])].filter(v => v != null && isFinite(v));
    const hi = Math.max(150, ...all.map(v => v * 1.6)), lo = Math.min(-50, ...all.map(v => v - 60));
    const shapes = mode === 'es' ? [{type: 'line', xref: 'x', yref: 'paper', x0: 0, x1: 0, y0: 0, y1: 1, line: {color: BODY, width: 1}}] : [];
    Plotly.react('curves', traces, {font, height: 420, margin: {l: 56, r: 12, t: 8, b: 76}, paper_bgcolor: '#fff', plot_bgcolor: '#fff',
      showlegend: true, legend: {orientation: 'h', y: -0.16, x: 0, font: {size: 11}}, hovermode: 'closest', shapes,
      xaxis: {title: {text: mode === 'es' ? 'excess supply, GW (offers less bids at each price)' : 'GW'}, gridcolor: FAINT, linecolor: LIGHT, zeroline: false},
      yaxis: {title: {text: '£ per MWh'}, range: [lo, hi], gridcolor: FAINT, linecolor: LIGHT, zerolinecolor: LIGHT}, uirevision: mode}, config);
    const hourStart = hourOk ? new Date(X.t0 + h * 3600000).toISOString().slice(0, 16).replace('T', ' ') : 'n/a';
    el('curvesTitle').textContent = "The exchanges' curves, hour from " + hourStart;
    el('curvesNote').textContent = (live.length ? '' : 'No exchange curves for this hour. ') +
      (mode === 'es' ? 'Each curve crosses zero at its price. The exchanges carry net positions (about a third of GB demand clears on N2EX), so only this ' +
        'difference of offers and bids is comparable with the model: volume a company supplies itself or holds under contract leaves both sides. Its slope at zero is the price response to a shift of demand. '
       : 'Supply includes the accepted block orders and the coupled interconnector flow (N2EX). These are the portfolios\' net positions, not the physical merit order. ') +
      prices.map(p => p[0] + ': £' + p[1].toFixed(2) + ' from the curves, £' + (p[2] == null ? 'n/a' : p[2].toFixed(2)) + ' published').join('; ') + '.';
  }

  function tiles() {
    const t = cur, x = D.extra, items = [];
    for (const m of M.markets) items.push([gbp(D.series['sim_' + m][t]), 'Simulated ' + m + ' price']);
    const obs = D.series['n2ex_day_ahead'] ? D.series['n2ex_day_ahead'][t] : null;
    if (D.series['n2ex_day_ahead']) items.push([gbp(obs), 'Observed (' + M.observed_name + ')']);
    if (obs != null && D.series.sim_DA[t] != null) items.push([gbp(D.series.sim_DA[t] - obs), 'Simulated less observed']);
    const mu = D.marginal.DA[t] >= 0 ? D.units[D.marginal.DA[t]] : null;
    items.push([mu ? mu.group : 'n/a', 'Sets the DA price' + (mu ? ': ' + mu.id : '')]);
    items.push([gw(x.demand[t]), 'GB demand the market serves']);
    items.push([gw(x.vre[t]), 'Wind and solar']);
    items.push([gw(x.residual[t]), 'Demand left after wind and solar']);
    items.push([gw(x.imports[t]) + ' / ' + gw(x.exports[t]), 'Imports / exports']);
    if (x.imbalance) {
      items.push([gw(x.vre_day_ahead[t]) + ' → ' + gw(x.vre[t]), 'Wind and solar: day ahead → outturn']);
      items.push([gw(x.demand_day_ahead[t]) + ' → ' + gw(x.demand[t]), 'Demand: day ahead → outturn']);
      items.push([gw(x.imbalance[t]), 'System long (+) or short (-) at intraday']);
    }
    const box = el('tiles'); box.textContent = '';
    for (const [v, l] of items) { const d = document.createElement('div'); d.className = 'tile';
      const a = document.createElement('div'); a.className = 'v'; a.textContent = v;
      const b = document.createElement('div'); b.className = 'l'; b.textContent = l; d.appendChild(a); d.appendChild(b); box.appendChild(d); }
  }

  function render() {
    el('date').value = stamp(cur).slice(0, 10); el('time').value = String(cur % perDay); el('market').value = market;
    history.replaceState(null, '', '#t=' + stamp(cur).replace(' ', 'T') + '&m=' + market);
    tiles(); drawDay(); drawBook(); drawCurves();
    Plotly.relayout('year', {shapes: [{type: 'line', xref: 'x', yref: 'paper', x0: stamp(cur).slice(0, 10), x1: stamp(cur).slice(0, 10), y0: 0, y1: 1, line: {color: INK, width: 1, dash: 'dot'}}]});
  }
  render();
  // a chart has its event methods only once it is drawn
  el('dayPrice').on('plotly_legendclick', () => {
    setTimeout(() => { for (const tr of el('dayPrice').data) { const key = Object.keys(M.series_names).find(s => M.series_names[s] === tr.name); if (key) dayVisible[key] = tr.visible; } }, 0); });
  for (const id of ['dayPrice', 'dayStack']) {
    el(id).on('plotly_click', ev => { const d = Date.parse(ev.points[0].x.replace(' ', 'T') + 'Z'); if (isFinite(d)) go(Math.round((d - t0) / stepMs)); });
  }
})().catch(err => { const box = document.createElement('p'); box.id = 'error'; box.style.cssText = 'padding:0 24px;color:#882255';
  box.textContent = 'The viewer stopped: ' + err + ' (it needs a current browser: Chrome, Edge, Firefox or Safari).';
  document.body.insertBefore(box, document.body.children[2]); });
</script>
</body>
</html>
"""
