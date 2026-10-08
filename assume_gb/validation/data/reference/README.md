<!--
SPDX-FileCopyrightText: ASSUME Developers

SPDX-License-Identifier: MIT
-->

# Reference data of the validation package

## `cacciarelli_cate_digitised.csv`

The causal effect of day-ahead wind and solar forecasts on GB day-ahead prices from Cacciarelli
et al. (2025), *Do we actually understand the impact of renewables on electricity prices? A causal
inference approach*, iEnergy (arXiv 2501.10423), citation key `cacciarelliWeActuallyUnderstand2025`.
Read by `assume_gb.validation.reference`.

| Column | Content |
| --- | --- |
| `market` | `APX` (the APX/EPEX half-hourly day-ahead auction; Fig. 4 of the published paper, Fig. 3 of arXiv v1) or `NordPool` (N2EX day-ahead; Fig. 5, Fig. S1 of arXiv v1) |
| `technology` | `wind` or `solar` |
| `predicted_penetration_pct` | day-ahead forecast / estimated load x 100 |
| `cate_gbp_per_mwh_per_gw` | the smoothed conditional average treatment effect: GBP/MWh per +1 GW of day-ahead forecast output (their code divides the MW forecast by 1000; rows are half-hours) |
| `ci80_low`, `ci80_high` | the 80% band: 10th and 90th percentile of 100 bootstrap estimates per window, smoothed |

**Provenance.** The authors' results files (`results_wind.csv`, `results_solar.csv`) are named in
their repository's README but were never committed (github.com/dcacciarelli/market-impact-renewables,
all 23 commits up to 262ae18 of 1 Dec 2025). The table was therefore traced from the figures of
the published PDF (7 Oct 2026): about +/-0.05 GBP/MWh on the curve and +/-0.1 on the band edges. Solar
`ci80_low` above 7.1% is blank because the figure's legend hides it. It is approximate.

**Check.** An independent pixel tracing of the arXiv v1 figure images (3600 x 3600 px, axes
calibrated on the tick marks; the APX images are byte-identical to the repository's README figures)
agrees with the table within 0.023 GBP/MWh on the curves (mean 0.003 to 0.009) and 0.094 on the
band edges (mean 0.010 to 0.018), at every point below 7.9% for solar.

**Use.** Choose the market of the observed price series (N2EX: `NordPool`; EPEX: `APX`),
interpolate linearly between points, and do not extrapolate beyond the penetration range of each
column. The estimates come from local partially linear double machine learning; a linear
regression run on observed and simulated prices is only a consistency check against them.

**Definitions differ from the scenario data.** In the authors' dataset (`data/
market_impact_renewables.csv.zip` of their repository), `wind_forecast` is transmission-connected
wind (2023 mean 7.0 GW) and `estimated_load` equals transmission system demand (ITSDO; 2023 mean
28.0 GW). The GB scenarios carry wind including embedded wind and demand net of nuclear and pumped
storage, on which the 2023 mean wind penetration is 36.8% against their 25.5%; the validation
therefore bins on transmission wind over the transmission system demand forecast
(`penetration_forecasts.csv` of each scenario folder, see the package README). Their solar
forecast barely varies within a day (mean by hour 0.85 GW at 04:00 to 1.37 GW at 14:00, never zero
at night), so their solar penetration is not the half-hourly solar share: the solar curves are
indicative only.
