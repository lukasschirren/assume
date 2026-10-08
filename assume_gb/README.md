<!--
SPDX-FileCopyrightText: ASSUME Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# assume_gb: the GB scenarios and the tool that builds them (local, not pushed)

This folder is listed in `.git/info/exclude`: nothing in it is pushed. It holds the data of the
Great Britain scenarios under `inputs/`, one folder per year, and the tool that builds them from
the merit-order model of the GB data repository. The framework itself is self-contained: everything
GB-specific that is code lives in `assume/` (strategies, unit parameters, forecaster fields) with
tests under `tests/` and docs under `docs/source/`; nothing there refers to this folder.

Why the data is here and not in `examples/inputs`: that folder holds the example scenarios the
framework publishes (tracked in git, a licence file beside every data file, copied into the Docker
image). Data that stays local goes in a folder of its own, which the framework takes as
`inputs_path` (`-i`). To publish a scenario later, move its folder to `examples/inputs` and add the
licence files; `observed_prices.csv` is licensed exchange data and stays out either way.

```bash
python -m assume_gb build --year 2023          # offer stack + forecasts + observed prices -> inputs/gb_2023
python -m assume_gb check --year 2023          # the scenario's offers against the model's own
python -m assume_gb run   --year 2023          # simulate a case and compare its prices (--case, --years FROM TO, --save)
python -m pytest assume_gb                     # the tool's tests and the first week of every built year
python -m assume_gb validate --year 2023 --arm learning_novdec_own_e100   # the validation against observed prices (validation/README.md)
assume -s gb_2023 -c day_ahead -i assume_gb/inputs -csv examples/outputs   # the framework's own CLI
```

The offer stack is written first, once per year, in the GB data repository:
`python -m meritorder.stack_export --years 2018 2025 --ro-fleet register --tag register` (with that
repository's interpreter; `--ro-fleet register` applies the model's three corrections against
Ofgem's RO register, and the folders here are built from that tag: `build --tag register`);
`GB_DATA_REPO` points at the repository if it is not the sibling folder. The other inputs the tool
reads there: `processed/forecast_state_{year}.csv` (the day-ahead forecasts), `raw/espeni.csv` and
`raw/indo_*.csv` (the outturns they are compared with), `processed/mid_target_{year}.csv` and
`processed/system_prices_{year}.csv` and the EPEX intraday auction files (observed prices).

## Study cases

Every year folder has the same cases on the same files. All auctions are uniform-price, once a day
for the 48 half-hours of the next day, in GBP/MWh between -500 and 3000, times in UTC.

| Case | Markets | What is in it |
| --- | --- | --- |
| `day_ahead` | `DA` at 09:00 D-1, `pay_as_clear` | the merit-order model's stack: every unit bids what it will deliver, at its marginal cost less what its support contract pays |
| `day_ahead_storage` | `DA` | plus the model's battery fleet as one storage unit, bidding a plan optimised on the no-storage merit-order price (`storage_energy_optimization_schedule`) |
| `day_ahead_intraday` | `DA`, then `ID` at 17:30 D-1, `complex_clearing` | wind, solar and demand bid their day-ahead forecasts in `DA`; in `ID` every unit trades the difference between its position and the outturn in both directions, at marginal cost |
| `learning` | `DA`, one auction per half-hour (see below) | gas plants (CCGT by default, `build --learning-techs`; the N largest with `--learning-units N`) learn their bids by reinforcement learning (`powerplant_energy_learning`, or `..._learning_support` for a plant with a contract) over the first weeks (`--learning-weeks`, default 4); 50 training episodes with ten gradient steps per simulated day (a step costs about 45 ms per learning unit), the policies saved in `learned_strategies/` of the folder. `gb_2023` is built with the ten largest plants over two weeks: about 1.5 h |
| `learning_year` | as `learning` | the whole year with the best policies of `learning`, no training |
| `*_week` | as above | the first week of delivery only: seconds to run, what the tests use (`learning_week`: four episodes, to try the set-up) |

The learning cases hold the same market one half-hour at a time: one auction per half-hour for
that half-hour, opening 15 hours ahead as the day-ahead auction does for the first half-hour of a
day, because the framework's learning strategies bid the first product of an auction only. With the
day-ahead units in it the prices are those of the day-ahead auction (no unit has an inter-temporal
constraint; tested), at 48 times the number of auctions: about ten minutes for a year. The agents
observe the merit-order price forecast (`forecasts_df.csv`), the residual load the framework derives
from the demand and the wind and solar units, the last twelve prices and their own cost and output,
and act on two bid prices (for the minimum power and the rest) up to `max_bid_price` (the highest
merit-order price of the year rounded up to the next hundred, `--max-bid-price`). A learning run
needs a database (`run --db`, by default a sqlite file in `results/`), which `python -m assume_gb
learning --db ... --simulation gb_2023_learning` then draws.

A year takes one to three minutes without output files (the intraday case seven to twelve, its
auction being an optimisation); the one-week cases a few seconds. `--save` saves the clearing
prices of a run into `results/prices/<run>.csv` (`--out DIR` writes them as `DIR/<run>_prices.csv`
instead); `--csv DIR` writes the framework's full output, about 600 MB a year. Everything a run
or an analysis keeps goes into `results/` in a fixed layout (`results/README.md`; the paths come
from `paths.results_dir`); `python -m assume_gb tidy --apply` moves files written in the old flat
layout into it. `dev/regress.py` is the before/after check for changes to the framework's
bookkeeping (see its docstring).

What the intraday case is and is not: the `ID` auction knows the outturn, so after it every unit
holds what it delivers and the `ID` price is the merit order of the outturn, the same price as the
`day_ahead` case. The `DA` price of this case is the merit order of the forecasts, so `ID - DA` is
what the forecast error costs at marginal cost. The error carries curtailment too: the wind outturn
is metered after the balancing mechanism turned wind down, the forecast is of what the wind would
give, so the forecast exceeds the outturn by 3 to 18 percent on average depending on the year (see
`forecast_errors` in `scenario_meta.json`), and a forecast above a unit's `max_power` (its highest
outturn of the year) is capped there, 0.9 percent of forecast wind in 2023. Before mid-2021 the
demand forecast is NESO's cardinal-point forecast, which runs 300 to 570 MW above the INDO outturn
on average; from then on it is the NDF, with a bias below 50 MW.

## Files of a year folder

| File | Content |
| --- | --- |
| `config.yaml` | the study cases and their markets |
| `powerplant_units.csv` | 173 to 198 units, see "Units"; `*_intraday.csv` the same with the forecast and rebalance strategies; `*_learning.csv` the same with the learning units under their owners; `unit_operators_learning.csv` the owners that learn |
| `demand_units.csv`, `demand_df.csv` | one inelastic demand unit bidding the price cap, and its demand in MW |
| `storage_units.csv` | the battery fleet |
| `forecasts_df.csv` | `price_DA`, the price forecast the battery bids on and the learning agents observe: the no-storage merit order, in every period (the storage strategy does not bid an auction whose forecast has a gap) |
| `availability_df.csv.gz` | share of `max_power` a unit can deliver, per half-hour, where it varies |
| `availability_forecast_df.csv.gz`, `demand_forecast_df.csv` | the same as forecast the day before, for wind, solar and demand |
| `fuel_prices_df.csv` | `natural gas`, `hard coal`, `oil` (GBP/MWh thermal), `co2` (GBP/t), `bmrp` (the baseload market reference price of the CfD scheme) and one price column per interconnector unit |
| `reference_prices.csv` | `scenario_merit_order` (the merit order of the scenario's own offers, NaN where there is no demand to clear), `model_no_storage`, `model_with_battery` (the merit-order model's prices), `admissible` (false where that model flags its own inputs) |
| `observed_prices.csv` | day-ahead: `n2ex_day_ahead`, `epex_day_ahead` (hourly auctions, licensed), `blend_day_ahead` (the two weighted by volume), `epex_hh_day_ahead` (the half-hourly auction, licensed); intraday: `epex_ida1`, `epex_ida2` (licensed), `apx_mid`, `system_price` (public). Git-ignored as a whole |
| `penetration_forecasts.csv` | `wind_tx_forecast_mw` (transmission-connected wind, WINDFOR) and `tsd_forecast_mw` (transmission system demand) or, where the forecast state has none, `nd_forecast_mw` (national demand), as forecast at 09:00 D-1: the penetration axis of the causal reference of `validation/`. Written by `build`, or alone by `python -m assume_gb penetration --years FROM TO`; not read by the framework |
| `scenario_meta.json` | the model run the inputs come from, counts, the battery fleet, the size of the forecast errors |

The series start one day before the first delivery day, because the auction for a day is held the
day before; that lead-in day repeats the first day's inputs and delivers nothing. The last row only
closes the index.

## Units

The columns `kind` and `support_regime` describe a unit and are not read by the framework. Columns
of the stack that offer under identical terms are merged into one unit, so a unit is a class of
plant, not a plant. The offer of each kind, with the framework column that carries it:

| `kind` | How it is written | Offer, GBP/MWh |
| --- | --- | --- |
| `thermal` | fuel, `efficiency`, `emission_factor`; `additional_cost` is a start-up premium per MWh | (fuel + carbon x emission factor) / efficiency + premium |
| `ro` | `support_scheme: premium`, `support_value` = certificates per MWh x certificate value | avoidable cost - premium |
| `fit` | `support_scheme: premium`, `support_value` = generation tariff | avoidable cost - tariff |
| `cfd_intermittent` | `support_scheme: cfd`, `support_value` = strike price; `support_neg_price_rule` | avoidable cost - strike; under the any-hour rule zero (the model: avoidable cost) |
| `cfd_baseload` | `support_scheme: cfd`, `support_reference: bmrp` | avoidable cost - (strike - `bmrp`); energy from waste offers the floor |
| `merchant_vre` | `additional_cost` 0 | 0 |
| `must_run` | negative `additional_cost` | -15 (gas plant at minimum stable generation), -10 (biomass), -5 (embedded) |
| `avoidable` | `additional_cost` | avoidable cost |
| `interconnector` | `fuel_type` is the unit's own price column | price of the neighbouring market net of losses |
| `unserved_demand` | `additional_cost` 3000 | the cap, so a half-hour the fleet cannot cover clears there |

Three devices of the merit-order model are kept: interconnectors are price steps, not fixed flows
(an import unit offers its capacity at the neighbouring price; the export capacity is part of the
demand and an export unit offers it back at that price); nuclear and pumped storage are not units,
their output is subtracted from the demand; the demand is therefore the GB demand less nuclear and
pumped-storage output plus the export capacity of the interconnectors.

## Reading the results

- The CSV export writes numbers with five significant digits; a database keeps them in full.
- Cash flows, costs and profits are price x MW per period, an hourly rate at half-hourly periods:
  multiply by 0.5 for GBP. Energy volumes in `market_meta` are MWh.
- The cost accounting takes the absolute value of the marginal cost; for the units whose "cost" is a
  price (interconnectors, unserved demand) or negative (must-run blocks) costs are not meaningful.
- The payments of the support contracts are in `support_cashflow` of `unit_dispatch`, beside the
  market cashflow `energy_cashflow`; the bids of the supported units follow from the contracts.
- Dispatch is written each time an auction reports back, up to that moment, so `unit_dispatch` and
  `market_dispatch` end at the close of the last auction of the run.
- In the `ID` auction a unit at the margin offers to sell more and to buy back at one price, and the
  clearing may accept both: a swap that changes nothing but the `supply_volume` of `market_meta`
  (0.3 percent of the half-hours of 2023); units that offer at the same price can swap among
  themselves in the same way. The traded volume is the net change of position per offer price,
  from `market_orders`: sum the accepted volumes of the `ID` orders per period and price.
- The simulated day-ahead market is one market for all demand. It is compared with both hourly
  auctions (N2EX, EPEX) and their volume-weighted blend on hourly means, and with the half-hourly
  EPEX auction period by period. The two hourly auctions were one price until 2020 and differ by
  3.6 to 11.6 GBP/MWh on average since 2021; N2EX carries about 70 percent of the volume.

## Figures

```bash
python -m assume_gb run --year 2023 --case day_ahead --csv C:/assume_runs --figures   # simulate, then draw
python -m assume_gb figures C:/assume_runs/gb_2023_day_ahead_storage --baseline C:/assume_runs/gb_2023_day_ahead
python -m assume_gb learning --db sqlite:///C:/assume_runs/learn.db --simulation gb_2023_rl_seed1 gb_2023_rl_seed2
```

`figures` reads the framework's CSV output of one run and the scenario folder behind it and writes,
to `results/figures/<run>/`, the standard set below as PNG and PDF, their captions (`captions.md`),
the headline numbers as a table (`00_summary.md`), the numbers the captions quote (`numbers.json`,
`numbers.csv`) and `viewer.html`. A run writes about 600 MB of CSV a year: put `--csv` outside
OneDrive. The first reading keeps a parquet copy of each table beside it; a year then draws in 20
to 40 s and the viewer in 10 to 20 s. The figures are plain academic figures (`style.py`: axes,
legends and short panel labels only, Okabe-Ito colours, Arial, 16:9); all text is in the captions:
the figure's message as a bold first sentence, computed from the run, then what the figure shows
and the source. Redrawing a figure (`--only 01,10`) replaces its own caption. The figures and the
viewer hold the licensed observed prices: they stay local like the data.

The set follows what validation and agent-based studies of electricity markets report. The
template for GB is Ward, Green and Staffell (2019, Getting prices right in structural electricity
market models, Energy Policy 129): a model must reproduce the distribution and the shape of prices,
not only their mean. Prices are compared with N2EX on hourly means, as in `compare`.

| Figure | What it shows | Following |
| --- | --- | --- |
| `00_summary.md` | a table, not a figure: the headline numbers: price statistics simulated and observed (mean, spread, percentiles, extremes, daily range, hours below zero, hours above the observed 99th percentile), the fit (R², correlation, bias, MAE, RMSE, ratio of standard deviations, slope), the market (demand served and its cost, imports and exports, shares, price setters, capture prices, storage) and an order-book check of every market | Ward et al. Table 2; Lago et al. (2021, Applied Energy) for the error metrics |
| `01_price_year` | daily mean price, simulated and observed, with the monthly mean error | the price time series of every backcast |
| `02_price_duration` | hourly prices sorted, both ends stretched (logit axis): spikes and negative prices | Ward et al. (log tails); Nitsch and Schimeczek (2022, AMIRIS back-test); ASSUME's own dashboard |
| `03_daily_shape` | price by hour of the day per quarter: mean, 10th and 90th percentile | Ward et al. Figs 1 and 12 |
| `04_fit` | hourly prices against observed (density), and the monthly mean, 10th and 90th percentile against observed with the RMSE of each | Ward et al. Figs 8 and 9 |
| `05_error_map` | the hourly error by day and hour: where the model misses | diagnostic |
| `06_price_vs_residual_load` | price against demand less wind and solar, simulated and observed, with binned medians: the supply curve each market reveals | Ward et al. Fig. 11 |
| `07_generation_mix` | energy sold per month by group of plant, exports below zero | Ward et al. Figs 13 and 14; Electric Insights |
| `08_week_high`, `08_week_low` | the weeks with the most and the least demand left after wind and solar: price above, the stack of what was sold below (no second axis) | Electric Insights; Energy-Charts |
| `09_price_setter` | which group of plant sets the price, by month and hour | Blume-Werry et al. (2021, Economics of Energy and Environmental Policy) |
| `10_supply_curves` | the offers of the cheapest, a typical and the dearest half-hour as merit orders, with the demand and the observed price | Ward et al. Fig. 4; the exchanges' aggregated curves |
| `11_capture_prices` | price earned per MWh by group against the average price (value factor), and market revenue | Hirth (2013, Energy Economics) |
| `12_storage` | storage case: battery output and price by time of day with and without the batteries, revenue per month | ASSUME (SoftwareX 2025) Fig. 2 for storage profits |
| `13_intraday` | intraday case: intraday less day-ahead price by how long or short the system turned out, simulated and observed (market index, system price, IDA1) | the spread analysis of this work |
| `14_vs_baseline` | with `--baseline`: prices and revenue by group against another run (a counterfactual, or the competitive run for strategic bidders) | Borenstein, Bushnell and Wolak (2002, AER) |
| `15_offers_vs_baseline` | with `--baseline`, when units bid differently: each unit's offer against its offer in the baseline (mark-up) and the Lerner index of the price-setting offer | as above; Ye et al. (2020, IEEE Trans. Smart Grid) |

`viewer.html` is the interactive snapshot: pick any half-hour of the year (keys, date and time,
jumps to the dearest, cheapest and worst-fitted half-hours, or a click on a chart) and see the
headline numbers of that half-hour, the year, the day (prices, then the stack of what was sold) and
the order book of either market as a merit order with the demand, the clearing point, the observed
price and the price-setting unit, every offer with its unit, MW, acceptance and support contract,
and the same book as a table. The address keeps the half-hour and the market
(`viewer.html#t=2023-11-30T17:00&m=ID`). It is one file with Plotly.js inside and needs no network.

`learning` draws a learning run from its database (learning runs need one): `20_learning_curves`
(reward and profit per episode, training and evaluation), `21_training_diagnostics` (critic and
actor losses, gradient norms, exploration noise) and `22_actions` (the actions by time of day in the
first and the last evaluation). Learning results vary between seeds: give several simulations of
the same experiment with different seeds, and the curves become their mean with a 95% bootstrap
interval (Henderson et al. 2018, Deep reinforcement learning that matters; Agarwal et al. 2021,
NeurIPS; Ye et al. 2020 used ten seeds). What the learned bids do to the market is `figures` of the
learning run with the competitive run as `--baseline` (figures 14 and 15).

Not in the set yet, and why: generation by fuel against observed (no observed generation in the
scenario folder), comparisons across years (a Taylor diagram of the eight years would summarise
correlation, spread and error at once), significance tests of one model against another
(Diebold-Mariano, as in Lago et al.), and for learning agents the interquartile mean over seeds,
profit gain against the competitive and the collusive outcome (Calvano et al. 2020, AER) and
explanations of the policy (SHAP, as in ASSUME's tutorial 9).

## Validation

`validation/` is the validation of the simulated day-ahead prices against the observed ones: a
scorecard of eight criteria (accuracy, error structure, distribution, Diebold-Mariano tests against
the naive forecast and the competitive run, mechanisms against causal estimates, markups by
scarcity, the value of the prices for decisions, the spread of the seeds) and a standard set of
figures, following the synthesis note on validating agent-based price models.
`python -m assume_gb validate --year 2023 --arm <learning arm>` writes them for a year with the
arm's seeds as the agent-based model, into `results/validation/`;
`python -m assume_gb penetration --years FROM TO` writes, where the GB data repository is, the
forecasts that put the binned wind and solar effects on the axis of the causal reference
(`penetration_forecasts.csv`). `validation/README.md` explains the reasoning, the windows (the
headline window holds no data used for tuning), the information sets and every output.
`results/README.md` describes the results folder: what each file is, which command writes it and how to evaluate runs.

## Portfolio agents and HPC

`build --learning-pick portfolio` makes one learning agent per owner of the gas plants and the
named wind farms (`--learning-techs CCGT Wind`), with the framework's portfolio strategy made
aware of support contracts (`portfolio_learning_support`): the agent sees its whole portfolio and
chooses a mark-up on marginal cost per cost bin (`--nbins`, also the least plants an owner needs;
`--max-markup`, a factor, default 2; `--min-portfolio-mw`, default 500), its plants keep their
day-ahead strategies (a wind farm under the RO is bid from its contract floor), and its reward is
its profit less the competitive profit of the same plants, contract payments included. It cannot
bid below cost, which is what made the plant-level agents collapse. Owners come from DUKES 5.11
for the gas plants (the end-of-year sheet of the year; `data/raw/dukes/DUKES_5.11.xlsx` in the GB
data repository, read with openpyxl) and from Ofgem's register of accredited RO stations for the
wind farms (`data/raw/ofgem/RO_Accredited_Stations_*.csv`; the accreditation holder, a project
company, reduced to the group that trades the output in `owners.WIND_GROUPS`); joint ventures are
assigned to the operating partner by judgement (noted there). The named wind farms are no longer
merged into classes, so a year holds about 185 units. For 2023 the agents are RWE, SSE, Orsted,
ScottishPower, Vattenfall, VPI, InterGen, EPUKi, Uniper and EDF, 60 plants; the owner maps of
2018-2020 miss some gas plants because older DUKES editions name them differently. The register
also dates every station's accreditation, from which its RO support ends after twenty years
(`owners.ro_register`, column `ro_end`); the merit-order model does not yet retire RO support, see
`HANDOVER.md`.

The Electricity Generator Levy: `python -m assume_gb egl RUN_FOLDER [--benchmark 75]
[--hedge-share 0.5] [--observed-prices] [--nuclear] [--validate]` settles the levy on a run the way
it is assessed (realised receipts of a group over the calendar year, benchmark price, 45 percent,
the first 10 million exempt, groups of 50 GWh or less out of scope, CfD output and fossil plant out
of scope), grouped by owner, with an optional share of output sold forward at a contract price, at
the run's prices or the observed N2EX price, with the nuclear fleet added from its metered output
(it is not a unit: the model nets it off demand), and a named company's share of a merged class
attributed to it through `unit_columns.csv` (Drax's RO units). `--validate` puts the settlement
beside what the levy raised: the OBR's accrued outturn (the liability of the period, the nearest
match to a calendar-year settlement, the calendar year taken as a quarter of one fiscal year and
three quarters of the next), HMRC's cash receipts by calendar and fiscal year (quarterly
instalments, which lag), the Autumn Statement 2022 costing and the EDF and Drax disclosures (all
from the GB data repository, `data/processed/egl_receipts_hmrc.csv` and `data/raw/manual/`), and
backs out the realised price each disclosure implies for the generation attributed to the company,
which shows what its forward sales did. `build --egl-rate 0.45 --egl-benchmark 75` puts
the levy on the in-scope units as the framework's per-period levy (`levy_rate`, `levy_benchmark`),
which the learning rewards see; it is the marginal view and overstates the annual settlement.

For sweeps and seeds, `variant` derives study cases with settings changed, each with its own
simulation id, database and policies:

```bash
python -m assume_gb variant --year 2023 --case learning_s0 --set seed=0
python -m assume_gb variant --year 2023 --case learning_s1 --set seed=1 --set learning_config.noise_sigma=0.2
python -m assume_gb run --year 2023 --case learning_s1 --save
python -m assume_gb learning --db sqlite:///assume_gb/results/db/gb_2023_learning_s0.db        --simulation gb_2023_learning_s0 gb_2023_learning_s1     # several ids = seeds, with intervals
```

### HPC (PBS Pro)

The framework is in the fork on GitHub; `assume_gb/` is not, and is copied beside it. Nothing reads
the GB data repository at run time. A run is single-threaded Python with a few torch threads, so a
cluster is used by running many cases at once, and a GPU does not help.

```bash
# once, on the cluster
git clone https://github.com/lukasschirren/assume.git && cd assume
#   then copy assume_gb/ into it from the laptop, without results and old policies, e.g.
#   rsync -av --exclude results --exclude learned_strategies --exclude __pycache__ assume_gb/ <user>@<login node>:assume/assume_gb/
module load miniforge/3 && miniforge-setup          # Imperial RCS: conda in ~/miniforge3
source ~/miniforge3/etc/profile.d/conda.sh                # as in train.pbs
conda create -n assume python=3.12 pip && conda activate assume
pip install torch --index-url https://download.pytorch.org/whl/cpu    # the CPU build, a tenth of the size
pip install -e ".[all]" openpyxl

# the study cases of a campaign (windows x seeds), written into config.yaml and hpc/cases.txt
bash assume_gb/hpc/make_cases.sh

# submit, from the repository root
qsub -v CASE=learning_week assume_gb/hpc/train.pbs      # the smoke test first: four episodes, 10 min
qsub -J 1-9 assume_gb/hpc/train.pbs                     # then one sub-job per line of cases.txt
qstat -t -u $USER                                       # the sub-jobs; qdel <job id> removes one
```

`hpc/train.pbs` runs one case: the training, its learning figures and, for each companion the
case has, the year with the trained policies (`<case>_year` with those of the best evaluation
episode, `<case>_year_last` with those at the end of training). It works on the node's own disk
(`$TMPDIR`) and copies back when the job ends, however it ends: the database to
`results/db/<run>.db`, the run's log to `results/logs/<run>.log` (written live), the framework's
log to `results/logs/<run>.assume.log`, the CSV output to `results/csv/`, the learning figures to
`results/learning/<run>/`, the prices of all runs to `results/prices/<run>.csv`. Jobs submitted
before 7 Oct 2026, 15:00 write the old flat layout: run `python -m assume_gb tidy --apply` after
they finish (`results/hpc` is a link to `results/logs` for their logs). The
policies are in `inputs/gb_<year>/learned_strategies/<run>/`. Resources are per run (4 cores,
16 GB, 24 h): change them in the file or with `qsub -l`. `RESULTS=<folder>` (with `qsub -v`) sends
the results elsewhere, `YEAR_CSV=1` keeps the year's full CSV output (about 600 MB a run).

`hpc/make_cases.sh` writes a campaign: `learning_<window>_s<seed>` for each window and seed, each
with its `_year` and `_year_last` companions (`CASES_FILE` names the list; submit it with
`qsub -J 1-N -v CASES_FILE=$PWD/<list> assume_gb/hpc/train.pbs`). By default the windows are the first two weeks of January, March and
September and the seeds 0 to 2; `WINDOWS`, `SEEDS`, `EPISODES`, `EXTRA` (further settings for
every case) and `TAG` (a suffix for the names) change that, see the head of the file. Run it before
submitting and never while jobs run, because it rewrites `config.yaml`. `hpc/evaluate.pbs` runs
year cases without training (an array over `hpc/cases_last.txt`), `hpc/tests.pbs` the framework's
tests on a node, and `python assume_gb/dev/campaign.py '<case glob>'` summarises a campaign. Afterwards (on the
cluster, or on the laptop after copying `results` and `learned_strategies` back) the seeds of a
window are pooled in one set of figures, each run from its own database:

```bash
R=assume_gb/results
python -m assume_gb learning --out $R/learning/gb_2023_learning_sep \
    --db sqlite:///$R/db/gb_2023_learning_sep_s0.db sqlite:///$R/db/gb_2023_learning_sep_s1.db sqlite:///$R/db/gb_2023_learning_sep_s2.db \
    --simulation gb_2023_learning_sep_s0 gb_2023_learning_sep_s1 gb_2023_learning_sep_s2
```

`observed_prices.csv` is licensed: copy it only if the cluster is a place it may be; without it a
run compares with the merit-order model only. A two-week episode takes 25 s without and about 55 s
with training for ten plant-level agents on one laptop core; the portfolio agents are slower (a
week of four episodes took 7 min).

## Sources

The fleet, the demand and the availability derive from Elexon, NESO and LCCC data and from government
registers (REPD, DUKES, the Renewables Obligation and CfD registers); the wind and solar profiles
from physical notifications (Elexon) and reanalysis capacity factors (renewables.ninja); the
neighbouring prices from ENTSO-E; the fuel and carbon prices from public quotations; the day-ahead
forecasts from Elexon and NESO; the observed prices from Nord Pool and EPEX (licensed) and Elexon
(public). The terms of each source apply to the files derived from it.
