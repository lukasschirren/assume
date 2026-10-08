<!--
SPDX-FileCopyrightText: ASSUME Developers

SPDX-License-Identifier: MIT
-->

# ASSUME for Great Britain: state of the work and how to continue (6 Oct 2026)

Written for whoever picks this up next, on the HPC or in another chat. It is self-contained; the
repository's `CLAUDE.md` (working conventions, framework gotchas, numbers) and
`assume_gb/README.md` (the scenario files and the tool's commands) hold the detail. The first
training array and a second campaign (September, four weeks, three arms: the stock cost bins,
cost bins by capacity, the same with the levy) have run on the HPC (section 3): the trained owners
bid close to competitively in every arm, because the supply curve is flat where gas sets the
price. A third campaign (6 Oct, array 4283919: the steepest winter window, 6 Nov–3 Dec 2023,
the agents of the new ownership data, the gas fringe made strategic) finds market power where
the supply curve is steep: one owner withholds 0.3–1.4 GW in the tight peak half-hours, and the
price there rises by about £20, three quarters of the premium observed in those same half-hours.
But the learning flips between this state and competitive bidding, run to run and within runs
(section 3; next steps in section 5).

## 1. What this is

[ASSUME](https://github.com/assume-framework/assume) is an agent-based electricity market
simulator (mango agents, markets cleared by configurable algorithms, deep-reinforcement-learning
bidding with MATD3) written for the German market. This fork (`lukasschirren/assume`) adapts it to
the GB wholesale market in three stages agreed with Lukas Schirren (Imperial College London):

1. **Day-ahead backcast** of 2018–2025 against his merit-order model and observed prices: done.
2. **Intraday auctions with forecast error**: built and run; on hold by decision.
3. **Learning agents**: the machinery works end to end with portfolio agents per owner (gas and
   RO wind); the first experiments ran; the experiment design (training window, seeds, sweeps)
   is the open question and the HPC is where it continues.

Rules that shaped everything (keep them):

- Nothing pushed to GitHub may read, import or point at the sibling data repository
  `../Agent-Based Electricity Pricing GB` (the user's data pipeline and merit-order model).
- GB behaviour lives in the framework's own structure (`assume/strategies/…`, unit parameters,
  forecaster fields, `tests/`, `docs/source/`), all additive and optional, so stock scenarios run
  unchanged. Framework behaviour is only changed after telling the user (three such changes so
  far, all in the stock portfolio learning strategy: see section 2).
- `assume_gb/` is local (listed in `.git/info/exclude`): the tool that builds the scenarios from
  the merit-order model's export, the scenario data, results and figures. It is never pushed.
  No GB data goes to GitHub for now. `observed_prices.csv` in every scenario folder holds
  licensed exchange data (N2EX, EPEX): keep it private.
- The merit-order model (`src/meritorder/comb.py` in the GB repo) is the validated reference; do
  not run it without need. Nothing has been committed in either repository.
- Environment: conda env `assume` (Python 3.12, `pip install -e ".[all]"`, plus `openpyxl`);
  never touch base Anaconda or the system Python; say what and why before installing anything.

## 2. What is built

### In the framework (pushable; 534 tests, ruff clean; on the cluster one fails on its timezone, see section 6)

| Piece | Where |
| --- | --- |
| Bidding with a support contract (RO as a premium, CfD with strike and negative-price rule, baseload CfD against a reference series) | `assume/strategies/support_strategies.py` (`powerplant_energy_naive_support`); columns `support_scheme`, `support_value`, `support_neg_price_rule`, `support_reference` on `PowerPlant` |
| Contract payments and a revenue levy in the cash flows | `PowerPlant.support_payment` → `outputs["support_cashflow"]`; `PowerPlant.levy_payment` (`levy_rate`, `levy_benchmark`: the share of receipts above a benchmark price) → `outputs["levy_cashflow"]`, both beside the market cashflow |
| Forecasts beside outturns, and trading the deviation | `availability_forecast` / `demand_forecast` on the forecasters, `availability_forecast_df.csv` / `demand_forecast_df.csv`; `assume/strategies/forecast_strategies.py` |
| Storage schedule optimisation | `assume/strategies/storage_strategies.py` (`storage_energy_optimization_schedule`, pyomo/HiGHS) |
| Learning with contracts and the levy | `assume/strategies/learning_support_strategies.py`: `powerplant_energy_learning_support` (plant-level) and `portfolio_learning_support` (one agent per owner, bids every plant from its contract floor plus a chosen mark-up, profit and competitive benchmark include contract payments and the levy). Hooks added to the stock strategies for this (`unit_income`; `unit_floor`, `unit_bid`, `unit_income`), defaults unchanged |
| Fixes to the stock `PortfolioLearningStrategy` (behaviour changes, told to the user) | cost bins scaled by the dearest plant (it raised when a plant cost more than any forecast price), inframarginal share scaled by installed capacity (it divided by the residual load, zero or negative in GB) |
| Cost bins by capacity (6 Oct, an option; **not yet committed**, made on the cluster) | `cost_bins: capacity` in `bidding_strategy_params`: `PortfolioLearningStrategy` splits a portfolio into bins of equal installed capacity by marginal cost (`capacity_cost_bins`) and keeps each unit's bin for its bid, and then allows fewer units than `nbins` (empty bins); default `count` (the stock quantiles) unchanged. Tests in `tests/test_portfolio_learning_strategy.py`, docs in `unit_operator.rst` and the release notes |
| Faster bookkeeping | `common/base.py`, `fast_pandas.py`, `utils.py`, `units/powerplant.py`: bit-identical results, a GB year in 60–100 s instead of 220–300 s |

### In the local tool `assume_gb/`

- `build --year Y --tag register --learning-techs CCGT Wind --learning-pick portfolio
  --learning-weeks 2` writes `assume_gb/inputs/gb_Y/` from the merit-order model's offer stack
  (exported once per year in the GB repo with `--ro-fleet register`), the day-ahead forecasts and
  the observed prices; 173–198 units (merged classes of plant; the named CCGTs and the RO wind
  farms are plants). All eight year folders are built this way.
- Study cases per folder: `day_ahead`, `day_ahead_storage`, `day_ahead_intraday`, `learning`,
  `learning_year`, each also as `*_week`; `variant` derives further cases (seeds, sweeps).
- `run --year Y --case C [--out DIR] [--csv DIR] [--db URI]` simulates and prints the comparison
  with the merit-order model and the observed prices; a learning case trains first.
- `egl RUN_FOLDER [--benchmark] [--hedge-share] [--observed-prices] [--nuclear] [--validate]`
  settles the Electricity Generator Levy on a run; `--validate` sets it beside the OBR's accrued
  outturn, HMRC's cash, the 2022 costing and the companies' disclosures, and backs out the
  realised price each disclosure implies.
- `figures RUN_FOLDER [--baseline RUN_FOLDER]` and `learning --db URI --simulation ID …` draw
  the standard figures; `hpc/train.pbs` runs one case per PBS job or array sub-job,
  `hpc/make_cases.sh` writes the cases of a campaign.
- `owners.py`: who owns which plant (DUKES 5.11 for gas; Ofgem's RO register for the wind farms,
  with each station's accreditation date and the day its RO support ends).
- Tests: `python -m pytest assume_gb` (28); `dev/regress.py` for framework bookkeeping changes;
  `dev/campaign.py` summarises a learning campaign from its databases and price files;
  `dev/learning_sets.py` writes the portfolio agents of a folder from its `unit_owners.csv`.

## 3. Results so far

- **Engine**: with the model's stack the simulated day-ahead price equals the merit order in every
  half-hour of every year (1e-13).
- **Fit to the observed day-ahead price** (hourly R², no storage): 2018 0.35, 2019 0.57, 2020
  0.60, 2021 0.39, 2022 0.67, 2023 0.79, 2024 0.68, 2025 0.61 against N2EX; EPEX hourly fits
  better in four of the five decoupled years (2023: 0.81). The bias (cost-based prices below
  observed: −1 to −2 £/MWh in 2018–2020, −20/−27 in 2021/2022, −5 to −6 in 2023–2025) is what
  mark-ups would have to explain.
- **Register fleet** (5 Oct): the model's corrections against Ofgem's RO register (1.1 GW of
  pre-2002 hydro out of the RO, Galloper and Aberdeen Bay in, farms at installed capacity) move
  the 2023 price in 0.4% of half-hours (MAE £0.004); the fit is unchanged.
- **Storage**: the schedule strategy fits at least as well as the model's co-optimised battery in
  seven of eight years.
- **Intraday**: the intraday auction with full information re-clears to the outturn merit order;
  its spread matches the APX market-index spread in conditional means, not half-hour by
  half-hour; EPEX IDA1 does not react to the eventual error. On hold.
- **Learning, plant-level** (4 Oct): the ten largest CCGTs collapsed into never being dispatched;
  the ten cheapest reached a third of their competitive profit. Single plants have little power
  in a stack whose CCGT costs lie within £126–168; hence portfolio agents.
- **Learning, portfolio agents** (5 Oct): ten owners in 2023 (RWE, SSE, Ørsted, ScottishPower,
  Vattenfall, VPI, InterGen, EPUKi, Uniper, EDF; 60 plants, gas and RO wind) bid through
  `portfolio_learning_support`; the one-week smoke test (four episodes) runs in 7 min and its
  barely trained mark-ups lift the week's price £1.1/MWh (£2.8 on the HPC, 6 Oct).
- **Portfolio agents, first array** (HPC, 5–6 Oct): two-week windows from 1 January, 1 March and
  1 September 2023 x seeds 0–2, 50 episodes each (39–60 min per training, 13–22 min for the year).
  - Benchmark rule: in every run 8–10 of the 10 owners end above their competitive profit, but
    only just: the owners' total is 0.4–4% above it (£1.3–9.4m over two weeks).
  - Rewards fall during training in March and September, flat in January. In March the five
    random exploration episodes earn reward ~120 (profit £257m) against 4–28 for any trained
    policy (£232m): joint random mark-ups lift the price, and once trained each owner undercuts.
    The agents converge towards competitive bidding; critic loss falls steadily (no sign of a
    numerical failure). Mark-ups end bang-bang: 25–54% of actions at cost, 15–45% at the cap (2).
  - `avg_reward_eval_policies` (best evaluation, what `_year` loads) is the first evaluation in
    four of nine runs, i.e. barely trained. So every year was also run with `last_policies`
    (cases `*_year_last`, `hpc/evaluate.pbs`): price over the merit order for the year, final
    policies, Jan +2.2 to +4.3, Mar +0.1 to +0.3, Sep +0.2 to +0.9 £/MWh; R² against N2EX Jan
    0.819–0.822, Mar 0.797–0.802, Sep 0.799–0.814 (baseline 0.791, bias −5.75, MAE 10.69).
  - January's better fit is not learned market power: there the gas-only owners (EPUKi,
    InterGen, Uniper, VPI) have a competitive profit of about zero (gas out of merit), so a high
    mark-up costs them nothing in training and lifts the price when applied to the rest of the
    year. Even in March and September these four earn only £0.4–1.8m in two weeks; profit sits
    with the wind owners.
  - **Wind owners have no lever.** The bid is floor + marginal cost x (mark-up − 1); an RO farm
    has marginal cost £3 and a floor of −£63 to −£162, so a mark-up of 2 moves its bid by £3.
    Ørsted, ScottishPower and Vattenfall (wind only) cannot change their bids in any way that
    matters; their ratios of ~1.00 come from the gas owners' price effect. Kept so by decision
    (6 Oct, section 4): withholding an RO MWh forgoes the price plus £96–132 of support, gas only
    its margin, so a rational owner withholds gas first and wind practically never.
  - **RWE and SSE had one mark-up for all their gas.** The cost bins are count quantiles of the
    plants' marginal costs; with 8–9 wind farms at £3 beside 4 CCGTs the median is £3, so the
    wind farms take bin 0 and all four CCGTs (RWE 5.4 GW, SSE 4.0 GW) share bin 1: marking up the
    dear plants prices out the cheap ones, and the second action falls on wind. Their "gas at
    cost" (mark-up 1.00 in every half-hour) is at least partly this. More bins do not help (the
    gas-only owners have two plants; RWE's gas only splits from six bins). Fix, 6 Oct: the option
    `cost_bins: capacity` of the stock `PortfolioLearningStrategy` (bins of equal installed
    capacity by cost; default `count` unchanged): RWE wind + Didcot | Pembroke, Staythorpe, Little
    Barford; SSE wind + Seabank | Keadby, Medway, Peterhead; VPI's Damhead moves to the dear bin;
    the other owners as before. Checked in the smoke test `learning_week_sep_cap_egl` from the
    bids: two mark-ups among RWE's and SSE's CCGTs in every half-hour.
  - Pooled figures per window: `results/learning/gb_2023_learning_{jan,mar,sep}_pooled/`.
- **Portfolio agents, September campaign** (HPC, 6 Oct, array 4281171; read with
  `python assume_gb/dev/campaign.py 'learning_sep4*'`, pooled figures in
  `results/learning/gb_2023_learning_sep4{,_cap,_cap_egl}_pooled/`): 1–28 September 2023, seeds
  0–2, 50 episodes (75–95 min of training, 12–21 min per year run), three arms: stock count bins,
  `cost_bins: capacity`, capacity bins with the levy (45% above £75 on the in-scope units).
  - Same pattern in all nine runs: random exploration loses (reward −50), the first trained
    episodes earn +50 with joint high mark-ups, then each owner learns to undercut and by episode
    30–40 the seeds converge on near-competitive bidding (reward ~6). The best evaluation is the
    first one (episode 10) in every run, so `_year` holds barely trained policies; `_year_last`
    is the trained state.
  - Final policies: 8–10 of 10 owners above the benchmark, but the owners' profit is only
    0.1–0.5% above competitive (£0.2–1.5m over four weeks, of about £280m); the September price
    +£0.05–0.27/MWh over the merit order; the year +£0.2–0.7 (R² against N2EX 0.798–0.811,
    baseline 0.791). With the best-evaluation policies the year is +£0.6–2.7 (R² up to 0.821),
    which is the exploration burst, not learned behaviour.
  - Arms, means over seeds: price lift in the window 0.09 (count), 0.17 (capacity), 0.18
    (capacity + levy) £/MWh; excess profit £0.6m, £1.0m, £0.9m. Capacity bins raise both in each
    of the three seeds (paired), but by little. The levy's paired differences change sign from
    seed to seed: no detectable effect on bids. It acts as a transfer: RWE's competitive profit
    over the four weeks falls by £1.6m, SSE's by £1.1m.
  - RWE, with its own action for its dear CCGTs (Pembroke, Staythorpe, Little Barford), marks them
    up by 1.00–1.32 (mean 1.08; 1.14 with the levy): its bidding at cost was not an artefact of
    the bins. SSE's dear bin (Keadby, Medway, Peterhead, £96–98) goes from 1.00 to 1.92 between
    seeds while SSE's profit equals its competitive profit: those plants are out of merit or at
    break-even, the reward does not depend on their mark-up, and its value is arbitrary. A
    mark-up at the cap on such a plant is not market power.
  - Why so little: the September supply curve is flat at the clearing price (baseline order book,
    `results/csv/gb_2023_day_ahead`): taking 0.5 / 1 / 2 GW of offers away raises the price by a
    median £0.00 / £0.71 / £1.74 (mean 0.76 / 1.73 / 3.43; under £1 in 79 / 56 / 28% of
    half-hours). Withholding a plant gives up its margin for a price rise of well under £1 on the
    rest of the portfolio.
  - Reading: in this set-up learned day-ahead market power is small; it does not explain the
    merit order's −£5.75 bias against N2EX, and a levy with no market power to curb changes no
    bids.
- **Winter campaign** (HPC, 6 Oct, array 4283919, `hpc/cases_novdec.txt`; read with
  `python assume_gb/dev/campaign.py 'learning_novdec_own_s?' 'learning_novdec_fringe_s?'
  'learning_novdec_fringe4_s?'`; pooled figures `results/learning/gb_2023_learning_novdec_*_pooled/`):
  6 Nov–3 Dec 2023, seeds 0–2, 50 episodes (76–99 min of training), capacity bins; arms `own`
  (10 owners of the new ownership data), `fringe` (13, single-plant gas operators and Ratcliffe
  added), `fringe4` (four bins).
  - Price over the merit order in the window, final policies: `own` +2.08 / +1.54 / +1.51 by
    seed; `fringe` +0.92 / +1.31 / +0.01; `fringe4` +0.78 / +0.02 / +0.46 £/MWh (September:
    +0.05 to +0.27). It sits in the 56 half-hours (2% of the window) that are steep (1 GW
    withheld raises the price above £5) and priced at £80 or more: there +£17.7 to +21.3 (+9.7 in
    `fringe4` s2), up to +£94; elsewhere +£0.05 to +1.34. The year runs stay within +£1.0.
  - Who: in each run with a rise, one owner withholds 0.3–1.4 GW of mid-merit CCGT in those
    half-hours (bids above the clearing price at a cost below it): SSE (Marchwood and Seabank,
    980 MW, `own` s0; Keadby, Peterhead, Medway, 1.4 GW, `own` s1; Peterhead, `fringe4` s0),
    Uniper (Connah's Quay, 530 MW, `fringe` s0), VPI (Rye House, 280 MW, `fringe4` s2). The
    withholder bears the cost, the others gain: in `own` (£m over four weeks against competitive)
    RWE +6.0 to +6.4 on 183, each gas-only owner +1.6 to +2.4 on 0.7–4.1 (so their ratios of
    1.5–2.8), SSE −5.0 in s0 (the benchmark rule fails there) and +1.3 in s1 and s2. Owners'
    profit 2.3–3.9% above competitive (£11–18m).
  - Against the data: in the same 56 half-hours N2EX is £29.1 above the merit order (EPEX
    half-hourly £26.8), elsewhere £5.0; the simulation's market-power state gives £21.3 and £0.7.
    Learned withholding reproduces about three quarters of the observed peak premium, where the
    supply curve is steep; the £5 elsewhere is the merit order's general bias.
  - Not converged: the evaluation reward jumps between a competitive state (0–4) and a
    market-power state (50–70) within runs (`own` s0 and s2 competitive at evaluations 5–8,
    back up at 9; `fringe4` s2 between 2 and 130), so the final policy depends on when training
    stops. A strategic fringe makes it weaker and less certain: one seed of each fringe arm ends
    fully competitive (profit ratio 1.000). Single-plant owners bid at cost and gain from
    others' withholding (3–11% above competitive).
- **Overnight campaign, 6–7 Oct** (arrays 4284526, 4284527, 4284530, 4284531, 4284549; all 35
  runs finished): **200 episodes settle it.** 2023 `own` with 200 episodes (seeds 0–4): every seed
  ends in the same state, stable over its last 10–15 evaluations: SSE withholds Keadby, Medway and
  Peterhead (1.4 GW) in the steep peak half-hours (seed 4 also InterGen, Coryton and Rocksavage,
  0.6 GW); the price there is £21.3–22.3 above the merit order (N2EX: £29.1), elsewhere +£0.85
  (N2EX +£5.0); window +£1.51–2.40; every owner above its competitive profit (the benchmark rule
  holds in all five), £ m over four weeks: RWE +6.0, SSE +1.3, each gas-only owner +1.4 to +2.4,
  total £18m (seed 4 £26m, profit ratio 1.039–1.056). The year with the final policies +£0.28–0.54.
  - With 100 episodes (10 seeds) the runs still flip throughout: steep-hour rise £0–21.9 (mean
    15.8), final state competitive in 2 of 10. The 13-owner `fringe` arm with 100 episodes ends in
    a market-power state in all 10 seeds, steep-hour rise mean £17.6, window +£0.36–3.19, but
    1–2 owners below their competitive profit in four seeds.
  - 2024 (18 Nov–15 Dec, 100 episodes, 5 seeds): mostly competitive, steep-hour rise mean £6.4
    (two seeds about £21) against an observed N2EX premium of £48.5 there (elsewhere 4.1).
    2025 (13 Jan–9 Feb): competitive in all five (rise £0.1) against an observed premium of £85.6
    in its 30 steep half-hours (elsewhere 4.3). Both need 200 episodes before they say anything:
    submitted 7 Oct morning (2024: 4286608, 2025: 4286609, 2023 `fringe` with 200: 4286610;
    `hpc/cases_{year}_{window}_{arm}_e200.txt`).
- **Validation (7 Oct, `assume_gb/validation`, the standard from now on)**: `python -m assume_gb
  validate --year Y --arm <arm> [--policies last]` writes the eight-criterion scorecard to
  `results/validation/gb_<year>_<arm>[_last]/`; quote the `headline` window (the year without the
  training window). 100-episode arms, headline MAE / bias / W1 / q99 error (competitive 2023:
  10.61 / −5.73 / 5.83 / −25.5): 2023 `own` best policies 10.24 / −4.50 / 4.67 / −19.6, last
  10.45 / −5.49; `fringe` best 10.14 / −4.31 / 4.51 / −14.5, last 10.36; DM test against the
  competitive run p < 0.001 in all four. Training window (in-sample): MAE 9.5–9.9 against 11.6,
  q99 error −7 to +0.1 against −66. 2024 headline 9.35 against 9.39; 2025 no change (both at
  100 episodes, not settled). Criterion 5's binned wind and solar effects need
  `penetration_forecasts.csv` in each folder (written where the GB repository is; copy it alone).
- **Exchange bid curves (7 Oct, `assume_gb/exchange_curves.py`, `bid_curves/`, licensed, local)**:
  N2EX (Nord Pool archive, daily JSON 2016-2025) and EPEX GB hourly (daily CSV 2020 onwards;
  2018-19 nested zips not read yet) resampled per hour onto one price grid,
  `bid_curves/derived/curves_<year>.npz`, built by `python -m assume_gb curves --years A B`
  (`hpc/curves.pbs`, 4 min for 2023-25). N2EX clears on its hourly curves plus the accepted block
  orders plus the coupled flow from NO2 (North Sea Link): reconstructed to £0.01-0.03 mean
  (95-99% within 10p); EPEX on its curves alone, £0.07-0.12 mean (69-83% within 10p; adding
  its block file makes it worse, so its curves hold the blocks). Only the excess supply
  S - D + shift is comparable with the model (the exchanges carry net positions, N2EX about a
  third of GB demand). `viewer.html` (2023-25 day-ahead, `results/figures/`) now sets the
  model's excess supply beside N2EX, EPEX and both, per hour, or shows the raw curves.
  Finding (2024, all hours): the price rise for 1 GW more demand is a median £3.0 on the
  exchanges together against £0.93 in the model, about three times in every price band and at
  peak and off-peak; yet over time the model reacts too strongly to demand (validation): its
  supply is a staircase, flat inside blocks of like plant and jumping between them, the observed
  curve smooth and moderately steep.
- **Non-convex thermal supply (7 Oct, `assume_gb/thermal.py`, `inputs/thermal_parameters.{md,csv}`,
  GB defaults)**: `python -m assume_gb thermal --years 2023 2025` writes `powerplant_units_nc.csv`,
  `_nc_avail.csv` and the cases `day_ahead_nc`, `day_ahead_nc_avail`: merchant CCGT, OCGT and coal
  get the minimum stable level, minimum up and down times (steps), ramps, hot/warm/cold start costs
  in GBP/MW (ENTSO-E wear at the year's GBP/EUR + start fuel at the year's mean fuel and carbon
  price; 2023 CCGT 80 / 134 / 175) and the flexable strategy; `CCGT_MSG` (the merit-order model's
  2.25 GW must-run block at -15) is removed; `_avail` also derates capacity by the CM de-rating
  factor (CCGT 0.913). The cases need `forecast_algorithms.price: price_naive_forecast` (the units
  must hold a price forecast; with `price_keep_given` the strategy raises KeyError 'DA' and its
  units do not bid: the first year runs of 7 Oct failed so). Full year, no fitting, MAE / RMSE / r
  competitive -> nc -> nc_avail: 2023 10.69 / 16.6 / 0.904 -> 10.49 / 15.6 / 0.922 -> 9.83 / 14.8 /
  0.927; 2024 9.53 -> 9.52 -> 9.13 (r 0.851 -> 0.892); 2025 9.93 -> 9.96 -> 9.60; q99 error -26.6 ->
  -19.9 (2023); the demand coefficient on the outturn 3.41 -> 3.02 (observed 2.29, now overlapping).
  Not fixed: the weekday swing (1.27 -> 1.24 against 0.37) and too few negative prices (0.3%
  against 1.2%). `validation/gb_<year>_nonconvex/`. RL on top: `powerplant_units_learning_nc_avail.csv`
  (`thermal.write_learning`), arm `learning_novdec_new_nc_e200_s0..4` (2023 winter, 200 episodes,
  capacity bins), submitted 7 Oct evening; smoke test `learning_week_novdec_new_nc` ran clean.
- **Price setters** (2023 baseline): gas sets the price in 39–58% of half-hours by month (least in
  January and December, most in September); Pembroke, Didcot, Staythorpe, West Burton and Grain
  most often.
- **Electricity Generator Levy, settled and validated** (benchmark £75, 45%, £10m allowance,
  50 GWh threshold, CfD and fossil out of scope; the nuclear fleet added from its metered output,
  Drax's units split out of the biomass class by capacity). 2023 on the baseline: at simulated
  prices £593m (EDF nuclear £213m, Drax £81m), at observed N2EX prices £973m (£314m, £113m).
  The OBR's accrued outturn for calendar 2023 is £975m (fiscal 2023-24 £1,200m; HMRC cash
  £1,137m in the calendar year, £1,473m in the fiscal year; the Autumn Statement 2022 costing
  £4,075m). The £380m between the two settlements is the model's −£5.8/MWh price bias times 45%
  of the in-scope output. By company: EDF's disclosed ~£200m implies a realised price of
  £86.9/MWh against a spot average of £93.5 (sold forward below spot), Drax's £205m implies
  £111.8 against £95.7 (sold forward above spot): forward sales moved the 2023 levy between
  companies more than they moved the total. 2024: spot (observed £72.6, simulated £67.1) is
  below the benchmark, every settlement is zero, yet £825m accrued (EDF >£400m, implying at least
  £98/MWh realised; Drax £161m, £107.5/MWh): that levy was paid on hedges struck in 2022–2023,
  out of reach of a spot model. 2025: £27m simulated, £176m observed (EDF £14m/£93m) against
  HMRC cash of £41m in fiscal 2025-26, an OBR forecast of nil and Drax's nil. Of the 2023 £593m,
  the named wind owners pay £57m (RWE £22m, Ørsted £20m, SSE £9m, ScottishPower £3m, Vattenfall
  £2m); £300m falls on the aggregated classes of small plants, which stand for many companies and
  so overstate it (one allowance and one threshold per class). The benchmark's CPI indexation
  from April 2024 is not applied (only FY2026/27, £82.61, is known); it could only lower the 2025
  figures.

## 4. Decisions taken on 5 and 6 Oct and what they led to

1. **Portfolio agents**, one per owner, the plants bid from their contract floors; owners from
   DUKES 5.11 (gas) and Ofgem's RO register (wind farms; the holder is a project company, reduced
   to the group that trades the output in `owners.WIND_GROUPS`; joint ventures to the operating
   partner by judgement: check Seabank → SSE, Greater Gabbard → SSE, London Array → RWE, West of
   Duddon Sands → Ørsted, Galloper → RWE). Owner maps of 2018–2020 miss some gas plants (older
   DUKES editions name them differently).
2. **Training window**: decided on 6 Oct (item 9). The candidates were September, or February/March, when gas sets the price
   most; whole-year training is feasible on the HPC. For validation of who sets the price, Elexon's
   Detailed System Prices (ISPSTACK) show the marginal units of the balancing mechanism; the
   day-ahead auction has no price-setter data, so the model's own price-setting unit is the
   comparison.
3. **Forward cover**: left out of the agents' rewards; the levy settlement shows its effect
   (`--hedge-share`).
4. **Benchmark rule**: a policy counts only if every agent reaches at least its competitive
   profit; the portfolio reward is defined against exactly that benchmark.
5. **RO register**: used for owners and for the register-corrected stacks. Each station's RO
   support ends twenty years after accreditation: of the REPD-matched RO fleet 0.8 GW has lost
   support by end-2023, 1.8 GW by 2025, 5.1 GW by 2030 (named farms: Crystal Rig phase 1 in
   2023, Black Law, Hadyard Hill and Farr in 2025, Barrow in 2026). The merit-order model keeps
   every accredited station under the RO regardless; retiring support at `ro_end` is a change to
   its agent build (`comb.py`, with its golden regression) that is still to be made there. The
   ROC certificates file in `data/raw/ofgem` is a partial extract and not usable as a check.
6. **Generator levy**: settled on runs by `egl` the way it is assessed, and available as a
   per-period levy on in-scope units (`build --egl-rate 0.45 --egl-benchmark 75`) that the
   learning rewards see, which is the marginal view and overstates the annual settlement. The
   benchmark is £75 in 2023 and CPI-indexed from April 2024 (£82.61 in FY2026/27): pass it per
   year. Spot prices were below the benchmark throughout 2024, so a levy experiment belongs to
   2023 (or 2025, where spot sits at the benchmark).
7. **Wind owners' action** (6 Oct): the mark-up on marginal cost stays; RO wind withholding is
   not worth its support, and the levy acts through the mixed owners (RWE, SSE, EDF), whose wind
   income above £75 it taxes and so lowers the gain from marking up their gas.
8. **Policies** (6 Oct): every campaign evaluates the year with both, the best evaluation
   (`<case>_year`) and the end of training (`<case>_year_last`); `make_cases.sh` writes both and
   `train.pbs` runs both.
9. **Window** (6 Oct): September (gas sets the price most often), four weeks; January is out.
10. **Cost bins by capacity** (6 Oct): an option of the stock strategy (default unchanged), used
    in the campaign's second and third arms; whether it becomes the set-up depends on that
    campaign.

## 5. Next steps

1. Done (5–6 Oct): the first array, three windows x three seeds (section 3). January is ruled out
   as a training window (gas owners have nothing at stake there); March and September both give
   near-competitive agents.
2. Done (6 Oct, array 4281171, `hpc/cases_sep4.txt`; results in section 3): September 1–28, 2023, seeds
   0–2, 50 episodes, three arms: `learning_sep4_s*` (stock count bins), `learning_sep4_cap_s*`
   (`cost_bins: capacity`), `learning_sep4_cap_egl_s*` (the same with the levy, 45% above £75,
   on the 112 in-scope units: `powerplant_units_learning_egl.csv`, written with
   `egl.unit_parameters` from the existing units file, no rebuild). Read it with
   `python assume_gb/dev/campaign.py 'learning_sep4*'` (per run: owners above the benchmark,
   profit ratio, excess £m, price lift in the window and over the year with both policies, R²
   and MAE against N2EX; per arm: each owner's mark-up per action and share at the cap).
   Answers: RWE and SSE do not mark up their dear gas much once they can; the levy changes no
   bids beyond the seeds' spread; the cap binds only on plants out of merit, where it does not
   matter, so a `max_markup=3` arm is not needed.
3. Done (6 Oct, chosen by the user: (b) and (c); results in section 3), array 4283919, `hpc/cases_novdec.txt`:
   6 Nov–3 Dec 2023, seeds 0–2, 50 episodes, capacity bins, three arms:
   `learning_novdec_own_s*` (the agents of the new ownership data as `build` would choose them:
   10 owners, EDF out, Triton Power in, Marchwood with SSE), `learning_novdec_fringe_s*` (every
   operator of a named CCGT, coal plant or wind farm with 500 MW: 13 owners, adding EIG, ESB and
   Calon with one CCGT each and Ratcliffe to Uniper; the unowned merged classes, about 7.5 GW of
   CCGT and the OCGT and oil classes, still bid at cost), `learning_novdec_fringe4_s*` (the same
   with four bins: RWE, SSE and Uniper get three mark-ups for their thermal plants). The agent
   files are `powerplant_units_learning_{own,fringe}.csv` and `unit_operators_learning_{own,fringe}.csv`,
   written on the HPC by `python assume_gb/dev/learning_sets.py` from the folder's
   `unit_owners.csv` with `build`'s own function (`--check` reproduces the 5 Oct files exactly).
   The window: the 28 days with the most half-hours that are both steep (1 GW withheld raises the
   price above £5) and priced at £80 or more (4.2%; the cold spell of 27 Nov–3 Dec has 10%).
   The earlier choices were: (a) take near-competitive day-ahead bidding as the finding and
   go to the counterfactuals as transfers (the levy by owner through `egl`); (b) look for market
   power where the supply curve is steep; (c) make the competitive fringe strategic (other
   owners' CCGTs bid at cost now), or give the large portfolios more than two bins.
   For (b), the steepness by month of 2023 (baseline order book; price rise when 1 GW of offers
   is taken away): the median is £0.37–0.75 in every month (July least, March most), so no month
   is steep on the whole; the steep half-hours (a rise above £5) are 2–4% of April–September and
   8–10% of October–January and March (p90 £4.6–5.2 in November–January). A training window in
   November–December 2023 holds the most of them; in January gas sets the price with margins
   near zero, which is why the first array found nothing at stake there.
4a. New ownership data (7 Oct, the standard from now on; earlier runs are not repeated): 2023
   default agents 11 owners (Calon with two CCGTs), `learning_cfd` 12 (Equinor added, CfD units in
   their companies' portfolios). First campaign on it: `learning_novdec_new_e200_s0..4` (array
   4287861) and `learning_novdec_cfd_e200_s0..4` (4287862), 2023 winter window, 200 episodes,
   capacity bins. A background script validates each 200-episode arm when its array ends
   (`results/validation/`).
4. Done overnight (6–7 Oct, chosen by the user: (i), (ii) and more), 100 episodes each:
   2023 6 Nov–3 Dec `learning_novdec_own_e100_s0..9` (array 4284526) and
   `learning_novdec_fringe_e100_s0..9` (4284527); 2024 18 Nov–15 Dec `own`, seeds 0–4 (4284530,
   `YEAR=2024`); 2025 13 Jan–9 Feb `own`, seeds 0–4 (4284531, `YEAR=2025`). Windows for 2024 and
   2025: the 28 winter days with most half-hours that are steep (1 GW withheld: above £5) and
   priced above the year's 75th percentile (£84.5, £93.4): 4.8% and 2.3%. Agent files for 2024
   and 2025 by `dev/learning_sets.py` (`own`: 10 and 11 owners; Calon has two CCGTs in 2025).
   Also 2023 `learning_novdec_own_e200_s0..4` (array 4284549): 200 episodes, against the 100-episode
   seeds 0–4, to see whether longer training settles the flipping (the exploration noise decays
   over all episodes, so these stay exploratory longer); training on a whole year would take
   about 20 h for 50 episodes and was not run.
   Read with `python assume_gb/dev/campaign.py --year Y '<glob>'`, which now also gives
   `reward_last3`, `ratio_last3` (mean of the last three evaluations) and each run's reward
   trajectory. The earlier plan was: the market-power state is reached in
   five of nine runs and the learning does not settle, so before reading numbers off one run:
   (i) more seeds per arm (10, to estimate how often the state is reached), (ii) longer training
   (100–150 episodes, to see whether it settles), (iii) report the mean over the last
   evaluations rather than the last one; then (iv) the same test on other tight windows (winter
   2022/23, 2024, 2025) against their observed peak premia.
5. In the GB repo: retire RO support at `ro_end` in the model's agent build, then re-export the
   stacks and rebuild. Also an owner-level aggregation of the small RO plants would let the levy
   settlement and the portfolios cover them.
6. Later: a multi-product learning strategy so agents bid the real 48-product auction (the
   learning cases use one auction per half-hour, which gives identical prices with competitive
   bidding).

## 6. Known pitfalls

- **Results layout (7 Oct, 15:00):** `results/` is now `prices/`, `db/`, `csv/`, `logs/` (`hpc` links
  there), `figures/`, `learning/`, `validation/` (`results/README.md`, paths from `assume_gb/paths.py`).
  Paths elsewhere in this file such as `results/<run>.db` or `results/hpc/` are the old flat layout.
  Jobs submitted before 15:00 on 7 Oct (arrays 4286608–10, 4287861–2) still write the old layout;
  `python -m assume_gb tidy --apply` moves such files once their jobs have ended (not before: it
  could move a database a job is still copying). The background validation moves each arm's own
  files when its array has finished, then validates it.
- **Copy single files to the HPC, not whole scenario folders.** On 6 Oct a copy of the laptop's
  `inputs/gb_*` replaced the HPC's: it deleted the policies trained on the HPC (both campaigns,
  18 runs; their databases, prices, logs and figures in `results/` survive), the campaign cases
  in `config.yaml` and the levy units file (`powerplant_units_learning_egl.csv`: rewrite it with
  `egl.unit_parameters`, section 5). Copy `unit_owners.csv` and the like one by one, or exclude
  `learned_strategies`, `config.yaml` and the HPC's own files (`*_egl.csv`, `*_own.csv`,
  `*_fringe.csv`).
- 7 Oct afternoon: a copy of the laptop's `assume_gb` replaced `config.yaml` (all winter cases,
  and the year companions the running jobs needed), `hpc/train.pbs` (back to the old activation,
  without `NON_INTERACTIVE` and `_year_last`) and `hpc/make_cases.sh`. Restored on the HPC: the
  cases (from the recorded settings; the base cases were unchanged), `train.pbs`, and
  `_year_last` in the laptop's new `make_cases.sh` (which keeps its `SOURCE` option). The laptop
  copies of these two scripts are the old ones: take the HPC's back.
- Files written on the laptop end lines with CRLF, files written on the HPC with LF; the readers
  do not mind, a byte comparison does.
- On the cluster `tests/test_utils.py::test_broken_timestamps` fails (533 of 534 pass): the nodes
  run on Europe/London, which in 1970 was on British Standard Time (UTC+1 all year), and the test
  takes the 2020 offset for the 1970 one. Not caused by this fork.
- A batch job cannot answer the framework's prompt about an existing policy folder: the job
  scripts set `NON_INTERACTIVE=1`, which deletes the folder and trains afresh. So never submit a
  case again while it runs: the second job deletes the first one's policies.
- `qstat -u $USER` shows an array as one line in state `B` ("begun": its sub-jobs have started),
  not a waiting or blocked job; `qstat -t -u $USER` shows the sub-jobs (`R` running, `Q` queued).
- An exception inside a bidding strategy during a run is logged by mango ("got exception in
  scheduled event") and the simulation then waits forever for the missing bids: a hung run with
  no CPU use means an exception, look in the log.
- The framework's CSV writer removes and recreates its folder; under OneDrive this can fail with
  WinError 183 (a sync race). Keep heavy output outside synced folders.
- Two runs of the same code differ unless `PYTHONHASHSEED` is fixed (order of agents, random
  tie-breaking among equal-price offers): prices are the same, dispatch of tied units is not.
- A missing value in a price forecast makes a pyomo/HiGHS solve never return.
- Money in the framework is price x MW per time step: at half-hourly steps profits and cash
  flows are hourly rates, halve them for GBP; the figures do this, `rl_params` does not.
- Learning costs: a gradient step costs about 45 ms per plant-level agent on a CPU; ten
  plant-level agents, two weeks, 50 episodes took 56 min; the portfolio agents' observations loop
  over every unit and time step once per episode (a 188-unit week took 7 min for four episodes).

## 7. Moving to the HPC

State on 6 Oct: set up on Imperial RCS in `~/assume` (env `~/miniforge3/envs/assume`, CPU
torch; `assume_gb` in `.git/info/exclude`). The smoke test and the nine-case array ran (array
4275841, all nine finished); the year runs with the last policies ran as array 4280070
(`hpc/evaluate.pbs`, `hpc/cases_last.txt`). Job scripts activate the env with `cd $PBS_O_WORKDIR`,
`source ~/miniforge3/etc/profile.d/conda.sh`, `conda activate assume` (the user's convention), and
set `NON_INTERACTIVE=1`: otherwise the framework asks whether to overwrite an existing policy
folder and the job fails on the prompt. A run uses one core and under 1 GB (it requests 4 cores
and 16 GB: ample).

Added on 6 Oct: `hpc/evaluate.pbs` (a year with trained policies, no training; array over
`hpc/cases_last.txt`), `hpc/tests.pbs` (ruff and the framework's tests on a node), the
`_year_last` companions in `make_cases.sh` and `train.pbs`, and `hpc/cases_sep4.txt` (the running
campaign). The framework change of 6 Oct (cost bins by capacity: `assume/strategies/
portfolio_learning_strategies.py`, `tests/test_portfolio_learning_strategy.py`,
`docs/source/unit_operator.rst`, `docs/source/release_notes.rst`) exists only in the cluster's
working tree: commit and push it from there, or copy it to the laptop, before the two diverge.
`assume_gb/dev/campaign.py` needs a minute per call on the login node (slow imports), as does
every `variant`; writing many cases is faster in one Python process (`assume_gb.__main__.variant`
in a loop, as was done for `cases_novdec.txt`).

Ownership on the HPC (6 Oct, evening): the laptop's ownership work is complete here
(`ownership.py`, `ownership.md`, `ownership_data/` with the verified joint-venture shares,
`owners.py`, `egl.py`, `__main__.py`, `convert.py`, `test_ownership.py`: 10 passed, 1 skipped,
the register test that needs the GB repository), and every folder's `unit_owners.csv` holds the
verified shares. With the new code `dev/learning_sets.py` writes the same `own` and `fringe` files
byte for byte and still reproduces the 5 Oct files (`--check`), so the inputs need no update. A
laptop rebuild of gb_2023 would change only the default learning files (to the `own` set) and
rewrite `config.yaml`: copy `unit_owners.csv` alone when owners change, then rerun
`dev/learning_sets.py`, and only between campaigns. The second framework change of 6 Oct (a
portfolio may run fewer units than `nbins` with `cost_bins: capacity`) is in the same uncommitted
working tree.

**How the pieces get there.** The framework changes are committed and pushed to the fork
(`lukasschirren/assume`, `main`), so the cluster clones the fork. `assume_gb/` is never pushed and
is copied into the clone by hand. Nothing reads the GB data repository at run time (tested with
the repository unreachable). The cluster is PBS Pro (Imperial RCS). A run is single-threaded
Python with a few torch threads: the cluster is used by running many cases at once, and a GPU
does not help.

**What is in `assume_gb/hpc/`.**

- `train.pbs`: one study case per job or array sub-job. It trains the case, draws its learning
  figures and, if the case has a `<case>_year` companion, runs the year with the trained
  policies. It works on the node's own disk (`$TMPDIR`) and copies everything back when the job
  ends, also when it fails. Resources per run: 4 cores, 16 GB, 24 h.
- `make_cases.sh`: writes the study cases of a campaign into the scenario's `config.yaml` and
  lists them in `cases.txt`, one per line. Each case is `learning_<window>_s<seed>` with its
  `_year` companion. Run it only while no jobs are running, because it rewrites `config.yaml`.
- `cases.txt`: the nine cases already written for 2023: the first two weeks of January, March
  and September, seeds 0 to 2, 50 episodes each. `gb_2023/config.yaml` also holds
  `learning_week_sep`, a four-episode smoke test in September.

**What was tested and what was not.** The job script was dry-run on the laptop with
`learning_week_sep`: 320 s, a mid-year training window works, and the database, logs, CSV output
and figures were copied back (also after a failed run). PBS itself and the conda activation lines
in `train.pbs` could not be tested; those lines are the part to check on the cluster.

**Set up, once.**

```bash
# on the cluster
git clone https://github.com/lukasschirren/assume.git && cd assume

# on the laptop, from the repository root (Git Bash): assume_gb without results and old policies
tar --exclude=assume_gb/results --exclude=learned_strategies --exclude=__pycache__ -czf assume_gb.tgz assume_gb
scp assume_gb.tgz <user>@<login node>:assume/

# on the cluster, in assume/
tar -xzf assume_gb.tgz
module load miniforge/3 && miniforge-setup
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda create -n assume python=3.12 pip && conda activate assume
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[all]" openpyxl
mkdir -p assume_gb/results/logs
```

The archive is about 50 MB and includes `observed_prices.csv` (licensed N2EX and EPEX data). Add
`--exclude=observed_prices.csv` if it must not go on the cluster; a run then compares with the
merit-order model only.

**Submit**, from the repository root on the login node.

```bash
qsub -v CASE=learning_week assume_gb/hpc/train.pbs     # smoke test first: four episodes, about 10 min
qsub -J 1-9 assume_gb/hpc/train.pbs                    # then the nine cases, sub-job N = line N of cases.txt
qstat -t -u $USER                                      # status; qdel <job id> removes a job
```

Change the resources in the file or with `qsub -l walltime=48:00:00`. A 50-episode run of two
weeks is expected to take several hours and the year afterwards up to an hour; neither has been
timed on the cluster. `qsub -v RESULTS=<folder>` sends the results elsewhere, `YEAR_CSV=1` keeps
the year's full CSV output (about 600 MB a run).

**Where the results are**, under `assume_gb/results/` (a run is `gb_<year>_<case>`; the whole
layout in `results/README.md`, reorganised on 7 Oct 2026: `python -m assume_gb tidy --apply` moves
what jobs submitted before still write in the old flat layout):

| File | Content |
| --- | --- |
| `logs/<run>.log` | the run's output, written live |
| `logs/<run>.assume.log` | the framework's log: look here if a run hangs with no CPU use |
| `db/<run>.db` | the learning database (`rl_params`, `rl_meta`, `rl_grad_params`) |
| `learning/<run>/` | the learning figures of that run |
| `prices/<run>.csv`, `prices/<run>_year.csv` | prices of the training window and of the year with the trained policies |
| `csv/<run>/` | the framework's CSV output of the final evaluation |

The policies are in `assume_gb/inputs/gb_<year>/learned_strategies/<run>/`.

**Pool the seeds of a window** in one set of figures with intervals, on the cluster or after
copying `results` back:

```bash
R=assume_gb/results
python -m assume_gb learning --out $R/learning/gb_2023_learning_sep \
    --db sqlite:///$R/db/gb_2023_learning_sep_s0.db sqlite:///$R/db/gb_2023_learning_sep_s1.db sqlite:///$R/db/gb_2023_learning_sep_s2.db \
    --simulation gb_2023_learning_sep_s0 gb_2023_learning_sep_s1 gb_2023_learning_sep_s2
```

**Further campaigns.** `make_cases.sh` takes `WINDOWS` (label:first delivery day:day after the
last), `SEEDS`, `EPISODES`, `EXTRA` (further settings for every case), `TAG` (a suffix for the
names) and `APPEND=1` (add to `cases.txt`). A sweep point on a four-week September window:

```bash
WINDOWS="sep:2023-09-01:2023-09-29" EXTRA="bidding_strategy_params.max_markup=3" TAG=m3 APPEND=1 bash assume_gb/hpc/make_cases.sh
```

**Read the first array before sweeping.** The nine runs are the first full trainings of the
portfolio set-up; until now it has only had four-episode smoke tests (reward 0.001 to 0.002, at
the competitive benchmark). Check that the reward rises above zero, which means above competitive
profit, and that the mark-ups do not sit at the cap (`max_markup` 2). Then compare the windows on
the year (`prices/<run>_year.csv` against N2EX) and choose the window (section 4.2).
