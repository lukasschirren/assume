<!--
SPDX-FileCopyrightText: ASSUME Developers

SPDX-License-Identifier: MIT
-->

# Validating the GB day-ahead model

How the agent-based model (ABM) of the GB day-ahead market is validated against observed prices
in this repository, and why. The reasoning follows the synthesis note *How should I validate an
agent-based electricity price model?* (7 Oct 2026); keys in brackets are its citation keys. The
code is the package beside this file.

## The reasoning in brief

Agent-based electricity market models are rarely validated against data, and when they are it
is done in heterogeneous ways [weidlichCriticalSurveyAgentbased2008]. The recent AMIRIS and ASSUME
back-tests report MAE, RMSE and correlation only [nitschBacktestingAgentbasedModel2021]
[maurerKnowYourTools2024]. Price forecasting offers many more metrics, but most of them restate
each other. The scorecard therefore has eight criteria, each answering a different question:

| # | Question | Criterion | Module |
| --- | --- | --- | --- |
| 1 | How large are the errors? | MAE as headline, RMSE for comparison with the AMIRIS and ASSUME back-tests, rMAE (MAE over that of the naive forecast) across years [lagoForecastingDayaheadElectricity2021] | `metrics` |
| 2 | Why are they large? | bias, ratio of standard deviations, Pearson r and the MSE decomposition; hour-of-day and weekday profiles, benchmarked by how well one observed year's profile matches the next [casarCanShadowPrices] | `metrics` |
| 3 | Is the distribution right? | Wasserstein-1, share of negative prices, errors at the 1st, 5th, 95th and 99th percentiles, mean above the 95th [nitschBacktestingAgentbasedModel2021] [bordignonCombiningDayaheadForecasts2013] | `metrics` |
| 4 | Is the model better than simpler ones? | one-sided Diebold-Mariano tests, multivariate (daily loss) and per period of the day [lagoForecastingDayaheadElectricity2021] [fraunholzAdvancedPriceForecasting2021] | `stats` |
| 5 | Are the mechanisms right? | one reduced-form regression run on observed and simulated prices alike; wind and solar effects by penetration against causal estimates [cacciarelliWeActuallyUnderstand2025] | `mechanisms`, `reference` |
| 6 | Is market power plausible? | markups over the competitive run by scarcity, simulated against observed [borensteinMeasuringMarketInefficiencies2002] [twomeyMonitoringMarketPower] | `mechanisms` |
| 7 | Does the error matter for decisions? | capture-price error of wind and solar; storage arbitrage revenue [nitkaCombiningPredictiveDistributions2023] | `value` |
| 8 | Is the seed spread informative? | CRPS and PIT across seeds [pinsonNonparametricProbabilisticForecasts2007] [nowotarskiRecentAdvancesElectricity2018] | `metrics` |

Left out, because they add nothing to the eight:

- MAPE and sMAPE: they are meaningless for prices near or below zero.
- MASE: rMAE does the same job without depending on the calibration window.
- R²: it equals 1 - MSE / Var(observed), so it moves one for one with RMSE.
- Pinball loss: CRPS integrates it.
- KS, t and Kruskal-Wallis tests on hourly prices: with thousands of autocorrelated hours, almost any difference is significant.
- The Giacomini-White test: the unconditional DM test answers the question.
- HHI and residual supply index: they describe the inputs, not the output. Scarcity becomes the variable markups are binned by instead.

Four identities explain the overlaps:

- **MSE decomposition:** MSE = (mu_s - mu_o)^2 + (sigma_s - r sigma_o)^2 + sigma_o^2 (1 - r^2), so bias, sigma ratio and r together determine RMSE.
- **W1:** with equal sample sizes, Wasserstein-1 is the MAE between the two sorted series, i.e. between the price duration curves.
- **CRPS:** for a single run, the CRPS is the absolute error, so CRPS across seeds is on the scale of the MAE.
- **rMAE:** rMAE ranks models as MAE does within one test period.

## Where the code lives, and why

- **Not in `tests/`.** That folder is the framework's software test suite: it is pushed, it is pytest's `testpaths`, and it must run without data. Validation asks a different question (does the model match the world, not is the code right). It needs licensed prices and run outputs, and it produces thesis artefacts.
- **Not in `assume/`.** It is analysis, not model logic. Its adapters read the local GB folders, which pushed code must not refer to.
- **In `assume_gb/`.** The local tool already holds the analysis of runs (`compare`, `results`, `figures`, `egl`) next to the scenario data, and it is never pushed.
- **One boundary keeps the package portable.** Every module except `adapters` takes pandas objects (the data contract) and reads no file but the package's own reference data (`data/reference/`). `adapters` is the only module that knows the folders of `assume_gb`.
- **Tests.** The package's own tests run on synthetic data: `python -m pytest assume_gb/validation`, which also runs as part of `python -m pytest assume_gb`.

## How to validate in this repository

### The series

| Role | What it is | Where it comes from |
| --- | --- | --- |
| `observed` | N2EX hourly day-ahead price by default; the EPEX half-hourly auction (from 2020) as an option | `assume_gb/inputs/gb_<year>/observed_prices.csv` (licensed, git-ignored). Periods the merit-order model flags as inadmissible (`admissible` in `reference_prices.csv`, e.g. corrupt demand in 2018 and 2020) are left out |
| `competitive` | the `day_ahead` case: every unit bids its marginal cost net of its support payment. It equals the merit order of the scenario's offers to 1e-13, and thereby the merit-order model's price (except the any-hour rule of 2025) | `assume_gb/results/prices/gb_<year>_day_ahead.csv` where saved, otherwise `scenario_merit_order` in `reference_prices.csv` |
| `abm` | the delivery year run with the trained policies of a learning arm, one column per seed | `assume_gb/results/prices/gb_<year>_<arm>_s<k>_year.csv` (policies of the best evaluation) or `..._year_last.csv` (at the end of training) |
| `naive` | the observed price of the same local delivery period one week earlier (the same wall-clock time across a clock change) | built from `observed`. Lago et al. use the day before on Tuesday to Friday, so rMAE values are not directly comparable with theirs |
| `lear` | the LEAR benchmark [lagoForecastingDayaheadElectricity2021] | none yet; the contract takes it when there is one |

The competitive run does three jobs:

- It is the accuracy benchmark Harder et al. use [harderFitPurposeModeling2023].
- It is the counterfactual for the markups of criterion 6.
- Subtracted from the observed price, it gives a Borenstein-style estimate of the observed markup [borensteinMeasuringMarketInefficiencies2002].

It has the same fleet as the learning cases: neither has storage.

Prices are compared from full-precision files, never from the framework's `market_meta.csv`, which keeps five significant digits. The simulation is half-hourly and N2EX is hourly, so models are averaged to the observed hours. An hour with a missing half-hour is left out, as `assume_gb.compare.hourly` does.

### Windows: tuned and untuned data

- **`headline`:** data never used for tuning anything. Every headline number comes from it, and so do the Diebold-Mariano tests and every comparison with the naive forecast and LEAR. For a learning arm it is the delivery year without the arm's training window. The contract always takes the calibration window out of it.
- **`calibration`:** the data used for tuning: the training window of the agents, i.e. `start_date` to `end_date` of the case `learning_<arm>_s0` in `config.yaml` without its lead-in day. It is scored separately.
- **`year`:** the full delivery year, reported as a secondary result and labelled in-sample when it overlaps the calibration window (`ValidationData.in_sample`).

Why this split:
- Lago et al. ask for a test period of at least a year, with the calibration window reported [lagoForecastingDayaheadElectricity2021].
- Tracking a time series separates in-sample calibration from out-of-sample testing [pangalloDataDrivenEconomicAgentBased2024].
- The headline window loses four of fifty-two weeks but keeps every season.

The reinforcement learning never sees an observed price: the agents learn from simulated profit against the competitive benchmark. Choices made after comparing with N2EX are tuning all the same, and belong in the reported set-up: which arm, which training window, any merit-order parameter set on these years.

### Information sets

The day-ahead market cleared on forecasts, while the ABM runs on the outturn of wind, solar and demand. The explanatory data therefore carry both:
- the outturn: `wind`, `solar`, `demand`;
- the 09:00 D-1 forecast vintage: `wind_forecast`, `solar_forecast`, `demand_forecast`.

By default each price is regressed on its own information set: the observed price on the forecasts, a simulated price on what drove the model (`ValidationData.information`). Regressing both on the outturn is the robustness variant.

`mechanisms.attenuation` reports two factors, computed after partialling out the controls:
- **Variance ratio, Var(F~)/Var(O~):** by this factor the observed coefficients shrink in the robustness variant, if the forecast error is uncorrelated with the forecast.
- **Cov(F~, O~)/Var(O~):** the same factor without that assumption.

The reverse case is not attenuated: a price formed on the outturn, regressed on an efficient forecast, keeps its coefficient.

### Mechanisms and the causal reference

- **The regression:** `price ~ wind + solar + demand + gas + carbon + C(hour) + C(dow) + C(month)`, with Newey-West errors over one day.
  - Wind, solar and demand coefficients are in GBP/MWh per GW, which is per GWh in an hour. Gas is per GBP/MWh thermal, carbon per GBP/t.
  - The comparison is term by term over the window and per regime: observed minus simulated estimate, and whether the 95% intervals overlap.
  - A regression on observed data captures correlation, not the causal effect. The same regression on both sides checks model against data; the causal estimates say how large the effect is [cacciarelliWeActuallyUnderstand2025].
- **The reference:** the conditional average treatment effects of Cacciarelli et al. with their 80% band, in GBP/MWh per +1 GW of day-ahead forecast output. They are in `data/reference/cacciarelli_cate_digitised.csv`, traced from Figs. 4 (APX) and 5 (Nord Pool) of the published paper, because the authors' results files are not in their repository; the README there has the provenance.
  - Use the curve of the market of the observed series: N2EX goes with Nord Pool, the EPEX auction with APX.
  - Interpolate linearly; do not extrapolate.
  - `mechanisms.compare_binned` estimates one slope per bin of predicted penetration (forecast / load x 100), for observed and simulated prices on the same bins, and sets the curve beside each bin.
- **Penetration on the authors' axis.** Their wind forecast is transmission-connected wind and their load transmission system demand. The scenarios carry wind including embedded wind and demand net of nuclear and pumped storage: the 2023 mean wind penetration is 36.8% on those against 25.5% on theirs.
  - The bins therefore use the reference definition (`mechanisms.penetration`): transmission-connected wind (`wind_tx_forecast`, Elexon WINDFOR at 09:00 D-1) over the transmission system demand forecast (`tsd_forecast`).
  - The two series come from `penetration_forecasts.csv` in each scenario folder. `python -m assume_gb penetration --years 2018 2025` writes it, and only it, where the GB data repository is; copy the file alone to other machines.
  - Where the forecast state has no transmission system demand forecast (`tsd_fc_mw`), the national demand forecast stands in as `nd_forecast`. It leaves out station load, pumping and exports, so penetration comes out somewhat higher; the `load` column of `compare_binned` names the load used.
  - `definition="scenario"` bins on the scenario's own forecasts, which is not the reference's axis.
- **The solar reference is indicative.** The authors' solar forecast barely varies within a day (0.85 GW at 04:00, 1.37 GW at 14:00 on average), so their solar axis is not the half-hourly solar share. `compare_binned` marks solar rows `indicative`.

### Markups

- The simulated markup is `abm - competitive`; the observed markup is `observed - competitive`. Both are binned by a scarcity measure and compared by distribution and Wasserstein-1 per bin.
- **Default measure:** residual demand over available capacity. Thermal availability is constant in the scenarios, so this is close to rescaled residual demand. Pass a better measure when there is one.
- **Contract positions are not modelled.** A markup shows the outcome, not the incentive behind it, since hedging changes whether a generator wants prices up or down [wolakEmpiricalAnalysisImpact2000a] [xuMarketPowerAbuse2025].

### Seeds

| Criterion | What it uses |
| --- | --- |
| Point metrics | the mean of the seeds (or the median) |
| Distribution checks (criteria 3 and 6) | all seeds pooled |
| CRPS and PIT | each seed as an ensemble member |

- Averaging seeds smooths the price and flatters MAE and RMSE, so the scorecard also reports the median and range of the single-seed values.
- A seed ensemble covers behavioural randomness, not input uncertainty, so expect the PIT histogram to show too little spread.

### Conventions

- **Units:** prices in GBP/MWh, power in GW.
- **Time:** local (Europe/London). A clock-change day keeps its 23 or 25 hours: the daily loss of the multivariate DM test is the mean over the local delivery day, and the per-period test follows the wall clock.
- **Moments:** population moments (divisor n), so that the MSE decomposition adds up exactly.
- **Common sample:** every model is scored on the same periods, those where the observed price and all models have a value.
- **Money:** the framework's money is price x MW per step, an hourly rate at half-hourly steps. The value criterion works in GBP with the length of the period.
- **Storage decision:** the battery arbitrages each complete local day with perfect foresight of a price (a linear programme with charge plus discharge at most the power, the usual stand-in for "not both at once"), back to its starting charge at midnight. "Decision value" is what the schedules optimised on a model's price earn at the observed price, over the observed perfect-foresight revenue: 1 at best, negative when trusting the model loses money.

## Running it

From the repository root, with the `assume` environment:

```bash
python -m assume_gb validate --year 2023 --arm learning_novdec_own_e100           # one year, the arm's seeds as "abm"
python -m assume_gb validate --year 2023 --arm learning_novdec_own_e100 --policies last
python -m assume_gb validate --years 2018 2025 --regime crisis=2021-07-01:2022-12-31   # the competitive run alone
python -m assume_gb validate --year 2023 --venue epex_hh_day_ahead --run storage=day_ahead_storage
python -m assume_gb penetration --years 2018 2025    # where the GB data repository is: the penetration axis
python -m pytest assume_gb/validation                # the package's tests, on synthetic data
```

`validate` writes into `assume_gb/results/validation/<scenario>_<arm>/`, with `_last` for `--policies last` and `<first year>_<last year>_competitive` for `--years` (`--out` elsewhere). It takes 30 s for a year with ten seeds. The folder holds licensed prices in its tables and figures: keep it local. `assume_gb/results/README.md` describes the whole results folder and how its runs are named.

| File | Content |
| --- | --- |
| `scorecard_<window>.csv`, `.tex`, `_seeds.csv` | the scorecard of the headline, calibration and year windows: one row per metric and series with direction, skill and stars; the booktabs table; the single seeds' spread |
| `fig1_weeks` ... `fig6_scorecard` | the core figures: weeks by rule, duration curves, error by hour and month, coefficients, markups by scarcity, the scorecard (PDF and PNG, 300 dpi) |
| `figS1_scatter` ... `figS5_price_vs_residual_demand` | the supplementary figures: density scatter, Taylor diagram, profiles, PIT and monthly CRPS, price against residual demand |
| `weeks.csv` | the weeks of `fig1_weeks`, each with the rule and the sentence that chose it, for the caption |
| `regressions.csv`, `regressions_outturn.csv`, `attenuation.csv`, `binned_wind.csv`, `binned_solar.csv` | criterion 5: each price on its own information set, all on the outturn, the attenuation factors, the binned effects beside the causal curve |
| `markups.csv`, `capture_prices.csv`, `storage_daily.csv`, `storage.csv` | criteria 6 and 7 |
| `notes.txt` | what was left out, and why (for example the binned effects without `penetration_forecasts.csv`) |

In Python, the same pieces on any data in the contract's form:

```python
from assume_gb.validation import adapters, mechanisms, metrics, report, stats, value
from assume_gb.validation.contract import ValidationData

# from the assume_gb folders, or ValidationData(observed, models, exog, periods=...) from anything
data = adapters.load(2023, "learning_novdec_own_e100")
sample = data.sample()  # headline window, common sample
metrics.mae(sample["abm"], sample["observed"])
stats.dm_matrix(sample)  # row A, column B: p-value of "B beats A"
mechanisms.compare_regressions(data)  # each price on its own information set
mechanisms.attenuation(data)
mechanisms.markups(data)
value.storage_summary(value.storage_revenue(sample, adapters.battery(2023)))
report.write(data, "out", market=adapters.market("n2ex_day_ahead"), battery=adapters.battery(2023))
```

How to read and report the results:

1. **Quote the headline window.** It holds no data used for tuning. `scorecard_calibration` and `scorecard_year` are secondary, and their `in_sample` column says when they overlap the training.
2. **Read `notes.txt` first.** An empty cell is a piece that was left out, for example the binned effects without `penetration_forecasts.csv`. It is not a zero.
3. **Read each row in its own direction.** The `direction` column says which way is better: lower, higher, closer to 1, closer to 0, or closer to the observed value in the `observed` column. `skill` compares a model with the naive forecast only where that is meaningful (accuracy, correlation, capture prices, the storage decision, CRPS). Stars mark a significant DM test of beating the naive forecast.
4. **For an ensemble, check the single seeds.** The scorecard scores the seed mean, which smooths noise and looks better than any one seed. `scorecard_headline_seeds.csv` gives the single seeds' median and range beside it.
5. **For captions, use `weeks.csv`.** It gives the rule and the sentence behind each week of `fig1_weeks`. The causal reference of the forest plot is the estimate over the penetration the window holds.

Reading the observed column of the profile rows: it is the year-to-year stability of the observed profiles, the year before against the test year. Across a change of regime it can be far from 1 (2022 against 2023: amplitude ratios 0.45 by hour and 0.37 by weekday, the crisis year's spreads), so it is the benchmark only where consecutive years are alike.

## Status

All ten steps are built and tested (`python -m pytest assume_gb/validation`: 61 tests on synthetic data, plus the command's test in `assume_gb/test_validate.py`):

| Step | Module |
| --- | --- |
| 1 | `contract`: data contract, windows, information sets, local calendar |
| 2 | `metrics`: criteria 1, 2, 3 and 8 |
| 3 | `stats`: Diebold-Mariano tests |
| 4 | `mechanisms`, `reference`: criteria 5 and 6 |
| 5 | `value`: criterion 7 |
| 6 | `scorecard`: criteria x models, skill against naive, CSV and LaTeX |
| 7, 8 | `style`, `figures`: core and supplementary figures |
| 9 | `adapters`: the `assume_gb` folders into the data contract |
| 10 | `report` and `python -m assume_gb validate` |

## Open points

- **Transmission system demand forecast:** the forecast state of the GB data repository has to carry a `tsd_fc_mw` column (TSDF at 09:00 D-1, built like `demand_fc_mw`) for the penetration to divide by the authors' load; until then the national demand forecast stands in.
- **LEAR benchmark:** none yet.
- **Penetration files:** `penetration_forecasts.csv` is not in any scenario folder yet: run `python -m assume_gb penetration --years 2018 2025` where the GB data repository is and copy the files alone. Until then the binned effects and the wind row of criterion 5 are left out (`notes.txt`).
