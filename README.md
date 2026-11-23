# Public stress-testing methods, side by side

A small, standalone project that forecasts bank balances, rates and net interest income under macro scenarios
with **methods published by the IMF, the Federal Reserve and the New York Fed**, and compares them
out-of-sample against simple benchmarks. It is independent of the `macroforecast` project and does not change it;
it reads the same kind of files (macro scenarios, a metadata file with expected signs, bank data).

```
pip install -r requirements.txt
python run.py            # full test, about 2 minutes on the example data
python run.py --quick    # origins every 4th quarter, about 30 seconds
python run.py --targets loans_panel nii
python run.py --mode separate --banks "Bank A" "Bank C"     # each bank on its own, two banks
python run.py --metric cvar                                 # rank by the average of the worst 10% of errors
```

Open `outputs/report.html`.

## The methods

| Method | Source | What it does |
|---|---|---|
| **IMF BMA** | IMF Global Solvency Stress Test (GFSR Oct 2020, Online Annex 4.1) | Panel regressions with bank fixed effects and a lagged dependent variable. Every combination of 0–3 screened drivers is estimated; equations whose long-run multipliers b/(1−ρ) have the wrong sign are dropped; the rest are **averaged** with weights exp(−BIC/2). No single model is chosen. |
| **Forecast combination** | Guerrieri & Welch (2012), Fed FEDS 2012-49 | One equation per macro variable (own lag + lags 1–4 of the variable), equal-weighted. |
| **CLASS-style** | Hirtle, Kovner, Vickery & Bhanot (2016), NY Fed CLASS model | One small fixed equation per target with drivers chosen on economic grounds (`class_drivers`), the same at every origin. A transparent challenger. |
| **Rate × balance** | Federal Reserve supervisory PPNR models | For income, each method forecasts the parts (rates and balances); the income is rebuilt from them instead of being forecast in dollars. |
| **Benchmarks** | | AR(1) with bank effects, drift (average growth), random walk (no change). |
| **Conformal bands** | | 80% / 95% bands from each method's own out-of-sample errors at each horizon, with a real-time coverage check. |

## Your choices

All in `config.yaml`; `restrictions`, `estimation`, `ranking` and `methods` can also be set under one target to
apply to that target only.

**How drivers may enter (`restrictions`)**

| Option | Effect |
|---|---|
| `signs: false` | No sign restrictions at all. |
| `apply_to: [bma, combination, class]` | Which methods impose the expected signs. BMA drops wrong-signed equations; the combination drops equations whose summed lag coefficients have the wrong sign; the CLASS-style equation is fixed, so a wrong sign is flagged in the report. |
| `free_sign: [VAR]` | Keep a driver, without a sign restriction. |
| `include_unsigned: true` | Series with no expected sign may also enter, with a free sign. |
| `exclude: [VAR]` | Never use a driver. |
| `force_in: [VAR@lag]` | A driver in every BMA (and combination) equation. If the data give it the wrong sign everywhere, its sign check is dropped and the report warns. |
| `not_together: [[A, B]]` | An equation may hold at most one of A and B (e.g. the same series for two regions). |
| `max_corr: 0.85` | Two screened drivers more correlated than this are not both kept; `null` for no limit. |

**Panel or bank by bank (`estimation`)**

* `mode: panel`: banks pooled, with bank fixed effects (one set of driver effects for all).
* `mode: separate`: each bank on its own: its own screening, BMA, scores and report section.
* `banks: all` or a list of names, e.g. `[Bank A, Bank C]`. In panel mode only those banks are pooled.
  A rate x balance target runs its parts with the same mode and banks.

**What a method is compared with (`ranking.compare_against`, `methods.run`)**

* `methods.run`: which methods to estimate.
* `compare_against`: the benchmarks shown as "vs ..." columns and in "beats ... (share of origins)". Any method can
  be a benchmark, e.g. `[class]` or `["bma (rate x balance)"]`.

**How methods are ranked (`ranking.metric`)**

The average error rewards a method that is right on average even if it sometimes deviates a lot. To penalise
those large deviations, rank by one of the tail metrics:

| Metric | Meaning |
|---|---|
| `mean` | Average error (for balances, % of the actual level: a MAPE). |
| `rmse` | Root mean squared error: squaring makes large deviations weigh more. |
| `p90` | The error exceeded in 10% of cases. |
| `cvar` | Average of the worst `tail_share` (default 10%) of errors: how bad the bad cases are. |
| `worst` | The single largest error. |
| `combined` | mean + `penalty` × (cvar − mean): the average, plus a charge for how much worse the bad cases are. `penalty: 0` is the plain average, `1` the tail only. |

Every metric is computed at each horizon over all origins and banks and then averaged over horizons 1–12, so the
naturally larger long-horizon errors don't dominate (except `worst`, the largest error at any horizon). The
leaderboard always shows all of them, whichever is used for the ranking.

## How the test works

* **Rolling origin.** From `oos.start` on, at every origin each method is re-estimated (the BMA and the combination
  also re-screen their drivers) on data up to that origin only, then forecasts 1–12 quarters ahead.
* **Conditional on the macro that happened**, the usual way to validate scenario models.
* **Errors on levels**: percent of the actual level for balances and income, units (percentage points) for rates.
  Acquisitions in the events file are removed from growth (organic change), as the Fed does for mergers.
* **Same rows for every equation** at an origin, so BIC values (and BMA weights) are comparable.
* **Coverage** of the bands is checked in real time: at each origin the band is built only from errors already
  known at that origin.

## Inputs (config.yaml)

* `macro.scenarios`: one CSV per scenario, `date` + one column per series, history then the scenario path. The last
  quarter where all scenarios agree is taken as the end of history.
* `macro.metadata`: `mnemonic, description, geography, theme, transform (dlog/diff), expected_sign`.
* `targets[]`: a CSV with `date`, an optional bank column, and either a `column` or an `expression` over columns
  (e.g. net interest income). `form: dlog` for balances and income, `diff` for rates. `signs:` is the metadata-style
  file whose `expected_sign` says which drivers may enter and with which sign. `events:` (optional) lists
  acquisitions to take out of growth.
* `rate_x_balance.parts`: the targets whose forecasts are combined with the target's `expression`.
* `candidates`: which geographies and themes may supply drivers, the lags allowed, and how many drivers are screened.

## Outputs (outputs/)

| File | What |
|---|---|
| `report.html` | Overview, the methods, and per target: leaderboard, error by horizon, scenarios with bands, every method under the severe scenario, BMA inclusion probabilities and heaviest equations, the CLASS equation, a scenario range check. |
| `leaderboards.csv` | Every method and target (or target and bank): the ranking score, average error, RMSE, 90th percentile, tail average, largest error, combined score, h1/h4/h8/h12, vs each benchmark, share of origins beating the first benchmark, band coverage. |
| `errors_<target>.csv` | Every out-of-sample error (method, origin, bank, horizon). |
| `scenarios_<target>.csv` | Scenario forecasts by method, scenario, bank (and total), with 80% / 95% bands. |
| `bma_inclusion.csv` | Inclusion probability of each screened driver in the final BMA. |
| `scenario_range_check.csv` | BMA drivers whose scenario values leave the range seen in history. |

## What the example run shows

On the simulated example data (the same data as the main project):

* The **IMF BMA is the most accurate method for 6 of the 8 balance and rate targets**, and second for the other 2
  (mortgage and deposit balances), where the fixed CLASS-style equation wins. For net interest income the best method
  is the BMA applied rate × balance. It recovers the drivers the loan data were simulated with
  (house prices and disposable income, a quarter later) with inclusion probabilities close to 1.
* The **Fed's rate × balance approach cuts the error of net interest income by about a third** for the BMA
  (17.7% vs 27.7% of the level), because NII is a small difference of two large products and is far noisier
  than its parts.
* Every macro model beats the random walk and AR(1) clearly for balances; for rates, AR(1) and drift are worse than
  the random walk, while the BMA beats it by 50–60%.
* **Bands are too narrow in real time** (roughly 50–80% coverage for a nominal 80%), because the early calibration
  years were calmer than 2020 and 2022. The scenario bands use every error, shocks included, and are wider.

These are results on simulated data; on real data run the same comparison before choosing.

## Layout

```
run.py                 the run: fit, test, scenarios, files, report
config.yaml            data, targets, method settings
methods/data.py        macro scenarios, sign files, targets (organic change)
methods/models.py      the equation form, screening, BMA, combination, CLASS-style, benchmarks
methods/evaluate.py    rolling-origin test, rate x balance, conformal bands, coverage, scenarios, range check
methods/report.py      the HTML report
tests/test_smoke.py    a fast end-to-end check (python -m pytest -q)
```

## Sources

* IMF, *Global Financial Stability Report*, October 2020, Chapter 4 Online Annex 4.1 (Global Solvency Stress Test methodology).
* Guerrieri, L. and Welch, M. (2012), *Can Macro Variables Used in Stress Testing Forecast the Performance of Banks?*, Federal Reserve FEDS 2012-49.
* Hirtle, B., Kovner, A., Vickery, J. and Bhanot, M. (2016), *Assessing Financial Stability: The Capital and Loss Assessment under Stress Scenarios (CLASS) Model*, Journal of Banking & Finance 69, S35–S55 (NY Fed Staff Report 663).
* Federal Reserve, *Supervisory Stress Test Model Documentation: Pre-Provision Net Revenue Models* and *Descriptions of Supervisory Models*.
