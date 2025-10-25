# Treasury Scenario Engine (Phase 1A)

Multi-horizon Treasury yield scenario engine combining an AFNS state-space model for the yield curve, a sticky Hidden Markov Model (HMM) for regime persistence, Monte Carlo path simulation with regime transitions, and a multi-horizon validation harness. The pipeline is fully configurable via `config.yaml` and orchestrated through `main.py`.

This README is technical by design, intended for quantitative developers and researchers.

---

## Contents
- Vision and roadmap
- Overview and pipeline
- Configuration (`config.yaml`)
- Data preparation (`src/utils.py`)
- AFNS model (`src/afns.py`)
- Regime model (`src/regime.py`)
- Scenario generation (`src/scenarios.py`)
- Validation (`src/validation.py`)
- Conformal layer (`src/conformal.py`)
- Reporting (`src/reports.py`)
- Orchestrator (`main.py`)
- Outputs and artifacts
- Quick start
- Reproducibility and performance
- Extending the system

---

## Vision and roadmap

Status: Phase 1 complete; moving to Phase 2.

### The Master Plan: Building an Institutional-Grade Quantitative Rate & Spread Forecasting Engine

#### The Vision
I am building the core quantitative engine that will power multiple revenue streams for Baker Group:
1. Reinvestment rate forecasting for bond swap optimization
2. Daily proxy pricing for 1000+ client portfolios
3. Non-parallel scenario analysis for risk management
4. Multi-year spread forecasting for strategic positioning

This is NOT just a bond swap tool — it's the foundational quant infrastructure that differentiates Baker Group from every other broker-dealer who's still using Excel and gut feel.

#### The Architecture I am Building
```text
ReinvestmentRateEngine (Master System)
├── TermStructureModel (AFNS base)
│   ├── Regime-conditional VAR(1) - different dynamics per regime
│   ├── Term-dependent volatility - short rates more volatile
│   └── Long-horizon mean reversion - 5yr forecasts converge to cycle
├── SpreadDynamics (PRIMARY FOCUS)
│   ├── Sector-specific models (Muni/Agency/MBS/CMO)
│   ├── Credit cycle dynamics - spreads lead/lag differently
│   ├── Regime-dependent basis - correlation changes in crisis
│   └── Liquidity premiums - wider in stress, sector-specific
├── RegimeForecaster
│   ├── Persistent regimes (months/years not days)
│   ├── Fed cycle alignment - matches policy cycles
│   └── Credit cycle integration - risk-on/risk-off persistence
└── ScenarioGenerator
    ├── Conditional paths - P(rates | regime path)
    ├── Fed policy scenarios - cuts/hikes/holds
    └── Multi-year horizons - 6M, 1Y, 2Y, 3Y, 5Y
```

#### Implementation Phases

- Phase 1: Fix Foundation & Prove Concept (Immediate)
  - Goal: Get from 15% to 85%+ coverage, extend to multi-year horizons
  - Changes:
    1. Fix covariance bug — use scaled Q matrix, not regime_covs
    2. Extend horizons to [6M, 1Y, 2Y, 3Y, 5Y]
    3. Increase regime persistence (alpha=100+ for multi-month regimes)
    4. Validate at multiple horizons, not just 1-day
  - Deliverable: Working system showing "3-year rates likely 3.5% [2.8%-4.2%]"

- Phase 2: Activate Spread Products
  - Goal: Spreads as first-class citizens, not afterthoughts
  - Changes:
    1. Compute spreads from BVMB/BVCSUG/IGUUAC data
    2. Fit sector-specific AR models with regime dependence
    3. Different dynamics for Munis vs Agencies vs Corporates
    4. Student-t innovations for realistic tail risk
  - Deliverable: "If Treasury 10yr goes to 3.5%, Munis likely at 3.1%, Agencies at 3.7%"

- Phase 3: Regime-Dependent Dynamics
  - Goal: Different market behavior in different regimes
  - Changes:
    1. Separate VAR(1) parameters per regime (A₀, A₁, A₂, A₃)
    2. Regime-specific mean reversion speeds
    3. Crisis regimes have different correlation structures
    4. Calibrate to historical crisis/recovery cycles
  - Deliverable: "In crisis regime, correlations flip and spreads gap 200bps overnight"

- Phase 4: Reinvestment Rate Calculator
  - Goal: The money maker — quantify reinvestment risk
  - Changes:
    1. For any bond: forecast distribution at maturity
    2. Compare: hold-to-maturity vs swap-today scenarios
    3. Calculate break-even reinvestment rates
    4. Probability of finding target yields in future
  - Deliverable: "85% chance your 4% bond reinvests <3.5% in 2028, costing $X vs swapping today"

- Phase 5: Proxy Pricing System (Month 2)
  - Goal: Daily valuations without hand-pricing thousands
  - Changes:
    1. Define ~70 proxy buckets
    2. Map client portfolios to proxies
    3. Daily forecast for each proxy point
    4. Aggregate to portfolio valuations
  - Deliverable: "Client's 800-bond portfolio: $102.4MM today, $98.2MM in rising rate scenario"

#### Why This Approach Wins

- For Bond Swaps:
  - Quantifies reinvestment risk with hard numbers
  - Proves why taking losses today nets positive
  - Shows probability distributions, not point estimates

- For Proxy Pricing:
  - Same engine drives daily valuations
  - Consistent methodology across all products
  - Reduces hand-pricing from thousands to ~70

- For Risk Management:
  - Non-parallel scenarios with regime awareness
  - Sector-specific spread dynamics
  - Multi-year horizons for ALM

- For Baker Group:
  - Differentiates from competitors using gut feel
  - Enables data-driven client conversations
  - Scales to 1000+ portfolios without adding staff

#### The Technical Edge

- State-space models with Kalman filtering (optimal state estimation)
- Regime-switching dynamics (markets behave differently in crisis)
- Sector-specific spreads (Munis ≠ Agencies ≠ Corporates)
- Multi-horizon consistency (6-month and 5-year forecasts align)

---

## Overview and pipeline

At a high level, the system executes the following steps:

1. Load configuration and market/economic data.
2. Fit an AFNS VAR(1) state-space model to extract Level/Slope/Curvature (L/S/C) factors and the transition/covariance parameters.
3. Fit a sticky Gaussian HMM on factors’ daily changes to obtain regime persistence and regime-dependent covariances.
4. Generate multi-horizon Monte Carlo paths with regime transitions and compute distributional summaries (quantiles, mean, std) under multiple Fed shock scenarios.
5. Validate forecasts via historical multi-horizon backtesting using proper state dynamics and regime-scaled volatility.
6. Produce plots and save tables.

Core model equations (qualitative):

- Measurement model (Nelson–Siegel loadings):
  \[ y_t(\tau) = h(\tau;\lambda)^\top f_t + \varepsilon_t, \quad h = [1, \; (1-e^{-\lambda\tau})/(\lambda\tau), \; (1-e^{-\lambda\tau})/(\lambda\tau) - e^{-\lambda\tau}] \]
- State dynamics (VAR(1) with drift):
  \[ f_{t+1} = A f_t + b + \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q) \]
- Regime transitions: first-order Markov chain with sticky prior (diagonal-boosted transition matrix) and model selection by BIC.

---

## Configuration (`config.yaml`)

All knobs live in `config.yaml`:

- `data`
  - `raw_path`: Parquet with yields and market features (e.g., MOVE/VIX/OIS/Fed).
  - `econ_reports`, `econ_jobs`: Parquets with economic event data (survey/actual/relevance).
  - `processed_path`: Directory for processed dataset artifact.
  - `tenors`: Yield columns to model (e.g., `USGG2YR Index`, ... `USGG30YR Index`).
  - `maturities`: Maturities in years aligned to `tenors` for loadings.
- `afns`
  - `estimation_window`: Present but not currently used (reserved for rolling estimation).
  - `lambda`: Nelson–Siegel decay parameter.
  - `measurement_noise_R`: Observation noise variance (bps²) per tenor; forms diagonal R.
- `regime`
  - `vol_window`: Rolling window for MOVE/VIX z-scores.
  - `inflation_halflife`: EWMA half-life for inflation surprise memory.
  - `n_states_range`: Candidate K values for HMM model selection.
  - `sticky_alpha`: Dirichlet mass added to self-transitions (enforces persistence).
  - `random_state`: RNG seed for reproducibility.
  - `volatility_scaling`:
    - `use_scaled_Q`: If true, scale state noise Q by regime.
    - `scale_method`: `trace_ratio` or `determinant_ratio`.
- `scenarios`
  - `confidence_levels`: Quantiles of interest for summaries.
  - `n_paths`: Monte Carlo paths per scenario.
  - `n_jobs`: Parallel workers; `-1` uses CPU count minus one.
  - `horizons`: Named horizons (e.g., `6m:126`, `1y:252`, ..., `5y:1260`).
  - `fed_shocks`: Policy shocks in bps (e.g., `[-200, -100, -50, 0, 50, 100, 200]`).
- `validation`
  - `test_horizons`: Horizons (days) to backtest; should align with scenario horizons.
  - `backtest`: `start_date`, `refit_frequency` (reserved), `min_training_years` (reserved).
  - `subsample_splits`: Regime subperiods for slicing analysis.
  - `dv01_weights`: Weight vector by tenor for weighted metrics (currently informational).
  - `coverage_tolerance`: Allowed deviation for coverage checks.
  - `multi_horizon`: `validate_consistency`, `max_horizon_years`.
- `output`
  - `figures_path`, `results_path`, `scenarios_path`.

---

## Data preparation (`src/utils.py`)

- `load_config`: YAML loader.
- `load_yield_data`
  - Reads `data/raw/RatesRegimes.parquet`. Handles `Date` column or index as timestamp.
  - Filters to configured `tenors` plus features: `MOVE Index`, `VIX Index`, `FEDL01 Index`, `USOSFR1/2/5 Curncy`.
  - Sorts and drops missing rows.
- `load_econ_data`
  - Reads `econ_reports` and `econ_jobs`, coerces dates, concatenates, sorts.
- `calculate_surprises`
  - Converts `Survey`/`Actual` to numeric; computes `Surprise` and z-scores by `Event` bucket.
- `build_regime_features`
  - MOVE/VIX z-scores over `vol_window`, OIS slope (5Y–1Y), Fed–OIS gap.
- `merge_econ_surprises`
  - Filters inflation events (CPI/PPI/PCE), aggregates daily `Surprise_z`, EWMA with `inflation_halflife` → `Inflation_mem`.
  - Merges into yield data, clears helper columns.
- `prepare_data`
  - Executes the full pipeline above, writes `data/processed/processed_data.parquet`.

The output DataFrame index is trading dates; columns include `tenors` plus regime features needed for HMM embedding.

---

## AFNS model (`src/afns.py`)

Implements an Arbitrage-Free Nelson–Siegel VAR(1) state-space with Kalman filtering and MLE.

- `AFNSModel(maturities, lambda_param, measurement_noise_R)`
  - Builds loadings matrix H from maturities and λ.
  - Observation noise `R = diag(measurement_noise_R)`.
- `fit(yields_df)`
  - Initializes factors via DNS proxies (L=(2Y+10Y)/2, S=10Y−2Y, C=2×5Y−2Y−10Y).
  - Parameterization: A (3×3), b (3×1), Q=diag(q11,q22,q33).
  - Objective: negative log-likelihood from Kalman filter; stability constraint `max|eig(A)| < 0.999`.
  - Optimizer: BFGS with progress reporting; then final Kalman smoothing pass for factors.
  - Diagnostics: per-maturity RMSE, overall RMSE, log-likelihood.
- Utilities
  - `fit_factors` (OLS factors), `reconstruct_yields`, `compute_residuals`, `estimate_covariance` (Ledoit–Wolf).
- Wrapper
  - `fit_afns_model(df, config)` returns `(df_factors, afns_model, Q)` and prints factor stats and correlations.

State vector is `f=[L,S,C]`. Measurement: `y = H f + ε`. Transition: `f' = A f + b + η`.

---

## Regime model (`src/regime.py`)

Sticky Gaussian HMM on daily factor changes Δf with model selection by BIC.

- `StickyHMM(n_states_range, sticky_alpha, random_state)`
  - Fits `GaussianHMM` for each K in `n_states_range` with full covariances.
  - Applies stickiness by adding `sticky_alpha` mass to diagonal of the transition matrix and row-normalizing.
  - BIC calculation includes means, full covariances, and transition degrees of freedom.
  - Selects K with minimal BIC; exposes `regime_means`, `regime_covs`, `transition_matrix`, and a time series of regime probabilities.
  - Expected duration per regime: `1/(1 - p_kk)`; printed for persistence diagnostics.
- Wrapper
  - `fit_regime_model(df, df_factors, afns_model, config)`
    - Uses `afns_model.factor_changes` as Δf.
    - Returns the fitted HMM object, most-likely `regime_labels` (Series), `regime_stats`, and `regime_covs` dict.

---

## Scenario generation (`src/scenarios.py`)

Monte Carlo path simulation with regime transitions; supports multiple named horizons and Fed shock overlays.

- `ScenarioGenerator(afns_model, hmm_model, spread_engine=None, horizons=None, n_paths=10000, n_jobs=-1, move_state='neutral', q_scaler=1.0)`
  - If `horizons` is not provided, defaults to `{6m,1y,2y,3y,5y}`.
  - `n_jobs=-1` uses CPU count minus one; falls back to 1.
  - Uses Lyapunov solution `P = solve_discrete_lyapunov(A, Q)` to initialize factor uncertainty.
- Path kernel per path
  1. Draw initial regime `s_0 ~ Categorical(gamma_T)` from current regime distribution.
  2. Draw initial factors `f_0 ~ N(f_T, P)`.
  3. For t=1..H: sample next regime via row of transition matrix, then propagate `f_{t+1}=A f_t + b + η_t` with `η_t ~ N(0, Q)` (optionally scaled by `q_scaler`).
  4. Transform factors to yields at each maturity via AFNS loadings.
- Fed shock mapping (initial condition only)
  - Rule of thumb: `+100bp policy` → `+0.80` to Level, `−0.10` to Slope (in percentage points).
- APIs
  - `generate_scenario(current_state, fed_shock_bp=0, random_seed=42)` → `(paths_df, percentiles_dict)` at the current horizon `self.horizon_days`.
  - `generate_multiple_scenarios(current_state, fed_shocks)` → dict of shocks to results (single horizon, backward-compat).
  - `generate_all_horizons(current_state, fed_shock_bp)` → dict of horizon name to `(paths, percentiles, horizon_days)`.
  - `format_scenario_output` → tabular summary from single-horizon output.

Notes
- `current_state` requires `factors` (L/S/C), last `yields`, and, for some flows, `regime_probs`.
- `main.py` uses `generate_all_horizons` and then collapses to the max horizon for legacy plots/CSV while preserving multi-horizon objects in-memory.

---

## Validation (`src/validation.py`)

Historical multi-horizon backtesting using the same state dynamics as scenarios, with regime-scaled volatility.

- `MultiHorizonValidator(afns_model, hmm_model, config)`
  - Uses `validation.test_horizons`; sets smaller `n_paths` (default 1000) for speed.
  - Regime scaling of Q: `trace_ratio` (default) or `determinant_ratio`.
- `simulate_paths(initial_state, horizon, n_paths)`
  - Initial `f_0` drawn around observed factors using `P_ss = solve_discrete_lyapunov(A, Q)` (scaled down) to reflect parameter uncertainty.
  - Regime transitions follow the sticky transition matrix; Q scaled by regime if configured.
- `factors_to_yields` maps factor paths to yield paths via AFNS loadings.
- `compute_metrics(forecasts, realized)` for each maturity
  - CRPS (efficient approximation), PIT (uniform target), coverage at 90/95/99, bias, RMSE, MAE.
- `run_backtest(df, df_factors, start_date)`
  - For each origin date and each horizon, simulates forecasts and scores against realized yields.
- `analyze_results(results_df)`
  - Aggregates metrics overall, by horizon, by tenor, and checks horizon consistency (RMSE should increase with horizon).
- `run_validation(...)` orchestrates and returns `{results_df, analysis, validator}`.

---

## Conformal layer (`src/conformal.py`)

Split conformal calibration to obtain distribution-free coverage guarantees conditioned on regime.

- `ConformalScenarioGenerator(base_generator, confidence_level=0.90)`
  - `calibrate(df, df_factors, regime_labels, calibration_dates, regime_covs)` computes residual quantiles per regime by simulating around current factors with regime-specific covariances.
  - `generate_scenario(current_state, fed_shock_bp, regime)` applies conformal residual bands to base median forecasts to yield calibrated `p5/p50/p95` per tenor.
- `compute_crps` provides a discrete-quantile CRPS approximation utility.

Note: In Phase 1A, conformal validation is scaffolded but skipped in `main.py`; it is ready to be integrated in a subsequent phase.

---

## Reporting (`src/reports.py`)

- `create_reinvestment_report(scenarios, current_state, config)`
  - Produces a table across shocks × horizons × maturities with `Forecast_P5/P50/P95`, mean/std, and reinvestment risk deltas relative to the current yield.
- `plot_multi_year_forecasts(scenarios, current_state, config)`
  - Visualizes the median path and 90% bands over years-forward for the no-shock scenario.

---

## Orchestrator (`main.py`)

Sequential pipeline with progress reporting and artifacts:

1. Load `config.yaml`.
2. `prepare_data`: build regime features, merge inflation surprises, save processed parquet.
3. `fit_afns_model`: fit AFNS by MLE + Kalman, obtain `A, b, Q` and filtered factors.
4. `fit_regime_model`: fit Sticky HMM on Δf, print persistence diagnostics.
5. Instantiate `ScenarioGenerator(horizons=...)`; for each Fed shock, call `generate_all_horizons`.
6. Collapse to the largest horizon for legacy fan chart/CSV while printing a multi-year forecast example (10Y).
7. `run_validation`: backtest across `validation.test_horizons`; print summary (CRPS, PIT, coverage, RMSE).
8. Generate figures (`scenario_fan.png`, `regime_analysis.png`) and save results:
   - `outputs/results/validation_results.csv`, `validation_summary.txt`
   - `outputs/scenarios/latest_scenarios.csv` (largest-horizon percentiles across shocks)

Headless plotting is enabled by setting `MPLCONFIGDIR` and `matplotlib` backend `Agg` for write-safe environments.

---

## Outputs and artifacts

- Figures (`output.figures_path`)
  - `scenario_fan.png`: per-maturity fan of p5–p95 across shocks at the max horizon.
  - `regime_analysis.png`: regime timeline and scatter overlays for 10Y and MOVE.
- Results (`output.results_path`)
  - `validation_results.csv`: per-origin, per-horizon, per-tenor metrics.
  - `validation_summary.txt`: overall and segmented metrics in human-readable form.
- Scenarios (`output.scenarios_path`)
  - `latest_scenarios.csv`: largest-horizon percentiles table across shocks and tenors.
- Processed data (`data.processed_path`)
  - `processed_data.parquet`: full dataset with regime features and surprises.

---

## Quick start

```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ensure raw inputs exist (see data paths in config.yaml)
python main.py
```

Artifacts will be written under `outputs/` and `data/processed/` as configured.

---

## Reproducibility and performance

- Parallelism: `scenarios.n_jobs=-1` uses CPU cores minus one; set to a fixed value for deterministic scheduling. Each path uses `random_seed + path_id` to ensure distinct draws.
- Path count: `scenarios.n_paths=10000` provides stable percentile estimates; reduce for speed.
- Lyapunov covariance: both scenarios and validation use `solve_discrete_lyapunov(A, Q)` for steady-state uncertainty. Validation scales this down for initial draws to avoid overdispersion.
- Regime scaling: Enable `regime.volatility_scaling.use_scaled_Q` to adjust Q by regime intensity; `trace_ratio` is a robust default.

---

## Extending the system

- Rolling/expanding estimation: `afns.estimation_window` and `validation.backtest.refit_frequency` are placeholders for future rolling refits.
- Credit spreads: `ScenarioGenerator` accepts an optional `spread_engine` to simulate spread term structures conditional on factors and regimes.
- Conformal guarantees: integrate `ConformalScenarioGenerator` post-calibration to replace/augment percentile bands with coverage-guaranteed intervals by regime.
- Additional factors/features: Extend `utils.build_regime_features` and update `regime` config accordingly.

---

## Notes and conventions

- Units are in percent for yields; R and Q are parameterized in consistent units (bps² comments denote intended interpretation of magnitudes).
- Maturities and `tenors` must remain aligned; the AFNS loadings are computed from `data.maturities` in years.
- Horizon names are arbitrary keys; horizon values are trading-day counts (≈252 per year).

---

## Technical write-up

This section provides a deeper, theory-forward exposition of the full stack: yield curve representation, state-space estimation, regime modeling, scenario generation, calibration guarantees, and validation methodology. Notation follows conventional time-series/state-space texts; trading days per year ≈ 252.

### 1. Yield curve representation via Arbitrage-Free Nelson–Siegel

- Nelson–Siegel functional form
  For maturity \(\tau \in \mathbb{R}_+\) and decay parameter \(\lambda>0\), define loadings
  \[
  h(\tau;\lambda) = \begin{bmatrix}
  1 \\
  \frac{1-e^{-\lambda\tau}}{\lambda\tau} \\
  \frac{1-e^{-\lambda\tau}}{\lambda\tau} - e^{-\lambda\tau}
  \end{bmatrix}.
  \]
  The instantaneous zero rate (annualized, percent) at \(\tau\) is \(y_t(\tau) = h(\tau;\lambda)^\top f_t + \varepsilon_t\) where \(f_t=[L_t,S_t,C_t]^\top\) are the level, slope, and curvature factors.

- Discretized observation model
  With a fixed grid of maturities \(\{\tau_m\}_{m=1}^M\), define \(H \in \mathbb{R}^{M\times 3}\) with rows \(h(\tau_m;\lambda)^\top\). Observed yields vector \(y_t\in\mathbb{R}^M\) satisfies
  \[
  y_t = H f_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, R),
  \]
  where \(R\) is diagonal (independent measurement noise across tenors). In code, `R = diag(measurement_noise_R)`; units are consistent with percent.

- Arbitrage-free nuance
  The AFNS family imposes no-arbitrage via a risk-neutral dynamic for \(f_t\) and a consistent mapping to discount factors. Here we adopt a reduced-form “AFNS-inspired” state-space with empirical Kalman MLE and do not explicitly construct the SDF; arbitrage consistency is approximated via stable dynamics and fixed \(\lambda\).

### 2. State dynamics and Kalman MLE

- VAR(1) with drift
  \[
  f_{t+1} = A f_t + b + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, Q).
  \]
  We estimate \(\theta = (A,b,Q)\) by maximizing the log-likelihood implied by the linear-Gaussian state-space model using a Kalman filter.

- Initialization
  The initial state \(f_0\) is set by OLS on the first observation, i.e., \(f_0 = (H^\top H)^{-1}H^\top y_0\); initial covariance \(P_0=10 I_3\). Factor seeds for parameter guesses are taken from DNS proxies: \(L=(y_{2Y}+y_{10Y})/2\), \(S=y_{10Y}-y_{2Y}\), \(C=2y_{5Y}-y_{2Y}-y_{10Y}\).

- Stability and identification
  We constrain \(\rho(A) < 1\) (spectral radius) by penalizing objectives when any eigenvalue magnitude exceeds 0.999, ensuring a stationary solution for the discrete Lyapunov equation.

- Log-likelihood
  For t=1..T, with innovations \(\nu_t = y_t - H \hat f_{t|t-1}\) and innovation covariance \(S_t = H P_{t|t-1} H^\top + R\), the (Gaussian) log-likelihood is
  \[
  \log \mathcal{L}(\theta) = -\tfrac{1}{2}\sum_t \Big[ M\log(2\pi) + \log\det S_t + \nu_t^\top S_t^{-1} \nu_t \Big].
  \]
  We use BFGS to optimize \(\theta\), re-running a final filtering pass at the optimum to obtain smoothed factor estimates.

- Steady-state covariance via Lyapunov
  The unconditional covariance satisfies \(P = A P A^\top + Q\). We compute \(P\) with `solve_discrete_lyapunov(A, Q)`; this is used as an initial dispersion around \(f_T\) in simulations.

### 3. Regime dynamics via sticky Gaussian HMM

- Observation for the HMM
  We model daily factor changes \(\Delta f_t = f_t - f_{t-1}\) under a K-state Gaussian HMM: \(\Delta f_t | s_t=k \sim \mathcal{N}(\mu_k, \Sigma_k)\). Persistence is governed by transition matrix \(\Pi\in[0,1]^{K\times K}\).

- Stickiness
  After fitting a standard GaussianHMM, we add \(\alpha\) mass to the diagonal of \(\Pi\) and renormalize rows:
  \[ \Pi^{(sticky)}_{ij} = \frac{\Pi_{ij} + \alpha\,\mathbb{1}\{i=j\}}{\sum_j (\Pi_{ij}+\alpha\,\mathbb{1}\{i=j\})}. \]
  The expected duration in regime k is \(\mathbb{E}[D_k] = 1/(1-\Pi_{kk})\). We select K by minimizing BIC over candidates in `n_states_range`.

- Role in simulation and validation
  The HMM provides (i) transition kernel for regime sampling and (ii) regime-indexed covariance scaling factors for state noise (trace-ratio or determinant-ratio).

### 4. Scenario generation: multi-horizon Monte Carlo with regime transitions

- Initial condition
  At as-of date T, we set mean state to the filtered \(f_T\). We draw an initial regime \(s_T\) from the latest posterior probabilities \(\gamma_T\). Factor initial draw uses \(\mathcal{N}(f_T, P)\) where P is the Lyapunov solution.

- Fed policy shock mapping
  We embed a small-signal mapping: a \(+100\) bp policy shock maps to \(+0.80\) in the level factor and \(-0.10\) in the slope factor (percent points), applied as an offset to \(f_T\) at t=0. This is a pragmatic rule-of-thumb; for structural consistency, a macro-yields joint model would be required.

- Evolution and measurement
  For horizon H in trading days and for each step, we sample the next regime using the row of \(\Pi\) corresponding to the current regime and propagate \(f\) by the VAR(1). Yields are computed via \(y=H f\) for the configured maturities. Repeating over many paths produces an empirical predictive distribution.

- Percentiles and aggregation
  For each maturity and horizon, we compute \(p5, p50, p95\), mean, std from the terminal draws. Multi-horizon outputs are assembled as a map: horizon → {paths, percentiles, horizon_days}.

### 5. Conformal calibration (optional, scaffolded)

- Split conformal residualization
  Given calibration dates \(\mathcal{C}\), we compute residuals between realized yields and simulated medians under regime-conditional draws and form empirical quantiles per regime and tenor. With target coverage \(1-\alpha\), split conformal prediction yields finite-sample marginal coverage guarantees under exchangeability assumptions, i.e., \(\mathbb{P}[Y \in \hat{C}(X)] \ge 1-\alpha\).

- Regime conditioning
  By computing residual quantiles per \(s\), intervals adapt to heteroskedasticity across regimes, which is crucial for long-horizon tails.

### 6. Validation methodology and diagnostics

- Simulation-consistent forecasting
  For each origin date and horizon h, we simulate factor paths with the same VAR(1)+HMM transition mechanics, transform to yields, and score against realized yields at t+h.

- Metrics
  - CRPS: \(\operatorname{CRPS}(F,y)= \int_{-\infty}^{\infty} (F(z)-\mathbb{1}\{z\ge y\})^2 dz\). We compute an efficient empirical approximation using pairwise distances of a subsample of forecast draws.
  - PIT: Probability integral transform \(U=F(Y)\) should be Uniform(0,1). We monitor mean(PIT) and (optionally) histogram shape.
  - Coverage: Indicator that \(Y\) falls within \([p_{\ell},p_u]\) at 90/95/99.
  - Bias/RMSE/MAE: Conventional point-forecast diagnostics using the forecast sample mean.

- Horizon-consistency check
  In well-behaved systems, uncertainty should increase with horizon; we verify that RMSE is nondecreasing across horizons.

### 7. Numerical aspects and practicalities

- Units and scaling
  Yields are in percent; Q and R magnitudes should reflect percent-scale variability (comments mention bps² for intuition). The Lyapunov P can be large; validation scales initial covariance down to avoid excessive dispersion at short horizons.

- Optimization
  BFGS is nonconvex in this setting; early termination due to numerical tolerances is acceptable if RMSE diagnostics are reasonable. Stability penalties prevent explosive A.

- Parallelism and seeding
  With `n_jobs>1`, multiprocess Monte Carlo is used. Each path seeds with `random_seed + path_id` to ensure reproducibility under fixed scheduling.

- Model risk and limitations
  - Fixed \(\lambda\) simplifies estimation but may bias curvature representation.
  - Shock mapping is heuristic; a structural macro term structure model would be required for policy-consistent transmission across the curve.
  - Gaussian innovations understate tail risk; regime-dependent scaling partially compensates but does not capture jumps.

### 8. Assumptions summary

- Linear-Gaussian state-space for factors with stationary VAR(1).
- Measurement noise independent across tenors (diagonal R).
- Regime process is first-order Markov with time-homogeneous transitions (after stickiness).
- Exchangeability on calibration residuals for conformal coverage guarantees.
- Trading-day calendar with 252 days per year; horizons expressed in trading days.

### 9. Extension ideas

- Time-varying \(\lambda\) or augmented factor basis (e.g., Svensson) with penalties for stability.
- Rolling re-estimation and parameter uncertainty propagation (e.g., sampling \(A,b,Q\)).
- Regime-dependent drift \(b_s\) and/or transition matrix conditioned on observables (input-driven HMM).
- Copula-based dependence to fuse spreads and macro variables with yields for joint scenarios.
