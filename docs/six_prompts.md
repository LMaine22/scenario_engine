**🚀 HERE WE GO - ALL 6 PROMPTS FOR CURSOR**

Copy each prompt to Cursor one at a time, test after each, then move to the next.

---

## **PROMPT 1: VAR(1) + Kalman Filter (src/afns.py)**

```
CONTEXT: We're upgrading our AFNS model from static OLS factor extraction to a full VAR(1) state-space model with Kalman filter estimation. This adds factor dynamics (mean-reversion, momentum) which our current system lacks.

GOAL: Replace the fit() method in src/afns.py to use VAR(1) + Kalman filter + Maximum Likelihood Estimation.

INSTRUCTIONS:
Read docs/var_kalman_detailed.md completely. Focus on these sections:
- Section 2: Factor Initialization (DNS approximation) - lines 98-134 in the MD
- Section 3: Kalman Filter implementation - lines 136-196
- Section 4: Maximum Likelihood Estimation - lines 198-361

WHAT TO IMPLEMENT:
1. Add initialize_factors_dns() method (from Section 2.1 of MD)
   - Uses DNS formulas: L = (UST_2 + UST_10)/2, S = UST_10 - UST_2, C = 2*UST_5 - UST_2 - UST_10

2. Add _kalman_filter() helper method (from Section 3 of MD)
   - Use filterpy.kalman.KalmanFilter library
   - Implements predict + update steps
   - Returns filtered_factors and log_likelihood
   - Fixed measurement noise R = 4.0 bps²

3. Add _objective() function for optimization (from Section 4.3 of MD)
   - Returns negative log-likelihood
   - Includes stability constraint: |eigenvalues(A)| < 0.999
   - Ensures Q diagonal is positive via np.abs()

4. REPLACE the fit() method entirely (from Section 4 of MD)
   - Initialize parameters from DNS factors
   - Use scipy.optimize.minimize with method='BFGS'
   - Optimize A (3×3), b (3×1), Q_diag (3×1) = 15 parameters total
   - Store: self.A, self.b, self.Q
   - Compute and store: self.factors, self.factor_changes
   - Compute RMSE by maturity for validation

WHAT TO KEEP FROM CURRENT CODE:
- Class structure (class AFNSModel)
- __init__() method with config integration
- compute_loadings() method (Nelson-Siegel loading functions)
- _compute_loading_matrix() method
- Any other helper methods not mentioned above

DEPENDENCIES TO ADD:
from scipy.optimize import minimize
from filterpy.kalman import KalmanFilter

SUCCESS CRITERIA:
- fit() returns dict with: A, b, Q, factors, factor_changes, log_likelihood, rmse_by_maturity
- factors is a DataFrame with columns ['L', 'S', 'C']
- factor_changes is factors.diff().dropna()
- Overall RMSE < 10 bps (should be ~3-5 bps for good fit)

TEST: After implementing, run on our data and check that factors evolve smoothly and RMSE is reasonable.
```

---

## **PROMPT 2: Sticky HMM (src/regime.py)**

```
CONTEXT: We're upgrading from static GMM clustering to a Sticky Hidden Markov Model that models regime TRANSITIONS. This allows us to forecast "probability of moving from calm to crisis" rather than assuming current regime persists forever.

GOAL: Replace the regime classification system with a Sticky HMM that fits on factor CHANGES (Δf_t) and includes a transition matrix.

INSTRUCTIONS:
Read docs/sticky_hmm_detailed.md completely. Focus on:
- Section 1: Model Structure (what HMM observes)
- Section 2: Stickiness Implementation (Dirichlet prior α=10)
- Section 3: Model Selection (BIC for choosing K)
- Section 4: Fitting Process (using hmmlearn library)

WHAT TO IMPLEMENT:
1. Create a NEW class structure (replace the old one completely):
   - Class: StickyHMM with __init__(n_states_range=[2,3,4,5], sticky_alpha=10.0, random_state=42)

2. Add _apply_sticky_prior() method (from Section 2.2 of MD)
   - Adds α to diagonal of transition matrix
   - Renormalizes rows to sum to 1
   - Formula: π[k,k] += α, then π[k,:] = π[k,:] / sum(π[k,:])

3. Add _compute_bic() method (from Section 3.1 of MD)
   - Formula: BIC = -2*log_likelihood + n_params*log(n_observations)
   - n_params = 3K + 6K + K² - K (means + covariances + transition matrix)

4. Add _compute_regime_duration() method (from Section 2.3 of MD)
   - Formula: duration[k] = 1 / (1 - π[k,k])

5. Implement fit() method (from Section 4 of MD):
   - Input: factor_changes DataFrame (Δf_t), NOT factor levels
   - Loop over K ∈ [2,3,4,5]
   - For each K: Fit GaussianHMM with covariance_type='full', n_iter=100
   - After fitting: Apply sticky prior to transmat_
   - Compute BIC for each K
   - Select K with lowest BIC
   - Store: self.n_states, self.model, self.regime_means, self.regime_covs, self.transition_matrix
   - Compute regime_probs: model.predict_proba(X) as DataFrame

6. Add predict_regime_probs() method (from Section 7.3 of MD)
   - Returns gamma_T (most recent regime distribution)
   - Checks for ambiguity (max prob < 0.45)

7. Add get_regime_timeline() and get_most_likely_regimes() helper methods

WHAT TO KEEP:
- Integration with our main.py structure
- Config parameter reading

DEPENDENCIES TO ADD:
from hmmlearn import hmm

CRITICAL DIFFERENCES FROM CURRENT CODE:
- Input is factor_changes (Δf_t), NOT [MOVE_z, VIX_z, OIS_slope, ...]
- Outputs transition_matrix π (not in current code)
- BIC model selection for K (not fixed at 4)
- Sticky prior (α=10) to enforce persistence

SUCCESS CRITERIA:
- fit() returns dict with: n_states, transition_matrix, regime_means, regime_covs, regime_probs
- transition_matrix is K×K with high diagonal (e.g., π[k,k] > 0.85)
- regime_probs is T×K DataFrame with regime posterior probabilities
- BIC selects K (usually 3 or 4)

TEST: After fitting, check that transition_matrix diagonal is high (regime persistence) and regimes align with known market events.
```

---

## **PROMPT 3: Path Simulator (src/scenarios.py)**

```
CONTEXT: We're upgrading from single-step Monte Carlo (instant jump to horizon) to multi-step path simulation where factors evolve day-by-day over 126 days, and regimes can TRANSITION during the path. This captures uncertainty accumulation and regime switches.

GOAL: Replace single-step scenario generation with 126-day path simulation including regime dynamics.

INSTRUCTIONS:
Read docs/path_simulator_detailed.md completely. Focus on:
- Section 2: Single Path Logic (the core algorithm)
- Section 3: MOVE Scaling (how volatility regime affects Q)
- Section 4: Parallelization (using multiprocessing)
- Section 5: Output Structure

WHAT TO IMPLEMENT:
1. Modify the ScenarioGenerator class:
   - Add parameters: n_paths=10000, horizon_days=126, n_jobs=(cpu_count-1)
   - Store: afns_model, regime_covs (by regime), hmm transition matrix

2. ADD _simulate_single_path() method (from Section 2 of MD):
   ```python
   def _simulate_single_path(self, path_id, sim_params):
       # Unpack: f_T, P_T, gamma_T, horizon_days, random_seed
       
       # Set seed
       path_seed = random_seed + path_id
       np.random.seed(path_seed)
       
       # Draw initial regime from gamma_T
       s_t = np.random.choice(len(gamma_T), p=gamma_T)
       
       # Draw initial factors from N(f_T, P_T)
       f_t = np.random.multivariate_normal(f_T, P_T)
       
       # Storage
       path_data = {'t': [0], 'regime': [s_t], 'L': [f_t[0]], 'S': [f_t[1]], 'C': [f_t[2]]}
       
       # Simulate forward
       for t in range(1, horizon_days + 1):
           # Regime transition
           s_t = np.random.choice(n_regimes, p=transition_matrix[s_t, :])
           
           # Factor evolution: f_{t+1} = A*f_t + b + η
           eta = np.random.multivariate_normal(np.zeros(3), Q * q_scaler)
           f_t = A @ f_t + b + eta
           
           # Store
           path_data['t'].append(t)
           path_data['regime'].append(s_t)
           path_data['L'].append(f_t[0])
           path_data['S'].append(f_t[1])
           path_data['C'].append(f_t[2])
       
       # Compute yields from factors
       for maturity in maturities:
           loadings = compute_loadings(maturity)
           yields = factors_array @ loadings
           path_data[f'UST_{maturity}'] = yields
       
       return pd.DataFrame(path_data)
   ```

3. Implement generate_scenario() method (from Section 2 of MD):
   - Initialize state: Get f_T (current factors), compute P_T via Lyapunov equation
   - Get gamma_T (current regime distribution) from HMM
   - Get MOVE q_scaler (0.8 for low, 1.0 for neutral, 1.3 for high) - read from Section 3
   - Apply Fed shock to factors: f_T_shocked = f_T + [shock_L, shock_S, shock_C]
   - Use multiprocessing.Pool to parallelize n_paths simulations
   - Combine all paths into single DataFrame
   - Compute percentiles at horizon: p5, p50, p95

4. ADD parallelization wrapper (from Section 4 of MD):
   ```python
   def _simulate_single_path_wrapper(self, args):
       path_id, sim_params = args
       return self._simulate_single_path(path_id, sim_params)
   
   # In generate_scenario:
   with Pool(processes=self.n_jobs) as pool:
       all_paths = list(tqdm(
           pool.imap(self._simulate_single_path_wrapper, 
                    [(i, sim_params) for i in range(n_paths)]),
           total=n_paths, desc="Simulating paths"
       ))
   ```

5. Compute P_T via Lyapunov equation (from Section 1.3 of MD):
   ```python
   from scipy.linalg import solve_discrete_lyapunov
   P_T = solve_discrete_lyapunov(A, Q)
   ```

WHAT TO KEEP:
- Fed shock scenarios structure (-100, -50, 0, +50, +100)
- Percentile output format
- Integration with main.py

WHAT TO REMOVE:
- Single-step factor shock generation
- Static regime assumption

DEPENDENCIES TO ADD:
from multiprocessing import Pool, cpu_count
from scipy.linalg import solve_discrete_lyapunov

CRITICAL CHANGES:
- n_paths = 10,000 (not 1,000)
- horizon_days = 126 (6 months) or 252 (12 months)
- Regimes transition during path using HMM transition_matrix
- Factors evolve via VAR(1): f_t = A*f_{t-1} + b + eta
- Parallelization for speed

SUCCESS CRITERIA:
- generate_scenario() returns paths_df with columns: [path_id, t, regime, L, S, C, UST_2, ..., UST_30]
- Percentiles dict with p5, p50, p95 for each tenor
- 10,000 paths × 126 days = 1.26M rows in paths_df
- Regimes vary during paths (not frozen)

TEST: Run one scenario and verify paths show regime transitions and factors evolve smoothly.
```

---

## **PROMPT 4: Spread Engine (NEW FILE: src/spreads.py)**

```
CONTEXT: We're adding credit spread modeling on top of treasury yields. Spreads follow a regime-dependent AR(1) model with factor loadings and Student-t errors (fat tails for credit blow-outs). This is OPTIONAL - if no spread data, system works fine with just yields.

GOAL: Create a new SpreadEngine class that models credit spreads conditionally on treasury factors and regimes.

INSTRUCTIONS:
Read docs/spread_simulation_detailed.md completely. Focus on:
- Section 1: AR(1) Model Structure
- Section 2: Student-t Errors
- Section 3: Regime Dependence
- Section 4: Integration in Path Simulation

CREATE NEW FILE: src/spreads.py

WHAT TO IMPLEMENT:
1. Class: SpreadEngine with __init__(buckets, log_floor_epsilon=1.0, random_state=42)
   - buckets: list of (sector, maturity) tuples, e.g., [('IG', 5), ('Agency', 10), ('Muni', 30)]

2. Signed log transform methods (from Section 1.2 of MD):
   ```python
   def _transform_to_log_space(self, spreads):
       abs_spreads = np.abs(spreads)
       sign = np.sign(spreads)
       return sign * np.log(abs_spreads + self.log_floor_epsilon)
   
   def _transform_from_log_space(self, log_spreads):
       sign = np.sign(log_spreads)
       abs_log = np.abs(log_spreads)
       return sign * (np.exp(abs_log) - self.log_floor_epsilon)
   ```

3. Add _fit_ar_model() method (from Section 1.3 of MD):
   - Fits: z_{t+1} = α + φ*z_t + β^L*ΔL + β^S*ΔS + β^C*ΔC + ε
   - Uses weighted least squares with regime probabilities as weights
   - Fits Student-t to residuals: (ν, σ) = scipy.stats.t.fit(residuals)
   - Constrains: 3 ≤ ν ≤ 30, σ > 0
   - Returns: dict with alpha, phi, beta_L, beta_S, beta_C, nu, sigma

4. Implement fit() method (from Section 3.3 of MD):
   - Input: spreads_df (columns like 'IGSpr_5'), factor_changes_df, regime_probs_df
   - For each bucket (sector, mat):
     - Transform spread to log space
     - For each regime k:
       - Fit AR(1) model weighted by P(regime=k|data)
       - Store params[(sector, mat, k)] = {alpha, phi, beta_L, beta_S, beta_C, nu, sigma}
   - Track bucket_windows for data availability
   - Skip buckets with <30 observations

5. Implement simulate_spreads() method (from Section 4 of MD):
   ```python
   def simulate_spreads(self, factors_path, regime_path, initial_spreads, random_seed=None):
       # factors_path: T×3 array [L, S, C]
       # regime_path: T array of regime indices
       # initial_spreads: dict {(sector, mat): value}
       
       delta_factors = np.diff(factors_path, axis=0, prepend=factors_path[0:1, :])
       spreads_sim = {}
       
       for sector, mat in self.buckets:
           z_path = np.zeros(T)
           z_path[0] = transform_to_log_space(initial_spreads[(sector, mat)])
           
           for t in range(1, T):
               regime = int(regime_path[t])
               params = self.params[(sector, mat, regime)]
               
               # AR(1) mean
               z_mean = (params['alpha'] + params['phi']*z_path[t-1] + 
                        params['beta_L']*delta_factors[t,0] +
                        params['beta_S']*delta_factors[t,1] +
                        params['beta_C']*delta_factors[t,2])
               
               # Student-t shock
               shock = scipy.stats.t.rvs(params['nu'], loc=0, scale=params['sigma'])
               z_path[t] = z_mean + shock
           
           spreads_sim[(sector, mat)] = transform_from_log_space(z_path)
       
       return spreads_sim
   ```

6. Add graceful failure:
   - If bucket params missing for regime, use fallback regime
   - If no regimes available, carry forward previous value
   - Log warnings but don't crash

DEPENDENCIES:
from scipy import stats
import numpy as np
import pandas as pd

INTEGRATION WITH SCENARIOS:
In src/scenarios.py _simulate_single_path(), after computing yields:
```python
if self.spread_engine is not None:
    spreads_sim = self.spread_engine.simulate_spreads(
        factors_path, regime_path, current_spreads, random_seed=path_seed
    )
    for (sector, mat), spread_path in spreads_sim.items():
        path_data[f'{sector}Spr_{mat}'] = spread_path
```

SUCCESS CRITERIA:
- fit() stores params[(sector, mat, regime)] for all buckets × regimes
- simulate_spreads() returns dict {(sector, mat): array of length T}
- Student-t errors have heavier tails than Gaussian (ν typically 3-15)
- System works even if spread data is missing (graceful degradation)

TEST: If you have IG/Agency/Muni spread data, fit the model. If not, skip this component - the system works fine with just yields.
```

---

## **PROMPT 5: CRPS + PIT Validation (src/validation.py)**

```
CONTEXT: We're adding proper probabilistic forecast evaluation beyond just coverage. CRPS measures full distribution quality (not just point forecast), and PIT checks calibration (are our probabilities honest?).

GOAL: Add CRPS and PIT computation to our existing validation framework.

INSTRUCTIONS:
Read docs/validation_metrics_detailed.md completely. Focus on:
- Section 1: CRPS definition and computation
- Section 2: PIT definition and interpretation
- Section 3: Coverage (we already have this)

WHAT TO ADD to existing src/validation.py:

1. Add _compute_crps() method (from Section 1.2 of MD):
   ```python
   def _compute_crps(self, forecast_samples, realized):
       """
       Compute Continuous Ranked Probability Score.
       CRPS = E|X - y| - 0.5*E|X - X'|
       
       Lower is better. Units: same as variable (e.g., bps for yields).
       """
       # Term 1: Mean absolute error
       term1 = np.mean(np.abs(forecast_samples - realized))
       
       # Term 2: Spread penalty (use 100 samples for efficiency)
       n = len(forecast_samples)
       if n > 100:
           idx = np.random.choice(n, 100, replace=False)
           samples = forecast_samples[idx]
       else:
           samples = forecast_samples
       
       diffs = []
       for i in range(len(samples)):
           for j in range(i+1, len(samples)):
               diffs.append(np.abs(samples[i] - samples[j]))
       
       term2 = 0.5 * np.mean(diffs) if diffs else 0
       
       return term1 - term2
   ```

2. Add _compute_pit() method (from Section 2.2 of MD):
   ```python
   def _compute_pit(self, forecast_samples, realized):
       """
       Compute Probability Integral Transform.
       PIT = Empirical CDF(realized)
       
       Should be Uniform[0,1] if well-calibrated.
       """
       pit = (forecast_samples < realized).mean()
       return pit
   ```

3. MODIFY backtest_scenarios() in existing validation to compute CRPS and PIT:
   - After computing p5, p50, p95, also compute:
     ```python
     crps = self._compute_crps(sim_yields[:, j], realized[j])
     pit = self._compute_pit(sim_yields[:, j], realized[j])
     ```
   - Add to results dict: 'crps': crps, 'pit': pit

4. MODIFY run_validation() summary to report:
   - Average CRPS by tenor
   - Average PIT (should be ~0.5)
   - PIT histogram check: if 0.45 < avg_pit < 0.55, print "✓ Well-calibrated"
   - CRPS < 20bp check for yields

5. Update validation_summary.txt output to include:
   ```
   CRPS Metrics:
     USGG2YR: X.XX bps
     USGG10YR: X.XX bps
   
   PIT Calibration:
     Average PIT: 0.XX (target: 0.50)
     Status: [Well-calibrated / Biased high / Biased low]
   ```

WHAT TO KEEP:
- All existing coverage computation
- Existing backtest framework
- DV01-weighted errors
- Cosine similarity

DEPENDENCIES TO ADD:
None (just numpy)

INTERPRETATION GUIDE (add to docstrings):
- CRPS < 10bp: Excellent
- CRPS 10-20bp: Good
- CRPS > 20bp: Poor
- PIT ~0.5: Unbiased
- PIT < 0.45: Over-predicting
- PIT > 0.55: Under-predicting
- PIT histogram should be flat (uniform)

SUCCESS CRITERIA:
- CRPS computed for every forecast
- PIT computed for every forecast
- Summary includes CRPS and PIT alongside coverage
- CRPS < 20bp for treasury yields (expect ~5-15bp)
- Average PIT between 0.45-0.55

TEST: After running validation, check that CRPS is reasonable and PIT is near 0.5.
```

---

## **PROMPT 6: Integration + Config Updates (main.py, config.yaml)**

```
CONTEXT: Final step - integrate all new components and update configuration for new parameters.

GOAL: Wire up VAR+Kalman, Sticky HMM, Path Simulator, Spread Engine, and CRPS/PIT validation into main.py orchestrator.

INSTRUCTIONS:
1. Update config.yaml to add new parameters:

```yaml
# AFNS MODEL (updated)
afns:
  lambda: 0.0609
  measurement_noise_R: 4.0  # NEW: Fixed measurement noise

# REGIME (updated)
regime:
  n_states_range: [2, 3, 4, 5]  # NEW: Try multiple K
  sticky_alpha: 10.0             # NEW: Stickiness parameter
  random_state: 42

# SCENARIOS (updated)
scenarios:
  n_paths: 10000              # NEW: 10K paths (was 1K)
  horizon_days: 126           # NEW: 6-month horizon
  confidence_levels: [0.05, 0.5, 0.95]
  n_jobs: -1                  # NEW: Use all CPUs minus 1
  
# SPREADS (NEW - optional)
spreads:
  buckets:  # Optional: only if you have spread data
    - ['IG', 5]
    - ['IG', 10]
    - ['Agency', 5]
    - ['Agency', 10]
    - ['Muni', 5]
    - ['Muni', 10]
  log_floor_epsilon: 1.0
```

2. Update main.py orchestration:

```python
from src.spreads import SpreadEngine  # NEW import

# Step 3: Fit AFNS (UPDATED - now uses VAR+Kalman)
pbar.set_description("[3/8] Fitting AFNS VAR(1) + Kalman filter")
df_factors, afns_model, cov_matrix = fit_afns_model(df, config)
pbar.update(1)

# Step 4: Fit regime (UPDATED - now Sticky HMM)
pbar.set_description("[4/8] Fitting Sticky HMM")
classifier, regime_labels, regime_stats, regime_covs = fit_regime_model(
    df, df_factors, afns_model, config
)
pbar.update(1)

# Step 4.5: Fit spreads (NEW - optional)
pbar.set_description("[5/8] Fitting spread models (optional)")
spread_engine = None
if 'spreads' in config and config['spreads'].get('buckets'):
    try:
        from src.spreads import SpreadEngine
        spread_engine = SpreadEngine(
            buckets=config['spreads']['buckets'],
            log_floor_epsilon=config['spreads'].get('log_floor_epsilon', 1.0)
        )
        # Check if spread data exists
        spread_cols = [f"{s}Spr_{m}" for s, m in config['spreads']['buckets']]
        if any(col in df.columns for col in spread_cols):
            spread_engine.fit(df, df_factors.diff().dropna(), classifier.regime_probs)
            print("  Spread models fitted successfully")
        else:
            print("  No spread data found, skipping spreads")
            spread_engine = None
    except Exception as e:
        print(f"  Spread fitting failed: {e}, continuing without spreads")
        spread_engine = None
pbar.update(1)

# Step 5: Generate scenarios (UPDATED - now path simulation)
pbar.set_description("[6/8] Generating 10K path scenarios")
scenarios, df_scenarios, current_state = run_scenario_analysis(
    df, df_factors, afns_model, classifier, regime_labels, regime_covs, 
    spread_engine, config,  # NEW: pass spread_engine
    fed_shocks=[-100, -50, 0, 50, 100]
)
pbar.update(1)

# Step 6: Validation (UPDATED - now includes CRPS/PIT)
pbar.set_description("[7/8] Running validation with CRPS/PIT")
validation_results = run_validation(
    df, df_factors, afns_model, classifier, regime_labels, regime_covs, 
    spread_engine, config
)
pbar.update(1)
```

3. Update fit_afns_model() in utils.py (or wherever it is):
   - Ensure it calls afns_model.fit() which now returns VAR parameters
   - Extract A, b, Q from results

4. Update fit_regime_model():
   - Pass factor_changes (not levels) to HMM
   - Extract transition_matrix from results

5. Update run_scenario_analysis():
   - Pass spread_engine (can be None)
   - Use path simulation (not single-step)
   - Return scenarios with 10K paths

6. Update validation summary output:
   - Include CRPS metrics
   - Include PIT calibration check
   - Keep existing coverage metrics

EXPECTED CHANGES IN OUTPUT:
- Scenarios now show regime transitions during paths
- Validation includes CRPS and PIT
- If spread data present, scenarios include spread forecasts
- Coverage should improve from ~75% to ~85-90%

SUCCESS CRITERIA:
- All components integrate cleanly
- System works with or without spread data
- Validation shows improved coverage
- CRPS and PIT metrics reported
- No crashes or errors

TEST: Run full pipeline end-to-end and verify all outputs are generated correctly.
```

---

## ✅ **EXECUTION PLAN**

**Run these prompts IN ORDER:**
1. Prompt 1 → Test AFNS fit
2. Prompt 2 → Test HMM fit  
3. Prompt 3 → Test path simulation
4. Prompt 4 → Test spreads (skip if no data)
5. Prompt 5 → Test CRPS/PIT
6. Prompt 6 → Integration

**After each prompt:**
- Review Cursor's code
- Test the component
- Fix any bugs
- Move to next

**Total time: 3-4 hours**

**Expected result: 85-90% coverage on 2022-2025 test set**

---

**🚀 GO! Start with Prompt 1 and report back after each component!**

answers to clarifying questions that cursor asked not that important just extra context for you: I have treasury yields (USGG series) and credit yields (IG, Agency, Muni) in RatesRegimes.parquet. 
See docs/RatesRegimes_DataDictionary.md for full list.

To get spreads, I need to compute:
- IGSpr_1 = IGUUAC01_BVLI_Index - USGG1YR_Index (if I had 1yr treasury)
- IGSpr_5 = IGUUAC05_BVLI_Index - USGG5YR_Index
- IGSpr_7 = IGUUAC07_BVLI_Index - USGG7YR_Index  
- IGSpr_10 = IGUUAC10_BVLI_Index - USGG10YR_Index

- AgencySpr_1 = BVCSUG01_BVLI_Index - USGG1YR_Index (if available)
- AgencySpr_2 = BVCSUG02_BVLI_Index - USGG2YR_Index
- AgencySpr_5 = BVCSUG05_BVLI_Index - USGG5YR_Index
- AgencySpr_7 = BVCSUG07_BVLI_Index - USGG7YR_Index
- AgencySpr_10 = BVCSUG10_BVLI_Index - USGG10YR_Index
- AgencySpr_20 = BVCSUG20_BVLI_Index - USGG20YR_Index (if I had 20yr treasury)

- MuniSpr_2 = BVMB2Y_Index - USGG2YR_Index
- MuniSpr_3 = BVMB3Y_Index - USGG3YR_Index
- MuniSpr_5 = BVMB5Y_Index - USGG5YR_Index
- MuniSpr_7 = BVMB7Y_Index - USGG7YR_Index
- MuniSpr_10 = BVMB10Y_Index - USGG10YR_Index
- MuniSpr_30 = BVMB30Y_Index - USGG30YR_Index

Before fitting SpreadEngine:
1. Compute spread columns where both yields exist
2. Add to dataframe with naming convention: {sector}Spr_{maturity}
3. If spreads can't be computed, SpreadEngine should skip gracefully with log warning

Implement SpreadEngine with graceful degradation:
- Try to compute spreads first
- If no spread data available, log warning and continue with yields only
- System should work perfectly fine with just treasuries
```

---

## **Question 2: Regime Features**

**Answer: a) Replace regime features with factor changes (as per documentation)**

**What to tell Cursor:**
```
Replace the current regime feature engineering entirely.

OLD (current system):
- Uses ['MOVE_z', 'VIX_z', 'OIS_slope', 'Fed_vs_OIS', 'Inflation_mem']
- Static GMM clustering
- No transitions

NEW (Sticky HMM):
- Uses ONLY factor changes: [ΔL, ΔS, ΔC] from AFNS model
- Input to HMM is factor_changes DataFrame from afns_model.factor_changes
- Learns regime transitions via transition matrix π
- This is the correct approach per documentation

Remove all the MOVE_z, VIX_z feature engineering from regime.py.
The new StickyHMM class should ONLY use factor changes.
```

---

## **Question 3: Parallelization Jobs**

**Answer: a) n_jobs = -1 (all CPUs minus 1) - make it configurable in config.yaml**

**What to tell Cursor:**
```
Default: n_jobs = -1 (use all CPUs minus 1)

In config.yaml:
scenarios:
  n_jobs: -1  # -1 means all CPUs minus 1, or specify a number like 4 or 7

In code:
if config['scenarios']['n_jobs'] == -1:
    n_jobs = max(1, cpu_count() - 1)
else:
    n_jobs = config['scenarios']['n_jobs']

This gives flexibility while having a sensible default.
```

---

## **Question 4: Validation Backtest**

**Answer: Keep daily 1-step ahead, add CRPS/PIT to existing framework**

**What to tell Cursor:**
```
Keep the daily 1-step ahead backtest structure (it's fine).

The "annual rolling origins" in the docs was for the old system's 1M/3M/12M 
point forecasts. Our new system does path simulation, so the validation approach is:

1. Keep existing structure:
   - Test period: 2022-2025 (or from test_start in config)
   - Daily 1-step ahead forecasts
   - Generate 1000 paths for each day
   - Compare realized next-day yield to forecast distribution

2. ADD to existing metrics:
   - Compute CRPS for each forecast
   - Compute PIT for each forecast
   - Keep coverage, DV01 errors, cosine similarity

3. Summary should show:
   - Coverage: XX% (existing)
   - CRPS: XX bps (NEW)
   - Average PIT: 0.XX (NEW - should be ~0.5)
   - DV01 MAE: XX (existing)
   - Cosine similarity: 0.XXXX (existing)

Do NOT change to annual origins - keep daily 1-step ahead.
Just enhance with CRPS and PIT metrics.
```

---

## 📋 **COPY THIS TO CURSOR:**
```
ANSWERS TO YOUR QUESTIONS:

1. SPREAD DATA: 
   Answer (b) - No direct spread columns, but can compute from RatesRegimes.parquet.
   See docs/RatesRegimes_DataDictionary.md for available data.
   
   Compute spreads as:
   - IGSpr_5 = IGUUAC05_BVLI_Index - USGG5YR_Index
   - AgencySpr_5 = BVCSUG05_BVLI_Index - USGG5YR_Index  
   - MuniSpr_5 = BVMB5Y_Index - USGG5YR_Index
   (Similar for other maturities where both exist)
   
   Implement SpreadEngine with graceful degradation if spreads can't be computed.

2. REGIME FEATURES:
   Answer (a) - Replace with factor changes only.
   
   Remove all ['MOVE_z', 'VIX_z', 'OIS_slope', 'Fed_vs_OIS', 'Inflation_mem'] features.
   New StickyHMM uses ONLY factor changes [ΔL, ΔS, ΔC] from afns_model.factor_changes.

3. PARALLELIZATION:
   Answer (a) - n_jobs = -1 (all CPUs minus 1), but make configurable.
   
   In config.yaml: scenarios.n_jobs: -1
   In code: n_jobs = max(1, cpu_count() - 1) if config == -1 else config value

4. VALIDATION BACKTEST:
   Answer: Keep daily 1-step ahead, ADD CRPS/PIT to existing tests.
   
   Do NOT change to annual rolling origins.
   Keep existing daily backtest structure and add CRPS/PIT metrics to each forecast.
   
   Summary should include:
   - Coverage (existing)
   - CRPS in bps (NEW)
   - Average PIT ~0.5 (NEW)  
   - DV01 errors (existing)
   - Cosine similarity (existing)

PROCEED WITH PROMPT 1 (VAR + Kalman filter implementation).



