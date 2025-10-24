<!-- 2a6d1b40-2e5a-489b-b48b-bbf3e60fda7f 4d176348-e58a-4b72-b03b-d5aa2c276ab4 -->
# Complete Scenario Engine Upgrade Plan

## Overview

Replace static factor extraction and instant-horizon scenarios with dynamic state-space models and multi-day path simulation. Add proper probabilistic validation metrics.

---

## Phase 1: AFNS VAR(1) + Kalman Filter (src/afns.py)

### Current State

- Uses OLS to extract factors: `fit_factors()` does least squares on loadings
- No factor dynamics (A, b, Q matrices)
- Static covariance estimation

### Changes Required

**1. Add new methods to AFNSModel class:**

```python
def initialize_factors_dns(self, yields_df):
    """Initialize factors using DNS approximation (lines 98-134 of docs)"""
    # L = (UST_2 + UST_10) / 2
    # S = UST_10 - UST_2  
    # C = 2*UST_5 - UST_2 - UST_10
    
def _kalman_filter(self, yields, A, b, Q):
    """Run Kalman filter (lines 136-196 of docs)"""
    # Use filterpy.kalman.KalmanFilter
    # Fixed R = 4.0 bps² measurement noise
    # Returns: filtered_factors, log_likelihood
    
def _objective(self, params, yields):
    """Negative log-likelihood for optimization (lines 198-361 of docs)"""
    # Unpack params: A (9), b (3), Q_diag (3) = 15 total
    # Stability constraint: |eigenvalues(A)| < 0.999
    # Ensure Q positive via np.abs()
    # Return: -log_likelihood
```

**2. Replace `fit_factors()` with `fit()` method:**

```python
def fit(self, yields_df, max_iter=100):
    """Main estimation via BFGS optimization"""
    # 1. Initialize DNS factors for parameter guesses
    # 2. Set initial A = 0.95*I, b = small drift, Q = DNS covariance
    # 3. Optimize using scipy.optimize.minimize(method='BFGS')
    # 4. Store: self.A, self.b, self.Q
    # 5. Run final Kalman filter to get filtered factors
    # 6. Compute self.factors (DataFrame with L, S, C)
    # 7. Compute self.factor_changes = factors.diff().dropna()
    # 8. Compute RMSE by maturity
    # 9. Return: dict with A, b, Q, factors, factor_changes, log_likelihood, rmse_by_maturity
```

**3. Add dependencies:**

```python
from scipy.optimize import minimize
from filterpy.kalman import KalmanFilter
```

**4. Update `__init__()` to accept `measurement_noise_R`:**

```python
def __init__(self, maturities, lambda_param=0.0609, measurement_noise_R=4.0):
    self.R = np.eye(len(maturities)) * measurement_noise_R
```

**5. Keep existing helper methods:**

- `compute_loadings()` - Nelson-Siegel loading functions
- `_compute_loadings()` - Build loading matrix H
- `reconstruct_yields()` - Convert factors to yields

**Success Criteria:**

- Overall RMSE < 10 bps (target: 3-5 bps)
- `factors` DataFrame has columns ['L', 'S', 'C']
- `factor_changes` is first difference of factors

---

## Phase 2: Sticky HMM (src/regime.py)

### Current State

- Uses GaussianMixture on 5D feature vector `[MOVE_z, VIX_z, OIS_slope, Fed_vs_OIS, Inflation_mem]`
- Static regime classification (no transitions)

### Changes Required

**1. Create new StickyHMM class (REPLACE RegimeClassifier entirely):**

```python
class StickyHMM:
    def __init__(self, n_states_range=[2,3,4,5], sticky_alpha=10.0, random_state=42):
        """Initialize with BIC-based model selection"""
        
    def _apply_sticky_prior(self, trans_matrix, alpha):
        """Add α to diagonal, renormalize (lines 49-75 of docs)"""
        # π[k,k] += α
        # Renormalize rows to sum to 1
        
    def _compute_bic(self, model, X):
        """BIC = -2*log_likelihood + n_params*log(n) (lines 77-100 of docs)"""
        # n_params = 3K + 6K + K² - K
        
    def _compute_regime_duration(self, trans_matrix):
        """Expected duration = 1 / (1 - π[k,k]) (lines 102-117 of docs)"""
        
    def fit(self, factor_changes):
        """Fit HMM on factor changes (lines 119-231 of docs)"""
        # Input: factor_changes DataFrame (Δf_t) NOT levels
        # Loop over K in [2,3,4,5]
        # For each K: Fit GaussianHMM with covariance_type='full', n_iter=100
        # Apply sticky prior to transmat_
        # Compute BIC
        # Select K with lowest BIC
        # Store: n_states, model, regime_means, regime_covs, transition_matrix
        # Compute regime_probs = model.predict_proba(X) as DataFrame
        # Return: dict with n_states, transition_matrix, regime_means, regime_covs, regime_probs
```

**2. Add helper methods:**

```python
def predict_regime_probs(self, factor_changes_recent):
    """Get current regime distribution (lines 233-262 of docs)"""
    
def get_regime_timeline(self):
    """Extract historical regime probabilities"""
    
def get_most_likely_regimes(self):
    """Get most likely regime at each time"""
```

**3. Update `fit_regime_model()` function:**

- Remove `compute_regime_statistics()` (uses old features)
- Remove `estimate_regime_covariances()` (HMM provides these)
- Input: `factor_changes` NOT `regime_features`
- Return: HMM object with transition matrix

**4. Add dependency:**

```python
from hmmlearn import hmm
```

**Success Criteria:**

- Transition matrix diagonal > 0.85 (regime persistence)
- BIC selects K (usually 3 or 4)
- regime_probs is T×K DataFrame

---

## Phase 3: Path Simulator (src/scenarios.py)

### Current State

- Single-step factor shock: `base_factors + random_shock`
- No time evolution, no regime transitions

### Changes Required

**1. Update ScenarioGenerator class:**

```python
class ScenarioGenerator:
    def __init__(self, afns_model, hmm_model, spread_engine=None, n_paths=10000, 
                 horizon_days=126, n_jobs=-1, move_state='neutral', q_scaler=1.0):
        """Initialize with path simulation parameters"""
        self.n_jobs = n_jobs if n_jobs != -1 else max(1, cpu_count() - 1)
```

**2. Add `_simulate_single_path()` method (lines 64-169 of docs):**

```python
def _simulate_single_path(self, path_id, sim_params):
    """Simulate one 126-day path"""
    # Unpack: f_T, P_T, gamma_T, horizon_days, random_seed
    
    # Set unique seed: path_seed = random_seed + path_id
    
    # Draw initial regime from gamma_T
    # Draw initial factors from N(f_T, P_T)
    
    # Loop t=1 to horizon_days:
    #   - Transition regime: s_t ~ Categorical(π[s_{t-1}, :])
    #   - Evolve factors: f_t = A*f_{t-1} + b + η, η ~ N(0, Q*q_scaler)
    #   - Store: t, regime, L, S, C
    
    # Compute yields from factors
    # If spread_engine exists: simulate spreads
    # Return: DataFrame with path_id, t, regime, L, S, C, UST_*, spreads
```

**3. Add parallelization wrapper:**

```python
def _simulate_single_path_wrapper(self, args):
    """Wrapper for multiprocessing.Pool"""
    path_id, sim_params = args
    return self._simulate_single_path(path_id, sim_params)
```

**4. Replace `generate_scenario()` method:**

```python
def generate_scenario(self, current_state, fed_shock_bp, regime):
    """Generate 10K paths with regime transitions"""
    # 1. Get f_T (current factors)
    # 2. Compute P_T via solve_discrete_lyapunov(A, Q)
    # 3. Get gamma_T (current regime distribution)
    # 4. Get MOVE q_scaler (0.8=low, 1.0=neutral, 1.3=high)
    # 5. Apply Fed shock: f_T_shocked = f_T + [shock_L, shock_S, shock_C]
    # 6. Use Pool to parallelize n_paths simulations
    # 7. Combine paths into DataFrame
    # 8. Compute percentiles at horizon: p5, p50, p95
    # 9. Return: paths_df, percentiles_dict
```

**5. Add dependencies:**

```python
from multiprocessing import Pool, cpu_count
from scipy.linalg import solve_discrete_lyapunov
from tqdm import tqdm
```

**Success Criteria:**

- 10,000 paths × 126 days = 1.26M rows in paths_df
- Regimes vary during paths (check path_data['regime'])
- Factors evolve smoothly (no jumps)

---

## Phase 4: Spread Engine (NEW FILE: src/spreads.py)

### Create New File

**1. Compute spreads in data preparation (src/utils.py):**

```python
def compute_spreads(df):
    """Compute credit spreads from RatesRegimes data"""
    spreads = {}
    
    # IG spreads (if available)
    if 'IGUUAC05 BVLI Index' in df.columns and 'USGG5YR Index' in df.columns:
        spreads['IGSpr_5'] = df['IGUUAC05 BVLI Index'] - df['USGG5YR Index']
    # (Similar for 1Y, 7Y, 10Y)
    
    # Agency spreads
    if 'BVCSUG05 BVLI Index' in df.columns:
        spreads['AgencySpr_5'] = df['BVCSUG05 BVLI Index'] - df['USGG5YR Index']
    # (Similar for 1Y, 2Y, 7Y, 10Y, 20Y)
    
    # Muni spreads  
    if 'BVMB5Y Index' in df.columns:
        spreads['MuniSpr_5'] = df['BVMB5Y Index'] - df['USGG5YR Index']
    # (Similar for 2Y, 3Y, 7Y, 10Y, 30Y)
    
    return pd.DataFrame(spreads, index=df.index)
```

**2. Create SpreadEngine class (src/spreads.py):**

```python
class SpreadEngine:
    def __init__(self, buckets=None, log_floor_epsilon=1.0, random_state=42):
        """buckets = [('IG', 5), ('Agency', 10), ('Muni', 30)]"""
        
    def _transform_to_log_space(self, spreads):
        """Signed log: sign(x) * log(|x| + ε)"""
        
    def _transform_from_log_space(self, log_spreads):
        """Inverse: sign(z) * (exp(|z|) - ε)"""
        
    def _fit_ar_model(self, z, delta_factors, regime_weights):
        """Fit AR(1) with Student-t errors (lines 86-167 of docs)"""
        # Regression: z_{t+1} = α + φ*z_t + β^L*ΔL + β^S*ΔS + β^C*ΔC + ε
        # Weighted least squares with regime_weights
        # Fit Student-t to residuals: ν, σ = stats.t.fit(residuals)
        # Constrain: 3 ≤ ν ≤ 30
        # Return: dict with alpha, phi, beta_L, beta_S, beta_C, nu, sigma
        
    def fit(self, spreads_df, factor_changes_df, regime_probs_df):
        """Fit regime-dependent AR models (lines 169-246 of docs)"""
        # For each (sector, mat) bucket:
        #   Transform to log space
        #   For each regime k:
        #     Fit AR model weighted by P(regime=k)
        #     Store params[(sector, mat, k)]
        # Gracefully skip missing buckets
        
    def simulate_spreads(self, factors_path, regime_path, initial_spreads, random_seed=None):
        """Simulate spread paths (lines 248-321 of docs)"""
        # For each bucket:
        #   Initialize z_0 = transform(initial_spread)
        #   For t=1 to T:
        #     Get regime params
        #     Compute AR mean: z_mean = α + φ*z_{t-1} + β·Δf_t
        #     Draw Student-t shock
        #     Update: z_t = z_mean + shock
        #   Transform back to spreads
        # Return: dict {(sector, mat): spread_path}
```

**3. Add to scenarios.py integration:**

```python
# In _simulate_single_path():
if self.spread_engine is not None:
    spreads_sim = self.spread_engine.simulate_spreads(
        factors_path, regime_path, current_spreads, random_seed=path_seed
    )
    for (sector, mat), spread_path in spreads_sim.items():
        path_data[f'{sector}Spr_{mat}'] = spread_path
```

**4. Dependencies:**

```python
from scipy import stats
```

**Success Criteria:**

- System works with or without spread data
- If spreads exist, params stored for each (sector, mat, regime)
- Student-t ν typically 3-15 (heavier tails than Gaussian)

---

## Phase 5: CRPS + PIT Validation (src/validation.py)

### Current State

- Only coverage and RMSE metrics
- No probabilistic calibration checks

### Changes Required

**1. Add new metric methods:**

```python
def _compute_crps(self, forecast_samples, realized):
    """Compute CRPS (lines 45-83 of docs)"""
    # CRPS = E|X - y| - 0.5*E|X - X'|
    # Term1: mean absolute error
    # Term2: pairwise spread (subsample 100 for efficiency)
    # Return: crps (lower is better)
    
def _compute_pit(self, forecast_samples, realized):
    """Compute PIT (lines 85-102 of docs)"""
    # PIT = Empirical CDF(realized)
    # pit = (forecast_samples < realized).mean()
    # Should be ~0.5 if unbiased
```

**2. Update `backtest_scenarios()` to compute new metrics:**

```python
# After generating sim_yields (1000 samples):
forecast_mean = sim_yields[:, j].mean()
rmse = np.sqrt((forecast_mean - realized[j]) ** 2)
mae = np.abs(forecast_mean - realized[j])
crps = self._compute_crps(sim_yields[:, j], realized[j])  # NEW
pit = self._compute_pit(sim_yields[:, j], realized[j])    # NEW
coverage = self._compute_coverage(sim_yields[:, j], realized[j])

# Add to results dict
```

**3. Update summary output:**

```python
# Print average CRPS by tenor
logger.info("CRPS Metrics:")
for tenor in tenors:
    avg_crps = results_df[results_df['tenor']==tenor]['crps'].mean()
    logger.info(f"  {tenor}: {avg_crps:.2f} bps")

# Print PIT calibration
pit_mean = results_df['pit'].mean()
logger.info(f"\nPIT Calibration:")
logger.info(f"  Average PIT: {pit_mean:.3f} (target: 0.50)")
if 0.45 <= pit_mean <= 0.55:
    logger.info("  ✓ Well-calibrated")
else:
    logger.warning("  ⚠ Biased forecast")
```

**4. Update validation_summary.txt output:**

Add sections for CRPS and PIT after existing metrics.

**Success Criteria:**

- CRPS < 20 bps for yields (expect 5-15 bps)
- Average PIT between 0.45-0.55
- Coverage remains 85-95%

---

## Phase 6: Integration + Config (main.py, config.yaml)

### Config Updates

**1. Update config.yaml:**

```yaml
afns:
  lambda: 0.0609
  measurement_noise_R: 4.0  # NEW

regime:
  n_states_range: [2, 3, 4, 5]  # NEW
  sticky_alpha: 10.0             # NEW
  random_state: 42               # NEW

scenarios:
  n_paths: 10000      # NEW (was 1000)
  horizon_days: 126   # NEW
  n_jobs: -1          # NEW
  confidence_levels: [0.05, 0.5, 0.95]

spreads:  # NEW section
  buckets:
    - ['IG', 5]
    - ['IG', 10]
    - ['Agency', 5]
    - ['Agency', 10]
    - ['Muni', 5]
    - ['Muni', 10]
  log_floor_epsilon: 1.0
```

**2. Update main.py orchestration:**

```python
# Step 3: Fit AFNS (UPDATED)
logger.info("[3/8] Fitting AFNS VAR(1) + Kalman filter")
afns_results = afns_model.fit(df[config['data']['tenors']])
df_factors = afns_results['factors']
factor_changes = afns_results['factor_changes']

# Step 4: Fit regime (UPDATED)
logger.info("[4/8] Fitting Sticky HMM")
hmm_model = StickyHMM(
    n_states_range=config['regime']['n_states_range'],
    sticky_alpha=config['regime']['sticky_alpha']
)
hmm_results = hmm_model.fit(factor_changes)

# Step 4.5: Compute spreads and fit spread models (NEW)
logger.info("[5/8] Computing spreads and fitting spread models")
spreads_df = compute_spreads(df)
spread_engine = None
if 'spreads' in config and config['spreads'].get('buckets'):
    try:
        spread_engine = SpreadEngine(buckets=config['spreads']['buckets'])
        spread_engine.fit(spreads_df, factor_changes, hmm_results['regime_probs'])
    except Exception as e:
        logger.warning(f"Spread fitting failed: {e}, continuing without spreads")

# Step 5: Generate scenarios (UPDATED)
logger.info("[6/8] Generating 10K path scenarios")
generator = ScenarioGenerator(
    afns_model, hmm_model, spread_engine,
    n_paths=config['scenarios']['n_paths'],
    horizon_days=config['scenarios']['horizon_days'],
    n_jobs=config['scenarios']['n_jobs']
)
scenarios, paths_df = generator.generate_scenario(...)

# Step 6: Validation (UPDATED - includes CRPS/PIT)
logger.info("[7/8] Running validation with CRPS/PIT")
validation_results = run_validation(...)
```

**3. Update data preparation (src/utils.py):**

Add `compute_spreads()` function to be called after loading data.

**Success Criteria:**

- All components integrate cleanly
- System works with or without spread data
- Validation includes CRPS and PIT
- No crashes or errors

---

## Implementation Order

1. **Phase 1**: AFNS VAR+Kalman (src/afns.py) - Foundation for everything
2. **Phase 2**: Sticky HMM (src/regime.py) - Depends on factor_changes from Phase 1
3. **Phase 5**: CRPS/PIT (src/validation.py) - Can be done in parallel with Phase 3/4
4. **Phase 3**: Path Simulator (src/scenarios.py) - Depends on Phases 1 & 2
5. **Phase 4**: Spread Engine (src/spreads.py) - Optional, depends on Phases 1 & 2
6. **Phase 6**: Integration (main.py, config.yaml) - Final assembly

---

## Testing Strategy

After each phase:

1. Run on small data subset to verify no crashes
2. Check output dimensions and types
3. Verify key metrics (RMSE, coverage, etc.)

Final integration test:

1. Run full pipeline end-to-end
2. Verify outputs in outputs/figures, outputs/results, outputs/scenarios
3. Check validation_summary.txt includes all metrics
4. Confirm 10K paths generate in reasonable time (<5 min on 8-core machine)

### To-dos

- [ ] Implement AFNS VAR(1) + Kalman filter in src/afns.py
- [ ] Replace regime classifier with Sticky HMM in src/regime.py
- [ ] Add CRPS and PIT metrics to src/validation.py
- [ ] Implement multi-step path simulator in src/scenarios.py
- [ ] Create SpreadEngine in src/spreads.py
- [ ] Integrate all components in main.py and update config.yaml
- [ ] Run full pipeline test and verify all outputs