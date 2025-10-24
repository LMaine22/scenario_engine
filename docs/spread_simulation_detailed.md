# Spread Simulation: Detailed Implementation Guide

## Overview

This document provides a comprehensive explanation of how credit spreads are simulated in the AFNS codebase. Spreads follow a regime-dependent AR(1) model with factor loadings and Student-t errors, capturing both mean-reverting dynamics and heavy-tailed shocks.

---

## 1. Spread Model Structure

### 1.1 AR(1) Model Formula

Credit spreads are modeled in **log space** using an autoregressive process with factor loadings:

**Core Equation:**
```
z_{t+1} = α + φ z_t + β^L ΔL_t + β^S ΔS_t + β^C ΔC_t + ε_t

where:
  z_t = signed log transform of spread at time t
  α = intercept (drift)
  φ = AR(1) coefficient (persistence)
  β^L, β^S, β^C = factor loadings (sensitivity to Level, Slope, Curvature changes)
  ΔL_t, ΔS_t, ΔC_t = factor changes from t-1 to t
  ε_t ~ t_ν(0, σ²) = Student-t error with ν degrees of freedom
```

**Code Location:** `src/afns/spreads.py`, lines 86-91

```python
def _fit_ar_model(self, z, delta_factors, regime_weights):
    """
    Fit AR(1) model with factor loadings for one sector/maturity/regime.
    
    z_{t+1} = α + φ z_t + β^L ΔL + β^S ΔS + β^C ΔC + ε
    """
```

### 1.2 What is z_t? (Signed Log Transform)

To handle both positive and negative spreads while stabilizing variance, spreads are transformed using a **signed logarithm**:

**Forward Transform:**
```
z = sign(spread) * log(|spread| + ε)

where:
  sign(spread) = +1 if spread ≥ 0, -1 if spread < 0
  |spread| = absolute value
  ε = floor parameter (default 1.0 bps)
```

**Code Location:** `src/afns/spreads.py`, lines 47-65

```python
def _transform_to_log_space(self, spreads):
    """
    Transform spreads to log space with floor.
    Uses signed log to handle negative spreads.
    
    Parameters
    ----------
    spreads : np.array or pd.Series
        Spread values in bps
        
    Returns
    -------
    np.array
        Log-transformed spreads
    """
    # Use signed log to handle negative spreads
    abs_spreads = np.abs(spreads)
    sign = np.sign(spreads)
    return sign * np.log(abs_spreads + self.log_floor_epsilon)
```

**Inverse Transform:**
```
spread = sign(z) * (exp(|z|) - ε)
```

**Code Location:** `src/afns/spreads.py`, lines 67-84

```python
def _transform_from_log_space(self, log_spreads):
    """
    Transform from log space back to spreads.
    Inverse of signed log transform.
    
    Parameters
    ----------
    log_spreads : np.array
        Log-transformed spreads
        
    Returns
    -------
    np.array
        Spreads in bps
    """
    sign = np.sign(log_spreads)
    abs_log = np.abs(log_spreads)
    return sign * (np.exp(abs_log) - self.log_floor_epsilon)
```

**Why Signed Log?**
- **Stabilizes variance:** Large spreads have proportional noise
- **Handles negatives:** Preserves sign while taking log of magnitude
- **Prevents zero:** Floor parameter ε = 1.0 bps ensures log(0) never occurs
- **Mean-reverting in log space:** More realistic than levels

**Example:**
```
spread = 100 bps → z = log(100 + 1) ≈ 4.62
spread = 500 bps → z = log(500 + 1) ≈ 6.22
spread = -50 bps → z = -log(50 + 1) ≈ -3.93
```

### 1.3 Parameter Estimation by Regime

Parameters are estimated separately **for each regime** using **weighted least squares**:

**Regression Setup:**

**Code Location:** `src/afns/spreads.py`, lines 106-128

```python
T = len(z) - 1

# Build regression matrices
y = z[1:]  # z_{t+1}
X = np.column_stack([
    np.ones(T),  # intercept
    z[:-1],  # z_t
    delta_factors[:-1, 0],  # ΔL
    delta_factors[:-1, 1],  # ΔS
    delta_factors[:-1, 2]   # ΔC
])

# Weighted least squares (weight by regime probability)
W = np.diag(regime_weights[1:])  # Skip first obs

# Weighted regression: (X'WX)^{-1} X'Wy
try:
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    beta = np.linalg.solve(XtWX + np.eye(5) * 1e-6, XtWy)  # Add ridge for stability
except:
    # Fallback to OLS if weighted fails
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
```

**Weight Matrix W:**
- Diagonal matrix with regime posterior probabilities
- W[t,t] = P(regime = k | data) at time t
- Observations likely in regime k get higher weight

**Why Weighted Least Squares?**
- Each regime may have different dynamics (calm vs. crisis)
- Weighting by regime probability focuses estimation on relevant periods
- Produces regime-specific parameters even though regimes overlap

**Parameter Extraction:**

**Code Location:** `src/afns/spreads.py`, lines 130-136

```python
# Extract parameters
alpha = beta[0]
phi = beta[1]
beta_L = beta[2]
beta_S = beta[3]
beta_C = beta[4]
```

**Ridge Regularization:**
- Small diagonal term `1e-6 * I` added to X'WX
- Ensures numerical stability when correlations are high
- Prevents singular matrix errors

---

## 2. Student-t Errors

### 2.1 Degrees of Freedom ν

The residuals are modeled as **Student-t distributed** rather than Gaussian to capture **heavy tails** (extreme widening events).

**Estimation Process:**

**Code Location:** `src/afns/spreads.py`, lines 137-156

```python
# Compute residuals
y_fitted = X @ beta
residuals = y - y_fitted

# Fit t-distribution to weighted residuals
weighted_residuals = residuals * np.sqrt(regime_weights[1:])
weighted_residuals = weighted_residuals[weighted_residuals != 0]  # Remove zeros

if len(weighted_residuals) > 10:
    try:
        # Fit t-distribution
        nu, loc, scale = stats.t.fit(weighted_residuals)
        nu = max(3.0, min(nu, 30.0))  # Constrain df between 3 and 30
        scale = max(scale, 1e-4)  # Ensure positive scale
    except:
        nu = 5.0
        scale = max(np.std(weighted_residuals), 1e-4)
else:
    nu = 5.0
    scale = max(np.std(residuals) if len(residuals) > 0 else 1.0, 1e-4)
```

**Method:**
1. Compute residuals: `ε_t = z_{t+1} - (α + φ z_t + β^L ΔL + ...)`
2. Weight residuals by regime probabilities
3. Fit Student-t distribution via **maximum likelihood** (`scipy.stats.t.fit()`)
4. Extract degrees of freedom `ν` and scale `σ`

**Constraints on ν:**
- **Lower bound:** ν ≥ 3.0 (ensures finite variance)
- **Upper bound:** ν ≤ 30.0 (prevents near-Gaussian, keeps fat tails)
- **Default:** ν = 5.0 if fitting fails or insufficient data

**Typical Values:**
- **Calm regime:** ν ≈ 10-15 (moderate tails)
- **Volatile regime:** ν ≈ 3-5 (heavy tails)
- **Gaussian limit:** ν → ∞ (no implementation uses this)

**Why Student-t?**
- Captures **tail risk** better than Gaussian
- Allows for **occasional large shocks** (spread blow-outs)
- Empirically fits credit spread changes well
- ν is **data-driven** (estimated, not hardcoded)

### 2.2 Drawing Student-t Errors in Simulation

During path simulation, Student-t shocks are drawn at each time step:

**Code Location:** `src/afns/spreads.py`, lines 305-317

```python
sigma = params.get('sigma', 0.1)
if not np.isfinite(sigma) or sigma <= 0:
    sigma = 0.1
scale = max(float(sigma), 1e-6)

nu = params.get('nu', 5.0)
if not np.isfinite(nu) or nu <= 0:
    nu = 5.0
nu = max(float(nu), 2.1)

shock = stats.t.rvs(nu, loc=0, scale=scale)
z_path[t] = z_mean + shock
```

**Function:** `scipy.stats.t.rvs(df, loc, scale)`
- `df = nu`: Degrees of freedom
- `loc = 0`: Mean (centered)
- `scale = σ`: Scale parameter (analogous to standard deviation)

**Distribution:**
```
ε_t ~ t_ν(0, σ²)

PDF: p(ε) ∝ [1 + ε²/(ν σ²)]^{-(ν+1)/2}
```

**Tail Behavior:**
- For ν = 5: P(|ε| > 3σ) ≈ 1.5% (vs. 0.3% for Gaussian)
- Tails decay as power law: p(ε) ~ |ε|^{-(ν+1)} for large |ε|
- Produces occasional **extreme widening events**

### 2.3 Safety Checks

**Code Location:** `src/afns/spreads.py`, lines 227-233

```python
if params['sigma'] <= 0 or not np.isfinite(params['sigma']):
    logger.warning("    Regime %d: sigma=%s invalid, setting to 0.01", k, params['sigma'])
    params['sigma'] = 0.01

if params['nu'] <= 0 or not np.isfinite(params['nu']):
    logger.warning("    Regime %d: nu=%s invalid, setting to 5.0", k, params['nu'])
    params['nu'] = 5.0
```

**Fallback Values:**
- σ = 0.01 if invalid (ensures minimal but positive noise)
- ν = 5.0 if invalid (moderate fat tails)

---

## 3. Regime Dependence

### 3.1 Parameter Storage Structure

**All parameters differ by regime.** The model stores a complete set of parameters for each **(sector, maturity, regime)** combination:

**Code Location:** `src/afns/spreads.py`, lines 39-40, 235

```python
# Model parameters: dict[(sector, maturity, regime)] -> params
self.params = {}

# Later in fitting:
self.params[(sector, mat, k)] = params
```

**Dictionary Key:** `(sector, maturity, regime)` tuple

**Example:**
```python
self.params[('IG', 5, 0)] = {
    'alpha': 0.01,
    'phi': 0.95,
    'beta_L': -0.02,
    'beta_S': 0.05,
    'beta_C': 0.03,
    'nu': 12.0,
    'sigma': 0.08,
    'residuals': [...]
}

self.params[('IG', 5, 1)] = {
    'alpha': -0.05,
    'phi': 0.90,
    'beta_L': -0.08,
    'beta_S': 0.15,
    'beta_C': 0.10,
    'nu': 4.5,
    'sigma': 0.25,
    'residuals': [...]
}
```

### 3.2 Do Parameters Differ by Regime?

**Yes, all parameters are regime-specific:**

| Parameter | Regime 0 (Calm) | Regime 1 (Volatile) | Interpretation |
|-----------|-----------------|---------------------|----------------|
| `α` (intercept) | ~0 | Negative | Volatile regimes may have spread compression |
| `φ` (AR coefficient) | ~0.95 | ~0.85 | Lower persistence in volatile regimes |
| `β^L` (Level loading) | -0.02 | -0.10 | Stronger negative correlation in stress |
| `β^S` (Slope loading) | 0.05 | 0.20 | Flattening curve widens spreads more in stress |
| `β^C` (Curvature loading) | 0.03 | 0.08 | Higher sensitivity in volatile regimes |
| `ν` (df) | 12 | 4 | **Much heavier tails** in volatile regimes |
| `σ` (scale) | 0.08 | 0.30 | **Much higher volatility** in volatile regimes |

**Key Insights:**
- **Persistence (φ):** Decreases in volatile regimes (faster mean-reversion)
- **Factor sensitivity (β):** Increases in magnitude during stress
- **Tail fatness (ν):** Lower in volatile regimes (more extreme events)
- **Volatility (σ):** Substantially higher in volatile regimes

### 3.3 Fitting Loop

**Code Location:** `src/afns/spreads.py`, lines 195-240

```python
for sector, mat in self.buckets:
    col = f'{sector}Spr_{mat}'
    # ... load data ...
    
    z = self._transform_to_log_space(series.values)
    
    for k in range(n_regimes):
        regime_col = f'Regime_{k}'
        regime_weights = regime_probs[regime_col].values
        
        params = self._fit_ar_model(z, delta_factors, regime_weights)
        
        # Safety checks
        if params['sigma'] <= 0 or not np.isfinite(params['sigma']):
            params['sigma'] = 0.01
        
        if params['nu'] <= 0 or not np.isfinite(params['nu']):
            params['nu'] = 5.0
        
        self.params[(sector, mat, k)] = params
        
        logger.info(
            "    Regime %d: φ=%.3f, β^L=%.3f, β^S=%.3f, β^C=%.3f, ν=%.1f, σ=%.3f",
            k, params['phi'], params['beta_L'], params['beta_S'], params['beta_C'], 
            params['nu'], params['sigma']
        )
```

**Outer Loop:** Iterate over all spread buckets
**Inner Loop:** Fit separate model for each regime k

---

## 4. Integration in Path Simulation

### 4.1 How `simulate_spreads()` Works

**High-Level Flow:**

1. **Initialize:** Transform initial spreads to log space
2. **Loop over time:** For each day t=1 to T:
   - Determine current regime
   - Retrieve regime-specific parameters
   - Compute expected z_t from AR(1) equation
   - Add Student-t shock
   - Store result
3. **Transform back:** Convert log-space path to spreads in bps

**Code Location:** `src/afns/spreads.py`, lines 248-321

### 4.2 Inputs and Outputs

**Function Signature:**

```python
def simulate_spreads(self, factors_path, regime_path, initial_spreads, random_seed=None):
    """
    Simulate spread paths given factor and regime scenarios.
    
    Parameters
    ----------
    factors_path : np.array
        Factor path (T x 3): [L, S, C]
    regime_path : np.array
        Regime path (T,)
    initial_spreads : dict
        Initial spread values {(sector, mat): value}
    random_seed : int, optional
        Random seed for this path
        
    Returns
    -------
    dict
        Simulated spreads {(sector, mat): np.array of length T}
    """
```

**Inputs:**
- `factors_path`: T×3 array of factor levels [L, S, C] at each time
- `regime_path`: Length-T array of regime indices (e.g., [0, 0, 1, 1, 0, ...])
- `initial_spreads`: Dictionary mapping (sector, mat) → initial spread value in bps
- `random_seed`: Optional seed for reproducibility

**Outputs:**
- Dictionary mapping (sector, mat) → length-T array of simulated spreads in bps

**Example:**
```python
spreads_sim = spread_engine.simulate_spreads(
    factors_path=np.array([[4.0, 0.5, 0.2], [4.1, 0.4, 0.3], ...]),  # T=126 rows
    regime_path=np.array([0, 0, 1, 1, 0, ...]),  # T=126 elements
    initial_spreads={('IG', 5): 80.0, ('HY', 5): 400.0},
    random_seed=42
)

# Output:
# spreads_sim = {
#     ('IG', 5): [80.0, 82.1, 85.3, 88.2, ...],  # 126 values
#     ('HY', 5): [400.0, 405.2, 420.5, 415.3, ...]  # 126 values
# }
```

### 4.3 Full Code Implementation

**Code Location:** `src/afns/spreads.py`, lines 248-321

```python
def simulate_spreads(self, factors_path, regime_path, initial_spreads, random_seed=None):
    """
    Simulate spread paths given factor and regime scenarios.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    T = len(factors_path)
    delta_factors = np.diff(factors_path, axis=0, prepend=factors_path[0:1, :])
    
    spreads_sim = {}
    
    for sector, mat in self.buckets:
        key = (sector, mat)
        if key not in initial_spreads:
            continue
        
        # Initialize log-space spread
        z = self._transform_to_log_space(np.array([initial_spreads[key]]))
        z_path = np.zeros(T)
        z_path[0] = z[0]
        
        # Simulate forward
        for t in range(1, T):
            regime = int(regime_path[t])
            
            # Handle missing regime parameters
            if (sector, mat, regime) not in self.params:
                available_regimes = [k for k in range(10) if (sector, mat, k) in self.params]
                if available_regimes:
                    regime = available_regimes[0]
                else:
                    z_path[t] = z_path[t-1]  # Carry forward
                    continue
            
            params = self.params[(sector, mat, regime)]
            
            # Compute AR(1) mean
            z_mean = (
                params['alpha'] +
                params['phi'] * z_path[t-1] +
                params['beta_L'] * delta_factors[t, 0] +
                params['beta_S'] * delta_factors[t, 1] +
                params['beta_C'] * delta_factors[t, 2]
            )
            
            # Safety checks on sigma
            sigma = params.get('sigma', 0.1)
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = 0.1
            scale = max(float(sigma), 1e-6)
            
            # Safety checks on nu
            nu = params.get('nu', 5.0)
            if not np.isfinite(nu) or nu <= 0:
                nu = 5.0
            nu = max(float(nu), 2.1)
            
            # Draw Student-t shock
            shock = stats.t.rvs(nu, loc=0, scale=scale)
            z_path[t] = z_mean + shock
        
        # Transform back to bps
        spreads_sim[key] = self._transform_from_log_space(z_path)
    
    return spreads_sim
```

### 4.4 Key Features

**Factor Changes Computation:**
```python
delta_factors = np.diff(factors_path, axis=0, prepend=factors_path[0:1, :])
```
- Computes ΔL, ΔS, ΔC at each time step
- `prepend` ensures first element is zero change (initialization)

**Regime Fallback:**
```python
if (sector, mat, regime) not in self.params:
    available_regimes = [k for k in range(10) if (sector, mat, k) in self.params]
    if available_regimes:
        regime = available_regimes[0]
    else:
        z_path[t] = z_path[t-1]  # Carry forward
        continue
```
- If regime k is not fitted for a bucket (insufficient data), use another regime
- If no regimes available, carry forward previous value (no change)

**Safety Checks:**
- Ensures σ > 0 and finite
- Ensures ν ≥ 2.1 and finite (minimum for finite variance)
- Prevents NaN propagation

---

## 5. Buckets

### 5.1 What Spread Buckets Are Modeled?

Buckets are **(sector, maturity)** combinations representing specific spread curves:

**Typical Buckets:**
- **Investment Grade (IG):** (IG, 1), (IG, 5), (IG, 10), (IG, 30)
- **High Yield (HY):** (HY, 1), (HY, 5), (HY, 10)
- **Financials (FIN):** (FIN, 1), (FIN, 5), (FIN, 10)
- **Utilities (UTL):** (UTL, 1), (UTL, 5), (UTL, 10)
- **Industrials (IND):** (IND, 1), (IND, 5), (IND, 10)

**Code Location:** `src/afns/spreads.py`, line 33

```python
def __init__(self, buckets=None, log_floor_epsilon=1.0, enable_jumps=False, random_state=42):
    self.buckets = buckets or []
```

**Bucket Specification:**
Passed as list of tuples:
```python
buckets = [
    ('IG', 1), ('IG', 5), ('IG', 10), ('IG', 30),
    ('HY', 5), ('HY', 10)
]

spread_engine = SpreadEngine(buckets=buckets)
```

**Column Naming Convention:**
- DataFrame column: `{sector}Spr_{maturity}`
- Example: `IGSpr_5` for IG 5-year spreads
- Automatically parsed during fitting

### 5.2 Coverage and Missing Data

**Data Availability Check:**

**Code Location:** `src/afns/spreads.py`, lines 195-209

```python
for sector, mat in self.buckets:
    col = f'{sector}Spr_{mat}'
    if col not in spreads_df.columns:
        logger.warning("  Spread column %s not found, skipping", col)
        continue
    
    series = pd.to_numeric(spreads_df[col], errors='coerce').dropna()
    common_index = series.index.intersection(factor_changes_df.index).intersection(regime_probs_df.index)
    series = series.loc[common_index].sort_index()
    
    if len(series) < 30:
        logger.warning("  %s has insufficient observations (%d), skipping", col, len(series))
        continue
```

**Handling:**
1. **Missing column:** Skip bucket with warning
2. **Insufficient data:** Skip if < 30 observations
3. **Index alignment:** Use intersection of spread, factor, and regime indices
4. **Coercion:** Convert to numeric, replace invalid values with NaN

**Bucket Windows Tracking:**

**Code Location:** `src/afns/spreads.py`, lines 211-215

```python
self.bucket_windows[(sector, mat)] = {
    'start': common_index.min(),
    'end': common_index.max(),
    'observations': int(len(common_index))
}
```

Stores metadata about data availability:
- Start date of available data
- End date of available data
- Total number of observations

**Usage:**
```python
windows = spread_engine.get_bucket_windows()
print(windows[('IG', 5)])
# {'start': Timestamp('2010-01-04'), 'end': Timestamp('2024-12-31'), 'observations': 3780}
```

### 5.3 Partial Coverage Example

**Scenario:** HY data starts later than IG data

```
Timeline:   2010        2015        2020        2025
IG data:    |==================================|
HY data:                |====================|
Factors:    |==================================|
Regimes:    |==================================|

Result:
- IG bucket: Fitted on 2010-2025 (full period)
- HY bucket: Fitted on 2015-2025 (partial period)
- Both buckets usable in simulation
```

**Code handles this automatically** via index intersection.

---

## 6. Integration in Simulator

### 6.1 Call from Simulator

**Code Location:** `src/afns/simulator.py`, lines 143-149

```python
# Simulate spreads
spreads_sim = self.spread_engine.simulate_spreads(
    factors_path, regime_path, current_spreads, random_seed=path_seed
)

for (sector, mat), spread_path in spreads_sim.items():
    path_data[f'{sector}Spr_{mat}'] = spread_path
```

**Context:**
- Called within `_simulate_single_path()` function
- After factors and regimes have been simulated
- Uses same random seed for reproducibility

### 6.2 Data Flow

```
1. AFNS Model → factors_path (T×3 array)
2. HMM Model → regime_path (T array)
3. SpreadEngine → spreads_sim {(sector, mat): spread_path}
4. Combine into path_data DataFrame
```

**Full Integration:**

```python
# From simulator.py _simulate_single_path():

# Step 1: Simulate factors
for t in range(1, horizon_days + 1):
    s_t = np.random.choice(len(gamma_T), p=self.hmm_model.transition_matrix[s_t, :])
    eta = np.random.multivariate_normal(np.zeros(3), self.afns_model.Q * self.q_scaler)
    f_t = self.afns_model.A @ f_t + self.afns_model.b + eta
    path_data['regime'].append(s_t)
    path_data['L'].append(f_t[0])
    path_data['S'].append(f_t[1])
    path_data['C'].append(f_t[2])

# Step 2: Convert to arrays
factors_path = np.column_stack([path_data['L'], path_data['S'], path_data['C']])
regime_path = np.array(path_data['regime'])

# Step 3: Simulate spreads
spreads_sim = self.spread_engine.simulate_spreads(
    factors_path, regime_path, current_spreads, random_seed=path_seed
)

# Step 4: Store in path_data
for (sector, mat), spread_path in spreads_sim.items():
    path_data[f'{sector}Spr_{mat}'] = spread_path
```

---

## 7. Parameter Interpretation

### 7.1 AR Coefficient φ

**Range:** Typically 0.85 - 0.98

**Interpretation:**
- φ = 0.95: Spread has 5% mean reversion per day (half-life ~14 days)
- φ = 0.90: Spread has 10% mean reversion per day (half-life ~7 days)
- φ = 1.00: Random walk (no mean reversion)

**Formula:**
```
Half-life = -log(2) / log(φ)
```

**Example:**
```python
phi = 0.95
half_life = -np.log(2) / np.log(0.95)  # ≈ 13.5 days
```

### 7.2 Factor Loadings β^L, β^S, β^C

**Typical Signs:**
- **β^L < 0:** Rising rates (higher level) → spreads compress (negative correlation)
- **β^S > 0:** Steepening curve → spreads widen (risk-on)
- **β^C ≈ 0:** Curvature has weak direct effect on spreads

**Example (IG 5yr, Calm Regime):**
```
β^L = -0.03: If Level rises 10 bps, spread tightens ~0.3 bps (in log space)
β^S = 0.08: If Slope increases 10 bps, spread widens ~0.8 bps (in log space)
β^C = 0.02: If Curvature increases 10 bps, spread widens ~0.2 bps (in log space)
```

### 7.3 Student-t Parameters

**ν (Degrees of Freedom):**
- ν = 3: Very heavy tails (similar to Cauchy for extreme values)
- ν = 5: Heavy tails (standard for financial data)
- ν = 10: Moderate tails
- ν = 30: Near-Gaussian

**σ (Scale):**
- Analogous to standard deviation
- σ = 0.1: Low spread volatility (~10% daily moves in log space)
- σ = 0.3: High spread volatility (~30% daily moves in log space)

**Tail Probability:**
```
P(|shock| > 3σ) ≈ 1.5% for ν=5 vs. 0.3% for Gaussian
```

---

## 8. Example Output

### 8.1 Fitted Parameters Log

**Code Location:** `src/afns/spreads.py`, lines 237-240

```
  Fitting IGSpr_5 [2010-01-04 → 2024-12-31 | 3780 obs]
    Regime 0: φ=0.960, β^L=-0.025, β^S=0.078, β^C=0.031, ν=11.2, σ=0.085
    Regime 1: φ=0.920, β^L=-0.085, β^S=0.165, β^C=0.095, ν=4.8, σ=0.235
    Regime 2: φ=0.880, β^L=-0.120, β^S=0.220, β^C=0.140, ν=3.5, σ=0.380
```

**Interpretation:**
- **Regime 0 (Calm):** High persistence, low sensitivity, moderate tails
- **Regime 1 (Volatile):** Lower persistence, higher sensitivity, heavy tails
- **Regime 2 (Crisis):** Lowest persistence, highest sensitivity, very heavy tails

### 8.2 Simulated Path Example

**Input:**
```python
factors_path = [
    [4.0, 0.5, 0.2],  # t=0
    [4.1, 0.4, 0.3],  # t=1: ΔL=+0.1, ΔS=-0.1, ΔC=+0.1
    [4.2, 0.3, 0.4],  # t=2: ΔL=+0.1, ΔS=-0.1, ΔC=+0.1
]
regime_path = [0, 0, 1]
initial_spreads = {('IG', 5): 80.0}
```

**Simulation (t=1):**
```
z[0] = log(80 + 1) = 4.394

z_mean[1] = α + φ*z[0] + β^L*ΔL + β^S*ΔS + β^C*ΔC
          = 0.01 + 0.96*4.394 + (-0.025)*0.1 + 0.078*(-0.1) + 0.031*0.1
          = 0.01 + 4.218 - 0.0025 - 0.0078 + 0.0031
          = 4.221

shock ~ t(ν=11.2, σ=0.085)  (draw random value, e.g., -0.05)
z[1] = z_mean[1] + shock = 4.221 - 0.05 = 4.171

spread[1] = exp(4.171) - 1 = 63.8 bps
```

**Simulation (t=2, regime switches to 1):**
```
z[1] = 4.171

z_mean[2] = 0.01 + 0.92*4.171 + (-0.085)*0.1 + 0.165*(-0.1) + 0.095*0.1
          = 0.01 + 3.837 - 0.0085 - 0.0165 + 0.0095
          = 3.831

shock ~ t(ν=4.8, σ=0.235)  (draw random value, e.g., +0.15)
z[2] = z_mean[2] + shock = 3.831 + 0.15 = 3.981

spread[2] = exp(3.981) - 1 = 52.6 bps
```

**Result:** Spread path = [80.0, 63.8, 52.6] bps (widening then tightening)

---

## 9. Usage Example

```python
from src.afns.spreads import SpreadEngine

# Initialize
buckets = [
    ('IG', 1), ('IG', 5), ('IG', 10),
    ('HY', 5), ('HY', 10)
]

spread_engine = SpreadEngine(
    buckets=buckets,
    log_floor_epsilon=1.0,
    random_state=42
)

# Fit models
spread_engine.fit(
    spreads_df=spreads_df,  # Columns: IGSpr_1, IGSpr_5, etc.
    factor_changes_df=afns.factor_changes,
    regime_probs_df=hmm.regime_probs
)

# Access fitted parameters
params_ig5_regime0 = spread_engine.params[('IG', 5, 0)]
print(f"φ = {params_ig5_regime0['phi']:.3f}")
print(f"ν = {params_ig5_regime0['nu']:.1f}")

# Simulate spreads
spreads_sim = spread_engine.simulate_spreads(
    factors_path=factors_path,  # From AFNS simulation
    regime_path=regime_path,    # From HMM simulation
    initial_spreads={('IG', 5): 80.0, ('HY', 5): 400.0},
    random_seed=42
)

# Access results
ig5_path = spreads_sim[('IG', 5)]  # Array of length T
print(f"IG 5yr path: {ig5_path[:10]}")
```

---

## 10. References

**Files:**
- `src/afns/spreads.py`: Full SpreadEngine implementation
- `src/afns/simulator.py`: Integration in path simulation (lines 143-149)

**Key Code Sections:**
- Lines 47-84: Signed log transforms
- Lines 86-167: AR(1) model fitting with WLS
- Lines 141-156: Student-t distribution fitting
- Lines 169-246: Main `fit()` function
- Lines 248-321: `simulate_spreads()` function

**Dependencies:**
- `scipy.stats.t`: Student-t distribution fitting and sampling
- `numpy`: Matrix operations and linear algebra
- `pandas`: Data alignment and handling

---

This implementation provides a sophisticated spread model that:
1. **Captures regime shifts** in spread dynamics
2. **Links spreads to rates** via factor loadings
3. **Models heavy tails** using Student-t errors
4. **Handles missing data** gracefully
5. **Integrates seamlessly** with factor and regime simulation

