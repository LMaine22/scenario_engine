# Monte Carlo Path Simulator: Detailed Implementation Guide

## Overview

This document provides a comprehensive explanation of the Monte Carlo path simulation implementation used in the AFNS codebase. The simulator generates thousands of scenario paths for yield curves, spreads, and regime dynamics over specified forecast horizons.

---

## 1. Simulation Setup

### 1.1 Number of Paths

**Default:** 10,000 paths per simulation

**Code Location:** `src/afns/simulator.py`, line 38
```python
def __init__(self, afns_model, hmm_model, spread_engine, n_paths=10000,
             horizons={'6m': 126, '12m': 252}, random_seed=42, ...):
    self.n_paths = n_paths
```

**Why 10,000?**
- Sufficient for stable percentile estimates
- Manageable computational cost with parallelization
- Standard in Monte Carlo risk applications

**Customization:** Can be reduced to 1,000 for testing or increased to 50,000 for high-precision tail risk analysis.

### 1.2 Time Horizons

**Standard Horizons:**
- **6-month:** 126 business days
- **12-month:** 252 business days

**Code Location:** `src/afns/simulator.py`, line 38
```python
horizons={'6m': 126, '12m': 252}
```

**Business Days Convention:**
- ~21 business days per month
- 252 business days per year (52 weeks × 5 days - holidays ≈ 252)

**Customization:** Any horizon can be specified, e.g., `{'3m': 63, '18m': 378}`.

### 1.3 Initial State Determination

The simulation starts from the **most recent state** at time T:

**Code Location:** `src/afns/simulator.py`, lines 217-236

```python
# Get current state from AFNS
f_T = self.afns_model.factors.iloc[-1].values  # Most recent factors

# Get steady-state covariance
try:
    from scipy.linalg import solve_discrete_lyapunov
    P_T = solve_discrete_lyapunov(self.afns_model.A, self.afns_model.Q)
except:
    P_T = self.afns_model.Q.copy()

# Get current regime distribution
gamma_T = self.hmm_model.regime_probs.iloc[-1].values

logger.info(f"  Current state:")
logger.info(f"    Factors [L, S, C]: {f_T}")
logger.info(f"    Regime probs: {gamma_T}")
```

**Initial Conditions:**
1. **f_T:** Most recent filtered factors `[L_T, S_T, C_T]`
2. **P_T:** Steady-state covariance from solving the Lyapunov equation:
   ```
   P = A P A' + Q
   ```
3. **gamma_T:** Most recent regime probability distribution `[γ_0, γ_1, ..., γ_{K-1}]`

---

## 2. Single Path Logic

### 2.1 Function Overview

**Code Location:** `src/afns/simulator.py`, lines 64-169, function `_simulate_single_path()`

Each path is simulated independently with a unique random seed for reproducibility.

### 2.2 Step 1: Draw Initial Regime

At t=0, draw the starting regime from the current regime distribution:

**Code Location:** `src/afns/simulator.py`, lines 92-93

```python
# Set seed for this path
path_seed = sim_params['random_seed'] + path_id
np.random.seed(path_seed)

# 1. Draw initial regime
s_t = np.random.choice(len(gamma_T), p=gamma_T)
```

**Example:**
- If `gamma_T = [0.1, 0.7, 0.2]` (K=3 regimes)
- Regime 1 is most likely (70% probability)
- But some paths will start in regime 0 or 2

This captures **regime uncertainty** at simulation start.

### 2.3 Step 2: Draw Initial Factors

Perturb the current factors f_T by sampling from the state covariance:

**Code Location:** `src/afns/simulator.py`, lines 95-96

```python
# 2. Draw initial factors
f_t = np.random.multivariate_normal(f_T, P_T)
```

**Purpose:**
- Reflects **parameter uncertainty** in factor estimates
- Generates initial diversity across paths
- P_T quantifies Kalman filter's uncertainty about f_T

**Distribution:**
```
f_0^(path) ~ N(f_T, P_T)
```

### 2.4 Step 3: Day-by-Day Evolution Loop

**Code Location:** `src/afns/simulator.py`, lines 114-131

```python
# Simulate forward
for t in range(1, horizon_days + 1):
    # Transition regime
    s_t = np.random.choice(
        len(gamma_T),
        p=self.hmm_model.transition_matrix[s_t, :]
    )
    
    # Propagate factors: f_{t+1} = A f_t + b + η
    eta = np.random.multivariate_normal(np.zeros(3), self.afns_model.Q * self.q_scaler)
    f_t = self.afns_model.A @ f_t + self.afns_model.b + eta
    
    # Store
    path_data['t'].append(t)
    path_data['regime'].append(s_t)
    path_data['L'].append(f_t[0])
    path_data['S'].append(f_t[1])
    path_data['C'].append(f_t[2])
```

#### a. Regime Transition

At each time step, the regime evolves according to the HMM transition matrix:

```python
s_{t+1} ~ Categorical(π[s_t, :])
```

where `π` is the sticky transition matrix.

**Effect:**
- If `s_t = k`, the next regime is drawn from row k of π
- High diagonal π[k,k] means regimes persist
- Rare transitions to other regimes occur stochastically

#### b. Factor Evolution

Factors evolve via the VAR(1) equation:

```python
f_{t+1} = A f_t + b + η_t

where:
  η_t ~ N(0, Q * q_scaler)
```

**Components:**
- `A f_t`: Autoregressive dynamics (persistence and cross-factor effects)
- `b`: Drift toward long-run mean
- `η_t`: Random shock (scaled by q_scaler for MOVE conditioning)

**Matrix Operations:**
```python
f_t = self.afns_model.A @ f_t + self.afns_model.b + eta
```

#### c. Yield Computation

After updating factors, compute yields for all maturities:

**Code Location:** `src/afns/simulator.py`, lines 137-141

```python
# Compute yields at each time step
for mat in self.afns_model.maturities:
    loadings = self.afns_model.compute_loadings(mat)
    yields_path = factors_path @ loadings
    path_data[f'UST_{mat}'] = yields_path
```

**Formula:**
```
y_t(τ) = H(τ)' f_t = [1, h_1(τ), h_2(τ)] @ [L_t, S_t, C_t]'
```

This is applied for each maturity τ ∈ {2, 3, 5, 10, 30} years.

#### d. Spread Simulation

Spreads are simulated using the AR model with Student-t errors:

**Code Location:** `src/afns/simulator.py`, lines 143-149

```python
# Simulate spreads
spreads_sim = self.spread_engine.simulate_spreads(
    factors_path, regime_path, current_spreads, random_seed=path_seed
)

for (sector, mat), spread_path in spreads_sim.items():
    path_data[f'{sector}Spr_{mat}'] = spread_path
```

**Inside `spread_engine.simulate_spreads()`:**

For each bucket (sector, maturity):
1. **Compute expected spread:**
   ```
   E[log(spr_t)] = β_0 + β_L * L_t + β_S * S_t + β_C * C_t + β_regime[s_t]
   ```

2. **Add AR(1) dynamics:**
   ```
   log(spr_t) = ρ * log(spr_{t-1}) + (1-ρ) * E[log(spr_t)] + ε_t
   ```

3. **Student-t errors:**
   ```
   ε_t ~ t_ν(0, σ²)
   ```
   where ν ≈ 5 degrees of freedom (heavier tails than Gaussian)

4. **Exponentiate:**
   ```
   spr_t = exp(log(spr_t))
   ```

**Purpose of Student-t:**
- Captures heavy tails in spread changes
- Generates realistic extreme widening events

---

## 3. MOVE Scaling

### 3.1 MOVE State Classification

The **MOVE Index** (ICE BofA MOVE Index) measures bond market volatility. The codebase classifies the current MOVE state into three categories:

**Code Location:** `src/afns/simulator.py`, lines 437-446

```python
def classify_move_state(move_series):
    if move_series is None or move_series.dropna().empty:
        return 'neutral', 1.0
    terciles = move_series.quantile([1/3, 2/3]).values
    latest = move_series.dropna().iloc[-1]
    if latest <= terciles[0]:
        return 'low', 0.8
    if latest >= terciles[1]:
        return 'high', 1.3
    return 'neutral', 1.0
```

**Classification:**
- **Low:** MOVE ≤ 33rd percentile → **q_scaler = 0.8**
- **Neutral:** 33rd < MOVE < 67th percentile → **q_scaler = 1.0**
- **High:** MOVE ≥ 67th percentile → **q_scaler = 1.3**

### 3.2 Covariance Scaling

The MOVE state scales the factor noise covariance Q:

**Code Location:** `src/afns/simulator.py`, line 123

```python
eta = np.random.multivariate_normal(np.zeros(3), self.afns_model.Q * self.q_scaler)
```

**Effect:**
```
η_t ~ N(0, Q * q_scaler)
```

**Examples:**
- **Low MOVE (q_scaler=0.8):**
  - Reduces factor volatility by 20%
  - Simulates calmer market environment
  - Tighter yield distributions

- **High MOVE (q_scaler=1.3):**
  - Increases factor volatility by 30%
  - Simulates stressed market conditions
  - Wider yield distributions

### 3.3 Initialization

**Code Location:** `src/afns/simulator.py`, lines 40, 49-50

```python
def __init__(self, ..., move_state=None, q_scaler=1.0):
    self.move_state = move_state or 'neutral'
    self.q_scaler = q_scaler
```

**Usage:**
```python
move_state, q_scaler = classify_move_state(move_series)
simulator = Simulator(..., move_state=move_state, q_scaler=q_scaler)
```

---

## 4. Parallelization

### 4.1 Parallelization Strategy

**Method:** `multiprocessing.Pool` for process-based parallelism

**Code Location:** `src/afns/simulator.py`, lines 263-278

```python
# Run simulations in parallel
if self.n_jobs > 1:
    logger.info(f"  Running parallel simulation with {self.n_jobs} workers...")
    with Pool(processes=self.n_jobs) as pool:
        all_paths = list(tqdm(
            pool.imap(self._simulate_single_path_wrapper, 
                     [(path_id, sim_params) for path_id in range(self.n_paths)]),
            total=self.n_paths,
            desc=f"  Simulating {horizon_name}",
            unit="path"
        ))
else:
    # Sequential execution (for debugging)
    logger.info(f"  Running sequential simulation...")
    all_paths = []
    for path_id in tqdm(range(self.n_paths), desc=f"  Simulating {horizon_name}", unit="path"):
        all_paths.append(self._simulate_single_path(path_id, sim_params))
```

### 4.2 CPU Core Allocation

**Code Location:** `src/afns/simulator.py`, line 48

```python
self.n_jobs = n_jobs if n_jobs is not None else max(1, cpu_count() - 1)
```

**Default:** Use all CPUs minus 1
- Leaves one core free for system tasks
- Example: 8-core machine → 7 workers

**Speedup:**
- 10,000 paths on 7 cores: ~7x faster than sequential
- Near-linear scaling for embarrassingly parallel task

### 4.3 Random Seed Management

Each path has a **unique, deterministic seed**:

**Code Location:** `src/afns/simulator.py`, lines 88-90

```python
# Set seed for this path
path_seed = sim_params['random_seed'] + path_id
np.random.seed(path_seed)
```

**Benefits:**
- **Reproducibility:** Re-running with same `random_seed` gives identical results
- **Independence:** Different paths use different seeds
- **Parallel-safe:** No seed conflicts between workers

**Example:**
- Base seed: 42
- Path 0: seed 42
- Path 1: seed 43
- Path 9999: seed 10041

### 4.4 Wrapper Function

**Code Location:** `src/afns/simulator.py`, lines 59-62

```python
def _simulate_single_path_wrapper(self, args):
    """Wrapper for parallel execution."""
    path_id, sim_params = args
    return self._simulate_single_path(path_id, sim_params)
```

**Purpose:**
- `multiprocessing.Pool.imap()` requires a single-argument function
- Wrapper unpacks the tuple `(path_id, sim_params)`

---

## 5. Output Structure

### 5.1 What is Stored Per Path?

**Code Location:** `src/afns/simulator.py`, lines 99-112, 163-168

```python
# Storage for this path
path_data = {
    't': [],
    'regime': [],
    'L': [],
    'S': [],
    'C': []
}

# ... (after simulation loop)

# Convert to DataFrame and add path_id
path_df = pd.DataFrame(path_data)
path_df['path_id'] = path_id

path_df['MOVE_state'] = self.move_state
path_df['SOFR_scenario'] = self.sofr_scenario
return path_df
```

**Columns in path_df:**
- `path_id`: Unique path identifier (0 to 9999)
- `t`: Time step (0 to horizon_days)
- `regime`: Current regime at time t
- `L`, `S`, `C`: Factor values at time t
- `UST_2`, `UST_3`, `UST_5`, `UST_10`, `UST_30`: Yield values
- `{Sector}Spr_{mat}`: Spread values for each bucket
- `Delta_UST_{mat}`: Yield changes from t=0
- `Delta_{Sector}Spr_{mat}`: Spread changes from t=0
- `MOVE_state`: 'low', 'neutral', or 'high'
- `SOFR_scenario`: 'cuts', 'base', or 'hikes' (placeholder)

**Dimensions:**
- Each path: `(horizon_days + 1)` rows (t=0, 1, ..., horizon_days)
- All paths combined: `(horizon_days + 1) × n_paths` rows
- Example: 126 × 10,000 = 1,260,000 rows for 6-month simulation

### 5.2 Path Combination

**Code Location:** `src/afns/simulator.py`, lines 281-282

```python
# Combine all paths
logger.info(f"  Combining {len(all_paths)} paths...")
paths_df = pd.concat(all_paths, ignore_index=True)
```

**Result:** A single DataFrame with all paths stacked vertically.

### 5.3 Percentile Computation

**Code Location:** `src/afns/simulator.py`, lines 294-330

```python
def compute_summaries(self, paths_df, horizon_days):
    # Extract final values (at horizon)
    final_df = paths_df[paths_df['t'] == horizon_days].copy()
    
    # Identify value columns (exclude metadata)
    value_cols = [c for c in final_df.columns 
                  if c not in ['path_id', 't', 'regime', 'MOVE_state', 'SOFR_scenario']]
    
    # Compute percentiles
    percentiles = [5, 25, 50, 75, 95]
    summary = {}
    
    for col in value_cols:
        summary[col] = {
            'mean': final_df[col].mean(),
            'std': final_df[col].std(),
        }
        for p in percentiles:
            summary[col][f'p{p}'] = final_df[col].quantile(p / 100)
    
    summary_df = pd.DataFrame(summary).T
    
    return summary_df
```

**Percentiles:**
- **p5:** 5th percentile (worst-case in lower tail)
- **p25:** 25th percentile (lower quartile)
- **p50:** 50th percentile (median)
- **p75:** 75th percentile (upper quartile)
- **p95:** 95th percentile (worst-case in upper tail)

**Usage:**
- Construct **fan charts** (p5-p95 bands)
- Report **Value-at-Risk (VaR):** 95th percentile of losses
- Identify **tail scenarios** beyond p95

### 5.4 Summary Statistics

**Computed for Each Variable:**
- `mean`: Average across all paths
- `std`: Standard deviation (forecast uncertainty)
- `p5`, `p25`, `p50`, `p75`, `p95`: Percentiles

**Example Row in summary_df:**
```
             mean    std     p5     p25    p50    p75    p95
UST_10       4.12   0.85   2.73   3.55   4.10   4.68   5.62
Delta_UST_10 0.15   0.85  -1.25  -0.43   0.12   0.70   1.64
```

### 5.5 Additional Outputs

**Tail Probabilities:**
```python
def compute_tail_probabilities(self, paths_df, horizon_days, thresholds):
    final_df = paths_df[paths_df['t'] == horizon_days].copy()
    
    results = {}
    for var, thresh_list in thresholds.items():
        for thresh in thresh_list:
            prob = (final_df[var] > thresh).mean()
            results[f'P({var} > {thresh})'] = prob
    
    return pd.Series(results)
```

**Code Location:** `src/afns/simulator.py`, lines 332-361

**Example:**
- `P(UST_10 > 5.0) = 0.23` (23% chance 10yr yield exceeds 5%)
- `P(Delta_UST_10 > 1.0) = 0.12` (12% chance 10yr rises >100bps)

**Variance Attribution:**
```python
def compute_attribution(self, paths_df, horizon_days):
    final_df = paths_df[paths_df['t'] == horizon_days].copy()
    
    # Compute variance of each factor
    var_L = final_df['L'].var()
    var_S = final_df['S'].var()
    var_C = final_df['C'].var()
    
    total_var = var_L + var_S + var_C
    
    attribution = {
        'Level': var_L / total_var * 100,
        'Slope': var_S / total_var * 100,
        'Curvature': var_C / total_var * 100
    }
    
    return attribution
```

**Code Location:** `src/afns/simulator.py`, lines 363-398

**Example:**
```
Level:      70%  (dominant uncertainty)
Slope:      25%
Curvature:   5%
```

---

## 6. Full Single Path Simulation Code

**Code Location:** `src/afns/simulator.py`, lines 64-169

```python
def _simulate_single_path(self, path_id, sim_params):
    """
    Simulate a single Monte Carlo path.
    
    Parameters
    ----------
    path_id : int
        Path identifier
    sim_params : dict
        Simulation parameters
        
    Returns
    -------
    pd.DataFrame
        Path data
    """
    # Unpack parameters
    f_T = sim_params['f_T']
    P_T = sim_params['P_T']
    gamma_T = sim_params['gamma_T']
    horizon_days = sim_params['horizon_days']
    current_yields = sim_params['current_yields']
    current_spreads = sim_params['current_spreads']
    
    # Set seed for this path
    path_seed = sim_params['random_seed'] + path_id
    np.random.seed(path_seed)
    
    # 1. Draw initial regime
    s_t = np.random.choice(len(gamma_T), p=gamma_T)
    
    # 2. Draw initial factors
    f_t = np.random.multivariate_normal(f_T, P_T)
    
    # Storage for this path
    path_data = {
        't': [],
        'regime': [],
        'L': [],
        'S': [],
        'C': []
    }
    
    # Add initial state
    path_data['t'].append(0)
    path_data['regime'].append(s_t)
    path_data['L'].append(f_t[0])
    path_data['S'].append(f_t[1])
    path_data['C'].append(f_t[2])
    
    # Simulate forward
    for t in range(1, horizon_days + 1):
        # Transition regime
        s_t = np.random.choice(
            len(gamma_T),
            p=self.hmm_model.transition_matrix[s_t, :]
        )
        
        # Propagate factors: f_{t+1} = A f_t + b + η
        eta = np.random.multivariate_normal(np.zeros(3), self.afns_model.Q * self.q_scaler)
        f_t = self.afns_model.A @ f_t + self.afns_model.b + eta
        
        # Store
        path_data['t'].append(t)
        path_data['regime'].append(s_t)
        path_data['L'].append(f_t[0])
        path_data['S'].append(f_t[1])
        path_data['C'].append(f_t[2])
    
    # Convert to arrays for processing
    factors_path = np.column_stack([path_data['L'], path_data['S'], path_data['C']])
    regime_path = np.array(path_data['regime'])
    
    # Compute yields at each time step
    for mat in self.afns_model.maturities:
        loadings = self.afns_model.compute_loadings(mat)
        yields_path = factors_path @ loadings
        path_data[f'UST_{mat}'] = yields_path
    
    # Simulate spreads
    spreads_sim = self.spread_engine.simulate_spreads(
        factors_path, regime_path, current_spreads, random_seed=path_seed
    )
    
    for (sector, mat), spread_path in spreads_sim.items():
        path_data[f'{sector}Spr_{mat}'] = spread_path
    
    # Compute changes (Δy and Δspread)
    for mat in self.afns_model.maturities:
        ust_col = f'UST_{mat}'
        path_data[f'Delta_{ust_col}'] = np.array(path_data[ust_col]) - current_yields[mat]
    
    for bucket in self.spread_engine.buckets:
        sector, mat = bucket
        spr_col = f'{sector}Spr_{mat}'
        path_data[f'Delta_{spr_col}'] = (
            np.array(path_data[spr_col]) - current_spreads[bucket]
        )
    
    # Convert to DataFrame and add path_id
    path_df = pd.DataFrame(path_data)
    path_df['path_id'] = path_id
    
    path_df['MOVE_state'] = self.move_state
    path_df['SOFR_scenario'] = self.sofr_scenario
    return path_df
```

---

## 7. Usage Example

```python
from src.afns.simulator import Simulator

# Initialize simulator
simulator = Simulator(
    afns_model=afns,
    hmm_model=hmm,
    spread_engine=spread_engine,
    n_paths=10000,
    horizons={'6m': 126, '12m': 252},
    random_seed=42,
    n_jobs=7,
    move_state='neutral',
    q_scaler=1.0
)

# Run 6-month simulation
paths_df, summary_df = simulator.simulate(horizon_name='6m')

# Access results
print("Summary statistics:")
print(summary_df.loc['UST_10'])

print("\nTail probabilities:")
tail_probs = simulator.compute_tail_probabilities(
    paths_df, 
    horizon_days=126,
    thresholds={'UST_10': [4.0, 5.0, 6.0]}
)
print(tail_probs)

# Save results
simulator.save_simulation(paths_df, summary_df, '6m')
```

---

## 8. References

**File:** `src/afns/simulator.py`

**Key Code Sections:**
- Lines 37-57: Simulator initialization
- Lines 64-169: Single path simulation (full logic)
- Lines 190-292: Main `simulate()` function
- Lines 263-278: Parallelization with Pool
- Lines 294-330: Summary statistics computation
- Lines 332-361: Tail probabilities
- Lines 363-398: Variance attribution
- Lines 437-446: MOVE state classification

**Dependencies:**
- `multiprocessing.Pool`: Parallel execution
- `numpy.random`: Random number generation
- `tqdm`: Progress bars

---

This implementation provides a scalable Monte Carlo framework that:
1. **Simulates** realistic yield and spread scenarios
2. **Incorporates** regime dynamics and factor uncertainty
3. **Scales** covariance based on MOVE state
4. **Parallelizes** efficiently across CPU cores
5. **Outputs** rich distributions and summary statistics

