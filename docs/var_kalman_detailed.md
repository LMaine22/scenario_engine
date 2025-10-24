# AFNS VAR(1) Model with Kalman Filter: Detailed Implementation Guide

## Overview

This document provides a comprehensive explanation of how the Arbitrage-Free Nelson-Siegel (AFNS) VAR(1) model with Kalman filter estimation is implemented in this codebase. The AFNS model uses a three-factor structure (Level, Slope, Curvature) to model yield curve dynamics through a state-space framework.

---

## 1. Mathematical Model

### 1.1 State-Space Equations

The AFNS model is formulated as a state-space system with the following equations:

**State Equation (Factor Dynamics):**
```
f_{t+1} = A f_t + b + η_t
```

where:
- `f_t = [L_t, S_t, C_t]'` is the 3×1 vector of latent factors at time t
  - `L_t`: Level factor (long-term mean of the yield curve)
  - `S_t`: Slope factor (short-term vs. long-term spread)
  - `C_t`: Curvature factor (medium-term hump/trough)
- `A` is the 3×3 transition matrix governing factor dynamics
- `b` is the 3×1 drift/intercept vector
- `η_t ~ N(0, Q)` is the state innovation noise, where `Q` is a 3×3 covariance matrix

**Observation Equation (Yield Curve):**
```
y_t = H f_t + ε_t
```

where:
- `y_t` is the n×1 vector of observed yields at time t (n = number of maturities)
- `H` is the n×3 loading matrix mapping factors to yields
- `ε_t ~ N(0, R)` is the measurement noise, where `R` is an n×n covariance matrix

### 1.2 Loading Matrix Structure

The loading matrix `H` is constructed using the Nelson-Siegel loading functions:

For maturity τ (in years), the loadings are:
```
H(τ) = [1, h_1(τ), h_2(τ)]

where:
  h_1(τ) = (1 - exp(-λτ)) / (λτ)
  h_2(τ) = h_1(τ) - exp(-λτ)
```

The parameter `λ` (lambda) controls the decay rate and is fixed at 0.075 by default.

### 1.3 Parameter Dimensions Summary

| Parameter | Dimension | Description |
|-----------|-----------|-------------|
| `A` | 3×3 | Factor transition matrix |
| `b` | 3×1 | Factor drift vector |
| `Q` | 3×3 | State noise covariance (diagonal) |
| `R` | n×n | Measurement noise covariance (diagonal, fixed) |
| `H` | n×3 | Loading matrix (deterministic from λ) |
| `f_t` | 3×1 | Factor state at time t |
| `y_t` | n×1 | Yield observations at time t |

Default configuration uses `n = 5` maturities: [2, 3, 5, 10, 30] years.

---

## 2. Factor Initialization

### 2.1 DNS Approximation

Before running the Kalman filter, initial factors are computed using the Dynamic Nelson-Siegel (DNS) approximation. This provides starting values that approximate the three latent factors from observed yields.

**DNS Formulas:**
```
L_t = (UST_2_t + UST_10_t) / 2
S_t = UST_10_t - UST_2_t
C_t = 2 * UST_5_t - UST_2_t - UST_10_t
```

**Code Location:** `src/afns/afns_model.py`, lines 98-134

**Implementation:**
```python
def initialize_factors_dns(self, yields_df):
    """
    Initialize factors using DNS approximation.
    
    Parameters
    ----------
    yields_df : pd.DataFrame
        DataFrame with columns UST_2, UST_5, UST_7, UST_10
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns L, S, C
    """
    logger.info("Initializing factors using DNS approximation")
    
    # DNS formulas
    columns = yields_df.columns
    required = {'UST_2', 'UST_5', 'UST_10'}
    if not required.issubset(columns):
        missing = required - set(columns)
        raise ValueError(f"Missing required columns for DNS seed: {missing}")

    L = (yields_df['UST_2'] + yields_df['UST_10']) / 2
    S = yields_df['UST_10'] - yields_df['UST_2']
    C = 2 * yields_df['UST_5'] - yields_df['UST_2'] - yields_df['UST_10']
    
    factors_df = pd.DataFrame({
        'L': L,
        'S': S,
        'C': C
    }, index=yields_df.index)
    
    return factors_df
```

### 2.2 Purpose of DNS Initialization

The DNS approximation serves two purposes:
1. **Parameter initialization**: Provides initial guesses for `A` and `b` by fitting a simple AR(1) model to DNS factors
2. **Kalman filter initialization**: The first observation's DNS factors provide the initial state `f_0` via least squares

---

## 3. Kalman Filter

### 3.1 Kalman Filter Initialization

**Code Location:** `src/afns/afns_model.py`, lines 136-196, function `_kalman_filter()`

The Kalman filter is initialized with:

```python
# Initialize Kalman filter
kf = KalmanFilter(dim_x=3, dim_z=self.n_maturities)

# Transition model: f_{t+1} = A f_t + b + η, η ~ N(0, Q)
kf.F = A
kf.Q = Q

# Observation model: y_t = H f_t + ε, ε ~ N(0, R)
kf.H = self.H
kf.R = self.R

# Initial state (use DNS approximation from first observation)
y0 = yields[0, :]
f0 = np.linalg.lstsq(self.H, y0, rcond=None)[0]
kf.x = f0
kf.P = np.eye(3) * 10  # Initial state covariance
```

**Initial State:**
- State mean `f_0`: Computed via least squares `H f_0 ≈ y_0`
- State covariance `P_0`: Identity matrix scaled by 10 (reflects uncertainty)

**Fixed Parameters:**
- Measurement noise `R`: Diagonal matrix with `R_ii = 4.0 bps²` (fixed)
- Loading matrix `H`: Computed deterministically from λ = 0.075

### 3.2 Prediction Step

At each time step t, the Kalman filter predicts the next state:

```python
# Predict
kf.predict()
```

This implements:
```
f̂_{t|t-1} = A f̂_{t-1|t-1} + b
P_{t|t-1} = A P_{t-1|t-1} A' + Q
```

where:
- `f̂_{t|t-1}`: Prior state estimate (prediction)
- `P_{t|t-1}`: Prior state covariance

### 3.3 Update Step

When an observation `y_t` is available, the filter updates the state estimate:

```python
# Update with observation
y_t = yields[t, :]
if not np.any(np.isnan(y_t)):
    kf.update(y_t)
    
    filtered_factors[t, :] = kf.x.flatten()
```

This implements:
```
Innovation: ν_t = y_t - H f̂_{t|t-1}
Innovation covariance: S_t = H P_{t|t-1} H' + R
Kalman gain: K_t = P_{t|t-1} H' S_t^{-1}

Posterior state: f̂_{t|t} = f̂_{t|t-1} + K_t ν_t
Posterior covariance: P_{t|t} = (I - K_t H) P_{t|t-1}
```

### 3.4 Likelihood Computation

The log-likelihood is accumulated during the filtering process:

```python
# Accumulate log likelihood
innovation = y_t - kf.H @ kf.x_prior
S = kf.H @ kf.P_prior @ kf.H.T + kf.R
log_likelihood += -0.5 * (
    self.n_maturities * np.log(2 * np.pi) +
    np.log(np.linalg.det(S) + 1e-10) +
    innovation.T @ np.linalg.inv(S) @ innovation
)
```

The log-likelihood at time t is:
```
ℓ_t = -0.5 * [n log(2π) + log|S_t| + ν_t' S_t^{-1} ν_t]

Total log-likelihood: L = Σ_t ℓ_t
```

This measures how well the model fits the observed yields.

---

## 4. Maximum Likelihood Estimation

### 4.1 Parameter Vector

The model parameters are packed into a single vector for optimization:

```python
params = [A.flatten(),  # 9 elements (3×3 matrix)
          b,            # 3 elements (vector)
          diag(Q)]      # 3 elements (diagonal covariance)
```

Total: **15 parameters** to estimate.

**Code Location:** `src/afns/afns_model.py`, lines 270-275

### 4.2 Optimization Method

The model uses **BFGS** (Broyden-Fletcher-Goldfarb-Shanno), a quasi-Newton optimization method.

**Code Location:** `src/afns/afns_model.py`, lines 281-287

```python
# Optimize
result = minimize(
    self._objective,
    params_init,
    args=(yields,),
    method='BFGS',
    options={'maxiter': max_iter, 'disp': False}
)
```

**Why BFGS?**
- Gradient-based method with super-linear convergence
- Approximates the Hessian matrix without computing it explicitly
- Well-suited for smooth likelihood surfaces

### 4.3 Objective Function

The objective function returns the **negative log-likelihood**:

**Code Location:** `src/afns/afns_model.py`, lines 198-229

```python
def _objective(self, params, yields):
    """
    Negative log-likelihood for optimization.
    
    Parameters
    ----------
    params : np.array
        Flattened parameters [A, b, Q_diag]
    yields : np.array
        Yield observations
        
    Returns
    -------
    float
        Negative log likelihood
    """
    # Unpack parameters
    A = params[:9].reshape(3, 3)
    b = params[9:12]
    Q_diag = np.abs(params[12:15])  # Constrain to positive
    Q = np.diag(Q_diag)
    
    # Ensure stability (eigenvalues of A < 1)
    eigvals = np.linalg.eigvals(A)
    if np.any(np.abs(eigvals) > 0.999):
        return 1e10
    
    try:
        _, log_likelihood = self._kalman_filter(yields, A, b, Q)
        return -log_likelihood
    except:
        return 1e10
```

### 4.4 Constraints

**Stability Constraint:**
- All eigenvalues of `A` must satisfy `|λ_i| < 0.999`
- Ensures the VAR(1) process is stationary
- Violated parameters return penalty value `1e10`

**Positivity Constraint:**
- Diagonal elements of `Q` are forced positive via `np.abs()`
- Ensures valid covariance matrix

**No other explicit constraints** are imposed (e.g., on parameter ranges).

### 4.5 Initial Parameter Guesses

**Code Location:** `src/afns/afns_model.py`, lines 263-275

```python
# Initialize parameters with simple AR(1) on DNS factors
A_init = np.eye(3) * 0.95  # Start with persistent diagonal
b_init = factors_init.mean().values * 0.05

factor_changes = factors_init.diff().dropna()
Q_init = np.diag(factor_changes.var().values)
```

Initial guesses are computed from DNS factors:
- `A_init`: Diagonal matrix with 0.95 on diagonal (high persistence)
- `b_init`: Small drift towards factor means
- `Q_init`: Diagonal covariance from DNS factor changes

---

## 5. Model Outputs

### 5.1 Fitted Parameters

After optimization, the fitted model returns:

**Code Location:** `src/afns/afns_model.py`, lines 294-361

```python
# Extract fitted parameters
self.A = result.x[:9].reshape(3, 3)
self.b = result.x[9:12]
Q_diag = np.abs(result.x[12:15])
self.Q = np.diag(Q_diag)
```

**Stored in Model:**
- `self.A` (3×3): Transition matrix
- `self.b` (3×1): Drift vector
- `self.Q` (3×3): State noise covariance (diagonal)
- `self.R` (n×n): Measurement noise covariance (fixed, not estimated)

### 5.2 Historical Filtered Factors

The Kalman filter produces historical factor estimates:

```python
# Run Kalman filter with fitted parameters
filtered_factors, log_likelihood = self._kalman_filter(yields, self.A, self.b, self.Q)

# Store factors
self.factors = pd.DataFrame(
    filtered_factors,
    index=yields_df.index,
    columns=['L', 'S', 'C']
)

# Compute factor changes for HMM
self.factor_changes = self.factors.diff().dropna()
```

**Output:** `self.factors` is a DataFrame with columns `['L', 'S', 'C']` containing the filtered factor estimates at each time point.

### 5.3 Reconstruction Quality

The model computes yield reconstruction errors:

```python
# Compute reconstruction errors
yields_fitted = (self.H @ self.factors.T).T
errors = yields - yields_fitted
rmse_by_maturity = np.sqrt(np.mean(errors**2, axis=0))
overall_rmse = np.sqrt(np.mean(errors**2))
```

**Target:** Overall RMSE < 5 bps (basis points)

### 5.4 Return Dictionary

The `fit()` method returns:

```python
results = {
    'A': self.A,                           # Transition matrix
    'b': self.b,                           # Drift vector
    'Q': self.Q,                           # State noise covariance
    'R': self.R,                           # Measurement noise covariance
    'log_likelihood': log_likelihood,      # Maximum log-likelihood
    'rmse_by_maturity': rmse_by_maturity, # RMSE for each maturity
    'overall_rmse': overall_rmse,         # Overall RMSE across maturities
    'factors': self.factors,               # Historical filtered factors
    'factor_changes': self.factor_changes  # Δf_t for HMM
}
```

---

## 6. Code Summary

### 6.1 Main Functions

| Function | Lines | Description |
|----------|-------|-------------|
| `__init__()` | 34-56 | Initialize model with λ, maturities, and measurement noise |
| `compute_loadings()` | 58-82 | Compute Nelson-Siegel loadings for a maturity |
| `_compute_loading_matrix()` | 84-96 | Build full loading matrix H |
| `initialize_factors_dns()` | 98-134 | DNS approximation for initial factors |
| `_kalman_filter()` | 136-196 | Run Kalman filter given parameters |
| `_objective()` | 198-229 | Negative log-likelihood for optimization |
| `fit()` | 231-361 | Main estimation function |

### 6.2 Key Dependencies

```python
from scipy.optimize import minimize      # BFGS optimizer
from filterpy.kalman import KalmanFilter # Kalman filter implementation
```

**External Library:** The implementation uses `filterpy.kalman.KalmanFilter` for the filtering operations, which handles the prediction and update steps internally.

---

## 7. Interpretation of Parameters

### 7.1 Transition Matrix A

The matrix `A` governs factor dynamics:
- **Diagonal elements** `A_ii`: Persistence of factor i
  - `A_11 ≈ 0.99`: Level is highly persistent
  - `A_22 ≈ 0.95`: Slope is persistent but mean-reverting
  - `A_33 ≈ 0.90`: Curvature is more mean-reverting
- **Off-diagonal elements** `A_ij` (i≠j): Cross-factor spillovers
  - Often small in practice

### 7.2 Drift Vector b

The vector `b` represents the unconditional drift:
- `b_i = (1 - A_ii) * E[f_i]` approximately
- Pulls factors toward their long-run means

### 7.3 Covariance Matrices

- **Q (State noise)**: Captures day-to-day factor volatility
  - `Q_11`: Level volatility (typically ~0.5 bps²)
  - `Q_22`: Slope volatility (typically ~2-3 bps²)
  - `Q_33`: Curvature volatility (typically ~1-2 bps²)
- **R (Measurement noise)**: Fixed at 4.0 bps² per maturity
  - Represents model error and data noise

---

## 8. Usage Example

```python
from src.afns.afns_model import AFNSModel

# Initialize model
afns = AFNSModel(
    lambda_param=0.075,
    maturities=[2, 3, 5, 10, 30],
    measurement_noise_R=4.0
)

# Fit to data
results = afns.fit(yields_df, max_iter=100)

# Access fitted parameters
print("Transition matrix A:")
print(afns.A)
print("\nDrift vector b:")
print(afns.b)
print("\nState covariance Q:")
print(afns.Q)

# Access filtered factors
factors_history = afns.factors  # DataFrame with L, S, C columns
```

---

## 9. References

**File:** `src/afns/afns_model.py`

**Key Code Sections:**
- Lines 34-56: Model initialization
- Lines 98-134: DNS factor initialization
- Lines 136-196: Kalman filter implementation
- Lines 198-229: Objective function with constraints
- Lines 231-361: Maximum likelihood estimation (fit method)

**Dependencies:**
- `scipy.optimize.minimize`: BFGS optimization
- `filterpy.kalman.KalmanFilter`: Kalman filtering
- `numpy`, `pandas`: Numerical operations

---

This implementation provides a statistically rigorous approach to yield curve modeling through:
1. **State-space representation** with latent factors
2. **Kalman filtering** for optimal state estimation
3. **Maximum likelihood** for parameter estimation
4. **Stability constraints** ensuring model soundness

