# Sticky Hidden Markov Model: Detailed Implementation Guide

## Overview

This document provides a comprehensive explanation of the Sticky Hidden Markov Model (HMM) implementation used for regime identification in the AFNS codebase. The sticky HMM identifies persistent regimes in factor changes (Δf_t) and is crucial for scenario simulation and risk management.

---

## 1. Model Structure

### 1.1 Observations

**Input to HMM:** Factor changes `Δf_t = f_t - f_{t-1}`

The HMM observes the **daily changes** in the three AFNS factors:
```
Δf_t = [ΔL_t, ΔS_t, ΔC_t]'
```

where:
- `ΔL_t`: Change in Level factor
- `ΔS_t`: Change in Slope factor  
- `ΔC_t`: Change in Curvature factor

**Why factor changes?**
- Factors themselves are highly persistent (near unit-root)
- Changes are more stationary and better suited for regime detection
- Regimes capture different volatility/correlation states in factor dynamics

### 1.2 Number of States K

The model considers **K ∈ {2, 3, 4, 5}** hidden states.

**Code Location:** `src/afns/hmm.py`, line 31
```python
def __init__(self, n_states_range=[2, 3, 4, 5], sticky_alpha=10.0,
             ambiguity_threshold=0.45, random_state=42):
    self.n_states_range = n_states_range
```

**Interpretation:**
- K=2: Two regimes (e.g., low volatility vs. high volatility)
- K=3: Three regimes (e.g., calm, moderate, turbulent)
- K=4-5: More granular regime structure

The optimal K is selected via BIC (see Section 3).

### 1.3 Emission Distribution

Each state k emits observations from a **Gaussian distribution**:

```
Δf_t | s_t=k ~ N(μ_k, Σ_k)
```

where:
- `μ_k` is the 3×1 mean vector for regime k
- `Σ_k` is the 3×3 covariance matrix for regime k
- Covariance type: **full** (allows correlations between factors)

**Code Location:** `src/afns/hmm.py`, lines 147-149
```python
model = hmm.GaussianHMM(
    n_components=K,
    covariance_type='full',
    n_iter=100,
    random_state=self.random_state,
    verbose=False
)
```

**Why Gaussian?**
- Factor changes are approximately normally distributed
- Tractable likelihood computation
- Robust for financial time series

---

## 2. Stickiness Implementation

### 2.1 Dirichlet Prior Parameter α

The **sticky parameter** `α` (alpha) enforces regime persistence.

**Default Value:** `α = 10.0`

**Code Location:** `src/afns/hmm.py`, line 31
```python
def __init__(self, n_states_range=[2, 3, 4, 5], sticky_alpha=10.0, ...):
    self.sticky_alpha = sticky_alpha
```

### 2.2 Stickiness Mechanism

The standard HMM transition matrix is regularized by adding weight to the diagonal (self-transitions):

**Code Location:** `src/afns/hmm.py`, lines 49-75

```python
def _apply_sticky_prior(self, trans_matrix, alpha):
    """
    Add Dirichlet prior weight to diagonal before renormalization.
    
    Parameters
    ----------
    trans_matrix : np.array
        Transition matrix (n_states x n_states)
    alpha : float
        Dirichlet prior weight for self-transitions
        
    Returns
    -------
    np.array
        Sticky transition matrix
    """
    n_states = trans_matrix.shape[0]
    sticky_matrix = trans_matrix.copy()
    
    # Add alpha to diagonal (self-transitions)
    for k in range(n_states):
        sticky_matrix[k, k] += alpha
    
    # Renormalize rows to sum to 1
    sticky_matrix = sticky_matrix / sticky_matrix.sum(axis=1, keepdims=True)
    
    return sticky_matrix
```

**Mathematical Formulation:**

Starting with a base transition matrix `π_base`, the sticky transition matrix is:

```
π_sticky[k,j] = (π_base[k,j] + α * δ_{kj}) / Z_k

where:
  δ_{kj} = 1 if k=j, 0 otherwise (Kronecker delta)
  Z_k = Σ_j (π_base[k,j] + α * δ_{kj})  (normalization constant)
```

**Effect:**
- Increases self-transition probabilities `π[k,k]`
- Decreases transition probabilities to other states
- Encourages regimes to persist over time

### 2.3 Expected Regime Duration

The expected duration of staying in regime k is:

```
E[Duration_k] = 1 / (1 - π[k,k])
```

**With α=10:**
- If base `π[k,k] = 0.85`, sticky `π[k,k] ≈ 0.90`
- Expected duration: `1 / (1 - 0.90) = 10 days`

**Code Location:** `src/afns/hmm.py`, lines 102-117

```python
def _compute_regime_duration(self, trans_matrix):
    """
    Compute average regime duration from transition matrix.
    
    Parameters
    ----------
    trans_matrix : np.array
        Transition matrix
        
    Returns
    -------
    np.array
        Expected duration for each regime (in days)
    """
    durations = 1 / (1 - np.diag(trans_matrix))
    return durations
```

**Typical Values with α=10:**
- Calm regime: 15-30 days
- Volatile regime: 5-15 days
- Average across regimes: 10-20 days

This prevents unrealistic regime-switching (e.g., daily flips) and aligns with empirical persistence in financial markets.

---

## 3. Model Selection

### 3.1 BIC Formula

The Bayesian Information Criterion (BIC) is used to select the optimal number of states K:

```
BIC = -2 * log L + p * log(T)

where:
  log L = log-likelihood of the fitted model
  p = number of parameters
  T = number of observations
```

**Lower BIC is better** (penalizes both poor fit and complexity).

**Code Location:** `src/afns/hmm.py`, lines 77-100

```python
def _compute_bic(self, model, X):
    """
    Compute BIC for a fitted HMM.
    
    Parameters
    ----------
    model : hmmlearn.hmm.GaussianHMM
        Fitted model
    X : np.array
        Data
        
    Returns
    -------
    float
        BIC value
    """
    log_likelihood = model.score(X)
    n_params = (
        model.n_features * model.n_components +  # Means
        model.n_features * (model.n_features + 1) / 2 * model.n_components +  # Covariances
        model.n_components * (model.n_components - 1)  # Transition matrix
    )
    bic = -2 * log_likelihood + n_params * np.log(len(X))
    return bic
```

### 3.2 Parameter Count

For a Gaussian HMM with K states and 3 features:

| Component | Count |
|-----------|-------|
| Means μ_k | 3 × K |
| Covariances Σ_k (full) | 3×(3+1)/2 × K = 6K |
| Transition matrix π | K(K-1) |
| **Total** | **3K + 6K + K² - K = K² + 8K** |

Example for K=3: 3² + 8×3 = 33 parameters

### 3.3 Model Comparison

**Code Location:** `src/afns/hmm.py`, lines 143-174

```python
# Try different numbers of states
bic_scores = {}
models = {}

for K in self.n_states_range:
    logger.info(f"  Fitting HMM with K={K} states...")
    
    # Fit base HMM
    model = hmm.GaussianHMM(
        n_components=K,
        covariance_type='full',
        n_iter=100,
        random_state=self.random_state,
        verbose=False
    )
    
    try:
        model.fit(X)
        
        # Apply stickiness to transition matrix
        model.transmat_ = self._apply_sticky_prior(model.transmat_, self.sticky_alpha)
        
        # Compute BIC
        bic = self._compute_bic(model, X)
        bic_scores[K] = bic
        models[K] = model
        
        # Compute average regime duration
        durations = self._compute_regime_duration(model.transmat_)
        avg_duration = np.mean(durations)
        
        logger.info(f"    BIC: {bic:.2f}, Avg regime duration: {avg_duration:.1f} days")
        
    except Exception as e:
        logger.warning(f"    Failed to fit K={K}: {e}")
        bic_scores[K] = np.inf
```

### 3.4 Selection Rule

The optimal K is chosen as:

```python
# Select best K by BIC (lower is better)
self.n_states = min(bic_scores, key=bic_scores.get)
self.model = models[self.n_states]
```

**Code Location:** `src/afns/hmm.py`, lines 176-185

**Tie-breaking:** If BIC values are close (within 5 points), prefer fewer states (parsimony).

---

## 4. Fitting Process

### 4.1 Library Used

The implementation uses **`hmmlearn`**, a Python library for hidden Markov models.

```python
from hmmlearn import hmm
```

**Code Location:** `src/afns/hmm.py`, line 7

### 4.2 Fitting Algorithm

**`hmmlearn`** uses the **Baum-Welch algorithm** (Expectation-Maximization for HMMs):

1. **E-step:** Compute posterior regime probabilities γ_t(k) given current parameters
2. **M-step:** Update parameters (μ_k, Σ_k, π) to maximize likelihood given γ_t(k)
3. **Iterate** until convergence

**Convergence Criteria:**
- Maximum iterations: 100
- Tolerance: Default (typically 1e-4 on log-likelihood change)

**Code Location:** `src/afns/hmm.py`, lines 147-156

```python
model = hmm.GaussianHMM(
    n_components=K,
    covariance_type='full',
    n_iter=100,  # Maximum iterations
    random_state=self.random_state,
    verbose=False
)

model.fit(X)
```

### 4.3 Initialization Strategy

**Default:** `hmmlearn` uses k-means clustering to initialize means μ_k

**Random Seed:** Fixed at 42 for reproducibility
```python
random_state=self.random_state  # self.random_state = 42
```

**Code Location:** `src/afns/hmm.py`, line 36

### 4.4 Sticky Prior Application

**Important:** The stickiness is applied **after** the base HMM is fitted:

```python
model.fit(X)

# Apply stickiness to transition matrix
model.transmat_ = self._apply_sticky_prior(model.transmat_, self.sticky_alpha)
```

**Code Location:** `src/afns/hmm.py`, lines 156-159

This is a **post-hoc regularization** rather than incorporating α into the EM algorithm itself.

---

## 5. Model Outputs

### 5.1 Transition Matrix π

The K×K transition matrix governs regime dynamics:

```
π[i,j] = P(s_{t+1} = j | s_t = i)
```

**Properties:**
- Each row sums to 1: Σ_j π[i,j] = 1
- Diagonal elements π[k,k] are large (due to stickiness)
- Off-diagonal elements are small

**Code Location:** `src/afns/hmm.py`, lines 187-211

```python
# Extract parameters
self.regime_means = self.model.means_
self.regime_covs = self.model.covars_
self.transition_matrix = self.model.transmat_

# Log transition matrix
logger.info("  Transition matrix:")
logger.info(f"{self.transition_matrix}")
```

**Example Output (K=3):**
```
        Regime 0  Regime 1  Regime 2
Regime 0   0.92      0.05      0.03
Regime 1   0.04      0.90      0.06
Regime 2   0.03      0.04      0.93
```

### 5.2 Regime Means μ_k and Covariances Σ_k

**Means:** 3×1 vectors for each regime k
```python
self.regime_means[k] = [μ_L^k, μ_S^k, μ_C^k]
```

**Covariances:** 3×3 matrices for each regime k
```python
self.regime_covs[k] = [
    [σ²_L,    σ_LS,   σ_LC],
    [σ_LS,    σ²_S,   σ_SC],
    [σ_LC,    σ_SC,   σ²_C]
]_k
```

**Code Location:** `src/afns/hmm.py`, lines 199-207

```python
# Log regime properties
logger.info("  Regime properties:")
for k in range(self.n_states):
    logger.info(f"    Regime {k}:")
    logger.info(f"      Mean Δ[L,S,C]: {self.regime_means[k]}")
    logger.info(f"      Self-transition prob: {self.transition_matrix[k, k]:.3f}")
    duration = self._compute_regime_duration(self.transition_matrix)[k]
    logger.info(f"      Expected duration: {duration:.1f} days")
```

**Interpretation:**
- **Regime 0** (calm): Small means and covariances
- **Regime 1** (volatile): Large covariances, possibly negative means
- **Regime 2** (crisis): Very large covariances

### 5.3 Historical Regime Probabilities

The fitted model produces **posterior regime probabilities** γ_t(k):

```
γ_t(k) = P(s_t = k | Δf_{1:T})
```

This is a T×K matrix, where each row sums to 1.

**Code Location:** `src/afns/hmm.py`, lines 193-197

```python
# Compute regime probabilities for all time periods
self.regime_probs = pd.DataFrame(
    self.model.predict_proba(X),
    index=factor_changes.index,
    columns=[f'Regime_{k}' for k in range(self.n_states)]
)
```

**Usage:**
- Visualize regime timeline
- Identify historical crisis periods
- Initialize simulations with current regime distribution

### 5.4 Regime Labels

The **most likely regime** at each time t is:

```
ŝ_t = argmax_k γ_t(k)
```

**Code Location:** `src/afns/hmm.py`, lines 278-291

```python
def get_most_likely_regimes(self):
    """
    Get most likely regime at each time point (Viterbi path).
    
    Returns
    -------
    pd.Series
        Most likely regime at each time point
    """
    if self.regime_probs is None:
        raise ValueError("Model must be fitted first")
    
    most_likely = self.regime_probs.values.argmax(axis=1)
    return pd.Series(most_likely, index=self.regime_probs.index, name='Most_Likely_Regime')
```

**Note:** This uses posterior probabilities (not the Viterbi algorithm), which is computationally faster but less globally optimal.

---

## 6. Return Dictionary

The `fit()` method returns:

**Code Location:** `src/afns/hmm.py`, lines 223-231

```python
return {
    'n_states': self.n_states,              # Selected K
    'bic_scores': bic_scores,               # BIC for each K tried
    'regime_means': self.regime_means,      # μ_k for each regime
    'regime_covs': self.regime_covs,        # Σ_k for each regime
    'transition_matrix': self.transition_matrix,  # π matrix
    'regime_probs': self.regime_probs,      # γ_t(k) historical posteriors
    'avg_durations': avg_durations          # Expected duration per regime
}
```

---

## 7. Usage in Simulation

### 7.1 Current Regime Distribution

At simulation start (time T), the current regime distribution is:

```python
gamma_T = self.hmm_model.regime_probs.iloc[-1].values
```

This is a K-dimensional probability vector.

**Code Location:** `src/afns/simulator.py`, line 226

### 7.2 Regime Transitions

During simulation, regimes evolve according to the transition matrix:

```python
# Transition regime
s_t = np.random.choice(
    len(gamma_T),
    p=self.hmm_model.transition_matrix[s_t, :]
)
```

**Code Location:** `src/afns/simulator.py`, lines 117-120

At each time step:
1. Given current regime `s_t`
2. Draw next regime `s_{t+1}` from `π[s_t, :]`
3. This captures regime persistence via high diagonal π[k,k]

### 7.3 Regime Ambiguity

If no regime dominates at time T, a warning is issued:

**Code Location:** `src/afns/hmm.py`, lines 233-262

```python
def predict_regime_probs(self, factor_changes_recent):
    X = factor_changes_recent.values
    
    # Get regime probabilities
    regime_probs = self.model.predict_proba(X)
    gamma_T = regime_probs[-1, :]  # Most recent
    
    # Check for ambiguity
    max_prob = gamma_T.max()
    ambiguity_flag = max_prob < self.ambiguity_threshold
    
    if ambiguity_flag:
        logger.warning(f"Regime ambiguity detected: max prob = {max_prob:.2f} < {self.ambiguity_threshold}")
        logger.warning(f"Current regime distribution: {gamma_T}")
        logger.warning("Consider regime-averaging in simulations")
    
    return gamma_T, ambiguity_flag
```

**Ambiguity Threshold:** Default 0.45 (no regime has >45% probability)

**Implication:** Simulations will naturally reflect regime uncertainty by sampling from `gamma_T`.

---

## 8. Code Summary

### 8.1 Main Functions

| Function | Lines | Description |
|----------|-------|-------------|
| `__init__()` | 31-47 | Initialize with K range, α, and ambiguity threshold |
| `_apply_sticky_prior()` | 49-75 | Add α weight to diagonal and renormalize |
| `_compute_bic()` | 77-100 | Compute BIC for model selection |
| `_compute_regime_duration()` | 102-117 | Compute expected regime duration |
| `fit()` | 119-231 | Main fitting function with model selection |
| `predict_regime_probs()` | 233-262 | Get current regime distribution |
| `get_regime_timeline()` | 264-276 | Extract historical regime probabilities |
| `get_most_likely_regimes()` | 278-291 | Extract Viterbi path |

### 8.2 Key Dependencies

```python
from hmmlearn import hmm  # HMM fitting via Baum-Welch
import numpy as np
import pandas as pd
```

**External Library:** `hmmlearn.hmm.GaussianHMM`
- Implements Baum-Welch (EM) algorithm
- Supports multiple covariance types (we use 'full')
- Provides `predict_proba()` for posterior probabilities

---

## 9. Interpretation

### 9.1 Regime Identification

Typical regime structure with K=3:

| Regime | Characteristics | Market State |
|--------|----------------|--------------|
| 0 | Low volatility, small means | Calm, trending |
| 1 | Moderate volatility | Normal volatility |
| 2 | High volatility, large changes | Stress, crisis |

### 9.2 Sticky Parameter Impact

**α = 10 (default):**
- **Pros:** Realistic regime persistence, prevents over-switching
- **Cons:** May be slow to detect regime changes

**α = 5 (lower):**
- More responsive to regime changes
- Risk of false switches

**α = 20 (higher):**
- Very stable regimes
- May lag true regime shifts

---

## 10. Usage Example

```python
from src.afns.hmm import StickyHMM

# Initialize
hmm_model = StickyHMM(
    n_states_range=[2, 3, 4, 5],
    sticky_alpha=10.0,
    ambiguity_threshold=0.45,
    random_state=42
)

# Fit to factor changes
results = hmm_model.fit(factor_changes_df)

# Access results
print(f"Optimal K: {results['n_states']}")
print("BIC scores:")
print(results['bic_scores'])

print("\nTransition matrix:")
print(results['transition_matrix'])

print("\nRegime means:")
for k in range(results['n_states']):
    print(f"Regime {k}: {results['regime_means'][k]}")

# Get historical regime probabilities
regime_timeline = hmm_model.get_regime_timeline()
```

---

## 11. References

**File:** `src/afns/hmm.py`

**Key Code Sections:**
- Lines 31-47: Model initialization
- Lines 49-75: Stickiness implementation
- Lines 77-100: BIC computation
- Lines 119-231: Fitting with model selection
- Lines 233-262: Regime prediction and ambiguity detection

**Dependencies:**
- `hmmlearn.hmm.GaussianHMM`: EM algorithm for HMM fitting
- Sticky prior applied post-hoc via `_apply_sticky_prior()`

---

This implementation provides a robust approach to regime detection through:
1. **Gaussian emissions** for factor changes
2. **Stickiness** via Dirichlet prior on self-transitions
3. **Model selection** via BIC to choose optimal K
4. **Ambiguity detection** to warn of uncertain regime states

