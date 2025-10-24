# Validation Metrics (CRPS and PIT): Detailed Implementation Guide

## Overview

This document provides a comprehensive explanation of the validation metrics used in the AFNS codebase, focusing on **CRPS** (Continuous Ranked Probability Score) and **PIT** (Probability Integral Transform). These proper scoring rules assess forecast quality and calibration in a rolling-origin backtest framework.

---

## 1. CRPS (Continuous Ranked Probability Score)

### 1.1 Mathematical Definition

The **Continuous Ranked Probability Score** measures the distance between a forecast distribution and a realized value.

**Formula:**
```
CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|

where:
  F = forecast distribution
  y = realized value
  X, X' = independent random draws from F
```

**Interpretation:**
- **Lower is better** (0 is perfect)
- Units: Same as the variable being forecast (e.g., bps for yields)
- Balances **sharpness** (narrow forecast) and **calibration** (accuracy)

**Components:**
- **Term 1:** Average absolute error between forecast samples and realized value
- **Term 2:** Average pairwise distance within forecast distribution (penalizes over-confidence)

**Why CRPS?**
- **Proper scoring rule:** Encourages honest forecasts
- **Works with full distribution:** Doesn't collapse to point forecast
- **Robust:** Handles heavy tails and asymmetry
- **Standard:** Widely used in weather forecasting and probabilistic ML

### 1.2 Code Implementation

**Code Location:** `src/afns/validation.py`, lines 45-83

```python
def _compute_crps(self, forecast_samples, realized):
    """
    Compute Continuous Ranked Probability Score.
    
    Parameters
    ----------
    forecast_samples : np.array
        Forecast distribution samples
    realized : float
        Realized value
        
    Returns
    -------
    float
        CRPS score (lower is better)
    """
    # CRPS = E|X - y| - 0.5 * E|X - X'|
    # where X, X' are independent draws from forecast distribution
    
    term1 = np.mean(np.abs(forecast_samples - realized))
    
    # Approximate second term with pairwise differences
    n = len(forecast_samples)
    if n > 100:
        # Subsample for efficiency
        idx = np.random.choice(n, 100, replace=False)
        samples = forecast_samples[idx]
    else:
        samples = forecast_samples
    
    diffs = []
    for i in range(len(samples)):
        for j in range(i+1, len(samples)):
            diffs.append(np.abs(samples[i] - samples[j]))
    
    term2 = 0.5 * np.mean(diffs) if diffs else 0
    
    crps = term1 - term2
    return crps
```

### 1.3 Computational Details

**Efficiency Optimization:**
- Full pairwise computation: O(n²) with n=10,000 samples → 50 million comparisons
- **Subsampling:** Use 100 random samples for term2 → 4,950 comparisons
- Speedup: ~10,000x faster with minimal accuracy loss

**Alternative Exact Formula:**
For Gaussian forecast `F ~ N(μ, σ²)`:
```
CRPS(F, y) = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]

where:
  z = (y - μ) / σ
  Φ = standard normal CDF
  φ = standard normal PDF
```

This is **not used** in the codebase (would require distributional assumptions), but could be added for speed.

### 1.4 Interpretation

**Example Values:**
- CRPS = 10 bps: Forecast is off by ~10 bps on average
- CRPS = 50 bps: Poor forecast (large errors or overconfident)
- CRPS = 2 bps: Excellent forecast

**Comparison to RMSE:**
- RMSE only uses forecast mean (point forecast)
- CRPS uses full distribution (better for probabilistic forecasts)
- CRPS ≈ RMSE when forecast is Gaussian and well-calibrated

**Target:** CRPS < 20 bps for yield forecasts at 6-12 month horizons.

### 1.5 Usage in Backtest

**Code Location:** `src/afns/validation.py`, lines 264

```python
crps = self._compute_crps(forecast_samples, realized)
```

Called for each (origin, horizon, variable) combination in the rolling backtest.

---

## 2. PIT (Probability Integral Transform)

### 2.1 Mathematical Definition

The **Probability Integral Transform** evaluates the **quantile rank** of the realized value within the forecast distribution.

**Formula:**
```
PIT = F(y)

where:
  F = forecast CDF
  y = realized value
```

**Interpretation:**
- PIT ∈ [0, 1]
- PIT = 0.5: Realized value is at the median of forecast
- PIT = 0.05: Realized value is in lower 5% of forecast (unexpected low)
- PIT = 0.95: Realized value is in upper 5% of forecast (unexpected high)

**Calibration Property:**
If forecasts are well-calibrated, **PIT should be uniform U(0,1)**:
```
PIT ~ Uniform(0, 1)  if F is correctly specified
```

**Why?**
- If forecasts are too narrow: PIT will cluster near 0 and 1 (realized values often outside forecast range)
- If forecasts are too wide: PIT will cluster near 0.5 (realized values always near median)
- If forecasts are biased: PIT will be skewed (e.g., systematically over-predicting)

### 2.2 Code Implementation

**Code Location:** `src/afns/validation.py`, lines 85-102

```python
def _compute_pit(self, forecast_samples, realized):
    """
    Compute Probability Integral Transform.
    
    Parameters
    ----------
    forecast_samples : np.array
        Forecast distribution samples
    realized : float
        Realized value
        
    Returns
    -------
    float
        PIT value (should be uniform [0,1] if calibrated)
    """
    pit = (forecast_samples < realized).mean()
    return pit
```

**Implementation:**
- Empirical CDF: `F(y) ≈ (# samples < y) / (total samples)`
- Simple and non-parametric
- Works for any distribution shape

### 2.3 PIT Histogram

**Expected Histogram:**
If forecasts are well-calibrated, the PIT histogram should be **flat** (uniform).

**Example:**
```
PIT Bin    Expected    Observed (Good)    Observed (Biased)
[0.0-0.1]     10%           9%                 5%    (under-predicting)
[0.1-0.2]     10%          11%                 8%
[0.2-0.3]     10%          10%                12%
[0.3-0.4]     10%           9%                11%
[0.4-0.5]     10%          11%                13%
[0.5-0.6]     10%          10%                15%
[0.6-0.7]     10%          11%                12%
[0.7-0.8]     10%           9%                10%
[0.8-0.9]     10%          10%                 8%
[0.9-1.0]     10%          10%                 6%
```

**Diagnostic:**
- **U-shaped:** Forecasts too narrow (over-confident)
- **Inverse U-shaped:** Forecasts too wide (under-confident)
- **Skewed left:** Systematic over-prediction
- **Skewed right:** Systematic under-prediction

### 2.4 Average PIT

**Expected Value:**
```
E[PIT] = 0.5  if forecasts are unbiased
```

**Code Location:** `src/afns/validation.py`, lines 302-304

```python
# PIT should be ~0.5 on average
pit_mean = results_df['pit'].mean()
logger.info(f"  Average PIT: {pit_mean:.3f} (target: 0.5)")
```

**Interpretation:**
- PIT mean = 0.5: Unbiased forecasts
- PIT mean < 0.5: Systematic over-prediction (realized values below forecast median)
- PIT mean > 0.5: Systematic under-prediction (realized values above forecast median)

**Target:** 0.45 ≤ Average PIT ≤ 0.55

### 2.5 Usage in Backtest

**Code Location:** `src/afns/validation.py`, lines 265

```python
pit = self._compute_pit(forecast_samples, realized)
```

Called for each (origin, horizon, variable) combination.

---

## 3. Coverage Calculation

### 3.1 Definition

**Coverage** measures whether realized values fall within forecast prediction intervals.

**Formula:**
```
Coverage = P(y ∈ [q_lower, q_upper])

where:
  q_lower = lower_pct percentile of forecast
  q_upper = upper_pct percentile of forecast
```

**Standard Interval:** [5%, 95%] → Expected coverage = 90%

### 3.2 Code Implementation

**Code Location:** `src/afns/validation.py`, lines 104-124

```python
def _compute_coverage(self, forecast_samples, realized, lower_pct=5, upper_pct=95):
    """
    Check if realized value falls in forecast interval.
    
    Parameters
    ----------
    forecast_samples : np.array
        Forecast distribution samples
    realized : float
        Realized value
    lower_pct, upper_pct : float
        Percentile bounds
        
    Returns
    -------
    bool
        True if realized is within interval
    """
    lower = np.percentile(forecast_samples, lower_pct)
    upper = np.percentile(forecast_samples, upper_pct)
    return lower <= realized <= upper
```

### 3.3 Coverage Rate

**Aggregated Coverage:**
```
Coverage Rate = (# forecasts with y ∈ [q5, q95]) / (# total forecasts)
```

**Code Location:** `src/afns/validation.py`, lines 266, 306-313

```python
coverage = self._compute_coverage(forecast_samples, realized)

# Later in summary:
coverage_mean = results_df['coverage'].mean()
logger.info(f"  Average coverage (5-95%): {coverage_mean:.3f} (target: 0.90)")

if 0.85 <= coverage_mean <= 0.95:
    logger.info("  ✓ Coverage is well-calibrated")
else:
    logger.warning("  ⚠ Coverage may be miscalibrated")
```

### 3.4 Interpretation

**Expected Coverage:** 90% for [5%, 95%] interval

**Diagnostics:**
- **Coverage = 95%:** Forecasts too wide (conservative)
- **Coverage = 85%:** Forecasts too narrow (over-confident)
- **Coverage = 70%:** Severely miscalibrated (dangerous)

**Target:** 85% ≤ Coverage ≤ 95%

---

## 4. Rolling Origin Backtest

### 4.1 Forecast Points

The backtest uses **annual rolling origins** to avoid overfitting:

**Code Location:** `src/afns/validation.py`, lines 144-152

```python
# Define rolling origins (annual steps)
origins = pd.date_range(
    self.initial_train_end,
    self.data_df.index[-1],
    freq='AS'  # Annual start
)[:-1]  # Exclude last one if too recent

logger.info(f"  Rolling origins: {len(origins)}")
logger.info(f"  Origins: {[str(d.date()) for d in origins]}")
```

**Example Origins:**
- 2019-01-01
- 2020-01-01
- 2021-01-01
- 2022-01-01

**Why Annual?**
- Provides ~4-6 out-of-sample test points (depending on data length)
- Avoids data mining (not too frequent)
- Balances training data volume and test coverage

### 4.2 Horizons Tested

**Standard Horizons:**
- **1-month:** 21 business days
- **3-month:** 63 business days
- **12-month:** 252 business days

**Code Location:** `src/afns/validation.py`, line 35

```python
def __init__(self, data_df, initial_train_end='2018-12-31',
             test_horizons=[21, 63, 252], config=None):
    self.test_horizons = test_horizons
```

### 4.3 Training/Test Split

**For each origin:**

**Code Location:** `src/afns/validation.py`, lines 165-180

```python
for origin in origins:
    logger.info(f"\n  Origin: {origin.date()}")
    
    # Split data
    train_data = self.data_df[self.data_df.index <= origin]
    
    if len(train_data) < 252:  # Need at least 1 year of data
        logger.warning(f"    Insufficient training data, skipping")
        continue
    
    # Fit models on training data
    logger.info(f"    Fitting models on {len(train_data)} observations...")
```

**Training Data:** All data up to and including origin date
**Test Data:** Realized values at `origin + horizon`

**Example:**
- Origin: 2020-01-01
- Training: Data from start to 2020-01-01
- Test (6-month): Realized values on 2020-07-01
- Test (12-month): Realized values on 2021-01-01

**Expanding Window:** Training data grows with each origin (not a rolling window).

---

## 5. Other Metrics

### 5.1 RMSE (Root Mean Squared Error)

**Formula:**
```
RMSE = sqrt(E[(ŷ - y)²])

where:
  ŷ = forecast mean (point forecast)
  y = realized value
```

**Code Location:** `src/afns/validation.py`, lines 261-262

```python
forecast_mean = forecast_samples.mean()
rmse = np.sqrt((forecast_mean - realized) ** 2)
```

**Note:** This is the RMSE for a **single forecast**. Aggregated RMSE is computed later:
```python
results_df.groupby(['horizon', 'variable']).agg({'rmse': 'mean'})
```

**Interpretation:**
- Point forecast accuracy metric
- **Does not** assess forecast uncertainty or calibration
- Complementary to CRPS

### 5.2 MAE (Mean Absolute Error)

**Formula:**
```
MAE = E[|ŷ - y|]
```

**Code Location:** `src/afns/validation.py`, line 263

```python
mae = np.abs(forecast_mean - realized)
```

**Interpretation:**
- More robust to outliers than RMSE
- Units same as variable (e.g., bps)
- Easier to interpret than RMSE

### 5.3 Summary Statistics

**Code Location:** `src/afns/validation.py`, lines 286-296

```python
if len(results_df) > 0:
    summary = results_df.groupby(['horizon', 'variable']).agg({
        'rmse': 'mean',
        'mae': 'mean',
        'crps': 'mean',
        'pit': 'mean',
        'coverage': 'mean'
    })
    
    logger.info("\nMean metrics by horizon and variable:")
    logger.info(f"\n{summary}")
```

**Output Example:**
```
                      rmse    mae   crps    pit  coverage
horizon variable                                         
21      UST_10       15.2   12.3   10.5   0.51      0.88
        Delta_UST_10 14.8   11.9   10.2   0.49      0.90
63      UST_10       28.5   23.1   22.3   0.52      0.89
        Delta_UST_10 27.9   22.7   21.8   0.48      0.91
252     UST_10       65.3   52.4   50.1   0.50      0.87
        Delta_UST_10 64.1   51.8   49.3   0.51      0.88
```

---

## 6. Code Locations Summary

### 6.1 Main Functions

| Function | Lines | Description |
|----------|-------|-------------|
| `__init__()` | 34-43 | Initialize validator with horizons and config |
| `_compute_crps()` | 45-83 | Compute CRPS from samples and realized |
| `_compute_pit()` | 85-102 | Compute PIT from samples and realized |
| `_compute_coverage()` | 104-124 | Check if realized in forecast interval |
| `run_backtest()` | 126-325 | Main rolling-origin backtest loop |
| `compare_baselines()` | 327-353 | Compare to random walk baseline |
| `test_lambda_sensitivity()` | 355-408 | Test AFNS sensitivity to λ parameter |

### 6.2 Metrics Computation

**For each (origin, horizon, variable):**

```python
# Compute metrics
forecast_mean = forecast_samples.mean()
rmse = np.sqrt((forecast_mean - realized) ** 2)
mae = np.abs(forecast_mean - realized)
crps = self._compute_crps(forecast_samples, realized)
pit = self._compute_pit(forecast_samples, realized)
coverage = self._compute_coverage(forecast_samples, realized)

# Store results
results['origin'].append(origin)
results['horizon'].append(horizon)
results['variable'].append(var)
results['rmse'].append(rmse)
results['mae'].append(mae)
results['crps'].append(crps)
results['pit'].append(pit)
results['coverage'].append(coverage)
```

**Code Location:** `src/afns/validation.py`, lines 260-276

---

## 7. Visualization (Conceptual)

### 7.1 CRPS Over Time

**Plot:** CRPS vs. forecast origin (for each horizon)

```
CRPS (bps)
    ^
 50 |           *
    |     *           *
 40 |                       *
    | *       *
 30 |   *               *
    +----------------------------> Origin
    2019  2020  2021  2022
```

**Interpretation:**
- Spikes in CRPS → Poor forecasts during turbulent periods
- Consistent low CRPS → Good model performance

### 7.2 PIT Histogram

**Plot:** Histogram of all PIT values

```
Frequency
    ^
 20 | ===  ===  ===  ===  ===  ===  ===  ===  ===  ===
    | ===  ===  ===  ===  ===  ===  ===  ===  ===  ===
 10 | ===  ===  ===  ===  ===  ===  ===  ===  ===  ===
    | ===  ===  ===  ===  ===  ===  ===  ===  ===  ===
  0 +-------------------------------------------------> PIT
    0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0
```

**Well-Calibrated:** Flat histogram (uniform distribution)

### 7.3 Coverage by Horizon

**Plot:** Coverage percentage vs. forecast horizon

```
Coverage (%)
    ^
100 |
    |
 90 |---  Target  ---  ---  ---  ---  ---
    |      *           *
 80 |          *               *
    +----------------------------> Horizon (days)
     21        63           252
```

**Interpretation:**
- Coverage near 90%: Good calibration
- Coverage declining with horizon: Model uncertainty grows appropriately

---

## 8. Calibration Checks

### 8.1 PIT Uniformity Test

**Statistical Test:** Kolmogorov-Smirnov test for uniformity

```python
from scipy.stats import kstest

pit_values = results_df['pit'].values
ks_stat, p_value = kstest(pit_values, 'uniform')

if p_value > 0.05:
    print("✓ PIT is uniform (well-calibrated)")
else:
    print("✗ PIT deviates from uniform (miscalibrated)")
```

**Not currently in codebase** but could be added for formal testing.

### 8.2 Coverage Confidence Interval

**For 90% nominal coverage:**

```
95% CI for coverage = 0.90 ± 1.96 * sqrt(0.90 * 0.10 / n)
```

where n = number of forecasts.

Example with n=20 forecasts:
```
CI = 0.90 ± 1.96 * sqrt(0.09 / 20) = 0.90 ± 0.13 = [0.77, 1.00]
```

**Code Location:** Currently logs warning if outside [0.85, 0.95] (lines 310-313)

---

## 9. Usage Example

```python
from src.afns.validation import RollingValidator
from src.afns.utils import load_config

# Load data and config
data_df = pd.read_parquet('outputs/data/aligned_data.parquet')
config = load_config('config.yaml')

# Initialize validator
validator = RollingValidator(
    data_df=data_df,
    initial_train_end='2018-12-31',
    test_horizons=[21, 63, 252],
    config=config
)

# Run backtest
results = validator.run_backtest(n_paths=1000)

# Access results
results_df = results['results_df']
summary = results['summary']

# Print summary
print(summary)

# Examine specific metrics
print("\nCRPS by horizon:")
print(results_df.groupby('horizon')['crps'].mean())

print("\nPIT by variable:")
print(results_df.groupby('variable')['pit'].mean())

# Check calibration
pit_mean = results_df['pit'].mean()
coverage_mean = results_df['coverage'].mean()

print(f"\n✓ Average PIT: {pit_mean:.3f} (target: 0.5)")
print(f"✓ Average coverage: {coverage_mean:.3f} (target: 0.9)")
```

---

## 10. References

**File:** `src/afns/validation.py`

**Key Code Sections:**
- Lines 45-83: CRPS computation
- Lines 85-102: PIT computation
- Lines 104-124: Coverage computation
- Lines 126-325: Rolling-origin backtest
- Lines 260-276: Metrics calculation loop
- Lines 286-313: Summary statistics and calibration checks

**Dependencies:**
- `numpy`: Numerical operations
- `pandas`: Data handling
- `scipy`: Could add statistical tests

**References:**
- Gneiting & Raftery (2007): "Strictly Proper Scoring Rules, Prediction, and Estimation"
- Gneiting & Katzfuss (2014): "Probabilistic Forecasting"

---

This implementation provides rigorous probabilistic forecast evaluation through:
1. **CRPS:** Comprehensive forecast quality metric
2. **PIT:** Calibration diagnostic (should be uniform)
3. **Coverage:** Interval reliability check
4. **Rolling origin:** Realistic out-of-sample testing
5. **Multiple horizons:** Performance across time scales

