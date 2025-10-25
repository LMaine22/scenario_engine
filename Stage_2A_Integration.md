# Stage 2A Integration Guide: Spread Foundation

## What Was Built

Stage 2A implements the **hierarchical spread decomposition** with regime-dependent dynamics:

```
Spread_sector(t) = β_sector · [L, S, C] + z_sector(t)
                   ↑                      ↑
              Systematic               Residual
           (Treasury effects)    (Credit/Liquidity)
```

**Residual dynamics (regime-dependent):**
```
z_t+1 = Γ_s · z_t + μ_s + ξ_t,  where ξ_t ~ N(0, Ψ_s)
```

## Files Created/Modified

1. **NEW: `src/spreads.py`**
   - `SpreadEngine` class with full hierarchical decomposition
   - Factor loading estimation via OLS
   - Regime-dependent residual VAR
   - Spread forecasting and simulation methods

2. **UPDATED: `src/utils.py`**
   - Added `compute_spreads()` function
   - Computes Muni/Agency/Corp spreads from Bloomberg data
   - Added `get_spread_dataframe()` helper
   - Integrated into `prepare_data()` pipeline

3. **NEW: `test_spreads_stage2a.py`**
   - Comprehensive test script
   - Validates all components work correctly

4. **NEW: `STAGE_2A_INTEGRATION.md`**
   - This file - integration instructions

## Installation Steps

### 1. Copy Files to Your Project

```bash
# From /mnt/user-data/outputs/ to your project root:
cp /mnt/user-data/outputs/spreads.py src/spreads.py
cp /mnt/user-data/outputs/utils.py src/utils.py
cp /mnt/user-data/outputs/test_spreads_stage2a.py test_spreads_stage2a.py
```

### 2. Run Test Script

```bash
python test_spreads_stage2a.py
```

**Expected output:**
- ✓ Spreads computed (Muni, Agency, Corp at 5Y)
- ✓ AFNS factors extracted
- ✓ Regime model fitted
- ✓ SpreadEngine fitted with factor loadings
- ✓ Regime-dependent VAR estimated
- ✓ Economic validation passed
- ✓ Forecast test passed

## What the Test Validates

### Factor Loadings (β)
Should match economic intuition:
- **Munis**: High Slope sensitivity (β_S dominant) - carry dynamics
- **Agencies**: High Level sensitivity (β_L dominant) - prepayment risk
- **Corps**: High Curvature/Level sensitivity - credit/recession signals

### Regime-Dependent Dynamics
For each regime (Calm, Volatile, Crisis):
- **Γ_s**: Mean reversion matrix (eigenvalues < 1 for stability)
- **μ_s**: Long-run mean spread levels
- **Ψ_s**: Covariance matrix (higher in crisis)

### Stability Check
All Γ_s matrices must have max|eigenvalue| < 1.0 for stable dynamics.

## Interpreting Results

### Example Factor Loadings:
```
Muni:    β = [-0.15, -0.45, 0.08]  ← Slope dominant (carry)
Agency:  β = [ 0.18,  0.12, 0.06]  ← Level dominant (prepay)
Corp:    β = [ 0.25,  0.10, 0.35]  ← Level+Curve (credit)
```

### Example Regime Parameters:
```
Calm Regime (0):
  Γ diagonal = [0.95, 0.97, 0.93]  ← Fast mean reversion
  μ = [0, 0, 0]                    ← Tight spreads
  
Crisis Regime (2):
  Γ diagonal = [0.60, 0.75, 0.50]  ← Slow mean reversion
  μ = [30, 20, 200]                ← Wide spreads (Corps blown out)
```

## Troubleshooting

### Issue: "Missing spread columns"
**Solution:** Check that your data has BVMB5Y, BVCSUG05, IGUUAC05 columns. If using different maturity, update `benchmark_maturity` parameter.

### Issue: "Unstable Γ matrix (max|eig| > 1)"
**Cause:** Insufficient data in that regime
**Solution:** This is okay - model will use pooled estimates. Check if you have <30 observations in that regime.

### Issue: "Factor loadings don't match intuition"
**Investigate:** 
1. Check if spreads are in correct units (should be bps)
2. Check if data quality issues (outliers, breaks)
3. Look at R² values - if <0.3, weak Treasury factor relationship

## Data Requirements

The system needs:
1. **Treasury yields**: USGG2YR, USGG5YR, USGG7YR, USGG10YR
2. **Muni yields**: BVMB5Y Index
3. **Agency yields**: BVCSUG05 BVLI Index
4. **Corp yields**: IGUUAC05 BVLI Index

All at daily frequency, aligned dates.

## Next Steps

Once Stage 2A test passes:
- **Stage 2B**: Integrate SpreadEngine with ScenarioGenerator
- **Stage 2C**: Update validation and full pipeline

Do NOT proceed to Stage 2B until test script shows all checks passing.

## Technical Details

### Estimation Procedure:

**Step 1: Factor Loadings**
```
For each sector:
  1. Regress Spread_sector ~ L + S + C
  2. Extract β coefficients
  3. Compute residuals z = Spread - β·factors
  4. Validate R² > 0.3 (Treasury factors explain >30%)
```

**Step 2: Residual VAR**
```
For each regime:
  1. Filter residuals to that regime
  2. Estimate VAR(1): z_t = Γ·z_{t-1} + μ + ξ_t
  3. Check stability: max|eig(Γ)| < 1
  4. Estimate Ψ from VAR residuals
```

### Key Methods in SpreadEngine:

- `fit()`: Main estimation routine
- `forecast_spreads()`: Standalone spread forecasting
- `simulate_spreads()`: Integration point for ScenarioGenerator
- `get_current_spreads()`: Decompose observed spread into systematic + residual
- `validate_economic_intuition()`: Check β makes sense

## Questions?

If test fails or results look wrong, share:
1. The exact error message
2. The factor loadings printed (β values)
3. The regime parameters (Γ, μ, Ψ)
4. Data summary (how many observations, date range)

---

**Status: Stage 2A Complete ✓**
**Next: Stage 2B - Scenario Integration**