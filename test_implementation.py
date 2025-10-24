"""
Quick validation checklist script
"""
import sys
import pandas as pd
import numpy as np
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Import modules
from src.afns import AFNSModel
from src.regime import StickyHMM
from src.scenarios import ScenarioGenerator

# Load data
print("Loading data...")
df = pd.read_parquet('data/raw/RatesRegimes.parquet')

# Set 'Date' column as index if it exists
if 'Date' in df.columns:
    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)
else:
    df.index = pd.to_datetime(df.index)

print(f"Data loaded: {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")

# Test 1: AFNS
print("\n" + "="*60)
print("Test 1: AFNS Model")
print("="*60)
tenors = config['data']['tenors']
maturities = config['data']['maturities']
yields_df = df[tenors]

afns = AFNSModel(
    maturities=maturities,
    lambda_param=config['afns']['lambda'],
    measurement_noise_R=config['afns']['measurement_noise_R']
)

results = afns.fit(yields_df, max_iter=100)

# Check AFNS results
assert 'A' in results, "❌ Missing transition matrix A"
assert 'transition_matrix' not in results, "❌ Should be in regime, not afns"
assert 'overall_rmse' in results, "❌ Missing overall_rmse"

rmse = results['overall_rmse']
print(f"\n✓ AFNS passed: RMSE = {rmse:.2f} bps")

if rmse < 10:
    print(f"  ✓ RMSE < 10bp threshold (good fit)")
else:
    print(f"  ⚠ RMSE >= 10bp (acceptable but could be better)")

# Test 2: HMM
print("\n" + "="*60)
print("Test 2: Sticky HMM")
print("="*60)

hmm = StickyHMM(
    n_states_range=config['regime']['n_states_range'],
    sticky_alpha=config['regime']['sticky_alpha'],
    random_state=config['regime']['random_state']
)

hmm_results = hmm.fit(afns.factor_changes)

# Check HMM results
assert 'transition_matrix' in hmm_results, "❌ Missing transition matrix"
assert hmm_results['transition_matrix'].shape[0] == hmm_results['n_states'], "❌ Transition matrix shape mismatch"
assert 'n_states' in hmm_results, "❌ Missing n_states"

trans_mat = hmm_results['transition_matrix']
n_states = hmm_results['n_states']
print(f"\n✓ HMM passed: {n_states} states detected")
print(f"  Transition matrix diagonal: {np.diag(trans_mat)}")

# Check diagonal > 0.80
diag_min = np.diag(trans_mat).min()
if diag_min > 0.80:
    print(f"  ✓ All diagonal elements > 0.80 (sticky regimes)")
else:
    print(f"  ⚠ Min diagonal = {diag_min:.3f} < 0.80 (less sticky)")

# Test 3: Paths
print("\n" + "="*60)
print("Test 3: Scenario Path Generation")
print("="*60)

# Set up current state
current_idx = -1  # Last date
current_date = df.index[current_idx]
current_factors = afns.factors.iloc[current_idx].values
current_yields = yields_df.iloc[current_idx].values

current_state = {
    'date': current_date,
    'yields': current_yields,
    'factors': current_factors,
    'regime': hmm.get_most_likely_regimes().iloc[-1],
    'spreads': {}
}

# Create generator with small number of paths for quick test
gen = ScenarioGenerator(
    afns_model=afns,
    hmm_model=hmm,
    spread_engine=None,
    n_paths=100,  # Small for testing
    horizon_days=10,  # Short horizon for testing
    n_jobs=1  # Single process for testing
)

# Generate a simple scenario
paths_df, percentiles = gen.generate_scenario(current_state, fed_shock_bp=0, random_seed=42)

# Check paths were generated
assert paths_df is not None and len(paths_df) > 0, "❌ No paths generated"
assert len(percentiles) > 0, "❌ No percentiles computed"

print(f"\n✓ Paths generated: {len(paths_df)} rows ({gen.n_paths} paths × {gen.horizon_days + 1} steps)")
print(f"  Percentiles computed for {len(percentiles)} tenors")

# Sample output
print(f"\n  Sample: 10yr yield at horizon")
ust10_stats = percentiles['UST_10']
print(f"    Current: {current_yields[4]:.2f}%")
print(f"    p5:  {ust10_stats['p5']:.2f}%")
print(f"    p50: {ust10_stats['p50']:.2f}%")
print(f"    p95: {ust10_stats['p95']:.2f}%")

# Final summary
print("\n" + "="*60)
print("VALIDATION SUMMARY")
print("="*60)
print(f"✓ Test 1: AFNS fit successful (RMSE = {rmse:.2f}bp)")
print(f"✓ Test 2: HMM fit successful ({n_states} states, min diag = {diag_min:.3f})")
print(f"✓ Test 3: Paths generated successfully")

print("\n" + "="*60)
print("CHECKLIST:")
print("="*60)
print(f"  {'✓' if rmse < 10 else '⚠'} RMSE < 10bp? {rmse:.2f}bp")
print(f"  {'✓' if diag_min > 0.80 else '⚠'} Transition matrix diagonal > 0.80? {diag_min:.3f}")
print(f"  ✓ Paths generated? Yes")

print("\n" + "="*60)
if rmse < 10 and diag_min > 0.80:
    print("✓ ALL CHECKS PASSED - Implementation likely correct!")
    print("\nNext step: Run 'python main.py' to check coverage > 80%")
else:
    print("⚠ Some checks did not meet ideal thresholds, but implementation is functional")
    print("  This may be acceptable depending on data characteristics")
print("="*60)

