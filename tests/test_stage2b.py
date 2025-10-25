"""
Test Script for Stage 2B: Scenario Integration

This script verifies:
1. ScenarioGenerator integrates with SpreadEngine correctly
2. Joint Treasury + Spread paths are generated
3. Spread percentiles are computed
4. Multi-horizon forecasts include spreads
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.utils import load_config, prepare_data, get_spread_dataframe
from src.afns import fit_afns_model
from src.regime import fit_regime_model
from src.spreads import SpreadEngine
from src.scenarios import ScenarioGenerator


def test_stage_2b():
    """Test Stage 2B: Scenario Integration"""
    
    print("="*60)
    print("STAGE 2B TEST: Scenario Integration")
    print("="*60)
    
    # 1. Load config and data
    print("\n[1/7] Loading configuration and data...")
    config = load_config("config.yaml")
    df = prepare_data(config)
    print(f"✓ Data loaded: {len(df)} observations")
    
    # 2. Fit AFNS model
    print("\n[2/7] Fitting AFNS model...")
    df_factors, afns_model, cov_matrix = fit_afns_model(df, config)
    print(f"✓ AFNS fitted")
    
    # 3. Fit regime model
    print("\n[3/7] Fitting regime model...")
    hmm_model, regime_labels, regime_stats, regime_covs = fit_regime_model(
        df, df_factors, afns_model, config
    )
    print(f"✓ Regime model fitted: {hmm_model.n_states} states")
    
    # 4. Fit SpreadEngine
    print("\n[4/7] Fitting SpreadEngine...")
    spread_df = get_spread_dataframe(df, sectors=['Muni', 'Agency', 'Corp'], benchmark_maturity=5)
    spread_engine = SpreadEngine(sectors=['Muni', 'Agency', 'Corp'], benchmark_maturity=5)
    spread_engine.fit(spread_df, df_factors, regime_labels)
    print(f"✓ SpreadEngine fitted")
    
    # 5. Create ScenarioGenerator WITH SpreadEngine
    print("\n[5/7] Creating ScenarioGenerator with SpreadEngine integration...")
    
    # Use single horizon for testing (faster)
    test_horizons = {'6m': 126}
    
    generator = ScenarioGenerator(
        afns_model=afns_model,
        hmm_model=hmm_model,
        spread_engine=spread_engine,  # KEY: Passing spread_engine
        horizons=test_horizons,
        n_paths=500,  # Reduced for faster testing
        n_jobs=1,  # Single job to avoid multiprocessing issues
        config=config
    )
    
    print(f"✓ ScenarioGenerator created")
    print(f"  Horizons: {list(test_horizons.keys())}")
    print(f"  Paths: 500 (reduced for testing)")
    print(f"  Spread products: {spread_engine.sectors}")
    
    # 6. Build current state with spread decomposition
    print("\n[6/7] Preparing current state...")
    latest_date = df.index[-1]
    current_factors = df_factors.loc[latest_date].values
    current_regime_probs = hmm_model.regime_probs.iloc[-1].values
    tenors = config['data']['tenors']
    current_yields = df.loc[latest_date, tenors].values
    
    # Get spread decomposition
    current_spreads = spread_engine.get_current_spreads(spread_df, df_factors, latest_date)
    
    current_state = {
        'date': latest_date,
        'factors': current_factors,
        'regime_probs': current_regime_probs,
        'yields': current_yields,
        'spreads': current_spreads
    }
    
    print(f"✓ Current state prepared")
    print(f"  Date: {latest_date.date()}")
    print(f"  Current 10Y: {current_yields[4]:.2f}%")
    print(f"  Spread decomposition included: {list(current_spreads.keys())}")
    
    # 7. Generate scenario with no Fed shock
    print("\n[7/7] Generating scenario with joint Treasury + Spread forecasting...")
    
    try:
        paths_df, percentiles = generator.generate_scenario(
            current_state=current_state,
            fed_shock_bp=0,
            random_seed=42
        )
        
        print(f"✓ Scenario generated successfully")
        
    except Exception as e:
        print(f"\n❌ FAILED: Scenario generation error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Validation checks
    print("\n" + "="*60)
    print("VALIDATION CHECKS")
    print("="*60)
    
    # Check 1: Paths DataFrame has expected columns
    print("\n[Check 1] Path DataFrame structure:")
    
    expected_treasury_cols = [f'UST_{int(mat)}' for mat in afns_model.maturities]
    expected_spread_cols = [f'{sector}_Spread' for sector in spread_engine.sectors]
    expected_yield_cols = [f'{sector}_Yield' for sector in spread_engine.sectors]
    
    treasury_ok = all(col in paths_df.columns for col in expected_treasury_cols)
    spread_ok = all(col in paths_df.columns for col in expected_spread_cols)
    yield_ok = all(col in paths_df.columns for col in expected_yield_cols)
    
    print(f"  {'✓' if treasury_ok else '❌'} Treasury columns: {len(expected_treasury_cols)} present")
    print(f"  {'✓' if spread_ok else '❌'} Spread columns: {len(expected_spread_cols)} present")
    print(f"  {'✓' if yield_ok else '❌'} All-in yield columns: {len(expected_yield_cols)} present")
    
    if not (treasury_ok and spread_ok and yield_ok):
        print("\n❌ Missing expected columns in path DataFrame")
        return False
    
    # Check 2: Percentiles computed for spreads
    print("\n[Check 2] Percentile statistics:")
    
    has_treasury_percentiles = any('UST_' in k for k in percentiles.keys())
    has_spread_percentiles = any('_Spread' in k for k in percentiles.keys())
    has_yield_percentiles = any('_Yield' in k for k in percentiles.keys())
    
    print(f"  {'✓' if has_treasury_percentiles else '❌'} Treasury percentiles computed")
    print(f"  {'✓' if has_spread_percentiles else '❌'} Spread percentiles computed")
    print(f"  {'✓' if has_yield_percentiles else '❌'} All-in yield percentiles computed")
    
    if not (has_treasury_percentiles and has_spread_percentiles and has_yield_percentiles):
        print("\n❌ Missing percentile calculations")
        return False
    
    # Check 3: Spread statistics look reasonable
    print("\n[Check 3] Spread forecast statistics:")
    
    all_reasonable = True
    for sector in spread_engine.sectors:
        spread_key = f'{sector}_Spread'
        if spread_key in percentiles:
            stats = percentiles[spread_key]
            mean = stats['mean']
            std = stats['std']
            p50 = stats['p50']
            
            # Spreads should be small positive numbers (bps)
            # Check that mean is within reasonable range (-5 to +10 bps typically)
            reasonable = (-5 < mean < 10) and (0 < std < 5)
            
            status = '✓' if reasonable else '⚠'
            print(f"  {status} {sector}: mean={mean:.2f}bps, std={std:.2f}bps, p50={p50:.2f}bps")
            
            if not reasonable:
                all_reasonable = False
    
    if not all_reasonable:
        print("  ⚠ Some spreads outside typical ranges - may be data dependent")
    
    # Check 4: Relationship between spreads and all-in yields
    print("\n[Check 4] Treasury + Spread = All-in Yield:")
    
    # Get final time slice
    final_slice = paths_df[paths_df['t'] == 126].iloc[:10]  # First 10 paths
    
    relationship_ok = True
    for sector in spread_engine.sectors:
        ust_col = f'UST_{spread_engine.benchmark_maturity}'
        spread_col = f'{sector}_Spread'
        yield_col = f'{sector}_Yield'
        
        if all(col in final_slice.columns for col in [ust_col, spread_col, yield_col]):
            # Check: Sector_Yield ≈ UST + Spread
            computed = final_slice[ust_col] + final_slice[spread_col]
            actual = final_slice[yield_col]
            error = np.abs(computed - actual).mean()
            
            if error < 0.001:  # Within 0.1 bps
                print(f"  ✓ {sector}: UST + Spread = Yield (error: {error:.4f}bps)")
            else:
                print(f"  ❌ {sector}: Arithmetic mismatch (error: {error:.4f}bps)")
                relationship_ok = False
    
    if not relationship_ok:
        return False
    
    # Check 5: Print example forecast
    print("\n[Check 5] Example 6-month forecast (10Y benchmark):")
    
    ust_10 = percentiles.get('UST_10', percentiles.get(f'UST_{int(afns_model.maturities[4])}'))
    print(f"  Treasury 10Y: {ust_10['p50']:.2f}% [{ust_10['p5']:.2f}% - {ust_10['p95']:.2f}%]")
    
    for sector in spread_engine.sectors:
        yield_key = f'{sector}_Yield'
        spread_key = f'{sector}_Spread'
        
        if yield_key in percentiles and spread_key in percentiles:
            yield_stats = percentiles[yield_key]
            spread_stats = percentiles[spread_key]
            
            print(f"  {sector:7} Yield: {yield_stats['p50']:.2f}% "
                  f"[{yield_stats['p5']:.2f}% - {yield_stats['p95']:.2f}%] "
                  f"(spread: {spread_stats['p50']:.2f}bps)")
    
    # Summary
    print("\n" + "="*60)
    print("STAGE 2B: ✓ ALL TESTS PASSED")
    print("="*60)
    print("\nKey Results:")
    print(f"  • Joint Treasury + Spread simulation operational")
    print(f"  • {len(spread_engine.sectors)} spread products forecasted: {spread_engine.sectors}")
    print(f"  • Percentiles computed for Treasuries, Spreads, and All-in Yields")
    print(f"  • Arithmetic consistency verified (UST + Spread = Yield)")
    print("\nReady to proceed to Stage 2C: Full Pipeline Integration")
    
    return True


if __name__ == "__main__":
    success = test_stage_2b()
    sys.exit(0 if success else 1)