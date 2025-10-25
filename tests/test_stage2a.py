"""
Test Script for Stage 2A: Spread Foundation

This script verifies:
1. Spreads are computed correctly from raw data
2. SpreadEngine fits hierarchical model successfully
3. Factor loadings make economic sense
4. Regime-dependent dynamics are estimated
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


def test_stage_2a():
    """Test Stage 2A: Spread Foundation"""
    
    print("="*60)
    print("STAGE 2A TEST: Spread Foundation")
    print("="*60)
    
    # 1. Load config and data
    print("\n[1/5] Loading configuration and data...")
    config = load_config("config.yaml")
    df = prepare_data(config)
    
    # Check that spreads were computed
    spread_cols = ['Muni_Spread_5Y', 'Agency_Spread_5Y', 'Corp_Spread_5Y']
    missing = [col for col in spread_cols if col not in df.columns]
    
    if missing:
        print(f"\n❌ FAILED: Missing spread columns: {missing}")
        return False
    
    print(f"✓ Spreads computed successfully")
    print(f"  Columns: {spread_cols}")
    
    # 2. Fit AFNS model (need factors for spread decomposition)
    print("\n[2/5] Fitting AFNS model to get Treasury factors...")
    df_factors, afns_model, cov_matrix = fit_afns_model(df, config)
    print(f"✓ AFNS model fitted")
    
    # 3. Fit regime model (need regimes for regime-dependent dynamics)
    print("\n[3/5] Fitting regime model...")
    hmm_model, regime_labels, regime_stats, regime_covs = fit_regime_model(
        df, df_factors, afns_model, config
    )
    print(f"✓ Regime model fitted with {hmm_model.n_states} states")
    
    # 4. Extract spread DataFrame
    print("\n[4/5] Preparing spread data for SpreadEngine...")
    spread_df = get_spread_dataframe(df, sectors=['Muni', 'Agency', 'Corp'], benchmark_maturity=5)
    print(f"✓ Spread DataFrame prepared: {spread_df.shape}")
    print(f"  Date range: {spread_df.index.min()} to {spread_df.index.max()}")
    
    # 5. Fit SpreadEngine
    print("\n[5/5] Fitting SpreadEngine with hierarchical decomposition...")
    spread_engine = SpreadEngine(sectors=['Muni', 'Agency', 'Corp'], benchmark_maturity=5)
    
    try:
        results = spread_engine.fit(spread_df, df_factors, regime_labels)
        print(f"✓ SpreadEngine fitted successfully")
    except Exception as e:
        print(f"\n❌ FAILED: SpreadEngine fitting error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Validate results
    print("\n" + "="*60)
    print("VALIDATION CHECKS")
    print("="*60)
    
    # Check 1: Factor loadings exist for all sectors
    print("\n[Check 1] Factor loadings estimated:")
    all_sectors_ok = True
    for sector in ['Muni', 'Agency', 'Corp']:
        if sector in spread_engine.factor_loadings:
            beta = spread_engine.factor_loadings[sector]
            print(f"  ✓ {sector}: β = [{beta[0]:.3f}, {beta[1]:.3f}, {beta[2]:.3f}]")
        else:
            print(f"  ❌ {sector}: Missing!")
            all_sectors_ok = False
    
    if not all_sectors_ok:
        return False
    
    # Check 2: Regime-dependent parameters exist
    print(f"\n[Check 2] Regime-dependent VAR parameters:")
    n_regimes = hmm_model.n_states
    regime_params_ok = True
    
    for regime in range(n_regimes):
        if regime in spread_engine.Gamma and regime in spread_engine.mu and regime in spread_engine.Psi:
            Gamma = spread_engine.Gamma[regime]
            mu = spread_engine.mu[regime]
            Psi = spread_engine.Psi[regime]
            
            # Check stability
            eigenvalues = np.linalg.eigvals(Gamma)
            max_eig = np.max(np.abs(eigenvalues))
            stable = max_eig < 1.0
            
            status = "✓" if stable else "⚠"
            print(f"  {status} Regime {regime}: max|eig|={max_eig:.3f}, μ={mu}, stable={stable}")
        else:
            print(f"  ❌ Regime {regime}: Missing parameters!")
            regime_params_ok = False
    
    if not regime_params_ok:
        return False
    
    # Check 3: Economic validation
    print(f"\n[Check 3] Economic intuition validation:")
    spread_engine.validate_economic_intuition()
    
    # Check 4: Test forecast functionality
    print(f"\n[Check 4] Testing spread forecasting...")
    try:
        # Use latest observation
        latest_date = df.index[-1]
        factors_current = df_factors.loc[latest_date].values
        regime_current = int(regime_labels.iloc[-1])
        
        # Get current spreads
        current_spreads = spread_engine.get_current_spreads(spread_df, df_factors, latest_date)
        z_current = np.array([
            current_spreads['Muni']['residual'],
            current_spreads['Agency']['residual'],
            current_spreads['Corp']['residual']
        ])
        
        # Forecast 6 months ahead
        spread_paths = spread_engine.forecast_spreads(
            factors_current=factors_current,
            regime_current=regime_current,
            z_current=z_current,
            horizon=126,
            n_paths=1000,
            random_seed=42
        )
        
        print(f"  ✓ Forecast generated for {len(spread_paths)} sectors")
        
        # Check dimensions
        for sector, paths in spread_paths.items():
            expected_shape = (1000, 127)  # n_paths × (horizon + 1)
            if paths.shape == expected_shape:
                print(f"    ✓ {sector}: shape {paths.shape}")
            else:
                print(f"    ❌ {sector}: shape {paths.shape}, expected {expected_shape}")
                return False
                
    except Exception as e:
        print(f"  ❌ Forecast test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "="*60)
    print("STAGE 2A: ✓ ALL TESTS PASSED")
    print("="*60)
    print("\nKey Results:")
    print(f"  • Spreads computed for 3 sectors at 5Y benchmark")
    print(f"  • Factor loadings estimated (β captures Treasury effects)")
    print(f"  • Residual VAR estimated for {n_regimes} regimes")
    print(f"  • Spread forecasting operational")
    print("\nReady to proceed to Stage 2B: Scenario Integration")
    
    return True


if __name__ == "__main__":
    success = test_stage_2a()
    sys.exit(0 if success else 1)