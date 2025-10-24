"""
Validation: Historical accuracy, coverage tests, DV01-weighted metrics, CRPS, PIT
"""
import numpy as np
import pandas as pd
from scipy import stats


def _compute_crps(forecast_samples, realized):
    """
    Compute Continuous Ranked Probability Score
    
    Args:
        forecast_samples: Array of forecast distribution samples
        realized: Realized value
        
    Returns:
        crps: CRPS score (lower is better, in same units as variable)
    """
    # CRPS = E|X - y| - 0.5 * E|X - X'|
    
    # Term 1: Mean absolute error
    term1 = np.mean(np.abs(forecast_samples - realized))
    
    # Term 2: Approximate spread penalty with subsampling for efficiency
    n = len(forecast_samples)
    if n > 100:
        # Subsample for efficiency
        idx = np.random.choice(n, 100, replace=False)
        samples = forecast_samples[idx]
    else:
        samples = forecast_samples
    
    # Compute pairwise differences
    diffs = []
    for i in range(len(samples)):
        for j in range(i+1, len(samples)):
            diffs.append(np.abs(samples[i] - samples[j]))
    
    term2 = 0.5 * np.mean(diffs) if diffs else 0
    
    crps = term1 - term2
    return crps


def _compute_pit(forecast_samples, realized):
    """
    Compute Probability Integral Transform
    
    Args:
        forecast_samples: Array of forecast distribution samples
        realized: Realized value
        
    Returns:
        pit: PIT value (should be ~0.5 if unbiased, uniform if calibrated)
    """
    # PIT = Empirical CDF(realized)
    pit = (forecast_samples < realized).mean()
    return pit


def backtest_scenarios(df, df_factors, afns_model, hmm_model, 
                       regime_labels, regime_covs, config):
    """
    Run historical backtest: generate scenarios and compare to realized outcomes
    
    Args:
        df: DataFrame with yield curve data
        df_factors: DataFrame with AFNS factors
        afns_model: Fitted AFNS model
        hmm_model: Fitted StickyHMM model
        regime_labels: Historical regime assignments (Series)
        regime_covs: Covariance matrices by regime
        config: Configuration dict
        
    Returns:
        backtest_results: DataFrame with forecasts and realized values
    """
    from tqdm import tqdm
    
    test_start = pd.Timestamp(config['validation']['test_start'])
    test_dates = df.index[df.index >= test_start]
    
    tenors = config['data']['tenors']
    results = []
    
    # For each test date, generate 1-day ahead scenario and compare to realized
    for i in tqdm(range(len(test_dates) - 1), desc="Backtesting", position=1, leave=False):
        date = test_dates[i]
        idx = df.index.get_loc(date)
        next_date = test_dates[i + 1]
        
        # Current state
        current_factors = df_factors.iloc[idx][['L', 'S', 'C']].values
        # regime_labels is a Series indexed by date, use .loc
        try:
            current_regime = regime_labels.loc[date]
        except:
            current_regime = regime_labels.iloc[-1]
        
        # Generate scenario (assume no Fed shock for 1-day ahead)
        # Use regime-specific covariance from HMM
        cov = regime_covs[current_regime]
        base_factors = current_factors.copy()
        
        # Generate random shocks
        factor_shocks = np.random.multivariate_normal(
            mean=np.zeros(3),
            cov=cov,
            size=1000
        )
        sim_factors = base_factors + factor_shocks
        sim_yields = afns_model.reconstruct_yields(sim_factors)
        
        # Compute percentiles
        confidence_levels = config['scenarios']['confidence_levels']
        p_low = np.percentile(sim_yields, confidence_levels[0]*100, axis=0)
        p50 = np.percentile(sim_yields, confidence_levels[1]*100, axis=0)
        p_high = np.percentile(sim_yields, confidence_levels[2]*100, axis=0)
        
        # Get realized yields
        next_idx = df.index.get_loc(next_date)
        realized = df.iloc[next_idx][tenors].values
        
        # Store results with CRPS and PIT
        for j, tenor in enumerate(tenors):
            # Compute metrics
            forecast_mean = sim_yields[:, j].mean()
            rmse = np.sqrt((forecast_mean - realized[j]) ** 2)
            mae = np.abs(forecast_mean - realized[j])
            crps = _compute_crps(sim_yields[:, j], realized[j])
            pit = _compute_pit(sim_yields[:, j], realized[j])
            coverage = (realized[j] >= p_low[j]) and (realized[j] <= p_high[j])
            
            results.append({
                'date': date,
                'next_date': next_date,
                'tenor': tenor,
                'regime': current_regime,
                'forecast_p10': p_low[j],
                'forecast_p50': p50[j],
                'forecast_p90': p_high[j],
                'realized': realized[j],
                'rmse': rmse,
                'mae': mae,
                'crps': crps,
                'pit': pit,
                'in_band': coverage
            })
    
    df_backtest = pd.DataFrame(results)
    
    return df_backtest


def compute_coverage(df_backtest):
    """
    Compute coverage rates: do 90% bands contain 90% of outcomes?
    
    Args:
        df_backtest: DataFrame from backtest_scenarios
        
    Returns:
        coverage_stats: Dict with coverage by tenor and regime
    """
    coverage_stats = {}
    
    # Overall coverage
    overall_coverage = df_backtest['in_band'].mean()
    coverage_stats['overall'] = overall_coverage
    
    # Coverage by tenor
    coverage_stats['by_tenor'] = df_backtest.groupby('tenor')['in_band'].mean().to_dict()
    
    # Coverage by regime
    coverage_stats['by_regime'] = df_backtest.groupby('regime')['in_band'].mean().to_dict()
    
    return coverage_stats


def compute_dv01_weighted_errors(df_backtest, config):
    """
    Compute DV01-weighted forecast errors
    
    Args:
        df_backtest: DataFrame from backtest_scenarios
        config: Configuration dict
        
    Returns:
        metrics: Dict with DV01-weighted MAE, RMSE
    """
    tenors = config['data']['tenors']
    dv01_weights = np.array(config['validation']['dv01_weights'])
    
    # Compute errors for each tenor
    errors = []
    for i, tenor in enumerate(tenors):
        tenor_data = df_backtest[df_backtest['tenor'] == tenor]
        error = tenor_data['realized'] - tenor_data['forecast_p50']
        errors.append(error.values)
    
    errors = np.array(errors)  # (n_tenors, n_dates)
    
    # Weight by DV01
    weighted_errors = errors * dv01_weights[:, np.newaxis]
    
    # Compute metrics
    mae = np.mean(np.abs(weighted_errors))
    rmse = np.sqrt(np.mean(weighted_errors**2))
    
    metrics = {
        'dv01_mae': mae,
        'dv01_rmse': rmse,
        'unweighted_mae': np.mean(np.abs(errors)),
        'unweighted_rmse': np.sqrt(np.mean(errors**2))
    }
    
    return metrics


def compute_shape_accuracy(df_backtest, config):
    """
    Compute cosine similarity between forecast and realized curve shapes
    
    Args:
        df_backtest: DataFrame from backtest_scenarios
        config: Configuration dict
        
    Returns:
        avg_cosine: Average cosine similarity
    """
    tenors = config['data']['tenors']
    dates = df_backtest['date'].unique()
    
    cosine_sims = []
    
    for date in dates:
        date_data = df_backtest[df_backtest['date'] == date]
        
        # Get forecast and realized curves
        forecast = date_data.set_index('tenor')['forecast_p50'].reindex(tenors).values
        realized = date_data.set_index('tenor')['realized'].reindex(tenors).values
        
        # Compute cosine similarity
        cos_sim = np.dot(forecast, realized) / (np.linalg.norm(forecast) * np.linalg.norm(realized))
        cosine_sims.append(cos_sim)
    
    return np.mean(cosine_sims)


def subsample_stability(df, df_factors, afns_model, hmm_model, 
                        regime_labels, regime_covs, config):
    """
    Test stability across different sub-periods
    
    Args:
        All model components and config
        
    Returns:
        stability_results: Dict with metrics by sub-period
    """
    from tqdm import tqdm
    
    subsample_splits = config['validation']['subsample_splits']
    stability_results = {}
    
    for i, (start, end) in enumerate(tqdm(subsample_splits, desc="Sub-sample tests", position=1, leave=False)):
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)
        
        # Filter data for this period
        mask_df = (df.index >= start_date) & (df.index <= end_date)
        df_sub = df[mask_df]
        
        # df_factors has same length as df, regime_labels has 1 less row
        mask_factors = (df_factors.index >= start_date) & (df_factors.index <= end_date)
        df_factors_sub = df_factors[mask_factors]
        
        # regime_labels is a Series, filter by date range
        regime_labels_sub = regime_labels[(regime_labels.index >= start_date) & 
                                          (regime_labels.index <= end_date)]
        
        # Run backtest on this subsample
        df_backtest_sub = backtest_scenarios(
            df_sub, df_factors_sub, afns_model, hmm_model,
            regime_labels_sub, regime_covs, config
        )
        
        # Only compute metrics if we have data
        if len(df_backtest_sub) > 0:
            # Compute metrics
            coverage = compute_coverage(df_backtest_sub)
            dv01_metrics = compute_dv01_weighted_errors(df_backtest_sub, config)
            shape_acc = compute_shape_accuracy(df_backtest_sub, config)
            
            stability_results[f"period_{i+1}"] = {
                'dates': (start, end),
                'coverage': coverage['overall'],
                'dv01_mae': dv01_metrics['dv01_mae'],
                'cosine_similarity': shape_acc
            }
        else:
            stability_results[f"period_{i+1}"] = {
                'dates': (start, end),
                'coverage': None,
                'dv01_mae': None,
                'cosine_similarity': None,
                'note': 'No data in test period'
            }
    
    return stability_results


def run_validation(df, df_factors, afns_model, hmm_model, 
                   regime_labels, regime_covs, config):
    """
    Main validation function
    
    Args:
        All model components and config
        
    Returns:
        validation_results: Dict with all validation metrics
    """
    print("\n=== Running Validation ===")
    
    # 1. Run backtest
    print("Running historical backtest...")
    df_backtest = backtest_scenarios(
        df, df_factors, afns_model, hmm_model,
        regime_labels, regime_covs, config
    )
    
    # 2. Compute coverage
    print("Computing coverage rates...")
    coverage_stats = compute_coverage(df_backtest)
    
    # 3. Compute DV01-weighted errors
    print("Computing DV01-weighted errors...")
    dv01_metrics = compute_dv01_weighted_errors(df_backtest, config)
    
    # 4. Compute shape accuracy
    print("Computing shape accuracy...")
    shape_acc = compute_shape_accuracy(df_backtest, config)
    
    # 5. Sub-sample stability
    print("Testing sub-sample stability...")
    stability = subsample_stability(
        df, df_factors, afns_model, hmm_model,
        regime_labels, regime_covs, config
    )
    
    # Compile results
    validation_results = {
        'coverage': coverage_stats,
        'dv01_metrics': dv01_metrics,
        'cosine_similarity': shape_acc,
        'stability': stability,
        'backtest_data': df_backtest
    }
    
    # Print summary with CRPS and PIT
    print("\n=== Validation Results ===")
    print(f"Overall Coverage: {coverage_stats['overall']:.1%}")
    print(f"Target: 90% (tolerance: {config['validation']['coverage_tolerance']})")
    
    print(f"\nDV01-Weighted MAE: {dv01_metrics['dv01_mae']:.4f}")
    print(f"Unweighted MAE: {dv01_metrics['unweighted_mae']:.4f}")
    
    print(f"\nCosine Similarity (Shape): {shape_acc:.4f}")
    
    # CRPS Metrics
    if 'crps' in df_backtest.columns:
        print(f"\nCRPS Metrics (bps):")
        tenors = config['data']['tenors']
        for tenor in tenors:
            avg_crps = df_backtest[df_backtest['tenor']==tenor]['crps'].mean()
            print(f"  {tenor}: {avg_crps:.2f} bps")
        
        overall_crps = df_backtest['crps'].mean()
        print(f"  Overall: {overall_crps:.2f} bps")
        
        if overall_crps < 20:
            print("  ✓ CRPS < 20 bps (good forecast quality)")
        else:
            print("  ⚠ CRPS > 20 bps (poor forecast quality)")
    
    # PIT Calibration
    if 'pit' in df_backtest.columns:
        pit_mean = df_backtest['pit'].mean()
        print(f"\nPIT Calibration:")
        print(f"  Average PIT: {pit_mean:.3f} (target: 0.50)")
        
        if 0.45 <= pit_mean <= 0.55:
            print("  ✓ Well-calibrated (unbiased forecast)")
        elif pit_mean < 0.45:
            print("  ⚠ Biased high (over-predicting)")
        else:
            print("  ⚠ Biased low (under-predicting)")
    
    print("\nSub-sample Stability:")
    for period, stats in stability.items():
        print(f"  {stats['dates'][0]} to {stats['dates'][1]}:")
        if stats['coverage'] is not None:
            print(f"    Coverage: {stats['coverage']:.1%}")
            print(f"    DV01 MAE: {stats['dv01_mae']:.4f}")
            print(f"    Cosine Sim: {stats['cosine_similarity']:.4f}")
        else:
            print(f"    {stats.get('note', 'No data')}")
    
    return validation_results


def validate_conformal(df, df_factors, afns_model, hmm_model,
                       regime_labels, regime_covs, config):
    """Validate conformal predictor using train/calibration/test split."""
    from src.conformal import ConformalScenarioGenerator, compute_crps

    print("\n=== Conformal Validation (Train/Cal/Test Split) ===")

    train_end = pd.Timestamp('2018-12-31')
    cal_end = pd.Timestamp('2021-12-31')
    test_start = pd.Timestamp('2022-01-01')

    train_mask = df.index <= train_end
    cal_mask = (df.index > train_end) & (df.index <= cal_end)
    test_mask = df.index >= test_start

    print(f"Train: {df.index[train_mask][0].date()} to {df.index[train_mask][-1].date()}")
    print(f"Calibration: {df.index[cal_mask][0].date()} to {df.index[cal_mask][-1].date()}")
    print(f"Test: {df.index[test_mask][0].date()} to {df.index[test_mask][-1].date()}")

    # Create a simple base generator for conformal (not the full path simulator)
    class SimpleGenerator:
        def __init__(self, afns_model, config):
            self.afns = afns_model
            self.config = config
    
    base_generator = SimpleGenerator(afns_model, config)
    conformal_generator = ConformalScenarioGenerator(base_generator, confidence_level=0.90)

    calibration_dates = df.index[cal_mask]
    conformal_generator.calibrate(df, df_factors, regime_labels, calibration_dates, regime_covs)

    df_test = df[test_mask]

    tenors = config['data']['tenors']
    results = []

    print("Generating conformal forecasts on test set...")
    for i in range(len(df_test) - 1):
        date = df_test.index[i]
        next_date = df_test.index[i + 1]

        idx = df.index.get_loc(date)
        current_factors = df_factors.iloc[idx][['L', 'S', 'C']].values
        current_yields = df.iloc[idx][tenors].values
        
        # Get regime from Series
        try:
            current_regime = int(regime_labels.loc[date])
        except:
            current_regime = int(regime_labels.iloc[-1])

        current_state = {
            'date': date,
            'yields': current_yields,
            'factors': current_factors,
            'regime': current_regime
        }

        # Generate simple single-step scenario for conformal
        cov = regime_covs[current_regime] if current_regime in regime_covs else regime_covs[0]
        factor_shocks = np.random.multivariate_normal(mean=np.zeros(3), cov=cov, size=1000)
        sim_factors = current_factors + factor_shocks
        sim_yields = afns_model.reconstruct_yields(sim_factors)
        
        # Create scenario dict matching old format
        scenario = {
            'percentiles': {}
        }
        for j, tenor in enumerate(tenors):
            q_lower, q_upper = conformal_generator.quantiles_by_regime[current_regime][tenor]
            forecast = np.median(sim_yields[:, j])
            scenario['percentiles'][tenor] = {
                'p5': forecast + q_lower,
                'p50': forecast,
                'p95': forecast + q_upper
            }

        next_idx = df.index.get_loc(next_date)
        realized = df.iloc[next_idx][tenors].values

        for j, tenor in enumerate(tenors):
            percs = scenario['percentiles'][tenor]
            results.append({
                'date': date,
                'next_date': next_date,
                'tenor': tenor,
                'regime': current_regime,
                'forecast_p5': percs['p5'],
                'forecast_p50': percs['p50'],
                'forecast_p95': percs['p95'],
                'realized': realized[j],
                'in_band': (realized[j] >= percs['p5']) and (realized[j] <= percs['p95'])
            })

    df_results = pd.DataFrame(results)

    coverage = df_results['in_band'].mean()
    coverage_by_tenor = df_results.groupby('tenor')['in_band'].mean()
    coverage_by_regime = df_results.groupby('regime')['in_band'].mean()

    dv01_metrics = compute_dv01_weighted_errors(df_results, config)

    shape_acc = compute_shape_accuracy(df_results, config)

    dates = df_results['date'].unique()
    crps_scores = []
    for date in dates:
        date_data = df_results[df_results['date'] == date]
        forecasts = []
        realized = []
        for _, row in date_data.iterrows():
            forecasts.append({
                'p5': row['forecast_p5'],
                'p50': row['forecast_p50'],
                'p95': row['forecast_p95']
            })
            realized.append(row['realized'])
        crps_scores.append(compute_crps(forecasts, np.array(realized)))

    avg_crps = np.mean(crps_scores)

    print(f"\n=== Test Set Results (Held-Out) ===")
    print(f"Coverage: {coverage:.1%}")
    print("Target: 90% ± 5% (85-95%)")
    print(f"\nCoverage by Tenor:")
    for tenor, cov in coverage_by_tenor.items():
        print(f"  {tenor}: {cov:.1%}")
    print(f"\nCoverage by Regime:")
    for regime, cov in coverage_by_regime.items():
        print(f"  Regime {regime}: {cov:.1%}")
    print(f"\nDV01-Weighted MAE: {dv01_metrics['dv01_mae']:.4f}")
    print(f"Cosine Similarity: {shape_acc:.4f}")
    print(f"Average CRPS: {avg_crps:.4f}")

    return {
        'coverage': coverage,
        'coverage_by_tenor': coverage_by_tenor.to_dict(),
        'coverage_by_regime': coverage_by_regime.to_dict(),
        'dv01_metrics': dv01_metrics,
        'cosine_similarity': shape_acc,
        'crps': avg_crps,
        'test_data': df_results
    }
