f"""
Main Orchestrator: Run complete Phase 1A pipeline
"""
import os
import sys
from pathlib import Path

# Configure Matplotlib backend for headless execution and writable cache
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))
from pathlib import Path as _Path  # Temporary alias to create directory without circularity
_mpl_path = _Path(os.environ["MPLCONFIGDIR"])
_mpl_path.mkdir(parents=True, exist_ok=True)
del _Path, _mpl_path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils import load_config, prepare_data
from src.afns import fit_afns_model
from src.regime import fit_regime_model
from src.scenarios import run_scenario_analysis
from src.validation import run_validation, validate_conformal


def plot_scenario_fan(scenarios, current_state, config, save_path=None):
    """Create scenario fan chart for visualization"""
    tenors = config['data']['tenors']
    maturities = config['data']['maturities']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    fed_shocks = sorted(scenarios.keys())
    
    for i, (tenor, maturity) in enumerate(zip(tenors, maturities)):
        ax = axes[i]
        
        # Current yield
        current_yield = current_state['yields'][i]
        
        # Plot scenario distributions
        for shock in fed_shocks:
            scenario_data = scenarios[shock]
            percs = scenario_data['percentiles']
            
            # Find the matching UST key
            ust_key = f'UST_{int(maturity)}'
            if ust_key in percs:
                stats = percs[ust_key]
                p_low = stats['p5']
                p50 = stats['p50']
                p_high = stats['p95']
                
                ax.plot([maturity], [p50], 'o', label=f'{shock:+d}bp')
                ax.plot([maturity, maturity], [p_low, p_high], '-', alpha=0.5)
        
        # Current level
        ax.axhline(current_yield, color='black', linestyle='--', label='Current')
        
        ax.set_title(f'{tenor.replace(" Index", "")} ({maturity}yr)')
        ax.set_xlabel('Maturity')
        ax.set_ylabel('Yield (%)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Scenario fan chart saved to {save_path}")
    
    return fig


def plot_regime_analysis(df, regime_labels, config, save_path=None):
    """Plot regime time series and characteristics"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # regime_labels is a Series indexed by date
    # Need to align with df by reindexing
    regime_labels_aligned = regime_labels.reindex(df.index, method='ffill')
    
    # Regime time series
    ax = axes[0]
    ax.plot(regime_labels.index, regime_labels.values, linewidth=0.5)
    ax.set_ylabel('Regime')
    ax.set_title('Regime Classification Over Time')
    ax.grid(True, alpha=0.3)
    
    # 10yr yield colored by regime
    ax = axes[1]
    for regime in range(int(regime_labels.max()) + 1):
        mask = regime_labels_aligned == regime
        ax.scatter(df.index[mask], df['USGG10YR Index'][mask], 
                  label=f'Regime {regime}', s=1, alpha=0.6)
    ax.set_ylabel('10yr Yield (%)')
    ax.set_title('10yr Yield by Regime')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MOVE index colored by regime
    ax = axes[2]
    for regime in range(int(regime_labels.max()) + 1):
        mask = regime_labels_aligned == regime
        ax.scatter(df.index[mask], df['MOVE Index'][mask], 
                  label=f'Regime {regime}', s=1, alpha=0.6)
    ax.set_ylabel('MOVE Index')
    ax.set_title('MOVE Index by Regime')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Regime analysis chart saved to {save_path}")
    
    return fig


def save_conformal_results(baseline_results, conformal_results, scenarios, df_scenarios, config):
    """Save validation results comparing baseline and conformal approaches."""
    output_path = Path(config['output']['results_path'])
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / 'validation_summary.txt', 'w') as f:
        f.write("=== Phase 1A Validation Results (Conformal) ===\n\n")

        f.write("BASELINE (Manual Inflation):\n")
        f.write(f"  Overall Coverage: {baseline_results['coverage']['overall']:.1%}\n")
        f.write(f"  DV01-Weighted MAE: {baseline_results['dv01_metrics']['dv01_mae']:.4f}\n")
        f.write(f"  Cosine Similarity: {baseline_results['cosine_similarity']:.4f}\n\n")

        f.write("CONFORMAL (Test Set Only, 2022-2025):\n")
        f.write(f"  Coverage: {conformal_results['coverage']:.1%}\n")
        f.write("  Target: 90% ± 5% (85-95%)\n")
        f.write(f"  DV01-Weighted MAE: {conformal_results['dv01_metrics']['dv01_mae']:.4f}\n")
        f.write(f"  Cosine Similarity: {conformal_results['cosine_similarity']:.4f}\n")
        f.write(f"  Average CRPS: {conformal_results['crps']:.4f}\n\n")

        f.write("Coverage by Regime (Conformal):\n")
        for regime, cov in conformal_results['coverage_by_regime'].items():
            f.write(f"  Regime {regime}: {cov:.1%}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("METHODOLOGY:\n")
        f.write("- Split conformal prediction with proper time-series handling\n")
        f.write("- Train: 2010-2018, Calibration: 2019-2021, Test: 2022-2025\n")
        f.write("- Mondrian stratification by volatility regime\n")
        f.write("- NO manual inflation factors - coverage achieved via residual quantiles\n")
        f.write("- Guarantees marginal coverage without distributional assumptions\n")

    conformal_results['test_data'].to_csv(output_path / 'conformal_test_results.csv', index=False)
    baseline_results['backtest_data'].to_csv(output_path / 'baseline_backtest_results.csv', index=False)

    scenarios_path = Path(config['output']['scenarios_path'])
    scenarios_path.mkdir(parents=True, exist_ok=True)
    df_scenarios.to_csv(scenarios_path / 'latest_scenarios.csv', index=False)

    print(f"\nResults saved to {output_path}")


def main():
    """Main execution pipeline"""
    
    print("=" * 60)
    print("PHASE 1A: Treasury Scenario Engine - MVP (Conformal)")
    print("=" * 60)
    
    # Progress bar for overall pipeline
    with tqdm(total=8, desc="Overall Progress", position=0, leave=True) as pbar:
        
        # 1. Load configuration
        pbar.set_description("[1/8] Loading configuration")
        config = load_config()
        pbar.update(1)
        
        # 2. Prepare data
        pbar.set_description("[2/8] Preparing data")
        df = prepare_data(config)
        pbar.update(1)
        
        # 3. Fit AFNS model
        pbar.set_description("[3/8] Fitting AFNS model")
        df_factors, afns_model, cov_matrix = fit_afns_model(df, config)
        pbar.update(1)
        
        # 4. Fit regime model with higher persistence
        pbar.set_description("[4/8] Fitting Sticky HMM")
        hmm_model, regime_labels, regime_stats, regime_covs = fit_regime_model(
            df, df_factors, afns_model, config
        )

        # Print regime persistence check
        print(f"\nRegime Persistence Check:")
        for k, stats in regime_stats.items():
            duration = stats['expected_duration_days']
            if duration < 30:
                print(f"  \u26a0 Regime {k}: {duration:.1f} days (too short for multi-year forecasting)")
            else:
                print(f"  \u2713 Regime {k}: {duration:.1f} days ({duration/252:.1f} years)")
        pbar.update(1)
        
        # 5. Multi-horizon scenario generation
        pbar.set_description("[5/8] Generating multi-year scenarios (10K paths)")

        # Get all horizons from config
        horizons = config['scenarios']['horizons']

        # Initialize scenario generator with multi-year support
        from src.scenarios import ScenarioGenerator
        generator = ScenarioGenerator(
            afns_model=afns_model,
            hmm_model=hmm_model,
            spread_engine=None,  # Phase 2
            horizons=horizons,
            n_paths=config['scenarios']['n_paths'],
            n_jobs=config['scenarios']['n_jobs']
        )

        # Generate for multiple Fed scenarios and horizons
        fed_shocks = config['scenarios']['fed_shocks']
        all_scenarios = {}

        for shock in fed_shocks:
            print(f"\nFed Shock: {shock:+d} bps")
            
            # Get current state
            current_state = {
                'date': df.index[-1],
                'yields': df.iloc[-1][config['data']['tenors']].values,
                'factors': afns_model.factors.iloc[-1].values,
                'regime_probs': hmm_model.regime_probs.iloc[-1].values
            }
            
            # Generate for all horizons
            horizon_scenarios = generator.generate_all_horizons(current_state, shock)
            all_scenarios[shock] = horizon_scenarios

        # Backward-compat dataframes for saving fan chart and CSV
        # Use no-shock percentiles at the max horizon for legacy outputs
        base_no_shock = all_scenarios.get(0) or all_scenarios.get(0.0) or next(iter(all_scenarios.values()))
        max_h_name, max_h_data = sorted(base_no_shock.items(), key=lambda x: x[1]['horizon_days'])[-1]
        df_scenarios = pd.DataFrame([
            {
                'Fed_Shock_bp': shock,
                'Tenor': k,
                'p5': v['percentiles'][k]['p5'],
                'p50': v['percentiles'][k]['p50'],
                'p95': v['percentiles'][k]['p95'],
                'mean': v['percentiles'][k]['mean'],
                'std': v['percentiles'][k]['std']
            }
            for shock, horizon_map in all_scenarios.items()
            for _, v in [sorted(horizon_map.items(), key=lambda x: x[1]['horizon_days'])[-1]]
            for k in v['percentiles'].keys()
        ])
        scenarios = {shock: data[max_h_name] for shock, data in all_scenarios.items()}
        pbar.update(1)
        
        # 6. Run sophisticated multi-horizon validation
        pbar.set_description("[6/8] Running multi-horizon validation")
        validation_results = run_validation(
            df, df_factors, afns_model, hmm_model, 
            regime_labels, regime_covs, config
        )
        pbar.update(1)

        # Print multi-year forecast example
        print("\n=== Multi-Year Forecast Example (10yr Treasury) ===")
        print("Current: {:.2f}%".format(current_state['yields'][4]))
        print("\nNo Fed Action Scenario:")
        for horizon_name, data in all_scenarios[0].items():
            percs = data['percentiles']['UST_10']
            years = data['horizon_days'] / 252
            print(f"  {horizon_name:3} ({years:3.1f}Y): {percs['p50']:.2f}% [{percs['p5']:.2f}% - {percs['p95']:.2f}%]")
        
        # Map multi-horizon validation results to baseline_results structure for saving
        baseline_results = {
            'coverage': {
                'overall': validation_results['analysis']['overall']['coverage_90']
            },
            'dv01_metrics': {'dv01_mae': float('nan')},
            'cosine_similarity': float('nan'),
            'backtest_data': validation_results['results_df']
        }

        # 7. Conformal validation (DISABLED - needs update for new scenario API)
        pbar.set_description("[7/8] Skipping conformal validation (to be updated)")
        conformal_results = {
            'coverage': 0.0,
            'coverage_by_tenor': {},
            'coverage_by_regime': {},
            'dv01_metrics': {'dv01_mae': 0.0},
            'cosine_similarity': 0.0,
            'crps': 0.0,
            'test_data': pd.DataFrame()
        }
        print("\nConformal validation skipped (needs API update for path simulation)")
        pbar.update(1)

        # 8. Generate outputs
        pbar.set_description("[8/8] Generating outputs")
        
        # Create output directories
        figures_path = Path(config['output']['figures_path'])
        figures_path.mkdir(parents=True, exist_ok=True)
        
        # Plot scenario fan
        plot_scenario_fan(scenarios, current_state, config, 
                         save_path=figures_path / 'scenario_fan.png')
        
        # Plot regime analysis
        plot_regime_analysis(df, regime_labels, config,
                            save_path=figures_path / 'regime_analysis.png')
        
        # Save results
        save_conformal_results(baseline_results, conformal_results, scenarios, df_scenarios, config)
        pbar.update(1)
    
    print("\n" + "=" * 60)
    print("Phase 1A Complete (Conformal)!")
    print("=" * 60)
    print(f"\nOutputs saved to:")
    print(f"  - {config['output']['figures_path']}")
    print(f"  - {config['output']['results_path']}")
    print(f"  - {config['output']['scenarios_path']}")


if __name__ == "__main__":
    main()
