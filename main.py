"""
Main Orchestrator: Run complete Phase 2 pipeline with Treasury + Spread forecasting
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
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils import load_config, prepare_data, get_spread_dataframe
from src.afns import fit_afns_model
from src.regime import fit_regime_model
from src.spreads import SpreadEngine
from src.scenarios import ScenarioGenerator
from src.validation import run_validation


def plot_scenario_fan(scenarios, current_state, config, save_path=None):
    """Create scenario fan chart for Treasury visualization"""
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


def plot_spread_scenarios(scenarios, current_state, spread_engine, config, save_path=None):
    """
    Create spread scenario visualization (Phase 2 Enhancement)
    
    Shows forecasted spreads for each sector under different Fed scenarios
    """
    if spread_engine is None or not spread_engine.is_fitted:
        print("Spread engine not available - skipping spread visualization")
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sectors = spread_engine.sectors
    fed_shocks = sorted(scenarios.keys())
    benchmark_mat = spread_engine.benchmark_maturity
    
    for i, (sector, ax) in enumerate(zip(sectors, axes)):
        spread_key = f'{sector}_Spread'
        yield_key = f'{sector}_Yield'
        
        # Get current values
        if 'spreads' in current_state and sector in current_state['spreads']:
            current_spread = current_state['spreads'][sector]['total']
        else:
            current_spread = 0.0
        
        # Plot forecasts for each Fed scenario
        for shock in fed_shocks:
            scenario_data = scenarios[shock]
            percs = scenario_data['percentiles']
            
            if spread_key in percs:
                stats = percs[spread_key]
                p_low = stats['p5']
                p50 = stats['p50']
                p_high = stats['p95']
                
                # Plot median with error bars
                x_pos = shock / 100  # Convert bps to position
                ax.errorbar(x_pos, p50, 
                           yerr=[[p50 - p_low], [p_high - p50]],
                           fmt='o', capsize=5, capthick=2,
                           label=f'{shock:+d}bp Fed')
        
        # Current spread
        ax.axhline(current_spread, color='black', linestyle='--', 
                  linewidth=2, label='Current')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_title(f'{sector} Spread ({benchmark_mat}Y)', fontweight='bold')
        ax.set_xlabel('Fed Policy Shock (bp)')
        ax.set_ylabel('Spread (bps)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Spread Forecasts by Fed Scenario', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Spread scenario chart saved to {save_path}")
    
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


def create_spread_forecast_table(scenarios, spread_engine, config, save_path=None):
    """
    Create detailed spread forecast table (Phase 2)
    
    Shows forecasted yields for Treasuries and each spread product
    """
    if spread_engine is None or not spread_engine.is_fitted:
        return None
    
    rows = []
    benchmark_mat = spread_engine.benchmark_maturity
    
    for shock in sorted(scenarios.keys()):
        scenario_data = scenarios[shock]
        percs = scenario_data['percentiles']
        
        # Treasury
        ust_key = f'UST_{benchmark_mat}'
        if ust_key in percs:
            ust_stats = percs[ust_key]
            rows.append({
                'Fed_Shock': f'{shock:+d}bp',
                'Product': f'{benchmark_mat}Y Treasury',
                'P5': f"{ust_stats['p5']:.2f}%",
                'Median': f"{ust_stats['p50']:.2f}%",
                'P95': f"{ust_stats['p95']:.2f}%",
                'Mean': f"{ust_stats['mean']:.2f}%",
                'Std': f"{ust_stats['std']:.2f}%"
            })
        
        # Spread products
        for sector in spread_engine.sectors:
            yield_key = f'{sector}_Yield'
            spread_key = f'{sector}_Spread'
            
            if yield_key in percs and spread_key in percs:
                yield_stats = percs[yield_key]
                spread_stats = percs[spread_key]
                
                rows.append({
                    'Fed_Shock': f'{shock:+d}bp',
                    'Product': f'{benchmark_mat}Y {sector}',
                    'P5': f"{yield_stats['p5']:.2f}%",
                    'Median': f"{yield_stats['p50']:.2f}%",
                    'P95': f"{yield_stats['p95']:.2f}%",
                    'Mean': f"{yield_stats['mean']:.2f}%",
                    'Std': f"{yield_stats['std']:.2f}% (spr: {spread_stats['p50']:.2f}bps)"
                })
    
    df_table = pd.DataFrame(rows)
    
    if save_path:
        df_table.to_csv(save_path, index=False)
        print(f"Spread forecast table saved to {save_path}")
    
    return df_table


def save_results(validation_results, scenarios, config, spread_engine=None):
    """Save validation and scenario results"""
    output_path = Path(config['output']['results_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save validation results
    validation_results['results_df'].to_csv(
        output_path / 'validation_results.csv', 
        index=False
    )
    
    # Save analysis summary
    with open(output_path / 'validation_summary.txt', 'w') as f:
        f.write("=== Phase 2 Validation Results ===\n\n")
        analysis = validation_results['analysis']
        
        f.write("Overall Metrics:\n")
        f.write(f"  Mean CRPS: {analysis['overall']['mean_crps']:.2f} bps\n")
        f.write(f"  Mean PIT: {analysis['overall']['mean_pit']:.3f}\n")
        f.write(f"  90% Coverage: {analysis['overall']['coverage_90']:.1%}\n")
        f.write(f"  95% Coverage: {analysis['overall']['coverage_95']:.1%}\n")
        
        f.write("\nBy Horizon:\n")
        for horizon, metrics in analysis['by_horizon'].items():
            years = horizon / 252
            f.write(f"  {horizon} days ({years:.1f}Y):\n")
            f.write(f"    CRPS: {metrics['mean_crps']:.2f} bps\n")
            f.write(f"    Coverage: {metrics['coverage_90']:.1%}\n")
    
    # Save scenarios (backward compatible format)
    scenarios_path = Path(config['output']['scenarios_path'])
    scenarios_path.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(scenarios_path / 'latest_scenarios.csv', index=False)
    
    # Save spread forecast table if available
    if spread_engine is not None and spread_engine.is_fitted:
        # Reconstruct scenarios dict for spread table
        # (simplified - using the last horizon from scenarios DataFrame)
        print(f"Spread forecasts included in scenario output")
    
    print(f"\nResults saved to {output_path}")


def main():
    """Main execution pipeline with Phase 2 spread integration"""
    
    print("=" * 60)
    print("PHASE 2: Treasury + Spread Scenario Engine")
    print("=" * 60)
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Run directory: {run_dir}")
    
    # Progress bar for overall pipeline
    with tqdm(total=10, desc="Overall Progress", position=0, leave=True) as pbar:
        
        # 1. Load configuration
        pbar.set_description("[1/10] Loading configuration")
        config = load_config("config.yaml")
        
        # Override output paths to use run directory
        config['output']['figures_path'] = str(run_dir / "figures")
        config['output']['results_path'] = str(run_dir / "results")
        config['output']['scenarios_path'] = str(run_dir / "scenarios")
        
        # Create subdirectories
        Path(config['output']['figures_path']).mkdir(parents=True, exist_ok=True)
        Path(config['output']['results_path']).mkdir(parents=True, exist_ok=True)
        Path(config['output']['scenarios_path']).mkdir(parents=True, exist_ok=True)
        
        pbar.update(1)
        
        # 2. Prepare data (now includes spread computation)
        pbar.set_description("[2/10] Preparing data + computing spreads")
        df = prepare_data(config)
        pbar.update(1)
        
        # 3. Fit AFNS model
        pbar.set_description("[3/10] Fitting AFNS model")
        df_factors, afns_model, cov_matrix = fit_afns_model(df, config)
        pbar.update(1)
        
        # 4. Fit regime model
        pbar.set_description("[4/10] Fitting Sticky HMM")
        hmm_model, regime_labels, regime_stats, regime_covs = fit_regime_model(
            df, df_factors, afns_model, config
        )

        # Print regime persistence check
        print(f"\nRegime Persistence Check:")
        for k, stats in regime_stats.items():
            duration = stats['expected_duration_days']
            if duration < 30:
                print(f"  ⚠ Regime {k}: {duration:.1f} days (too short for multi-year forecasting)")
            else:
                print(f"  ✓ Regime {k}: {duration:.1f} days ({duration/252:.1f} years)")
        pbar.update(1)
        
        # 5. Fit SpreadEngine (Phase 2)
        pbar.set_description("[5/10] Fitting SpreadEngine")
        
        spread_engine = None
        if config.get('spreads', {}).get('enabled', True):
            try:
                sectors = config['spreads'].get('sectors', ['Muni', 'Agency', 'Corp'])
                benchmark_maturity = config['spreads'].get('benchmark_maturity', 5)
                
                print(f"\nFitting SpreadEngine for {sectors} at {benchmark_maturity}Y...")
                spread_df = get_spread_dataframe(df, sectors, benchmark_maturity)
                spread_engine = SpreadEngine(sectors=sectors, benchmark_maturity=benchmark_maturity)
                spread_engine.fit(spread_df, df_factors, regime_labels)
                
                # Validate economic intuition
                spread_engine.validate_economic_intuition()
                
                print(f"✓ SpreadEngine fitted successfully")
            except Exception as e:
                print(f"⚠ SpreadEngine fitting failed: {e}")
                print("Continuing with Treasury-only forecasts...")
                spread_engine = None
        else:
            print("Spreads disabled in config - running Treasury-only")
        
        pbar.update(1)
        
        # 6. Multi-horizon scenario generation with spreads
        pbar.set_description("[6/10] Generating multi-year scenarios")

        # Get all horizons from config
        horizons = config['scenarios']['horizons']

        # Initialize scenario generator with spread engine
        generator = ScenarioGenerator(
            afns_model=afns_model,
            hmm_model=hmm_model,
            spread_engine=spread_engine,  # Phase 2 integration
            horizons=horizons,
            n_paths=config['scenarios']['n_paths'],
            n_jobs=config['scenarios']['n_jobs'],
            config=config
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
                'regime_probs': hmm_model.regime_probs.iloc[-1].values,
                'move': df.iloc[-1]['MOVE Index'] if 'MOVE Index' in df.columns else 80.0
            }
            
            # Add spread decomposition if available
            if spread_engine is not None and spread_engine.is_fitted:
                current_state['spreads'] = spread_engine.get_current_spreads(
                    spread_df, df_factors, df.index[-1]
                )
            
            # Generate for all horizons
            horizon_scenarios = generator.generate_all_horizons(current_state, shock)
            all_scenarios[shock] = horizon_scenarios

        # Backward-compat dataframes for saving
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
        
        # 7. Run validation
        pbar.set_description("[7/10] Running multi-horizon validation")
        validation_results = run_validation(
            df, df_factors, afns_model, hmm_model, 
            regime_labels, regime_covs, config
        )
        pbar.update(1)

        # 8. Print multi-year forecast example
        pbar.set_description("[8/10] Generating forecast summary")
        print("\n=== Multi-Year Forecast Example (10yr Treasury) ===")
        print("Current: {:.2f}%".format(current_state['yields'][4]))
        print("\nNo Fed Action Scenario:")
        for horizon_name, data in all_scenarios[0].items():
            percs = data['percentiles']['UST_10']
            years = data['horizon_days'] / 252
            print(f"  {horizon_name:3} ({years:3.1f}Y): {percs['p50']:.2f}% [{percs['p5']:.2f}% - {percs['p95']:.2f}%]")
        
        # Print spread forecasts if available
        if spread_engine is not None and spread_engine.is_fitted:
            print(f"\n=== Spread Product Forecasts ({max_h_name}) ===")
            max_h_percs = all_scenarios[0][max_h_name]['percentiles']
            
            # Treasury
            ust_key = f'UST_{spread_engine.benchmark_maturity}'
            if ust_key in max_h_percs:
                ust = max_h_percs[ust_key]
                print(f"  {spread_engine.benchmark_maturity}Y Treasury: {ust['p50']:.2f}% [{ust['p5']:.2f}% - {ust['p95']:.2f}%]")
            
            # Spreads
            for sector in spread_engine.sectors:
                yield_key = f'{sector}_Yield'
                spread_key = f'{sector}_Spread'
                if yield_key in max_h_percs and spread_key in max_h_percs:
                    yld = max_h_percs[yield_key]
                    spr = max_h_percs[spread_key]
                    print(f"  {spread_engine.benchmark_maturity}Y {sector:7}: {yld['p50']:.2f}% "
                          f"[{yld['p5']:.2f}% - {yld['p95']:.2f}%] "
                          f"(spread: {spr['p50']:+.2f}bps)")
        
        pbar.update(1)
        
        # 9. Generate outputs
        pbar.set_description("[9/10] Generating visualizations")
        
        # Create output directories
        figures_path = Path(config['output']['figures_path'])
        figures_path.mkdir(parents=True, exist_ok=True)
        
        # Plot Treasury scenario fan
        plot_scenario_fan(scenarios, current_state, config, 
                         save_path=figures_path / 'scenario_fan.png')
        
        # Plot spread scenarios (Phase 2)
        if spread_engine is not None and spread_engine.is_fitted:
            plot_spread_scenarios(scenarios, current_state, spread_engine, config,
                                save_path=figures_path / 'spread_scenarios.png')
            
            # Create spread forecast table
            create_spread_forecast_table(scenarios, spread_engine, config,
                                        save_path=Path(config['output']['results_path']) / 'spread_forecasts.csv')
        
        # Plot regime analysis
        plot_regime_analysis(df, regime_labels, config,
                            save_path=figures_path / 'regime_analysis.png')
        
        pbar.update(1)
        
        # 10. Save results
        pbar.set_description("[10/10] Saving results")
        save_results(validation_results, df_scenarios, config, spread_engine)
        pbar.update(1)
    
    print("\n" + "=" * 60)
    print("Phase 2 Complete!")
    print("=" * 60)
    print(f"\n📁 Run directory: {run_dir}")
    print(f"\nOutputs saved to:")
    print(f"  - figures/  (scenario fan, spread scenarios, regime analysis)")
    print(f"  - results/  (validation results, spread forecasts)")
    print(f"  - scenarios/  (latest scenarios CSV)")
    
    if spread_engine is not None and spread_engine.is_fitted:
        print(f"\n✓ Spread products included: {spread_engine.sectors}")


if __name__ == "__main__":
    main()