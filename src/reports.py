"""
Reporting utilities for multi-year scenarios
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def create_reinvestment_report(scenarios, current_state, config):
    """
    Create detailed reinvestment rate forecasts
    
    Args:
        scenarios: Multi-horizon scenarios from generator
        current_state: Current market state
        config: Configuration
        
    Returns:
        DataFrame with reinvestment projections
    """
    rows = []
    
    for shock, horizon_data in scenarios.items():
        for horizon_name, data in horizon_data.items():
            percentiles = data['percentiles']
            horizon_days = data['horizon_days']
            horizon_years = horizon_days / 252
            
            for mat_idx, maturity in enumerate(config['data']['maturities']):
                ust_key = f'UST_{int(maturity)}'
                if ust_key in percentiles:
                    stats = percentiles[ust_key]
                    
                    rows.append({
                        'Fed_Shock': shock,
                        'Horizon': horizon_name,
                        'Horizon_Years': horizon_years,
                        'Maturity': maturity,
                        'Current_Yield': current_state['yields'][mat_idx],
                        'Forecast_P5': stats['p5'],
                        'Forecast_P50': stats['p50'],
                        'Forecast_P95': stats['p95'],
                        'Forecast_Mean': stats['mean'],
                        'Forecast_Std': stats['std'],
                        'Prob_Below_Current': None  # Calculate from paths if needed
                    })
    
    df = pd.DataFrame(rows)
    
    # Calculate reinvestment risk metrics
    df['Reinvestment_Risk'] = df['Current_Yield'] - df['Forecast_P50']
    df['Worst_Case_Risk'] = df['Current_Yield'] - df['Forecast_P5']
    df['Best_Case_Gain'] = df['Forecast_P95'] - df['Current_Yield']
    
    return df


def plot_multi_year_forecasts(scenarios, current_state, config, save_path=None):
    """
    Create visualization of multi-year rate forecasts
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Focus on no-shock scenario
    base_scenario = scenarios[0]
    
    for mat_idx, (maturity, ax) in enumerate(zip(config['data']['maturities'], axes)):
        if mat_idx >= 6:  # Only 6 maturities
            break
            
        ust_key = f'UST_{int(maturity)}'
        current = current_state['yields'][mat_idx]
        
        # Extract forecasts by horizon
        horizons = []
        p5s = []
        p50s = []
        p95s = []
        
        for horizon_name, data in sorted(base_scenario.items(), 
                                        key=lambda x: x[1]['horizon_days']):
            if ust_key in data['percentiles']:
                years = data['horizon_days'] / 252
                horizons.append(years)
                p5s.append(data['percentiles'][ust_key]['p5'])
                p50s.append(data['percentiles'][ust_key]['p50'])
                p95s.append(data['percentiles'][ust_key]['p95'])
        
        # Add current (t=0)
        horizons = [0] + horizons
        p5s = [current] + p5s
        p50s = [current] + p50s
        p95s = [current] + p95s
        
        # Plot
        ax.plot(horizons, p50s, 'b-', label='Median', linewidth=2)
        ax.fill_between(horizons, p5s, p95s, alpha=0.3, label='90% Band')
        ax.axhline(current, color='black', linestyle='--', label='Current')
        
        # If below 4%, highlight reinvestment risk zone
        if current >= 4.0:
            ax.axhspan(0, 4.0, alpha=0.1, color='red', label='Risk Zone (<4%)')
        
        ax.set_xlabel('Years Forward')
        ax.set_ylabel('Yield (%)')
        ax.set_title(f'{maturity}Y Treasury')
        ax.grid(True, alpha=0.3)
        if mat_idx == 0:
            ax.legend()
        
        ax.set_xlim(0, 5)
    
    plt.suptitle('Multi-Year Reinvestment Rate Forecasts', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Multi-year forecast plot saved to {save_path}")
    
    return fig


