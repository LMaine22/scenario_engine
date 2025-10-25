# Validation Framework Structure
# Step 1: Load and explore the regime transitions in your data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats

# Load your data
df = pd.read_parquet( 'data/raw/RatesRegimes.parquet')

# 1. REGIME IDENTIFICATION MODULE
class RegimeDetector:
    """Identify market regimes from multiple indicators"""
    
    def __init__(self, df):
        self.df = df
        self.regimes = pd.DataFrame(index=df.index)
        
    def identify_volatility_regime(self):
        """Classify based on MOVE and VIX levels"""
        # MOVE-based regime
        self.regimes['vol_regime'] = pd.cut(
            self.df['MOVE Index'],
            bins=[0, 80, 120, 500],
            labels=['calm', 'elevated', 'stress']
        )
        
    def identify_policy_regime(self):
        """Classify based on Fed Funds rate and trajectory"""
        self.regimes['policy_regime'] = pd.cut(
            self.df['FEDL01 Index'],
            bins=[-1, 0.5, 2.5, 4.0, 10],
            labels=['ZIRP', 'normalizing', 'neutral', 'restrictive']
        )
        
        # Add rate of change
        self.regimes['fed_momentum'] = self.df['FEDL01 Index'].diff(252).fillna(0)
        self.regimes['hiking_cycle'] = (self.regimes['fed_momentum'] > 0.5).astype(int)
        
    def identify_curve_regime(self):
        """Classify based on curve shape"""
        self.regimes['curve_2_10'] = self.df['USGG10YR Index'] - self.df['USGG2YR Index']
        self.regimes['curve_regime'] = pd.cut(
            self.regimes['curve_2_10'],
            bins=[-500, -0.5, 0.5, 2.0, 500],
            labels=['inverted', 'flat', 'normal', 'steep']
        )
        
    def create_composite_regime(self):
        """Combine indicators into master regime"""
        # Simple scoring system - can be replaced with your Markov switching
        regime_score = (
            (self.df['MOVE Index'] > 100).astype(int) * 2 +
            (self.df['FEDL01 Index'] > 3).astype(int) * 1 +
            (self.regimes['curve_2_10'] < 0).astype(int) * 2 +
            (self.df['VIX Index'] > 25).astype(int) * 1
        )
        
        self.regimes['composite_regime'] = pd.cut(
            regime_score,
            bins=[-1, 1, 3, 10],
            labels=['stable', 'transition', 'stress']
        )
        
        return self.regimes

# 2. PT ASSUMPTIONS MODULE
class PTAssumptions:
    """Encode PT's fixed assumptions for comparison"""
    
    def __init__(self, start_date='2021-12-01'):
        self.start_date = pd.to_datetime(start_date)
        self.assumptions = {}
        
    def set_deposit_assumptions(self):
        """PT's fixed deposit parameters"""
        self.assumptions['nmd_decay'] = 0.03  # 3% annual
        self.assumptions['surge_pct'] = 0.15  # 15% uniform
        self.assumptions['deposit_lag'] = 90  # 3 month lag
        self.assumptions['deposit_beta'] = {
            'savings': 0.20,
            'checking': 0.15,
            'mmda': 0.35
        }
        
    def set_reinvestment_assumptions(self):
        """PT's reinvestment formulas"""
        self.assumptions['reinvestment'] = {
            'consumer_fixed': 'PRIME + 4.78',
            'first_mortgage': '10CMT + 1.89',
            'commercial': 'PRIME - 0.92',
            'deposits': 'FFT - 1.00, floor 3.50'
        }
        
    def project_forward(self, market_data, horizon=24):
        """Generate PT's projections using their fixed rules"""
        projections = pd.DataFrame()
        
        # Starting from self.start_date
        proj_dates = pd.date_range(self.start_date, periods=horizon, freq='M')
        
        for date in proj_dates:
            # PT assumes everything stays constant
            fed_funds = market_data.loc[self.start_date, 'FEDL01 Index']
            
            # Apply PT's simplistic formulas
            proj_dict = {
                'date': date,
                'pt_deposit_cost': max(fed_funds - 1.0, 3.50),  # Their floor
                'pt_deposit_beta': 0.22,  # Their "net" beta
                'pt_decay_rate': 0.03,
                'pt_surge': 0.15
            }
            projections = pd.concat([projections, pd.DataFrame([proj_dict])], ignore_index=True)
        
        return projections

# 3. REALITY TRACKER MODULE  
class RealityTracker:
    """Track what actually happened vs PT projections"""
    
    def __init__(self, df, start_date='2021-12-01'):
        self.df = df[df.index >= start_date]
        self.actual_metrics = pd.DataFrame(index=self.df.index)
        
    def calculate_actual_funding_cost(self):
        """Approximate actual funding costs from market data"""
        # Using Fed Funds as proxy - in reality would use Call Report data
        self.actual_metrics['actual_funding_cost'] = (
            self.df['FEDL01 Index'] * 0.75  # Rough beta approximation
        )
        
    def calculate_actual_spreads(self):
        """Track actual spread evolution"""
        self.actual_metrics['actual_mbs_oas'] = self.df['LUMSOAS Index']
        self.actual_metrics['actual_corp_oas'] = self.df['LUACOAS Index']
        
    def calculate_stress_indicators(self):
        """Identify stress periods PT would miss"""
        self.actual_metrics['march_23_shock'] = (
            (self.df.index >= '2023-03-01') & 
            (self.df.index <= '2023-04-01')
        ).astype(int) * 100  # Scale for visibility
        
        return self.actual_metrics

# 4. COMPARISON ENGINE
class ValidationEngine:
    """Compare PT projections to reality"""
    
    def __init__(self, pt_proj, actual, regimes):
        self.pt = pt_proj
        self.actual = actual
        self.regimes = regimes
        
    def calculate_errors(self):
        """Quantify PT's prediction errors"""
        errors = pd.DataFrame()
        
        # Align dates
        common_dates = self.actual.index.intersection(
            pd.to_datetime(self.pt['date'])
        )
        
        for date in common_dates:
            pt_value = self.pt[self.pt['date'] == date]['pt_deposit_cost'].values[0]
            actual_value = self.actual.loc[date, 'actual_funding_cost']
            
            error_dict = {
                'date': date,
                'pt_projection': pt_value,
                'actual': actual_value,
                'error_bps': (actual_value - pt_value) * 100,
                'regime': self.regimes.loc[date, 'composite_regime']
            }
            errors = pd.concat([errors, pd.DataFrame([error_dict])], ignore_index=True)
        
        return errors

# 5. VISUALIZATION MODULE
def create_destruction_chart(errors_df, regimes_df):
    """The chart that destroys PT"""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Top: Regime indicator
    regime_colors = {'stable': 'green', 'transition': 'yellow', 'stress': 'red'}
    for regime, color in regime_colors.items():
        mask = regimes_df['composite_regime'] == regime
        axes[0].fill_between(
            regimes_df.index[mask], 0, 1, 
            color=color, alpha=0.3, label=regime
        )
    axes[0].set_ylabel('Regime')
    axes[0].legend()
    axes[0].set_title('Market Regime vs PT Prediction Errors')
    
    # Middle: PT vs Reality
    axes[1].plot(errors_df['date'], errors_df['pt_projection'], 
                label='PT Projection', linestyle='--', color='blue')
    axes[1].plot(errors_df['date'], errors_df['actual'], 
                label='Actual', color='red', linewidth=2)
    axes[1].set_ylabel('Funding Cost (%)')
    axes[1].legend()
    axes[1].fill_between(errors_df['date'], 
                        errors_df['pt_projection'], 
                        errors_df['actual'],
                        color='red', alpha=0.2)
    
    # Bottom: Prediction Error
    axes[2].bar(errors_df['date'], errors_df['error_bps'], 
               color='darkred', alpha=0.7)
    axes[2].set_ylabel('PT Error (bps)')
    axes[2].set_xlabel('Date')
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Highlight March 2023
    for ax in axes:
        ax.axvspan(pd.Timestamp('2023-03-01'), pd.Timestamp('2023-04-01'), 
                  color='gray', alpha=0.3)
    
    plt.tight_layout()
    return fig

# 6. EXECUTION SCRIPT
def run_validation():
    """Execute the complete validation"""
    
    print("Loading data...")
    df = pd.read_parquet('data/raw/RatesRegimes.parquet')
    df.set_index('Date', inplace=True)
    
    print("Detecting regimes...")
    regime_detector = RegimeDetector(df)
    regime_detector.identify_volatility_regime()
    regime_detector.identify_policy_regime()
    regime_detector.identify_curve_regime()
    regimes = regime_detector.create_composite_regime()
    
    print("Setting PT assumptions...")
    pt_model = PTAssumptions(start_date='2021-12-01')
    pt_model.set_deposit_assumptions()
    pt_model.set_reinvestment_assumptions()
    pt_projections = pt_model.project_forward(df, horizon=36)
    
    print("Tracking reality...")
    reality = RealityTracker(df, start_date='2021-12-01')
    reality.calculate_actual_funding_cost()
    reality.calculate_actual_spreads()
    actual_metrics = reality.calculate_stress_indicators()
    
    print("Calculating errors...")
    validator = ValidationEngine(pt_projections, actual_metrics, regimes)
    errors = validator.calculate_errors()
    
    print("Creating visualizations...")
    fig = create_destruction_chart(errors, regimes)
    
    # Summary statistics
    print("\n=== PT FAILURE METRICS ===")
    print(f"Average Prediction Error: {errors['error_bps'].mean():.0f} bps")
    print(f"Maximum Prediction Error: {errors['error_bps'].max():.0f} bps")
    print(f"Error During Stress Regime: {errors[errors['regime']=='stress']['error_bps'].mean():.0f} bps")
    
    return errors, regimes, fig

# Run it
if __name__ == "__main__":
    errors, regimes, fig = run_validation()
    plt.show()