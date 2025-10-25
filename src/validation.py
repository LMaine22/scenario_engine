"""
Validation: Multi-horizon backtesting with proper VAR dynamics and regime-scaled covariances
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import solve_discrete_lyapunov
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MultiHorizonValidator:
    """
    Sophisticated multi-horizon validation with proper state-space dynamics
    """
    
    def __init__(self, afns_model, hmm_model, config):
        """
        Initialize validator with models and config
        
        Args:
            afns_model: Fitted AFNS model with VAR parameters
            hmm_model: Fitted Sticky HMM model
            config: Configuration dict
        """
        self.afns = afns_model
        self.hmm = hmm_model
        self.config = config
        
        # Extract key parameters
        self.horizons = config['validation']['test_horizons']
        self.confidence_levels = config['scenarios']['confidence_levels']
        self.n_paths = 1000  # Fewer paths for validation speed
        
        # Regime-dependent volatility scaling
        self.use_scaled_Q = config['regime']['volatility_scaling']['use_scaled_Q']
        self.scale_method = config['regime']['volatility_scaling']['scale_method']
        
        logger.info(f"Initialized MultiHorizonValidator with horizons: {self.horizons}")
    
    def compute_regime_scaling_factors(self) -> Dict[int, float]:
        """
        Compute volatility scaling factor for each regime
        
        Returns:
            Dict mapping regime -> scaling factor
        """
        scaling_factors = {}
        
        # Get regime covariances (these are covariances of CHANGES)
        regime_covs = self.hmm.regime_covs
        
        if self.scale_method == 'trace_ratio':
            # Use trace (total variance) as scaling metric
            base_trace = np.trace(regime_covs[0])
            for k in range(self.hmm.n_states):
                scaling_factors[k] = np.trace(regime_covs[k]) / base_trace
                
        elif self.scale_method == 'determinant_ratio':
            # Use determinant (total volume) as scaling metric
            base_det = np.linalg.det(regime_covs[0]) ** (1/3)
            for k in range(self.hmm.n_states):
                regime_det = np.linalg.det(regime_covs[k]) ** (1/3)
                scaling_factors[k] = regime_det / base_det if base_det > 0 else 1.0
                
        else:
            # No scaling
            scaling_factors = {k: 1.0 for k in range(self.hmm.n_states)}
        
        logger.info(f"Regime scaling factors: {scaling_factors}")
        return scaling_factors
    
    def simulate_paths(self, initial_state: Dict, horizon: int, 
                      n_paths: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths using proper VAR dynamics with regime transitions
        
        Args:
            initial_state: Dict with 'factors', 'regime_probs', 'date'
            horizon: Forecast horizon in days
            n_paths: Number of paths (default from config)
            
        Returns:
            factor_paths: (n_paths, horizon+1, 3) array of factor paths
            regime_paths: (n_paths, horizon+1) array of regime paths
        """
        if n_paths is None:
            n_paths = self.n_paths
        
        # Extract initial conditions
        f_0 = initial_state['factors']  # (3,) array
        gamma_0 = initial_state['regime_probs']  # (K,) array
        
        # Get steady-state uncertainty from Lyapunov equation
        try:
            P_ss = solve_discrete_lyapunov(self.afns.A, self.afns.Q)
        except:
            logger.warning("Lyapunov solution failed, using Q as approximation")
            P_ss = self.afns.Q * 10
        
        # Get regime scaling factors
        scaling_factors = self.compute_regime_scaling_factors()
        
        # Storage
        factor_paths = np.zeros((n_paths, horizon + 1, 3))
        regime_paths = np.zeros((n_paths, horizon + 1), dtype=int)
        
        # Simulate each path
        for p in range(n_paths):
            # Draw initial regime from current distribution
            s_t = np.random.choice(self.hmm.n_states, p=gamma_0)
            
            # Draw initial factors from steady-state distribution
            # This captures parameter uncertainty
            f_t = np.random.multivariate_normal(f_0, P_ss * 0.1)  # Scale down P_ss
            
            # Store initial state
            factor_paths[p, 0, :] = f_t
            regime_paths[p, 0] = s_t
            
            # Evolve forward
            for t in range(1, horizon + 1):
                # Regime transition
                transition_probs = self.hmm.transition_matrix[s_t, :]
                s_t = np.random.choice(self.hmm.n_states, p=transition_probs)
                
                # Scale Q by regime
                if self.use_scaled_Q:
                    Q_scaled = self.afns.Q * scaling_factors[s_t]
                else:
                    Q_scaled = self.afns.Q
                
                # VAR dynamics: f_{t+1} = A f_t + b + eta
                eta = np.random.multivariate_normal(np.zeros(3), Q_scaled)
                f_t = self.afns.A @ f_t + self.afns.b + eta
                
                # Store
                factor_paths[p, t, :] = f_t
                regime_paths[p, t] = s_t
        
        return factor_paths, regime_paths
    
    def factors_to_yields(self, factor_paths: np.ndarray) -> np.ndarray:
        """
        Convert factor paths to yield paths using AFNS loadings
        
        Args:
            factor_paths: (n_paths, n_times, 3) array
            
        Returns:
            yield_paths: (n_paths, n_times, n_maturities) array
        """
        n_paths, n_times, _ = factor_paths.shape
        n_mats = len(self.afns.maturities)
        
        yield_paths = np.zeros((n_paths, n_times, n_mats))
        
        for m, mat in enumerate(self.afns.maturities):
            # Compute loadings for this maturity
            tau = mat
            lam = self.afns.lambda_param
            h_0 = 1.0
            h_1 = (1 - np.exp(-lam * tau)) / (lam * tau)
            h_2 = h_1 - np.exp(-lam * tau)
            loadings = np.array([h_0, h_1, h_2])
            
            # Apply loadings: y = h'f
            yield_paths[:, :, m] = factor_paths @ loadings
        
        return yield_paths
    
    def compute_metrics(self, forecasts: np.ndarray, realized: np.ndarray) -> Dict:
        """
        Compute comprehensive validation metrics
        
        Args:
            forecasts: (n_paths, n_maturities) array of forecasts
            realized: (n_maturities,) array of realized values
            
        Returns:
            Dict with metrics including CRPS, PIT, coverage
        """
        n_paths, n_mats = forecasts.shape
        metrics = {}
        
        for m in range(n_mats):
            forecast_m = forecasts[:, m]
            realized_m = realized[m]
            
            # CRPS (Continuous Ranked Probability Score)
            crps = self._compute_crps(forecast_m, realized_m)
            
            # PIT (Probability Integral Transform)
            pit = self._compute_pit(forecast_m, realized_m)
            
            # Coverage at different levels
            coverage_90 = self._compute_coverage(forecast_m, realized_m, 5, 95)
            coverage_95 = self._compute_coverage(forecast_m, realized_m, 2.5, 97.5)
            coverage_99 = self._compute_coverage(forecast_m, realized_m, 0.5, 99.5)
            
            # Bias (mean forecast error)
            bias = np.mean(forecast_m) - realized_m
            
            # RMSE
            rmse = np.sqrt(np.mean((forecast_m - realized_m) ** 2))
            
            # MAE
            mae = np.mean(np.abs(forecast_m - realized_m))
            
            metrics[f'mat_{m}'] = {
                'crps': crps,
                'pit': pit,
                'coverage_90': coverage_90,
                'coverage_95': coverage_95,
                'coverage_99': coverage_99,
                'bias': bias,
                'rmse': rmse,
                'mae': mae
            }
        
        return metrics
    
    def _compute_crps(self, forecast_samples: np.ndarray, realized: float) -> float:
        """Compute CRPS with efficient subsampling"""
        term1 = np.mean(np.abs(forecast_samples - realized))
        
        # Subsample for efficiency
        n = len(forecast_samples)
        n_sub = min(n, 100)
        idx = np.random.choice(n, n_sub, replace=False)
        samples = forecast_samples[idx]
        
        # Pairwise differences
        diffs = []
        for i in range(len(samples)):
            for j in range(i+1, len(samples)):
                diffs.append(np.abs(samples[i] - samples[j]))
        
        term2 = 0.5 * np.mean(diffs) if diffs else 0
        
        return term1 - term2
    
    def _compute_pit(self, forecast_samples: np.ndarray, realized: float) -> float:
        """Compute PIT (should be uniform if calibrated)"""
        return (forecast_samples < realized).mean()
    
    def _compute_coverage(self, forecast_samples: np.ndarray, realized: float,
                         lower_pct: float, upper_pct: float) -> bool:
        """Check if realized value falls in prediction interval"""
        lower = np.percentile(forecast_samples, lower_pct)
        upper = np.percentile(forecast_samples, upper_pct)
        return (lower <= realized <= upper)
    
    def run_backtest(self, df: pd.DataFrame, df_factors: pd.DataFrame,
                    start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Run comprehensive multi-horizon backtest
        
        Args:
            df: DataFrame with yield data
            df_factors: DataFrame with factor data
            start_date: Start of test period
            end_date: End of test period
            
        Returns:
            DataFrame with backtest results
        """
        if start_date is None:
            start_date = self.config['validation']['backtest']['start_date']
        if end_date is None:
            end_date = df.index[-1]
        
        # Filter to test period
        test_mask = (df.index >= pd.Timestamp(start_date)) & \
                   (df.index <= pd.Timestamp(end_date))
        df_test = df[test_mask]
        df_factors_test = df_factors[test_mask]
        
        results = []
        tenors = self.config['data']['tenors']
        
        # For each origin date
        for origin_idx in tqdm(range(len(df_test)), desc="Backtesting", leave=False):
            origin_date = df_test.index[origin_idx]
            
            # Skip if not enough future data for longest horizon
            max_horizon = max(self.horizons)
            if origin_idx + max_horizon >= len(df_test):
                continue
            
            # Get initial state
            factors_0 = df_factors_test.iloc[origin_idx].values
            
            # Get regime probabilities at origin
            try:
                gamma_0 = self.hmm.regime_probs.loc[origin_date].values
            except:
                # If exact date not found, use nearest
                nearest_idx = self.hmm.regime_probs.index.get_indexer([origin_date], method='nearest')[0]
                gamma_0 = self.hmm.regime_probs.iloc[nearest_idx].values
            
            initial_state = {
                'factors': factors_0,
                'regime_probs': gamma_0,
                'date': origin_date
            }
            
            # For each horizon
            for horizon in self.horizons:
                if origin_idx + horizon >= len(df_test):
                    continue
                
                # Simulate paths
                factor_paths, regime_paths = self.simulate_paths(
                    initial_state, horizon, n_paths=self.n_paths
                )
                
                # Convert to yields
                yield_paths = self.factors_to_yields(factor_paths)
                
                # Extract forecasts at horizon
                forecasts = yield_paths[:, -1, :]  # (n_paths, n_maturities)
                
                # Get realized yields
                realized_date = df_test.index[origin_idx + horizon]
                realized = df_test.loc[realized_date, tenors].values
                
                # Compute metrics
                metrics = self.compute_metrics(forecasts, realized)
                
                # Store results
                for m, tenor in enumerate(tenors):
                    mat_metrics = metrics[f'mat_{m}']
                    results.append({
                        'origin_date': origin_date,
                        'horizon_days': horizon,
                        'target_date': realized_date,
                        'tenor': tenor,
                        'maturity': self.afns.maturities[m],
                        'realized': realized[m],
                        'forecast_mean': np.mean(forecasts[:, m]),
                        'forecast_p5': np.percentile(forecasts[:, m], 5),
                        'forecast_p50': np.percentile(forecasts[:, m], 50),
                        'forecast_p95': np.percentile(forecasts[:, m], 95),
                        **mat_metrics
                    })
        
        return pd.DataFrame(results)
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze backtest results and compute summary statistics
        
        Args:
            results_df: DataFrame from run_backtest
            
        Returns:
            Dict with analysis results
        """
        analysis = {}
        
        # Overall metrics
        analysis['overall'] = {
            'mean_crps': results_df['crps'].mean(),
            'mean_pit': results_df['pit'].mean(),
            'coverage_90': results_df['coverage_90'].mean(),
            'coverage_95': results_df['coverage_95'].mean(),
            'mean_bias': results_df['bias'].mean(),
            'mean_rmse': results_df['rmse'].mean()
        }
        
        # By horizon
        analysis['by_horizon'] = {}
        for horizon in self.horizons:
            horizon_data = results_df[results_df['horizon_days'] == horizon]
            analysis['by_horizon'][horizon] = {
                'mean_crps': horizon_data['crps'].mean(),
                'mean_pit': horizon_data['pit'].mean(),
                'coverage_90': horizon_data['coverage_90'].mean(),
                'mean_rmse': horizon_data['rmse'].mean()
            }
        
        # By tenor
        analysis['by_tenor'] = {}
        for tenor in self.config['data']['tenors']:
            tenor_data = results_df[results_df['tenor'] == tenor]
            analysis['by_tenor'][tenor] = {
                'mean_crps': tenor_data['crps'].mean(),
                'mean_pit': tenor_data['pit'].mean(),
                'coverage_90': tenor_data['coverage_90'].mean()
            }
        
        # Horizon consistency check
        # Short horizons should have lower RMSE than long horizons
        horizons_sorted = sorted(self.horizons)
        rmse_by_horizon = [
            analysis['by_horizon'][h]['mean_rmse'] 
            for h in horizons_sorted
        ]
        analysis['horizon_consistency'] = all(
            rmse_by_horizon[i] <= rmse_by_horizon[i+1] 
            for i in range(len(rmse_by_horizon)-1)
        )
        
        return analysis


def run_validation(df, df_factors, afns_model, hmm_model, 
                   regime_labels, regime_covs, config):
    """
    Main validation function with sophisticated multi-horizon testing
    
    Returns:
        Dict with comprehensive validation results
    """
    print("\n=== Running Multi-Horizon Validation ===")
    
    # Initialize validator
    validator = MultiHorizonValidator(afns_model, hmm_model, config)
    
    # Run backtest
    print("Running historical backtest at multiple horizons...")
    print(f"Horizons: {validator.horizons} days")
    
    results_df = validator.run_backtest(df, df_factors)
    
    # Analyze results
    print("Analyzing results...")
    analysis = validator.analyze_results(results_df)
    
    # Print summary
    print("\n=== Validation Results ===")
    print(f"Overall Metrics:")
    print(f"  Mean CRPS: {analysis['overall']['mean_crps']:.2f} bps")
    print(f"  Mean PIT: {analysis['overall']['mean_pit']:.3f} (target: 0.500)")
    print(f"  90% Coverage: {analysis['overall']['coverage_90']:.1%} (target: 90%)")
    print(f"  95% Coverage: {analysis['overall']['coverage_95']:.1%} (target: 95%)")
    print(f"  Mean Bias: {analysis['overall']['mean_bias']:.2f} bps")
    
    print(f"\nBy Horizon:")
    for horizon, metrics in analysis['by_horizon'].items():
        years = horizon / 252
        print(f"  {horizon} days ({years:.1f}Y):")
        print(f"    CRPS: {metrics['mean_crps']:.2f} bps")
        print(f"    Coverage: {metrics['coverage_90']:.1%}")
        print(f"    RMSE: {metrics['mean_rmse']:.2f} bps")
    
    # Check calibration
    pit_mean = analysis['overall']['mean_pit']
    if 0.45 <= pit_mean <= 0.55:
        print("\n✓ PIT well-calibrated (unbiased forecasts)")
    else:
        print(f"\n⚠ PIT = {pit_mean:.3f} indicates bias")
    
    coverage_90 = analysis['overall']['coverage_90']
    if 0.85 <= coverage_90 <= 0.95:
        print("✓ Coverage well-calibrated")
    else:
        print(f"⚠ Coverage = {coverage_90:.1%} (miscalibrated)")
    
    if analysis['horizon_consistency']:
        print("✓ Horizon consistency maintained (uncertainty grows with time)")
    else:
        print("⚠ Horizon inconsistency detected")
    
    return {
        'results_df': results_df,
        'analysis': analysis,
        'validator': validator
    }
