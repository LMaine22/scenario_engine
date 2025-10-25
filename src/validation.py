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
from multiprocessing import Pool, cpu_count
from functools import partial

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
        
        # Parallel processing
        self.n_jobs = config.get('validation', {}).get('n_jobs', -1)
        if self.n_jobs == -1:
            self.n_jobs = max(1, cpu_count() - 1)
        
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
        
        # Vectorized initial conditions
        # Draw initial regimes for all paths at once
        initial_regimes = np.random.choice(self.hmm.n_states, size=n_paths, p=gamma_0)
        regime_paths[:, 0] = initial_regimes
        
        # Draw initial factors for all paths (vectorized)
        initial_factors = np.random.multivariate_normal(f_0, P_ss * 0.1, size=n_paths)
        factor_paths[:, 0, :] = initial_factors
        
        # For each time step (can't fully vectorize due to regime dependencies)
        for t in range(1, horizon + 1):
            # Vectorized regime transitions
            for p in range(n_paths):
                current_regime = regime_paths[p, t-1]
                transition_probs = self.hmm.transition_matrix[current_regime, :]
                regime_paths[p, t] = np.random.choice(self.hmm.n_states, p=transition_probs)
            
            # Vectorized factor evolution for all paths
            prev_factors = factor_paths[:, t-1, :]  # (n_paths, 3)
            
            # Group paths by regime for efficient noise generation
            for regime in range(self.hmm.n_states):
                mask = regime_paths[:, t] == regime
                n_regime = np.sum(mask)
                
                if n_regime > 0:
                    # Scale Q by regime
                    if self.use_scaled_Q:
                        Q_scaled = self.afns.Q * scaling_factors[regime]
                    else:
                        Q_scaled = self.afns.Q
                    
                    # Generate noise for all paths in this regime at once
                    eta = np.random.multivariate_normal(np.zeros(3), Q_scaled, size=n_regime)
                    
                    # VAR dynamics: f_{t+1} = A f_t + b + eta (vectorized)
                    factor_paths[mask, t, :] = (prev_factors[mask] @ self.afns.A.T + 
                                                self.afns.b + eta)
        
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
    
    def _process_single_origin(self, args):
        """Helper function to process a single origin date (for parallel processing)"""
        origin_idx, origin_date, df_test, df_factors_test, tenors = args
        results = []
        
        try:
            # Current observations  
            current_yields = df_test.loc[origin_date, tenors].values
            current_factors = df_factors_test.loc[origin_date].values
            
            # Get regime probabilities at origin
            try:
                current_gamma = self.hmm.regime_probs.loc[origin_date].values
            except:
                # If exact date not found, use nearest
                nearest_idx = self.hmm.regime_probs.index.get_indexer([origin_date], method='nearest')[0]
                current_gamma = self.hmm.regime_probs.iloc[nearest_idx].values
            
            initial_state = {
                'factors': current_factors,
                'regime_probs': current_gamma,
                'yields': current_yields
            }
            
            # For each horizon
            for horizon in self.horizons:
                # Check if we have data horizon days in the future
                target_date = origin_date + pd.Timedelta(days=horizon)
                if target_date > df_test.index[-1]:
                    continue
                
                # Simulate paths
                factor_paths, regime_paths = self.simulate_paths(
                    initial_state, horizon, n_paths=self.n_paths
                )
                
                # Convert to yields
                yield_paths = self.factors_to_yields(factor_paths)
                forecasts = yield_paths[:, -1, :]  # Terminal values
                
                # Get realized yields - find the closest date to target
                realized_date = df_test.index[df_test.index.get_indexer([target_date], method='nearest')[0]]
                realized = df_test.loc[realized_date, tenors].values
                
                # CRISIS ADAPTATION: Scale intervals based on MOVE
                if 'MOVE Index' in df_test.columns:
                    current_move = df_test.loc[origin_date, 'MOVE Index']
                    move_baseline = 80.0  # Normal MOVE level
                    
                    # Scale forecast dispersion when volatility is high
                    if current_move > move_baseline * 1.5:  # > 120
                        vol_multiplier = current_move / move_baseline
                        # Widen the forecast distribution
                        forecast_mean = np.mean(forecasts, axis=0)
                        forecasts = forecast_mean + (forecasts - forecast_mean) * vol_multiplier
                
                # Compute metrics
                metrics = self.compute_metrics(forecasts, realized)
                
                # Store results
                for m, tenor in enumerate(tenors):
                    mat_metrics = metrics[f'mat_{m}']
                    
                    # Build result dict with all fields explicitly
                    result = {
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
                    }
                    
                    # Add metrics explicitly, with defaults if missing
                    result['crps'] = mat_metrics.get('crps', 0.0)
                    result['pit'] = mat_metrics.get('pit', 0.5)
                    result['coverage_90'] = mat_metrics.get('coverage_90', False)
                    result['coverage_95'] = mat_metrics.get('coverage_95', False)
                    result['coverage_99'] = mat_metrics.get('coverage_99', False)
                    result['bias'] = mat_metrics.get('bias', 0.0)
                    result['rmse'] = mat_metrics.get('rmse', 0.0)
                    result['mae'] = mat_metrics.get('mae', 0.0)
                    
                    results.append(result)
                    
        except Exception as e:
            logger.warning(f"Error processing origin date {origin_date}: {str(e)}")
            
        return results
    
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
        # Test-mode overrides (optional, from config)
        max_test_points = self.config['validation']['backtest'].get('max_test_points', None)
        test_frequency = self.config['validation']['backtest'].get('test_frequency', 'daily')
        n_paths_override = self.config['validation'].get('n_paths_validation', self.n_paths)
        self.n_paths = n_paths_override

        if start_date is None:
            start_date = self.config['validation']['backtest']['start_date']
        if end_date is None:
            end_date = df.index[-1]
        
        # Filter to test period
        test_mask = (df.index >= pd.Timestamp(start_date)) & \
                   (df.index <= pd.Timestamp(end_date))
        df_test = df[test_mask]
        df_factors_test = df_factors[test_mask]

        # Downsample frequency for testing
        if test_frequency == 'monthly' and len(df_test) > 0:
            # first day of month in index
            monthly_mask = df_test.index.to_series().dt.day == 1
            if monthly_mask.any():
                df_test = df_test[monthly_mask]
                df_factors_test = df_factors_test[monthly_mask]
        
        # Limit number of test origins
        if max_test_points and len(df_test) > max_test_points:
            step = max(1, len(df_test) // max_test_points)
            df_test = df_test.iloc[::step][:max_test_points]
            df_factors_test = df_factors_test.iloc[::step][:max_test_points]
        
        print(f"Testing on {len(df_test)} points (reduced for speed)")
        
        results = []
        tenors = self.config['data']['tenors']
        
        # Check if we should use parallel processing
        use_parallel = self.n_jobs > 1 and len(df_test) > 10
        
        if use_parallel:
            # Prepare arguments for parallel processing
            parallel_args = []
            for origin_idx in range(len(df_test)):
                origin_date = df_test.index[origin_idx]
                
                # Skip if not enough future data for longest horizon
                max_horizon = max(self.horizons)
                max_target_date = origin_date + pd.Timedelta(days=max_horizon)
                if max_target_date > df_test.index[-1]:
                    continue
                    
                parallel_args.append((origin_idx, origin_date, df_test, df_factors_test, tenors))
            
            # Process in parallel with progress bar
            print(f"Processing {len(parallel_args)} origins with {self.n_jobs} workers...")
            with Pool(processes=self.n_jobs) as pool:
                batch_results = list(tqdm(
                    pool.imap(self._process_single_origin, parallel_args),
                    total=len(parallel_args),
                    desc="Backtesting (parallel)",
                    leave=False
                ))
            
            # Flatten results
            for batch in batch_results:
                results.extend(batch)
        else:
            # Original sequential processing
            for origin_idx in tqdm(range(len(df_test)), desc="Backtesting", leave=False):
                origin_date = df_test.index[origin_idx]
                
                # Skip if not enough future data for longest horizon
                max_horizon = max(self.horizons)
                max_target_date = origin_date + pd.Timedelta(days=max_horizon)
                if max_target_date > df_test.index[-1]:
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
                # Check if we have data horizon days in the future
                target_date = origin_date + pd.Timedelta(days=horizon)
                if target_date > df_test.index[-1]:
                    continue
                
                # Simulate paths
                factor_paths, regime_paths = self.simulate_paths(
                    initial_state, horizon, n_paths=self.n_paths
                )
                
                # Convert to yields
                yield_paths = self.factors_to_yields(factor_paths)
                
                # Extract forecasts at horizon
                forecasts = yield_paths[:, -1, :]  # (n_paths, n_maturities)
                
                # Get realized yields - find the closest date to target
                realized_date = df_test.index[df_test.index.get_indexer([target_date], method='nearest')[0]]
                realized = df_test.loc[realized_date, tenors].values
                
                # CRISIS ADAPTATION: Scale intervals based on MOVE
                if 'MOVE Index' in df_test.columns:
                    current_move = df_test.loc[origin_date, 'MOVE Index']
                    move_baseline = 80.0  # Normal MOVE level
                    
                    # Scale forecast dispersion when volatility is high
                    if current_move > move_baseline * 1.5:  # > 120
                        vol_multiplier = current_move / move_baseline
                        # Widen the forecast distribution
                        forecast_mean = np.mean(forecasts, axis=0)
                        forecasts = forecast_mean + (forecasts - forecast_mean) * vol_multiplier
                
                # Compute metrics
                metrics = self.compute_metrics(forecasts, realized)
                
                # Store results
                for m, tenor in enumerate(tenors):
                    mat_metrics = metrics[f'mat_{m}']
                    
                    # Build result dict with all fields explicitly
                    result = {
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
                    }
                    
                    # Add metrics explicitly, with defaults if missing
                    result['crps'] = mat_metrics.get('crps', 0.0)
                    result['pit'] = mat_metrics.get('pit', 0.5)
                    result['coverage_90'] = mat_metrics.get('coverage_90', False)
                    result['coverage_95'] = mat_metrics.get('coverage_95', False)
                    result['coverage_99'] = mat_metrics.get('coverage_99', False)
                    result['bias'] = mat_metrics.get('bias', 0.0)
                    result['rmse'] = mat_metrics.get('rmse', 0.0)
                    result['mae'] = mat_metrics.get('mae', 0.0)
                    
                    results.append(result)
        
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
