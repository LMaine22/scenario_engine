"""
Scenario Generation: Multi-step path simulation with regime transitions
Updated for Phase 2B: Joint Treasury + Spread forecasting
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import solve_discrete_lyapunov
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


class ScenarioGenerator:
    """Generate multi-step yield curve scenarios with regime transitions and spread products"""
    
    def __init__(self, afns_model, hmm_model, spread_engine=None, 
                 horizons=None, n_paths=10000, n_jobs=-1, 
                 move_state='neutral', q_scaler=1.0, config=None):
        """
        Enhanced for multi-year horizon support with joint Treasury + Spread forecasting
        
        Args:
            afns_model: Fitted AFNS model
            hmm_model: Fitted Sticky HMM
            spread_engine: Optional SpreadEngine for credit spreads (Phase 2)
            horizons: Dict of horizon_name -> days, e.g. {'6m': 126, '3y': 756}
            n_paths: Number of Monte Carlo paths
            n_jobs: Parallel workers (-1 for all cores)
            move_state: Volatility state (not used currently)
            q_scaler: Scaling factor for Q matrix (not used currently)
            config: Configuration dict (for accessing data tenors)
        """
        self.afns = afns_model
        self.hmm = hmm_model
        self.spread_engine = spread_engine
        self.n_paths = n_paths
        self.move_state = move_state
        self.q_scaler = q_scaler
        self.config = config
        
        # Multi-horizon support
        if horizons is None:
            self.horizons = {
                '6m': 126,
                '1y': 252,
                '2y': 504,
                '3y': 756,
                '5y': 1260
            }
        else:
            self.horizons = horizons
        
        # Default single horizon attribute used by generate_scenario
        self.horizon_days = max(self.horizons.values()) if isinstance(self.horizons, dict) else 126
        
        # Set number of jobs
        if n_jobs == -1:
            self.n_jobs = max(1, cpu_count() - 1)
        else:
            self.n_jobs = n_jobs
        
    def _simulate_single_path_wrapper(self, args):
        """Wrapper for multiprocessing.Pool"""
        path_id, sim_params = args
        return self._simulate_single_path(path_id, sim_params)
    
    def _simulate_single_path(self, path_id, sim_params):
        """
        Simulate a single Monte Carlo path with joint Treasury + Spread dynamics
        
        Args:
            path_id: Path identifier
            sim_params: Dict with simulation parameters
            
        Returns:
            path_df: DataFrame with simulated path including spreads
        """
        # Unpack parameters
        f_T = sim_params['f_T']
        P_T = sim_params['P_T']
        gamma_T = sim_params['gamma_T']
        horizon_days = sim_params['horizon_days']
        random_seed = sim_params['random_seed']
        z_initial = sim_params.get('z_initial', None)  # Initial spread residuals
        
        # Set unique seed for this path
        path_seed = random_seed + path_id
        np.random.seed(path_seed)
        
        # 1. Draw initial regime from gamma_T
        n_regimes = len(gamma_T)
        s_t = np.random.choice(n_regimes, p=gamma_T)
        
        # 2. Draw initial factors from N(f_T, P_T)
        f_t = np.random.multivariate_normal(f_T, P_T)
        
        # Storage for this path
        path_data = {
            't': [0],
            'regime': [s_t],
            'L': [f_t[0]],
            'S': [f_t[1]],
            'C': [f_t[2]]
        }
        
        # 3. Simulate forward
        for t in range(1, horizon_days + 1):
            # Transition regime
            s_t = np.random.choice(n_regimes, p=self.hmm.transition_matrix[s_t, :])
            
            # Propagate factors: f_{t+1} = A*f_t + b + η
            eta = np.random.multivariate_normal(np.zeros(3), self.afns.Q * self.q_scaler)
            f_t = self.afns.A @ f_t + self.afns.b + eta
            
            # Store
            path_data['t'].append(t)
            path_data['regime'].append(s_t)
            path_data['L'].append(f_t[0])
            path_data['S'].append(f_t[1])
            path_data['C'].append(f_t[2])
        
        # Convert to arrays for yield computation
        factors_path = np.column_stack([path_data['L'], path_data['S'], path_data['C']])
        regime_path = np.array(path_data['regime'])
        
        # 4. Compute Treasury yields at each time step
        for mat in self.afns.maturities:
            # Compute loadings for this maturity
            tau = mat
            lam = self.afns.lambda_param
            h_0 = 1.0
            h_1 = (1 - np.exp(-lam * tau)) / (lam * tau)
            h_2 = h_1 - np.exp(-lam * tau)
            loadings = np.array([h_0, h_1, h_2])
            
            # Compute yields for all time steps
            yields_path = factors_path @ loadings
            path_data[f'UST_{int(mat)}'] = yields_path
        
        # 5. Simulate spreads if spread_engine exists (Phase 2B Integration)
        if self.spread_engine is not None and self.spread_engine.is_fitted:
            try:
                # Initialize spread residuals if not provided
                if z_initial is None:
                    # Use long-run mean of initial regime
                    z_initial = self.spread_engine.mu[regime_path[0]]
                
                # Simulate spread paths conditional on factor and regime paths
                spread_paths_dict = self.spread_engine.simulate_spreads(
                    factor_paths=factors_path,
                    regime_paths=regime_path,
                    z_initial=z_initial,
                    random_seed=path_seed
                )
                
                # Add spread paths to output
                for sector, spread_path in spread_paths_dict.items():
                    path_data[f'{sector}_Spread'] = spread_path
                    
                    # Also compute all-in yields for each sector
                    # For simplicity, use benchmark maturity (5Y)
                    if f'UST_{self.spread_engine.benchmark_maturity}' in path_data:
                        treasury_yield = path_data[f'UST_{self.spread_engine.benchmark_maturity}']
                        path_data[f'{sector}_Yield'] = treasury_yield + spread_path
                        
            except Exception as e:
                # Gracefully skip spreads if there's an error
                import logging
                logging.warning(f"Spread simulation failed for path {path_id}: {e}")
        
        # Convert to DataFrame
        path_df = pd.DataFrame(path_data)
        path_df['path_id'] = path_id
        path_df['MOVE_state'] = self.move_state
        
        return path_df
    
    def generate_scenario(self, current_state, fed_shock_bp=0, random_seed=42):
        """
        Generate multi-step scenario paths with regime transitions
        Now includes joint Treasury + Spread forecasting
        
        Args:
            current_state: Dict with current state information
                Required: 'factors', 'regime_probs'
                Optional: 'spreads' (dict with current spread decomposition)
            fed_shock_bp: Fed Funds shock in basis points
            random_seed: Random seed for reproducibility
            
        Returns:
            paths_df: DataFrame with all simulated paths
            percentiles_dict: Dict with percentiles at horizon
        """
        print(f"\n=== Generating {self.n_paths} Scenario Paths ===")
        print(f"Horizon: {self.horizon_days} days")
        print(f"Fed shock: {fed_shock_bp:+d} bps")
        print(f"Parallel workers: {self.n_jobs}")
        
        if self.spread_engine is not None and self.spread_engine.is_fitted:
            print(f"Including {len(self.spread_engine.sectors)} spread products: {self.spread_engine.sectors}")
        
        # 1. Get current factors
        f_T = current_state['factors']
        
        # 2. Compute steady-state covariance via Lyapunov equation
        try:
            P_T = solve_discrete_lyapunov(self.afns.A, self.afns.Q)
        except:
            # Fallback to Q if Lyapunov fails
            P_T = self.afns.Q.copy()
        
        # 3. Get current regime distribution
        gamma_T = current_state['regime_probs']
        
        # 4. Apply Fed shock to initial factors
        # Rule of thumb: 100bp Fed shock → 0.8bp Level, -0.1bp Slope
        level_impact = fed_shock_bp / 100 * 0.80
        slope_impact = fed_shock_bp / 100 * -0.10
        f_T_shocked = f_T.copy()
        f_T_shocked[0] += level_impact  # Level
        f_T_shocked[1] += slope_impact  # Slope
        
        # 5. Extract initial spread residuals if spread_engine is active
        z_initial = None
        if self.spread_engine is not None and self.spread_engine.is_fitted:
            if 'spreads' in current_state and current_state['spreads']:
                # Use provided spread decomposition
                spread_decomp = current_state['spreads']
                z_initial = np.array([
                    spread_decomp.get(sector, {}).get('residual', 0.0)
                    for sector in self.spread_engine.sectors
                ])
            else:
                # Use long-run mean of current regime
                regime_current = np.argmax(gamma_T)
                z_initial = self.spread_engine.mu[regime_current]
        
        # 6. Prepare simulation parameters
        sim_params = {
            'f_T': f_T_shocked,
            'P_T': P_T,
            'gamma_T': gamma_T,
            'horizon_days': self.horizon_days,
            'random_seed': random_seed,
            'z_initial': z_initial
        }
        
        # 7. Run simulations in parallel
        print("Simulating paths...")
        if self.n_jobs > 1:
            with Pool(processes=self.n_jobs) as pool:
                all_paths = list(tqdm(
                    pool.imap(self._simulate_single_path_wrapper,
                             [(i, sim_params) for i in range(self.n_paths)]),
                    total=self.n_paths,
                    desc="  Progress"
                ))
        else:
            # Sequential execution (for debugging)
            all_paths = []
            for i in tqdm(range(self.n_paths), desc="  Progress"):
                all_paths.append(self._simulate_single_path(i, sim_params))
        
        # 8. Combine all paths
        print("Combining paths...")
        paths_df = pd.concat(all_paths, ignore_index=True)
        
        # 9. Compute percentiles at horizon
        print("Computing percentiles...")
        final_df = paths_df[paths_df['t'] == self.horizon_days].copy()
        
        # Get yield columns
        yield_cols = [f'UST_{int(mat)}' for mat in self.afns.maturities]
        
        # CRISIS ADAPTATION: Scale uncertainty based on current MOVE (if available)
        vol_multiplier = 1.0
        if 'move' in current_state:
            current_move = current_state['move']
            move_baseline = 80.0  # Normal MOVE level
            
            if current_move > move_baseline * 1.5:  # Crisis threshold (>120)
                vol_multiplier = min(current_move / move_baseline, 3.0)  # Cap at 3x
                print(f"Crisis mode: MOVE={current_move:.1f}, scaling intervals by {vol_multiplier:.1f}x")
                
                # Scale the dispersion of yields around their mean
                for col in yield_cols:
                    col_mean = final_df[col].mean()
                    final_df[col] = col_mean + (final_df[col] - col_mean) * vol_multiplier
        
        # Compute percentiles for Treasuries
        percentiles_dict = {}
        for col in yield_cols:
            percentiles_dict[col] = {
                'p5': final_df[col].quantile(0.05),
                'p50': final_df[col].quantile(0.50),
                'p95': final_df[col].quantile(0.95),
                'mean': final_df[col].mean(),
                'std': final_df[col].std()
            }
        
        # Compute percentiles for spreads (Phase 2B)
        if self.spread_engine is not None and self.spread_engine.is_fitted:
            for sector in self.spread_engine.sectors:
                spread_col = f'{sector}_Spread'
                yield_col = f'{sector}_Yield'
                
                if spread_col in final_df.columns:
                    percentiles_dict[spread_col] = {
                        'p5': final_df[spread_col].quantile(0.05),
                        'p50': final_df[spread_col].quantile(0.50),
                        'p95': final_df[spread_col].quantile(0.95),
                        'mean': final_df[spread_col].mean(),
                        'std': final_df[spread_col].std()
                    }
                
                if yield_col in final_df.columns:
                    percentiles_dict[yield_col] = {
                        'p5': final_df[yield_col].quantile(0.05),
                        'p50': final_df[yield_col].quantile(0.50),
                        'p95': final_df[yield_col].quantile(0.95),
                        'mean': final_df[yield_col].mean(),
                        'std': final_df[yield_col].std()
                    }
        
        print(f"Simulation complete: {len(paths_df)} total rows")
        print(f"  Paths: {self.n_paths}")
        print(f"  Time steps per path: {self.horizon_days + 1}")
        if self.spread_engine is not None and self.spread_engine.is_fitted:
            print(f"  Spread products included: {', '.join(self.spread_engine.sectors)}")
        
        return paths_df, percentiles_dict
    
    def generate_multiple_scenarios(self, current_state, fed_shocks=None):
        """
        Generate multiple Fed policy scenarios
        
        Args:
            current_state: Dict with current state
            fed_shocks: List of Fed shock magnitudes (default: [-100, -50, 0, 50, 100])
            
        Returns:
            all_scenarios: Dict mapping shock -> (paths_df, percentiles_dict)
        """
        if fed_shocks is None:
            fed_shocks = [-100, -50, 0, 50, 100]
        
        all_scenarios = {}
        for shock in fed_shocks:
            print(f"\n{'='*60}")
            print(f"Fed Shock: {shock:+d} bps")
            print('='*60)
            paths_df, percentiles_dict = self.generate_scenario(current_state, shock)
            all_scenarios[shock] = {
                'paths': paths_df,
                'percentiles': percentiles_dict
            }
        
        return all_scenarios

    def generate_all_horizons(self, current_state, fed_shock_bp=0):
        """
        Generate scenarios for all configured horizons
        
        Returns:
            Dict mapping horizon_name -> (paths_df, percentiles)
        """
        all_horizons = {}
        
        for horizon_name, horizon_days in self.horizons.items():
            print(f"\nGenerating {horizon_name} ({horizon_days} days) scenarios...")
            
            # Update horizon for this run
            self.horizon_days = horizon_days
            
            # Generate scenarios
            paths_df, percentiles = self.generate_scenario(
                current_state, fed_shock_bp
            )
            
            all_horizons[horizon_name] = {
                'paths': paths_df,
                'percentiles': percentiles,
                'horizon_days': horizon_days
            }
        
        return all_horizons


def format_scenario_output(all_scenarios):
    """
    Format scenarios into readable DataFrame
    
    Args:
        all_scenarios: Dict mapping shock -> scenario results
        
    Returns:
        df_scenarios: DataFrame with formatted results
    """
    rows = []
    
    for shock, scenario_data in all_scenarios.items():
        percentiles = scenario_data['percentiles']
        
        for tenor, stats in percentiles.items():
            rows.append({
                'Fed_Shock_bp': shock,
                'Tenor': tenor,
                'p5': stats['p5'],
                'p50': stats['p50'],
                'p95': stats['p95'],
                'mean': stats['mean'],
                'std': stats['std']
            })
    
    df_scenarios = pd.DataFrame(rows)
    
    return df_scenarios


def run_scenario_analysis(df, df_factors, afns_model, hmm_model, 
                          regime_labels, regime_covs, spread_engine, config, 
                          as_of_date=None, fed_shocks=None):
    """
    Main function to run scenario analysis with path simulation
    Updated for Phase 2B: Now includes spread forecasting
    
    Args:
        df: DataFrame with yield curve data
        df_factors: DataFrame with AFNS factors
        afns_model: Fitted AFNS model
        hmm_model: Fitted StickyHMM model
        regime_labels: Historical regime assignments
        regime_covs: Covariance matrices by regime (not used in new implementation)
        spread_engine: SpreadEngine for credit spreads (Phase 2B)
        config: Configuration dict
        as_of_date: Date for scenario (default: latest)
        fed_shocks: List of Fed shocks to analyze
        
    Returns:
        all_scenarios: Dict of scenario results
        df_scenarios: Formatted DataFrame
        current_state: Dict with current state information
    """
    # Default to latest date
    if as_of_date is None:
        as_of_date = df.index[-1]
    
    # Get current state
    current_idx = df.index.get_loc(as_of_date)
    tenors = config['data']['tenors']
    current_yields = df.iloc[current_idx][tenors].values
    current_factors = afns_model.factors.iloc[current_idx].values  # [L, S, C]
    
    # regime_labels is a Series indexed by date (from factor_changes)
    try:
        current_regime_probs = hmm_model.regime_probs.loc[as_of_date].values
    except:
        current_regime_probs = hmm_model.regime_probs.iloc[-1].values
    
    # Build current state dict
    current_state = {
        'date': as_of_date,
        'yields': current_yields,
        'factors': current_factors,
        'regime_probs': current_regime_probs,
        'spreads': {}
    }
    
    # Add spread decomposition if spread_engine is available (Phase 2B)
    if spread_engine is not None and spread_engine.is_fitted:
        try:
            from src.utils import get_spread_dataframe
            spread_df = get_spread_dataframe(df, spread_engine.sectors, spread_engine.benchmark_maturity)
            current_spreads = spread_engine.get_current_spreads(spread_df, df_factors, as_of_date)
            current_state['spreads'] = current_spreads
        except Exception as e:
            print(f"Warning: Could not extract current spreads: {e}")
    
    # Initialize generator with path simulation and spread integration
    generator = ScenarioGenerator(
        afns_model=afns_model,
        hmm_model=hmm_model,
        spread_engine=spread_engine,
        n_paths=config['scenarios'].get('n_paths', 10000),
        horizons=config['scenarios'].get('horizons'),
        n_jobs=config['scenarios'].get('n_jobs', -1),
        config=config
    )
    
    # Generate scenarios
    if fed_shocks is None:
        fed_shocks = config['scenarios'].get('fed_shocks', [-100, 0, 100])
    
    all_scenarios = generator.generate_multiple_scenarios(current_state, fed_shocks)
    
    # Format output
    df_scenarios = format_scenario_output(all_scenarios)
    
    # Print summary
    print(f"\n=== Scenario Analysis Summary ===")
    print(f"As of: {as_of_date.date()}")
    print(f"Current 10yr yield: {current_yields[4]:.2f}%")
    print(f"\nScenario Results (at {list(config['scenarios']['horizons'].keys())[-1]}):")
    
    for shock in sorted(all_scenarios.keys()):
        percs = all_scenarios[shock]['percentiles']
        ust10_stats = percs.get('UST_10', percs.get(f'UST_{int(afns_model.maturities[4])}'))
        
        print(f"  Fed {shock:+4d}bp: {ust10_stats['p50']:.2f}% "
              f"[{ust10_stats['p5']:.2f}% - {ust10_stats['p95']:.2f}%]", end='')
        
        # Print spread products if available
        if spread_engine is not None and spread_engine.is_fitted:
            spread_info = []
            for sector in spread_engine.sectors:
                yield_key = f'{sector}_Yield'
                if yield_key in percs:
                    spread_info.append(f"{sector}={percs[yield_key]['p50']:.2f}%")
            if spread_info:
                print(f" | {', '.join(spread_info)}")
        else:
            print()
    
    return all_scenarios, df_scenarios, current_state