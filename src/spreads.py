"""
Spread Engine: Hierarchical decomposition with regime-dependent dynamics

Architecture:
    Spread_sector(t) = β_sector · f_t + z_sector(t)
    
    Where:
    - β_sector: (3×1) factor loadings - how sector responds to Treasury [L,S,C]
    - z_sector(t): Residual spread - credit/liquidity component NOT explained by Treasuries
    
    Residual dynamics:
    z_t = [z_Muni, z_Agency, z_Corp]  (3D vector)
    z_{t+1} = Γ_s · z_t + μ_s + ξ_t,  where ξ_t ~ N(0, Ψ_s)
    
    All parameters (Γ, μ, Ψ) are regime-dependent:
    - Calm regime: Fast mean reversion, tight spreads, low volatility
    - Volatile regime: Slower reversion, moderate widening, higher vol
    - Crisis regime: Very slow reversion, massive widening, explosive vol with contagion
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import solve_discrete_lyapunov
from sklearn.linear_model import LinearRegression
import logging

logger = logging.getLogger(__name__)


class SpreadEngine:
    """
    Hierarchical spread decomposition with regime-dependent residual dynamics
    """
    
    def __init__(self, sectors=['Muni', 'Agency', 'Corp'], benchmark_maturity=5):
        """
        Initialize spread engine
        
        Args:
            sectors: List of sector names (must match spread column naming)
            benchmark_maturity: Maturity to use for spread modeling (default 5yr)
        """
        self.sectors = sectors
        self.n_sectors = len(sectors)
        self.benchmark_maturity = benchmark_maturity
        
        # Factor loadings: β_sector for each sector (estimated in fit())
        self.factor_loadings = {}  # Dict: sector -> (3,) array [β_L, β_S, β_C]
        
        # Residual spread data
        self.residuals = None  # DataFrame with columns [z_Muni, z_Agency, z_Corp]
        
        # Regime-dependent residual VAR parameters
        self.n_regimes = None
        self.Gamma = {}  # Dict: regime -> (3×3) mean reversion matrix
        self.mu = {}     # Dict: regime -> (3,) long-run mean
        self.Psi = {}    # Dict: regime -> (3×3) covariance matrix
        
        # Fitted flag
        self.is_fitted = False
        
    def fit(self, spread_df, factors_df, regime_labels):
        """
        Fit hierarchical spread model
        
        Stage 1: Estimate factor loadings β_sector via OLS
        Stage 2: Extract residuals z_sector
        Stage 3: Fit regime-dependent VAR on residuals
        
        Args:
            spread_df: DataFrame with spread columns [Muni_Spread_5Y, Agency_Spread_5Y, Corp_Spread_5Y]
            factors_df: DataFrame with AFNS factors [L, S, C]
            regime_labels: Series with regime assignments (0, 1, 2, ...)
            
        Returns:
            results: Dict with fitted parameters and diagnostics
        """
        logger.info("\n=== Fitting Hierarchical Spread Model ===")
        
        # Align data
        common_idx = spread_df.index.intersection(factors_df.index).intersection(regime_labels.index)
        spread_df = spread_df.loc[common_idx]
        factors_df = factors_df.loc[common_idx]
        regime_labels = regime_labels.loc[common_idx]
        
        self.n_regimes = int(regime_labels.max()) + 1
        
        # Stage 1: Estimate factor loadings for each sector
        logger.info("Stage 1: Estimating factor loadings β_sector via OLS...")
        
        X = factors_df[['L', 'S', 'C']].values  # (T, 3)
        
        residuals_dict = {}
        
        for sector in self.sectors:
            spread_col = f'{sector}_Spread_{self.benchmark_maturity}Y'
            
            if spread_col not in spread_df.columns:
                raise ValueError(f"Missing spread column: {spread_col}")
            
            y = spread_df[spread_col].values  # (T,)
            
            # OLS: Spread = β_0 + β_L*L + β_S*S + β_C*C + residual
            reg = LinearRegression()
            reg.fit(X, y)
            
            # Extract loadings (no intercept - spreads should be zero-mean after factor effects)
            beta = reg.coef_  # (3,) array
            self.factor_loadings[sector] = beta
            
            # Compute fitted values and residuals
            spread_fitted = X @ beta
            residual = y - spread_fitted
            residuals_dict[f'z_{sector}'] = residual
            
            # R² and diagnostics
            ss_res = np.sum(residual**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2 = 1 - (ss_res / ss_tot)
            
            logger.info(f"  {sector}:")
            logger.info(f"    Factor loadings β: [L={beta[0]:.3f}, S={beta[1]:.3f}, C={beta[2]:.3f}]")
            logger.info(f"    R²: {r2:.3f} (Treasury factors explain {r2*100:.1f}% of spread variation)")
            logger.info(f"    Residual std: {residual.std():.2f} bps")
        
        # Store residuals as DataFrame
        self.residuals = pd.DataFrame(residuals_dict, index=common_idx)
        
        # Stage 2: Fit regime-dependent VAR on residuals
        logger.info("\nStage 2: Fitting regime-dependent residual VAR...")
        
        # Compute residual changes for VAR estimation
        residual_changes = self.residuals.diff().dropna()
        regime_labels_aligned = regime_labels.loc[residual_changes.index]
        
        for regime in range(self.n_regimes):
            logger.info(f"\n  Regime {regime}:")
            
            # Filter to this regime
            regime_mask = regime_labels_aligned == regime
            z_regime = self.residuals.loc[regime_labels_aligned.index][regime_mask]
            
            if len(z_regime) < 30:
                logger.warning(f"    Only {len(z_regime)} observations - using pooled estimates")
                # Use pooled estimates as fallback
                z_regime = self.residuals
            
            # Estimate VAR(1): z_t = Γ·z_{t-1} + μ + ξ_t
            z_t = z_regime.iloc[1:].values  # (T-1, 3)
            z_tm1 = z_regime.iloc[:-1].values  # (T-1, 3)
            
            # OLS for each equation
            Gamma_regime = np.zeros((self.n_sectors, self.n_sectors))
            mu_regime = np.zeros(self.n_sectors)
            residuals_var = []
            
            for i in range(self.n_sectors):
                # Fit: z_{i,t} = Γ_{i,·} @ z_{t-1} + μ_i + ξ_{i,t}
                reg = LinearRegression()
                reg.fit(z_tm1, z_t[:, i])
                
                Gamma_regime[i, :] = reg.coef_
                mu_regime[i] = reg.intercept_
                
                # Collect residuals for covariance
                fitted = z_tm1 @ reg.coef_ + reg.intercept_
                resid = z_t[:, i] - fitted
                residuals_var.append(resid)
            
            residuals_var = np.column_stack(residuals_var)  # (T-1, 3)
            Psi_regime = np.cov(residuals_var.T)
            
            # Store parameters
            self.Gamma[regime] = Gamma_regime
            self.mu[regime] = mu_regime
            self.Psi[regime] = Psi_regime
            
            # Diagnostics
            eigenvalues = np.linalg.eigvals(Gamma_regime)
            max_eig = np.max(np.abs(eigenvalues))
            is_stable = max_eig < 1.0
            
            # Mean reversion speed (diagonal elements)
            diag_gamma = np.diag(Gamma_regime)
            half_lives = -np.log(2) / np.log(diag_gamma + 1e-10)
            
            logger.info(f"    Γ diagonal (mean reversion): {diag_gamma}")
            logger.info(f"    Half-lives (days): {half_lives}")
            logger.info(f"    Long-run means μ: {mu_regime}")
            logger.info(f"    Max |eigenvalue|: {max_eig:.3f} {'(stable)' if is_stable else '(unstable!)'}")
            logger.info(f"    Volatility (diag Ψ): {np.sqrt(np.diag(Psi_regime))}")
            
            # Check for contagion (off-diagonal correlation)
            corr_matrix = np.corrcoef(residuals_var.T)
            max_corr = np.max(np.abs(corr_matrix - np.eye(self.n_sectors)))
            logger.info(f"    Max cross-correlation: {max_corr:.3f}")
        
        self.is_fitted = True
        
        # Return summary results
        results = {
            'factor_loadings': self.factor_loadings,
            'residuals': self.residuals,
            'regime_params': {
                'Gamma': self.Gamma,
                'mu': self.mu,
                'Psi': self.Psi
            },
            'n_regimes': self.n_regimes
        }
        
        logger.info("\n=== Spread Model Fitting Complete ===")
        
        return results
    
    def forecast_spreads(self, factors_current, regime_current, z_current=None, 
                        horizon=126, n_paths=1000, random_seed=42):
        """
        Forecast spread distributions at horizon
        
        Args:
            factors_current: Current AFNS factors [L, S, C]
            regime_current: Current regime (integer)
            z_current: Current residual spreads (optional, uses long-run mean if None)
            horizon: Forecast horizon in days
            n_paths: Number of Monte Carlo paths
            random_seed: Random seed
            
        Returns:
            spread_paths: Dict mapping sector -> (n_paths, horizon+1) array of spread forecasts
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before forecasting")
        
        np.random.seed(random_seed)
        
        # Initialize residuals
        if z_current is None:
            # Use long-run mean of current regime
            z_current = self.mu[regime_current]
        else:
            z_current = np.array(z_current)
        
        # Storage for residual paths
        z_paths = np.zeros((n_paths, horizon + 1, self.n_sectors))
        z_paths[:, 0, :] = z_current
        
        # Simulate residual paths (assuming regime stays constant - will be updated by ScenarioGenerator)
        Gamma_s = self.Gamma[regime_current]
        mu_s = self.mu[regime_current]
        Psi_s = self.Psi[regime_current]
        
        for t in range(1, horizon + 1):
            # VAR dynamics: z_t = Γ·z_{t-1} + μ + ξ_t
            z_prev = z_paths[:, t-1, :]  # (n_paths, 3)
            
            # Generate noise
            xi = np.random.multivariate_normal(np.zeros(self.n_sectors), Psi_s, size=n_paths)
            
            # Propagate
            z_paths[:, t, :] = (z_prev @ Gamma_s.T) + mu_s + xi
        
        # Convert residual paths to spread paths
        # Spread = β·factors + z
        # For simplicity, assume factors constant at factors_current (will be updated in joint simulation)
        systematic_component = factors_current @ np.column_stack([
            self.factor_loadings[sector] for sector in self.sectors
        ])  # (3,) @ (3, n_sectors) = (n_sectors,)
        
        spread_paths = {}
        for i, sector in enumerate(self.sectors):
            # Total spread = systematic + residual
            spread_paths[sector] = z_paths[:, :, i] + systematic_component[i]
        
        return spread_paths
    
    def simulate_spreads(self, factor_paths, regime_paths, z_initial, random_seed=42):
        """
        Simulate spread paths conditional on Treasury factor and regime paths
        
        This is the key method for integration with ScenarioGenerator
        
        Args:
            factor_paths: (horizon+1, 3) array of factor paths [L, S, C]
            regime_paths: (horizon+1,) array of regime paths
            z_initial: (3,) array of initial residual spreads [z_Muni, z_Agency, z_Corp]
            random_seed: Random seed
            
        Returns:
            spread_paths: Dict mapping sector -> (horizon+1,) array of spread values
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before simulating")
        
        np.random.seed(random_seed)
        
        horizon = len(factor_paths) - 1
        
        # Initialize residual path
        z_path = np.zeros((horizon + 1, self.n_sectors))
        z_path[0, :] = z_initial
        
        # Simulate residuals with regime-dependent dynamics
        for t in range(1, horizon + 1):
            regime_t = int(regime_paths[t])
            
            # Get regime-specific parameters
            Gamma_s = self.Gamma[regime_t]
            mu_s = self.mu[regime_t]
            Psi_s = self.Psi[regime_t]
            
            # Generate noise
            xi = np.random.multivariate_normal(np.zeros(self.n_sectors), Psi_s)
            
            # VAR dynamics: z_t = Γ·z_{t-1} + μ + ξ_t
            z_path[t, :] = (z_path[t-1, :] @ Gamma_s.T) + mu_s + xi
        
        # Compute systematic component at each time
        # β·f_t for each sector
        beta_matrix = np.column_stack([
            self.factor_loadings[sector] for sector in self.sectors
        ])  # (3, n_sectors)
        
        systematic_paths = factor_paths @ beta_matrix  # (horizon+1, n_sectors)
        
        # Total spread = systematic + residual
        spread_paths = {}
        for i, sector in enumerate(self.sectors):
            spread_paths[sector] = systematic_paths[:, i] + z_path[:, i]
        
        return spread_paths
    
    def get_current_spreads(self, spread_df, factors_df, date):
        """
        Extract current spread levels and decompose into systematic + residual
        
        Args:
            spread_df: DataFrame with spread data
            factors_df: DataFrame with factors
            date: Date to extract
            
        Returns:
            Dict with 'total', 'systematic', 'residual' for each sector
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before decomposition")
        
        current_spreads = {}
        
        factors = factors_df.loc[date, ['L', 'S', 'C']].values
        
        for sector in self.sectors:
            spread_col = f'{sector}_Spread_{self.benchmark_maturity}Y'
            total_spread = spread_df.loc[date, spread_col]
            
            # Systematic component: β·f
            systematic = factors @ self.factor_loadings[sector]
            
            # Residual: observed - systematic
            residual = total_spread - systematic
            
            current_spreads[sector] = {
                'total': total_spread,
                'systematic': systematic,
                'residual': residual
            }
        
        return current_spreads
    
    def validate_economic_intuition(self):
        """
        Validate that factor loadings match economic intuition
        
        Expected patterns:
        - Munis: High Slope sensitivity (steep = more carry = tighter)
        - Agencies: High Level sensitivity (prepayment risk)
        - Corps: High Curvature sensitivity (inversion = recession = wider)
        """
        if not self.is_fitted:
            raise ValueError("Must fit model first")
        
        logger.info("\n=== Economic Validation of Factor Loadings ===")
        
        for sector in self.sectors:
            beta = self.factor_loadings[sector]
            logger.info(f"\n{sector}:")
            logger.info(f"  β_Level = {beta[0]:.3f}")
            logger.info(f"  β_Slope = {beta[1]:.3f}")
            logger.info(f"  β_Curvature = {beta[2]:.3f}")
            
            # Interpretation
            if sector == 'Muni':
                if abs(beta[1]) > abs(beta[0]) and abs(beta[1]) > abs(beta[2]):
                    logger.info("  ✓ Slope dominant (expected for tax-exempt carry dynamics)")
                else:
                    logger.warning("  ⚠ Expected Slope to dominate for Munis")
            
            elif sector == 'Agency':
                if abs(beta[0]) > abs(beta[1]):
                    logger.info("  ✓ Level dominant (expected for prepayment-sensitive)")
                else:
                    logger.warning("  ⚠ Expected Level to dominate for Agencies")
            
            elif sector == 'Corp':
                if abs(beta[2]) > 0.15 or abs(beta[0]) > 0.15:
                    logger.info("  ✓ Curvature/Level sensitive (expected for credit)")
                else:
                    logger.warning("  ⚠ Expected stronger credit sensitivity")