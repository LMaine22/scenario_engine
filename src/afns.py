"""
AFNS Model: Extract Level, Slope, Curvature factors from yield curve
Uses VAR(1) state-space model with Kalman filter estimation
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import solve_discrete_lyapunov
from filterpy.kalman import KalmanFilter
from sklearn.covariance import LedoitWolf


class AFNSModel:
    """Arbitrage-Free Nelson-Siegel model"""
    
    def __init__(self, maturities, lambda_param=0.0609, measurement_noise_R=4.0):
        """
        Args:
            maturities: List of maturities in years [2, 3, 5, 7, 10, 30]
            lambda_param: Decay parameter (default 0.0609 is rule of thumb)
            measurement_noise_R: Measurement noise variance in bps² (default 4.0)
        """
        self.maturities = np.array(maturities)
        self.n_maturities = len(maturities)
        self.lambda_param = lambda_param
        self.loadings = self._compute_loadings()
        self.H = self.loadings.T  # n_maturities × 3 loading matrix
        self.R = np.eye(self.n_maturities) * measurement_noise_R
        
        # VAR(1) parameters (estimated in fit())
        self.A = None
        self.b = None
        self.Q = None
        self.factors = None
        self.factor_changes = None
        
    def _compute_loadings(self):
        """Compute factor loadings for each maturity"""
        tau = self.maturities
        lam = self.lambda_param
        
        # Level loading: always 1
        level = np.ones_like(tau)
        
        # Slope loading: (1 - exp(-λτ)) / (λτ)
        slope = (1 - np.exp(-lam * tau)) / (lam * tau)
        
        # Curvature loading: ((1 - exp(-λτ)) / (λτ)) - exp(-λτ)
        curvature = ((1 - np.exp(-lam * tau)) / (lam * tau)) - np.exp(-lam * tau)
        
        # Stack into matrix (3 factors x n_maturities)
        return np.vstack([level, slope, curvature])
    
    def initialize_factors_dns(self, yields_df):
        """
        Initialize factors using DNS approximation
        
        Args:
            yields_df: DataFrame with yield columns
            
        Returns:
            factors_df: DataFrame with columns L, S, C
        """
        # Extract required yields (assuming column names like 'USGG2YR Index')
        cols = yields_df.columns
        
        # Find 2yr, 5yr, 10yr yields
        ust_2 = None
        ust_5 = None
        ust_10 = None
        
        for col in cols:
            if '2YR' in col or '2Y' in col:
                ust_2 = yields_df[col]
            elif '5YR' in col or '5Y' in col:
                ust_5 = yields_df[col]
            elif '10YR' in col or '10Y' in col:
                ust_10 = yields_df[col]
        
        if ust_2 is None or ust_5 is None or ust_10 is None:
            raise ValueError("Missing required yields (2Y, 5Y, 10Y) for DNS initialization")
        
        # DNS formulas
        L = (ust_2 + ust_10) / 2
        S = ust_10 - ust_2
        C = 2 * ust_5 - ust_2 - ust_10
        
        factors_df = pd.DataFrame({
            'L': L,
            'S': S,
            'C': C
        }, index=yields_df.index)
        
        return factors_df
    
    def _kalman_filter(self, yields, A, b, Q):
        """
        Run Kalman filter with given parameters
        
        Args:
            yields: Array (T, n_maturities) of yield observations
            A: State transition matrix (3, 3)
            b: State drift vector (3,)
            Q: State noise covariance (3, 3)
            
        Returns:
            filtered_factors: Array (T, 3) of filtered factor estimates
            log_likelihood: Scalar log-likelihood
        """
        T = len(yields)
        
        # Initialize Kalman filter
        kf = KalmanFilter(dim_x=3, dim_z=self.n_maturities)
        
        # State transition: f_{t+1} = A f_t + b + η
        kf.F = A
        kf.B = None  # No control input
        kf.Q = Q
        
        # Observation: y_t = H f_t + ε
        kf.H = self.H
        kf.R = self.R
        
        # Initial state: use least squares on first observation
        y0 = yields[0, :]
        f0 = np.linalg.lstsq(self.H, y0, rcond=None)[0]
        kf.x = f0
        kf.P = np.eye(3) * 10  # Initial uncertainty
        
        # Storage
        filtered_factors = np.zeros((T, 3))
        log_likelihood = 0.0
        
        # Filter forward
        for t in range(T):
            # Predict
            kf.predict()
            
            # Add drift b manually (filterpy doesn't support constant drift in predict)
            kf.x = kf.x + b
            
            y_t = yields[t, :]
            
            # Update if observation is valid
            if not np.any(np.isnan(y_t)):
                # Compute innovation before update
                innovation = y_t - kf.H @ kf.x
                S = kf.H @ kf.P @ kf.H.T + kf.R
                
                # Update
                kf.update(y_t)
                
                # Accumulate log-likelihood
                try:
                    log_likelihood += -0.5 * (
                        self.n_maturities * np.log(2 * np.pi) +
                        np.log(np.linalg.det(S) + 1e-10) +
                        innovation.T @ np.linalg.solve(S, innovation)
                    )
                except:
                    log_likelihood += -1e10  # Penalize if matrix is singular
            
            filtered_factors[t, :] = kf.x.flatten()
        
        return filtered_factors, log_likelihood
    
    def _objective(self, params, yields):
        """
        Negative log-likelihood for optimization
        
        Args:
            params: Flattened parameters [A (9), b (3), Q_diag (3)]
            yields: Yield observations
            
        Returns:
            neg_log_likelihood: Scalar objective value
        """
        # Unpack parameters
        A = params[:9].reshape(3, 3)
        b = params[9:12]
        Q_diag = np.abs(params[12:15])  # Ensure positive
        Q = np.diag(Q_diag)
        
        # Stability constraint: eigenvalues < 0.999
        eigvals = np.linalg.eigvals(A)
        if np.any(np.abs(eigvals) > 0.999):
            return 1e10
        
        # Run Kalman filter
        try:
            _, log_likelihood = self._kalman_filter(yields, A, b, Q)
            return -log_likelihood
        except:
            return 1e10
    
    def fit(self, yields_df, max_iter=100):
        """
        Fit AFNS model using Maximum Likelihood Estimation
        
        Args:
            yields_df: DataFrame with yield curve data
            max_iter: Maximum iterations for optimization
            
        Returns:
            results: Dict with fitted parameters and diagnostics
        """
        from tqdm import tqdm
        
        print("\n=== Fitting AFNS VAR(1) + Kalman Filter ===")
        
        # Extract yields matrix
        yields = yields_df.values
        
        # 1. Initialize factors using DNS
        print("Initializing factors using DNS approximation...")
        factors_init = self.initialize_factors_dns(yields_df)
        
        # 2. Initialize parameters from DNS factors
        print("Setting initial parameter guesses...")
        
        # A: Start with high persistence on diagonal
        A_init = np.eye(3) * 0.95
        
        # b: Small drift toward factor means
        b_init = factors_init.mean().values * 0.05
        
        # Q: Diagonal covariance from DNS factor changes
        factor_changes_init = factors_init.diff().dropna()
        Q_init = np.diag(factor_changes_init.var().values)
        
        # Pack into parameter vector
        params_init = np.concatenate([
            A_init.flatten(),
            b_init,
            np.diag(Q_init)
        ])
        
        # 3. Optimize using BFGS with progress bar
        print("Running maximum likelihood optimization via BFGS...")
        print(f"  Optimizing {len(params_init)} parameters (A, b, Q)")
        print(f"  This may take 20-30 seconds...")
        
        # Create progress bar
        pbar = tqdm(total=max_iter, desc="  BFGS iterations", unit="iter", leave=False)
        self._iteration_count = 0
        
        def callback(xk):
            """Callback to update progress bar"""
            self._iteration_count += 1
            pbar.update(1)
        
        result = minimize(
            self._objective,
            params_init,
            args=(yields,),
            method='BFGS',
            callback=callback,
            options={'maxiter': max_iter, 'disp': False}
        )
        
        pbar.close()
        
        if not result.success:
            print(f"\n  ⚠ Warning: {result.message}")
            print(f"  This is often OK - optimizer stopped due to numerical precision limits")
            print(f"  Check RMSE below to verify fit quality")
        else:
            print(f"\n  ✓ Converged in {result.nit} iterations")
        
        # 4. Extract fitted parameters
        self.A = result.x[:9].reshape(3, 3)
        self.b = result.x[9:12]
        Q_diag = np.abs(result.x[12:15])
        self.Q = np.diag(Q_diag)
        
        # 5. Run final Kalman filter to get filtered factors
        print("Running final Kalman filter...")
        filtered_factors, log_likelihood = self._kalman_filter(yields, self.A, self.b, self.Q)
        
        # 6. Store factors as DataFrame
        self.factors = pd.DataFrame(
            filtered_factors,
            index=yields_df.index,
            columns=['L', 'S', 'C']
        )
        
        # 7. Compute factor changes
        self.factor_changes = self.factors.diff().dropna()
        
        # 8. Compute RMSE by maturity
        yields_fitted = (self.H @ self.factors.T).T
        errors = yields - yields_fitted
        rmse_by_maturity = np.sqrt(np.mean(errors**2, axis=0))
        overall_rmse = np.sqrt(np.mean(errors**2))
        
        # Print diagnostics
        print(f"\nFitted VAR(1) Parameters:")
        print(f"  Transition matrix A:")
        print(f"{self.A}")
        print(f"  Drift vector b: {self.b}")
        print(f"  State covariance Q (diagonal): {np.diag(self.Q)}")
        print(f"\nLog-likelihood: {log_likelihood:.2f}")
        print(f"Overall RMSE: {overall_rmse:.4f} bps")
        print(f"\nRMSE by maturity:")
        for i, mat in enumerate(self.maturities):
            print(f"  {mat}yr: {rmse_by_maturity[i]:.4f} bps")
        
        # 9. Return results dict
        results = {
            'A': self.A,
            'b': self.b,
            'Q': self.Q,
            'R': self.R,
            'log_likelihood': log_likelihood,
            'rmse_by_maturity': rmse_by_maturity,
            'overall_rmse': overall_rmse,
            'factors': self.factors,
            'factor_changes': self.factor_changes
        }
        
        return results
    
    def fit_factors(self, yields):
        """
        Extract L/S/C factors from yield curve using OLS
        
        Args:
            yields: Array of shape (n_dates, n_maturities)
            
        Returns:
            factors: Array of shape (n_dates, 3) with [Level, Slope, Curvature]
        """
        # OLS: yields = loadings.T @ factors
        # factors = (loadings @ loadings.T)^-1 @ loadings @ yields.T
        
        L = self.loadings  # (3, n_maturities)
        
        # Solve for factors
        factors = np.linalg.lstsq(L.T, yields.T, rcond=None)[0].T
        
        return factors
    
    def reconstruct_yields(self, factors):
        """
        Reconstruct yield curve from L/S/C factors
        
        Args:
            factors: Array of shape (n_dates, 3) with [Level, Slope, Curvature]
            
        Returns:
            yields: Array of shape (n_dates, n_maturities)
        """
        return factors @ self.loadings
    
    def compute_residuals(self, yields, factors):
        """
        Compute residuals: actual - fitted
        
        Args:
            yields: Actual yields (n_dates, n_maturities)
            factors: Fitted factors (n_dates, 3)
            
        Returns:
            residuals: Array of shape (n_dates, n_maturities)
        """
        fitted = self.reconstruct_yields(factors)
        return yields - fitted
    
    def estimate_covariance(self, factor_returns, method='ledoit_wolf'):
        """
        Estimate factor covariance matrix with shrinkage
        
        Args:
            factor_returns: Daily changes in factors (n_dates, 3)
            method: 'ledoit_wolf' or 'sample'
            
        Returns:
            cov_matrix: (3, 3) covariance matrix
        """
        if method == 'ledoit_wolf':
            lw = LedoitWolf()
            cov_matrix = lw.fit(factor_returns).covariance_
        else:
            cov_matrix = np.cov(factor_returns.T)
        
        return cov_matrix


def fit_afns_model(df, config):
    """
    Main function to fit AFNS model and extract factors using VAR(1) + Kalman filter
    
    Args:
        df: DataFrame with yield curve data
        config: Configuration dict
        
    Returns:
        df_factors: DataFrame with [L, S, C] factors
        afns: Fitted AFNS model object
        cov_matrix: Factor covariance matrix (Q)
    """
    # Get configuration
    tenors = config['data']['tenors']
    maturities = config['data']['maturities']
    measurement_noise_R = config['afns'].get('measurement_noise_R', 4.0)
    
    # Extract yields DataFrame
    yields_df = df[tenors]
    
    # Initialize and fit AFNS with VAR(1) + Kalman filter
    afns = AFNSModel(
        maturities=maturities,
        lambda_param=config['afns']['lambda'],
        measurement_noise_R=measurement_noise_R
    )
    
    # Fit model using maximum likelihood
    results = afns.fit(yields_df, max_iter=100)
    
    # Extract factors DataFrame
    df_factors = results['factors']
    
    # Print summary statistics
    print(f"\nFactor Summary Statistics:")
    print(df_factors.describe())
    
    print(f"\nFactor Correlations:")
    print(df_factors.corr())
    
    # Return factors, model, and covariance matrix
    return df_factors, afns, results['Q']
