"""Conformal Prediction: Distribution-free calibration with guaranteed coverage"""
import numpy as np
import pandas as pd
from scipy import stats


class ConformalScenarioGenerator:
    """Split conformal prediction for time-series scenario generation"""
    def __init__(self, base_generator, confidence_level=0.90):
        """Initialize with base scenario generator and target confidence level."""
        self.base = base_generator
        self.alpha = 1 - confidence_level
        self.residuals_by_regime = {}
        self.quantiles_by_regime = {}
        self.is_calibrated = False

    def calibrate(self, df, df_factors, regime_labels, calibration_dates, regime_covs):
        """Calibrate conformal quantiles using a dedicated calibration set."""
        tenors = self.base.config['data']['tenors']
        cal_mask = df.index.isin(calibration_dates)
        df_cal = df[cal_mask]
        df_factors_cal = df_factors[cal_mask]
        
        # regime_labels is a Series, filter by dates that are in calibration set
        regime_labels_cal = regime_labels[regime_labels.index.isin(calibration_dates)]
        n_regimes = int(regime_labels.max()) + 1
        
        # Store regime_covs for use in calibration
        self.regime_covs = regime_covs
        self.residuals_by_regime = {
            regime: {tenor: [] for tenor in tenors}
            for regime in range(n_regimes)
        }
        print(f"\nCalibrating conformal predictor on {len(df_cal)-1} observations...")
        for i in range(len(df_cal) - 1):
            current_factors = df_factors_cal.iloc[i][['L', 'S', 'C']].values
            
            # Get current regime from regime_labels Series
            date = df_cal.index[i]
            try:
                current_regime = int(regime_labels_cal.loc[date])
            except:
                current_regime = int(regime_labels_cal.iloc[0])
            
            # Generate random shocks using regime covariance
            cov = self.regime_covs[current_regime] if current_regime in self.regime_covs else self.regime_covs[0]
            factor_shocks = np.random.multivariate_normal(
                mean=np.zeros(3),
                cov=cov,
                size=1000
            )
            sim_factors = current_factors + factor_shocks
            sim_yields = self.base.afns.reconstruct_yields(sim_factors)
            forecast = np.median(sim_yields, axis=0)
            realized = df_cal.iloc[i + 1][tenors].values
            residuals = realized - forecast
            for j, tenor in enumerate(tenors):
                self.residuals_by_regime[current_regime][tenor].append(residuals[j])
        self.quantiles_by_regime = {}
        for regime in range(n_regimes):
            self.quantiles_by_regime[regime] = {}
            for tenor in tenors:
                residuals = np.array(self.residuals_by_regime[regime][tenor])
                if len(residuals) > 10:
                    n = len(residuals)
                    lower_pct = 100 * (n + 1) * (self.alpha / 2) / n
                    upper_pct = 100 * (1 - (n + 1) * (self.alpha / 2) / n)
                    q_lower = np.percentile(residuals, lower_pct)
                    q_upper = np.percentile(residuals, upper_pct)
                else:
                    all_residuals = [
                        res
                        for r in range(n_regimes)
                        for res in self.residuals_by_regime[r][tenor]
                    ]
                    all_residuals = np.array(all_residuals)
                    if len(all_residuals) == 0:
                        q_lower = 0.0
                        q_upper = 0.0
                    else:
                        q_lower = np.percentile(all_residuals, 100 * self.alpha / 2)
                        q_upper = np.percentile(all_residuals, 100 * (1 - self.alpha / 2))
                self.quantiles_by_regime[regime][tenor] = (q_lower, q_upper)
        self.is_calibrated = True
        print("Conformal calibration complete!")

    def generate_scenario(self, current_state, fed_shock_bp, regime):
        """Generate conformal-adjusted scenario percentiles for a given state."""
        if not self.is_calibrated:
            raise ValueError("Must call calibrate() before generating scenarios")
        base_scenario = self.base.generate_scenario(current_state, fed_shock_bp, regime)
        tenors = self.base.config['data']['tenors']
        conformal_scenario = {
            'base_yields': base_scenario['base_yields'],
            'percentiles': {},
            'mean_yields': base_scenario['mean_yields'],
            'std_yields': base_scenario['std_yields']
        }
        for tenor in tenors:
            base_forecast = base_scenario['percentiles'][tenor]['p50']
            q_lower, q_upper = self.quantiles_by_regime[regime][tenor]
            conformal_scenario['percentiles'][tenor] = {
                'p5': base_forecast + q_lower,
                'p50': base_forecast,
                'p95': base_forecast + q_upper
            }
        return conformal_scenario


def compute_crps(forecasts, realized, percentiles=(5, 50, 95)):
    """Compute a simplified CRPS estimate from discrete quantile forecasts."""
    crps_scores = []
    for i in range(len(realized)):
        p5 = forecasts[i]['p5']
        p50 = forecasts[i]['p50']
        p95 = forecasts[i]['p95']
        y = realized[i]
        crps = (
            0.05 * abs(y - p5) +
            0.45 * abs(y - p50) +
            0.45 * abs(y - p95) +
            0.05 * (p95 - p5)
        )
        crps_scores.append(crps)
    return np.mean(crps_scores) if crps_scores else np.nan
