"""
Regime Classification: Sticky Hidden Markov Model on factor changes
"""
import numpy as np
import pandas as pd
from hmmlearn import hmm


class StickyHMM:
    """Sticky Hidden Markov Model with BIC-based model selection"""
    
    def __init__(self, n_states_range=[2, 3, 4, 5], sticky_alpha=10.0, 
                 ambiguity_threshold=0.45, random_state=42):
        """
        Args:
            n_states_range: List of K values to try for model selection
            sticky_alpha: Dirichlet prior weight for self-transitions
            ambiguity_threshold: Warn if max regime prob < this threshold
            random_state: Random seed for reproducibility
        """
        self.n_states_range = n_states_range
        self.sticky_alpha = sticky_alpha
        self.ambiguity_threshold = ambiguity_threshold
        self.random_state = random_state
        
        # Fitted parameters (set in fit())
        self.n_states = None
        self.model = None
        self.regime_means = None
        self.regime_covs = None
        self.transition_matrix = None
        self.regime_probs = None
        
    def _apply_sticky_prior(self, trans_matrix, alpha):
        """
        Add Dirichlet prior weight to diagonal and renormalize
        
        Args:
            trans_matrix: Transition matrix (K × K)
            alpha: Dirichlet prior weight for self-transitions
            
        Returns:
            sticky_matrix: Regularized transition matrix
        """
        n_states = trans_matrix.shape[0]
        sticky_matrix = trans_matrix.copy()
        
        # Add alpha to diagonal (self-transitions)
        for k in range(n_states):
            sticky_matrix[k, k] += alpha
        
        # Renormalize rows to sum to 1
        sticky_matrix = sticky_matrix / sticky_matrix.sum(axis=1, keepdims=True)
        
        return sticky_matrix
    
    def _compute_bic(self, model, X):
        """
        Compute Bayesian Information Criterion
        
        Args:
            model: Fitted GaussianHMM
            X: Data
            
        Returns:
            bic: BIC value (lower is better)
        """
        log_likelihood = model.score(X)
        n_features = model.n_features
        n_components = model.n_components
        
        # Parameter count
        n_params = (
            n_features * n_components +  # Means
            n_features * (n_features + 1) / 2 * n_components +  # Full covariances
            n_components * (n_components - 1)  # Transition matrix (rows sum to 1)
        )
        
        bic = -2 * log_likelihood + n_params * np.log(len(X))
        return bic
    
    def _compute_regime_duration(self, trans_matrix):
        """
        Compute expected regime duration
        
        Args:
            trans_matrix: Transition matrix
            
        Returns:
            durations: Expected duration for each regime (in days)
        """
        durations = 1 / (1 - np.diag(trans_matrix))
        return durations
    
    def fit(self, factor_changes):
        """
        Fit Sticky HMM with BIC-based model selection
        
        Args:
            factor_changes: DataFrame with columns [L, S, C] (factor changes Δf_t)
            
        Returns:
            results: Dict with fitted parameters
        """
        print("\n=== Fitting Sticky HMM ===")
        
        # Convert to numpy array
        X = factor_changes.values
        
        # Try different numbers of states
        bic_scores = {}
        models = {}
        
        print(f"Testing K ∈ {self.n_states_range} states...")
        
        for K in self.n_states_range:
            print(f"  Fitting HMM with K={K} states...")
            
            # Fit base HMM
            model = hmm.GaussianHMM(
                n_components=K,
                covariance_type='full',
                n_iter=100,
                random_state=self.random_state,
                verbose=False
            )
            
            try:
                model.fit(X)
                
                # Apply stickiness to transition matrix
                model.transmat_ = self._apply_sticky_prior(model.transmat_, self.sticky_alpha)
                
                # Compute BIC
                bic = self._compute_bic(model, X)
                bic_scores[K] = bic
                models[K] = model
                
                # Compute average regime duration
                durations = self._compute_regime_duration(model.transmat_)
                avg_duration = np.mean(durations)
                
                print(f"    BIC: {bic:.2f}, Avg regime duration: {avg_duration:.1f} days")
                
            except Exception as e:
                print(f"    Failed to fit K={K}: {e}")
                bic_scores[K] = np.inf
        
        # Select best K by BIC (lower is better)
        self.n_states = min(bic_scores, key=bic_scores.get)
        self.model = models[self.n_states]
        
        print(f"\nSelected K={self.n_states} states (lowest BIC)")
        
        # Extract parameters
        self.regime_means = self.model.means_
        self.regime_covs = self.model.covars_
        self.transition_matrix = self.model.transmat_
        
        # Compute regime probabilities for all time periods
        self.regime_probs = pd.DataFrame(
            self.model.predict_proba(X),
            index=factor_changes.index,
            columns=[f'Regime_{k}' for k in range(self.n_states)]
        )
        
        # Log transition matrix
        print(f"\nTransition Matrix:")
        print(self.transition_matrix)
        
        # Log regime properties
        print(f"\nRegime Properties:")
        for k in range(self.n_states):
            print(f"  Regime {k}:")
            print(f"    Mean Δ[L,S,C]: {self.regime_means[k]}")
            print(f"    Self-transition prob: {self.transition_matrix[k, k]:.3f}")
            duration = self._compute_regime_duration(self.transition_matrix)[k]
            print(f"    Expected duration: {duration:.1f} days")
        
        # Compute average durations
        avg_durations = self._compute_regime_duration(self.transition_matrix)
        
        # Return results dict
        results = {
            'n_states': self.n_states,
            'bic_scores': bic_scores,
            'regime_means': self.regime_means,
            'regime_covs': self.regime_covs,
            'transition_matrix': self.transition_matrix,
            'regime_probs': self.regime_probs,
            'avg_durations': avg_durations
        }
        
        return results
    
    def predict_regime_probs(self, factor_changes_recent):
        """
        Get current regime distribution with ambiguity detection
        
        Args:
            factor_changes_recent: Recent factor changes
            
        Returns:
            gamma_T: Most recent regime distribution
            ambiguity_flag: True if no regime dominates
        """
        X = factor_changes_recent.values
        
        # Get regime probabilities
        regime_probs = self.model.predict_proba(X)
        gamma_T = regime_probs[-1, :]  # Most recent
        
        # Check for ambiguity
        max_prob = gamma_T.max()
        ambiguity_flag = max_prob < self.ambiguity_threshold
        
        if ambiguity_flag:
            print(f"Warning: Regime ambiguity detected (max prob = {max_prob:.2f} < {self.ambiguity_threshold})")
            print(f"Current regime distribution: {gamma_T}")
            print("Consider regime-averaging in simulations")
        
        return gamma_T, ambiguity_flag
    
    def get_regime_timeline(self):
        """
        Extract historical regime probabilities
        
        Returns:
            regime_probs: DataFrame with regime probabilities over time
        """
        if self.regime_probs is None:
            raise ValueError("Model must be fitted first")
        
        return self.regime_probs
    
    def get_most_likely_regimes(self):
        """
        Get most likely regime at each time point
        
        Returns:
            most_likely: Series with regime labels
        """
        if self.regime_probs is None:
            raise ValueError("Model must be fitted first")
        
        most_likely = self.regime_probs.values.argmax(axis=1)
        return pd.Series(most_likely, index=self.regime_probs.index, name='Most_Likely_Regime')


def fit_regime_model(df, df_factors, afns_model, config):
    """
    Main function to fit Sticky HMM on factor changes
    
    Args:
        df: DataFrame with data (not used in new implementation)
        df_factors: DataFrame with AFNS factors
        afns_model: Fitted AFNS model
        config: Configuration dict
        
    Returns:
        hmm_model: Fitted StickyHMM
        regime_labels: Most likely regime at each time point
        regime_stats: Dict with regime statistics
        regime_covs: Dict mapping regime -> covariance matrix
    """
    # Get factor changes from AFNS model
    factor_changes = afns_model.factor_changes
    
    # Initialize and fit Sticky HMM
    hmm_model = StickyHMM(
        n_states_range=config['regime'].get('n_states_range', [2, 3, 4, 5]),
        sticky_alpha=config['regime'].get('sticky_alpha', 10.0),
        random_state=config['regime'].get('random_state', 42)
    )
    
    # Fit HMM
    hmm_results = hmm_model.fit(factor_changes)
    
    # Get most likely regimes for compatibility with existing code
    regime_labels = hmm_model.get_most_likely_regimes()  # Keep as Series with index
    
    # Extract regime covariances from HMM (these are for factor changes, not levels)
    regime_covs = {}
    for k in range(hmm_model.n_states):
        regime_covs[k] = hmm_model.regime_covs[k]
    
    # Compute regime statistics
    regime_stats = {}
    for k in range(hmm_model.n_states):
        # Count observations where this is the most likely regime
        n_obs = (regime_labels.values == k).sum()
        pct_obs = n_obs / len(regime_labels) * 100
        
        regime_stats[k] = {
            'n_obs': n_obs,
            'pct_obs': pct_obs,
            'mean_delta_L': hmm_model.regime_means[k][0],
            'mean_delta_S': hmm_model.regime_means[k][1],
            'mean_delta_C': hmm_model.regime_means[k][2],
            'self_transition_prob': hmm_model.transition_matrix[k, k],
            'expected_duration_days': hmm_results['avg_durations'][k]
        }
    
    print(f"\nRegime Statistics:")
    for k, stats in regime_stats.items():
        print(f"  Regime {k}: {stats['n_obs']} obs ({stats['pct_obs']:.1f}%), "
              f"duration={stats['expected_duration_days']:.1f} days")
    
    return hmm_model, regime_labels, regime_stats, regime_covs
