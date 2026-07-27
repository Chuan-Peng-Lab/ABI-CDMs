"""
Model comparison metrics evaluator.

Provides a single class that caches shared TrialSummary computations and
computes RMSE, G-square, chi-square, and aBIC without redundant recomputation.

Two types of metrics are supported:

1. **Quantile-based metrics** (RMSE, G-square, count-based aBIC):
   - Uses bin statistics from cdf_bin_stats and caf_bin_stats
   - Cutoffs from all trials, proportions within each bin

2. **Proportion-based metrics** (flexDDM-style chi-square, aBIC):
   - Uses cdf_props and caf_props
   - CDF cutoffs from correct trials only, proportions relative to total trials
"""

import numpy as np

from utils_preprocessing import (
    compute_quantile_bin_stats,
    normalize_behavioral_trials,
)

__all__ = ["ModelMetricEvaluator"]


class ModelMetricEvaluator:
    """
    Evaluator for model comparison metrics.
    
    Computes metrics from empirical and predicted behavioral data.
    All statistics are computed in a single pass using compute_quantile_bin_stats.
    
    Parameters
    ----------
    empirical_df : pd.DataFrame
        Empirical/observed data with columns: rt, accuracy, congruency, subject_id
    predicted_df : pd.DataFrame
        Model predicted data with same format
    config : dict, optional
        Config for all statistics:
        - groupby: columns to group by (default ["subject", "condition"])
        - n_bins_caf, n_bins_cdf: bins for quantile stats
        - n_bins_cdf_props, n_bins_caf_props: bins for proportion stats
    """
    
    def __init__(self, empirical_df, predicted_df, config=None):
        self.empirical_df = empirical_df
        self.predicted_df = predicted_df
        self.config = config
        
        # Cached results
        self._empirical_normalized = None
        self._predicted_normalized = None
        self._empirical_summary = None
        self._predicted_summary = None

    def prepare(self):
        """Prepare all statistics (lazy computation)."""
        if self._empirical_summary is None:
            self._empirical_normalized = normalize_behavioral_trials(self.empirical_df)
            self._predicted_normalized = normalize_behavioral_trials(self.predicted_df)
            self._empirical_summary = compute_quantile_bin_stats(
                self._empirical_normalized, config=self.config
            )
            self._predicted_summary = compute_quantile_bin_stats(
                self._predicted_normalized, config=self.config
            )
        return self

    # =========================================================================
    # RMSE Metrics (Quantile Bin Stats)
    # =========================================================================
    
    def compute_rmse(self, caf_scale_base=None):
        """
        Compute RMSE between empirical and predicted data.
        
        Uses quantile bin statistics:
        - CAF RMSE: accuracy rate per RT bin
        - CDF RMSE: mean RT per RT bin
        
        Parameters
        ----------
        caf_scale_base : int, optional
            Base multiplier for CAF component weighting.
            Defaults to self.config["caf_scale_base"] (500) to produce
            the current manuscript results.
            RMSE_total = RMSE_CDF + caf_weight * RMSE_CAF
            where caf_weight = (n_cdf_points / n_caf_points) * caf_scale_base.
        
        Returns
        -------
        dict
            - rmse: weighted combined RMSE
            - caf_rmse: RMSE for CAF (accuracy)
            - cdf_rmse: RMSE for CDF (mean RT)
            - caf_scale_base: base multiplier used
            - caf_weight: effective CAF weight
            - n_caf_points: number of CAF data points
            - n_cdf_points: number of CDF data points
        """
        self.prepare()
        
        if caf_scale_base is None:
            caf_scale_base = self.config.get("caf_scale_base", 500) if self.config else 500
        
        emp_caf = self._empirical_summary.caf_bin_stats.sort_values(
            ["condition", "bin"]
        )["prop_correct"].to_numpy(dtype=float)
        pred_caf = self._predicted_summary.caf_bin_stats.sort_values(
            ["condition", "bin"]
        )["prop_correct"].to_numpy(dtype=float)
        
        emp_cdf = self._empirical_summary.cdf_bin_stats.sort_values(
            ["condition", "bin"]
        )["mean_rt_ms"].to_numpy(dtype=float)
        pred_cdf = self._predicted_summary.cdf_bin_stats.sort_values(
            ["condition", "bin"]
        )["mean_rt_ms"].to_numpy(dtype=float)
        
        n_caf_points = len(emp_caf)
        n_cdf_points = len(emp_cdf)
        
        # Weight CAF RMSE more heavily due to different units
        caf_weight = (n_cdf_points / n_caf_points) * caf_scale_base
        
        caf_rmse = float(np.sqrt(np.mean((emp_caf - pred_caf) ** 2)))
        cdf_rmse = float(np.sqrt(np.mean((emp_cdf - pred_cdf) ** 2)))
        rmse = (caf_weight * caf_rmse) + cdf_rmse
        
        return {
            "rmse": rmse,
            "caf_rmse": caf_rmse,
            "cdf_rmse": cdf_rmse,
            "caf_scale_base": caf_scale_base,
            "caf_weight": caf_weight,
            "n_caf_points": n_caf_points,
            "n_cdf_points": n_cdf_points,
        }

    # =========================================================================
    # G-square Metrics (Quantile Bin Stats - Count-based)
    # =========================================================================
    
    def compute_g_square(self):
        """
        Compute G-square (likelihood ratio chi-square) statistic.
        
        Uses count data from quantile bin statistics.
        G^2 = 2 * sum(O_i * ln(O_i / E_i))
        
        Returns
        -------
        dict
            - g_square: G-square statistic
        """
        self.prepare()
        
        emp_cdf = self._empirical_summary.cdf_bin_stats.sort_values(["condition", "bin"])
        pred_cdf = self._predicted_summary.cdf_bin_stats.sort_values(["condition", "bin"])
        
        # Concatenate correct and error counts
        O_i = np.concatenate([
            emp_cdf["n_correct"].to_numpy(dtype=float),
            emp_cdf["n_error"].to_numpy(dtype=float),
        ])
        E_i = np.concatenate([
            pred_cdf["n_correct"].to_numpy(dtype=float),
            pred_cdf["n_error"].to_numpy(dtype=float),
        ])
        
        safe_E = np.where(E_i <= 0, 1e-12, E_i)
        safe_O = np.where(O_i <= 0, 1e-12, O_i)
        
        g_sq_terms = 2 * O_i * np.log(safe_O / safe_E)
        g_sq_terms = np.where(O_i == 0.0, 0.0, g_sq_terms)
        
        g_square = float(np.sum(g_sq_terms))
        return {"g_square": g_square}

    def compute_abic(self, n_param=None, n_trial=None):
        """
        Compute aBIC (approximate Bayesian Information Criterion).
        
        Uses G-square from quantile bin statistics.
        aBIC = G^2 + k * ln(N)
        
        Parameters
        ----------
        n_param : int
            Number of model parameters
        n_trial : int
            Number of trials
            
        Returns
        -------
        dict
            - g_square: G-square statistic
            - aBIC: approximate BIC
        """
        if n_param is None or n_trial is None:
            raise ValueError("compute_abic requires both n_param and n_trial")
        
        g_sq_result = self.compute_g_square()
        g_square = g_sq_result["g_square"]
        abic = float(g_square + (n_param * np.log(n_trial)))
        
        return {"g_square": g_square, "aBIC": abic}

    # =========================================================================
    # Proportion-based Metrics (flexDDM style)
    # =========================================================================
    
    def compute_chi_square_proportions(self, n_trial=None):
        """
        Compute chi-square (G-square) using CDF/CAF proportions.
        
        This follows the flexDDM-style calculation:
        - CDF proportions: cutoffs from correct trials only
        - CAF proportions: cutoffs from all trials
        - Proportions relative to total trials
        
        G^2 = 2 * n_trial * sum(empirical * log(empirical / predicted))
        
        Parameters
        ----------
        n_trial : int, optional
            Number of trials. If None, uses len of empirical data.
        
        Returns
        -------
        dict
            - chi_square: chi-square (G-square) statistic
        """
        self.prepare()
        
        if n_trial is None:
            n_trial = len(self.empirical_df)
        
        emp_cdf = self._empirical_summary.cdf_props.sort_values(
            ["condition", "bin"]
        )["prop"].to_numpy(dtype=float)
        pred_cdf = self._predicted_summary.cdf_props.sort_values(
            ["condition", "bin"]
        )["prop"].to_numpy(dtype=float)
        
        emp_caf = self._empirical_summary.caf_props.sort_values(
            ["condition", "bin"]
        )["prop"].to_numpy(dtype=float)
        pred_caf = self._predicted_summary.caf_props.sort_values(
            ["condition", "bin"]
        )["prop"].to_numpy(dtype=float)
        
        empirical = np.concatenate([emp_cdf, emp_caf])
        predicted = np.concatenate([pred_cdf, pred_caf])
        
        # Replace zeros with small values
        empirical_safe = np.maximum(empirical, 1e-6)
        predicted_safe = np.maximum(predicted, 1e-6)
        
        # G-square (log-likelihood ratio)
        g_sq_terms = 2 * n_trial * empirical_safe * np.log(empirical_safe / predicted_safe)
        g_sq_terms = np.where(empirical <= 0, 0.0, g_sq_terms)
        
        chi_square = float(np.sum(g_sq_terms))
        chi_square = np.nan_to_num(chi_square, nan=1e12)
        
        return {"g_square": chi_square}

    def compute_abic_proportions(self, n_param=None, n_trial=None):
        """
        Compute aBIC using proportion-based calculation (flexDDM style).
        
        BIC = -2 * sum(n_trial * empirical * log(predicted)) + k * ln(N)
        
        Parameters
        ----------
        n_param : int
            Number of model parameters
        n_trial : int
            Number of trials
            
        Returns
        -------
        dict
            - chi_square: G-square statistic
            - aBIC: approximate BIC
        """
        if n_param is None or n_trial is None:
            raise ValueError("compute_abic_proportions requires both n_param and n_trial")
        
        self.prepare()
        
        emp_cdf = self._empirical_summary.cdf_props.sort_values(["condition", "bin"])["prop"].to_numpy(dtype=float)
        pred_cdf = self._predicted_summary.cdf_props.sort_values(["condition", "bin"])["prop"].to_numpy(dtype=float)
        emp_caf = self._empirical_summary.caf_props.sort_values(["condition", "bin"])["prop"].to_numpy(dtype=float)
        pred_caf = self._predicted_summary.caf_props.sort_values(["condition", "bin"])["prop"].to_numpy(dtype=float)
        
        empirical = np.concatenate([emp_cdf, emp_caf])
        predicted = np.concatenate([pred_cdf, pred_caf])
        
        predicted_safe = np.maximum(predicted, 1e-6)
        
        # BIC calculation (flexDDM style)
        finalsum = np.sum(n_trial * empirical * np.log(predicted_safe))
        abic = float(-2 * finalsum + n_param * np.log(n_trial))
        
        # Also compute chi_square
        chi_sq_result = self.compute_chi_square_proportions(n_trial=n_trial)
        
        return {
            "g_square": chi_sq_result["g_square"],
            "aBIC": abic,
        }

    # =========================================================================
    # Unified Compute Method
    # =========================================================================
    
    def compute(self, metrics=("rmse", "g_square", "abic", "chi_square_props", "abic_props"), 
                n_param=None, n_trial=None):
        """
        Compute specified metrics.
        
        Parameters
        ----------
        metrics : tuple
            Metrics to compute. Options:
            - "rmse": RMSE using quantile bin stats
            - "g_square": G-square using quantile bin stats
            - "abic": aBIC using quantile bin stats (requires n_param, n_trial)
            - "chi_square_props": chi-square using CDF/CAF proportions
            - "abic_props": aBIC using CDF/CAF proportions (requires n_param, n_trial)
        n_param : int, optional
            Number of model parameters (required for abic, abic_props)
        n_trial : int, optional
            Number of trials (required for abic, abic_props)
            
        Returns
        -------
        dict
            Dictionary with computed metrics
        """
        results = {}
        for metric in metrics:
            if metric == "rmse":
                results["rmse"] = self.compute_rmse()
            elif metric == "g_square":
                results["g_square"] = self.compute_g_square()
            elif metric == "abic":
                results["abic"] = self.compute_abic(n_param=n_param, n_trial=n_trial)
            elif metric == "chi_square_props":
                results["chi_square_props"] = self.compute_chi_square_proportions(n_trial=n_trial)
            elif metric == "abic_props":
                results["abic_props"] = self.compute_abic_proportions(n_param=n_param, n_trial=n_trial)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        return results
    
    # =========================================================================
    # Convenience Properties
    # =========================================================================
    
    @property
    def empirical_summary(self):
        """Get empirical TrialSummary."""
        self.prepare()
        return self._empirical_summary
    
    @property
    def predicted_summary(self):
        """Get predicted TrialSummary."""
        self.prepare()
        return self._predicted_summary