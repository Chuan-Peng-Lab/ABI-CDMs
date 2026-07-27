"""
Preprocessing utilities for behavioral trial data.

Provides functions to normalize raw behavioral trial DataFrames
into a canonical schema for downstream analysis and modeling.
"""

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "normalize_behavioral_trials",
    "DEFAULT_SCHEMA",
    "TrialSummary",
    "compute_quantile_bin_stats",
    "DEFAULT_CONFIG",
]

DEFAULT_SCHEMA = {
    "subject_col": "subject_id",
    "rt_col": "rt",
    "response_col": "accuracy",     # 1=correct, 0=error
    "congruency_col": "congruency", # 1=congruent, 0=incongruent
}

DEFAULT_CONFIG = {
    "groupby": ["subject", "condition"],
    "n_bins_caf": 5,    # number of bins for CAF (Conditional Accuracy Function)
    "n_bins_cdf": 10,   # number of bins for CDF (Cumulative Distribution Function)
    "n_bins_cdf_props": 6,  # number of bins for CDF proportions (flexDDM style)
    "n_bins_caf_props": 4,  # number of bins for CAF proportions (flexDDM style)
}


@dataclass
class TrialSummary:
    """
    Container for normalized trial data and behavioral statistics.
    
    Contains two types of statistics:
    
    **Quantile Bin Stats** (for RMSE, G-square):
    - cdf_bin_stats: RT distribution stats per bin (cutoffs from all trials)
    - caf_bin_stats: Accuracy stats per RT bin (cutoffs from all trials)
    - Proportions computed within each bin: n_correct / n_trials_in_bin
    
    **Proportion Stats** (for flexDDM-style chi-square, aBIC):
    - cdf_props: Correct trial proportions (cutoffs from correct trials only)
    - caf_props: Error trial proportions (cutoffs from all trials)
    - Proportions computed relative to total: n_correct / n_total_trials
    """
    
    normalized: pd.DataFrame           # output of normalize_behavioral_trials
    grouping: list                      # e.g. ["subject", "condition"]
    config: dict                        # config parameters used
    cdf_bin_stats: pd.DataFrame         # RT distribution per bin (for RMSE)
    caf_bin_stats: pd.DataFrame         # Accuracy per RT bin (for RMSE)
    cdf_props: pd.DataFrame = field(default_factory=pd.DataFrame)  # CDF proportions (flexDDM style)
    caf_props: pd.DataFrame = field(default_factory=pd.DataFrame)  # CAF proportions (flexDDM style)


def _validate_positive_bin_count(value, name):
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def compute_quantile_bin_stats(df, config=None) -> "TrialSummary":
    """
    Compute per-group quantile bin statistics from a normalized trial DataFrame.
    
    This function computes two sets of behavioral measures:
    
    **Quantile Bin Stats** (for RMSE, G-square):
    - CDF bin stats: mean RT per bin, cutoffs from ALL trials
    - CAF bin stats: accuracy rate per bin, cutoffs from ALL trials
    - Proportions computed within each bin
    
    **Proportion Stats** (for flexDDM-style chi-square, aBIC):
    - CDF props: proportion of correct trials per bin, cutoffs from CORRECT trials
    - CAF props: proportion of error trials per bin, cutoffs from ALL trials
    - Proportions computed relative to total trials
    
    Parameters
    ----------
    df : pd.DataFrame
        Normalized output from normalize_behavioral_trials. Must contain columns:
        subject, rt_ms, condition, is_correct, is_error.
    config : dict, optional
        Configuration dict with keys:
        - "groupby": list of columns to group by (default ["subject", "condition"])
        - "n_bins_caf": number of bins for CAF bin stats (default 5)
        - "n_bins_cdf": number of bins for CDF bin stats (default 10)
        - "n_bins_cdf_props": number of bins for CDF proportions (default 6)
        - "n_bins_caf_props": number of bins for CAF proportions (default 4)
        If None, uses DEFAULT_CONFIG.
    
    Returns
    -------
    TrialSummary
        Dataclass containing:
        - normalized: input DataFrame
        - grouping: columns used for grouping
        - config: config dict used
        - cdf_bin_stats: DataFrame with RT distribution per bin
        - caf_bin_stats: DataFrame with accuracy per RT bin
        - cdf_props: DataFrame with CDF proportions (flexDDM style)
        - caf_props: DataFrame with CAF proportions (flexDDM style)
    """
    if config is None:
        config = copy.deepcopy(DEFAULT_CONFIG)
    else:
        merged = copy.deepcopy(DEFAULT_CONFIG)
        merged.update(config)
        config = merged
    
    unknown_keys = set(config) - set(DEFAULT_CONFIG)
    if unknown_keys:
        raise ValueError(f"Unknown config keys: {unknown_keys}")
    
    groupby_cols = config["groupby"]
    
    # Bin counts
    n_bins_caf = config["n_bins_caf"]
    n_bins_cdf = config["n_bins_cdf"]
    n_bins_cdf_props = config["n_bins_cdf_props"]
    n_bins_caf_props = config["n_bins_caf_props"]
    
    _validate_positive_bin_count(n_bins_caf, "n_bins_caf")
    _validate_positive_bin_count(n_bins_cdf, "n_bins_cdf")
    _validate_positive_bin_count(n_bins_cdf_props, "n_bins_cdf_props")
    _validate_positive_bin_count(n_bins_caf_props, "n_bins_caf_props")
    
    # Generate quantiles: n_bins bins require n_bins-1 cutoffs
    caf_quantiles = [i / n_bins_caf for i in range(1, n_bins_caf)]
    cdf_quantiles = [i / n_bins_cdf for i in range(1, n_bins_cdf)]
    cdf_props_quantiles = [i / n_bins_cdf_props for i in range(1, n_bins_cdf_props)]
    caf_props_quantiles = [i / n_bins_caf_props for i in range(1, n_bins_caf_props)]
    
    # Records for each type of statistics
    caf_records = []
    cdf_records = []
    cdf_props_records = []
    caf_props_records = []
    
    for group_keys, group in df.groupby(groupby_cols):
        group = group.reset_index(drop=True)
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        
        key_dict = dict(zip(groupby_cols, group_keys))
        
        if len(group) == 0:
            continue
        
        rt_values = group["rt_ms"].values
        n_total = len(group)
        
        correct_trials = group[group["is_correct"] == 1]
        n_correct = len(correct_trials)
        
        # ====================================================================
        # CAF Bin Stats: accuracy per RT bin (cutoffs from all trials)
        # ====================================================================
        
        if len(caf_quantiles) > 0:
            caf_cutoffs = np.nanpercentile(rt_values, [q * 100 for q in caf_quantiles])
            caf_bins = np.digitize(rt_values, caf_cutoffs)
        else:
            caf_cutoffs = np.array([])
            caf_bins = np.zeros_like(rt_values, dtype=int)
        
        for bin_idx in range(n_bins_caf):
            mask = caf_bins == bin_idx
            
            n_trials = int(mask.sum())
            n_correct_bin = int(group.loc[mask, "is_correct"].sum()) if n_trials > 0 else 0
            n_error_bin = int(group.loc[mask, "is_error"].sum()) if n_trials > 0 else 0
            prop_correct = n_correct_bin / n_trials if n_trials > 0 else 0.0
            prop_error = n_error_bin / n_trials if n_trials > 0 else 0.0
            mean_rt_ms = float(group.loc[mask, "rt_ms"].mean()) if n_trials > 0 else np.nan
            
            cutoff_lower = np.nan if bin_idx == 0 else float(caf_cutoffs[bin_idx - 1])
            cutoff_upper = np.nan if bin_idx == n_bins_caf - 1 else float(caf_cutoffs[bin_idx])
            
            record = {**key_dict, "bin": bin_idx,
                      "n_trials": n_trials, "n_correct": n_correct_bin, "n_error": n_error_bin,
                      "prop_correct": prop_correct, "prop_error": prop_error,
                      "mean_rt_ms": mean_rt_ms,
                      "cutoff_lower": cutoff_lower, "cutoff_upper": cutoff_upper}
            caf_records.append(record)
        
        # ====================================================================
        # CDF Bin Stats: RT distribution (cutoffs from all trials)
        # ====================================================================
        
        if len(cdf_quantiles) > 0:
            cdf_cutoffs = np.nanpercentile(rt_values, [q * 100 for q in cdf_quantiles])
            cdf_bins = np.digitize(rt_values, cdf_cutoffs)
        else:
            cdf_cutoffs = np.array([])
            cdf_bins = np.zeros_like(rt_values, dtype=int)
        
        for bin_idx in range(n_bins_cdf):
            mask = cdf_bins == bin_idx
            
            n_trials = int(mask.sum())
            n_correct_bin = int(group.loc[mask, "is_correct"].sum()) if n_trials > 0 else 0
            n_error_bin = int(group.loc[mask, "is_error"].sum()) if n_trials > 0 else 0
            prop_correct = n_correct_bin / n_trials if n_trials > 0 else 0.0
            prop_error = n_error_bin / n_trials if n_trials > 0 else 0.0
            mean_rt_ms = float(group.loc[mask, "rt_ms"].mean()) if n_trials > 0 else np.nan
            
            cutoff_lower = np.nan if bin_idx == 0 else float(cdf_cutoffs[bin_idx - 1])
            cutoff_upper = np.nan if bin_idx == n_bins_cdf - 1 else float(cdf_cutoffs[bin_idx])
            
            record = {**key_dict, "bin": bin_idx,
                      "n_trials": n_trials, "n_correct": n_correct_bin, "n_error": n_error_bin,
                      "prop_correct": prop_correct, "prop_error": prop_error,
                      "mean_rt_ms": mean_rt_ms,
                      "cutoff_lower": cutoff_lower, "cutoff_upper": cutoff_upper}
            cdf_records.append(record)
        
        # ====================================================================
        # CDF Proportions: correct trials per bin (cutoffs from correct trials)
        # ====================================================================
        
        if len(cdf_props_quantiles) > 0 and n_correct > 0:
            cdf_props_cutoffs = np.nanpercentile(
                correct_trials["rt_ms"].values, 
                [q * 100 for q in cdf_props_quantiles]
            )
        elif len(cdf_props_quantiles) > 0:
            cdf_props_cutoffs = np.array([0.0] * len(cdf_props_quantiles))
        else:
            cdf_props_cutoffs = np.array([])
        
        for bin_idx in range(n_bins_cdf_props):
            if len(cdf_props_cutoffs) == 0:
                mask = correct_trials.index >= 0
            elif bin_idx == 0:
                mask = correct_trials["rt_ms"] <= cdf_props_cutoffs[0]
            elif bin_idx == n_bins_cdf_props - 1:
                mask = correct_trials["rt_ms"] > cdf_props_cutoffs[-1]
            else:
                mask = (correct_trials["rt_ms"] > cdf_props_cutoffs[bin_idx - 1]) & \
                       (correct_trials["rt_ms"] <= cdf_props_cutoffs[bin_idx])
            
            n_correct_in_bin = int(mask.sum())
            prop = n_correct_in_bin / n_total if n_total > 0 else 0.0
            
            record = {**key_dict, "bin": bin_idx, "prop": prop}
            cdf_props_records.append(record)
        
        # ====================================================================
        # CAF Proportions: error trials per bin (cutoffs from all trials)
        # ====================================================================
        
        if len(caf_props_quantiles) > 0 and n_total > 0:
            caf_props_cutoffs = np.nanpercentile(
                rt_values, 
                [q * 100 for q in caf_props_quantiles]
            )
        elif len(caf_props_quantiles) > 0:
            caf_props_cutoffs = np.array([0.0] * len(caf_props_quantiles))
        else:
            caf_props_cutoffs = np.array([])
        
        for bin_idx in range(n_bins_caf_props):
            if len(caf_props_cutoffs) == 0:
                mask = group.index >= 0
            elif bin_idx == 0:
                mask = group["rt_ms"] <= caf_props_cutoffs[0]
            elif bin_idx == n_bins_caf_props - 1:
                mask = group["rt_ms"] > caf_props_cutoffs[-1]
            else:
                mask = (group["rt_ms"] > caf_props_cutoffs[bin_idx - 1]) & \
                       (group["rt_ms"] <= caf_props_cutoffs[bin_idx])
            
            bin_trials = group[mask]
            n_error_in_bin = int(bin_trials["is_error"].sum()) if len(bin_trials) > 0 else 0
            prop = n_error_in_bin / n_total if n_total > 0 else 0.0
            
            record = {**key_dict, "bin": bin_idx, "prop": prop}
            caf_props_records.append(record)
    
    return TrialSummary(
        normalized=df,
        grouping=groupby_cols,
        config=config,
        cdf_bin_stats=pd.DataFrame(cdf_records),
        caf_bin_stats=pd.DataFrame(caf_records),
        cdf_props=pd.DataFrame(cdf_props_records),
        caf_props=pd.DataFrame(caf_props_records),
    )


def normalize_behavioral_trials(df, schema=None):
    """
    Convert a raw behavioral trial DataFrame into a canonical schema.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing raw behavioral trial data.
    schema : dict, optional
        Dictionary specifying input column names. Keys should be:
        - "subject_col": column name for subject identifier
        - "rt_col": column name for reaction time (expected in seconds, will be converted to ms)
        - "response_col": column name for response accuracy (1=correct, 0=error)
        - "congruency_col": column name for congruency (1=congruent, 0=incongruent)
        If None, uses DEFAULT_SCHEMA.

    Returns
    -------
    pd.DataFrame
        DataFrame with canonical schema columns:
        - subject: subject identifier
        - rt_ms: reaction time in milliseconds
        - condition: "congruent" or "incongruent"
        - is_correct: 1=correct, 0=error
        - is_error: 1=error, 0=correct

    Raises
    ------
    ValueError
        If schema is missing required keys, DataFrame is missing required columns,
        if any RT values are negative, or if congruency/response columns contain
        unexpected values.
    """
    if schema is None:
        schema = DEFAULT_SCHEMA.copy()

    # Validate schema has required keys
    required_keys = {"subject_col", "rt_col", "response_col", "congruency_col"}
    missing_keys = required_keys - schema.keys()
    if missing_keys:
        raise ValueError(f"schema is missing required keys: {missing_keys}")

    # Extract column names from schema
    subject_col = schema["subject_col"]
    rt_col = schema["rt_col"]
    response_col = schema["response_col"]
    congruency_col = schema["congruency_col"]

    # Validate that required columns exist in DataFrame
    required_cols = [subject_col, rt_col, response_col, congruency_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

    # Validate RT values are non-negative
    if (df[rt_col] < 0).any():
        raise ValueError(
            "RT values must be non-negative. Got negative values which may indicate "
            "sentinel values or wrong units."
        )

    # Validate congruency column contains only 0 or 1
    valid_congruency = df[congruency_col].isin([0, 1])
    if not valid_congruency.all():
        bad = df.loc[~valid_congruency, congruency_col].unique()
        raise ValueError(
            f"Unexpected values in '{congruency_col}': {list(bad)}. Expected 0 or 1."
        )

    # Validate response column contains only 0 or 1
    valid_response = df[response_col].isin([0, 1])
    if not valid_response.all():
        bad = df.loc[~valid_response, response_col].unique()
        raise ValueError(
            f"Unexpected values in '{response_col}': {list(bad)}. Expected 0 or 1."
        )

    # Create output DataFrame with canonical columns
    result = pd.DataFrame()

    # Copy subject column
    result["subject"] = df[subject_col]

    # Convert RT to milliseconds (input is in seconds)
    result["rt"] = df[rt_col]
    result["rt_ms"] = df[rt_col] * 1000

    # Map congruency: 1 -> "congruent", 0 -> "incongruent"
    result["condition"] = np.where(df[congruency_col] == 1, "congruent", "incongruent")

    # Map response accuracy: 1 -> is_correct=1, 0 -> is_correct=0
    result["is_correct"] = np.where(df[response_col] == 1, 1, 0)

    # is_error is inverse of is_correct
    result["is_error"] = 1 - result["is_correct"]

    return result