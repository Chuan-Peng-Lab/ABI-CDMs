"""
Shared analysis utilities for individual-differences preprocessing.

Originally defined inline in `23individual_analysis_preprocess.py`.
Extracted here so both the original 4-model pipeline and the DMC_v2
extension can import the same functions.

Functions
---------
rename_columns_with_model   – append model name to parameter columns
concat_dfs_by_subj          – merge multiple model DataFrames by subject
calculate_indices           – RT / accuracy / delta analysis per subject
compute_model_prediction_indices – aBIC, G-square, RMSE across tasks
merge_params_and_indices    – combine fitted params with behavioural indices
get_col_names               – extract author_year & task_name from task_id
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from tqdm import tqdm
from joblib import Parallel, delayed

from utils_pydmc import Ob
from utils import timer
from model_metrics import ModelMetricEvaluator
from default_settings import PARAMS_KEY_NAME_MAPPING


# ── Column formatting ────────────────────────────────────────────────────

def rename_columns_with_model(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Rename columns by appending model name to each parameter, except subject_id.

    Example:
        >>> df = pd.DataFrame({'subject_id': [1], 'a': [0.5], 't': [0.3]})
        >>> rename_columns_with_model(df, 'DDM')
           subject_id  $a|DDM$  $t|DDM$
        0         1      0.5      0.3
    """
    tmp_df = df.copy()
    cols_to_rename = [col for col in tmp_df.columns if col != 'subject_id']
    rename_dict = {
        col: f'${PARAMS_KEY_NAME_MAPPING[model_name][col].strip("$")}|{model_name}$'
        for col in cols_to_rename
    }
    return tmp_df.rename(columns=rename_dict)


def concat_dfs_by_subj(df_dict: dict) -> pd.DataFrame:
    """
    Concatenate multiple model DataFrames by subject ID.

    Example:
        >>> df_dict = {
        ...     'DDM': pd.DataFrame({'subject_id': [1], 'a': [0.5]}),
        ...     'DMC': pd.DataFrame({'subject_id': [1], 'v': [0.3]})
        ... }
        >>> concat_dfs_by_subj(df_dict)
           subject_id  $a|DDM$  $v|DMC$
        0         1      0.5      0.3
    """
    return pd.concat(
        [rename_columns_with_model(df, model_name)
         for model_name, df in df_dict.items()],
        axis=0
    ).groupby('subject_id', as_index=False).first()


# ── Behavioural indices ──────────────────────────────────────────────────

def calculate_indices(df: pd.DataFrame) -> dict:
    """
    Perform RT and accuracy cost analysis for each subject.

    Parameters
    ----------
    df : DataFrame with columns ['subject_id', 'accuracy', 'rt', 'congruency']

    Returns
    -------
    dict with keys:
        trial_data, subject_indices, subject_caf,
        subject_summary, subject_delta
    """
    results = []
    df['subject_id'] = df['subject_id'].factorize()[0]
    subjects = df['subject_id'].unique()

    ob_res = Ob(df)
    ob_res_caf = ob_res.caf_subject
    ob_res_delta = ob_res.delta_subject
    ob_res_summary = ob_res.summary_subject

    for subject_id in subjects:
        subj_data = df[df['subject_id'] == subject_id].copy()
        subj_data_delta = ob_res_delta[ob_res_delta['Subject'] == subject_id].copy()

        # congruency: 1=congruent, 0=incongruent
        # Effect coding: congruent=1, incongruent=-1
        subj_data['congruency_effect'] = subj_data['congruency'].map({1: 1, 0: -1})

        # 1. RT analysis (rt ~ congruency)
        X_rt = subj_data[['congruency_effect']].values
        y_rt = subj_data['rt'].values
        rt_model = LinearRegression(fit_intercept=True)
        rt_model.fit(X_rt, y_rt)
        rt_intercept = rt_model.intercept_   # Mean RT
        rt_beta = rt_model.coef_[0]           # RT cost (half)
        rt_cost = rt_beta * -2                # Full RT cost

        # 2. Accuracy analysis (accuracy ~ congruency)
        X_acc = subj_data[['congruency_effect']].values
        y_acc = subj_data['accuracy'].values
        try:
            acc_model = LogisticRegression(fit_intercept=True, max_iter=1000)
            acc_model.fit(X_acc, y_acc)
            acc_intercept = acc_model.intercept_[0]
            acc_beta = acc_model.coef_[0][0]
            avg_accuracy_prob = 1 / (1 + np.exp(-acc_intercept))
            prob_congruent = 1 / (1 + np.exp(-(acc_intercept + acc_beta)))
            prob_incongruent = 1 / (1 + np.exp(-(acc_intercept - acc_beta)))
            error_cost = (prob_congruent - prob_incongruent)
        except Exception:
            acc_model_linear = LinearRegression(fit_intercept=True)
            acc_model_linear.fit(X_acc, y_acc)
            acc_intercept = acc_model_linear.intercept_
            acc_beta = acc_model_linear.coef_[0]
            avg_accuracy_prob = acc_intercept
            error_cost = acc_beta * 2

        # 3. Delta RT slope (delta ~ mean_bin)
        X_delta = subj_data_delta[['mean_effect']].values
        y_delta = subj_data_delta['mean_bin'].values
        delta_model = LinearRegression(fit_intercept=True)
        delta_model.fit(X_delta, y_delta)
        delta_slope = delta_model.coef_[0]

        results.append({
            "subject_id": int(subject_id),
            "rt_avg": float(rt_intercept) * 1000,
            "rt_cost": float(rt_cost) * 1000,
            "acc_avg": float(avg_accuracy_prob),
            "error_cost": float(error_cost),
            "delta_slope": delta_slope
        })

    return {
        "trial_data": df,
        "subject_indices": pd.DataFrame(results).sort_values("subject_id"),
        "subject_caf": ob_res_caf,
        "subject_summary": ob_res_summary,
        "subject_delta": ob_res_delta
    }


# ── Model prediction indices ─────────────────────────────────────────────

@timer
def compute_model_prediction_indices(
    indices_dict: dict,
    map_nparams_of_models: dict,
    parallel: bool = True,
    n_jobs: int = -1,
    show_progress: bool = True,
    preprocess_config: dict = None,
    caf_scale_base: int = None,
) -> pd.DataFrame:
    """
    Compute model prediction indices (aBIC, G-square, RMSE) for all
    tasks, models, and subjects.

    Parameters
    ----------
    indices_dict : dict
        {task_id: {"obs": {...}, "ppd": {model: {...}}}}
    map_nparams_of_models : dict
        {model_name: n_params}
    caf_scale_base : int, optional
        Base multiplier for CAF component weighting in RMSE.
        Passed through to ModelMetricEvaluator.compute_rmse().
        Default (None) uses the evaluator's internal default of 500.
    """
    if preprocess_config is None:
        preprocess_config = {
            "n_bins_caf": 5,
            "n_bins_cdf": 10,
            "n_bins_caf_props": 5,
            "n_bins_cdf_props": 10
        }

    def _process_single_subject(task_id, model_name, sub_id,
                                 obs_sub_df, pp_sub_df, n_param,
                                 _caf_scale_base=None):
        try:
            n_trial = obs_sub_df.shape[0]
            evaluator = ModelMetricEvaluator(
                empirical_df=obs_sub_df,
                predicted_df=pp_sub_df,
                config=preprocess_config,
            )
            rmse_res = evaluator.compute_rmse(caf_scale_base=_caf_scale_base)
            abic_res = evaluator.compute_abic_proportions(
                n_param=n_param, n_trial=n_trial
            )
            return {
                "task_id": task_id,
                "model": model_name,
                "subject_id": sub_id,
                "aBIC": abic_res.get("aBIC", np.nan),
                "g_square": abic_res.get("g_square", np.nan),
                "RMSE": rmse_res.get("rmse", np.nan),
                "caf_rmse": rmse_res.get("caf_rmse", np.nan),
                "cdf_rmse": rmse_res.get("cdf_rmse", np.nan),
                "caf_scale_base": rmse_res.get("caf_scale_base", np.nan),
                "caf_weight": rmse_res.get("caf_weight", np.nan),
                "n_caf_points": rmse_res.get("n_caf_points", np.nan),
                "n_cdf_points": rmse_res.get("n_cdf_points", np.nan),
            }
        except Exception as e:
            print(f"  [WARN] Task {task_id}, Model {model_name}, "
                  f"Sub {sub_id}: {e}")
            return None

    all_results = []
    task_iterator = (tqdm(indices_dict.items(), desc="Tasks")
                     if show_progress else indices_dict.items())

    for task_id, task_indices in task_iterator:
        obs_data_full = task_indices["obs"]["trial_data"]
        obs_grouped = dict(list(obs_data_full.groupby("subject_id")))

        for model_name, n_param in map_nparams_of_models.items():
            if model_name not in task_indices["ppd"]:
                continue

            pp_data_full = task_indices["ppd"][model_name]["trial_data"]
            pp_grouped = dict(list(pp_data_full.groupby("subject_id")))

            common_subjects = (set(obs_grouped.keys())
                               & set(pp_grouped.keys()))
            parallel_tasks = [
                (task_id, model_name, sub_id,
                 obs_grouped[sub_id], pp_grouped[sub_id], n_param,
                 caf_scale_base)
                for sub_id in common_subjects
            ]

            if parallel:
                batch_results = Parallel(n_jobs=n_jobs)(
                    delayed(_process_single_subject)(*args)
                    for args in parallel_tasks
                )
            else:
                batch_results = [_process_single_subject(*args)
                                 for args in parallel_tasks]

            all_results.extend([r for r in batch_results if r is not None])

    final_df = pd.DataFrame(all_results)
    if not final_df.empty:
        cols_order = ["task_id", "model", "subject_id",
                       "aBIC", "g_square", "RMSE"]
        existing_cols = [c for c in cols_order if c in final_df.columns]
        remaining = [c for c in final_df.columns if c not in cols_order]
        final_df = final_df[existing_cols + remaining]

    return final_df


# ── Merging helpers ──────────────────────────────────────────────────────

def merge_params_and_indices(df_dict: dict, indices_dict: dict) -> pd.DataFrame:
    """Merge fitted parameters with behavioural indices per task."""
    tmp_list = []
    for task_id_i, df_dict_i in df_dict.items():
        tmp_merge_i = pd.merge(
            df_dict_i,
            indices_dict[task_id_i]["obs"]["subject_indices"],
            on="subject_id"
        )
        tmp_merge_i["task_id"] = task_id_i
        tmp_list.append(tmp_merge_i)

    return pd.concat(tmp_list, axis=0, ignore_index=True)


def get_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """Extract author_year and task_name from task_id column."""
    tmp_df = df.copy()
    tmp_df[['author_year', 'task_name']] = (
        tmp_df['task_id'].str.extract(r'([a-z]+[0-9]{4})([a-z]+)')
    )
    return tmp_df
