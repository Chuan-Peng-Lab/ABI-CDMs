#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import copy
from typing import List, Sequence

from nsbi_module.project_paths import DATA_DIR, INTERMEDIATE_DIR, ensure_output_directories




def processing_data(df, task_id, retest=False):
    """
    Preprocess raw behavioral data for use with CDMs_NSBI.

    Required input columns: 'subject', 'acc', 'rt', 'congruency'
    Optional (if retest=True): 'session'

    Output columns:
        - 'subject_id' (int)
        - 'accuracy' (binary: 0 or 1)
        - 'rt' (float, in [0, 999])
        - 'congruency' (binary: 1 for 'congruent', 0 for 'incongruent')
        - 'session' (if retest=True; no transformation applied)

    Parameters
    ----------
    df : pandas.DataFrame
        Raw trial-level data.
    retest : bool, default False
        Whether to include the 'session' column.

    Returns
    -------
    pandas.DataFrame
        Cleaned and standardized DataFrame.
    """
    import pandas as pd
    import numpy as np

    df_tmp = df.copy()

    # --- Column existence checks ---
    required_cols = ["subject", "acc", "rt", "congruency"]
    if retest:
        required_cols.append("session")

    missing_cols = [col for col in required_cols if col not in df_tmp.columns]
    if missing_cols:
        raise ValueError(f"{task_id}: Missing required columns: {missing_cols}")

    # --- Check and validate 'congruency' values ---
    allowed_congruency_vals = {"congruent", "incongruent"}
    unique_vals = set(df_tmp["congruency"].dropna().unique())
    invalid_vals = unique_vals - allowed_congruency_vals
    if invalid_vals:
        raise ValueError(f"{task_id}: Invalid 'congruency' values found: {sorted(invalid_vals)}. "
                         f"Allowed: {sorted(allowed_congruency_vals)}")

    # Map to binary: congruent → 1, incongruent → 0
    df_tmp["congruency"] = df_tmp["congruency"].map({"congruent": 1, "incongruent": 0})
    if df_tmp["congruency"].isnull().any():
        raise ValueError("{task_id}: Encountered unmappable 'congruency' values after mapping.")

    # --- Check and validate 'acc' (accuracy) ---
    # Accept int/float 0/1
    acc = df_tmp["acc"].astype(int)
    if not acc.isin([0, 1]).all():
        invalid_acc = sorted(acc[~acc.isin([0, 1])].unique())
        raise ValueError(f"{task_id}: Invalid 'acc' values: {invalid_acc}. Only 0 and 1 are allowed.")
    df_tmp["accuracy"] = acc

    # --- Check and validate 'rt' (reaction time) ---
    rt = df_tmp["rt"]
    # Enforce range: 0 < rt <= 999 (common safe bound; adjust if needed)
    # Note: rt == 0 is suspicious (instant response); typically excluded
    if not ((rt > 0) & (rt <= 999)).all():
        invalid_rt = rt[(rt <= 0) | (rt > 999)].unique()
        raise ValueError(f"{task_id}: 'rt' out of valid range (0, 999]: found values {sorted(invalid_rt)}")

    # --- Check and validate 'subject' → convert to int 'subject_id' ---
    df_tmp["subject_id"] = df['subject'].astype(int)

    # --- Select and order output columns ---
    selected_cols = ["subject_id", "accuracy", "rt", "congruency"]
    if retest:
        # No transformation applied to 'session'; assume user ensures integrity
        if not pd.api.types.is_numeric_dtype(df_tmp["session"]) and not pd.api.types.is_string_dtype(df_tmp["session"]):
            raise TypeError(f"{task_id}: 'session' should be numeric or string-like.")
        selected_cols.append("session")

    return df_tmp[selected_cols].reset_index(drop=True)

def filter_data_by_quality(
    data_dict,
    acc_threshold=0.6,
    dropped_ratio_threshold=0.10,
    rt_range=(0.15, 3.0)
):
    """
    Filters datasets based on Reaction Time (RT) range and Subject Performance.
    Returns a NEW dictionary (does not modify the original).

    Args:
        data_dict (dict): Dictionary {task_name: dataframe} containing data.
        acc_threshold (float): Minimum accuracy required. Default 0.6.
        dropped_ratio_threshold (float): Max allowed ratio of dropped trials per subject. Default 0.10 (10%).
        rt_range (tuple): (min_rt, max_rt). Default (0.15, 3.0).

    Returns:
        dict: A deep copy of the dictionary containing filtered dataframes.
    """
    # Create a deep copy to ensure the original dictionary is not modified
    cleaned_dict = copy.deepcopy(data_dict)
    rt_min, rt_max = rt_range

    for name, df in cleaned_dict.items():
        # --- 1. RT Filtering (First Pass) ---
        # Keep trials strictly within the defined RT range
        rt_filtered_df = df.query(f'{rt_min} < rt < {rt_max}').copy()

        # --- 2. Subject Exclusion Logic ---

        # Calculate stats per subject
        subj_total = df.groupby('subject_id').size()          # Total trials (before RT filter)
        subj_kept = rt_filtered_df.groupby('subject_id').size() # Kept trials (after RT filter)
        subj_acc = rt_filtered_df.groupby('subject_id')['accuracy'].mean() # Accuracy on kept trials

        # Create evaluation DataFrame
        subj_eval = pd.DataFrame({'total': subj_total, 'kept': subj_kept, 'acc': subj_acc})

        # Fill NaNs for subjects who lost ALL trials (kept=0, acc=NaN -> 0)
        subj_eval[['kept', 'acc']] = subj_eval[['kept', 'acc']].fillna(0)

        # Calculate Dropped Ratio: (Total - Kept) / Total
        subj_eval['dropped_ratio'] = (subj_eval['total'] - subj_eval['kept']) / subj_eval['total']

        # Identify bad subjects: Low Accuracy OR High Dropped Ratio
        bad_subj_mask = (subj_eval['acc'] < acc_threshold) | (subj_eval['dropped_ratio'] > dropped_ratio_threshold)
        excluded_subjects = subj_eval[bad_subj_mask].index.tolist()

        # --- 3. Final Filtering ---
        # Remove excluded subjects
        final_df = rt_filtered_df[~rt_filtered_df['subject_id'].isin(excluded_subjects)].copy()

        # Update the dictionary
        cleaned_dict[name] = final_df

    return cleaned_dict

def find_subjects_in_multiple_tasks(df, subject_col = "subject", task_col = "task", n_tasks = None):
    """
    Find subjects that appear in multiple tasks.

    This function groups the data by subject and counts the number of unique tasks
    each subject appears in, then returns a list of subjects that appear in more
    than one task.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least 'subject' and 'task' columns

    Returns
    -------
    list
        List of subject identifiers that appear in multiple tasks

    Example
    -------
    >>> df = pd.DataFrame({
    ...     'subject': [1, 1, 2, 2, 3],
    ...     'task': ['flanker', 'stroop', 'flanker', 'flanker', 'simon']
    ... })
    >>> find_subjects_in_multiple_tasks(df)
    [1, 2]
    """
    if n_tasks is None:
        n_tasks = df[task_col].nunique()
    subjects_in_multiple_tasks = (
        df.groupby(subject_col)[task_col]
        .nunique()
        .loc[lambda x: x >= n_tasks]
        .index.tolist()
    )
    return subjects_in_multiple_tasks

def filter_common_subject_ids(df_sequence: Sequence[pd.DataFrame], subject_col = "subject_id") -> List[pd.DataFrame]:
    """
    Filters a sequence of DataFrames to retain only rows with subject_id values
    common to all DataFrames in the sequence.

    Parameters
    ----------
    df_sequence : Sequence[pd.DataFrame]
        A list or tuple of pandas DataFrames. Each must contain a 'subject_id' column.

    Returns
    -------
    List[pd.DataFrame]
        A list of filtered DataFrames (copies), preserving the input order.
    """
    # 1. Validation
    if not df_sequence:
        raise ValueError("Input sequence must not be empty.")

    # Check if subject_col exists in all DataFrames (lazy check)
    if any(subject_col not in df.columns for df in df_sequence):
        raise KeyError(f"One or more DataFrames are missing the {subject_col} column.")

    # 2. Identify common IDs using set intersection
    # We create a generator of sets and unpack it (*) into set.intersection
    common_ids = set.intersection(*(set(df[subject_col]) for df in df_sequence))

    # 3. Filter DataFrames using list comprehension
    return [df[df[subject_col].isin(common_ids)].copy() for df in df_sequence]

def calculate_statistics_summary(original_dict, processed_dict):
    """
    Generates a summary DataFrame by comparing the original raw data
    with the processed (cleaned) data.

    Args:
        original_dict (dict): The original raw data (reference for initial counts).
        processed_dict (dict): The cleaned data (after all filter steps).

    Returns:
        pd.DataFrame: Summary statistics.
    """
    stats = []

    for name, final_df in processed_dict.items():
        # Get the original dataframe for comparison
        if name not in original_dict:
            continue

        original_df = original_dict[name]

        # --- 1. Basic Counts ---
        total_trials_initial = len(original_df)
        kept_trials = len(final_df)
        dropped_trials = total_trials_initial - kept_trials

        # Global dropped ratio
        dropped_ratio = dropped_trials / total_trials_initial if total_trials_initial > 0 else 0

        # Subject counts
        # We calculate excluded subjects by comparing unique IDs in original vs final
        original_subjs = set(original_df['subject_id'].unique())
        final_subjs = set(final_df['subject_id'].unique())
        n_subj = len(final_subjs)
        n_excluded_subj = len(original_subjs) - n_subj

        # --- 2. Descriptive Statistics (on Final Data) ---
        if not final_df.empty:
            mean_trials = final_df.groupby('subject_id').size().mean()
            mean_accuracy = final_df['accuracy'].mean()

            # RT Stats
            quantiles = final_df['rt'].quantile([0.25, 0.50, 0.75])
            max_rt_val = final_df['rt'].max()
            rt_25, rt_50, rt_75 = quantiles[0.25], quantiles[0.50], quantiles[0.75]
        else:
            mean_trials = 0
            mean_accuracy = 0
            rt_25 = rt_50 = rt_75 = max_rt_val = 0

        stats.append({
            'task_id': name,
            'n_subj': n_subj,
            'n_excluded_subj': n_excluded_subj,
            'mean_trials_per_subj': round(mean_trials),
            'total_trials': kept_trials,
            'total_trials_initial': total_trials_initial,
            'dropped_trials': dropped_trials,
            'dropped_perc': round(dropped_ratio, 3) * 100,
            'mean_accuracy': round(mean_accuracy, 3),
            'rt_25%': round(rt_25, 3),
            'rt_median': round(rt_50, 3),
            'rt_75%': round(rt_75, 3),
            'rt_max': round(max_rt_val, 3)
        })

    # --- 3. Format Output DataFrame ---
    stats_df = pd.DataFrame(stats)

    # Extract metadata using regex (Author + Year + Taskname)
    try:
        stats_df[['author_year', 'task_name']] = stats_df['task_id'].str.extract(r'([a-z]+[0-9]{4})([a-z]+)')
    except:
        stats_df['author_year'] = np.nan
        stats_df['task_name'] = np.nan

    # Reorder columns
    cols_order = [
        'task_id', 'author_year', 'task_name',
        'n_subj', 'n_excluded_subj',
        'mean_trials_per_subj',
        'total_trials', 'total_trials_initial', 'dropped_trials', 'dropped_perc',
        'mean_accuracy',
        'rt_25%', 'rt_median', 'rt_75%', 'rt_max'
    ]

    # Select columns that exist
    stats_df = stats_df[[c for c in cols_order if c in stats_df.columns]]

    # Sort
    if 'author_year' in stats_df.columns:
        stats_df = stats_df.sort_values(by=['author_year'])

    return stats_df




df_ulrich2015 = pd.read_csv(DATA_DIR / "ulrich2015.csv", index_col=0)
df_whitehead2019 = pd.read_csv(DATA_DIR / "whitehead2019.csv")
df_hedge2018 = pd.read_csv(DATA_DIR / "hedge2018.csv")
df_kucina2023 = pd.read_csv(DATA_DIR / "kucina2023.csv")
df_eisenberg2019 = pd.read_csv(DATA_DIR / "eisenberg2019.csv", index_col=0)
df_lee2025 = pd.read_csv(DATA_DIR / "lee2025.csv", index_col=0)
# Generate session_rank for each subject based on session order
df_lee2025.rename(columns={'session': 'session_id'}, inplace=True)
df_lee2025['session'] = df_lee2025.groupby(['task','subject'])['session_id'].rank(method='dense')
df_lee2025.sort_values(by=['task', 'subject', 'session'], inplace=True)
df_reymermet2018 = pd.read_csv(DATA_DIR / "reymermet2018.csv").query("congruency in ['congruent', 'incongruent']")
df_clayson2024 = pd.read_csv(DATA_DIR / "clayson2024.csv")
df_clayson2025 = pd.read_csv(DATA_DIR / "clayson2025.csv")

# Author-Year DataFrame dictionary
author_year_df_dict = {
    'ulrich2015': df_ulrich2015,
    'whitehead2019': df_whitehead2019,
    'hedge2018': df_hedge2018.query("session == 1"),
    'kucina2023': df_kucina2023,
    'eisenberg2019': df_eisenberg2019.query("session == 1"),
    'lee2025': df_lee2025.query("session == 1"),
    'reymermet2018': df_reymermet2018,
    'clayson2024': df_clayson2024,
    'clayson2025': df_clayson2025
}

# Author-Year-Task DataFrame dictionary
author_year_task_df_dict = {
    'ulrich2015flanker': author_year_df_dict["ulrich2015"].query("task=='flanker'"),
    'hedge2018flanker': author_year_df_dict["hedge2018"].query("task=='flanker'"),
    'reymermet2018flanker': author_year_df_dict["reymermet2018"].query("task=='arrowflanker'"),
    'whitehead2019flanker': author_year_df_dict["whitehead2019"].query("task=='flanker'"),
    'eisenberg2019flanker': author_year_df_dict["eisenberg2019"].query("task=='flanker'"),
    'kucina2023flanker': author_year_df_dict["kucina2023"].query("task=='flanker'"),
    'clayson2025flanker': df_clayson2025.query("task=='flanker'"),
    'lee2025flanker': author_year_df_dict["lee2025"].query("task=='flanker'"),
    'ulrich2015simon': author_year_df_dict["ulrich2015"].query("task=='simon'"),
    'hedge2018simon': author_year_df_dict["hedge2018"].query("task=='simon'"),
    'reymermet2018simon': author_year_df_dict["reymermet2018"].query("task=='simon'"),
    'whitehead2019simon': author_year_df_dict["whitehead2019"].query("task=='simon'"),
    'eisenberg2019simon': author_year_df_dict["eisenberg2019"].query("task=='simon'"),
    'kucina2023simon': author_year_df_dict["kucina2023"].query("task=='simon'"),
    'hedge2018stroop': author_year_df_dict["hedge2018"].query("task=='stroop'"),
    'reymermet2018stroop': author_year_df_dict["reymermet2018"].query("task=='colorstroop'"),
    'whitehead2019stroop': author_year_df_dict["whitehead2019"].query("task=='stroop'"),
    'eisenberg2019stroop': author_year_df_dict["eisenberg2019"].query("task=='stroop'"),
    'kucina2023stroop': author_year_df_dict["kucina2023"].query("task=='stroop'"),
    # 'kucina2023stroopon': author_year_df_dict["kucina2023"].query("task=='stroopon'"),
    'clayson2025stroop': df_clayson2025.query("task=='stroop'"),
    'lee2025stroop': author_year_df_dict["lee2025"].query("task=='stroop'"),
}




author_year_task_df_dict = {key:processing_data(df, task_id=key) for key,df in author_year_task_df_dict.items()}




cleaned_data = filter_data_by_quality(
    author_year_task_df_dict,
    acc_threshold=0.6,
    dropped_ratio_threshold=0.10
)




# Define the groups of keys you want to process together
task_groups = [
    ["clayson2025flanker", "clayson2025stroop"],
    ["eisenberg2019stroop", "eisenberg2019flanker", "eisenberg2019simon"],
    ["reymermet2018stroop", "reymermet2018simon", "reymermet2018flanker"],
    ["lee2025flanker", "lee2025stroop"]
]

# Iterate through each group to filter and update the dictionary
for keys in task_groups:
    # 1. Extract the list of DataFrames corresponding to the current keys
    dfs_to_process = [cleaned_data[k] for k in keys]

    # 2. Apply the filtering function
    filtered_dfs = filter_common_subject_ids(dfs_to_process)

    # 3. Update the dictionary effectively using zip (avoids manual indexing like result[0])
    for key, df in zip(keys, filtered_dfs):
        cleaned_data[key] = df




stats_df = calculate_statistics_summary(
    original_dict=author_year_task_df_dict,
    processed_dict=cleaned_data
)
stats_df




total_subjects = stats_df.groupby("author_year")[["n_subj"]].mean().reset_index()
total_subjects.loc[total_subjects["author_year"] == "hedge2018", "n_subj"] = 155
total_subjects.loc[total_subjects["author_year"] == "kucina2023", "n_subj"] = 120
print("The total number of subjects:", total_subjects["n_subj"].sum())
print("The total number of trials:", stats_df["total_trials"].sum())
total_subjects




stats_df.groupby("task_name")[["n_subj","total_trials"]].sum()




ensure_output_directories()
with pd.HDFStore(INTERMEDIATE_DIR / "datasets_cross_sectional.h5") as store:
    for key, df in cleaned_data.items():
        store.put(key, df)


# ## For retest datasets
#



df_eisenberg2019_filtered = (
    df_eisenberg2019
    .groupby('subject')
    .filter(lambda x: x['session'].nunique() > 1)
)

df_clayson2024_filtered = (
    df_clayson2024.query("task=='flanker'")
    # consider different types of flanker as different sessions
    .assign(
        session=lambda x: x["task_subname"].map(
            {"ffa": 1, "ffb": 2, "ffc": 3, "flk": 4}
        )
    )
    .groupby("subject")
    .filter(lambda x: x["session"].nunique() > 1)
)

author_year_task_df_dict = {
    'hedge2018flanker': df_hedge2018.query("task=='flanker'"),
    'hedge2018stroop': df_hedge2018.query("task=='stroop'"),
    'lee2025flanker': df_lee2025.query("task=='flanker'"),
    'lee2025stroop': df_lee2025.query("task=='stroop'"),
    'eisenberg2019flanker': df_eisenberg2019_filtered.query("task == 'flanker'"),
    'eisenberg2019stroop': df_eisenberg2019_filtered.query("task == 'stroop'"),
    'eisenberg2019simon': df_eisenberg2019_filtered.query("task == 'simon'"),
    'clayson2024flanker': df_clayson2024_filtered
}

tmp_author_year_task_df_dict = {key:processing_data(df, task_id=key, retest=True) for key,df in author_year_task_df_dict.items()}
task_groups = [
    ["lee2025flanker", "lee2025stroop"]
]
for keys in task_groups:
    dfs_to_process = [tmp_author_year_task_df_dict[k] for k in keys]
    filtered_dfs = filter_common_subject_ids(dfs_to_process)
    for key, df in zip(keys, filtered_dfs):
        tmp_author_year_task_df_dict[key] = df




stats = []
for name, df in tmp_author_year_task_df_dict.items():
    n_subj = df['subject_id'].nunique()

    trials_per_subj = df.groupby('subject_id').size()
    mean_trials = trials_per_subj.mean()

    sessions_per_subj = df.groupby(['subject_id'])['session'].nunique()
    mean_sessions = sessions_per_subj.mean()

    total_trials = len(df)
    filtered_df = df.query('0.15 < rt < 3')
    # Update the dictionary with the filtered DataFrame
    tmp_author_year_task_df_dict[name] = filtered_df

    kept_trials = len(filtered_df)
    dropped_trials = total_trials - kept_trials
    dropped_ratio = dropped_trials / total_trials if total_trials > 0 else 0

    stats.append({
        'task': name,
        'n_subj': n_subj,
        'mean_trials_per_subj': round(mean_trials),
        'mean_sessions_per_subj': round(mean_sessions),
        'mean_trials_per_session': round(mean_trials/mean_sessions),
        'total_trials': total_trials,
        'dropped_trials': dropped_trials,
        'dropped_perc': round(dropped_ratio,3)*100
    })
stats_df = pd.DataFrame(stats)
stats_df




tmp_author_year_task_df_dict["meta_data"] = stats_df

with pd.HDFStore(INTERMEDIATE_DIR / "datasets_retest.h5") as store:
    for key, df in tmp_author_year_task_df_dict.items():
        store.put(key, df)






