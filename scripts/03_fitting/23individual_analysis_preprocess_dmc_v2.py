#!/usr/bin/env python
# coding: utf-8

# # DMC_v2 (driftdm_dmc) — Individual Analysis Preprocessing
#
# This script:
# 1. Registers the driftdm_dmc simulator (from `34_ulrich2015_driftdm_dmc.py`)
# 2. Reads DMC_v2 fitted parameters & posterior predictive data from
#    `22fitting_and_prediction.h5`
# 3. Computes subject-level behavioural indices (calculate_indices)
# 4. Computes model prediction indices (aBIC, g_square, RMSE)
# 5. Merges the new DMC_v2 results with the **existing** 4-model CSV files
# 6. Saves two new CSV files:
#    - `23subj_indices_across_models_and_tasks_dmc_v2.csv`
#    - `23model_prediction_indices_dmc_v2.csv`
#
# Existing source files are NOT modified.

# ## 0. Imports

# %%
import pandas as pd
import sys
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

from nsbi_module.utils import FitStore
from nsbi_module.default_settings import PARAMS_KEY_NAME_MAPPING

# ── Shared analysis utilities ──
from nsbi_module.analysis_utils import (
    concat_dfs_by_subj,
    calculate_indices, compute_model_prediction_indices,
    get_col_names,
)

# ## 1. Register driftdm_dmc model (via shared loader)
#
# Calls dmc_v2_loader which handles registration + dmc_v2 alias.
# This must run before PARAMS_KEY_NAME_MAPPING is used downstream.

# %%
from nsbi_module.dmc_v2_loader import (
    get_dmc_v2_model, 
    STORE_KEY_PREFIX, 
    MODEL_REG_NAME, 
    DRIFTDMC_DMC_CONFIG  # <-- ADD THIS IMPORT
)
get_dmc_v2_model   # just import — actual registration happens lazily
_ = get_dmc_v2_model()   # force registration now

param_keys = DRIFTDMC_DMC_CONFIG["param_keys"]
PARAMS_KEY_NAME_MAPPING[STORE_KEY_PREFIX] = {k: k for k in param_keys}

print("[OK] driftdm_dmc registered. dmc_v2 alias added to PARAMS_KEY_NAME_MAPPING.")
print(f"     dmc_v2 params: {list(PARAMS_KEY_NAME_MAPPING.get(STORE_KEY_PREFIX, {}).keys())}")


# ## 3. Load DMC_v2 fits and datasets

# %%
STORE_KEY_PREFIX = "dmc_v2"
STORE_FITS_PATH = "22fitting_and_prediction.h5"
STORE_DATASETS_PATH = "21preprocessed_datasets.h5"

# ── Dataset filter (prefix match; None = all datasets with cached fits) ──
# TARGET_FILTER = ["hedge2018", "reymermet2018"]
TARGET_FILTER = None   # uncomment to process all datasets

# ── Load DMC_v2 fitted parameters & PPD ──
print("[..] Loading DMC_v2 fits from", STORE_FITS_PATH)
store_fits = FitStore(STORE_FITS_PATH)

# Discover task names from the HDF5 key list
task_names = set()
for _, row in store_fits.key_list_df.iterrows():
    key = row["key"]
    if STORE_KEY_PREFIX in key:
        # Extract task name: "clayson2025flanker_dmc_v2_fitted" → "clayson2025flanker"
        task_name = key.split(f"_{STORE_KEY_PREFIX}")[0]
        task_names.add(task_name)

task_names = sorted(task_names)
print(f"  Found {len(task_names)} tasks: {task_names[:3]}...")

# ── Apply dataset filter ──
if TARGET_FILTER is not None:
    task_names = [
        t for t in task_names
        if any(t.startswith(f) for f in TARGET_FILTER)
    ]
    print(f"  Filtered to {len(task_names)} task(s): {task_names}")

# ── Load observational datasets ──
print("[..] Loading datasets from", STORE_DATASETS_PATH)
df_obs_dict = {}
with pd.HDFStore(STORE_DATASETS_PATH) as store_ds:
    for key in store_ds.keys():
        key = key.lstrip("/")
        if "meta_data" in key:
            continue
        df_obs_dict[key] = store_ds[key]

# ## 4. Compute DMC_v2 behavioural indices

# %%
print("\n" + "=" * 60)
print("  COMPUTING DMC_v2 BEHAVIOURAL INDICES")
print("=" * 60)

indices_dict_dmc_v2 = {}
fitted_df_dict_dmc_v2 = {}

for task_name in tqdm(task_names, desc="Behavioural indices"):
    try:
        key_fitted = f"{task_name}_{STORE_KEY_PREFIX}_fitted"
        key_predicted = f"{task_name}_{STORE_KEY_PREFIX}_predicted"

        # --- 1. Validation Checks ---
        if not store_fits.isexist(key_fitted) or not store_fits.isexist(key_predicted):
            print(f"  [SKIP] {task_name}: Keys missing in HDF5.")
            continue

        fitted_dict = store_fits[key_fitted]
        ppd_dict = store_fits[key_predicted]

        if not fitted_dict or not ppd_dict:
            print(f"  [SKIP] {task_name}: Nodes are empty (likely interrupted fit).")
            continue

        # --- 2. Robust Data Extraction ---
        fit_key = STORE_KEY_PREFIX if STORE_KEY_PREFIX in fitted_dict else list(fitted_dict.keys())[0]
        ppd_key = STORE_KEY_PREFIX if STORE_KEY_PREFIX in ppd_dict else list(ppd_dict.keys())[0]

        fitted_params = fitted_dict[fit_key]
        pp_data = ppd_dict[ppd_key]

        # --- 3. Index Calculation ---
        indices_result = {
            "obs": calculate_indices(df_obs_dict[task_name]),
            "ppd": {
                STORE_KEY_PREFIX: calculate_indices(pp_data)
            }
        }
        indices_dict_dmc_v2[task_name] = indices_result

        # --- 4. Parameter Merging ---
        # Now that the exact parameter names are injected into the mapping,
        # concat_dfs_by_subj will know exactly which columns to extract.
        fitted_df_dict_dmc_v2[task_name] = concat_dfs_by_subj({STORE_KEY_PREFIX: fitted_params})

    except Exception as e:
        print(f"  [FAIL] {task_name}: {type(e).__name__} - {e}")

store_fits.close_store()
print(f"[OK] Behavioural indices computed for {len(indices_dict_dmc_v2)} tasks.")

# ## 5. Merge with old CSV — Subject indices (params + behaviour)

# %%
print("\n" + "=" * 60)
print("  MERGING DMC_v2 INTO SUBJECT INDICES CSV")
print("=" * 60)

OLD_CSV_PATH = "23subj_indices_across_models_and_tasks.csv"
NEW_CSV_PATH = "23subj_indices_across_models_and_tasks_dmc_v2.csv"

# ── Step 5a: Generate DMC_v2 parameter + behaviour DataFrame ──
print("[..] Building DMC_v2 param+behaviour table ...")
dmc_v2_rows = []
for task_name in task_names:
    if task_name not in fitted_df_dict_dmc_v2:
        continue
    if task_name not in indices_dict_dmc_v2:
        continue

    params_df = fitted_df_dict_dmc_v2[task_name]
    behav_df = indices_dict_dmc_v2[task_name]["obs"]["subject_indices"]

    merged = pd.merge(params_df, behav_df, on="subject_id")
    merged["task_id"] = task_name
    dmc_v2_rows.append(merged)

df_dmc_v2 = pd.concat(dmc_v2_rows, axis=0, ignore_index=True)
df_dmc_v2 = get_col_names(df_dmc_v2)
print(f"  DMC_v2 table: {df_dmc_v2.shape[0]} rows × {df_dmc_v2.shape[1]} cols")

# ── Step 5b: Load old CSV & merge horizontally ──
print(f"[..] Loading old CSV: {OLD_CSV_PATH} ...")
df_old = pd.read_csv(OLD_CSV_PATH)
print(f"  Old table: {df_old.shape[0]} rows × {df_old.shape[1]} cols")

# Merge on (subject_id, task_id)
merge_cols = ["subject_id", "task_id"]
df_new = pd.merge(df_old, df_dmc_v2, on=merge_cols, how="left", suffixes=("", "_dmc_v2"))

# Drop duplicate columns (author_year, task_name — keep originals)
for col_suffix in ["_dmc_v2"]:
    dup_cols = [c for c in df_new.columns if c.endswith(col_suffix)]
    if dup_cols:
        df_new = df_new.drop(columns=dup_cols)

print(f"  New table: {df_new.shape[0]} rows × {df_new.shape[1]} cols")
print(f"  Added columns: {df_new.shape[1] - df_old.shape[1]}")

# ── Step 5c: Save ──
df_new.to_csv(NEW_CSV_PATH, index=False)
print(f"[OK] Saved → {NEW_CSV_PATH}")

# ## 6. Compute DMC_v2 model prediction indices (aBIC, RMSE)

# %%
print("\n" + "=" * 60)
print("  COMPUTING DMC_v2 MODEL PREDICTION INDICES")
print("=" * 60)

map_nparams_dmc_v2 = {STORE_KEY_PREFIX: 6}

model_pred_dmc_v2 = compute_model_prediction_indices(
    indices_dict_dmc_v2,
    map_nparams_dmc_v2,
    parallel=True,
    n_jobs=32,
    show_progress=True
)
model_pred_dmc_v2 = get_col_names(model_pred_dmc_v2)
print(f"  DMC_v2 predictions: {model_pred_dmc_v2.shape[0]} rows")

# ## 7. Merge with old CSV — Model prediction indices

# %%
OLD_PRED_CSV = "23model_prediction_indices.csv"
NEW_PRED_CSV = "23model_prediction_indices_dmc_v2.csv"

print(f"[..] Loading old predictions: {OLD_PRED_CSV} ...")
df_pred_old = pd.read_csv(OLD_PRED_CSV)
print(f"  Old table: {df_pred_old.shape[0]} rows × {df_pred_old.shape[1]} cols")
print(f"  Old models: {sorted(df_pred_old['model'].unique())}")

# ── Concatenate vertically ──
df_pred_new = pd.concat([df_pred_old, model_pred_dmc_v2], axis=0, ignore_index=True)
print(f"  New table: {df_pred_new.shape[0]} rows × {df_pred_new.shape[1]} cols")
print(f"  New models: {sorted(df_pred_new['model'].unique())}")

# ── Save ──
df_pred_new.to_csv(NEW_PRED_CSV, index=False)
print(f"[OK] Saved → {NEW_PRED_CSV}")

# ## 8. Summary

# %%
print("\n" + "=" * 60)
print("  DMC_v2 INDIVIDUAL ANALYSIS — COMPLETE")
print("=" * 60)
print(f"  Subject indices   → {NEW_CSV_PATH}")
print(f"    Rows: {df_new.shape[0]}, Cols: {df_new.shape[1]}")
print("    Models: DDM, DMC, SSP, DSTP, dmc_v2")
print(f"  Prediction indices → {NEW_PRED_CSV}")
print(f"    Rows: {df_pred_new.shape[0]}, Cols: {df_pred_new.shape[1]}")
print(f"    Models: {sorted(df_pred_new['model'].unique())}")
