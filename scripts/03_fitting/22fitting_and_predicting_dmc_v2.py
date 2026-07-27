#!/usr/bin/env python
# coding: utf-8

# # DMC_v2 (driftdm_dmc) — Fitting & Posterior Predictive for All Datasets
#
# This script fits the driftdm_dmc model (referred to as DMC_v2 in output)
# to all 21 validation datasets, generates posterior predictive data,
# and saves results into the **existing** `22fitting_and_prediction.h5` FitStore.
#
# Output keys use the `dmc_v2` prefix to avoid conflicts with the
# original 4-model results (DDM/DMC/SSP/DSTP).

# ## 0. Imports

# %%
import pandas as pd
import sys
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

from nsbi_module.utils import FitStore
from nsbi_module.dmc_v2_loader import get_dmc_v2_model, STORE_KEY_PREFIX, DEFAULT_CHECKPOINT

# ## 1. Register & load DMC_v2 model
#
# Model definition, registration, and loading are centralised in
# `nsbi_module/dmc_v2_loader.py`.  Every script imports a single function.

# %%
MODEL_REG_NAME = "driftdm_dmc"
DATA_PATH = "./21preprocessed_datasets.h5"
SAVE_NAME = "22fitting_and_prediction.h5"  # reuse existing store

# ── Dataset filter (prefix match; None = all datasets) ──
# TARGET_FILTER = ["hedge2018", "reymermet2018"]
TARGET_FILTER = None   # uncomment to process all 21 datasets

# ── Load the NSBI model (auto-registers on first call) ──
print(f"\n[..] Loading {MODEL_REG_NAME} from {DEFAULT_CHECKPOINT} ...")
m_dmc_v2 = get_dmc_v2_model()
print("[OK] Model loaded.")

# ── Load all validation datasets ──
print(f"\n[..] Loading datasets from {DATA_PATH} ...")
df_dict = {}
with pd.HDFStore(DATA_PATH) as store:
    for key in store.keys():
        key = key.lstrip("/")
        if "meta_data" in key:
            continue
        df_dict[key] = store[key]
        print(f"  {key}: {len(df_dict[key])} rows, "
              f"{df_dict[key]['subject_id'].nunique()} subjects")
print(f"[OK] {len(df_dict)} datasets loaded.")

# ── Apply dataset filter ──
if TARGET_FILTER is not None:
    df_dict = {
        k: v for k, v in df_dict.items()
        if any(k.startswith(f) for f in TARGET_FILTER)
    }
    print(f"[OK] Filtered to {len(df_dict)} dataset(s): {sorted(df_dict.keys())}")

# ## 3. Fit DMC_v2 to selected datasets

# %%
print("\n" + "=" * 60)
print("  FITTING DMC_v2 (driftdm_dmc) TO SELECTED DATASETS")
print("=" * 60)

fit_store = FitStore(SAVE_NAME)

for task_name, df_obs in tqdm(df_dict.items(), desc="Fitting tasks"):
    key_fitted = f"{task_name}_{STORE_KEY_PREFIX}_fitted"
    key_trace  = f"{task_name}_{STORE_KEY_PREFIX}_fitted_trace"

    if fit_store.isexist(key_fitted):
        print(f"  [SKIP] {key_fitted} already exists")
        continue

    try:
        # Raw posterior draws
        trace = m_dmc_v2.fit_data(
            df_obs, return_infdata=False, batchsize=32, show_progress=False
        )
        # Subject-level summary (mean of posterior per subject)
        summary = m_dmc_v2.df_summary(trace)

        fit_store[key_trace]  = {STORE_KEY_PREFIX: trace}
        fit_store[key_fitted] = {STORE_KEY_PREFIX: summary}
        print(f"  [DONE] {task_name}")
    except Exception as e:
        print(f"  [FAIL] {task_name}: {e}")

fit_store.close_store()
print("\n[OK] Fitting complete. Results saved to:", SAVE_NAME)

# ## 4. Generate Posterior Predictive Data (PPD)

# %%
N_SAMPLES = 200   # posterior samples per subject
N_TRIALS  = 250    # trials per sample → 200 × 250 = 50000 total per subject

print("\n" + "=" * 60)
print("  POSTERIOR PREDICTIVE SIMULATION (DMC_v2)")
print("=" * 60)

fit_store = FitStore(SAVE_NAME)

for task_name, df_obs in tqdm(df_dict.items(), desc="PPC tasks"):
    key_predicted = f"{task_name}_{STORE_KEY_PREFIX}_predicted"
    key_trace     = f"{task_name}_{STORE_KEY_PREFIX}_fitted_trace"

    if fit_store.isexist(key_predicted):
        print(f"  [SKIP] {key_predicted} already exists")
        continue

    if not fit_store.isexist(key_trace):
        print(f"  [WARN] No trace for {task_name}, skipping PPC")
        continue

    try:
        # Retrieve trace (stored as dict: {"dmc_v2": DataFrame})
        trace_dict = fit_store[key_trace]
        trace = trace_dict[STORE_KEY_PREFIX]

        # Drop chain/draw if present
        cols_to_drop = [c for c in ["chain", "draw"] if c in trace.columns]
        if cols_to_drop:
            trace = trace.drop(columns=cols_to_drop)

        # Sample N_SAMPLES posterior sets per subject
        sampled = trace.groupby("subject_id").sample(
            n=N_SAMPLES, replace=True
        ).copy()
        sampled["_batch_id"] = sampled.groupby("subject_id").cumcount()

        # Build batched inputs
        batched = {}
        for bid, grp in sampled.groupby("_batch_id"):
            batched[bid] = grp.drop(columns=["_batch_id"])

        # Simulate
        collected = []
        for i in range(N_SAMPLES):
            pp = m_dmc_v2.posterior_predictive(batched[i], n_trial=N_TRIALS)
            collected.append(pp)

        pp_merged = pd.concat(collected, ignore_index=True)
        fit_store[key_predicted] = {STORE_KEY_PREFIX: pp_merged}
        print(f"  [DONE] {task_name} — {len(pp_merged)} rows")
    except Exception as e:
        print(f"  [FAIL] {task_name}: {e}")

fit_store.close_store()
print("\n[OK] Posterior predictive simulation complete.")

# ## 5. Summary

# %%
print("\n" + "=" * 60)
print("  DMC_v2 FITTING & PPD — COMPLETE")
print("=" * 60)
print(f"  HDF5 file : {SAVE_NAME}")
print(f"  Key prefix: {STORE_KEY_PREFIX}")
print(f"  Datasets  : {len(df_dict)}")
print("\n  Stored keys per dataset:")
print(f"    {{task}}_{STORE_KEY_PREFIX}_fitted_trace  → raw posterior draws")
print(f"    {{task}}_{STORE_KEY_PREFIX}_fitted        → subject-level parameter means")
print(f"    {{task}}_{STORE_KEY_PREFIX}_predicted      → posterior predictive trials")
