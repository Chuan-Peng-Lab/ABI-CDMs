#!/usr/bin/env python
# coding: utf-8
"""
Phase 3: RMSE Scaling Sensitivity Analysis
===========================================
Recomputes model prediction indices with RMSE component breakdown,
then derives sensitivity across CAF scaling factors algebraically.

Strategy:
  1. One full computation with default caf_scale_base=500 (returns caf_rmse, cdf_rmse,
     n_caf_points, n_cdf_points, caf_weight for each subject/task/model).
  2. Derive RMSE for other scaling factors:
     RMSE(s) = cdf_rmse + (n_cdf / n_caf) * s * caf_rmse
  3. Identify winners and compute agreement with default.

Output:
  - 33_rmse_scaling_sensitivity.csv
  - 33_rmse_scaling_summary.csv
  - figs/S_rmse_scaling_sensitivity.svg  & .png
"""
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

from utils import timer, FitStore
from analysis_utils import (
    compute_model_prediction_indices,
    calculate_indices,
    concat_dfs_by_subj,
    get_col_names,
)

plt.rcParams["font.sans-serif"] = ["Arial", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("white")

# ---------------------------------------------------------------------------
# 0. Constants
# ---------------------------------------------------------------------------
STORE_DATASETS_PATH = "21preprocessed_datasets.h5"
STORE_FITS_PATH = "22fitting_and_prediction.h5"
CAF_SCALE_BASES = [0, 125, 250, 375, 500, 750, 1000]
MAP_NPARAMS = {"DDM": 4, "DMC": 5, "SSP": 6, "DSTP": 7}
MODELS_MAIN = ["DSTP", "DMC", "SSP", "DDM"]
MODEL_COLORS = {
    "DSTP": "#81cef0", "DMC": "#95d8c3",
    "SSP": "#92d28e", "DDM": "#b0d97c",
}
MODEL_EDGE = {
    "DSTP": "#4dc3eb", "DMC": "#75cfb6",
    "SSP": "#6fc96f", "DDM": "#9dd25b",
}


def strip_trailing_whitespace(path):
    p = Path(path)
    t = p.read_text(encoding="utf-8")
    p.write_text("\n".join(l.rstrip() for l in t.splitlines()) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Load indices_dict from HDF5 stores
# ---------------------------------------------------------------------------
@timer
def get_indices_dict():
    """Load & compute indices_dict from HDF5 (reuses preprocess logic)."""
    import pandas as pd
    from joblib import Parallel, delayed

    datasets_dict = {}
    with pd.HDFStore(STORE_DATASETS_PATH, mode="r") as s:
        for key in s.keys():
            k = key[1:]
            if "meta_data" not in k:
                datasets_dict[k] = s[key]

    store_fits = FitStore(STORE_FITS_PATH)
    fits_keys = store_fits.key_list_df["key"].str.split("_", expand=True).loc[:, 0].values
    fits_predicted = {}
    try:
        for kn in fits_keys:
            fits_predicted[kn] = store_fits[f"{kn}_predicted"]
    finally:
        store_fits.close_store()

    def _process(key_name, obs_df, pred_dict):
        try:
            r = {
                "obs": calculate_indices(obs_df),
                "ppd": {m: calculate_indices(pp) for m, pp in pred_dict.items()},
            }
            return key_name, r
        except Exception as e:
            print(f"  [ERR] {key_name}: {e}")
            return key_name, None

    tasks = [(kn, datasets_dict[kn], fits_predicted[kn]) for kn in fits_predicted]
    results = Parallel(n_jobs=-1)(delayed(_process)(*t) for t in tasks)

    indices_dict = {}
    for kn, r in results:
        if r is not None:
            indices_dict[kn] = r
    return indices_dict


# ---------------------------------------------------------------------------
# 2. One full metric computation
# ---------------------------------------------------------------------------
@timer
def compute_base_metrics(indices_dict):
    """Compute metrics once with caf_scale_base=500, capturing all components."""
    return compute_model_prediction_indices(
        indices_dict,
        MAP_NPARAMS,
        parallel=True,
        n_jobs=-1,
        show_progress=True,
        caf_scale_base=500,
    )


# ---------------------------------------------------------------------------
# 3. Derive sensitivity
# ---------------------------------------------------------------------------
def derive_sensitivity(base_df):
    """
    From one base run (caf_scale_base=500), derive RMSE for all scaling factors.
    RMSE(s) = cdf_rmse + (n_cdf/n_caf) * s * caf_rmse
    """
    rows = []
    base_df = base_df.copy()
    base_df = base_df.dropna(subset=["caf_rmse", "cdf_rmse", "n_caf_points", "n_cdf_points"])
    base_df = base_df[base_df["model"].isin(MODELS_MAIN)]

    # Identify default winner (at caf_scale_base=500)
    idx_default = base_df.groupby(["task_id", "subject_id"])["RMSE"].idxmin()
    default_winners = base_df.loc[idx_default, ["task_id", "subject_id", "model"]].rename(
        columns={"model": "default_winner_model"}
    )

    for s in CAF_SCALE_BASES:
        tmp = base_df.copy()
        w = (tmp["n_cdf_points"] / tmp["n_caf_points"]) * s
        tmp["caf_weight"] = w
        tmp["caf_scale_base"] = s
        tmp["RMSE"] = tmp["cdf_rmse"] + w * tmp["caf_rmse"]
        rows.append(tmp)

    all_df = pd.concat(rows, ignore_index=True)

    # Identify winner per (caf_scale_base, task_id, subject_id)
    idx_winner = all_df.groupby(["caf_scale_base", "task_id", "subject_id"])["RMSE"].idxmin()
    all_df["is_winner"] = False
    all_df.loc[idx_winner, "is_winner"] = True

    # Merge default winner
    all_df = all_df.merge(default_winners, on=["task_id", "subject_id"], how="left")
    all_df["same_as_default"] = all_df["model"] == all_df["default_winner_model"]

    # Extract author_year, task_name
    all_df = get_col_names(all_df)

    cols_order = [
        "task_id", "author_year", "task_name", "subject_id",
        "model", "caf_scale_base", "caf_weight",
        "caf_rmse", "cdf_rmse", "RMSE",
        "is_winner", "default_winner_model", "same_as_default",
    ]
    return all_df[cols_order]


# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------
def compute_summary(all_df):
    summary_rows = []

    # --- Overall ---
    for s, grp in all_df.groupby("caf_scale_base"):
        winner_only = grp[grp["is_winner"]]
        n_cases = grp.groupby(["task_id", "subject_id"]).ngroups
        for m in MODELS_MAIN:
            wn = (winner_only["model"] == m).sum()
            agreement = winner_only["same_as_default"].mean() if len(winner_only) > 0 else np.nan
            summary_rows.append({
                "caf_scale_base": s, "group_type": "overall", "group": "all",
                "model": m, "n_cases": n_cases,
                "winner_n": wn, "winner_prop": wn / n_cases if n_cases > 0 else 0,
                "agreement_with_default": agreement,
            })

    # --- Per task ---
    for (s, task), grp in all_df.groupby(["caf_scale_base", "task_name"]):
        winner_only = grp[grp["is_winner"]]
        n_cases = grp.groupby(["task_id", "subject_id"]).ngroups
        for m in MODELS_MAIN:
            wn = (winner_only["model"] == m).sum()
            agreement = winner_only["same_as_default"].mean() if len(winner_only) > 0 else np.nan
            summary_rows.append({
                "caf_scale_base": s, "group_type": "task", "group": task,
                "model": m, "n_cases": n_cases,
                "winner_n": wn, "winner_prop": wn / n_cases if n_cases > 0 else 0,
                "agreement_with_default": agreement,
            })

    # --- Per dataset (author_year) ---
    for (s, ds), grp in all_df.groupby(["caf_scale_base", "author_year"]):
        winner_only = grp[grp["is_winner"]]
        n_cases = grp.groupby(["task_id", "subject_id"]).ngroups
        for m in MODELS_MAIN:
            wn = (winner_only["model"] == m).sum()
            agreement = winner_only["same_as_default"].mean() if len(winner_only) > 0 else np.nan
            summary_rows.append({
                "caf_scale_base": s, "group_type": "dataset", "group": ds,
                "model": m, "n_cases": n_cases,
                "winner_n": wn, "winner_prop": wn / n_cases if n_cases > 0 else 0,
                "agreement_with_default": agreement,
            })

    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------
def plot_sensitivity(summary_df, save_svg, save_png):
    """4-panel figure: A) overall winners, B) agreement, C) task, D) dataset heatmap."""
    fig = plt.figure(figsize=(11, 9))

    # Panel A: Overall winner proportions across scaling factors
    ax_a = fig.add_subplot(2, 2, 1)
    overall = summary_df[summary_df["group_type"] == "overall"]
    x = overall["caf_scale_base"].unique()
    x_sorted = sorted(x)
    bottom = np.zeros(len(x_sorted))
    for m in MODELS_MAIN:
        vals = []
        for s in x_sorted:
            row = overall[(overall["caf_scale_base"] == s) & (overall["model"] == m)]
            vals.append(row["winner_prop"].values[0] * 100 if len(row) > 0 else 0)
        ax_a.bar(range(len(x_sorted)), vals, bottom=bottom, label=m,
                 color=MODEL_COLORS[m], edgecolor=MODEL_EDGE[m], linewidth=1.0, width=0.7)
        bottom += np.array(vals)
    ax_a.set_xticks(range(len(x_sorted)))
    ax_a.set_xticklabels([str(s) for s in x_sorted])
    ax_a.set_ylim(0, 100)
    ax_a.set_ylabel("Winner proportion (%)")
    ax_a.set_xlabel("CAF scale base")
    ax_a.set_title("A: Overall winner proportions", fontweight="bold", loc="left")
    ax_a.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.05))
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # Panel B: Agreement with default (500)
    ax_b = fig.add_subplot(2, 2, 2)
    agree = summary_df[summary_df["group_type"] == "overall"].drop_duplicates(
        ["caf_scale_base", "agreement_with_default"]
    )
    ax_b.plot(agree["caf_scale_base"], agree["agreement_with_default"] * 100, "o-", color="#2c7bb6", lw=2)
    ax_b.axhline(100, color="gray", ls="--", alpha=0.5)
    ax_b.set_ylabel("Agreement with default (%)")
    ax_b.set_xlabel("CAF scale base")
    ax_b.set_ylim(80, 105)
    ax_b.set_title("B: Agreement with default (s=500)", fontweight="bold", loc="left")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # Panel C: Task-level winners at selected factors
    ax_c = fig.add_subplot(2, 2, 3)
    selected_scales = [0, 500, 1000]
    tasks = sorted(summary_df[summary_df["group_type"] == "task"]["group"].unique())
    bar_width = 0.25
    x_pos = np.arange(len(tasks))
    for i, s in enumerate(selected_scales):
        task_data = summary_df[(summary_df["group_type"] == "task") & (summary_df["caf_scale_base"] == s)]
        # Get dominant model per task
        dom = task_data.loc[task_data.groupby("group")["winner_prop"].idxmax()]
        props = []
        for t in tasks:
            r = dom[dom["group"] == t]
            props.append(r["winner_prop"].values[0] * 100 if len(r) > 0 else 0)
        ax_c.bar(x_pos + i * bar_width - bar_width, props, bar_width,
                 label=f"s={s}", color=[plt.cm.Blues(0.3 + i * 0.3)], edgecolor="black", linewidth=0.5)
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(tasks, fontsize=9)
    ax_c.set_ylabel("Dominant model win %")
    ax_c.set_title("C: Dominant model win% by task", fontweight="bold", loc="left")
    ax_c.legend(fontsize=8)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # Panel D: Dataset heatmap — agreement with default
    ax_d = fig.add_subplot(2, 2, 4)
    ds_data = summary_df[summary_df["group_type"] == "dataset"].drop_duplicates(
        ["caf_scale_base", "group", "agreement_with_default"]
    )
    ds_pivot = ds_data.pivot(index="group", columns="caf_scale_base", values="agreement_with_default")
    sns.heatmap(ds_pivot * 100, annot=True, fmt=".0f", cmap="RdYlGn", vmin=70, vmax=100,
                ax=ax_d, cbar_kws={"label": "Agreement (%)"}, linewidths=0.5)
    ax_d.set_title("D: Dataset agreement with default", fontweight="bold", loc="left")

    plt.tight_layout()
    plt.savefig(save_svg, bbox_inches="tight")
    strip_trailing_whitespace(save_svg)
    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_svg}")
    print(f"Saved: {save_png}")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  PHASE 3: RMSE SCALING SENSITIVITY")
    print("=" * 60)

    # Step 1: Load indices
    print("\n[1/5] Loading indices_dict from HDF5 stores...")
    indices_dict = get_indices_dict()
    print(f"  Loaded {len(indices_dict)} tasks.")

    # Step 2: Compute base metrics
    print("\n[2/5] Computing base metrics (caf_scale_base=500)...")
    base_df = compute_base_metrics(indices_dict)
    print(f"  Base metrics: {base_df.shape[0]} rows × {base_df.shape[1]} cols")

    # Step 3: Derive sensitivity
    print("\n[3/5] Deriving sensitivity across scaling factors...")
    all_df = derive_sensitivity(base_df)
    sensitivity_path = "33_rmse_scaling_sensitivity.csv"
    all_df.to_csv(sensitivity_path, index=False)
    print(f"  Saved: {sensitivity_path} ({all_df.shape[0]} rows)")

    # Step 4: Summary
    print("\n[4/5] Computing summary statistics...")
    summary_df = compute_summary(all_df)
    summary_path = "33_rmse_scaling_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path} ({summary_df.shape[0]} rows)")

    # Quick check
    default_agreement = summary_df[
        (summary_df["group_type"] == "overall") &
        (summary_df["caf_scale_base"] == 0)
    ]["agreement_with_default"].values[0]
    print(f"\n  Agreement at s=0 with default: {default_agreement*100:.1f}%")
    if default_agreement < 0.9:
        print("  [WARNING] Low agreement — some model selections may differ at extreme scaling!")

    # Step 5: Plot
    print("\n[5/5] Generating figure...")
    plot_sensitivity(
        summary_df,
        save_svg="../figs/S_rmse_scaling_sensitivity.svg",
        save_png="../figs/S_rmse_scaling_sensitivity.png",
    )

    print("\n  PHASE 3 COMPLETE.")


if __name__ == "__main__":
    main()
