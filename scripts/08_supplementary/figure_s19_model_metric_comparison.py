#!/usr/bin/env python
# coding: utf-8
"""
Phase 5: BIC/G-Square Supplementary Model Comparison
=====================================================
Normalizes column naming, computes RMSE/G-square/aBIC model comparison
summaries by task and dataset, and generates a supplementary figure.

Input:
  - results/intermediate/model_prediction_indices_extended_dmc.csv

Output:
  - 33_model_metric_supplement.csv
  - figures/supplement/figure_s19_model_metric_comparison.svg and .png
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

from nsbi_module.study_labels import format_author_year
from nsbi_module.project_paths import (
    INTERMEDIATE_DIR,
    SUPPLEMENT_FIGURES_DIR,
    TABLES_DIR,
    ensure_output_directories,
)

plt.rcParams["font.sans-serif"] = ["Arial", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("white")

INPUT_CSV = INTERMEDIATE_DIR / "model_prediction_indices_extended_dmc.csv"
MODELS_MAIN = ["DSTP", "DMC", "SSP", "DDM"]
CRITERIA = ["RMSE", "g_square", "aBIC"]
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
# 1. Load & normalize
# ---------------------------------------------------------------------------
def load_and_normalize():
    """Load CSV, normalize column names, filter models."""
    df = pd.read_csv(INPUT_CSV)

    # Normalize: ensure both 'g_square' and 'chi_square' exist
    if "g_square" not in df.columns and "chi_square" in df.columns:
        df["g_square"] = df["chi_square"]
    if "chi_square" not in df.columns and "g_square" in df.columns:
        df["chi_square"] = df["g_square"]

    # Backfill g_square from chi_square where g_square is NaN
    # The main-model rows may store the equivalent G-square value only in chi_square.
    if "chi_square" in df.columns and "g_square" in df.columns:
        n_backfilled = df["g_square"].isna() & df["chi_square"].notna()
        df.loc[n_backfilled, "g_square"] = df.loc[n_backfilled, "chi_square"]
        print(f"  Backfilled g_square from chi_square: {n_backfilled.sum()} rows")

    # Filter: exclude lee2025, only main models
    df = df.query("author_year != 'lee2025'")
    df = df[df["model"].isin(MODELS_MAIN)].copy()

    print(f"  Loaded {df.shape[0]} rows, {df['task_id'].nunique()} tasks")
    print(f"  Models: {sorted(df['model'].unique())}")
    return df


# ---------------------------------------------------------------------------
# 2. Compute winner proportions
# ---------------------------------------------------------------------------
def compute_winner_proportions(df):
    """Compute winner counts/proportions for each criterion × group_type."""
    rows = []

    for criterion in CRITERIA:
        if criterion not in df.columns:
            print(f"  [WARN] {criterion} not in columns, skipping.")
            continue

        valid = df.dropna(subset=[criterion]).copy()

        # --- Overall ---
        idx_winner = valid.groupby(["task_id", "subject_id"])[criterion].idxmin()
        winners = valid.loc[idx_winner]
        n_cases = len(idx_winner)
        for m in MODELS_MAIN:
            wn = (winners["model"] == m).sum()
            rows.append({
                "criterion": criterion, "group_type": "overall", "group": "all",
                "model": m, "winner_n": wn,
                "winner_prop": wn / n_cases if n_cases > 0 else 0,
                "n_cases": n_cases,
            })

        # --- Per task ---
        for task, grp in winners.groupby("task_name"):
            n = len(grp)
            for m in MODELS_MAIN:
                wn = (grp["model"] == m).sum()
                rows.append({
                    "criterion": criterion, "group_type": "task", "group": task,
                    "model": m, "winner_n": wn,
                    "winner_prop": wn / n if n > 0 else 0,
                    "n_cases": n,
                })

        # --- Per dataset ---
        for ds, grp in winners.groupby("author_year"):
            n = len(grp)
            for m in MODELS_MAIN:
                wn = (grp["model"] == m).sum()
                rows.append({
                    "criterion": criterion, "group_type": "dataset", "group": ds,
                    "model": m, "winner_n": wn,
                    "winner_prop": wn / n if n > 0 else 0,
                    "n_cases": n,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Plot
# ---------------------------------------------------------------------------
def plot_multi_criterion(summary_df, save_svg, save_png):
    """3-column subplot: RMSE, G-square, aBIC winner proportions by dataset."""
    datasets = sorted(summary_df[summary_df["group_type"] == "dataset"]["group"].unique())
    # Format dataset labels
    ds_labels = [format_author_year(d) for d in datasets]

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

    for ax_i, criterion in enumerate(CRITERIA):
        ax = axes[ax_i]
        ds_data = summary_df[
            (summary_df["group_type"] == "dataset") &
            (summary_df["criterion"] == criterion)
        ].copy()

        x = np.arange(len(datasets))
        bottom = np.zeros(len(datasets))
        for m in MODELS_MAIN:
            vals = []
            for d in datasets:
                r = ds_data[(ds_data["group"] == d) & (ds_data["model"] == m)]
                vals.append(r["winner_prop"].values[0] * 100 if len(r) > 0 else 0)
            ax.bar(x, vals, bottom=bottom, label=m,
                   color=MODEL_COLORS[m], edgecolor=MODEL_EDGE[m], linewidth=1.0, width=0.65)
            bottom += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels(ds_labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(criterion.replace("_", " "), fontweight="bold", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax_i == 0:
            ax.set_ylabel("Winner proportion (%)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m], ec=MODEL_EDGE[m], lw=1)
               for m in MODELS_MAIN]
    fig.legend(handles, MODELS_MAIN, loc="upper center", ncol=4,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_svg, bbox_inches="tight")
    strip_trailing_whitespace(save_svg)
    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_svg}")


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    ensure_output_directories()
    print("=" * 60)
    print("  PHASE 5: BIC/G-SQUARE SUPPLEMENT")
    print("=" * 60)

    print("\n[1/3] Loading & normalizing data...")
    df = load_and_normalize()

    print("\n[2/3] Computing winner proportions...")
    summary_df = compute_winner_proportions(df)

    # Quick report
    for criterion in CRITERIA:
        overall = summary_df[
            (summary_df["group_type"] == "overall") &
            (summary_df["criterion"] == criterion)
        ]
        max_row = overall.loc[overall["winner_prop"].idxmax()]
        print(f"  {criterion}: dominant model = {max_row['model']} "
              f"({max_row['winner_prop']*100:.1f}%)")

    csv_path = TABLES_DIR / "figure_s19_model_metric_comparison.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({summary_df.shape[0]} rows)")

    print("\n[3/3] Generating figure...")
    plot_multi_criterion(
        summary_df,
        save_svg=SUPPLEMENT_FIGURES_DIR / "figure_s19_model_metric_comparison.svg",
        save_png=SUPPLEMENT_FIGURES_DIR / "figure_s19_model_metric_comparison.png",
    )

    print("\n  PHASE 5 COMPLETE.")


if __name__ == "__main__":
    main()
