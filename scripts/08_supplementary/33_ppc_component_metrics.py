#!/usr/bin/env python
# coding: utf-8
"""
Phase 4: Component-Wise CAF and Delta Posterior Predictive Metrics
==================================================================
Computes CAF_RMSE, Delta_RMSE, and Delta_slope_error for each
dataset × model combination from the PPC plotting data pickle.

Uses 24_ppc_process_data_dict.pkl which stores:
  plotting_data[dataset]["observed"]["caf"] / ["delta"]
  plotting_data[dataset]["models"][model]["caf"] / ["delta"]

Output:
  - 33_ppc_component_metrics.csv
  - figs/S_ppc_component_metrics.svg  & .png
"""
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["Arial", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("white")

MODEL_COLORS = {
    "DDM": "#b0d97c", "SSP": "#92d28e",
    "DMC": "#95d8c3", "DSTP": "#81cef0",
}
MODELS = ["DSTP", "DMC", "SSP", "DDM"]
PPC_PKL = "../06_ppc/24_ppc_process_data_dict.pkl"


def strip_trailing_whitespace(path):
    p = Path(path)
    t = p.read_text(encoding="utf-8")
    p.write_text("\n".join(l.rstrip() for l in t.splitlines()) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Compute component metrics
# ---------------------------------------------------------------------------
def compute_component_metrics(plotting_data):
    """Compute CAF_RMSE, Delta_RMSE, Delta_slope_error per dataset × model."""
    rows = []

    for ds_name, ds_data in plotting_data.items():
        obs_caf = ds_data["observed"]["caf"]
        obs_delta = ds_data["observed"]["delta"]

        # Extract task info from dataset name
        import re
        m = re.match(r"([a-zA-Z]+)(\d{4})([a-zA-Z]+)", ds_name)
        author_year = f"{m.group(1)}{m.group(2)}" if m else ds_name
        task_name = m.group(3) if m else ""

        for model_name, model_data in ds_data["models"].items():
            pred_caf = model_data["caf"]
            pred_delta = model_data["delta"]

            # --- CAF RMSE ---
            # CAF data has columns: bin, comp, incomp, effect
            caf_cols = ["comp", "incomp"]
            obs_caf_vals = obs_caf[caf_cols].to_numpy().ravel()
            pred_caf_vals = pred_caf[caf_cols].to_numpy().ravel()
            caf_rmse = float(np.sqrt(np.mean((obs_caf_vals - pred_caf_vals) ** 2)))
            n_caf_bins = len(obs_caf)

            # --- Delta RMSE ---
            # Delta data has columns: bin, mean_comp, mean_incomp, mean_bin, mean_effect
            if "mean_effect" in obs_delta.columns:
                obs_delta_vals = obs_delta["mean_effect"].to_numpy()
                pred_delta_vals = pred_delta["mean_effect"].to_numpy()
                delta_rmse = float(np.sqrt(np.mean((obs_delta_vals - pred_delta_vals) ** 2)))
            else:
                delta_rmse = np.nan
            n_delta_bins = len(obs_delta)

            # --- Delta slope error ---
            if "mean_bin" in obs_delta.columns and "mean_effect" in obs_delta.columns:
                X_obs = obs_delta["mean_bin"].values.reshape(-1, 1)
                y_obs = obs_delta["mean_effect"].values
                try:
                    lr_obs = LinearRegression().fit(X_obs, y_obs)
                    obs_slope = float(lr_obs.coef_[0])
                except Exception:
                    obs_slope = np.nan

                X_pred = pred_delta["mean_bin"].values.reshape(-1, 1)
                y_pred = pred_delta["mean_effect"].values
                try:
                    lr_pred = LinearRegression().fit(X_pred, y_pred)
                    pred_slope = float(lr_pred.coef_[0])
                except Exception:
                    pred_slope = np.nan

                delta_slope_error = abs(obs_slope - pred_slope) if (
                    not np.isnan(obs_slope) and not np.isnan(pred_slope)
                ) else np.nan
            else:
                obs_slope = np.nan
                pred_slope = np.nan
                delta_slope_error = np.nan

            rows.append({
                "dataset": ds_name,
                "author_year": author_year,
                "task_name": task_name,
                "model": model_name,
                "CAF_RMSE": caf_rmse,
                "Delta_RMSE": delta_rmse,
                "Delta_slope_observed": obs_slope,
                "Delta_slope_predicted": pred_slope,
                "Delta_slope_error": delta_slope_error,
                "n_caf_bins": n_caf_bins,
                "n_delta_bins": n_delta_bins,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Plot
# ---------------------------------------------------------------------------
def plot_component_metrics(df, save_svg, save_png):
    """3-panel figure: CAF_RMSE, Delta_RMSE, Delta_slope_error by task × model."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    tasks = sorted(df["task_name"].dropna().unique())
    x = np.arange(len(tasks))
    bar_width = 0.18

    for ax_i, (metric, title) in enumerate([
        ("CAF_RMSE", "A: CAF Accuracy RMSE"),
        ("Delta_RMSE", "B: Delta RT RMSE"),
        ("Delta_slope_error", "C: |Delta Slope Error|"),
    ]):
        ax = axes[ax_i]
        for j, m in enumerate(MODELS):
            vals = []
            for t in tasks:
                row = df[(df["task_name"] == t) & (df["model"] == m)]
                vals.append(row[metric].mean() if len(row) > 0 else 0)
            ax.bar(x + j * bar_width - 1.5 * bar_width, vals, bar_width,
                   label=m, color=MODEL_COLORS.get(m, "#999"),
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in tasks])
        ax.set_title(title, fontweight="bold", loc="left", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax_i == 0:
            ax.set_ylabel("RMSE")
        if ax_i == 2:
            ax.legend(fontsize=8, frameon=False, ncol=1, loc="upper left")

    plt.tight_layout()
    plt.savefig(save_svg, bbox_inches="tight")
    strip_trailing_whitespace(save_svg)
    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_svg}")


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  PHASE 4: COMPONENT-WISE PPC METRICS")
    print("=" * 60)

    print("\n[1/3] Loading PPC plotting data...")
    with open(PPC_PKL, "rb") as f:
        plotting_data = pickle.load(f)
    print(f"  Loaded {len(plotting_data)} datasets.")

    print("\n[2/3] Computing component metrics...")
    df = compute_component_metrics(plotting_data)
    csv_path = "33_ppc_component_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({df.shape[0]} rows)")

    # Quick summary
    print("\n  Mean CAF_RMSE by model:")
    for m in MODELS:
        print(f"    {m}: {df[df['model'] == m]['CAF_RMSE'].mean():.4f}")
    print("  Mean Delta_RMSE by model:")
    for m in MODELS:
        print(f"    {m}: {df[df['model'] == m]['Delta_RMSE'].mean():.4f}")
    print("  Mean Delta_slope_error by model:")
    for m in MODELS:
        print(f"    {m}: {df[df['model'] == m]['Delta_slope_error'].mean():.4f}")

    print("\n[3/3] Generating figure...")
    plot_component_metrics(
        df,
        save_svg="../figs/S_ppc_component_metrics.svg",
        save_png="../figs/S_ppc_component_metrics.png",
    )

    print("\n  PHASE 4 COMPLETE.")


if __name__ == "__main__":
    main()
