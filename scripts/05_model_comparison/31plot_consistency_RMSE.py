#!/usr/bin/env python
# coding: utf-8
"""
Standalone script: generate Fig 4 – cross-task model consistency.

Layout (2 panels):
  A: Model stability bar chart (which model is most consistent across tasks)
  B: Stacked consistency by dataset

Usage:
    python 31plot_consistency_RMSE.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

import sys

from utils_ind_diff import *  # noqa: F403

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["Arial", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 17
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14
sns.set_style("white")

ipython = globals().get("get_ipython")
if ipython is not None:
    shell = ipython()
    if shell is not None:
        shell.run_line_magic("load_ext", "autoreload")
        shell.run_line_magic("autoreload", "2")


def strip_trailing_whitespace(path):
    """Strip trailing whitespace from generated SVG files for clean diffs."""
    figure_path = Path(path)
    text = figure_path.read_text(encoding="utf-8")
    figure_path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 1.  Constants  (matching Fig 2 style)
# ---------------------------------------------------------------------------
metric = "RMSE"
models_sorted = ["DSTP", "DMC", "SSP", "DDM"]
model_colors = {
    "DSTP": "#81cef0",
    "DMC": "#95d8c3",
    "SSP": "#92d28e",
    "DDM": "#b0d97c",
}
model_edge_colors = {
    "DSTP": "#4dc3eb",
    "DMC": "#75cfb6",
    "SSP": "#6fc96f",
    "DDM": "#9dd25b",
}

# ---------------------------------------------------------------------------
# 2.  Load data  (shared with 31prediciontion_comparison_RMSE.py)
# ---------------------------------------------------------------------------
model_prediction_indices_long = pd.read_csv("../03_fitting/23model_prediction_indices_dmc_v2.csv")
model_prediction_indices_long = model_prediction_indices_long.query(
    "author_year != 'lee2025'"
)
model_prediction_indices_long = model_prediction_indices_long[
    model_prediction_indices_long["model"].isin(models_sorted)
].copy()
model_prediction_indices_long = rank_models_by_metric(  # noqa: F405
    model_prediction_indices_long, metric=metric
)

# ---------------------------------------------------------------------------
# 3.  Filter to studies with cross-task data
# ---------------------------------------------------------------------------
selected_author_years = [
    'eisenberg2019',   # flanker, simon, stroop
    'whitehead2019',   # flanker, simon, stroop
    'ulrich2015',      # simon, flanker
    'clayson2025',     # flanker, stroop
    'reymermet2018',   # flanker, simon, stroop
    'hedge2018',       # flanker, stroop  (simon filtered out)
]

model_pred_indices_long_cross_task = model_prediction_indices_long[
    model_prediction_indices_long['author_year'].isin(selected_author_years)
]
model_pred_indices_long_cross_task = model_pred_indices_long_cross_task.query(
    "task_id != 'hedge2018simon'"
)
best_model_cross_task = get_best_model_by_metric(  # noqa: F405
    model_pred_indices_long_cross_task, metric=metric
)

# ---------------------------------------------------------------------------
# 4.  Compute consistency metrics
# ---------------------------------------------------------------------------
df = best_model_cross_task
df_par, author_summary, model_stability_df = calculate_par_metrics(  # noqa: F405
    df, group_col="author_year"
)
df_par['author_year'] = df_par['author_year'].apply(format_author_year)  # noqa: F405

# ---------------------------------------------------------------------------
# 5.  Build figure (A: stability bar, B: stacked consistency)
# ---------------------------------------------------------------------------
full_color_map = model_colors.copy()
full_color_map["Inconsistent"] = "#ffffff"

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(11, 5),
    gridspec_kw={"width_ratios": [0.35, 0.65], "wspace": 0.3},
)

# --- Panel A: Model stability bar chart ---
plot_bar_with_text(  # noqa: F405
    model_stability_df,
    x_col='winner_model',
    y_col='percentage',
    count_col='n_subj',
    edge_color=model_edge_colors,
    ax=ax1,
    palette=model_colors,
    models_sorted=models_sorted,
)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.set_ylabel("Percentage of subjects (%)")
add_subplot_label(ax1, 'A', x_offset=-0.16, y_offset=1.04, fontsize=18)  # noqa: F405

# Nature-style axis for panel A
apply_nature_bar_axis_style(  # noqa: F405
    ax1, xlabel="", ylabel="Percentage of subjects (%)",
    label_fontsize=17, tick_fontsize=14,
)

# --- Panel B: Stacked consistency plot ---
plot_stacked_consistency(  # noqa: F405
    df_par,
    group_col='author_year',
    custom_colors=full_color_map,
    ax=ax2,
    rotate_xticks=32,
    ylabel=None,
    alpha=1.0,
    label_threshold=10,
    edge_color=model_edge_colors,
    show_legend=False,
    models_sorted=models_sorted.copy(),  # copy: function mutates the list
)
add_subplot_label(ax2, 'B', x_offset=-0.11, y_offset=1.04, fontsize=18)  # noqa: F405

# Nature-style axis for panel B
apply_nature_bar_axis_style(  # noqa: F405
    ax2, xlabel="", ylabel="Percentage of subjects (%)",
    label_fontsize=17, tick_fontsize=14,
)

# --- Shared legend ---
handles = [
    mpatches.Patch(facecolor=model_colors[model],
                   edgecolor=model_edge_colors[model], label=model)
    for model in models_sorted
]
# Append Inconsistent (white with gray edge for visibility)
handles.append(
    mpatches.Patch(facecolor="#ffffff", edgecolor="#cccccc",
                   linewidth=1.0, label="Inconsistent")
)
fig.legend(
    handles=handles,
    loc="upper center",
    ncol=5,
    bbox_to_anchor=(0.5, 1.02),
    frameon=False,
    fontsize=14,
)

plt.tight_layout(rect=[0, 0, 1, 0.9])

# ---------------------------------------------------------------------------
# 6.  Save
# ---------------------------------------------------------------------------
consistency_svg = f"../figs/31_model_comparision_consistency_across_task_{metric}.svg"
plt.savefig(consistency_svg, bbox_inches='tight')
strip_trailing_whitespace(consistency_svg)
plt.savefig(
    f"../figs/31_model_comparision_consistency_across_task_{metric}.png",
    dpi=300,
    bbox_inches='tight',
)
plt.close(fig)

print(f"Saved: {consistency_svg}")
print(f"Saved: ../figs/31_model_comparision_consistency_across_task_{metric}.png")
