#!/usr/bin/env python
# coding: utf-8
"""
Standalone script: generate the 31model_comparision_percentage_RMSE figure only.

Panel A: stacked bars by task (sorted by DSTP/DMC proportion)
Panel B: stacked bars by dataset (sorted by DSTP/DMC proportion)

Usage:
    python 31plot_percentage_RMSE.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

import sys

from nsbi_module.utils_ind_diff import *  # noqa: F403

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
# 1. Load and prepare data
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

summary_df = summarize_model_performance(  # noqa: F405
    model_prediction_indices_long, group_col="task_id", metric=metric
)

best_model = get_best_model_by_metric(  # noqa: F405
    model_prediction_indices_long, metric=metric
)
best_model["author_year"] = best_model["author_year"].apply(format_author_year)  # noqa: F405
best_model["author_year"] = best_model["author_year"].replace(
    {"Reymermet 2018": "Rey-Mermet 2018"}
)
best_model["task_name"] = best_model["task_name"].apply(format_task_name)  # noqa: F405
best_model["task_id"] = best_model["task_id"].apply(format_task_id)  # noqa: F405
best_model["task_id"] = best_model["task_id"].str.replace(
    "Reymermet 2018", "Rey-Mermet 2018", regex=False
)

best_model_proportion = calc_best_model_proportion(  # noqa: F405
    best_model, group_cols=["task_id", "task_name", "author_year"]
)

# ---------------------------------------------------------------------------
# 2. Compute proportions sorted by DSTP/DMC dominance
# ---------------------------------------------------------------------------
task_proportion = calc_best_model_proportion(  # noqa: F405
    best_model, group_cols=["task_name"]
)
task_order = order_groups_by_model_proportions(  # noqa: F405
    task_proportion,
    group_col="task_name",
    sort_models=("DSTP", "DMC"),
)

dataset_proportion = calc_best_model_proportion(  # noqa: F405
    best_model, group_cols=["author_year"]
)
dataset_order = order_groups_by_model_proportions(  # noqa: F405
    dataset_proportion,
    group_col="author_year",
    sort_models=("DSTP", "DMC"),
)

# ---------------------------------------------------------------------------
# 3. Plot: stacked percentage bars (Panel A + B)
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(11, 5),
    gridspec_kw={"width_ratios": [0.38, 0.62], "wspace": 0.25},
)

plot_stacked_model_proportion_bar(  # noqa: F405
    task_proportion,
    colors=model_colors,
    group_col="task_name",
    models_sorted=models_sorted,
    group_order=task_order,
    edge_color=model_edge_colors,
    xlabel="",
    ylabel="Percentage of subjects (%)",
    ax=ax1,
    label_threshold=15,
)
add_subplot_label(ax1, "A", x_offset=-0.16, y_offset=1.04, fontsize=18)  # noqa: F405

plot_stacked_model_proportion_bar(  # noqa: F405
    dataset_proportion,
    colors=model_colors,
    group_col="author_year",
    models_sorted=models_sorted,
    group_order=dataset_order,
    edge_color=model_edge_colors,
    xlabel="",
    ylabel="",
    rotate_x_labels=32,
    ax=ax2,
    label_threshold=15,
)
add_subplot_label(ax2, "B", x_offset=-0.11, y_offset=1.04, fontsize=18)  # noqa: F405

handles = [
    mpatches.Patch(
        facecolor=model_colors[model],
        edgecolor=model_edge_colors[model],
        label=model,
    )
    for model in models_sorted
]
fig.legend(
    handles=handles,
    loc="upper center",
    ncol=len(models_sorted),
    bbox_to_anchor=(0.5, 1.02),
    frameon=False,
    fontsize=14,
)

plt.tight_layout(rect=[0, 0, 1, 0.9])
percentage_svg = f"../figs/31model_comparision_percentage_{metric}.svg"
plt.savefig(percentage_svg, bbox_inches="tight")
strip_trailing_whitespace(percentage_svg)
plt.savefig(
    f"../figs/31model_comparision_percentage_{metric}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)
print(f"Saved: {percentage_svg}")
print(f"Saved: ../figs/31model_comparision_percentage_{metric}.png")
