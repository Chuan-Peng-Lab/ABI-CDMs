#!/usr/bin/env python
# coding: utf-8
"""
Standalone script: generate Fig 5 – retest model consistency.

Layout (2 panels):
  A: Model stability bar chart (which model is most stable across retest sessions)
  B: Pie chart grid (rows = task, cols = study) showing consistent-model distribution

Usage:
    python 31plot_retest_RMSE.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
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
# 2.  Load retest data
# ---------------------------------------------------------------------------
model_prediction_indices_long = pd.read_csv("../03_fitting/23model_prediction_indices_retest.csv")
best_model_retest = get_best_model_by_metric(  # noqa: F405
    model_prediction_indices_long,
    group=['subject_id', 'task_id', 'author_year', 'task_name', "session_id"],
    metric=metric,
)

# Compute consistency metrics (with session-level granularity)
df = best_model_retest
df_par, _, model_stability_df = calculate_par_metrics(  # noqa: F405
    df, another_group="task_name", par_threshold=0.7,
)
df_par['task_name'] = df_par['task_name'].apply(format_task_name)  # noqa: F405
df_par['author_year'] = df_par['author_year'].apply(format_author_year)  # noqa: F405

# ---------------------------------------------------------------------------
# 3.  Prepare pie-grid data for Panel B
# ---------------------------------------------------------------------------
plot_data = df_par.copy()
inconsistent_label = "Inconsistent"
plot_data['consistent_model'] = plot_data['consistent_model'].fillna(inconsistent_label)

unique_models = sorted(
    [m for m in plot_data['consistent_model'].unique() if m != inconsistent_label]
)
all_categories = unique_models + [inconsistent_label]

# Build author_year labels with N counts
n_counts = plot_data['author_year'].value_counts()
original_order = sorted(plot_data['author_year'].unique())

def get_fmt_label(year):
    return f"{year}\n(N={n_counts.get(year, 0)})"

label_map = {year: get_fmt_label(year) for year in original_order}
plot_data['author_year_fmt'] = plot_data['author_year'].map(label_map)
fmt_order = [get_fmt_label(year) for year in original_order]

# Task order (matching existing convention)
task_order = ['Flanker', 'Simon', 'Stroop']

# Color maps
color_map = model_colors.copy()
color_map[inconsistent_label] = '#FFFFFF'

# ---------------------------------------------------------------------------
# 4.  Build combined figure
#     Left (width 0.35): Panel A – bar chart
#     Right (width 0.65): Panel B – pie grid (3 rows × N_studies cols)
# ---------------------------------------------------------------------------
N_studies = len(fmt_order)
N_tasks = len(task_order)

fig = plt.figure(figsize=(11.5, 5.5))

# GridSpec: 1 row × 2 columns for main panels
from matplotlib.gridspec import GridSpec  # noqa: E402

gs_main = GridSpec(1, 2, figure=fig, width_ratios=[0.35, 0.65], wspace=0.15)

# ── Panel A: model stability bar chart ────────────────────────────────────
ax_a = fig.add_subplot(gs_main[0])
plot_bar_with_text(  # noqa: F405
    model_stability_df,
    x_col='winner_model',
    y_col='percentage',
    count_col='n_subj',
    edge_color=model_edge_colors,
    ax=ax_a,
    palette=model_colors,
    models_sorted=models_sorted,
)
ax_a.tick_params(axis='both', which='major', labelsize=12)
add_subplot_label(ax_a, 'A', x_offset=-0.18, y_offset=1.04, fontsize=18)  # noqa: F405
apply_nature_bar_axis_style(  # noqa: F405
    ax_a, xlabel="", ylabel="Percentage of subjects (%)",
    label_fontsize=15, tick_fontsize=12,
)

for patch in ax_a.patches:
    patch.set_zorder(3)

# ── Panel B: pie chart grid ───────────────────────────────────────────────
gs_pie = gs_main[1].subgridspec(
    N_tasks, N_studies, hspace=0.05, wspace=0.05,
)

def _filter_autopct(pct):
    return f'{pct:.0f}%' if pct > 10 else ''

ax_b = None
for i, task in enumerate(task_order):
    for j, study_fmt in enumerate(fmt_order):
        ax = fig.add_subplot(gs_pie[i, j])
        if i == 0 and j == 0:
            ax_b = ax

        # Column title (top row only)
        if i == 0:
            ax.set_title(study_fmt, fontsize=13, fontweight='bold', pad=4)

        # Row label (rightmost column only)
        if j == N_studies - 1:
            ax.text(
                1.3, 0.5, task,
                transform=ax.transAxes,
                fontsize=14, fontweight='bold', rotation=270,
                ha='center', va='center'
            )

        # Filter data for this cell
        cell_data = plot_data[
            (plot_data['task_name'] == task) &
            (plot_data['author_year_fmt'] == study_fmt)
        ]

        if cell_data.empty:
            ax.axis('off')
            continue

        counts = cell_data['consistent_model'].value_counts()
        if counts.empty:
            ax.axis('off')
            continue

        # Build colors in consistent order
        pie_colors = [color_map.get(cat, '#cccccc') for cat in counts.index]
        wedges, texts, autotexts = ax.pie(
            counts,
            colors=pie_colors,
            radius=1.0,
            startangle=90,
            wedgeprops={'edgecolor': 'gray', 'linewidth': 0.8, 'alpha': 1.0},
            autopct=_filter_autopct,
            textprops={'fontsize': 10, 'color': 'black'},
        )
        for at in autotexts:
            at.set_fontweight('bold')
        ax.axis('off')

add_subplot_label(  # noqa: F405
    ax_b, 'B',
    x_offset=-0.35, y_offset=1.08, fontsize=18,
)

# ── Shared legend ─────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(facecolor=model_colors[model],
                   edgecolor=model_edge_colors[model], label=model)
    for model in models_sorted if model in unique_models
]
handles.append(
    mpatches.Patch(facecolor='#ffffff', edgecolor='#cccccc',
                   linewidth=1.0, label='Inconsistent')
)
fig.legend(
    handles=handles,
    loc="upper center",
    ncol=len(handles),
    bbox_to_anchor=(0.5, 1.06),
    frameon=False,
    fontsize=13,
)

plt.tight_layout(rect=[0, 0, 1, 0.88])

# ---------------------------------------------------------------------------
# 5.  Save
# ---------------------------------------------------------------------------
retest_svg = f"../figs/31_model_comparision_retest_consistency_{metric}.svg"
plt.savefig(retest_svg, bbox_inches='tight')
strip_trailing_whitespace(retest_svg)
plt.savefig(
    f"../figs/31_model_comparision_retest_consistency_{metric}.png",
    dpi=300,
    bbox_inches='tight',
)
plt.close(fig)

print(f"Saved: {retest_svg}")
print(f"Saved: ../figs/31_model_comparision_retest_consistency_{metric}.png")
