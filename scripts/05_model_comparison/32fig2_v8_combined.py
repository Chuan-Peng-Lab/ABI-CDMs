#!/usr/bin/env python
# coding: utf-8
"""
Standalone script: generate v8 Fig 2 – combined model comparison,
cross-task consistency, and retest consistency.

Two layouts supported:
  3x2 – 3 rows × 2 columns (original portrait)
  2x3 – 2 rows × 3 columns (landscape)

Usage:
    python 32fig2_v8_combined.py                  # default: 2x3
    python 32fig2_v8_combined.py --layout 3x2     # 3x2 only
    python 32fig2_v8_combined.py --layout 2x3     # 2x3 only
    python 32fig2_v8_combined.py --layout both    # both layouts
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

import sys
from nsbi_module.utils_ind_diff import (
    add_subplot_label,
    apply_nature_bar_axis_style,
    calc_best_model_proportion,
    calculate_par_metrics,
    format_task_id,
    format_task_name,
    get_best_model_by_metric,
    order_groups_by_model_proportions,
    plot_bar_with_text,
    plot_stacked_consistency,
    plot_stacked_model_proportion_bar,
    rank_models_by_metric,
    summarize_model_performance,
)
from nsbi_module.study_labels import format_study_label, format_study_abbrev

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["Arial", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 8.5
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 8.5
plt.rcParams["ytick.labelsize"] = 8.5
plt.rcParams["legend.fontsize"] = 10
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
# 1. Constants
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
full_color_map = model_colors.copy()
full_color_map["Inconsistent"] = "#ffffff"

FONT = {
    "panel": 13,
    "section": 12,
    "legend": 10,
    "axis_label": 10,
    "tick": 8.5,
    "stack_label": 7,
    "bar_label": 8,
    "n_label": 8,
    "pie_pct": 7,
    "pie_col": 8,
    "pie_row": 9,
}


def set_axis_text_sizes(ax, tick_size=None, text_size=None):
    """Keep locally generated annotations visually consistent."""
    if tick_size is None:
        tick_size = FONT["tick"]
    if text_size is None:
        text_size = FONT["bar_label"]
    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    for text in ax.texts:
        if text.get_text().startswith("N="):
            text.set_fontsize(FONT["n_label"])
        else:
            text.set_fontsize(text_size)


# ---------------------------------------------------------------------------
# 2. Shared data preparation for model comparison (A & B)
# ---------------------------------------------------------------------------
model_prediction_indices_long = pd.read_csv("../03_fitting/23model_prediction_indices_dmc_v2.csv")
model_prediction_indices_long = model_prediction_indices_long.query(
    "author_year != 'lee2025'"
)
model_prediction_indices_long = model_prediction_indices_long[
    model_prediction_indices_long["model"].isin(models_sorted)
].copy()
model_prediction_indices_long = rank_models_by_metric(
    model_prediction_indices_long, metric=metric
)

summary_df = summarize_model_performance(
    model_prediction_indices_long, group_col="task_id", metric=metric
)

best_model = get_best_model_by_metric(model_prediction_indices_long, metric=metric)
best_model["author_year"] = best_model["author_year"].apply(format_study_label)
best_model["task_name"] = best_model["task_name"].apply(format_task_name)
best_model["task_id"] = best_model["task_id"].apply(format_task_id)
best_model["task_id"] = best_model["task_id"].str.replace(
    "Reymermet 2018", "Rey-Mermet 2018", regex=False
)

best_model_proportion = calc_best_model_proportion(
    best_model, group_cols=["task_id", "task_name", "author_year"]
)

# A: task proportions
task_proportion = calc_best_model_proportion(
    best_model, group_cols=["task_name"]
)
task_order = order_groups_by_model_proportions(
    task_proportion,
    group_col="task_name",
    sort_models=("DSTP", "DMC"),
)

# B: dataset proportions (abbreviated labels to avoid rotation)
dataset_proportion = calc_best_model_proportion(
    best_model, group_cols=["author_year"]
)
dataset_proportion["author_year"] = dataset_proportion["author_year"].apply(format_study_abbrev)
dataset_order = order_groups_by_model_proportions(
    dataset_proportion,
    group_col="author_year",
    sort_models=("DSTP", "DMC"),
)


# ---------------------------------------------------------------------------
# 3. Cross-task consistency data (C & D)
# ---------------------------------------------------------------------------
selected_cross_task_studies = [
    "eisenberg2019",
    "whitehead2019",
    "ulrich2015",
    "clayson2025",
    "reymermet2018",
    "hedge2018",
]
model_pred_cross_task = model_prediction_indices_long[
    model_prediction_indices_long["author_year"].isin(selected_cross_task_studies)
]
model_pred_cross_task = model_pred_cross_task.query("task_id != 'hedge2018simon'")

best_model_cross_task = get_best_model_by_metric(
    model_pred_cross_task, metric=metric
)

df_cross = best_model_cross_task
df_par_cross, author_summary_cross, model_stability_df_cross = calculate_par_metrics(
    df_cross, group_col="author_year"
)
df_par_cross["author_year"] = df_par_cross["author_year"].apply(format_study_abbrev)


# ---------------------------------------------------------------------------
# 4. Retest consistency data (E & F)
# ---------------------------------------------------------------------------
model_prediction_indices_retest = pd.read_csv("../03_fitting/23model_prediction_indices_retest.csv")
best_model_retest = get_best_model_by_metric(
    model_prediction_indices_retest,
    group=["subject_id", "task_id", "author_year", "task_name", "session_id"],
    metric=metric,
)

df_retest = best_model_retest
df_par_retest, _, model_stability_df_retest = calculate_par_metrics(
    df_retest, another_group="task_name", par_threshold=0.7,
)
df_par_retest["task_name"] = df_par_retest["task_name"].apply(format_task_name)
df_par_retest["author_year"] = df_par_retest["author_year"].apply(format_study_label)

# Pie-grid data preparation
plot_data_retest = df_par_retest.copy()
inconsistent_label = "Inconsistent"
plot_data_retest["consistent_model"] = plot_data_retest["consistent_model"].fillna(
    inconsistent_label
)

unique_models = sorted(
    [m for m in plot_data_retest["consistent_model"].unique()
     if m != inconsistent_label]
)
all_categories = unique_models + [inconsistent_label]

n_counts = plot_data_retest["author_year"].value_counts()
original_order = sorted(plot_data_retest["author_year"].unique())


def get_fmt_label(year):
    abbrev = format_study_abbrev(year)
    return f"{abbrev}\n(N={n_counts.get(year, 0)})"


label_map = {year: get_fmt_label(year) for year in original_order}
plot_data_retest["author_year_fmt"] = plot_data_retest["author_year"].map(label_map)
fmt_order = [get_fmt_label(year) for year in original_order]

task_order_retest = ["Flanker", "Simon", "Stroop"]


# ---------------------------------------------------------------------------
# 5. Shared helpers
# ---------------------------------------------------------------------------

def _filter_autopct(pct):
    return f"{pct:.0f}%" if pct > 10 else ""


def _add_shared_legend(fig, loc, bbox_to_anchor):
    """Add the 5-item legend at the specified position."""
    handles = [
        mpatches.Patch(facecolor=model_colors[model],
                       edgecolor=model_edge_colors[model], label=model)
        for model in models_sorted
    ]
    handles.append(
        mpatches.Patch(facecolor="#ffffff", edgecolor="#cccccc",
                       linewidth=1.0, label="Inconsistent")
    )
    fig.legend(
        handles=handles,
        loc=loc,
        ncol=5,
        bbox_to_anchor=bbox_to_anchor,
        frameon=False,
        fontsize=FONT["legend"],
    )


def _draw_pie_grid(fig, f_gs_cell, f_pos):
    """Draw the retest consistency pie grid (panel F) inside a given cell."""
    N_studies = len(fmt_order)
    N_tasks = len(task_order_retest)
    gs_pie = f_gs_cell.subgridspec(
        N_tasks + 1,
        N_studies,
        height_ratios=[0.22, 1, 1, 1],
        hspace=0.08,
        wspace=0.04,
    )

    # F panel label
    fig.text(
        f_pos.x0 - 0.012,
        f_pos.y1 - 0.035,
        "F",
        fontsize=FONT["panel"],
        fontweight="bold",
        ha="right",
        va="bottom",
    )

    for i, task in enumerate(task_order_retest):
        for j, study_fmt in enumerate(fmt_order):
            ax = fig.add_subplot(gs_pie[i + 1, j])

            if i == 0:
                ax.set_title(
                    study_fmt,
                    fontsize=FONT["axis_label"],
                    fontweight="bold",
                    pad=2,
                    linespacing=0.9,
                )

            if j == N_studies - 1:
                row_pos = ax.get_position(fig)
                fig.text(
                    f_pos.x1 + 0.018,
                    (row_pos.y0 + row_pos.y1) / 2,
                    task,
                    fontsize=FONT["pie_row"],
                    fontweight="bold",
                    rotation=270,
                    ha="center",
                    va="center",
                )

            cell_data = plot_data_retest[
                (plot_data_retest["task_name"] == task) &
                (plot_data_retest["author_year_fmt"] == study_fmt)
            ]

            if cell_data.empty:
                ax.axis("off")
                continue

            counts = cell_data["consistent_model"].value_counts()
            if counts.empty:
                ax.axis("off")
                continue

            pie_colors = [full_color_map.get(cat, "#cccccc") for cat in counts.index]
            wedges, texts, autotexts = ax.pie(
                counts,
                colors=pie_colors,
                radius=1.0,
                startangle=90,
                wedgeprops={"edgecolor": "gray", "linewidth": 0.8, "alpha": 1.0},
                autopct=_filter_autopct,
                textprops={"fontsize": FONT["pie_pct"], "color": "black"},
            )
            for at in autotexts:
                at.set_fontweight("bold")
            ax.axis("off")


# ---------------------------------------------------------------------------
# 6. Figure builders
# ---------------------------------------------------------------------------

def build_figure_3x2():
    """3 rows × 2 columns (original portrait layout).

    Row 0: A (left) | B (right)
    Row 1: C (left) | D (right)
    Row 2: E (left) | F (right)
    """
    fig = plt.figure(figsize=(6, 9))

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        width_ratios=[4, 6],
        hspace=0.35,
        wspace=0.22,
        left=0.075,
        right=0.93,
        top=0.90,
        bottom=0.065,
    )

    # ── Row 0: A & B ──────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    plot_stacked_model_proportion_bar(
        task_proportion, colors=model_colors, group_col="task_name",
        models_sorted=models_sorted, group_order=task_order,
        edge_color=model_edge_colors, xlabel="", ylabel="", ax=ax_a,
        label_threshold=15, label_fontsize=FONT["axis_label"],
        tick_fontsize=FONT["tick"], segment_label_fontsize=8,
        show_n_prefix=False,
    )
    add_subplot_label(ax_a, "A", x_offset=-0.17, y_offset=1.04, fontsize=FONT["panel"])

    ax_b = fig.add_subplot(gs[0, 1])
    plot_stacked_model_proportion_bar(
        dataset_proportion, colors=model_colors, group_col="author_year",
        models_sorted=models_sorted, group_order=dataset_order,
        edge_color=model_edge_colors, xlabel="", ylabel="", ax=ax_b,
        label_threshold=15, label_fontsize=FONT["axis_label"],
        tick_fontsize=FONT["tick"], segment_label_fontsize=FONT["stack_label"],
        show_n_prefix=False,
    )
    add_subplot_label(ax_b, "B", x_offset=-0.10, y_offset=1.04, fontsize=FONT["panel"])

    for ax_ab in (ax_a, ax_b):
        ax_ab.set_yticks(np.arange(0, 100, 25))
        ax_ab.tick_params(axis="x", labelsize=FONT["axis_label"])

    # ── Row 1: C & D ──────────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    plot_bar_with_text(
        model_stability_df_cross, x_col="winner_model", y_col="percentage",
        count_col="n_subj", edge_color=model_edge_colors, ax=ax_c,
        palette=model_colors, models_sorted=models_sorted, show_n_prefix=False,
    )
    apply_nature_bar_axis_style(
        ax_c, xlabel="", ylabel="",
        label_fontsize=FONT["axis_label"], tick_fontsize=FONT["tick"],
    )
    set_axis_text_sizes(ax_c)
    ax_c.set_yticks([0, 7, 15, 25])
    add_subplot_label(ax_c, "C", x_offset=-0.17, y_offset=1.04, fontsize=FONT["panel"])
    ax_c.tick_params(axis="x", labelsize=FONT["axis_label"])
    for patch in ax_c.patches:
        patch.set_zorder(3)

    ax_d = fig.add_subplot(gs[1, 1])
    plot_stacked_consistency(
        df_par_cross, group_col="author_year", custom_colors=full_color_map,
        ax=ax_d, ylabel=None, alpha=1.0, label_threshold=10,
        edge_color=model_edge_colors, show_legend=False,
        models_sorted=models_sorted.copy(),
    )
    apply_nature_bar_axis_style(
        ax_d, xlabel="", ylabel="",
        label_fontsize=FONT["axis_label"], tick_fontsize=FONT["tick"],
    )
    set_axis_text_sizes(ax_d, text_size=FONT["stack_label"])
    add_subplot_label(ax_d, "D", x_offset=-0.10, y_offset=1.04, fontsize=FONT["panel"])
    ax_d.tick_params(axis="x", labelsize=FONT["axis_label"])

    # ── Row 2: E & F ──────────────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[2, 0])
    plot_bar_with_text(
        model_stability_df_retest, x_col="winner_model", y_col="percentage",
        count_col="n_subj", edge_color=model_edge_colors, ax=ax_e,
        palette=model_colors, models_sorted=models_sorted, show_n_prefix=False,
    )
    apply_nature_bar_axis_style(
        ax_e, xlabel="", ylabel="",
        label_fontsize=FONT["axis_label"], tick_fontsize=FONT["tick"],
    )
    set_axis_text_sizes(ax_e)
    ax_e.set_yticks([0, 7, 15, 25, 35])
    add_subplot_label(ax_e, "E", x_offset=-0.17, y_offset=1.04, fontsize=FONT["panel"])
    ax_e.tick_params(axis="x", labelsize=FONT["axis_label"])
    for patch in ax_e.patches:
        patch.set_zorder(3)

    # F pie grid
    f_pos = gs[2, 1].get_position(fig)
    _draw_pie_grid(fig, gs[2, 1], f_pos)

    # ── Row titles ────────────────────────────────────────────────────────
    def _row_title_y(row_idx, pad=0.02):
        return gs[row_idx, 0].get_position(fig).y1 + pad

    fig.text(
        0.5, _row_title_y(0), "Proportions of best-fitting models",
        fontsize=FONT["section"], fontweight="bold", ha="center", va="bottom",
    )
    fig.text(
        0.5, _row_title_y(1), "Cross-task consistency of winning models",
        fontsize=FONT["section"], fontweight="bold", ha="center", va="bottom",
    )
    fig.text(
        0.5, _row_title_y(2), "Cross-temporal consistency of winning models",
        fontsize=FONT["section"], fontweight="bold", ha="center", va="bottom",
    )

    # ── Legend at top ─────────────────────────────────────────────────────
    _add_shared_legend(fig, loc="upper center", bbox_to_anchor=(0.5, 0.975))

    return fig


def build_figure_2x3():
    """2 rows × 3 columns (landscape layout).

    Row 0: A (task freq) | C (cross-task stability) | E (retest stability)
    Row 1: B (dataset freq) | D (cross-task stacked) | F (retest pie grid)
    """
    fig = plt.figure(figsize=(10.5, 6.5))

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        hspace=0.38,
        wspace=0.22,
        left=0.055,
        right=0.95,
        top=0.91,
        bottom=0.10,
    )

    # ── Row 0 col 0: A ──────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    plot_stacked_model_proportion_bar(
        task_proportion, colors=model_colors, group_col="task_name",
        models_sorted=models_sorted, group_order=task_order,
        edge_color=model_edge_colors, xlabel="", ylabel="", ax=ax_a,
        label_threshold=15, label_fontsize=FONT["axis_label"],
        tick_fontsize=FONT["tick"], segment_label_fontsize=8,
        show_n_prefix=False,
    )
    add_subplot_label(ax_a, "A", x_offset=-0.14, y_offset=1.04, fontsize=FONT["panel"])
    ax_a.set_yticks(np.arange(0, 100, 25))
    ax_a.tick_params(axis="x", labelsize=FONT["axis_label"])

    # ── Row 0 col 1: C ──────────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 1])
    plot_bar_with_text(
        model_stability_df_cross, x_col="winner_model", y_col="percentage",
        count_col="n_subj", edge_color=model_edge_colors, ax=ax_c,
        palette=model_colors, models_sorted=models_sorted, show_n_prefix=False,
    )
    apply_nature_bar_axis_style(
        ax_c, xlabel="", ylabel="",
        label_fontsize=FONT["axis_label"], tick_fontsize=FONT["tick"],
    )
    set_axis_text_sizes(ax_c)
    ax_c.set_yticks([0, 7, 15, 25])
    add_subplot_label(ax_c, "C", x_offset=-0.10, y_offset=1.04, fontsize=FONT["panel"])
    ax_c.tick_params(axis="x", labelsize=FONT["axis_label"])
    for patch in ax_c.patches:
        patch.set_zorder(3)

    # ── Row 0 col 2: E ──────────────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[0, 2])
    plot_bar_with_text(
        model_stability_df_retest, x_col="winner_model", y_col="percentage",
        count_col="n_subj", edge_color=model_edge_colors, ax=ax_e,
        palette=model_colors, models_sorted=models_sorted, show_n_prefix=False,
    )
    apply_nature_bar_axis_style(
        ax_e, xlabel="", ylabel="",
        label_fontsize=FONT["axis_label"], tick_fontsize=FONT["tick"],
    )
    set_axis_text_sizes(ax_e)
    ax_e.set_yticks([0, 7, 15, 25, 35])
    add_subplot_label(ax_e, "E", x_offset=-0.08, y_offset=1.04, fontsize=FONT["panel"])
    ax_e.tick_params(axis="x", labelsize=FONT["axis_label"])
    for patch in ax_e.patches:
        patch.set_zorder(3)

    # ── Row 1 col 0: B ──────────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[1, 0])
    plot_stacked_model_proportion_bar(
        dataset_proportion, colors=model_colors, group_col="author_year",
        models_sorted=models_sorted, group_order=dataset_order,
        edge_color=model_edge_colors, xlabel="", ylabel="", ax=ax_b,
        label_threshold=15, label_fontsize=FONT["axis_label"],
        tick_fontsize=FONT["tick"], segment_label_fontsize=FONT["stack_label"],
        show_n_prefix=False,
    )
    add_subplot_label(ax_b, "B", x_offset=-0.14, y_offset=1.04, fontsize=FONT["panel"])
    ax_b.set_yticks(np.arange(0, 100, 25))
    ax_b.tick_params(axis="x", labelsize=FONT["axis_label"])

    # ── Row 1 col 1: D ──────────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    plot_stacked_consistency(
        df_par_cross, group_col="author_year", custom_colors=full_color_map,
        ax=ax_d, ylabel=None, alpha=1.0, label_threshold=10,
        edge_color=model_edge_colors, show_legend=False,
        models_sorted=models_sorted.copy(),
    )
    apply_nature_bar_axis_style(
        ax_d, xlabel="", ylabel="",
        label_fontsize=FONT["axis_label"], tick_fontsize=FONT["tick"],
    )
    set_axis_text_sizes(ax_d, text_size=FONT["stack_label"])
    add_subplot_label(ax_d, "D", x_offset=-0.10, y_offset=1.04, fontsize=FONT["panel"])
    ax_d.tick_params(axis="x", labelsize=FONT["axis_label"])

    # ── Row 1 col 2: F ──────────────────────────────────────────────────────
    f_pos = gs[1, 2].get_position(fig)
    _draw_pie_grid(fig, gs[1, 2], f_pos)

    # ── Column titles ──────────────────────────────────────────────────────
    def _col_title_x(col_idx):
        pos = gs[0, col_idx].get_position(fig)
        return (pos.x0 + pos.x1) / 2

    col_title_y = gs[0, 0].get_position(fig).y1 + 0.02

    fig.text(
        _col_title_x(0), col_title_y,
        "Proportions of\nbest-fitting models",
        fontsize=FONT["section"], fontweight="bold", ha="center", va="bottom",
    )
    fig.text(
        _col_title_x(1), col_title_y,
        "Cross-task consistency\nof winning models",
        fontsize=FONT["section"], fontweight="bold", ha="center", va="bottom",
    )
    fig.text(
        _col_title_x(2), col_title_y,
        "Cross-temporal consistency\nof winning models",
        fontsize=FONT["section"], fontweight="bold", ha="center", va="bottom",
    )

    # ── Legend at bottom ───────────────────────────────────────────────────
    _add_shared_legend(fig, loc="lower center", bbox_to_anchor=(0.5, 0.0))

    return fig


# ---------------------------------------------------------------------------
# 7. Save & CLI
# ---------------------------------------------------------------------------

def save_figure(fig, layout_name):
    """Save figure to SVG and PNG with layout-aware filenames."""
    out_svg = f"../figs/32fig2_v8_combined_{layout_name}.svg"
    out_png = f"../figs/32fig2_v8_combined_{layout_name}.png"

    plt.savefig(out_svg, bbox_inches="tight")
    strip_trailing_whitespace(out_svg)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_svg}")
    print(f"Saved: {out_png}")


BUILDERS = {
    "3x2": build_figure_3x2,
    "2x3": build_figure_2x3,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate v8 Fig 2 with selectable layout."
    )
    parser.add_argument(
        "--layout", "-l",
        choices=["3x2", "2x3", "both"],
        default="2x3",
        help="Which layout(s) to generate (default: 2x3).",
    )
    args = parser.parse_args()

    if args.layout == "both":
        for name in ["2x3", "3x2"]:
            print(f"\n--- Building layout: {name} ---")
            fig = BUILDERS[name]()
            save_figure(fig, name)
    else:
        fig = BUILDERS[args.layout]()
        save_figure(fig, args.layout)
