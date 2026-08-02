#!/usr/bin/env python
"""Build v8 Fig4 by composing Fig9b, Fig8a, Fig8b, and Fig10 functions.

The compositor uses data-level outputs from R for the reliability panels and
recreates the existing factor-space panels with matplotlib/seaborn functions.
It does not stitch previously exported SVG/PNG files.
"""

from __future__ import annotations

import argparse
import subprocess
from itertools import combinations
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy import stats
from scipy.stats import gaussian_kde

from nsbi_module.project_paths import INTERMEDIATE_DIR, MAIN_FIGURES_DIR, REPO_ROOT, ensure_output_directories

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = MAIN_FIGURES_DIR
EXPORT_DIR = INTERMEDIATE_DIR / "figure_04_reliability"
R_EXPORT_SCRIPT = SCRIPT_DIR / "export_figure_04_reliability_data.R"

FACTOR_CAUTION = "Decision Caution"
FACTOR_NDT = "Non-decision time"
FACTOR_EFFICIENCY = "Processing Efficiency"
FACTOR_INHIBITION = "Inhibitory process"
FACTOR_ORDER = [
    "Non-decision time",
    "Processing Efficiency",
    "Decision Caution",
    "Inhibitory process",
]

FACTOR_COLORS = {
    "Processing Efficiency": "#8491B4",
    "Decision Caution": "#3b6cb3",
    "Non-decision time": "#7E6148",
    "Inhibitory process": "#0099B4",
}
TASK_COLORS = {
    "Flanker": "#b0d97c",
    "Stroop": "#95d8c3",
    "Simon": "#81cef0",
}
SOURCE_COLORS = {
    "Labs": "#81cef0",
    "Tasks": "#95d8c3",
}
STUDY_ABBREVIATIONS = {
    "Clayson2025": "C25",
    "Eisenberg2019": "E19",
    "Hedge2018": "H18",
    "Kucina2023": "K23",
    "Lee2025": "L25",
    "Reymermet2018": "R18",
    "Ulrich2015": "U15",
    "Whitehead2019": "W19",
}

PALETTE = sns.color_palette("husl", 20)
PALETTE_TASK = [PALETTE[0], PALETTE[12], PALETTE[14]]
PALETTE_LAB = sns.color_palette("Set2", 10)[1:9]

LABEL_FONTSIZE = 9
TICK_FONTSIZE = 8
LEGEND_FONTSIZE = 8
POINT_SIZE = 24
ALPHA_SCATTER = 0.24


def run_r_export() -> None:
    """Regenerate the CSV inputs used by the top-row reliability panels."""
    subprocess.run(
        ["Rscript", str(R_EXPORT_SCRIPT)],
        cwd=REPO_ROOT,
        check=True,
    )


def require_reliability_exports() -> None:
    """Validate that all R-exported inputs are present."""
    required = [
        EXPORT_DIR / "fig9b_cross_task_icc_draws.csv",
        EXPORT_DIR / "fig8a_temporal_icc_subgroups.csv",
        EXPORT_DIR / "fig8b_temporal_sd_draws.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"Missing reliability export(s): {names}")


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.05) -> None:
    """Add a compact bold panel label just outside an axes."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def clean_legend(legend) -> None:
    """Remove legend frame/background while preserving labels and handles."""
    if legend is None:
        return
    legend.set_frame_on(False)
    frame = legend.get_frame()
    frame.set_facecolor("none")
    frame.set_edgecolor("none")
    frame.set_alpha(0)


def configure_style(font_scale: float, point_size: float | None = None) -> None:
    """Scale plot text and scatter marker size from CLI parameters."""
    global LABEL_FONTSIZE, TICK_FONTSIZE, LEGEND_FONTSIZE, POINT_SIZE

    LABEL_FONTSIZE = max(1, LABEL_FONTSIZE * font_scale)
    TICK_FONTSIZE = max(1, TICK_FONTSIZE * font_scale)
    LEGEND_FONTSIZE = max(1, LEGEND_FONTSIZE * font_scale)
    if point_size is not None:
        POINT_SIZE = point_size


def add_row_title(
    fig: plt.Figure,
    axes: tuple[plt.Axes, plt.Axes, plt.Axes],
    title: str,
    y_offset: float,
) -> None:
    """Place a row title centered above a three-panel row."""
    boxes = [ax.get_position() for ax in axes]
    left = min(box.x0 for box in boxes)
    right = max(box.x1 for box in boxes)
    top = max(box.y1 for box in boxes)
    fig.text(
        (left + right) / 2,
        top + y_offset,
        title,
        ha="center",
        va="bottom",
        fontsize=LABEL_FONTSIZE + 1,
        fontweight="bold",
    )


def kde_ridge(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    value_col: str,
    group_col: str,
    group_order: list[str],
    colors: dict[str, str],
    xlabel: str,
    xlim: tuple[float, float],
    scale: float = 0.75,
    interval_width: float = 0.95,
    ytick_label_map: dict[str, str] | None = None,
) -> None:
    """Draw compact horizontal ridgelines with median and interval markers."""
    grid = np.linspace(xlim[0], xlim[1], 300)
    y_positions = np.arange(len(group_order))

    for y_pos, group in zip(y_positions, group_order):
        values = data.loc[data[group_col] == group, value_col].dropna().to_numpy()
        if values.size < 3:
            continue
        kde = gaussian_kde(values)
        density = kde(grid)
        density = density / density.max() * scale
        color = colors[group]

        ax.fill_between(
            grid,
            y_pos,
            y_pos + density,
            color=color,
            alpha=0.72,
            linewidth=0,
        )
        ax.plot(grid, y_pos + density, color="white", linewidth=0.9)

        lo = np.quantile(values, (1 - interval_width) / 2)
        hi = np.quantile(values, 1 - (1 - interval_width) / 2)
        med = np.median(values)
        interval_y = y_pos - 0.08
        ax.hlines(interval_y, lo, hi, color="black", linewidth=1.0)
        ax.plot(med, interval_y, "o", color="black", markersize=3)

    ax.set_xlim(*xlim)
    ax.set_yticks(y_positions)
    ytick_labels = [ytick_label_map.get(group, group) if ytick_label_map else group for group in group_order]
    ax.set_yticklabels(ytick_labels, fontsize=TICK_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e8e8e8", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True)


def plot_fig9b_icc_ridge(ax: plt.Axes, icc_draws: pd.DataFrame) -> None:
    """Plot Fig9b: posterior distributions of cross-task ICC."""
    med_order = (
        icc_draws.groupby("Factor_Type")["ICC"]
        .median()
        .sort_values()
        .index.tolist()
    )
    kde_ridge(
        ax,
        icc_draws,
        value_col="ICC",
        group_col="Factor_Type",
        group_order=med_order,
        colors=FACTOR_COLORS,
        xlabel="Cross-task ICC",
        xlim=(-0.03, 0.30),
        scale=0.78,
        ytick_label_map={
            "Non-decision time": "Non-decision\ntime",
            "Processing Efficiency": "Processing\nEfficiency",
            "Decision Caution": "Decision\nCaution",
            "Inhibitory process": "Inhibitory\nprocess",
        },
    )


def plot_fig8a_temporal_icc(fig: plt.Figure, cell, subgroup_iccs: pd.DataFrame) -> None:
    """Plot Fig8a as four compact forest mini-panels inside one grid cell."""
    inner = GridSpecFromSubplotSpec(4, 1, subplot_spec=cell, hspace=0.34)
    first_ax = None

    for idx, factor in enumerate(FACTOR_ORDER):
        ax = fig.add_subplot(inner[idx, 0])
        if first_ax is None:
            first_ax = ax

        factor_df = subgroup_iccs.loc[subgroup_iccs["Factor_Type"] == factor].copy()
        tasks = sorted(factor_df["task_name"].dropna().unique())
        y_lookup = {task: pos for pos, task in enumerate(tasks)}
        rng = np.random.default_rng(20260218 + idx)
        y = factor_df["task_name"].map(y_lookup).to_numpy(dtype=float)
        y = y + rng.normal(0, 0.055, size=len(y))

        ax.axvline(
            factor_df["ICC"].mean(),
            color=FACTOR_COLORS[factor],
            linewidth=1.1,
            alpha=0.9,
        )
        ax.scatter(
            factor_df["ICC"],
            y,
            s=14,
            color=FACTOR_COLORS[factor],
            alpha=0.62,
            edgecolor="none",
        )
        ax.set_xlim(0, 0.8)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(tasks, fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", length=0, pad=1)
        ax.set_title(factor, loc="left", fontsize=8, fontweight="bold", pad=1)
        ax.grid(axis="x", color="#e8e8e8", linewidth=0.6)
        ax.grid(axis="y", visible=False)
        sns.despine(ax=ax, left=True)

        if idx < len(FACTOR_ORDER) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Cross-temporal ICC", fontsize=LABEL_FONTSIZE)

    if first_ax is not None:
        add_panel_label(first_ax, "C", x=-0.34, y=1.22)


def plot_fig8b_sd_comparison(ax: plt.Axes, sd_draws: pd.DataFrame) -> None:
    """Plot Fig8b: posterior SD distributions for task and lab effects."""
    df = sd_draws.copy()
    source_map = {"Tasks": "Across Tasks", "Labs": "Across Studies"}
    df["Source"] = df["Source"].map(source_map)
    order = ["Across Tasks", "Across Studies"]
    palette = {
        "Across Tasks": SOURCE_COLORS["Tasks"],
        "Across Studies": SOURCE_COLORS["Labs"],
    }

    sns.violinplot(
        data=df,
        x="Source",
        y="Sigma",
        hue="Source",
        order=order,
        palette=palette,
        inner=None,
        cut=0,
        linewidth=0,
        alpha=0.70,
        legend=False,
        ax=ax,
    )

    for idx, source in enumerate(order):
        values = df.loc[df["Source"] == source, "Sigma"].dropna().to_numpy()
        lo, hi = np.quantile(values, [0.025, 0.975])
        med = np.median(values)
        ax.vlines(idx, lo, hi, color="black", linewidth=1.0)
        ax.plot(idx, med, "o", color="black", markersize=3)

    ax.set_xlabel("")
    ax.set_ylabel("Estimated SD", fontsize=LABEL_FONTSIZE)
    ax.set_ylim(0, 0.8)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE * 0.95, rotation=8)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    sns.despine(trim=True, ax=ax)


def plot_factor_scatter(
    df: pd.DataFrame,
    x_factor: str,
    y_factor: str,
    hue_col: str,
    palette,
    ax: plt.Axes,
    show_centroids: bool = True,
    max_points_per_group: int | None = None,
) -> tuple[plt.Axes, list[mlines.Line2D]]:
    """Draw a factor-space scatter panel with centroid legend handles."""
    if isinstance(df[hue_col].dtype, pd.CategoricalDtype):
        hue_order = df[hue_col].cat.categories.tolist()
    else:
        hue_order = sorted(df[hue_col].unique())

    colours = sns.color_palette(palette, n_colors=len(hue_order))
    df_plot = df.copy()
    if max_points_per_group is not None and max_points_per_group > 0:
        df_plot = df_plot.groupby(hue_col, group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max_points_per_group), random_state=42)
        )

    sns.scatterplot(
        data=df_plot,
        x=x_factor,
        y=y_factor,
        hue=hue_col,
        hue_order=hue_order,
        palette=colours,
        alpha=ALPHA_SCATTER,
        s=POINT_SIZE,
        linewidth=0,
        ax=ax,
        legend=False,
    )

    ax.axhline(y=0, color="grey", linestyle="--", linewidth=1.1, alpha=0.45)
    ax.axvline(x=0, color="grey", linestyle="--", linewidth=1.1, alpha=0.45)

    legend_handles: list[mlines.Line2D] = []
    if show_centroids:
        centroids = df.groupby(hue_col)[[x_factor, y_factor]].mean().reindex(hue_order)
        for (group, row), colour in zip(centroids.iterrows(), colours):
            ax.scatter(
                row[x_factor],
                row[y_factor],
                marker="D",
                s=34,
                color=colour,
                edgecolors="black",
                linewidths=0.55,
                zorder=5,
            )
            legend_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    marker="D",
                    linestyle="None",
                    markerfacecolor=colour,
                    markeredgecolor="black",
                    markeredgewidth=0.55,
                    markersize=5.5,
                    label=str(group),
                )
            )

    ax.set_xlabel(x_factor, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(y_factor, fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    sns.despine(ax=ax)
    return ax, legend_handles


def compute_centroid_distances(df: pd.DataFrame, x_factor: str, y_factor: str, group_col: str) -> pd.DataFrame:
    """Calculate pairwise Euclidean distances between group centroids."""
    centroids = df.groupby(group_col)[[x_factor, y_factor]].mean()
    labels = centroids.index.tolist()
    records = []
    for (i1, lbl1), (i2, lbl2) in combinations(enumerate(labels), 2):
        p1 = centroids.iloc[i1].values
        p2 = centroids.iloc[i2].values
        records.append(
            {
                "Grouping": group_col,
                "Group_A": lbl1,
                "Group_B": lbl2,
                "Distance": np.linalg.norm(p1 - p2),
            }
        )
    return pd.DataFrame(records)


def plot_distance_comparison(distance_df: pd.DataFrame, ax: plt.Axes) -> plt.Axes:
    """Draw the task-vs-lab centroid distance panel."""
    df = distance_df.copy()
    df["Grouping"] = df["Grouping"].map(
        {"task_name": "Across Tasks", "author_year": "Across Studies"}
    )
    sns.boxplot(
        x="Grouping",
        y="Distance",
        hue="Grouping",
        data=df,
        palette=["#95d8c3", "#81cef0"],
        width=0.5,
        linewidth=0.9,
        fliersize=0,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        x="Grouping",
        y="Distance",
        data=df,
        color=".25",
        size=2.4,
        alpha=0.55,
        jitter=True,
        ax=ax,
    )

    groups = df["Grouping"].dropna().unique()
    if len(groups) == 2:
        group1 = df.loc[df["Grouping"] == groups[0], "Distance"]
        group2 = df.loc[df["Grouping"] == groups[1], "Distance"]
        _, p_val = stats.mannwhitneyu(group1, group2)
        text = "ns" if p_val > 0.05 else "*" if p_val > 0.01 else "**" if p_val > 0.001 else "***"
        y_max = df["Distance"].max()
        y_min = df["Distance"].min()
        y_range = max(y_max - y_min, 1e-6)
        y_h = y_max + y_range * 0.08
        y_text = y_h + y_range * 0.03
        ax.plot([0, 0, 1, 1], [y_h, y_h + y_range * 0.03, y_h + y_range * 0.03, y_h], lw=1.0, c="k")
        p_text = f"{p_val:.1e}" if p_val < 0.001 else f"{p_val:.3f}"
        ax.text(0.5, y_text, f"{text}\np={p_text}", ha="center", va="bottom", fontsize=TICK_FONTSIZE * 1.05)
        ax.set_ylim(top=y_text + y_range * 0.2)

    ax.set_xlabel("")
    ax.set_ylabel("Euclidean distance", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE * 0.95, rotation=8)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    sns.despine(trim=True, ax=ax)
    return ax


def plot_factor_row(
    axes: tuple[plt.Axes, plt.Axes, plt.Axes],
    df: pd.DataFrame,
    x_factor: str,
    y_factor: str,
    labels: tuple[str, str, str],
) -> tuple[list[mlines.Line2D], list[mlines.Line2D]]:
    """Draw one row of the Fig10 factor-space panels."""
    ax_task, ax_lab, ax_dist = axes
    _, task_handles = plot_factor_scatter(
        df,
        x_factor,
        y_factor,
        hue_col="task_name",
        palette=PALETTE_TASK,
        ax=ax_task,
        show_centroids=True,
    )
    add_panel_label(ax_task, labels[0])

    _, lab_handles = plot_factor_scatter(
        df,
        x_factor,
        y_factor,
        hue_col="author_year",
        palette=PALETTE_LAB,
        ax=ax_lab,
        show_centroids=True,
    )
    add_panel_label(ax_lab, labels[1])

    dist_task = compute_centroid_distances(df, x_factor, y_factor, "task_name")
    dist_lab = compute_centroid_distances(df, x_factor, y_factor, "author_year")
    dist_all = pd.concat([dist_task, dist_lab], ignore_index=True)
    plot_distance_comparison(dist_all, ax=ax_dist)
    add_panel_label(ax_dist, labels[2], x=-0.18)

    return task_handles, lab_handles


def load_factor_space_data() -> pd.DataFrame:
    """Load the EFA factor-space data used by the original Fig10 script."""
    df = pd.read_csv(INTERMEDIATE_DIR / "factor_scores.csv")
    df["task_name"] = df["task_name"].str.capitalize()
    df["author_year"] = (
        df["author_year"]
        .str.capitalize()
        .replace(STUDY_ABBREVIATIONS)
    )
    return df


def build_figure(
    run_export: bool = True,
    output_stem: str = "44fig4_v8_combined_3x3",
    selectable: bool = False,
    layout: str = "full",
    figsize: tuple[float, float] | None = None,
    width_ratios: tuple[float, float, float] = (3.0, 3.0, 2.0),
    height_ratios: tuple[float, float, float] = (1.35, 1.0, 1.0),
    wspace: float = 0.34,
    hspace: float = 0.70,
    font_scale: float = 1.08,
    point_size: float | None = 18,
    row_title_y_offset: float = 0.055,
    bottom_margin: float | None = None,
    task_legend_anchor: tuple[float, float] = (0.25, 0.145),
    lab_legend_anchor: tuple[float, float] = (0.66, 0.145),
    legend_font_scale: float = 0.92,
    legend_loc: str = "upper center",
) -> plt.Figure:
    """Build and save the full, reliability-only, or factor-space layout."""
    ensure_output_directories()
    valid_layouts = {"full", "reliability", "factor"}
    if layout not in valid_layouts:
        choices = ", ".join(sorted(valid_layouts))
        raise ValueError(f"Unknown layout '{layout}'. Choose from: {choices}.")

    configure_style(font_scale=font_scale, point_size=point_size)
    include_reliability = layout in {"full", "reliability"}
    include_factor = layout in {"full", "factor"}

    if run_export and include_reliability:
        run_r_export()
    if include_reliability:
        require_reliability_exports()
        icc_draws = pd.read_csv(EXPORT_DIR / "fig9b_cross_task_icc_draws.csv")
        temporal_icc = pd.read_csv(EXPORT_DIR / "fig8a_temporal_icc_subgroups.csv")
        sd_draws = pd.read_csv(EXPORT_DIR / "fig8b_temporal_sd_draws.csv")
    if include_factor:
        factor_df = load_factor_space_data()

    sns.set_theme(style="white", context="paper")
    if figsize is None:
        figsize = {
            "full": (8.0, 9.0),
            "reliability": (8.0, 3.0),
            "factor": (8.0, 6.0),
        }[layout]

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    n_rows = {"full": 3, "reliability": 1, "factor": 2}[layout]
    grid_kwargs = {
        "figure": fig,
        "width_ratios": width_ratios,
        "wspace": wspace,
        "hspace": hspace,
    }
    if layout == "full":
        grid_kwargs["height_ratios"] = height_ratios
    gs = GridSpec(
        n_rows,
        3,
        **grid_kwargs,
    )

    if include_reliability:
        ax_a = fig.add_subplot(gs[0, 0])
        plot_fig9b_icc_ridge(ax_a, icc_draws)
        add_panel_label(ax_a, "B")

        plot_fig8a_temporal_icc(fig, gs[0, 1], temporal_icc)

        ax_c = fig.add_subplot(gs[0, 2])
        plot_fig8b_sd_comparison(ax_c, sd_draws)
        add_panel_label(ax_c, "D")

    if include_factor:
        factor_start_row = 1 if layout == "full" else 0
        factor_panel_labels = (
            (("E", "F", "G"), ("H", "I", "J"))
            if layout == "full"
            else (("A", "B", "C"), ("D", "E", "F"))
        )
        row_2_axes = tuple(
            fig.add_subplot(gs[factor_start_row, col])
            for col in range(3)
        )
        row_3_axes = tuple(
            fig.add_subplot(gs[factor_start_row + 1, col])
            for col in range(3)
        )

        task_handles, lab_handles = plot_factor_row(
            row_2_axes,
            factor_df,
            FACTOR_CAUTION,
            FACTOR_NDT,
            factor_panel_labels[0],
        )
        task_handles_2, lab_handles_2 = plot_factor_row(
            row_3_axes,
            factor_df,
            FACTOR_EFFICIENCY,
            FACTOR_INHIBITION,
            factor_panel_labels[1],
        )

        task_handles = task_handles or task_handles_2
        lab_handles = lab_handles or lab_handles_2

    if bottom_margin is None:
        bottom_margin = 0.20 if include_factor else 0.16
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.975,
        bottom=bottom_margin,
    )

    if include_factor:
        add_row_title(
            fig,
            row_2_axes,
            "Factor Space 1: Decision Caution x Non-decision Time",
            y_offset=row_title_y_offset,
        )
        add_row_title(
            fig,
            row_3_axes,
            "Factor Space 2: Processing Efficiency x Inhibitory Process",
            y_offset=row_title_y_offset,
        )
        task_legend = fig.legend(
            handles=task_handles,
            loc=legend_loc,
            bbox_to_anchor=task_legend_anchor,
            ncol=len(task_handles),
            fontsize=LEGEND_FONTSIZE * legend_font_scale,
            title="Tasks",
            title_fontsize=(LEGEND_FONTSIZE + 1) * legend_font_scale,
            frameon=False,
            columnspacing=1.0,
            handletextpad=0.45,
            labelspacing=0.35,
        )
        lab_legend = fig.legend(
            handles=lab_handles,
            loc=legend_loc,
            bbox_to_anchor=lab_legend_anchor,
            ncol=4,
            fontsize=LEGEND_FONTSIZE * legend_font_scale,
            title="Studies",
            title_fontsize=(LEGEND_FONTSIZE + 1) * legend_font_scale,
            frameon=False,
            columnspacing=0.9,
            handletextpad=0.4,
            labelspacing=0.3,
        )
        clean_legend(task_legend)
        clean_legend(lab_legend)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_svg = FIG_DIR / f"{output_stem}.svg"
    out_png = FIG_DIR / f"{output_stem}.png"
    if selectable:
        # Keep text as real <text> (selectable/editable/searchable). Must be set
        # AFTER sns.set_theme() above, which resets rcParams and would otherwise
        # revert svg.fonttype back to the default 'path'.
        plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    svg_text = out_svg.read_text(encoding="utf-8")
    out_svg.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_svg}")
    print(f"Saved: {out_png}")
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-r-export",
        action="store_true",
        help="Use existing reliability CSV exports instead of regenerating them.",
    )
    parser.add_argument(
        "--output-stem",
        default="figure_04_05_panels",
        help="Output filename stem under figures/main.",
    )
    parser.add_argument(
        "--layout",
        choices=("full", "reliability", "factor"),
        default="full",
        help="Panel layout: full B-J grid, reliability B-D, or factor-space E-J.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches. Defaults depend on --layout.",
    )
    parser.add_argument(
        "--width-ratios",
        nargs=3,
        type=float,
        default=(3.0, 3.0, 2.0),
        metavar=("COL1", "COL2", "COL3"),
        help="Relative column widths. Default: 3 3 2.",
    )
    parser.add_argument(
        "--height-ratios",
        nargs=3,
        type=float,
        default=(1.35, 1.0, 1.0),
        metavar=("ROW1", "ROW2", "ROW3"),
        help="Relative row heights. Default: 1 1 1.",
    )
    parser.add_argument("--wspace", type=float, default=0.34, help="Horizontal panel spacing.")
    parser.add_argument("--hspace", type=float, default=0.70, help="Vertical panel spacing.")
    parser.add_argument("--font-scale", type=float, default=1.08, help="Multiplier for label/tick/legend text.")
    parser.add_argument("--point-size", type=float, default=18, help="Scatter point size for Fig10 panels.")
    parser.add_argument(
        "--row-title-y-offset",
        type=float,
        default=0.055,
        help="Figure-coordinate offset above each Fig10 row for the row title.",
    )
    parser.add_argument(
        "--bottom-margin",
        type=float,
        default=None,
        help="Bottom figure margin. Defaults depend on --layout.",
    )
    parser.add_argument(
        "--task-legend-anchor",
        nargs=2,
        type=float,
        default=(0.25, 0.145),
        metavar=("X", "Y"),
        help="Figure-coordinate anchor for the Tasks legend.",
    )
    parser.add_argument(
        "--lab-legend-anchor",
        nargs=2,
        type=float,
        default=(0.66, 0.145),
        metavar=("X", "Y"),
        help="Figure-coordinate anchor for the Studies legend.",
    )
    parser.add_argument(
        "--legend-font-scale",
        type=float,
        default=0.92,
        help="Additional multiplier for bottom legend text.",
    )
    parser.add_argument(
        "--legend-loc",
        default="upper center",
        help="Matplotlib legend loc for bottom legends. Default aligns legend tops.",
    )
    parser.add_argument(
        "--selectable",
        action="store_true",
        help="Keep SVG text editable/searchable without changing the output name. "
             "PDF; depends on Arial/Segoe UI being installed on the viewer.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.selectable:
        plt.rcParams["svg.fonttype"] = "none"
    build_figure(
        run_export=not args.skip_r_export,
        output_stem=args.output_stem,
        selectable=args.selectable,
        layout=args.layout,
        figsize=tuple(args.figsize) if args.figsize else None,
        width_ratios=tuple(args.width_ratios),
        height_ratios=tuple(args.height_ratios),
        wspace=args.wspace,
        hspace=args.hspace,
        font_scale=args.font_scale,
        point_size=args.point_size,
        row_title_y_offset=args.row_title_y_offset,
        bottom_margin=args.bottom_margin,
        task_legend_anchor=tuple(args.task_legend_anchor),
        lab_legend_anchor=tuple(args.lab_legend_anchor),
        legend_font_scale=args.legend_font_scale,
        legend_loc=args.legend_loc,
    )
