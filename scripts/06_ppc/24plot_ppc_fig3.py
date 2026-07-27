#!/usr/bin/env python
# coding: utf-8
"""
Standalone script: generate Fig 3 – merged CAF + Delta PPC for selected studies.

Layout (7 rows × 3 columns):
  Rows 0–2:  CAF   (Eisenberg 2019 / Hedge 2018 / Rey-Mermet 2018)
  Row 3:     spacer for "Delta Plots" section title
  Rows 4–6:  Delta (same studies)

Usage:
    python 24plot_ppc_fig3.py
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import numpy as np

import sys
from nsbi_module.plotting import restructure_plotting_data  # noqa: E402
from nsbi_module.study_labels import format_study_abbrev  # noqa: E402

# ---------------------------------------------------------------------------
# 0.  Constants  (matching 24PPC.py)
# ---------------------------------------------------------------------------
# Strategy 2 (2026-07-24): reuse the four EFA factor colors from Fig4
# (44fig4_efa_svg.py FACTORS c0) for within-manuscript consistency.
# Revision (2026-07-24): DDM is the reference baseline model -> render it as a
# neutral gray so it recedes behind the three substantive models. DMC takes the
# vacated blue (DC) so it stays clearly distinct from DSTP's teal.
#   DDM  -> baseline gray #808080 (neutral reference)
#   SSP  -> NDT #7E6148 (brown)
#   DMC  -> DC  #3b6cb3 (blue, was DDM)
#   DSTP -> IP  #0099B4 (teal)
MODEL_COLORS = {
    'DDM':  '#808080',
    'SSP':  '#7E6148',
    'DMC':  '#3b6cb3',
    'DSTP': '#0099B4',
}
DEFAULT_TASKS      = ['flanker', 'simon', 'stroop']
DEFAULT_TASK_ORDER = ['Flanker', 'Simon', 'Stroop']
DESIRED_MODEL_ORDER = ["DDM", "DMC", "SSP", "DSTP"]

# Enhanced font sizes (larger than defaults)
STUDY_LABEL_FONTSIZE = 12
STUDY_LABEL_FONTWEIGHT = 'bold'
TICK_LABELSIZE = 9
SECTION_LABEL_FONTSIZE = 14

# Per-cell figsize
CELL_W, CELL_H = 3.2, 2.5
N_COLS = 3  # flanker, simon, stroop

# Study name formatting uses the reusable mapping from nsbi_module.study_labels
# (e.g. 'eisenberg2019' → 'E19', 'reymermet2018' → 'R18').

# ---------------------------------------------------------------------------
# 1.  Load pre-computed plotting data
# ---------------------------------------------------------------------------
plotting_data = pickle.load(open("24_ppc_process_data_dict.pkl", "rb"))

# Filter to selected studies (matching 24PPC.py)
selected_keys = [k for k in plotting_data
                 if k.startswith(('eisenberg', 'hedge', 'reymermet'))]
selected_data = {k: plotting_data[k] for k in selected_keys}

# Restructure for rows=studies × columns=tasks grid
structured, studies, tasks_col = restructure_plotting_data(
    selected_data, DEFAULT_TASKS, DEFAULT_TASK_ORDER
)

# Collect all model names found in the data
models = set()
for study_data in structured.values():
    for task_data in study_data.values():
        if "models" in task_data:
            models.update(task_data["models"].keys())
models = [m for m in DESIRED_MODEL_ORDER if m in models]

# ── Helper: draw CAF cell ──────────────────────────────────────────────────
def _draw_caf_cell(ax, data):
    """Draw observed (black markers) + model (colored lines) CAF curves."""
    obs_df = data["observed"]["caf"]
    for cond_key, col_name in [("comp", "comp"), ("incomp", "incomp")]:
        style = "-" if cond_key == "comp" else "--"
        if col_name in obs_df.columns:
            ax.plot(obs_df["bin"], obs_df[col_name],
                    color="black", linestyle=style,
                    marker="o", markersize=4, linewidth=2)

    for model_name in models:
        if model_name not in data.get("models", {}):
            continue
        pred_df = data["models"][model_name]["caf"]
        c = MODEL_COLORS.get(model_name, "#aaaaaa")
        for cond_key, col_name in [("comp", "comp"), ("incomp", "incomp")]:
            style = "-" if cond_key == "comp" else "--"
            if col_name in pred_df.columns:
                ax.plot(pred_df["bin"], pred_df[col_name],
                        color=c, linestyle=style, linewidth=2)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE)


# ── Helper: draw Delta cell ────────────────────────────────────────────────
def _draw_delta_cell(ax, data):
    """Draw observed (black markers) + model (colored lines) delta curves."""
    obs_df = data["observed"]["delta"]
    ax.plot(obs_df["mean_bin"], obs_df["mean_effect"],
            color='black', linestyle='-', linewidth=2,
            marker='o', markersize=4)

    for model_name in models:
        if model_name not in data.get("models", {}):
            continue
        pred_df = data["models"][model_name]["delta"]
        c = MODEL_COLORS.get(model_name, "#aaaaaa")
        ax.plot(pred_df["mean_bin"], pred_df["mean_effect"],
                color=c, linestyle='--', linewidth=2)

    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE)


# ── Helper: label study on leftmost column ─────────────────────────────────
def _set_study_label(ax, study: str):
    """Set bold abbreviated study name on y-axis (e.g. 'E19', 'H18')."""
    abbrev = format_study_abbrev(study)
    ax.set_ylabel(
        abbrev,
        fontsize=STUDY_LABEL_FONTSIZE,
        fontweight=STUDY_LABEL_FONTWEIGHT,
        labelpad=12,
    )


# ── Helper: hide empty cell while preserving outer labels ──────────────────
def _hide_empty_cell(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── Helper: set x-axis label ───────────────────────────────────────────────
def _set_xlabel(ax, label: str):
    ax.set_xlabel(label, fontsize=9)


def _compute_axis_limits() -> Tuple[Tuple[float, float] | None, Dict[str, Tuple[float, float]]]:
    """Compute shared CAF limits and per-study Delta limits."""
    caf_ylim = None
    delta_ylims: Dict[str, Tuple[float, float]] = {}

    y_min_all, y_max_all = np.inf, -np.inf
    for study in studies:
        for task in tasks_col:
            if task in structured[study]:
                y_vals = structured[study][task]["observed"]["caf"][["comp", "incomp"]]
                y_vals = y_vals.values.flatten()
                y_vals = y_vals[~np.isnan(y_vals)]
                if len(y_vals) > 0:
                    y_min_all = min(y_min_all, np.min(y_vals))
                    y_max_all = max(y_max_all, np.max(y_vals))
    if y_min_all != np.inf:
        margin = (y_max_all - y_min_all) * 0.08
        caf_ylim = (y_min_all - margin, y_max_all + margin)

    for study in studies:
        y_min_all, y_max_all = np.inf, -np.inf
        for task in tasks_col:
            if task in structured[study]:
                y_vals = structured[study][task]["observed"]["delta"]["mean_effect"]
                y_vals = y_vals.values.flatten()
                y_vals = y_vals[~np.isnan(y_vals)]
                if len(y_vals) > 0:
                    y_min_all = min(y_min_all, np.min(y_vals))
                    y_max_all = max(y_max_all, np.max(y_vals))
        if y_min_all != np.inf:
            margin = (y_max_all - y_min_all) * 0.08
            delta_ylims[study] = (y_min_all - margin, y_max_all + margin)

    return caf_ylim, delta_ylims


def _draw_section_grid(
    fig,
    spec,
    section_name: str,
    x_label: str,
    caf_ylim: Tuple[float, float] | None,
    delta_ylims: Dict[str, Tuple[float, float]],
    *,
    delta_row_ylims: Dict[int, Tuple[float, float]] | None = None,
    wspace: float = 0.12,
    hspace: float = 0.25,
):
    """Draw one 3 x 3 section grid and return its axes."""
    section_gs = spec.subgridspec(
        len(studies),
        len(tasks_col),
        wspace=wspace,
        hspace=hspace,
    )
    section_axes = np.empty((len(studies), len(tasks_col)), dtype=object)

    for i, study in enumerate(studies):
        for j, task in enumerate(tasks_col):
            ax = fig.add_subplot(section_gs[i, j])
            section_axes[i, j] = ax

            if i == 0:
                ax.set_title(task, fontsize=12, fontweight='bold', pad=6)
            if j == 0:
                _set_study_label(ax, study)
            if i == len(studies) - 1:
                _set_xlabel(ax, x_label)
            else:
                ax.tick_params(axis='x', labelbottom=False)
            if j != 0:
                ax.tick_params(axis='y', labelleft=False)

            if task not in structured[study]:
                _hide_empty_cell(ax)
                continue

            data = structured[study][task]
            if section_name == "CAF":
                _draw_caf_cell(ax, data)
                if caf_ylim:
                    ax.set_ylim(*caf_ylim)
            else:
                _draw_delta_cell(ax, data)
                if delta_row_ylims and i in delta_row_ylims:
                    ax.set_ylim(*delta_row_ylims[i])
                    ax.set_yticks(np.arange(delta_row_ylims[i][0], delta_row_ylims[i][1] + 1, 50))
                elif study in delta_ylims:
                    ax.set_ylim(*delta_ylims[study])

    return section_axes


def _section_bounds(axes):
    """Return the bounding box limits for an axes grid in figure coordinates."""
    boxes = [ax.get_position() for ax in axes.flat]
    return (
        min(box.x0 for box in boxes),
        min(box.y0 for box in boxes),
        max(box.x1 for box in boxes),
        max(box.y1 for box in boxes),
    )


def _add_panel_label(fig, axes, label: str, *, y_offset: float = 0.03):
    """Place a bold panel label just outside the top-left of an axes grid."""
    x0, _, _, y1 = _section_bounds(axes)
    fig.text(
        x0 - 0.055,
        y1 + y_offset,
        label,
        fontsize=24,
        fontweight='bold',
        ha='right',
        va='center',
    )


def _legend_line(fig, x0, x1, y, color, linestyle='-', marker=None):
    """Draw a legend sample line in figure coordinates."""
    line = mlines.Line2D(
        [x0, x1],
        [y, y],
        transform=fig.transFigure,
        color=color,
        linestyle=linestyle,
        linewidth=2,
        marker=marker,
        markersize=4,
        clip_on=False,
    )
    fig.add_artist(line)


def _legend_text(fig, x, y, label, *, weight='normal', ha='left'):
    """Draw legend text in figure coordinates."""
    fig.text(
        x,
        y,
        label,
        fontsize=11,
        fontweight=weight,
        ha=ha,
        va='center',
    )


def _draw_shared_legend(
    fig,
    *,
    y_mid: float,
    y_gap: float = 0.008,
    x_positions: Dict[str, float] | None = None,
):
    """Draw the shared Data and Model legend without automatic padding."""
    y_top = y_mid + y_gap
    y_bottom = y_mid - y_gap
    x = {
        "data_title": 0.155,
        "obs_line0": 0.255,
        "obs_line1": 0.295,
        "obs_text": 0.310,
        "cond_line0": 0.455,
        "cond_line1": 0.495,
        "cond_text": 0.510,
        "model_title": 0.665,
        "model_col1_line0": 0.755,
        "model_col2_line0": 0.865,
    }
    if x_positions:
        x.update(x_positions)

    _legend_text(fig, x["data_title"], y_mid, 'Data', weight='bold')
    _legend_line(fig, x["obs_line0"], x["obs_line1"], y_mid, 'black', marker='o')
    _legend_text(fig, x["obs_text"], y_mid, 'Observed')
    _legend_line(fig, x["cond_line0"], x["cond_line1"], y_top, 'gray', linestyle='-')
    _legend_text(fig, x["cond_text"], y_top, 'Congruent')
    _legend_line(fig, x["cond_line0"], x["cond_line1"], y_bottom, 'gray', linestyle='--')
    _legend_text(fig, x["cond_text"], y_bottom, 'Incongruent')

    _legend_text(fig, x["model_title"], y_mid, 'Model', weight='bold')
    model_positions = {
        'DDM': (x["model_col1_line0"], y_top),
        'DMC': (x["model_col1_line0"], y_bottom),
        'SSP': (x["model_col2_line0"], y_top),
        'DSTP': (x["model_col2_line0"], y_bottom),
    }
    for model_name in DESIRED_MODEL_ORDER:
        if model_name not in models:
            continue
        x, y = model_positions[model_name]
        _legend_line(fig, x, x + 0.040, y, MODEL_COLORS.get(model_name, '#aaaaaa'))
        _legend_text(fig, x + 0.055, y, model_name)


def build_vertical_figure():
    """Build the original vertical A-over-B layout."""
    caf_ylim, delta_ylims = _compute_axis_limits()
    height_ratios = [1, 1, 1, 0.35, 1, 1, 1]
    fig = plt.figure(figsize=(CELL_W * N_COLS, CELL_H * sum(height_ratios) * 0.85))
    gs = gridspec.GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[3, 0.35, 3],
        hspace=0.22,
        top=0.92,
        bottom=0.085,
    )

    axes_caf = _draw_section_grid(
        fig, gs[0, 0], "CAF", "RT Bin / Quantile", caf_ylim, delta_ylims
    )
    axes_delta = _draw_section_grid(
        fig, gs[2, 0], "Delta", "Mean RT (ms)", caf_ylim, delta_ylims
    )

    spacer_pos = gs[1, 0].get_position(fig)
    fig.text(
        (spacer_pos.x0 + spacer_pos.x1) / 2,
        (spacer_pos.y0 + spacer_pos.y1) / 2 - 0.005,
        "Delta Plots",
        fontsize=SECTION_LABEL_FONTSIZE,
        fontweight='bold',
        ha='center',
        va='center',
    )
    fig.text(
        0.5,
        0.965,
        'CAF',
        fontsize=16,
        fontweight='bold',
        ha='center',
        va='center',
    )
    _add_panel_label(fig, axes_caf, "A", y_offset=0.045)
    _add_panel_label(fig, axes_delta, "B", y_offset=0.045)
    _draw_shared_legend(fig, y_mid=0.010)
    return fig


def build_horizontal_figure():
    """Build the side-by-side A/B layout."""
    caf_ylim, delta_ylims = _compute_axis_limits()
    fig = plt.figure(figsize=(15.2, 7.4))
    gs = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1, 1],
        wspace=0.18,
        left=0.055,
        right=0.985,
        top=0.82,
        bottom=0.14,
    )

    axes_caf = _draw_section_grid(
        fig, gs[0, 0], "CAF", "RT Bin / Quantile", caf_ylim, delta_ylims,
        wspace=0.12, hspace=0.28,
    )
    axes_delta = _draw_section_grid(
        fig, gs[0, 1], "Delta", "Mean RT (ms)", caf_ylim, delta_ylims,
        delta_row_ylims={0: (0, 230), 1: (0, 230), 2: (0, 130)},
        wspace=0.12, hspace=0.28,
    )

    for title, axes in [("CAF", axes_caf), ("Delta Plots", axes_delta)]:
        x0, _, x1, y1 = _section_bounds(axes)
        fig.text(
            (x0 + x1) / 2,
            y1 + 0.070,
            title,
            fontsize=SECTION_LABEL_FONTSIZE,
            fontweight='bold',
            ha='center',
            va='center',
        )

    _add_panel_label(fig, axes_caf, "A", y_offset=0.075)
    _add_panel_label(fig, axes_delta, "B", y_offset=0.075)
    _draw_shared_legend(
        fig,
        y_mid=0.045,
        y_gap=0.012,
        x_positions={
            "data_title": 0.105,
            "obs_line0": 0.155,
            "obs_line1": 0.185,
            "obs_text": 0.198,
            "cond_line0": 0.285,
            "cond_line1": 0.315,
            "cond_text": 0.328,
            "model_title": 0.545,
            "model_col1_line0": 0.625,
            "model_col2_line0": 0.735,
        },
    )
    return fig


def _save_figure(fig, output_base: str):
    """Save a figure as SVG and PNG."""
    out_base = Path(output_base)
    out_svg = out_base.with_suffix(".svg")
    out_png = out_base.with_suffix(".png")

    fig.savefig(out_svg, bbox_inches='tight')
    svgtxt = out_svg.read_text(encoding='utf-8')
    out_svg.write_text(
        "\n".join(line.rstrip() for line in svgtxt.splitlines()) + "\n",
        encoding='utf-8',
    )
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {out_svg}")
    print(f"Saved: {out_png}")


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Generate vertical and/or horizontal PPC Fig 3 layouts."
    )
    parser.add_argument(
        "--layout",
        choices=["vertical", "horizontal", "both"],
        default="both",
        help="Which layout to generate. Default: both.",
    )
    return parser.parse_args()


def main():
    """Generate the requested PPC figure layout(s)."""
    args = parse_args()
    if args.layout in {"vertical", "both"}:
        _save_figure(build_vertical_figure(), "../figs/24_ppc_combined_fig3")
    if args.layout in {"horizontal", "both"}:
        _save_figure(build_horizontal_figure(), "../figs/24_ppc_combined_fig3_horizontal")


if __name__ == "__main__":
    main()
