#!/usr/bin/env python
"""
Standalone script: supplementary Fig. S5 - full CAF PPC grid, two-block layout.

Background
----------
The published S5 (``figs/24_ppc_caf.svg`` -> Supplementary ``image5.png``) was
produced by the *old* flat layout of ``plot_distribution_curve`` (rows = 21
datasets, cols = 4 models), which no longer exists in
``nsbi_module.plotting``. The result is extremely tall (21 rows x 4 cols).

This script regenerates the same content in a two-block ("8-column") layout:
  - Left block : datasets 1-11  (4 model columns)
  - Right block: datasets 12-21 (4 model columns)
  - Each block carries a rotated (90 deg), two-line study+task label
    (e.g. "Clayson2025\nFlanker") on the RIGHT side of its 4 columns.
  - No y-axis "Accuracy" label (per 2026-07-30 request).

Visual encoding is kept identical to the current S5 for manuscript consistency:
  color      = condition   (Congruent #1f77b4 blue / Incongruent #d62728 red)
  line style = data source (Observed solid + circle markers / Model dashed)

Usage:
    python scripts/06_ppc/figure_s05_caf.py --selectable
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib import gridspec

from nsbi_module.plotting import parse_dataset_name
from nsbi_module.project_paths import INTERMEDIATE_DIR, SUPPLEMENT_FIGURES_DIR

plt.rcParams["font.sans-serif"] = ["Arial", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# 0.  Constants
# ---------------------------------------------------------------------------
# Condition colors exactly as in the current S5 (figs/24_ppc_caf.svg).
COND_COLORS = {"comp": "#1f77b4", "incomp": "#d62728"}
LEGEND_GRAY = "#808080"

DESIRED_MODEL_ORDER = ["DDM", "DMC", "SSP", "DSTP"]
KNOWN_TASKS = ["flanker", "simon", "stroop"]

# Enlarged fonts (figure is large; text must survive downscaling).
TICK_LABELSIZE = 12
AXIS_LABEL_FONTSIZE = 13
STUDY_LABEL_FONTSIZE = 14
MODEL_TITLE_FONTSIZE = 18
SUPTITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 15

# Per-cell size (inches) and grid spacing.
CELL_W, CELL_H = 2.3, 1.65
SPACER_RATIO = 0.5  # inter-block spacer column width, relative to one cell
WSPACE, HSPACE = 0.16, 0.22


# ---------------------------------------------------------------------------
# 1.  Load pre-computed plotting data
# ---------------------------------------------------------------------------
def load_plotting_data() -> dict:
    with (INTERMEDIATE_DIR / "ppc_data.pkl").open("rb") as input_file:
        return pickle.load(input_file)


# ---------------------------------------------------------------------------
# 2.  Helpers
# ---------------------------------------------------------------------------
def _format_row_label(ds_name: str) -> str:
    """'clayson2025flanker' -> 'Clayson2025\nFlanker' (two lines)."""
    study, task = parse_dataset_name(ds_name, KNOWN_TASKS)
    study_label = study[0].upper() + study[1:]
    return f"{study_label}\n{task}"


def _draw_caf_cell(ax, data: dict, model_name: str) -> None:
    """Observed (solid + markers) and model (dashed) CAF, colored by condition."""
    obs_df = data.get("observed", {}).get("caf")
    if obs_df is not None:
        for cond, col in [("comp", "comp"), ("incomp", "incomp")]:
            if col in obs_df.columns:
                ax.plot(
                    obs_df["bin"],
                    obs_df[col],
                    color=COND_COLORS[cond],
                    linestyle="-",
                    marker="o",
                    markersize=6,
                    linewidth=2.5,
                )

    pred_df = data.get("models", {}).get(model_name, {}).get("caf")
    if pred_df is not None:
        for cond, col in [("comp", "comp"), ("incomp", "incomp")]:
            if col in pred_df.columns:
                ax.plot(
                    pred_df["bin"],
                    pred_df[col],
                    color=COND_COLORS[cond],
                    linestyle="--",
                    linewidth=3,
                )

    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)


def _hide_cell(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_block(
    fig,
    gs,
    plotting_data: dict,
    block_datasets: list[str],
    models: list[str],
    col0: int,
    n_rows: int,
) -> None:
    """Draw one block of ``len(block_datasets)`` rows x ``len(models)`` cols."""
    n_cols = len(models)
    for i in range(n_rows):
        has_data = i < len(block_datasets)
        ds = block_datasets[i] if has_data else None

        for j, model_name in enumerate(models):
            ax = fig.add_subplot(gs[i, col0 + j])

            if not has_data:
                _hide_cell(ax)
                continue

            data = plotting_data[ds]
            if "caf" not in data.get("observed", {}):
                _hide_cell(ax)
                continue

            _draw_caf_cell(ax, data, model_name)

            # Column titles on the first row of the block
            if i == 0:
                ax.set_title(
                    model_name,
                    fontsize=MODEL_TITLE_FONTSIZE,
                    fontweight="bold",
                    pad=8,
                )

            # X labels only on the block's last data row
            if i == len(block_datasets) - 1:
                ax.set_xlabel("RT Bin / Quantile", fontsize=AXIS_LABEL_FONTSIZE)
            else:
                ax.tick_params(axis="x", labelbottom=False)

            # Rotated two-line study+task label on the RIGHT of each block
            if j == n_cols - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(
                    _format_row_label(ds),
                    rotation=90,
                    fontsize=STUDY_LABEL_FONTSIZE,
                    fontweight="bold",
                    labelpad=14,
                    va="center",
                )


def _draw_shared_legend(fig) -> None:
    """Single-row legend at the bottom: Observed / Model / conditions."""
    legend_handles = [
        mlines.Line2D(
            [], [], color=LEGEND_GRAY, marker="o", markersize=9,
            linewidth=3.5, label="Observed",
        ),
        mlines.Line2D(
            [], [], color=LEGEND_GRAY, linestyle="--",
            linewidth=3.5, label="Model",
        ),
        mlines.Line2D(
            [], [], color=COND_COLORS["comp"], linewidth=3.5, label="Congruent",
        ),
        mlines.Line2D(
            [], [], color=COND_COLORS["incomp"], linewidth=3.5, label="Incongruent",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=True,
        shadow=False,
        handlelength=2.6,
        columnspacing=2.2,
    )


# ---------------------------------------------------------------------------
# 3.  Build figure
# ---------------------------------------------------------------------------
def build_figure(plotting_data: dict, left_n: int | None = None) -> plt.Figure:
    datasets = sorted(plotting_data.keys())

    models = set()
    for ds in datasets:
        models.update(plotting_data[ds].get("models", {}).keys())
    models = [m for m in DESIRED_MODEL_ORDER if m in models]
    n_cols = len(models)

    n = len(datasets)
    if left_n is None:
        left_n = (n + 1) // 2
    left_ds, right_ds = datasets[:left_n], datasets[left_n:]
    n_rows = max(len(left_ds), len(right_ds))

    fig_w = (CELL_W * (2 * n_cols + SPACER_RATIO + WSPACE * (2 * n_cols))) / 0.93
    fig_h = (CELL_H * (n_rows + HSPACE * (n_rows - 1))) / 0.885
    fig = plt.figure(figsize=(fig_w, fig_h))

    width_ratios = [1] * n_cols + [SPACER_RATIO] + [1] * n_cols
    gs = gridspec.GridSpec(
        n_rows,
        2 * n_cols + 1,
        figure=fig,
        width_ratios=width_ratios,
        wspace=WSPACE,
        hspace=HSPACE,
        left=0.03,
        right=0.965,
        top=0.94,
        bottom=0.055,
    )

    _draw_block(fig, gs, plotting_data, left_ds, models, col0=0, n_rows=n_rows)
    _draw_block(fig, gs, plotting_data, right_ds, models, col0=n_cols + 1, n_rows=n_rows)

    fig.suptitle(
        "Conditional Accuracy Function (CAF)",
        fontsize=SUPTITLE_FONTSIZE,
        fontweight="bold",
        y=0.985,
    )
    _draw_shared_legend(fig)
    return fig


# ---------------------------------------------------------------------------
# 4. Save the canonical supplementary figure.
# ---------------------------------------------------------------------------
def _save_figure(fig, output_base: Path, *, selectable: bool = False) -> None:
    out_base = Path(output_base)
    out_svg = out_base.with_suffix(".svg")
    out_png = out_base.with_suffix(".png")
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    if selectable:
        plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    svgtxt = out_svg.read_text(encoding="utf-8")
    if 'style="fill: #ffffff"' not in svgtxt:
        svg_start = svgtxt.find(">")
        white_background = '\n<rect width="100%" height="100%" fill="white"/>'
        svgtxt = svgtxt[: svg_start + 1] + white_background + svgtxt[svg_start + 1 :]
    out_svg.write_text(
        "\n".join(line.rstrip() for line in svgtxt.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {out_svg}")
    print(f"Saved: {out_png}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the two-block (2 x 4 model columns) S5 CAF grid."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SUPPLEMENT_FIGURES_DIR,
        help="Directory for SVG and PNG outputs.",
    )
    parser.add_argument(
        "--output-stem",
        default="figure_s05_caf",
        help="Filename stem for the outputs.",
    )
    parser.add_argument(
        "--left-n",
        type=int,
        default=None,
        help="Number of datasets in the left block (default: ceil(n/2)).",
    )
    parser.add_argument(
        "--selectable",
        action="store_true",
        help="Keep SVG labels as searchable/selectable text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plotting_data = load_plotting_data()
    fig = build_figure(plotting_data, left_n=args.left_n)
    _save_figure(
        fig,
        args.output_dir.resolve() / args.output_stem,
        selectable=args.selectable,
    )


if __name__ == "__main__":
    main()
