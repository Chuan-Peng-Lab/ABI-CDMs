"""Generate the DSTP-versus-DMC model-comparison plot for Figure S9."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from nsbi_module.project_paths import INTERMEDIATE_DIR, SUPPLEMENT_FIGURES_DIR
from nsbi_module.utils_ind_diff import (
    calc_best_model_proportion,
    format_author_year,
    format_task_id,
    format_task_name,
    get_best_model_by_metric,
    plot_best_model_proportion_barh,
    rank_models_by_metric,
)


MODEL_COLORS = {"DSTP": "#81cef0", "DMC": "#95d8c3"}


def parse_args() -> argparse.Namespace:
    """Parse the model-index input and SVG output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=INTERMEDIATE_DIR / "model_prediction_indices_extended_dmc.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SUPPLEMENT_FIGURES_DIR / "figure_s09_dstp_vs_dmc.svg",
    )
    return parser.parse_args()


def plot_comparison(
    data: pd.DataFrame,
    figsize: tuple[float, float] = (7.0, 5.0),
    save_name: Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot subject-level winning-model proportions by study and task."""
    subset = data.query("author_year != 'lee2025'")
    subset = subset[subset["model"].isin(MODEL_COLORS)].copy()
    ranked = rank_models_by_metric(subset, metric="RMSE")
    winners = get_best_model_by_metric(ranked, metric="RMSE")
    winners["author_year"] = winners["author_year"].apply(format_author_year)
    winners["task_name"] = winners["task_name"].apply(format_task_name)
    winners["task_id"] = winners["task_id"].apply(format_task_id)
    proportions = calc_best_model_proportion(
        winners,
        group_cols=["task_id", "task_name", "author_year"],
    )

    figure, axis = plt.subplots(figsize=figsize)
    plot_best_model_proportion_barh(
        proportions,
        order_by="author_year",
        colors=MODEL_COLORS,
        models_sorted=["DSTP", "DMC"],
        alpha=1,
        ax=axis,
    )
    figure.tight_layout()
    if save_name is not None:
        save_name.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_name, format="svg", bbox_inches="tight")
    return figure, axis


def main() -> None:
    """Load the extended fit summary and write Figure S9."""
    args = parse_args()
    plot_comparison(pd.read_csv(args.input), save_name=args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
