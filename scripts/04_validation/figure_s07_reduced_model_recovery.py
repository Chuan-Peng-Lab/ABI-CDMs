"""Generate recovery diagnostics for the reduced supplementary models."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from matplotlib.figure import Figure

from nsbi_module.NSBI_CDMs import NSBICDM
from nsbi_module.dmc_v2_loader import get_dmc_v2_model
from nsbi_module.project_paths import CHECKPOINTS_DIR, SUPPLEMENT_FIGURES_DIR


def parse_args() -> argparse.Namespace:
    """Parse diagnostic sample sizes and output location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-batch", type=int, default=100)
    parser.add_argument("--n-posterior", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=SUPPLEMENT_FIGURES_DIR)
    return parser.parse_args()


def _standard_model(model_name: str, checkpoint_name: str) -> NSBICDM:
    """Load a registered reduced model from its canonical checkpoint."""
    return NSBICDM(model_name, checkpoint_path=CHECKPOINTS_DIR / checkpoint_name)


def _figures(result: object) -> list[Figure]:
    """Normalize BayesFlow diagnostic return values to Matplotlib figures."""
    if isinstance(result, Figure):
        return [result]
    if isinstance(result, (list, tuple)):
        return [item for item in result if isinstance(item, Figure)]
    figure = getattr(result, "figure", None)
    return [figure] if isinstance(figure, Figure) else []


def main() -> None:
    """Run recovery diagnostics and save every returned panel as SVG."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaders: dict[str, Callable[[], NSBICDM]] = {
        "dmc_fixed_shape": lambda: _standard_model(
            "DMC_fixed_alpha", "DMC_fixed_alpha"
        ),
        "ssp_fixed_ratio": lambda: _standard_model(
            "SSP_fixed_ratio", "SSP_fixed_ratio"
        ),
        "dstp_fixed_ratio": lambda: _standard_model(
            "DSTP_fixed_ratio", "DSTP_fixed_ratio"
        ),
        "driftdm_aligned_dmc": lambda: get_dmc_v2_model(
            CHECKPOINTS_DIR / "driftdm_dmc"
        ),
    }

    for label, load_model in loaders.items():
        model = load_model()
        result = model.plot_trained_result(
            n_batch=args.n_batch,
            n_posterior=args.n_posterior,
        )
        figures = _figures(result)
        if not figures:
            raise RuntimeError(f"No Matplotlib figure returned for {label}.")
        for index, figure in enumerate(figures, start=1):
            suffix = "" if len(figures) == 1 else f"_{index:02d}"
            output = args.output_dir / f"figure_s07_{label}{suffix}.svg"
            figure.savefig(output, format="svg", bbox_inches="tight")
            print(f"Saved {output}")


if __name__ == "__main__":
    main()
