"""Generate manuscript Figure 4 from EFA and reliability components."""

import matplotlib.pyplot as plt

from _figure_04_05_panels import build_figure
from _figure_04_efa_diagram import compose_fig4, compute_scene


def main() -> None:
    """Build the reliability grid, then compose it below the EFA diagram."""
    build_figure(
        output_stem="_figure_04_reliability_panels",
        selectable=True,
        layout="reliability",
    )
    plt.close("all")
    compose_fig4(
        compute_scene(),
        grid_stem="_figure_04_reliability_panels",
        out_stem="figure_04_latent_factors",
    )


if __name__ == "__main__":
    main()
