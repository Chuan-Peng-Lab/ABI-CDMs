"""Generate manuscript Figure 5 from the estimated factor scores."""

from _figure_04_05_panels import build_figure


def main() -> None:
    """Build the two-row factor-space figure with editable SVG text."""
    build_figure(
        output_stem="figure_05_factor_space",
        selectable=True,
        layout="factor",
    )


if __name__ == "__main__":
    main()
