"""Convert the five canonical manuscript SVG figures to one-page PDFs."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from nsbi_module.project_paths import MAIN_FIGURES_DIR


FIGURE_STEMS = (
    "figure_01_workflow",
    "figure_02_model_comparison",
    "figure_03_posterior_predictive_checks",
    "figure_04_latent_factors",
    "figure_05_factor_space",
)


def find_inkscape() -> str:
    """Return the available Inkscape executable."""
    command = shutil.which("inkscape.com") or shutil.which("inkscape")
    if command is None:
        raise FileNotFoundError("Inkscape was not found on PATH.")
    return command


def convert_svg(command: str, source: Path, destination: Path) -> None:
    """Convert one SVG file to a PDF while preserving vector content."""
    if not source.is_file():
        raise FileNotFoundError(f"Figure source does not exist: {source}")
    subprocess.run(
        [
            command,
            str(source),
            "--export-type=pdf",
            "--export-area-page",
            f"--export-filename={destination}",
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MAIN_FIGURES_DIR / "pdf",
        help="Destination directory; defaults to figures/main/pdf.",
    )
    return parser.parse_args()


def main() -> None:
    """Export every canonical main figure to PDF."""
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inkscape = find_inkscape()

    for stem in FIGURE_STEMS:
        destination = output_dir / f"{stem}.pdf"
        convert_svg(inkscape, MAIN_FIGURES_DIR / f"{stem}.svg", destination)
        print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
