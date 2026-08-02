"""Canonical repository paths for ABI-CDMs scripts.

All paths are derived from this file, so commands can be launched from the
repository root or from any script directory without changing behavior.
"""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

DATA_DIR = REPO_ROOT / "data"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

FIGURES_DIR = REPO_ROOT / "figures"
MAIN_FIGURES_DIR = FIGURES_DIR / "main"
SUPPLEMENT_FIGURES_DIR = FIGURES_DIR / "supplement"

RESULTS_DIR = REPO_ROOT / "results"
INTERMEDIATE_DIR = RESULTS_DIR / "intermediate"
TABLES_DIR = RESULTS_DIR / "tables"


def ensure_output_directories() -> None:
    """Create the generated-output directories used by analysis scripts."""
    for path in (
        MAIN_FIGURES_DIR,
        SUPPLEMENT_FIGURES_DIR,
        INTERMEDIATE_DIR,
        TABLES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
