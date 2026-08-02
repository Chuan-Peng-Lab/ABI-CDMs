"""Train the reduced DMC estimator used in Supplementary Figure S7."""

from nsbi_module.NSBI_CDMs import NSBICDM
from nsbi_module.project_paths import CHECKPOINTS_DIR


def main() -> None:
    """Train DMC with the automatic-activation shape fixed at two."""
    model = NSBICDM(
        "DMC_fixed_alpha",
        checkpoint_path=CHECKPOINTS_DIR / "DMC_fixed_alpha",
    )
    model.run(epochs=500, verbose=0, keep_optimizer=True)


if __name__ == "__main__":
    main()
