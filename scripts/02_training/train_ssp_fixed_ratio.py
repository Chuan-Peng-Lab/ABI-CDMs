"""Train the reduced SSP estimator used in Supplementary Figure S7."""

from nsbi_module.NSBI_CDMs import NSBICDM
from nsbi_module.project_paths import CHECKPOINTS_DIR


def main() -> None:
    """Train SSP using the shrinkage-rate-to-width ratio parameterization."""
    model = NSBICDM(
        "SSP_fixed_ratio",
        checkpoint_path=CHECKPOINTS_DIR / "SSP_fixed_ratio",
    )
    model.run(
        epochs=500,
        batch_size=16,
        num_batches_per_epoch=200,
        verbose=0,
        keep_optimizer=True,
    )


if __name__ == "__main__":
    main()
