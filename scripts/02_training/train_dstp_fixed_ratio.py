"""Train the reduced DSTP estimator used in Supplementary Figure S7."""

from nsbi_module.NSBI_CDMs import NSBICDM
from nsbi_module.project_paths import CHECKPOINTS_DIR


def main() -> None:
    """Train DSTP using the flanker-to-selection drift ratio."""
    model = NSBICDM(
        "DSTP_fixed_ratio",
        checkpoint_path=CHECKPOINTS_DIR / "DSTP_fixed_ratio",
    )
    model.run(
        epochs=500,
        batch_size=32,
        num_batches_per_epoch=200,
        verbose=0,
        keep_optimizer=True,
    )


if __name__ == "__main__":
    main()
