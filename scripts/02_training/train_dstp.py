"""Train the amortized posterior estimator for the DSTP model."""

from nsbi_module.NSBI_CDMs import NSBICDM
from nsbi_module.project_paths import CHECKPOINTS_DIR


def main() -> None:
    """Train the DSTP estimator and save its checkpoint."""
    model = NSBICDM("DSTP", checkpoint_path=CHECKPOINTS_DIR / "DSTP")
    model.run(
        epochs=500,
        batch_size=32,
        num_batches_per_epoch=200,
        verbose=0,
        keep_optimizer=True,
    )


if __name__ == "__main__":
    main()
