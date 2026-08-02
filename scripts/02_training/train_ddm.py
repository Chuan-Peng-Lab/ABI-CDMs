"""Train the amortized posterior estimator for the DDM."""

from nsbi_module.NSBI_CDMs import NSBICDM
from nsbi_module.project_paths import CHECKPOINTS_DIR


def main() -> None:
    """Train the DDM estimator and save its checkpoint."""
    model = NSBICDM("DDM", checkpoint_path=CHECKPOINTS_DIR / "DDM")
    model.run(
        epochs=250,
        batch_size=64,
        num_batches_per_epoch=250,
        verbose=0,
        keep_optimizer=True,
    )


if __name__ == "__main__":
    main()
