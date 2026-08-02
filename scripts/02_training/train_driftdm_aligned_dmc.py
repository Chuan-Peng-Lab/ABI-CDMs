"""Train the six-parameter dRiftDM-aligned DMC estimator."""

from nsbi_module.dmc_v2_loader import get_dmc_v2_model
from nsbi_module.project_paths import CHECKPOINTS_DIR


def main() -> None:
    """Train DMC_v2 with fixed shape and centered starting point."""
    model = get_dmc_v2_model(CHECKPOINTS_DIR / "driftdm_dmc")
    model.run(
        epochs=500,
        batch_size=32,
        num_batches_per_epoch=200,
        verbose=0,
        keep_optimizer=True,
    )


if __name__ == "__main__":
    main()
