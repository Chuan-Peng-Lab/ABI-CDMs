"""Train the seven-parameter DMC estimator used in Figure S8."""

from nsbi_module.dmc_variable_start_loader import get_dmc_variable_start_model
from nsbi_module.project_paths import CHECKPOINTS_DIR


def main() -> None:
    """Train the dRiftDM-aligned DMC with starting-point variability."""
    model = get_dmc_variable_start_model(
        CHECKPOINTS_DIR / "driftdm_dmc_vs"
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
