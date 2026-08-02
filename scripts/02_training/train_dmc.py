"""Train the amortized posterior estimator for the DMC."""

from nsbi_module.NSBI_CDMs import NSBICDM
from nsbi_module.project_paths import CHECKPOINTS_DIR


def main() -> None:
    """Train the DMC estimator and save its checkpoint."""
    model = NSBICDM("DMC", checkpoint_path=CHECKPOINTS_DIR / "DMC")
    model.run(epochs=500, verbose=0, keep_optimizer=True)


if __name__ == "__main__":
    main()
