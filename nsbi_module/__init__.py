"""
ABI-CDMs: Amortized Bayesian Inference of Conflict Diffusion Models.

A Python package for amortized Bayesian inference of cognitive process models
(DDM, DMC, SSP, DSTP) using neural network-based simulation-based inference.
"""

__version__ = "1.1.0"
__all__ = [
    "NSBICDM",
    "CDMsTrainer",
    "CDMsSimulator",
]


def __getattr__(name: str):
    """Load model classes only when they are requested.

    This keeps lightweight modules such as ``project_paths`` usable during
    preprocessing without importing the full BayesFlow training stack.
    """
    if name == "NSBICDM":
        from nsbi_module.NSBI_CDMs import NSBICDM

        return NSBICDM
    if name in {"CDMsTrainer", "CDMsSimulator"}:
        from nsbi_module.trainer import CDMsSimulator, CDMsTrainer

        return {"CDMsTrainer": CDMsTrainer, "CDMsSimulator": CDMsSimulator}[name]
    raise AttributeError(f"module 'nsbi_module' has no attribute {name!r}")
