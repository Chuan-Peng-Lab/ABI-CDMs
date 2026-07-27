"""
NSBI-CDMs: Neural Simulation-Based Inference of Conflict Diffusion Models.

A Python package for amortized Bayesian inference of cognitive process models
(DDM, DMC, SSP, DSTP) using neural network-based simulation-based inference.
"""

from nsbi_module.NSBI_CDMs import CDMs_NSBI as NSBICDM
from nsbi_module.trainer import CDMsTrainer
from nsbi_module.simulators import CDMsSimulator
from nsbi_module.model_metrics import (
    compute_rmse,
    compute_g_square,
    get_best_model_by_metric,
)

__version__ = "1.0.0"
__all__ = [
    "NSBICDM",
    "CDMsTrainer",
    "CDMsSimulator",
    "compute_rmse",
    "compute_g_square",
    "get_best_model_by_metric",
]
