"""
NSBI-CDMs: Neural Simulation-Based Inference of Conflict Diffusion Models.

A Python package for amortized Bayesian inference of cognitive process models
(DDM, DMC, SSP, DSTP) using neural network-based simulation-based inference.
"""

from nsbi_module.NSBI_CDMs import NSBICDM
from nsbi_module.trainer import CDMsTrainer, CDMsSimulator

__version__ = "1.0.0"
__all__ = [
    "NSBICDM",
    "CDMsTrainer",
    "CDMsSimulator",
]
