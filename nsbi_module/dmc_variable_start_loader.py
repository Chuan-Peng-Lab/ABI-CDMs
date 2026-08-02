"""Loader for the seven-parameter dRiftDM-aligned DMC estimator.

This specification extends the six-parameter DMC_v2 model with symmetric
Beta-distributed starting-point variability. It is the ABI model used by the
current Supplementary Figure S8 comparison pipeline.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from nsbi_module.dmc_v2_loader import (
    _dmc_drift_rate,
    _draw_truncated_normal,
    _make_balanced_conditions,
)
from nsbi_module.project_paths import CHECKPOINTS_DIR


@njit
def _sample_symmetric_beta(alpha: float) -> float:
    """Sample Beta(alpha, alpha) by rejection from a uniform proposal."""
    for _ in range(1000):
        value = np.random.random()
        acceptance = (4.0 * value * (1.0 - value)) ** (alpha - 1.0)
        if np.random.random() < acceptance:
            return value
    return 0.5


@njit
def _experiment_kernel(
    muc: float,
    b: float,
    non_dec: float,
    sd_non_dec: float,
    tau: float,
    a_shape: float,
    amplitude: float,
    alpha: float,
    num_trials: int,
    sigma: float,
    dt: float,
    max_steps: int,
    return_nan: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate balanced congruent/incongruent trials."""
    condition = _make_balanced_conditions(num_trials)
    rt = np.empty(num_trials, dtype=np.float64)
    choice = np.empty(num_trials, dtype=np.float64)
    sqrt_dt = np.sqrt(dt)

    for trial in range(num_trials):
        automatic_amplitude = amplitude if condition[trial] == 1 else -amplitude
        start = _sample_symmetric_beta(alpha)
        evidence = (2.0 * start - 1.0) * b
        ndt = _draw_truncated_normal(
            non_dec,
            sd_non_dec,
            0.0,
            2.0 * non_dec + 5.0 * sd_non_dec,
        )

        hit_boundary = False
        for step in range(1, max_steps + 1):
            time = step * dt
            drift = _dmc_drift_rate(
                time,
                muc,
                tau,
                a_shape,
                automatic_amplitude,
            )
            evidence += drift * dt + sigma * sqrt_dt * np.random.normal()
            if evidence >= b:
                rt[trial] = time + ndt
                choice[trial] = 1.0
                hit_boundary = True
                break
            if evidence <= -b:
                rt[trial] = time + ndt
                choice[trial] = 0.0
                hit_boundary = True
                break

        if not hit_boundary:
            rt[trial] = max_steps * dt + ndt
            choice[trial] = 1.0 if evidence >= 0.0 else 0.0
            if return_nan:
                rt[trial] = np.nan
                choice[trial] = np.nan

    return rt, choice, condition


def dmc_variable_start_simulator(
    muc: float,
    b: float,
    non_dec: float,
    sd_non_dec: float,
    tau: float,
    A: float,
    alpha: float,
    a_shape: float = 2.0,
    num_trials: int = 50,
    sigma: float = 1.0,
    dt: float = 0.0075,
    max_steps: int = 400,
    return_nan: bool = False,
) -> dict[str, np.ndarray]:
    """Simulate one experiment from the variable-start DMC specification."""
    rt, choice, condition = _experiment_kernel(
        muc,
        b,
        non_dec,
        sd_non_dec,
        tau,
        a_shape,
        A,
        alpha,
        int(num_trials),
        sigma,
        dt,
        int(max_steps),
        return_nan,
    )
    return {
        "rt": rt,
        "choice": choice,
        "condition": condition.astype(np.float64),
    }


MODEL_CONFIG = {
    "prior_range": {
        "muc": [0.5, 9.0],
        "b": [0.15, 1.20],
        "non_dec": [0.15, 0.60],
        "sd_non_dec": [0.005, 0.10],
        "tau": [0.015, 0.25],
        "A": [0.005, 0.30],
        "alpha": [2, 8],
    },
    "param_keys": ["muc", "b", "non_dec", "sd_non_dec", "tau", "A", "alpha"],
    "param_names": [
        r"$\mu_c$",
        r"$b$",
        r"$t_{er}$",
        r"$s_t$",
        r"$\tau$",
        r"$A$",
        r"$\alpha$",
    ],
    "simulator_type": "experiment",
}

MODEL_NAME = "driftdm_dmc_vs"
STORE_KEY_PREFIX = "dmc_vs"
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / MODEL_NAME


def register() -> None:
    """Register the simulator and parameter configuration once."""
    from nsbi_module import default_settings, simulators

    if MODEL_NAME in default_settings.MODEL_CONFIG:
        return
    simulators.TRIAL_SIMULATOR[MODEL_NAME] = dmc_variable_start_simulator
    default_settings.MODEL_CONFIG[MODEL_NAME] = MODEL_CONFIG
    default_settings.PARAMS_KEY_NAME_MAPPING = default_settings.get_param_mappings(
        default_settings.MODEL_CONFIG
    )
    default_settings.PARAMS_KEY_NAME_MAPPING[STORE_KEY_PREFIX] = (
        default_settings.PARAMS_KEY_NAME_MAPPING[MODEL_NAME]
    )


def get_dmc_variable_start_model(checkpoint_path=DEFAULT_CHECKPOINT):
    """Return a ready-to-use variable-start DMC estimator."""
    from nsbi_module.NSBI_CDMs import NSBICDM

    register()
    return NSBICDM(MODEL_NAME, checkpoint_path=checkpoint_path)
