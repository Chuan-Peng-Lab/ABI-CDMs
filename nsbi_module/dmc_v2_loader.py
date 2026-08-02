"""
Shared loader for the driftdm_dmc (DMC_v2) model.

This module centralises the simulator definition, model registration,
and instantiation so that every analysis script can use:

    from nsbi_module.dmc_v2_loader import get_dmc_v2_model
    m = get_dmc_v2_model()

without duplicating 150+ lines of numba kernels and config dictionaries.
"""

from numba import njit
import numpy as np

from nsbi_module.project_paths import CHECKPOINTS_DIR

# ── Simulator kernels (identical to 34_ulrich2015_driftdm_dmc.py) ──────────


@njit(fastmath=True)
def _dmc_drift_rate(t, muc, tau, a_shape, A_cond):
    if abs(a_shape - 2.0) < 1e-8:
        mu_auto = (A_cond / tau) * np.exp(1.0 - t / tau) * (1.0 - t / tau)
    else:
        t_adj = t + 1e-6
        base = (t_adj * np.e) / ((a_shape - 1.0) * tau)
        exp_part = np.exp(-t_adj / tau)
        poly_part = base ** (a_shape - 1.0)
        deriv_part = (a_shape - 1.0) / t_adj - 1.0 / tau
        mu_auto = A_cond * exp_part * poly_part * deriv_part
    return muc + mu_auto


@njit
def _draw_truncated_normal(mean, sd, lower, upper):
    if sd <= 0.0:
        return mean
    for _ in range(100):
        x = mean + sd * np.random.normal()
        if lower <= x <= upper:
            return x
    if x < lower:
        return lower
    if x > upper:
        return upper
    return x


@njit
def _shuffle_int_array(x):
    n = x.shape[0]
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = x[i]
        x[i] = x[j]
        x[j] = tmp


@njit
def _make_balanced_conditions(num_trials):
    condition = np.empty(num_trials, dtype=np.int32)
    half = num_trials // 2
    for i in range(half):
        condition[i] = 1
    for i in range(half, num_trials):
        condition[i] = 0
    if num_trials % 2 == 1:
        condition[num_trials - 1] = np.random.randint(0, 2)
    _shuffle_int_array(condition)
    return condition


@njit
def _driftdm_dmc_experiment_kernel(
    muc, b, non_dec, sd_non_dec, tau, a_shape, A,
    num_trials, sigma, dt, max_steps, var_non_dec, return_nan
):
    condition = _make_balanced_conditions(num_trials)
    rt = np.empty(num_trials, dtype=np.float64)
    choice = np.empty(num_trials, dtype=np.float64)
    sqrt_dt = np.sqrt(dt)

    for i in range(num_trials):
        cond_code = condition[i]
        A_cond = A if cond_code == 1 else -A
        evidence = 0.0

        if var_non_dec:
            ndt = _draw_truncated_normal(
                non_dec, sd_non_dec, 0.0,
                2.0 * non_dec + 5.0 * sd_non_dec
            )
        else:
            ndt = non_dec

        hit_boundary = False
        step = 0
        while step < max_steps:
            step += 1
            t = step * dt
            drift = _dmc_drift_rate(t, muc, tau, a_shape, A_cond)
            noise = sigma * sqrt_dt * np.random.normal()
            evidence += drift * dt + noise
            if evidence >= b:
                rt[i] = t + ndt
                choice[i] = 1.0
                hit_boundary = True
                break
            elif evidence <= -b:
                rt[i] = t + ndt
                choice[i] = 0.0
                hit_boundary = True
                break

        if not hit_boundary:
            rt[i] = max_steps * dt + ndt
            if return_nan:
                choice[i] = np.nan
                rt[i] = np.nan
            else:
                choice[i] = 1.0 if evidence >= 0.0 else 0.0

    return rt, choice, condition


def driftdm_dmc_experiment_simulator(
    muc: float, b: float, non_dec: float, sd_non_dec: float,
    tau: float, A: float,
    a_shape: float = 2.0, num_trials: int = 50,
    sigma: float = 1.0, dt: float = 0.0075,
    max_steps: int = 400, var_non_dec: bool = True,
    return_nan: bool = False,
):
    num_trials = int(num_trials)
    max_steps = int(max_steps)
    rt, choice, condition = _driftdm_dmc_experiment_kernel(
        muc, b, non_dec, sd_non_dec, tau, a_shape, A,
        num_trials, sigma, dt, max_steps, var_non_dec, return_nan,
    )
    return {"rt": rt, "choice": choice, "condition": condition.astype(np.float64)}


# ── Model config ──────────────────────────────────────────────────────────

DRIFTDMC_DMC_CONFIG = {
    "prior_range": {
        "muc": [0.5, 9.0],
        "b": [0.15, 1.20],
        "non_dec": [0.15, 0.60],
        "sd_non_dec": [0.005, 0.10],
        "tau": [0.015, 0.25],
        "A": [0.005, 0.30],
    },
    "param_keys": ["muc", "b", "non_dec", "sd_non_dec", "tau", "A"],
    "param_names": [
        r"$\mu_c$", r"$b$", r"$t_{er}$",
        r"$s_{t}$", r"$\tau$", r"$A$",
    ],
    "simulator_type": "experiment",
}

# ── Public API ────────────────────────────────────────────────────────────

MODEL_REG_NAME = "driftdm_dmc"
STORE_KEY_PREFIX = "dmc_v2"
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "driftdm_dmc"
_PARAM_KEYS = DRIFTDMC_DMC_CONFIG["param_keys"]


def register():
    """
    Register the driftdm_dmc simulator and config into the global registries.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    from nsbi_module import default_settings, simulators

    if MODEL_REG_NAME in default_settings.MODEL_CONFIG:
        return  # already registered

    simulators.TRIAL_SIMULATOR[MODEL_REG_NAME] = driftdm_dmc_experiment_simulator
    default_settings.MODEL_CONFIG[MODEL_REG_NAME] = DRIFTDMC_DMC_CONFIG
    default_settings.PARAMS_KEY_NAME_MAPPING = default_settings.get_param_mappings(
        default_settings.MODEL_CONFIG
    )

    # Create dmc_v2 alias for downstream code
    from nsbi_module.default_settings import PARAMS_KEY_NAME_MAPPING
    PARAMS_KEY_NAME_MAPPING[STORE_KEY_PREFIX] = (
        PARAMS_KEY_NAME_MAPPING[MODEL_REG_NAME]
    )


def get_dmc_v2_model(checkpoint_path=DEFAULT_CHECKPOINT):
    """
    Return a ready-to-use NSBICDM instance for driftdm_dmc.

    Automatically registers the model on first call.
    """
    from nsbi_module.NSBI_CDMs import NSBICDM
    register()
    return NSBICDM(MODEL_REG_NAME, checkpoint_path=checkpoint_path)
