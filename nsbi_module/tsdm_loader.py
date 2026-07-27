"""
Shared loader for the tsdm (Two-Stage Dual-Mechanism) diffusion model.

This module centralises the simulator definition, model registration,
and instantiation so that every analysis script can use:

    from tsdm_loader import get_tsdm_model
    m = get_tsdm_model()

Model: TSDM with 9 parameters (wc, b, mu_r, sigma_r, r_pc, r_pa, kp, kd, te).

The TSDM model consists of two stages:
  1. Perceptual stage (duration tp = t_nd - te):
     Parallel processing of task-relevant (controlled) and task-irrelevant
     (automatic) stimulus attributes, modulated by attention after 100 ms.
  2. Decision stage: Evidence accumulation via drift-diffusion with
     time-varying drift rates that differ between congruent and incongruent
     conditions.

Key theoretical assumptions:
  - Total attention resources are fixed at 2.
  - Proactive control determines initial allocation wc.
  - Reactive control governs the speed of attentional shifts (kp, kd).
  - The incongruent condition triggers faster attention reallocation (kd > kp).

All internal computations use milliseconds (ms); output RT is converted to
seconds to match the NSBI framework convention.
"""

from numba import njit
import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# Shared Numba utility functions
# ═════════════════════════════════════════════════════════════════════════════


@njit
def _draw_truncated_normal(mean, sd, lower, upper):
    """Sample from a truncated normal distribution via rejection."""
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
    """Fisher-Yates shuffle in-place."""
    n = x.shape[0]
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = x[i]
        x[i] = x[j]
        x[j] = tmp


@njit
def _make_balanced_conditions(num_trials):
    """Create a balanced (≈50/50) array of condition labels.
    
    Returns:
        np.ndarray (int32): 1 = congruent, 0 = incongruent.
    """
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


# ═════════════════════════════════════════════════════════════════════════════
# TSDM attention functions
# ═════════════════════════════════════════════════════════════════════════════


@njit
def _AR(t, wc_resource, k):
    """Attention resource allocated to the automatic channel.

    AR(t) = (2 - wc) / (1 + t^k / 1e15)

    Parameters:
        t: Time since attention onset (ms).
        wc_resource: Initial resource allocation to controlled channel.
        k: Attention shift rate parameter (kp for perceptual/congruent,
           kd for incongruent decision stage).

    Returns:
        Attention resource fraction for the automatic channel.
        Controlled channel receives (2 - AR(t)).
    """
    return wc_resource / (1.0 + np.power(t, k) / 1e15)


@njit
def _perceptual_pa(t, dt_ms, r_pa, wc_resource, kp):
    """Instantaneous perceptual input strength of the automatic channel.
    
    For t <= 100 ms: p_a(t) = r_pa * t  (linear growth)
    For t > 100 ms:  p_a(t) = r_pa * dt_ms * AR(dt_ms, wc_resource, kp)
    
    where dt_ms = t - 100 is time since attention onset.
    
    Parameters:
        t: Current time point (ms), must be >= 1.
        dt_ms: Time since attention onset = max(0, t - 100).
        r_pa: Base drift rate of automatic channel.
        wc_resource: 2 - wc (resources allocated to auto channel initially).
        kp: Perceptual attention shift rate.
    """
    if t <= 100.0:
        return r_pa * t
    else:
        ar_val = _AR(dt_ms, wc_resource, kp)
        return r_pa * dt_ms * ar_val


@njit
def _perceptual_pc(t, dt_ms, r_pc, wc_resource, kp):
    """Instantaneous perceptual input strength of the controlled channel.
    
    For t <= 100 ms: p_c(t) = r_pc * t  (linear growth)
    For t > 100 ms:  p_c(t) = r_pc * dt_ms * (2 - AR(dt_ms))
    
    where dt_ms = t - 100.
    """
    if t <= 100.0:
        return r_pc * t
    else:
        ar_val = _AR(dt_ms, wc_resource, kp)
        return r_pc * dt_ms * (2.0 - ar_val)


@njit
def _compute_perceptual_accumulation(r_pa, r_pc, wc, kp, tp):
    """Numerically integrate perceptual input strength over [0, tp] ms.
    
    Returns:
        (Sp_a, Sp_c): Cumulative perceptual evidence for automatic and
        controlled channels respectively.
    """
    wc_resource = 2.0 - wc  # initial auto-channel resources
    Sp_a = 0.0
    Sp_c = 0.0

    # Integrate with dt = 1 ms
    tp_int = int(tp)
    if tp_int < 1:
        tp_int = 1

    for t in range(1, tp_int + 1):
        t_f = float(t)
        dt_ms = max(0.0, t_f - 100.0)
        Sp_a += _perceptual_pa(t_f, dt_ms, r_pa, wc_resource, kp)
        Sp_c += _perceptual_pc(t_f, dt_ms, r_pc, wc_resource, kp)

    return Sp_a, Sp_c


# ═════════════════════════════════════════════════════════════════════════════
# Decision-stage drift rate functions
# ═════════════════════════════════════════════════════════════════════════════


@njit
def _drift_congruent(t_dec, tp, wc_resource, kp, Sp_a, Sp_c, drift_scale):
    """Drift rate for congruent trials.

    mu_total(t) = Sp_a * AR(tp-100+t) + Sp_c * (2 - AR(tp-100+t))

    The AR function continues from where it left off in the perceptual stage.
    """
    t_attention = (tp - 100.0) + t_dec
    if t_attention < 0.0:
        t_attention = 0.0
    ar_val = _AR(t_attention, wc_resource, kp)
    mu = Sp_a * ar_val + Sp_c * (2.0 - ar_val)
    return mu * drift_scale


@njit
def _drift_incongruent(t_dec, wc_resource, kd, Sp_a, Sp_c, drift_scale):
    """Drift rate for incongruent trials.

    AR_ai(t) = (2-wc) / (1 + t^kd / 1e15)         (no ×2 factor — corrected)
    mu_total(t) = Sp_c * (2 - AR_ai(t)) - Sp_a * AR_ai(t)

    Note: The AR function resets at decision onset (t_dec=0) because conflict
    detection triggers a fresh reactive control process.
    """
    ar_val = _AR(t_dec, wc_resource, kd)
    mu = Sp_c * (2.0 - ar_val) - Sp_a * ar_val
    return mu * drift_scale


# ═════════════════════════════════════════════════════════════════════════════
# Experiment-level Numba kernel
# ═════════════════════════════════════════════════════════════════════════════


@njit
def _tsdm_experiment_kernel(
    wc, b, mu_r, sigma_r, r_pc, r_pa, kp, kd,
    te_min, te_max, num_trials, sigma, dt, max_steps,
):
    """Numba-compiled TSDM experiment simulator.

    For each trial:
      1. Sample non-decision time t_nd ~ TruncatedNormal(mu_r, sigma_r).
      2. Sample motor execution time te ~ Uniform(te_min, te_max).
      3. Compute perceptual stage duration tp = t_nd - te.
      4. Numerically integrate perceptual inputs → Sp_a, Sp_c.
      5. Run drift-diffusion with time-varying drift rate until boundary hit.

    Parameters:
        wc: Controlled-channel attention weight [0, 2].
        b: Decision boundary (ms).
        mu_r: Mean non-decision time (ms).
        sigma_r: SD of non-decision time (ms).
        r_pc: Controlled-channel base drift rate.
        r_pa: Automatic-channel base drift rate.
        kp: Perceptual attention shift rate.
        kd: Incongruent decision attention shift rate.
        te_min, te_max: Motor execution time range (ms).
        num_trials: Number of trials per experiment.
        sigma: Diffusion coefficient (fixed at 4.0).
        dt: Time step (ms, fixed at 1.0).
        max_steps: Maximum decision-stage steps.

    Returns:
        (rt, choice, condition): arrays of response times (ms), choices,
        and condition labels.
    """
    condition = _make_balanced_conditions(num_trials)
    n_trials = condition.shape[0]
    rt = np.empty(n_trials, dtype=np.float64)
    choice = np.empty(n_trials, dtype=np.float64)
    sqrt_dt = np.sqrt(dt)

    # Drift-rate scaling: perceptual evidence values can be large
    # (on the order of 10^3–10^4).  Dividing by 1000 maps drift rates
    # into a range that yields plausible decision times (≈ 50–500 ms)
    # when combined with boundaries b ∈ [45, 85].
    drift_scale = 1.0 / 1000.0

    wc_resource = 2.0 - wc  # initial auto-channel attention resources

    for i in range(n_trials):
        cond_code = condition[i]

        # ── Sample non-decision time and motor time ───────────────────────
        ndt_lower = 0.0
        ndt_upper = 2.0 * mu_r + 5.0 * sigma_r

        # Guard against degenerate tp: resample t_nd until tp > 0
        for _ in range(100):
            t_nd = _draw_truncated_normal(mu_r, sigma_r, ndt_lower, ndt_upper)
            te = np.random.uniform(te_min, te_max)
            tp = t_nd - te
            if tp >= 1.0:
                break
        else:
            # Fallback (extremely unlikely)
            tp = 1.0
            t_nd = te + tp

        # ── Perceptual stage: accumulate Sp_a, Sp_c ──────────────────────
        Sp_a, Sp_c = _compute_perceptual_accumulation(r_pa, r_pc, wc, kp, tp)

        # ── Decision stage: Euler-Maruyama diffusion ─────────────────────
        evidence = 0.0
        hit_boundary = False

        for step in range(1, max_steps + 1):
            t_dec = float(step) * dt  # decision time so far (ms)

            if cond_code == 1:
                # Congruent: evidence from both channels agrees
                mu_total = _drift_congruent(
                    t_dec, tp, wc_resource, kp, Sp_a, Sp_c, drift_scale
                )
            else:
                # Incongruent: evidence from channels opposes
                mu_total = _drift_incongruent(
                    t_dec, wc_resource, kd, Sp_a, Sp_c, drift_scale
                )

            noise = sigma * sqrt_dt * np.random.normal()
            evidence += mu_total * dt + noise

            if evidence >= b:
                rt[i] = t_dec + t_nd   # decision time + non-decision time (ms)
                choice[i] = 1.0         # correct (upper boundary)
                hit_boundary = True
                break
            elif evidence <= -b:
                rt[i] = t_dec + t_nd
                choice[i] = 0.0         # error (lower boundary)
                hit_boundary = True
                break

        if not hit_boundary:
            rt[i] = float(max_steps) * dt + t_nd
            # Force choice by sign of final evidence position
            choice[i] = 1.0 if evidence >= 0.0 else 0.0

    return rt, choice, condition


# ═════════════════════════════════════════════════════════════════════════════
# Python simulator wrapper
# ═════════════════════════════════════════════════════════════════════════════


def tsdm_experiment_simulator(
    wc: float, b: float, mu_r: float, sigma_r: float,
    r_pc: float, r_pa: float, kp: float, kd: float,
    te: float = 115.0,
    te_min: float = 100.0, te_max: float = 130.0,
    num_trials: int = 50,
    sigma: float = 4.0, dt: float = 1.0,
    max_steps: int = 10000,
):
    """Experiment-level TSDM simulator.

    Simulates a full experiment (balanced congruent/incongruent trials) for
    a single subject with the given TSDM parameter set.

    Parameters:
        wc: Controlled-channel attention weight.
        b: Decision boundary (ms).
        mu_r: Mean non-decision time (ms).
        sigma_r: SD of non-decision time (ms).
        r_pc: Controlled-channel base drift rate.
        r_pa: Automatic-channel base drift rate.
        kp: Perceptual attention shift rate.
        kd: Incongruent decision attention shift rate.
        te: Motor execution time baseline (ms) — when te_min==te_max==te,
            motor time is deterministic at this value.

    Returns:
        dict with keys:
            "rt":       Response times in seconds (float64).
            "choice":   1.0 = correct, 0.0 = error.
            "condition": 1.0 = congruent, 0.0 = incongruent.
    """
    num_trials = int(num_trials)
    max_steps = int(max_steps)

    rt_ms, choice, condition = _tsdm_experiment_kernel(
        wc, b, mu_r, sigma_r, r_pc, r_pa, kp, kd,
        te_min, te_max, num_trials, sigma, dt, max_steps,
    )

    # Convert RT from milliseconds to seconds for NSBI framework compatibility
    return {
        "rt": rt_ms / 1000.0,
        "choice": choice,
        "condition": condition.astype(np.float64),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Model configuration
# ═════════════════════════════════════════════════════════════════════════════

TSDM_CONFIG = {
    "prior_range": {
        "wc":      [0.0, 2.0],      # controlled-channel attention weight
        "b":       [45.0, 85.0],     # decision boundary (ms)
        "mu_r":    [200.0, 400.0],   # mean non-decision time (ms)
        "sigma_r": [20.0, 50.0],     # SD of non-decision time (ms)
        "r_pc":    [0.0, 5.0],       # controlled-channel base drift rate
        "r_pa":    [0.0, 5.0],       # automatic-channel base drift rate
        "kp":      [1.0, 5.0],       # perceptual attention shift rate
        "kd":      [2.0, 6.0],       # incongruent decision shift rate
    },
    "param_keys": [
        "wc", "b", "mu_r", "sigma_r",
        "r_pc", "r_pa", "kp", "kd",
    ],
    "param_names": [
        r"$w_c$", r"$b$", r"$\mu_R$", r"$\sigma_R$",
        r"$r_{pc}$", r"$r_{pa}$", r"$k_p$", r"$k_d$",
    ],
    "simulator_type": "experiment",
}


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

MODEL_REG_NAME = "tsdm"
STORE_KEY_PREFIX = "tsdm"
DEFAULT_CHECKPOINT = "../../checkpoints/tsdm"
_PARAM_KEYS = TSDM_CONFIG["param_keys"]


def register():
    """Register the TSDM simulator and config into global registries.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    import simulators
    import default_settings

    if MODEL_REG_NAME in default_settings.MODEL_CONFIG:
        return  # already registered

    simulators.TRIAL_SIMULATOR[MODEL_REG_NAME] = tsdm_experiment_simulator
    default_settings.MODEL_CONFIG[MODEL_REG_NAME] = TSDM_CONFIG
    default_settings.PARAMS_KEY_NAME_MAPPING = default_settings.get_param_mappings(
        default_settings.MODEL_CONFIG
    )

    # Create alias for downstream analysis code
    from default_settings import PARAMS_KEY_NAME_MAPPING
    PARAMS_KEY_NAME_MAPPING[STORE_KEY_PREFIX] = (
        PARAMS_KEY_NAME_MAPPING[MODEL_REG_NAME]
    )


def get_tsdm_model(checkpoint_path=DEFAULT_CHECKPOINT):
    """Return a ready-to-use NSBICDM instance for the TSDM model.

    Automatically registers the model on first call.

    Usage:
        from tsdm_loader import get_tsdm_model
        model = get_tsdm_model()
        sim_data = model.simulate_data(n_trial=500, params=[{...}])
        history = model.run(epochs=250, batch_size=32, num_batches_per_epoch=200)
    """
    from NSBI_CDMs import NSBICDM
    register()
    return NSBICDM(MODEL_REG_NAME, checkpoint_path=checkpoint_path)
