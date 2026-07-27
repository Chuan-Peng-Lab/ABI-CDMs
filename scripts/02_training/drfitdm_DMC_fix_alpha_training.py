#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys

import simulators as simulators
import default_settings as default_settings
from NSBI_CDMs import NSBICDM
from plotting import plot_rt_dists
import time

%load_ext autoreload
%autoreload 2

# ## Generative model (Simulator)

# In[ ]:


# from driftdm_dmc import DMCParameters, DMCModel


# def driftdm_dmc_experiment_simulator(
#     muc: float,
#     b: float,
#     non_dec: float,
#     sd_non_dec: float,
#     tau: float,
#     A: float,
#     a_shape: float = 2,
#     num_trials: int = 50,
#     alpha: float = 4.0,
#     sigma: float = 1.0,
#     t_max: float = 3.0,
#     dt: float = 0.0075,
#     dx: float = 0.02,
#     var_non_dec: bool = True,
#     var_start: bool = True,
#     seed: int | None = None,
# ):
#     """
#     Experiment-level simulator for driftdm_dmc.

#     This function directly generates multiple trials for one simulated subject.

#     Output convention:
#         rt:
#             Response time in seconds.

#         choice:
#             Accuracy-coded response.
#             Here, the upper boundary is treated as the correct response.
#             Therefore, raw upper-boundary choice = 1 is kept as accuracy = 1,
#             and lower-boundary choice = 0 is treated as accuracy = 0.

#         condition:
#             Congruency-coded condition.
#             Compatible condition is coded as 1.
#             Incompatible condition is coded as 0.

#     Notes:
#         The original driftdm_dmc model uses string labels:
#             "comp" and "incomp"

#         The NSBI framework expects numeric labels:
#             1 and 0

#         Therefore, this function maps:
#             "comp"   -> 1
#             "incomp" -> 0
#     """

#     if seed is not None:
#         np.random.seed(seed)

#     num_trials = int(num_trials)

#     params = DMCParameters(
#         muc=muc,
#         b=b,
#         non_dec=non_dec,
#         sd_non_dec=sd_non_dec,
#         tau=tau,
#         a=a_shape,
#         A=A,
#         alpha=alpha,
#         sigma=sigma,
#     )

#     model = DMCModel(
#         params=params,
#         t_max=t_max,
#         dt=dt,
#         dx=dx,
#         var_non_dec=var_non_dec,
#         var_start=var_start,
#     )

#     # Create a balanced condition vector.
#     # 1 = compatible, 0 = incompatible.
#     half = num_trials // 2
#     condition_codes = np.array([1] * half + [0] * half, dtype=np.int32)

#     # If the number of trials is odd, add one randomly selected condition.
#     if num_trials % 2 == 1:
#         extra_condition = np.random.choice([0, 1])
#         condition_codes = np.append(condition_codes, extra_condition)

#     # Shuffle trial order.
#     np.random.shuffle(condition_codes)

#     rt = np.empty(num_trials, dtype=np.float64)
#     choice = np.empty(num_trials, dtype=np.int32)
#     condition = np.empty(num_trials, dtype=np.int32)

#     for i, cond_code in enumerate(condition_codes):
#         cond_label = "comp" if cond_code == 1 else "incomp"

#         trial = model.simulate_trial(condition=cond_label)

#         rt[i] = trial["rt"]

#         # Upper boundary is defined as the correct response.
#         # Therefore, raw choice can be used as accuracy.
#         choice[i] = 1 if trial["choice"] == 1 else 0

#         # Numeric condition coding required by NSBI.
#         condition[i] = cond_code

#     return {
#         "rt": rt,
#         "choice": choice,
#         "condition": condition,
#     }


# In[4]:


import numpy as np
from numba import njit


@njit(fastmath=True)
def _dmc_drift_rate(t, muc, tau, a_shape, A_cond):
    """
    Compute the DMC time-varying drift rate.

    Total drift:
        mu(t) = muc + mu_auto(t)

    For a_shape = 2:
        mu_auto(t) = A / tau * exp(1 - t / tau) * (1 - t / tau)

    For general a_shape:
        this follows the derivative-of-gamma formulation used in DMC.
    """

    if abs(a_shape - 2.0) < 1e-8:
        mu_auto = (A_cond / tau) * np.exp(1.0 - t / tau) * (1.0 - t / tau)
    else:
        # Small offset avoids numerical instability at t = 0.
        t_adj = t + 1e-6

        base = (t_adj * np.e) / ((a_shape - 1.0) * tau)
        exp_part = np.exp(-t_adj / tau)
        poly_part = base ** (a_shape - 1.0)
        deriv_part = (a_shape - 1.0) / t_adj - 1.0 / tau

        mu_auto = A_cond * exp_part * poly_part * deriv_part

    return muc + mu_auto


@njit
def _draw_truncated_normal(mean, sd, lower, upper):
    """
    Draw one sample from a truncated normal distribution.

    This is used for non-decision time variability.
    A small rejection loop is sufficient because sd_non_dec is usually small.
    """

    if sd <= 0.0:
        return mean

    for _ in range(100):
        x = mean + sd * np.random.normal()
        if lower <= x <= upper:
            return x

    # Fallback if rejection sampling fails.
    if x < lower:
        return lower
    if x > upper:
        return upper
    return x


@njit
def _shuffle_int_array(x):
    """
    In-place Fisher-Yates shuffle for a 1D integer array.
    """

    n = x.shape[0]
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = x[i]
        x[i] = x[j]
        x[j] = tmp


@njit
def _make_balanced_conditions(num_trials):
    """
    Create a balanced condition vector.

    Coding:
        1 = compatible
        0 = incompatible
    """

    condition = np.empty(num_trials, dtype=np.int32)

    half = num_trials // 2

    for i in range(half):
        condition[i] = 1

    for i in range(half, num_trials):
        condition[i] = 0

    # If num_trials is odd, randomly overwrite the last trial.
    if num_trials % 2 == 1:
        condition[num_trials - 1] = np.random.randint(0, 2)

    _shuffle_int_array(condition)

    return condition


@njit
def _driftdm_dmc_experiment_kernel(
    muc,
    b,
    non_dec,
    sd_non_dec,
    tau,
    a_shape,
    A,
    num_trials,
    sigma,
    dt,
    max_steps,
    var_non_dec,
    return_nan
):
    """
    Numba-compiled experiment-level DMC simulator.

    This function simulates all trials for one synthetic subject.

    Boundary convention:
        upper boundary  = correct response
        lower boundary  = error response

    Output coding:
        choice:
            1 = correct
            0 = error

        condition:
            1 = compatible
            0 = incompatible
    """

    condition = _make_balanced_conditions(num_trials)

    rt = np.empty(num_trials, dtype=np.float64)
    choice = np.empty(num_trials, dtype=np.float64)

    sqrt_dt = np.sqrt(dt)

    for i in range(num_trials):

        cond_code = condition[i]

        if cond_code == 1:
            A_cond = A
        else:
            A_cond = -A

        # Fixed starting point at the center between two boundaries.
        # Evidence space is [-b, +b].
        evidence = 0.0

        # Draw non-decision time.
        if var_non_dec:
            ndt = _draw_truncated_normal(
                non_dec,
                sd_non_dec,
                0.0,
                2.0 * non_dec + 5.0 * sd_non_dec
            )
        else:
            ndt = non_dec

        hit_boundary = False
        step = 0

        while step < max_steps:
            step += 1
            t = step * dt

            drift = _dmc_drift_rate(
                t,
                muc,
                tau,
                a_shape,
                A_cond
            )

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
                # Forced choice at timeout.
                # Since upper boundary is correct, positive evidence is coded as correct.
                if evidence >= 0.0:
                    choice[i] = 1.0
                else:
                    choice[i] = 0.0

    return rt, choice, condition


def driftdm_dmc_experiment_simulator(
    muc: float,
    b: float,
    non_dec: float,
    sd_non_dec: float,
    tau: float,
    A: float,
    a_shape: float = 2.0,
    num_trials: int = 50,
    sigma: float = 1.0,
    dt: float = 0.0075,
    max_steps: int = 400,
    var_non_dec: bool = True,
    return_nan: bool = False,
):
    """
    Fast experiment-level simulator for driftdm-style DMC.

    This function is designed for NSBI / BayesFlow training.
    It does not depend on driftdm_dmc.py.

    Parameters expected by the neural prior:
        muc
        b
        non_dec
        sd_non_dec
        tau
        a_shape
        A

    Fixed simulation parameters:
        sigma:
            Diffusion noise scale. Usually fixed to 1.0 for identifiability.

        dt:
            Time-step size in seconds.

        max_steps:
            Maximum number of accumulation steps.
            Effective maximum decision time = dt * max_steps.

        var_non_dec:
            Whether to use trial-level non-decision time variability.

    Returns:
        dict with keys:
            rt
            choice
            condition
    """

    num_trials = int(num_trials)
    max_steps = int(max_steps)

    rt, choice, condition = _driftdm_dmc_experiment_kernel(
        muc,
        b,
        non_dec,
        sd_non_dec,
        tau,
        a_shape,
        A,
        num_trials,
        sigma,
        dt,
        max_steps,
        var_non_dec,
        return_nan,
    )

    return {
        "rt": rt,
        "choice": choice,
        "condition": condition.astype(np.float64),
    }

# In[17]:


driftdm_dmc_config = {
    "prior_range": {
        "muc": [0.5, 9.0],
        "b": [0.15, 1.20],
        "non_dec": [0.15, 0.60],
        "sd_non_dec": [0.005, 0.10],
        "tau": [0.015, 0.25],
        # "a_shape": [1.2, 3.0],
        "A": [0.005, 0.30],
    },
    "param_keys": [
        "muc",
        "b",
        "non_dec",
        "sd_non_dec",
        "tau",
        # "a_shape",
        "A",
    ],
    "param_names": [
        r"$\mu_c$",
        r"$b$",
        r"$t_{er}$",
        r"$s_{t}$",
        r"$\tau$",
        # r"$a_{shape}$",
        r"$A$",
    ],

    # Tell load_generator that this is already an experiment-level simulator.
    "simulator_type": "experiment",
}


# Register the experiment-level simulator.
simulators.TRIAL_SIMULATOR["driftdm_dmc"] = driftdm_dmc_experiment_simulator

# Register the model configuration.
default_settings.MODEL_CONFIG["driftdm_dmc"] = driftdm_dmc_config

# Refresh parameter-name mappings.
default_settings.PARAMS_KEY_NAME_MAPPING = default_settings.get_param_mappings(
    default_settings.MODEL_CONFIG
)


# Initialize the NSBI model.
driftdm_model = NSBICDM("driftdm_dmc")


# Optional prior predictive check.
test_params = {
    "muc": 4.0,
    "b": 0.6,
    "non_dec": 0.3,
    "sd_non_dec": 0.02,
    "tau": 0.04,
    "A": 0.1,
}

sim_data = driftdm_model.simulate_data(
    n_trial=500,
    params=[test_params],
)

sim_data.head()

# In[16]:


sim_data.groupby(["congruency", "accuracy"])["rt"].describe()

# ### prior check
# 
# adapted from flexDDM

# In[6]:


plot_rt_dists(sim_data, flip=True)

# In[ ]:


sim_data10 = driftdm_model.simulate_data(10, n_trial=500)

plot_rt_dists(sim_data10)

# ## Train

# In[21]:


# print cost time
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# In[22]:


history = driftdm_model.run(
    epochs=250,
    batch_size=32,
    num_batches_per_epoch=200, 
    # verbose=0,
    # save=False, 
    keep_optimizer=True
)

# In[24]:


# print cost time
import time
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# ### check

# In[23]:


driftdm_model.plot_trained_result()
