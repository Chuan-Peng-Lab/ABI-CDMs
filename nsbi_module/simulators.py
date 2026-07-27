import numpy as np
from numba import njit
from numba_stats import norm
from typing import Callable
import inspect
##------------------------------------------------
#                drift function
#------------------------------------------------

@njit(fastmath=True)
def SSP_drift(t: float, p: float, sda: float, rd: float) -> np.array:
    """Drift function of Spotlight Shrinkage model for Diffusion model in conflict tasks by Evans et al. (2020)

    Args:
        t (float): time of sequential sampling in seconds
        p (float): point of drift rate
        sda (float): inital deviation of drift rate
        rd (float): reduction rate of inital deviation

    Returns:
        np.array: np.array include scalar drift rate
    """
    # calculate current sd of spotlight
    sd_t = sda - (rd * t)
    sd_t = max(sd_t, 0.001)

    x = np.array([0.5], dtype=np.float32)
    # find area of spotlight over target and flanker
    # NOTE numba_stats 1.7.0: norm.cdf only support x is np.array, mu and sigma is float
    a_target = norm.cdf(x, 0.0, sd_t) - norm.cdf(-x, 0.0, sd_t)
    a_flanker = 1 - a_target

    # current drift rate
    drift = p * (a_target - a_flanker)
    # drift = 2 * p * (a_target - 0.5)

    return drift

@njit(fastmath=True)
def SSPfsrr_drift(t: float, p: float, rd_sda_ratio: float) -> np.array:
    """Drift function of Spotlight Shrinkage model for Diffusion model in conflict tasks by Evans et al. (2020)

    Args:
        t (float): time of sequential sampling in seconds
        p (float): point of drift rate
        rd_sda_ratio (float): ratio of sda to rd

    Returns:
        np.array: np.array include scalar drift rate
    """
    # calculate current sd of spotlight
    sd_t = 1 - (rd_sda_ratio * t)   # sd_t = sda - (rd * t) >> sd_t = sda - (sda * rd_sda_ratio * t) with sda = 1
    sd_t = max(sd_t, 0.001)
    

    x = np.array([0.5], dtype=np.float32)
    # find area of spotlight over target and flanker
    # NOTE numba_stats 1.7.0: norm.cdf only support x is np.array, mu and sigma is float
    a_target = norm.cdf(x, 0.0, sd_t) - norm.cdf(-x, 0.0, sd_t)
    a_flanker = 1 - a_target

    # current drift rate
    drift = p * (a_target - a_flanker)
    # drift = 2 * p * (a_target - 0.5)

    return drift

@njit(fastmath=True)
def DMC_drift(t: float, vc: float, shape: float, peak: float, tau: float) -> float:
    """Drift function for Diffusion model in conflict tasks that follows a gamma function by Evans et al. (2020)

    Arguments
    ---------
        t: float
            Timepoints at which to evaluate the drift.
        vc: float
            The drift for Processing Efficiency. 
        shape: float 
            Shape parameter of the gamma distribution
        peak: float
            Scalar parameter that scales the peak of
            the gamma distribution.
            (Note this function follows a gamma distribution
            but does not integrate to 1)
        tau: float
            tau is the characteristic time.

    Return
    ------
        float
            The gamma drift evaluated at the supplied timepoints t without congruency condition.
    """

    t = max(t, 0.01)
    term1 = peak * np.exp(-t / tau)
    term2 = np.power((t * np.e) / ((shape - 1) * tau), (shape - 1))
    term3 = ((shape - 1) / t) - (1 / tau)
    va = term1 * term2 * term3

    drift = va + vc
    return drift

##------------------------------------------------
#                Trial simulator
#------------------------------------------------
@njit
def DDM_trial(a, ndt, v, dc=1.0, dt=0.001, max_steps=2e5, return_NAN=True):
    """
    Simulates a trial from the Drift Diffusion Model (DDM).
    
    Parameters:
    - a (float): Decision boundary separation
    - ndt (float): Non-decision time (ms)
    - v (float): Drift rate
    - dc (float, optional): Diffusion coefficient. Defaults to 1.0.
    - dt (float, optional): Time step size (ms). Defaults to 0.001.
    - max_steps (float, optional): Maximum number of simulation steps. Defaults to 2e5.
    - return_NAN (bool, optional): Whether to return NaN for undecided trials. Defaults to True.
    
    Returns:
    - rt (float): Response time in seconds
    - choice (int): Choice made (-1 or 1)
    - v_view (list): Drift values over time
    - trajectory (list): Evidence accumulation over time
    """
    
    n_steps = 0.0
    evidence = a * 0.5

    v_view = [v]
    trajectory = [evidence]
    
    # Simulate a single DM path
    while evidence > 0 and evidence < a and n_steps < max_steps:
        # DDM equation
        evidence += v * dt + np.sqrt(dt) * dc * np.random.normal()
        trajectory.append(evidence)

        # Increment step
        n_steps += 1.0
        v_view.append(v)

    rt = n_steps * dt + ndt

    if evidence >= a:
        choice = 1
    elif evidence <= 0:
        choice = -1
    elif return_NAN:
        choice = np.nan
        rt = np.nan
    else:
        choice = 1 if evidence >= trajectory[0] else -1

    return rt, choice, v_view, trajectory

@njit
def DDM_conflict_trial(a, ndt, v_c, v_i, congruency=1, dc=1.0, dt=0.001, max_steps=2e5, return_NAN=True):
    """
    Simulates a trial from the Drift Diffusion Model with conflict conditions.
    
    Parameters:
    - a (float): Decision boundary separation
    - ndt (float): Non-decision time (ms)
    - v_c (float): Drift rate for congruent trials
    - v_i (float): Drift rate for incongruent trials
    - congruency (int, optional): Congruency condition (1=congruent, -1=incongruent). Defaults to 1.
    - dc (float, optional): Diffusion coefficient. Defaults to 1.0.
    - dt (float, optional): Time step size (ms). Defaults to 0.001.
    - max_steps (float, optional): Maximum number of simulation steps. Defaults to 2e5.
    - return_NAN (bool, optional): Whether to return NaN for undecided trials. Defaults to True.
    
    Returns:
    - rt (float): Response time in seconds
    - choice (int): Choice made (-1 or 1)
    - v_view (list): Drift values over time
    - trajectory (list): Evidence accumulation over time
    """
    v = v_c if congruency == 1 else v_i
    return DDM_trial(a, ndt, v, dc=dc, dt=dt, max_steps=max_steps, return_NAN=return_NAN)

@njit
def SSP_trial(a, ndt, p, sd_a, r_d, congruency=1, dc=1.0, dt=0.001, max_steps=2e5, return_NAN=True):
    """
    Simulates a trial from the Stimulus Selection Process model.
    
    Parameters:
    - a (float): Decision boundary separation
    - ndt (float): Non-decision time (ms)
    - p (float): Baseline drift rate
    - sd_a (float): Drift rate variability
    - r_d (float): Response decay parameter
    - congruency (int, optional): Congruency condition (1=congruent, -1=incongruent). Defaults to 1.
    - dc (float, optional): Diffusion coefficient. Defaults to 1.0.
    - dt (float, optional): Time step size (ms). Defaults to 0.001.
    - max_steps (float, optional): Maximum number of simulation steps. Defaults to 2e5.
    - return_NAN (bool, optional): Whether to return NaN for undecided trials. Defaults to True.
    
    Returns:
    - rt (float): Response time in seconds
    - choice (int): Choice made (-1 or 1)
    - v_view (list): Drift values over time
    - trajectory (list): Evidence accumulation over time
    """
    
    evidence = a * 0.5  # fix start point to 0.5
    n_steps = 0.0

    v_view = []
    trajectory = []

    # Simulate a single DM path
    while evidence > 0 and evidence < a and n_steps < max_steps:
        if congruency == 1:
            drift = p
        else:
            # NOTE: SSP_drift return array, so we need [0]
            drift = SSP_drift(n_steps * dt, p, sd_a, r_d)[0].item()

        trajectory.append(evidence)
        v_view.append(drift)

        # DDM equation
        evidence += drift * dt + np.sqrt(dt) * dc * np.random.normal()

        # Increment step
        n_steps += 1.0

    rt = n_steps * dt + ndt

    if evidence >= a:
        choice = 1
    elif evidence <= 0:
        choice = -1
    elif return_NAN:
        choice = np.nan
        rt = np.nan
    else:
        choice = 1 if evidence >= trajectory[0] else -1

    return rt, choice, v_view, trajectory

@njit
def DMC_trial(a, ndt, v_c, alpha, eta, tau, congruency=1, dc=1.0, dt=0.001, max_steps=2e5, return_NAN=True):
    """
    Simulates a single trial of the Drift Diffusion Model for decision making with time-varying drift.
    
    Parameters:
    - a (float): Decision boundary separation
    - ndt (float): Non-decision time (ms)
    - v_c (float): Peak drift rate
    - alpha (float): Peak time of drift rate
    - eta (float): Shape parameter of drift rate function
    - tau (float): Time constant of drift rate function
    - congruency (int, optional): Congruency condition (1=congruent, -1=incongruent). Defaults to 1.
    - dc (float, optional): Diffusion coefficient. Defaults to 1.0.
    - dt (float, optional): Time step size (ms). Defaults to 0.001.
    - max_steps (float, optional): Maximum number of simulation steps. Defaults to 2e5.
    - return_NAN (bool, optional): Whether to return NaN for undecided trials. Defaults to True.
    
    Returns:
    - rt (float): Response time in seconds
    - choice (int): Choice made (-1 or 1)
    - v_view (list): Drift values over time
    - trajectory (list): Evidence accumulation over time
    """
    
    n_steps = 0.0
    evidence = a * 0.5  # fix start point to 0.5
    congruency = 1 if congruency == 1 else -1
    trajectory = [evidence]
    v_view = [v_c]
    
    # Simulate a single DM path
    while evidence > 0 and evidence < a and n_steps < max_steps:
        n_steps += 1.0
        
        # DDM equation
        v = DMC_drift(n_steps*dt*1000, v_c, alpha, eta * congruency, tau)
        evidence += v * dt + np.sqrt(dt) * dc * np.random.normal()

        v_view.append(v)
        trajectory.append(evidence)

    rt = n_steps * dt + ndt

    if evidence >= a:
        choice = 1
    elif evidence <= 0:
        choice = -1
    elif return_NAN:
        choice = np.nan
        rt = np.nan
    else:
        choice = 1 if evidence >= trajectory[0] else -1

    return rt, choice, v_view, trajectory

@njit
def DSTP_trial(a, ndt, vta, vfl, vss, vp2, ass, congruency=1, dc=1.0, dt=.001, max_steps=2e5, return_NAN=True):
    """
    Simulates a trial from the Dual Stage Two-Phase model.
    
    Parameters:
    - a (float): Decision boundary separation
    - ndt (float): Non-decision time (ms)
    - vta (float): Target drift rate in first phase
    - vfl (float): Flanker drift rate in first phase
    - vss (float): Stimulus selection drift rate
    - vp2 (float): Drift rate in second phase
    - ass (float): Stimulus selection boundary
    - congruency (int, optional): Congruency condition (1=congruent, -1=incongruent). Defaults to 1.
    - dc (float, optional): Diffusion coefficient. Defaults to 1.0.
    - dt (float, optional): Time step size (ms). Defaults to 0.001.
    - max_steps (float, optional): Maximum number of simulation steps. Defaults to 2e5.
    - return_NAN (bool, optional): Whether to return NaN for undecided trials. Defaults to True.
    
    Returns:
    - rt (float): Response time in seconds
    - choice (int): Choice made (-1 or 1)
    - v_view (list): Drift values over time
    - trajectory (list): Evidence accumulation over time
    """
    
    n_steps = 0.
    sqrt_st = np.sqrt(dt) * dc
    evidence = a * 0.5  # fix start point to 0.5
    # initiate the X of start point for stimulus selection
    X_ss = ass * 0.5  # fix start point of selection to 0.5
    
    # Drift rate in the first phase
    congruency = 1 if congruency == 1 else -1
    drift = vta + congruency * vfl

    v_view = [drift]
    trajectory = [evidence]

    # Simulate a single DM path
    while (evidence > 0 and evidence < a and n_steps < max_steps):
        noise = np.random.normal(loc=0, scale=1, size=2)
        # Stimulus selection
        X_ss += vss * dt + noise[0] * sqrt_st
        # Drift rate in the second phase
        if X_ss >= ass:  # select target
            drift = vp2
        elif X_ss <= 0:  # select flanker
            # the flanker have same direction as to target or error
            drift = vp2 if congruency == 1 else -vp2
        
        v_view.append(drift)

        # DDM equation
        evidence += drift * dt + noise[1] * sqrt_st
        
        trajectory.append(evidence)

        # Increment step
        n_steps += 1.0

    rt = n_steps * dt + ndt
    
    if evidence >= a:
        choice = 1
    elif evidence <= 0:
        choice = -1
    elif return_NAN:
        choice = np.nan
        rt = np.nan
    else:
        choice = 1 if evidence >= trajectory[0] else -1

    return rt, choice, v_view, trajectory

##------------------------------------------------
#                Trial Simulator with fixed parameters
#------------------------------------------------
@njit
def DMC_fixed_alpha_trial(a, ndt, v_c, eta, tau, congruency=1, dc=1.0, dt=0.001, max_steps=2e5, return_NAN=True):
    """
    Simulates a single trial of the Drift Diffusion Model with fixed alpha parameter.
    
    Parameters:
    - a (float): Decision boundary separation
    - ndt (float): Non-decision time (ms)
    - v_c (float): Peak drift rate
    - eta (float): Peak time of drift rate
    - tau (float): Time constant of drift rate function
    - congruency (int, optional): Congruency condition (1=congruent, -1=incongruent). Defaults to 1.
    - dc (float, optional): Diffusion coefficient. Defaults to 1.0.
    - dt (float, optional): Time step size (ms). Defaults to 0.001.
    - max_steps (float, optional): Maximum number of simulation steps. Defaults to 2e5.
    - return_NAN (bool, optional): Whether to return NaN for undecided trials. Defaults to True.
    
    Returns:
    - rt (float): Response time in seconds
    - choice (int): Choice made (-1 or 1)
    - v_view (list): Drift values over time
    - trajectory (list): Evidence accumulation over time
    """
    
    n_steps = 0.0
    alpha = 2  # also termed alpha, fix to 2
    evidence = a * 0.5  # fix start point to 0.5
    congruency = 1 if congruency == 1 else -1
    trajectory = [evidence]
    v_view = [v_c]
    
    # Simulate a single DM path
    while evidence > 0 and evidence < a and n_steps < max_steps:
        n_steps += 1.0
        
        # DDM equation
        v = DMC_drift(n_steps*dt*1000, v_c, alpha, eta * congruency,  tau)
        evidence += v * dt + np.sqrt(dt) * dc * np.random.normal()

        v_view.append(v)
        trajectory.append(evidence)

    rt = n_steps * dt + ndt

    if evidence >= a:
        choice = 1
    elif evidence <= 0:
        choice = -1
    elif return_NAN:
        choice = np.nan
        rt = np.nan
    else:
        choice = 1 if evidence >= trajectory[0] else -1

    return rt, choice, v_view, trajectory

@njit
def DSTP_fixed_ratio_trial(a, ndt, vta, vfl_vss_ratio, vp2, ass, congruency=1, dc=1.0, dt=.001, max_steps=2e5, return_NAN=True):
    """
    Simulates a trial from the Dual Stage Two-Phase model with fixed ratio between vfl and vss.
    
    Parameters:
    - a (float): Decision boundary separation
    - ndt (float): Non-decision time (ms)
    - vta (float): Target drift rate in first phase
    - vfl_vss_ratio (float): Ratio between vfl and vss
    - vp2 (float): Drift rate in second phase
    - ass (float): Stimulus selection boundary
    - congruency (int, optional): Congruency condition (1=congruent, -1=incongruent). Defaults to 1.
    - dc (float, optional): Diffusion coefficient. Defaults to 1.0.
    - dt (float, optional): Time step size (ms). Defaults to 0.001.
    - max_steps (float, optional): Maximum number of simulation steps. Defaults to 2e5.
    - return_NAN (bool, optional): Whether to return NaN for undecided trials. Defaults to True.
    
    Returns:
    - rt (float): Response time in seconds
    - choice (int): Choice made (-1 or 1)
    - v_view (list): Drift values over time
    - trajectory (list): Evidence accumulation over time
    """
    
    n_steps = 0.
    sqrt_st = np.sqrt(dt) * dc
    evidence = a * 0.5  # fix start point to 0.5
    # initiate the X of start point for stimulus selection
    X_ss = ass * 0.5  # fix start point of selection to 0.5
    
    # Drift rate in the first phase
    vss = 4
    vfl = vss * vfl_vss_ratio
    congruency = 1 if congruency == 1 else -1
    drift = vta + congruency * vfl

    v_view = [drift]
    trajectory = [evidence]

    # Simulate a single DM path
    while (evidence > 0 and evidence < a and n_steps < max_steps):
        noise = np.random.normal(loc=0, scale=1, size=2)
        # Stimulus selection
        X_ss += vss * dt + noise[0] * sqrt_st
        # Drift rate in the second phase
        if X_ss >= ass:  # select target
            drift = vp2
        elif X_ss <= 0:  # select flanker
            # the flanker have same direction as to target or error
            drift = vp2 * congruency
        
        v_view.append(drift)

        # DDM equation
        evidence += drift * dt + noise[1] * sqrt_st
        
        trajectory.append(evidence)

        # Increment step
        n_steps += 1.0

    rt = n_steps * dt + ndt
    
    if evidence >= a:
        choice = 1
    elif evidence <= 0:
        choice = -1
    elif return_NAN:
        choice = np.nan
        rt = np.nan
    else:
        choice = 1 if evidence >= trajectory[0] else -1

    return rt, choice, v_view, trajectory

@njit
def SSP_fixed_ratio_trial(a, ndt, p, rd_sda_ratio, congruency=1, dc=1.0, dt=0.001, max_steps=2e5, return_NAN=True):
    """
    Simulates a trial from the Stimulus Selection Process model with fixed ratio between rd and sda.
    
    Parameters:
    - a (float): Decision boundary separation
    - ndt (float): Non-decision time (ms)
    - p (float): Baseline drift rate
    - rd_sda_ratio (float): Ratio between rd and sda
    - congruency (int, optional): Congruency condition (1=congruent, -1=incongruent). Defaults to 1.
    - dc (float, optional): Diffusion coefficient. Defaults to 1.0.
    - dt (float, optional): Time step size (ms). Defaults to 0.001.
    - max_steps (float, optional): Maximum number of simulation steps. Defaults to 2e5.
    - return_NAN (bool, optional): Whether to return NaN for undecided trials. Defaults to True.
    
    Returns:
    - rt (float): Response time in seconds
    - choice (int): Choice made (-1 or 1)
    - v_view (list): Drift values over time
    - trajectory (list): Evidence accumulation over time
    """
    
    evidence = a * 0.5  # fix start point to 0.5
    n_steps = 0.0

    v_view = []
    trajectory = []

    # Simulate a single DM path
    while evidence > 0 and evidence < a and n_steps < max_steps:
        if congruency == 1:
            drift = p
        else:
            drift = SSPfsrr_drift(n_steps * dt, p, rd_sda_ratio)[0].item()

        trajectory.append(evidence)
        v_view.append(drift)

        # DDM equation
        evidence += drift * dt + np.sqrt(dt) * dc * np.random.normal()

        # Increment step
        n_steps += 1.0

    rt = n_steps * dt + ndt

    if evidence >= a:
        choice = 1
    elif evidence <= 0:
        choice = -1
    elif return_NAN:
        choice = np.nan
        rt = np.nan
    else:
        choice = 1 if evidence >= trajectory[0] else -1

    return rt, choice, v_view, trajectory

##------------------------------------------------
#                Experiment Simulator
#------------------------------------------------

TRIAL_SIMULATOR= {
    "DMC": DMC_trial,
    "DMC_fixed_alpha": DMC_fixed_alpha_trial,
    "SSP": SSP_trial,
    "SSP_fixed_ratio": SSP_fixed_ratio_trial,
    "DSTP": DSTP_trial,
    "DSTP_fixed_ratio": DSTP_fixed_ratio_trial,
    "DDM": DDM_conflict_trial
}


def CDMs_experiment_simulator_wrapper(
    trial_simulator:Callable, 
    excluded_params = ['congruency', 'dc', 'dt', 'max_steps', 'return_NAN'],
    n_output = 2,
    dc=1,
    dt=0.001,
    max_steps=1500,
    return_NAN=False
    ):
    """wrapper trial simulator to experiment simulator.

    Parameters
    ----------
    trial_simulator : Callable or str
        such as DMC_trial, SSP_trial or DSTP_trial

    Returns
    -------
    wrapper function. 

    Example
    -------
    >>> CDMs_experiment = CDMs_experiment_simulator_wrapper(DMC_trial)
    >>> params = {
    >>>     "a": 0.5,
    >>>     "ter": 0.30,
    >>>     "p": 4,
    >>>     "sd.start": 1.5,
    >>>     "rate": 20
    >>> }
    >>> params=np.array(list(params.values()))
    >>> context = np.array([
    >>>     [1, 1],
    >>>     [0, 1],
    >>>     [1, 0],
    >>> CDMs_experiment(params,context)
    """
    
    def _experiment_simulator(num_trials=50,**kwargs):
        
        sig = inspect.signature(trial_simulator)
        param_keys = [sig for sig in sig.parameters.keys() if sig not in excluded_params]
        params_tuple = tuple(kwargs[param] for param in param_keys)

        # Call the Numba-optimized version
        half = num_trials // 2
        condition = np.array([1] * half + [-1] * half)
        
        # If odd number, add one more randomly chosen value
        if num_trials % 2 == 1:
            last_value = np.random.choice([-1, 1])
            condition = np.append(condition, last_value)
        
        data = np.empty((num_trials, n_output))
        for i in range(num_trials):
            data[i] = trial_simulator(
                *params_tuple, 
                congruency=condition[i],
                dc=dc,
                dt=dt,
                max_steps=max_steps,
                return_NAN=return_NAN
            )[:n_output]

        # Ensure choice and condition are mapped to 0 and 1
        choice_data = data[:, 1]
        choice_data = np.where(choice_data == 1, 1, 0)
        condition_data = condition
        condition_data = np.where(condition_data == 1, 1, 0)

        return dict(rt=data[:,0], choice=choice_data, condition=condition_data)

    return _experiment_simulator
