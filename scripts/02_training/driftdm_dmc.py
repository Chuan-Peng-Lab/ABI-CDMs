"""
https://github.com/bucky2177/dRiftDM

DMC (Diffusion Model for Conflict Tasks) - Python Implementation
Based on Ulrich et al. (2015): Cognitive Psychology

This module implements the Diffusion Model for Conflict Tasks for modeling
response times and choice behavior in conflict tasks (Simon, Stroop, Flanker).
"""

import numpy as np
from scipy.stats import norm, beta
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DMCParameters:
    """DMC Model Parameters"""
    muc: float = 4.0          # Drift rate of controlled process
    b: float = 0.6            # Boundary separation
    non_dec: float = 0.3      # Non-decision time (seconds)
    sd_non_dec: float = 0.02  # SD of non-decision time
    tau: float = 0.04         # Scale parameter of gamma distribution
    a: float = 2              # Shape parameter of gamma distribution (recommended: 2)
    A: float = 0.1            # Amplitude of automatic process
    alpha: float = 4          # Shape parameter of beta distribution for starting point
    sigma: float = 1.0        # Diffusion constant
    
    # Parameter ranges for optimization
    PARAM_RANGES = {
        'muc': (0.5, 9.0),
        'b': (0.15, 1.20),
        'non_dec': (0.15, 0.60),
        'sd_non_dec': (0.005, 0.1),
        'tau': (0.015, 0.25),
        'a': (1.2, 3.0),
        'A': (0.005, 0.3),
        'alpha': (2, 8)
    }


class DMCModel:
    """
    Diffusion Model for Conflict Tasks
    
    The DMC combines two decision processes:
    - Controlled process with constant drift rate (muc)
    - Automatic process modeled as derivative of gamma function
    
    Total drift rate: μ(t) = muc + A/τ * e^(1 - t/τ) * (1 - t/τ)  [for a=2]
    """
    
    def __init__(
        self,
        params: Optional[DMCParameters] = None,
        t_max: float = 3.0,
        dt: float = 0.0075,
        dx: float = 0.02,
        var_non_dec: bool = True,
        var_start: bool = True
    ):
        """
        Initialize DMC Model
        
        Parameters
        ----------
        params : DMCParameters, optional
            Model parameters. Uses defaults if not provided.
        t_max : float
            Maximum time for simulation (seconds)
        dt : float
            Time step for discretization
        dx : float
            Space step for discretization
        var_non_dec : bool
            Include variability in non-decision time (truncated normal)
        var_start : bool
            Include variability in starting point (beta distribution)
        """
        self.params = params or DMCParameters()
        self.t_max = t_max
        self.dt = dt
        self.dx = dx
        self.var_non_dec = var_non_dec
        self.var_start = var_start
        
        # Create time and space grids
        self.t_vec = np.arange(0, t_max + dt, dt)
        self.x_vec = np.arange(-1, 1 + dx, dx)  # Evidence space
        
    def drift_rate(
        self,
        t: np.ndarray,
        condition: str = 'comp',
        include_auto: bool = True,
        tol: float = 0.001
    ) -> np.ndarray:
        """
        Calculate drift rate μ(t) at time t
        
        For compatible condition: A > 0 (automatic process aids)
        For incompatible condition: A < 0 (automatic process hinders)
        
        Parameters
        ----------
        t : np.ndarray
            Time values
        condition : str
            'comp' (compatible) or 'incomp' (incompatible)
        include_auto : bool
            Include automatic process component
        tol : float
            Tolerance for numerical stability when a ≠ 2
            
        Returns
        -------
        np.ndarray
            Drift rate at each time point
        """
        muc = self.params.muc
        tau = self.params.tau
        a = self.params.a
        A = self.params.A
        
        # Adjust amplitude based on condition
        A_cond = A if condition == 'comp' else -A
        
        # Controlled process: constant drift rate
        mu_controlled = muc
        
        # Automatic process: derivative of gamma distribution
        if include_auto:
            if np.abs(a - 2) < 1e-6:  # a = 2 (recommended case)
                # μ_auto(t) = A/τ * e^(1 - t/τ) * (1 - t/τ)
                mu_auto = (A_cond / tau) * np.exp(1 - t / tau) * (1 - t / tau)
            else:  # General case for a ≠ 2
                t_adj = t + tol  # Avoid singularity at t=0
                # μ_auto(t) = A * e^(-t/τ) * ((t*e)/(a-1)τ)^(a-1) * ((a-1)/t - 1/τ)
                base = (t_adj * np.e) / ((a - 1) * tau)
                exp_part = np.exp(-t_adj / tau)
                poly_part = base ** (a - 1)
                deriv_part = (a - 1) / t_adj - 1 / tau
                mu_auto = A_cond * exp_part * poly_part * deriv_part
        else:
            mu_auto = 0
        
        return mu_controlled + mu_auto
    
    def drift_rate_integral(
        self,
        t: np.ndarray,
        condition: str = 'comp'
    ) -> np.ndarray:
        """
        Integral of drift rate ∫μ(τ)dτ
        
        Parameters
        ----------
        t : np.ndarray
            Time values
        condition : str
            'comp' or 'incomp'
            
        Returns
        -------
        np.ndarray
            Integrated drift rate
        """
        muc = self.params.muc
        tau = self.params.tau
        a = self.params.a
        A = self.params.A
        
        A_cond = A if condition == 'comp' else -A
        
        # ∫muc dt = muc * t
        controlled_int = muc * t
        
        # ∫A/τ * e^(1 - t/τ) * (1 - t/τ) dt = A * e^(-t/τ) * ((t*e)/(a-1)τ)^(a-1)
        base = (t * np.e) / ((a - 1) * tau)
        auto_int = A_cond * np.exp(-t / tau) * (base ** (a - 1))
        
        return controlled_int + auto_int
    
    def starting_point_distribution(self, x: np.ndarray) -> np.ndarray:
        """
        Beta-shaped starting point distribution (symmetric around x=0)
        
        P(x) ∝ Beta(x; α, α)
        
        Parameters
        ----------
        x : np.ndarray
            Evidence space values [-1, 1]
            
        Returns
        -------
        np.ndarray
            Probability density
        """
        if not self.var_start:
            # Dirac delta at x=0
            result = np.zeros_like(x)
            result[np.argmin(np.abs(x))] = 1.0 / self.dx
            return result
        
        alpha = self.params.alpha
        
        # Map x from [-1, 1] to [0, 1]
        xx = (x + 1) / 2
        
        # Beta distribution
        pdf = beta.pdf(xx, alpha, alpha) / 2  # Factor 1/2 for Jacobian
        
        # Normalize
        pdf = pdf / (np.sum(pdf) * self.dx)
        
        return pdf
    
    def nondecision_time_distribution(self, t: np.ndarray) -> np.ndarray:
        """
        Truncated normal distribution for non-decision time
        
        N(t; μ, σ²) truncated to [0, t_max]
        
        Parameters
        ----------
        t : np.ndarray
            Time values
            
        Returns
        -------
        np.ndarray
            Probability density
        """
        if not self.var_non_dec:
            # Dirac delta at non_dec
            result = np.zeros_like(t)
            idx = np.argmin(np.abs(t - self.params.non_dec))
            result[idx] = 1.0 / self.dt
            return result
        
        non_dec = self.params.non_dec
        sd_non_dec = self.params.sd_non_dec
        
        # Truncated normal
        pdf = norm.pdf(t, non_dec, sd_non_dec)
        
        # Normalize to integrate to 1
        pdf = pdf / (np.sum(pdf) * self.dt)
        
        return pdf
    
    def simulate_trial(
        self,
        condition: str = 'comp',
        seed: Optional[int] = None
    ) -> Dict:
        """
        Simulate a single trial from the DMC model
        
        Uses Euler discretization to solve the stochastic differential equation:
        dx(t) = μ(t, cond) dt + σ dW(t)
        
        Parameters
        ----------
        condition : str
            'comp' (compatible) or 'incomp' (incompatible)
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        dict
            Contains: rt (response time), choice (0 or 1), trajectory
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Draw starting point
        p_x = self.starting_point_distribution(self.x_vec)
        p_x = p_x / np.sum(p_x)
        x_samples = np.random.choice(self.x_vec, p=p_x)
        
        # Draw non-decision time
        p_t = self.nondecision_time_distribution(self.t_vec)
        p_t = p_t / np.sum(p_t)
        ndt_samples = np.random.choice(self.t_vec, p=p_t)
        
        # Initialize accumulation
        x = x_samples
        b = self.params.b
        sigma = self.params.sigma
        
        # Simulate evidence accumulation
        trajectory = [x]
        times = [0]
        
        t = 0
        while t < self.t_max:
            # Drift rate at current time
            mu_t = self.drift_rate(np.array([t]), condition=condition)[0]
            
            # Euler step
            dW = np.random.normal(0, np.sqrt(self.dt))
            x = x + mu_t * self.dt + sigma * dW
            
            trajectory.append(x)
            t += self.dt
            times.append(t)
            
            # Check boundary conditions
            if x >= b:
                return {
                    'rt': t + ndt_samples,
                    'choice': 1,  # Upper boundary
                    'trajectory': np.array(trajectory),
                    'times': np.array(times),
                    'correct': condition == 'comp'
                }
            elif x <= -b:
                return {
                    'rt': t + ndt_samples,
                    'choice': 0,  # Lower boundary
                    'trajectory': np.array(trajectory),
                    'times': np.array(times),
                    'correct': condition == 'incomp'
                }
        
        # Timeout (didn't hit boundary)
        return {
            'rt': self.t_max + ndt_samples,
            'choice': np.sign(x) if x != 0 else 1,
            'trajectory': np.array(trajectory),
            'times': np.array(times),
            'correct': None
        }
    
    def simulate_data(
        self,
        n_trials: int = 500,
        conditions: Optional[List[str]] = None,
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate multiple trials from DMC model
        
        Parameters
        ----------
        n_trials : int
            Number of trials per condition
        conditions : list, optional
            Conditions to simulate. Defaults to ['comp', 'incomp']
        seed : int, optional
            Random seed
            
        Returns
        -------
        dict
            Simulation results with RT, choice, error, condition
        """
        if conditions is None:
            conditions = ['comp', 'incomp']
        
        if seed is not None:
            np.random.seed(seed)
        
        results = {
            'rt': [],
            'choice': [],
            'correct': [],
            'condition': [],
            'error': []
        }
        
        for condition in conditions:
            for _ in range(n_trials):
                trial = self.simulate_trial(condition=condition)
                results['rt'].append(trial['rt'])
                results['choice'].append(trial['choice'])
                results['correct'].append(trial['correct'])
                results['condition'].append(condition)
                results['error'].append(0 if trial['correct'] else 1)
        
        # Convert to numpy arrays
        for key in results:
            results[key] = np.array(results[key])
        
        return results
    
    def calculate_quantiles(
        self,
        condition: str = 'comp',
        quantiles: List[float] = None,
        n_trials: int = 10000
    ) -> Dict[str, np.ndarray]:
        """
        Calculate RT quantiles for correct and error responses
        
        Parameters
        ----------
        condition : str
            'comp' or 'incomp'
        quantiles : list
            Quantile values (default: [0.1, 0.3, 0.5, 0.7, 0.9])
        n_trials : int
            Number of trials to simulate
            
        Returns
        -------
        dict
            Quantiles for correct and error responses
        """
        if quantiles is None:
            quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        np.random.seed(42)
        
        correct_rts = []
        error_rts = []
        
        for _ in range(n_trials):
            trial = self.simulate_trial(condition=condition)
            if trial['correct']:
                correct_rts.append(trial['rt'])
            else:
                error_rts.append(trial['rt'])
        
        correct_rts = np.array(correct_rts)
        error_rts = np.array(error_rts)
        
        return {
            'correct': np.quantile(correct_rts, quantiles),
            'error': np.quantile(error_rts, quantiles),
            'quantiles': quantiles
        }
    
    def plot_drift_rate(self, condition: str = 'comp', ax=None):
        """Plot drift rate function over time"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        mu = self.drift_rate(self.t_vec, condition=condition)
        
        ax.plot(self.t_vec, mu, 'b-', linewidth=2, label=f'Drift rate ({condition})')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Drift rate', fontsize=12)
        ax.set_title(f'DMC Drift Rate Function - {condition} condition', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_trajectories(self, condition: str = 'comp', n_trials: int = 50, ax=None):
        """Plot simulated trajectories"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        for i in range(n_trials):
            trial = self.simulate_trial(condition=condition)
            ax.plot(trial['times'], trial['trajectory'], alpha=0.3, linewidth=0.8)
        
        b = self.params.b
        ax.axhline(y=b, color='g', linestyle='--', linewidth=2, label='Upper boundary (+b)')
        ax.axhline(y=-b, color='r', linestyle='--', linewidth=2, label='Lower boundary (-b)')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Evidence', fontsize=12)
        ax.set_title(f'Simulated Trajectories - {condition} condition', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_rt_distribution(self, condition: str = 'comp', n_trials: int = 2000, ax=None):
        """Plot RT distribution for correct and error responses"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        np.random.seed(42)
        data = self.simulate_data(n_trials=n_trials // 2, conditions=[condition])
        
        correct_rts = data['rt'][data['correct'] == True]
        error_rts = data['rt'][data['correct'] == False]
        
        ax.hist(correct_rts, bins=30, alpha=0.6, label=f'Correct (n={len(correct_rts)})', color='blue')
        ax.hist(error_rts, bins=30, alpha=0.6, label=f'Error (n={len(error_rts)})', color='red')
        
        ax.set_xlabel('Response Time (s)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'RT Distribution - {condition} condition', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        return ax


# Summary statistics
def calculate_summary_stats(data: Dict[str, np.ndarray]) -> Dict:
    """
    Calculate summary statistics from simulated data
    
    Parameters
    ----------
    data : dict
        Simulation results from simulate_data()
        
    Returns
    -------
    dict
        Summary statistics including RT, accuracy, etc.
    """
    conditions = np.unique(data['condition'])
    stats = {}
    
    for cond in conditions:
        mask = data['condition'] == cond
        cond_rts = data['rt'][mask]
        cond_correct = data['correct'][mask]
        
        stats[cond] = {
            'mean_rt': np.mean(cond_rts),
            'sd_rt': np.std(cond_rts),
            'accuracy': np.mean(cond_correct),
            'mean_rt_correct': np.mean(cond_rts[cond_correct]),
            'mean_rt_error': np.mean(cond_rts[~cond_correct]),
            'n_correct': np.sum(cond_correct),
            'n_error': np.sum(~cond_correct)
        }
    
    # Conflict effect (incomp - comp)
    if 'comp' in stats and 'incomp' in stats:
        stats['conflict_effect'] = stats['incomp']['mean_rt'] - stats['comp']['mean_rt']
    
    return stats
