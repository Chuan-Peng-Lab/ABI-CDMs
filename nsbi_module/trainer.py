from default_settings import DEFAULT_NET_CONFIG

import numpy as np
import pandas as pd

import os
from pathlib import Path
import warnings

os.environ["KERAS_BACKEND"] = "torch"
import bayesflow as bf
from abc import ABC, abstractmethod
import keras

from typing import Callable, Dict, List, Optional


def random_num_trials(
    rng: Optional[np.random.Generator],
    min_obs: Optional[int] = None,
    max_obs: Optional[int] = None,
) -> int:
    """
    Generate a random number of trials for simulation.
    
    This function is called internally during each backpropagation step to 
    determine how many observations/trials to generate for a simulation.
    
    Parameters:
    -----------
    rng : np.random.Generator or None
        Random number generator instance
    min_obs : int, optional
        Minimum number of observations
    max_obs : int, optional
        Maximum number of observations
        
    Returns:
    --------
    int
        Randomly selected number of trials between min_obs and max_obs (inclusive)
    """
    if rng is None:
        rng = np.random.default_rng()

    if min_obs is None or max_obs is None:
        raise ValueError("Both min_obs and max_obs must be provided and not None.")

    return rng.integers(min_obs, max_obs + 1)


def make_uniform_prior_func(param_keys, prior_bounds):
    """
    Create a uniform prior sampling function.
    
    Creates a function that samples parameters uniformly from specified bounds.
    
    Parameters:
    -----------
    param_keys : list
        Names of parameters to sample
    prior_bounds : dict
        Dictionary mapping parameter names to (min, max) bounds
        
    Returns:
    --------
    function
        A callable that generates parameter samples
    """
    def prior_func(rng=None):
        if rng is None:
            rng_instance = np.random
        else:
            rng_instance = rng

        bounds_array = np.array(list(prior_bounds.values()))
        lows = bounds_array[:, 0]
        highs = bounds_array[:, 1]
        samples = rng_instance.uniform(lows, highs)
        return dict(zip(param_keys, samples))
    
    return prior_func


class CDMsSimulator(ABC):
    """
    Base class for Cognitive Diffusion Model Simulators.
    
    Provides a framework for creating simulators that can generate synthetic
    data from cognitive models with specified priors.
    """

    def __init__(
        self,
        model_name,
        prior,
        param_keys,
        param_names,
        experiment_simulator,
        varied_num_trials=None,
        rng=None,
        df_col = ["rt", "choice", "condition"],
        subject_col="subject_id",
        **kwargs,
    ):
        """
        Initialize the simulator with model configuration.
        
        Parameters:
        -----------
        model_name : str
            Name identifier for the model
        prior : callable or dict
            Function to sample parameters or dictionary of parameter bounds
        param_keys : list
            List of parameter names
        param_names : list
            Human-readable parameter names
        experiment_simulator : callable
            Function that runs the actual simulation
        varied_num_trials : list or None
            [min_trials, max_trials] for variable trial count simulations
        rng : np.random.Generator or None
            Random number generator
        subject_col : str
            Column name for subject identification in output data
        """
        
        self._rng = rng or np.random.default_rng()
        self.model_name = model_name
        self.experiment_simulator = experiment_simulator
        self.varied_num_trials = varied_num_trials
        self.param_keys = param_keys
        self.param_names = param_names
        self.subject_col = subject_col
        self.df_col = df_col

        # Configure prior distribution
        if isinstance(prior, Callable):
            self.prior = prior
        elif isinstance(prior, dict):
            self.prior = make_uniform_prior_func(self.param_keys, prior)
        else:
            raise TypeError("Prior must be callable or dictionary")

        # Setup simulator with appropriate trial handling
        if varied_num_trials is None:
            self.simulator: bf.simulators.Simulator = bf.make_simulator(
                [self.prior, experiment_simulator]
            )
        elif isinstance(varied_num_trials, List):
            def _random_num_trials():
                num_trials = random_num_trials(
                    self._rng,
                    min_obs=varied_num_trials[0],
                    max_obs=varied_num_trials[1],
                )
                return dict(num_trials=num_trials)

            self.simulator: bf.simulators.Simulator = bf.make_simulator(
                [self.prior, experiment_simulator], meta_fn=_random_num_trials
            )

    def sample_prior(self, n_sim: int = 1):
        """
        Sample parameters from the prior distribution.
        
        Parameters:
        -----------
        n_sim : int
            Number of parameter sets to sample
            
        Returns:
        --------
        list
            List of parameter dictionaries
        """
        return [self.prior() for _ in range(n_sim)]

    def simulate_data(
        self,
        n_sim: int = 1,
        n_trial: Optional[int] = None,
        params: Optional[List] = None,
    ) -> pd.DataFrame:
        """
        Simulate experimental data from the cognitive model.
        
        Parameters:
        -----------
        n_sim : int
            Number of simulations to run (if params not provided)
        n_trial : int or None
            Number of trials per simulation
        params : list or None
            Pre-specified parameters for simulations
            
        Returns:
        --------
        pandas.DataFrame
            Combined simulation results with subject identifiers
        """
        # Use provided parameters or sample from prior
        if params is None:
            params_list = self.sample_prior(n_sim)
        else:
            assert set(params[0].keys()) == set(self.param_keys), f"Parameters must match model parameters {self.param_keys}."
            params_list = params

        # Generate data for each parameter set
        dfs = []
        for subject_id, param_dict in enumerate(params_list):
            # Run simulation with specified trial count
            if n_trial is not None:
                sim_data = self.experiment_simulator(num_trials=n_trial, **param_dict)
            else:
                sim_data = self.experiment_simulator(**param_dict)

            # Format output with subject identifier
            sim_df = pd.DataFrame(sim_data)
            sim_df[self.subject_col] = subject_id
            for param_i, value_i in param_dict.items():
                sim_df[param_i] = value_i
            dfs.append(sim_df)

        output_df = pd.concat(dfs, ignore_index=True)
        # check output_df columns is consistent with df_col
        if set(output_df.columns) == set(self.df_col):
            warnings.warn(
                f"Output data columns do not match {self.df_col}. Please check your simulation function."
            )
        # else:
        #     print("Output data columns:", output_df.columns)

        return output_df

    @property
    def params_mapping_from_key_to_name(self):
        """
        Create a mapping from parameter keys to their human-readable names.
        
        Returns:
        --------
        dict
            Dictionary mapping parameter keys to parameter names
        """
        return dict(zip(self.param_keys, self.param_names))

    def get_param_name_from_param_key(self, param_key):
        """
        Get the human-readable name for a given parameter key.
        
        Parameters:
        -----------
        param_key : str
            The parameter key to look up
            
        Returns:
        --------
        str
            The human-readable name for the parameter key
            
        Raises:
        -------
        KeyError
            If the param_key is not found in param_keys
        """
        if param_key not in self.param_keys:
            raise KeyError(f"Parameter key '{param_key}' not found. Available keys: {self.param_keys}")
        
        key_to_name_mapping = self.params_mapping_from_key_to_name
        return key_to_name_mapping[param_key]
        

class Experiment(ABC):
    """
    Abstract base class for standardized experiments.
    
    Defines the interface that all experiment types must implement.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass


class CDMsTrainer(Experiment):
    """
    Trainer for Cognitive Diffusion Models using BayesFlow.
    
    Handles the full training pipeline including network setup, 
    simulation, training, and model persistence.
    """

    def __init__(
        self,
        cdms_simulator: CDMsSimulator,
        adapter: Optional[bf.Adapter] = None,
        checkpoint_path: Optional[str] = None,
        net_config: Optional[Dict] = None,
    ):
        """
        Initialize the trainer with model components.
        
        Parameters:
        -----------
        cdms_simulator : CDMsSimulator
            Simulator instance for generating training data
        adapter : bf.Adapter or None
            Data adapter for preprocessing inputs
        checkpoint_path : str or None
            Directory to save/load model checkpoints
        net_config : dict or None
            Network configuration overrides
        """
        
        self.cdms_simulator = cdms_simulator
        self.model_name = self.cdms_simulator.model_name
        self.param_keys = self.cdms_simulator.param_keys
        self.param_names = self.cdms_simulator.param_names

        # Setup checkpoint path
        if checkpoint_path is None:
            self.checkpoint_path = Path("checkpoints") / self.model_name
            print(f"Checkpoint path not provided, using {self.checkpoint_path}")
        else:
            self.checkpoint_path = Path(checkpoint_path)

        # Configure networks with defaults and overrides
        default_net_config: Dict = DEFAULT_NET_CONFIG
        summary_network_config: Dict = {}
        inference_network_config: Dict = {}
        workflow_config: Dict = {}
        if "summary_network_settings" in default_net_config:
            summary_network_config: Dict = default_net_config["summary_network_settings"]
        if "inference_network_settings" in default_net_config:
            inference_network_config: Dict = default_net_config["inference_network_settings"]
        if "workflow_settings" in default_net_config:
            workflow_config: Dict = default_net_config["workflow_settings"]
        if net_config is not None:
            if "summary_network_settings" in net_config:
                summary_network_config.update(net_config["summary_network_settings"])
            if "inference_network_settings" in net_config:
                inference_network_config.update(net_config["inference_network_settings"])
            if "workflow_settings" in net_config:
                workflow_config.update(net_config["workflow_settings"])

        # Initialize network components
        summary_network = bf.networks.SetTransformer(**summary_network_config)
        inference_network = bf.networks.FlowMatching(coupling_kwargs=dict(subnet_kwargs=dict(dropout=inference_network_config["dropout"])))
        self.summary_network = summary_network
        self.inference_network = inference_network

        # Setup adapter and workflow
        self.adapter = adapter or self._adapter()
        
        self.workflow:bf.BasicWorkflow = bf.BasicWorkflow(
            simulator=self.cdms_simulator.simulator,
            adapter=self.adapter,
            inference_network=self.inference_network,
            summary_network=self.summary_network,
            inference_conditions=None,
            checkpoint_name=self.model_name,
            checkpoint_path=self.checkpoint_path,
            **workflow_config
        )

        save_name = self.checkpoint_path / f"{self.model_name}.keras"
        if save_name.exists():
            print(f"find trained weights and load it: {save_name}")
            self.load(save_name)

    def get_param_name(self, param_key):
        """
        Get the human-readable name for a given parameter key.
        
        Parameters:
        -----------
        param_key : str
            The parameter key to look up
            
        Returns:
        --------
        str
            The human-readable name for the parameter key
            
        Raises:
        -------
        KeyError
            If the param_key is not found in param_keys
        """
        
        return self.cdms_simulator.get_param_name_from_param_key(param_key)

    def simulate(self, *args, **kwargs):
        """Generate simulation samples using the underlying simulator."""
        return self.cdms_simulator.simulator.sample(*args, **kwargs)

    def simulate_data(self, *args, **kwargs):
        """Generate formatted simulation data."""
        return self.cdms_simulator.simulate_data(*args, **kwargs)

    def _adapter(self):
        """
        Create default data adapter for preprocessing simulation outputs.
        
        Returns:
        --------
        bf.Adapter
            Configured adapter for transforming simulation data
        """
        return (
            bf.Adapter()
            .convert_dtype("float64", "float32")
            # .sqrt("num_trials")
            .broadcast("num_trials", to="rt")
            .as_set(self.cdms_simulator.df_col)
            .concatenate(self.cdms_simulator.param_keys, into="inference_variables")
            .concatenate(self.cdms_simulator.df_col, into="summary_variables")
            .nan_to_num("summary_variables", 0) # Note: to fix the issue of NaNs in the summary variables, we set them to 0
            .rename("num_trials", "inference_conditions")
        )

    def _prior_mean(self, n_draws=10000):
        """
        Calculate prior means and standard deviations.
        
        Parameters:
        -----------
        n_draws : int
            Number of samples to draw from prior for statistics calculation
        """
        prior_list = self.cdms_simulator.sample_prior(n_sim=n_draws)
        df = pd.DataFrame(prior_list)
        self.prior_means = df.mean(axis=1)
        self.prior_stds = df.std(axis=1)

    def run(
        self,
        epochs=100,
        num_batches_per_epoch=250,
        batch_size=64,
        train_mode="online",
        save=True,
        **kwargs,
    ):
        """
        Train the model using specified training mode.
        
        Parameters:
        -----------
        epochs : int
            Number of training epochs
        num_batches_per_epoch : int
            Number of batches per epoch
        batch_size : int
            Size of each training batch
        train_mode : str
            Either "online" or "offline"
            
        Returns:
        --------
        History object from training process
        """
        
        validation_data = kwargs.pop("validation_data", batch_size*2)

        history_all = {"loss": [], "val_loss": []}
        existed_history_obj = getattr(self.workflow, "history", None)
        existed_history_dict = {}
        existed_epochs_list = []  # Track epochs from previous runs
        total_epochs = 0

        if existed_history_obj is not None:
            # Case 1: workflow.history is a Keras History object
            if hasattr(existed_history_obj, 'history') and isinstance(getattr(existed_history_obj, 'history', None), dict):
                existed_history_dict = existed_history_obj.history
                # Safely get epoch list, defaulting to range based on history length if missing
                existed_epochs_list = getattr(existed_history_obj, 'epoch', list(range(len(next(iter(existed_history_dict.values()))))) if existed_history_dict else [])
            # Case 2: workflow.history is already a dict
            elif isinstance(existed_history_obj, dict):
                existed_history_dict = existed_history_obj
                # If it's a dict, try to get epoch list, defaulting based on loss length
                existed_epochs_list = existed_history_obj.get('epoch', list(range(len(existed_history_dict.get('loss', [])))))
            else:
                print(f"Warning: Found workflow.history of unexpected type: {type(existed_history_obj)}. Cannot merge history.")
                existed_history_dict = {}
                existed_epochs_list = []

        if existed_history_dict and "val_loss" in existed_history_dict:
            prev_epochs = len(existed_history_dict["val_loss"])
            best_val = np.min(existed_history_dict["val_loss"])
            total_epochs = prev_epochs
            history_all["loss"].extend(existed_history_dict.get("loss", []))
            history_all["val_loss"].extend(existed_history_dict["val_loss"])
            print(f"Found existing training history ({prev_epochs} epochs, best val_loss = {best_val:.4f})")

        # Perform training based on mode
        if train_mode == "online":
            current_history = self.workflow.fit_online(
                epochs=epochs,
                batch_size=batch_size,
                num_batches_per_epoch=num_batches_per_epoch,
                validation_data=validation_data,
                **kwargs,
            )
        elif train_mode == "offline":
            data = kwargs.pop("data", None)
            if data is None:
                raise ValueError("No data provided for offline training")
            current_history = self.workflow.fit_offline(
                data=data,
                epochs=epochs,
                batch_size=batch_size,
                num_batches_per_epoch=num_batches_per_epoch,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

        # Extract current losses and epochs
        if hasattr(current_history, 'history') and isinstance(getattr(current_history, 'history', None), dict):
            current_history_dict = current_history.history
        
            current_epochs_list = list(range(total_epochs, total_epochs + len(current_history_dict.get('loss', []))))
            current_loss = current_history_dict.get("loss", [])
            current_val = current_history_dict.get("val_loss", [])

            self.workflow.history.history["total_epochs"] = len(current_loss)

        if existed_history_obj is not None:
            # Merge histories
            history_all["loss"].extend(current_loss)
            if current_val:  # Extend val_loss if current run produced it
                history_all["val_loss"].extend(current_val)
            else:
                # Extend val_loss with NaN if current run didn't produce it
                print("Warning: Current training run did not produce 'val_loss'. Extending with NaN.")
                history_all["val_loss"].extend([float('nan')] * len(current_loss))

            total_epochs += len(current_loss)

            # Merge epoch lists
            merged_epochs_list = existed_epochs_list + current_epochs_list

            # Update epoch list in the original history object if it has one
            if hasattr(existed_history_obj, 'epoch'):
                existed_history_obj.epoch = merged_epochs_list
            # Also update epoch list if it was a dict
            elif isinstance(existed_history_obj, dict):
                existed_history_obj['epoch'] = merged_epochs_list

            # Add total epochs and epoch list to the merged history
            history_all["total_epochs"] = total_epochs
            history_all['epoch'] = merged_epochs_list

            # Update the workflow's history attribute
            if hasattr(existed_history_obj, 'history'):  # Keras History object
                existed_history_obj.history = history_all
                self.workflow.history = existed_history_obj
            elif isinstance(existed_history_obj, dict):  # Dictionary
                existed_history_obj.update(history_all)  # Merge the dictionaries
                self.workflow.history = existed_history_obj

        self.history = self.workflow.history.history

        if save:
            self.save()

        return self.workflow.history

    def save(self, save_name:Optional[Path]=None):
        """
        Save the trained model to disk.
        
        Parameters:
        -----------
        save_name : str or Path, optional
            Path to save model; defaults to checkpoint path
        """
        import pickle
        if save_name is None:
            save_name_approximator = self.checkpoint_path / f"{self.model_name}.keras"
            save_name_history = self.checkpoint_path / f"{self.model_name}.pkl"
        else: 
            save_name_approximator = save_name.with_suffix(".keras")
            save_name_history = save_name.with_suffix(".pkl")
        
        # Ensure directory exists
        save_name_approximator.parent.mkdir(parents=True, exist_ok=True)

        with open(save_name_history, 'wb') as f:
            pickle.dump(self.workflow.history, f)

        self.workflow.approximator.save(save_name_approximator)

    def load(self, save_name:Optional[Path]=None):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        save_name : str or Path, optional
            Path to load model from; defaults to checkpoint path
        """
        import pickle
        if save_name is None:
            save_name_approximator = self.checkpoint_path / f"{self.model_name}.keras"
            save_name_history = self.checkpoint_path / f"{self.model_name}.pkl"
        else: 
            save_name_approximator = save_name.with_suffix(".keras")
            save_name_history = save_name.with_suffix(".pkl")

        with open(save_name_history, 'rb') as f:
            self.workflow.history = pickle.load(f)
            self.history = self.workflow.history.history
        
        self.workflow.approximator = keras.saving.load_model(save_name_approximator)
        # NOTE: we need to force setting the adapter
        # init adapter
        self.adapter(self.cdms_simulator.simulator.sample(1))
        # set adapter
        self.workflow.approximator.adapter = self.adapter

    def plot_trained_result(self, n_batch=100, n_posterior=500, **kwargs):
        """
        Generate diagnostic plots for trained model.
        
        Parameters:
        -----------
        n_batch : int
            Number of test batches to generate
        n_posterior : int
            Number of posterior samples for diagnostics
            
        Returns:
        --------
        matplotlib.Figure
            Diagnostic plots figure
        """
        sims_data = self.cdms_simulator.simulator.sample(n_batch)
        
        fig = self.workflow.plot_default_diagnostics(
            test_data=sims_data, 
            num_samples=n_posterior,
            variable_keys = self.param_keys,
            variable_names = self.param_names,
            **kwargs
        )
        
        return fig