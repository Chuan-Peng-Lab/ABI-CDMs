from .simulators import TRIAL_SIMULATOR, CDMs_experiment_simulator_wrapper
from .default_settings import MODEL_CONFIG, CONTEXT_CONFIG

from .trainer import CDMsSimulator, CDMsTrainer
from .utils_pydmc import PlotFit
import bayesflow as bf

import numpy as np
import pandas as pd
import xarray as xr
import arviz as az
import re
from typing import Optional, Union, Callable, List, Dict
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_generator(
    model_name: str,
    trial_simulator: Optional[Callable] = None,
    experiment_simulator: Optional[Callable] = None,
    simulator_type: Optional[str] = None,
    wrapper_kwargs: Optional[Dict] = None,
    **kwargs
) -> CDMsSimulator:
    """
    Load a BayesFlow-compatible CDM generator.

    This function supports three use cases:

    1. A full experiment-level simulator is provided directly.
       In this case, the function will use it as-is.

    2. A trial-level simulator is provided directly.
       In this case, the function will wrap it with CDMs_experiment_simulator_wrapper.

    3. No simulator is provided.
       In this case, the function will search TRIAL_SIMULATOR by model_name.
       The model config can specify whether the registered simulator is
       trial-level or experiment-level via the key "simulator_type".

    Expected experiment-level simulator signature:
        experiment_simulator(num_trials=..., **params)

    Expected experiment-level simulator output:
        {
            "rt": np.ndarray,
            "choice": np.ndarray,
            "condition": np.ndarray
        }
    """

    if model_name not in MODEL_CONFIG:
        raise ValueError(f"{model_name} is not defined in MODEL_CONFIG.")

    if trial_simulator is not None and experiment_simulator is not None:
        raise ValueError(
            "Provide either trial_simulator or experiment_simulator, not both."
        )

    model_config = MODEL_CONFIG[model_name]
    prior_dict = model_config["prior_range"]

    wrapper_kwargs = wrapper_kwargs or {}

    # Case 1: use a user-provided experiment-level simulator directly.
    if experiment_simulator is not None:
        selected_experiment_simulator = experiment_simulator

    # Case 2: wrap a user-provided trial-level simulator.
    elif trial_simulator is not None:
        selected_experiment_simulator = CDMs_experiment_simulator_wrapper(
            trial_simulator,
            **wrapper_kwargs
        )

    # Case 3: no simulator is explicitly provided; search the global registry.
    else:
        if model_name not in TRIAL_SIMULATOR:
            raise ValueError(
                f"{model_name} is not found in TRIAL_SIMULATOR, "
                "and no simulator was explicitly provided."
            )

        registered_simulator = TRIAL_SIMULATOR[model_name]

        # Default to trial-level for backward compatibility.
        resolved_simulator_type = (
            simulator_type
            or model_config.get("simulator_type", "trial")
        )

        if resolved_simulator_type == "experiment":
            selected_experiment_simulator = registered_simulator

        elif resolved_simulator_type == "trial":
            selected_experiment_simulator = CDMs_experiment_simulator_wrapper(
                registered_simulator,
                **wrapper_kwargs
            )

        else:
            raise ValueError(
                "simulator_type must be either 'trial' or 'experiment'. "
                f"Got: {resolved_simulator_type}"
            )

    CDMs_generator = CDMsSimulator(
        model_name,
        prior_dict,
        model_config["param_keys"],
        model_config["param_names"],
        selected_experiment_simulator,
        CONTEXT_CONFIG["n_trials_range"],
        **kwargs
    )

    return CDMs_generator

def fmt_mem(bytes_val):
    gb = bytes_val / (1024**3)
    mb = bytes_val / (1024**2)
    return f"{gb:.2f} GB ({mb:.1f} MB)"

class NSBICDM(CDMsTrainer):
    """
    Customer NSBI for CDMs (conflict diffusion models).

    Members
    ---
    model_name(str):
        DDM, DMC, SSP, DSTP
    param_names(list):
        The names of parameters in the model.
    obs_df(pd.DataFrame): The DataFrame is expected to have the following columns:
        - 'rt': Reaction time values, expected to be in seconds and less than 998.
        - 'accuracy': Response values, which should only be 0 or 1 (usually 0 for error and 1 for correct response).
        - 'congruency': Congruency values, which should only be 0 or 1.
        - 'subject_id'(optional): Subject index values, which should be unique for each subject.
    df_col(list):
        The columns of the dataframe must contain the columns in this list.
        Note the order of data for input in network
    check_point_path(str):
        The location of trainned check points.

    """

    def __init__(
        self,
        model: str = "DMC",
        obs_df: pd.DataFrame = None,
        df_col=["rt", "accuracy", "congruency"],
        subject_col="subject_id",
        **kwargs,
    ):
        generator = load_generator(model)

        self.df_col = df_col
        self.subject_col = subject_col
        if obs_df is not None:
            self._check_df(obs_df)
            self.obs_df = obs_df

        super().__init__(cdms_simulator=generator, **kwargs)

    def _check_df(self, df: pd.DataFrame):
        """
        Validates the input DataFrame to ensure it meets specific criteria.

        Parameters:
        df (pd.DataFrame): The DataFrame to be validated.

        Raises:
        AssertionError: If the DataFrame does not contain the required columns.
        AssertionError: If the 'congruency' column contains values other than 0 or 1.
        AssertionError: If the 'accuracy' column contains values other than 0 or 1.
        AssertionError: If any value in the 'rt' column is 998 or greater.

        The DataFrame is expected to have the following columns:
        - 'rt': Reaction time values, expected to be in seconds and less than 998.
        - 'accuracy': Response values, which should only be 0 or 1 (usually 0 for error and 1 for correct response).
        - 'congruency': Congruency values, which should only be 0 or 1.

        Example:
        >>> df = pd.DataFrame({
        ...     'rt': [0.5, 0.7, 0.9],
        ...     'accuracy': [1, 0, 1],
        ...     'congruency': [0, 1, 0]
        ... })
        >>> _check_df(df)
        """

        assert all([i in df.columns for i in self.df_col]), (
            f"dataframe should contain the columns {self.df_col}"
        )
        assert set(df[self.df_col[1]].unique()).issubset({0, 1}), (
            f"The values in the {self.df_col[1]} column must only be 0 or 1"
        )
        assert set(df[self.df_col[2]].unique()).issubset({0, 1}), (
            f"The values in the {self.df_col[2]} column must only be 0 or 1"
        )
        assert all(df["rt"] < 998), (
            "The values in the rt column must be in unit of second and less than 998"
        )
        grouped = self.subject_col in df.columns
        # if not grouped:
        #     warnings.warn(
        #         f"subject_col:{self.subject_col} is not in dataframe. ", UserWarning
        #     )

        return grouped

    def simulate_data(self, *args, **kwargs):
        sim_data = super().simulate_data(*args, **kwargs)
        sim_data.rename(
            columns={"choice": self.df_col[1], "condition": self.df_col[2]},
            inplace=True,
        )
        return sim_data

    def sample(
        self,
        forward_dict: Dict,
        n_posterior: int = 5000,
        n_chain: int = 1,
        batchsize: Optional[int] = None,
        show_progress: bool = False
    ) -> pd.DataFrame:
        """
        Generate samples from the model's posterior distribution based on observed data.

        Supports batching and optional progress tracking for long-running inference.

        Parameters:
        -----------
        forward_dict : Dict
            Must contain 'rt', 'choice', 'condition' as (n_subjects, n_trials) arrays.
            Optional 'num_trials' (int) — auto-inferred if missing.

        n_posterior : int, default=5000
            Number of posterior draws per chain.

        n_chain : int, default=1
            Number of independent MCMC chains.

        batchsize : int or None, default=None
            Max number of subjects per batch. If None, process all subjects at once.

        show_progress : bool, default=False
            Whether to display progress bars. Only active when:
            - tqdm is installed,
            - batchsize > 5, and
            - either n_chain > 1 or number of batches > 1.

        Returns:
        --------
        pandas.DataFrame — posterior samples in long format with columns:
            ['subject_id', 'chain', 'draw', 'a', 'ndt', 'v_c', 'v_i', ...]

        Examples:
        ---------
        1. Basic usage (single chain, all subjects at once):
        
            >>> data = {
            ...     "rt": np.random.uniform(0.3, 2.0, size=(20, 100)),   # 20 subjects, 100 trials
            ...     "choice": np.random.randint(0, 2, size=(20, 100)),
            ...     "condition": np.random.choice([0, 1], size=(20, 100))
            ... }
            >>> df = model.sample(data, n_posterior=1000, n_chain=1)

        2. Memory-efficient sampling for large cohorts (e.g., 500 subjects):
        
            # Split into batches of 50 subjects (10 batches total)
            >>> df = model.sample(
            ...     data, 
            ...     n_posterior=2000,
            ...     n_chain=4,
            ...     batchsize=50,       # ← critical for avoiding OOM
            ...     show_progress=True  # ← will show progress bars (batchsize=50 > 5)
            ... )

        """
        from tqdm import tqdm 
        import gc
        import torch
        
        required_keys = {"rt", "choice", "condition"}
        assert required_keys.issubset(forward_dict.keys()), \
            f"forward_dict must contain keys: {required_keys}"

        rt = np.asarray(forward_dict["rt"])
        n_subjects, n_trials = rt.shape[0], rt.shape[1]

        if "num_trials" not in forward_dict:
            forward_dict["num_trials"] = n_trials
        else:
            assert forward_dict["num_trials"] == n_trials, \
                f"Mismatch: provided num_trials={forward_dict['num_trials']}, but rt.shape[1]={n_trials}"

        if batchsize is None:
            batchsize = n_subjects
        elif batchsize <= 0:
            raise ValueError("batchsize must be a positive integer or None.")

        n_batches = (n_subjects + batchsize - 1) // batchsize
        effective_batchsize = min(batchsize, n_subjects)

        # Calculate total work units for progress bar
        total_iterations = n_chain * n_batches

        # Decide whether to show progress
        should_show = show_progress and total_iterations > 3
        if should_show:
            pbar = tqdm(total=total_iterations, desc="Sampling", position=0, leave=True)
        else:
            pbar = None

        all_results = []
        try:
            # Single loop over all work units (chains * batches)
            for chain_idx in range(n_chain):
                chain_results = []
                
                # Process batches for this chain
                batch_start_indices = list(range(0, n_subjects, batchsize))
                
                for start in batch_start_indices:
                    if pbar:
                        pbar.set_description(f"Chain {chain_idx+1}/{n_chain}" if n_chain > 1 else "Sampling")
                        
                    end = min(start + batchsize, n_subjects)
                    batch_subject_ids = list(range(start, end))
                    batch_size_local = len(batch_subject_ids)

                    batch_dict = {
                        k: v[batch_subject_ids] if isinstance(v, np.ndarray) and v.ndim >= 2 else v
                        for k, v in forward_dict.items()
                    }

                    posterior_draws_dict = self.workflow.sample(
                        conditions=batch_dict,
                        num_samples=n_posterior,
                    )

                    # Shape validation
                    first_val = next(iter(posterior_draws_dict.values()))
                    if first_val.ndim < 2:
                        raise ValueError(
                            f"Expected posterior samples of shape (n_subjects, n_draws, ...), "
                            f"got {first_val.shape} for parameter '{list(posterior_draws_dict.keys())[0]}'"
                        )
                    assert first_val.shape[0] == batch_size_local
                    assert first_val.shape[1] == n_posterior

                    # Build DataFrame
                    subject_id = np.repeat(batch_subject_ids, n_posterior)
                    draw = np.tile(np.arange(n_posterior), batch_size_local)

                    data_dict = {'subject_id': subject_id, 'draw': draw}
                    for key, value in posterior_draws_dict.items():
                        # Ensure at least 2D: handle scalar outputs that may be squeezed to (D,) when B=1
                        value = np.asarray(value)
                        if value.ndim == 1:
                            # Case: (D,) → assume it's for 1 subject: reshape to (1, D)
                            if value.shape[0] == n_posterior:
                                value = value[None, :]  # (D,) → (1, D)
                            else:
                                raise ValueError(
                                    f"1D array for parameter '{key}' has length {value.shape[0]}, "
                                    f"but expected n_posterior={n_posterior}."
                                )
                        elif value.ndim == 3:
                            if value.shape[-1] == 1:
                                value = value.squeeze(-1)  # (B, D, 1) → (B, D)
                            else:
                                raise ValueError(
                                    f"Parameter '{key}' has shape {value.shape} with >1 trailing dim. "
                                    "Only scalar parameters (no extra dims) are supported."
                                )
                        # Now value should be (B, D)
                        if value.ndim != 2:
                            raise ValueError(
                                f"Parameter '{key}' final shape {value.shape} not supported. "
                                "Expected (B, D) after normalization."
                            )
                        if value.shape[0] != batch_size_local or value.shape[1] != n_posterior:
                            raise ValueError(
                                f"Parameter '{key}' shape {value.shape} mismatches "
                                f"batch_size_local={batch_size_local}, n_posterior={n_posterior}."
                            )
                        data_dict[key] = value.flatten()

                    batch_df = pd.DataFrame(data_dict)
                    batch_df.insert(1, "chain", chain_idx)
                    chain_results.append(batch_df)

                    del first_val
                    del posterior_draws_dict
                    del data_dict
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Update progress bar
                    if pbar:
                        pbar.update(1)

                if chain_results:
                    chain_df = pd.concat(chain_results, axis=0, ignore_index=True)
                    all_results.append(chain_df)
                    
        finally:
            # Ensure progress bar is properly closed
            if pbar:
                pbar.close()

        if not all_results:
            raise RuntimeError("No results generated. Check input data and workflow.")

        final_df = pd.concat(all_results, axis=0, ignore_index=True)

        return final_df

    def df_to_forward_dict(self, df:pd.DataFrame):
        """
        Convert a long-format DataFrame to a dictionary suitable for forward modeling.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame containing trial data in long format. Must contain columns
            specified by self.subject_col and self.df_col (response time, choice, condition).
            
        Returns:
        --------
        dict
            A dictionary containing:
            - "rt": np.array of response times in wide format (subjects x trials)
            - "choice": np.array of choices in wide format (subjects x trials)
            - "condition": np.array of conditions in wide format (subjects x trials)
            - "num_trials": int16, number of trials per subject
        """

        # Sort and group
        df = df.sort_values([self.subject_col]).reset_index(drop=True)
        df['trial_idx'] = df.groupby(self.subject_col).cumcount()
        
        # Pivot to wide format — automatically introduces NaN for missing cells
        rt = df.pivot(index=self.subject_col, columns='trial_idx', values=self.df_col[0]).values
        choice = df.pivot(index=self.subject_col, columns='trial_idx', values=self.df_col[1]).values
        condition = df.pivot(index=self.subject_col, columns='trial_idx', values=self.df_col[2]).values
        
        # Ensure dtype consistency (important for NN input)
        rt = rt.astype(np.float32)
        choice = choice.astype(np.float32)
        condition = condition.astype(np.float32)
        
        forward_dict = {
            "rt":rt,
            "choice":choice,
            "condition": condition,
            "num_trials": np.int16(rt.shape[1])
        }
        return forward_dict

    def fit_data(
        self, obs_df: pd.DataFrame, condition=None, return_infdata=False, **kwargs
    ):
        """
        Fits the model to the provided observational data using neural posterior estimation.

        This function performs Bayesian parameter inference by fitting the model to 
        observed behavioral data. It supports both group-level and hierarchical modeling 
        based on specified conditions. The method can return results either as a pandas 
        DataFrame or as an ArviZ InferenceData object for further analysis.

        Parameters:
        -----------
        obs_df : pd.DataFrame
            The DataFrame containing the observational data. Must include columns for 
            reaction times, choices, conditions, and subject identifiers as defined 
            in the model configuration.
            
        condition : str or list, optional
            Column name(s) to group by for hierarchical modeling. Can be a single 
            string for one grouping variable or a list of strings for multiple 
            grouping variables. If provided, separate posteriors are estimated for 
            each group. Default is None (pool all data).
            
        return_infdata : bool, optional
            Whether to return results as an ArviZ InferenceData object (True) or 
            as a pandas DataFrame (False). The InferenceData format provides better 
            support for Bayesian analysis workflows. Default is False.
            
        **kwargs : dict
            Additional keyword arguments passed to the sampling method. Common options include:
            - n_posterior: Number of posterior samples to draw
            - batchsize: Batch size for neural network evaluation
            - show_progress: Whether to display progress bar
            
        Returns:
        --------
        az.InferenceData or pd.DataFrame
            If return_infdata=True, returns an ArviZ InferenceData object containing 
            posterior samples and (if possible) observed data. Otherwise, returns a 
            pandas DataFrame with posterior draws.
            
        Example:
        --------
        >>> import pandas as pd
        >>> obs_df = pd.DataFrame({
        ...     'subject_id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        ...     'rt': [0.601, 0.451, 0.418, 0.451, 0.401, 
        ...            0.520, 0.390, 0.440, 0.380, 0.460],
        ...     'accuracy': [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        ...     'congruency': [1, 1, 1, 0, 1, 1, 0, 1, 0, 1]
        ... })
        >>> # Fit model with hierarchical structure by subject
        >>> posterior_samples = model.fit_data(
        ...     obs_df, 
        ...     n_posterior=50, 
        ...     batchsize=5, 
        ...     show_progress=True,
        ...     return_infdata=False
        ... )
        >>> print(posterior_samples.head())
        """

        grouped = self._check_df(obs_df)
        if not grouped:
            obs_df[self.subject_col] = 0
        coords = kwargs.pop("coords", [])
        # subj_param_colname_type = kwargs.pop("subj_param_colname_type", 0)

        group_list = []
        if condition is not None:
            if isinstance(condition, str):
                group_list.append(condition)
            elif isinstance(condition, list):
                group_list = condition

        if group_list:
            posterior_draws_df = (
                obs_df.groupby(group_list)
                .apply(lambda x: self.sample(self.df_to_forward_dict(x), **kwargs), include_groups=False) # type: ignore
                .reset_index()
            )
            posterior_draws_df.drop(f"level_{len(group_list)}", axis=1, inplace=True)

        else:
            posterior_draws_df = self.sample(self.df_to_forward_dict(obs_df), **kwargs)

        # TODO: fix some convert issues
        if return_infdata:
            if group_list:
                pivot_df = posterior_draws_df.pivot(
                    index=["chain", "draw"], columns=group_list, values=self.param_keys
                )
                # names = list(pivot_df.columns.names)
                # pivot_df.columns = pivot_df.columns.map(
                #     partial(
                #         generate_column_name, names=names, type=subj_param_colname_type
                #     )
                # )
                # pivot_df.columns = [
                #     subj_param_colname(col[0], col[1], type=subj_param_colname_type)
                #     if col[1] else col[1] for col in pivot_df.columns
                # ]
                posterior_draws_df = pivot_df
            else:
                posterior_draws_df.set_index(keys=["chain", "draw"], inplace=True)

            infdata = self.to_infdata({"posterior": posterior_draws_df}, coords)
            try:
                obs_data = obs_df.copy()
                obs_data = obs_data.convert_dtypes()
                if "trial" not in obs_data.columns:
                    obs_data["trial"] = obs_data.groupby(self.subject_col).cumcount()
                    obs_data["trial"] = obs_data["trial"].astype("int")
                obs_data.reset_index(inplace=True)
                obs_data[self.df_col[0]] = obs_data[self.df_col[0]].astype("float32")
                obs_data[self.df_col[1]] = obs_data[self.df_col[1]].astype("int")
                obs_data[self.subject_col] = obs_data[self.subject_col].astype("int")
                obs_data.index.name = "obs_id"
                xdata_observed = xr.Dataset.from_dataframe(obs_data)
                xdata_observed = xdata_observed.set_coords(["subject_id", "trial"])
                infdata.add_groups({"observed_data": xdata_observed})
            except Exception as e:
                warnings.warn(f"Error when adding observed data to InferenceData: {e}")

            return infdata
        else:
            return posterior_draws_df

    def to_infdata(self, df_dict: dict, coords=[]):
        return to_infdata(df_dict, coords)

    def az_summary(self, infdata=None, half_a=False, type=0, **kwargs):
        """
        Generates a summary of the posterior distribution using ArviZ.

        Parameters:
        infdata (az.InferenceData, optional): An ArviZ InferenceData object containing posterior samples.
            If not provided, the method uses the InferenceData object stored in the model instance.
        half_a (bool, optional): If True, divides the 'a' parameter values by 2. Default is False.
        type (int, optional): Determines the pattern used to extract parameter and subject information from the parameter names.
            0 for 'subj' pattern and 1 for '^' pattern. Default is 0.
        **kwargs: Additional keyword arguments to pass to the az.summary method.

        Returns:
        pd.DataFrame: A DataFrame with the summary statistics of the posterior distribution.

        The method uses ArviZ's summary function to generate statistics such as mean, standard deviation,
        and highest density interval (HDI) for the parameters of the model. It then processes the summary
        DataFrame to extract and reformat parameter and subject information based on the specified type.
        If half_a is True, it halves the values of the 'a' parameter.

        Example:
        >>> model_summary = model.az_summary(infdata=my_infdata, half_a=True)
        >>> print(model_summary)
        """

        if infdata is None:
            infdata = self.infdata

        param_df = az.summary(infdata, kind="stats", **kwargs).reset_index(
            names="param_name"
        )
        # col_values = ['mean', 'sd', "hdi_3%", "hdi_97%"]
        col_values = list(param_df.columns[1:5])

        match type:
            case 0:
                pattern = r"\$(.*)\|subj\((\d+)\)\$"
            case 1:
                pattern = r"\$(.*)^(\d+)"

        param_df[["param", self.subject_col]] = param_df["param_name"].str.extract(
            pattern
        )
        param_df["param"] = param_df["param"].apply(lambda x: f"${x}$")

        param_df[self.subject_col] = param_df[self.subject_col].astype(int)

        if half_a:
            param_df.loc[param_df["param"] == "a", col_values] = (
                param_df.loc[param_df["param"] == "a", col_values] / 2
            )

        param_df = param_df.pivot(
            index=self.subject_col, columns="param", values=col_values
        )

        new_index = pd.MultiIndex.from_tuples(
            [(level_0, param) for level_0 in col_values for param in self.param_names],
            names=[None, "param"],
        )
        param_df = param_df.reindex(columns=new_index)

        param_df.reset_index(inplace=True)
        param_df.columns.names = [None, None]

        return param_df

    def df_summary(
        self,
        df: pd.DataFrame,
        condition=None,
        type: Union[str, Callable] = "mean",
        **kwargs,
    ):
        """
        Generates a summary of the provided DataFrame based on specified aggregation methods.

        Parameters:
        df (pd.DataFrame): The DataFrame to summarize.
        condition (str or list, optional): The condition(s) to group by for hierarchical modeling.
            This can be a single column name (str) or a list of column names. Default is None.
        type (str or Callable, optional): The aggregation method to apply.
            It can be a string ('mean', 'describe', 'all') or a callable function. Default is 'mean'.

        Returns:
        pd.DataFrame: A DataFrame with the summarized statistics.

        The function groups the DataFrame by the specified conditions and applies the specified aggregation method
        to each group. The available string options for the type parameter are:
        - 'mean': Computes the mean of each numeric column.
        - 'describe': Generates descriptive statistics for each numeric column.
        - 'all': Generates all descriptive statistics for each numeric column.

        If a callable function is provided, it is applied to each group.

        Example:
        >>> df = pd.DataFrame({
        ...     'rt': [0.5, 0.7, 0.9, 0.6, 0.8],
        ...     'accuracy': [1, 0, 1, 1, 0],
        ...     'congruency': [0, 1, 0, 1, 0],
        ...     'subject': ['subj1', 'subj1', 'subj2', 'subj2', 'subj3']
        ... })
        >>> model_summary = model.df_summary(df, condition='subject', type='mean')
        >>> print(model_summary)
        """

        group_list = [self.subject_col]
        if condition is not None:
            if isinstance(condition, str):
                group_list.extend([condition])
            elif isinstance(condition, list):
                group_list.extend(condition)
        group_df = df.set_index(["chain", "draw"]).groupby(group_list)

        if isinstance(type, Callable):
            return group_df.apply(type, include_groups=False)
        elif isinstance(type, str):
            match type:
                case "mean":
                    return group_df.mean().reset_index()
                case "describe":
                    return group_df.describe().reset_index()
                case "all":
                    return group_df.describe().reset_index()

    def plot_prior_check(self, n_samples=100, **kwargs):
        """
        Generates a plot to check the prior distribution of the model parameters.

        This method generates a plot to visualize the prior distribution of the model parameters.
        It uses the prior_check method of the model to generate the prior samples and then plots
        the histograms of the parameters.

        Returns:
        None

        Example:
        >>> model.plot_prior_check()
        """

        sim_draws = self.cdms_simulator.simulator.sample(n_samples)

        f = bf.diagnostics.plots.pairs_samples(
            samples=sim_draws,
            variable_keys=self.param_keys,
            variable_names=self.param_names,
            **kwargs,
        )

    # def gen_datasets(
    #     self,
    #     param_array: Union[pd.DataFrame, pd.Series, np.ndarray] = None,
    #     idx: Union[str, int] = None,
    #     vary_param: np.ndarray = np.linspace(40, 120, 40 + 1),
    #     n_trial: int = 5000,
    #     n_prior: int = 100,
    #     save_data: bool = True,
    #     return_traindata: bool = True,
    # ):
    #     """_summary_

    #     Parameters
    #     ----------
    #     param_array : _type_
    #         _description_
    #     idx : _type_
    #         _description_
    #     vary_param : _type_, optional
    #         _description_, by default np.linspace(40,120,40+1)
    #     n_trial : int, optional
    #         _description_, by default 5000

    #     Returns
    #     -------
    #     _type_
    #         _description_

    #     Example:
    #     ------
    #     >>> m_DDM = CDMs_NSBI("DDM")
    #     >>> m_DDM.gen_datasets(DDM_estimates.values, 3, np.array([0,1,2,3,4]))
    #     """

    #     if isinstance(idx, str) and not isinstance(param_array, np.ndarray):
    #         param_matrix = param_matrix = gen_param_matrix(
    #             param_array.values, self.param_names.index(idx), vary_param
    #         )
    #     elif isinstance(idx, int) and isinstance(param_array, np.ndarray):
    #         param_matrix = gen_param_matrix(param_array, idx, vary_param)
    #     else:
    #         # raise ValueError(
    #         #     "when idx is int, param_array must be np.ndarray; while idx is str, param_array should be pd.Dataframe or pd.Series"
    #         # )
    #         param_matrix = self.model.prior(n_prior)["prior_draws"]

    #     outcome = {}
    #     if save_data:
    #         datalabel = DataLabel(self.model_name)
    #         for i in param_matrix:
    #             key = generate_key(i.round(3))
    #             if key in datalabel.store:
    #                 df = datalabel.get_dataframe(key)
    #             else:
    #                 df = self.simulate_data(n_trial=n_trial, params=i).iloc[:, -3:]
    #                 datalabel.add_dataframe(key, df)
    #             outcome[key] = df
    #         datalabel.close_store()
    #     else:
    #         if not return_traindata:
    #             outcome = []
    #             for i in param_matrix:
    #                 outcome.append(
    #                     {
    #                         "param": pd.Series(i, index=self.param_names),
    #                         "sim_data": self.simulate_data(
    #                             n_trial=n_trial, params=i
    #                         ).iloc[:, -3:],
    #                     }
    #                 )

    #             return outcome
    #         else:
    #             for i in param_matrix:
    #                 key = generate_key(i.round(3))
    #                 outcome[key] = self.simulate_data(n_trial=n_trial, params=i).iloc[
    #                     :, -3:
    #                 ]

    #     train_data = TrainData(
    #         model_name=self.model_name,
    #         parameters=DataFeature(values=param_matrix, label=self.param_names),
    #         outcome=outcome,
    #     )

    #     return train_data

    # def params_recovery(self, ground_truth, n_trial=5000, **kwargs):
    #     """
    #     Evaluates the model's ability to recover the specified ground truth parameters by simulating data
    #     and fitting the model to it.

    #     Parameters:
    #     ground_truth (np.ndarray): An array of ground truth parameters to use for simulating the data.
    #     n_trial (int, optional): The number of trials for each simulation. Default is 5000.
    #     **kwargs: Additional keyword arguments to pass to the fit_data method.

    #     Returns:
    #     az.InferenceData: An ArviZ InferenceData object containing the posterior samples after fitting the model
    #         to the simulated data.

    #     The function first simulates data using the specified ground truth parameters and the number of trials.
    #     It then fits the model to the simulated data using the fit_data method and returns the resulting
    #     InferenceData object, which contains the posterior samples.

    #     Example:
    #     >>> ground_truth_params = np.array([1.0, 0.5, 0.3])
    #     >>> recovery_result = model.params_recovery(ground_truth_params, n_trial=1000)
    #     >>> print(recovery_result)
    #     """

    #     sim_df = self.simulate_data(n_trial=n_trial, params=ground_truth)
    #     out = self.fit_data(sim_df, **kwargs)

    #     return out

    # def plot_params_recovery(self, ground_truth=None, n_trial=5000, **kwargs):
    #     if ground_truth is None:
    #         ground_truth = self.model.prior.prior()
    #     infdata = self.params_recovery(
    #         ground_truth, n_trial=n_trial, return_infdata=True, **kwargs
    #     )
    #     axes = az.plot_posterior(infdata, ref_val=list(ground_truth))

    #     return axes

    def posterior_predictive(self, params: Union[List, pd.DataFrame], n_trial=5000):
        if isinstance(params, pd.DataFrame):
            params_list = [
                df_i.drop(columns=[self.subject_col]).iloc[0].to_dict()
                for _, df_i in params.groupby(self.subject_col)
            ]
        elif isinstance(params, List):
            params_list = params

        predictive_df = self.simulate_data(n_trial=n_trial, params=params_list)

        return predictive_df

    # def plot_ppc(
    #     self,
    #     obs_data: pd.DataFrame,
    #     ppc_data: pd.DataFrame = None,
    #     params=None,
    #     flip_rt=True,
    #     **kwargs,
    # ):
    #     from nsbi_module.plotting import plot_ppc_rt_dists

    #     if ppc_data is None:
    #         if params is None:
    #             raise ValueError("Either ppc_data or params must be provided")
    #         ppc_data = self.posterior_predictive(params)

    #     assert all([self._check_df(obs_data), self._check_df(ppc_data)]), (
    #         "To make sure obs_data and ppc_data have same nessessary columns"
    #     )

    #     obs_data = obs_data.copy()
    #     ppc_data = ppc_data.copy()

    #     rt_col, res_col = self.df_col[:2]
    #     if len(self.df_col) > 2:
    #         hue = self.df_col[2]
    #     else:
    #         hue = None

    #     if flip_rt:
    #         obs_data[rt_col] = obs_data.apply(
    #             lambda row: row[rt_col] if row[res_col] == 1 else -row[rt_col], axis=1
    #         )
    #         ppc_data[rt_col] = ppc_data.apply(
    #             lambda row: row[rt_col] if row[res_col] == 1 else -row[rt_col], axis=1
    #         )

    #     if hasattr(self.model, "context_config"):
    #         context_config = self.model.context_config
    #         factor_levels = context_config.get("factor_levels", None)
    #         if factor_levels is not None:
    #             tmp_factor = list(factor_levels.keys())[0]
    #             codes, unique = pd.factorize(list(factor_levels.values())[0])
    #             map_dict = dict(zip(codes, unique))

    #             ppc_data[tmp_factor] = ppc_data[tmp_factor].map(map_dict)
    #             obs_data[tmp_factor] = obs_data[tmp_factor].map(map_dict)

    #     plot_ppc_rt_dists(
    #         obs_data, ppc_data, self.subject_col, hue=hue, dv=rt_col, **kwargs
    #     )

    # def params_recovery_by_nobs(
    #     self, ground_truth: np.ndarray, n_trial=[50, 100, 200, 500]
    # ):
    #     recovery_df = pd.DataFrame()
    #     for i in n_trial:
    #         infdata = self.params_recovery(ground_truth, n_trial=i)
    #         mean_estimates = az.summary(infdata, kind="stats")[
    #             ["mean", "sd"]
    #         ].reset_index(names="params")
    #         mean_estimates["n_obs"] = i

    #         recovery_df = recovery_df._append(mean_estimates, ignore_index=True)

    #     return recovery_df

    # def plot_params_recovery_by_nobs(
    #     self,
    #     recovery_df: pd.DataFrame,
    #     ground_truth: Union[pd.DataFrame, np.ndarray, pd.Series] = None,
    #     reg_order=3,
    #     scatter_kws={"color": "gray", "alpha": 0.3},
    #     n_col=4,
    #     **kwargs,
    # ):
    #     from nsbi_module.plotting import plot_params_recovery_by_nobs

    #     plot_params_recovery_by_nobs(
    #         recovery_df,
    #         ground_truth,
    #         order=reg_order,
    #         n_col=n_col,
    #         scatter_kws=scatter_kws,
    #         **kwargs,
    #     )

    # def cross_fit(
    #     self, sim_datasets: "TrainData", other_model: "NSBICDM", return_summary=True
    # ):
    #     parmap = ParMap(self.model_name)
    #     assert len(self.param_names) == sim_datasets.parameters.values.shape[1], (
    #         f"number of parameters is not matched: {sim_datasets.parameters.values}"
    #     )

    #     for param, data_i in sim_datasets.outcome_iter:
    #         if not parmap.isexist(param, other_model=other_model.model_name):
    #             if return_summary:
    #                 fit_params: pd.Series = (
    #                     other_model.fit_data(data_i, return_infdata=False)
    #                     .iloc[:, 2:]
    #                     .mean(axis=0)
    #                 )
    #             else:
    #                 fit_params: pd.DataFrame = other_model.fit_data(
    #                     data_i, return_infdata=False
    #                 )

    #             parmap.add_value(
    #                 key=pd.Series(param, index=self.param_names),
    #                 other_model=other_model.model_name,
    #                 value=fit_params,
    #             )

    #     param_map_df = parmap.get_dataframe(sim_datasets.parameters.values)
    #     parmap.close_store()

    #     return param_map_df


def to_infdata(df_dict: dict, coords=[]):
    xr_dict = {}
    for i, j in df_dict.items():
        xrdata = xr.Dataset.from_dataframe(j)
        if coords:
            xrdata = xrdata.set_coords(coords)
        xr_dict[i] = xrdata

    idata = az.InferenceData(**xr_dict)

    return idata


def subj_param_colname(param_name, subj_id, type=0):
    match type:
        case 0:
            pattern = rf"$\1|subj({subj_id})$"
        case 1:
            pattern = rf"$\1^{subj_id}$"

    return re.sub(r"\$(.*?)\$", pattern, param_name)


# def generate_column_name(tup, names=None, type=0):
#     para_name = subj_param_colname(tup[0], tup[1], type=type)
#     if para_name[-1] == "$":
#         para_name = para_name[:-1]

#     if names is not None:
#         for i in range(2, len(names)):
#             para_name += f"|{names[i]}({tup[i]})"

#     return para_name + "$" if para_name[0] == "$" else para_name


def gen_param_matrix(param_array: np.ndarray, idx, vary_param: np.ndarray):
    if param_array.ndim == 1:
        param_array = param_array[np.newaxis, :]

    n_row = param_array.shape[0]
    n_col = param_array.shape[1]
    assert idx < n_col, "idx is out of range"

    param_matrix = np.tile(param_array, (vary_param.size, 1))
    param_matrix[:, idx] = np.tile(vary_param, n_row)

    return param_matrix


@dataclass
class DataFeature:
    values: np.ndarray
    label: List[str]


class DataLabel:
    def __init__(self, model_name: str):
        self.store = pd.HDFStore(f"{model_name}_sim_datasets.h5")
        self.label = ["rt", "response", "congruency"]

    def add_dataframe(self, key: str, df: pd.DataFrame):
        self.store.put(key, df)

    def get_dataframe(self, key: str) -> pd.DataFrame:
        return self.store.get(key)

    def close_store(self):
        self.store.close()


@dataclass
class TrainData:
    model_name: str
    parameters: DataFeature
    outcome: Dict[str, pd.DataFrame]

    def __add__(self, other: "TrainData"):
        """_summary_

        Parameters
        ----------
        other : TrainData
            _description_

        Example
        -------
        >>> a = TrainData(...)
        >>> b = TrainData(...)
        >>> c = a + b
        """

        assert self.model_name == other.model_name, "Model name is not the same"
        assert self.parameters.label == other.parameters.label, (
            "Parameter label is not the same"
        )

        parameters = np.concatenate(
            [self.parameters.values, other.parameters.values], axis=0
        ).copy()
        outcome = self.outcome.copy()
        outcome.update(other.outcome)

        return TrainData(
            model_name=self.model_name,
            parameters=DataFeature(values=parameters, label=self.parameters.label),
            outcome=outcome,
        )

    @property
    def outcome_iter(self):
        """
        A property that returns an iterator of tuples, where each tuple contains
        a parameter value and its corresponding outcome value.

        Returns:
            iterator: An iterator of tuples (parameter_value:ndarray, outcome_value:dict={key,data}).
        """
        return zip(self.parameters.values, self.outcome.values())

    @property
    def outcome_merged(self):
        return [
            {"sim_data": data, "param": param}
            for data, (_, param) in zip(
                self.outcome.values(),
                pd.DataFrame(
                    self.parameters.values, columns=self.parameters.label
                ).iterrows(),
            )
        ]


def generate_key(params):
    param_str = str(params)
    param_str = param_str.strip("[] .")
    param_str = re.sub(r"\s+", "__", param_str)
    # Replace invalid characters with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", param_str)
    # return param_str
    return f"s_{sanitized}_e"


class NSBICDMs:
    """
    A class for fitting and analyzing DMC-NSBI models.

    Attributes:
    models (Dict[str, CDMs_NSBI]): A dictionary of CDMs_NSBI model instances to be fitted.
    data (pd.DataFrame, optional): The observational data to be used for fitting the models. Default is None.
        - 'rt': Reaction time values, expected to be in seconds and less than 998.
        - 'accuracy': Response values, which should only be 0 or 1 (usually 0 for error and 1 for correct response).
        - 'congruency': Congruency values, which should only be 0 or 1.
        - 'subject_id'(optional): Subject index values, which should be unique for each subject.
    condition (str or list, optional): The condition(s) to group by for hierarchical modeling.
        This can be a single column name (str) or a list of column names. Default is None.

    The class provides methods for fitting the models to the data, summarizing the results,
    and simulating new data based on the fitted models.

    Example:
    >>> # Initialize CDMs_NSBI models with specific checkpoint paths
    >>> m_DDM = CDMs_NSBI("DDM", checkpoint_path="../../checkpoints/DDM")
    >>> m_DMC = CDMs_NSBI("DMC", checkpoint_path="../../checkpoints/DMC")
    >>> m_SSP = CDMs_NSBI("SSP", checkpoint_path="../../checkpoints/SSP")
    >>> m_DSTP = CDMs_NSBI("DSTP", checkpoint_path="../../checkpoints/DSTP")

    >>> # Group the initialized models into a dictionary
    >>> models = {
    ...     "DDM": m_DDM,
    ...     "DMC": m_DMC,
    ...     "SSP": m_SSP,
    ...     "DSTP": m_DSTP
    ... }

    >>> # Create a sample DataFrame containing observational data
    >>> data = pd.DataFrame({
    ...     'subject_id': ['1', '1', '2', '2', '3']  # Subject indices
    ...     'rt': [0.5, 0.7, 0.9, 0.6, 0.8],  # Reaction times
    ...     'accuracy': [1, 0, 1, 1, 0],  # Response values (1 for correct, 0 for error)
    ...     'congruency': [0, 1, 0, 1, 0],  # Congruency values (0 or 1)
    ... })

    >>> # Create an instance of CDMs_NSBI_Fit to fit the models to the data
    >>> fit = CDMs_NSBI_Fit(models=models, data=data, condition='subject_id')
    """

    def __init__(
        self,
        models: Dict[str, NSBICDM],
        data: pd.DataFrame = None,
        condition=None,
        group_list=["subject_id"],
    ):
        self.models = models
        self.models_name = list(models.keys())
        self.data = data

        self.condition = condition

        if condition is not None:
            if isinstance(condition, str):
                group_list.extend([condition])
            elif isinstance(condition, list):
                group_list.extend(condition)
        self.group_list = group_list

    def __get__(self, model_name: str):
        if model_name in self.models:
            return self.models[model_name]

    def fit_data(
        self, data: pd.DataFrame = None, condition=None, return_summary=True, **kwargs
    ):
        """
        Fits the models to the provided or stored observational data.

        Parameters:
        data (pd.DataFrame, optional): The DataFrame containing the observational data to fit the models to.
            - 'rt': Reaction time values, expected to be in seconds and less than 998.
            - 'accuracy': Response values, which should only be 0 or 1 (usually 0 for error and 1 for correct response).
            - 'congruency': Congruency values, which should only be 0 or 1.
            - 'subject_id'(optional): Subject index values, which should be unique for each subject.
        condition (str or list, optional): The condition(s) to group by for hierarchical modeling.
            This can be a single column name (str) or a list of column names.
            If not provided, the stored condition in the instance is used.

        Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are model names and values are DataFrames
            containing the fitted parameters for each model.

        Example:
        >>> fit = CDMs_NSBI_Fit(models=models, data=data, condition='subject_id')
        >>> fitted_parameters = fit.fit_data()
        >>> print(fitted_parameters['DDM'].head())
        """

        if data is None:
            data = self.data
            if data is None:
                raise Exception("No data provided")
        if condition is None:
            condition = self.condition

        fitted_params = {}
        for model_name, model in self.models.items():
            posterior = model.fit_data(
                data, condition=condition, return_infdata=False, **kwargs
            )

            if return_summary:
                posterior = model.df_summary(posterior, condition, **kwargs) # type: ignore
            fitted_params[model_name] = posterior

        self.fitted_params = fitted_params

        return fitted_params

    def df_summary(
        self,
        df_dict: Dict,
        condition=None,
        type: Union[str, Callable] = "mean",
        **kwargs,
    ):
        if condition is None:
            condition = self.condition

        fitted_params = {}
        df_dict_keys = df_dict.keys()
        for model_name, model in self.models.items():
            if model_name in df_dict_keys:
                posterior_summary = model.df_summary(
                    df_dict[model_name], condition, **kwargs
                )

                fitted_params[model_name] = posterior_summary

        return fitted_params

    def posterior_predictive(
        self, params_dict: Dict[str, Union[List, pd.DataFrame]], n_trial: int = 5000
    ) -> Dict[str, pd.DataFrame]:
        """
        Generates posterior predictive data for each model using the provided parameter samples.

        Parameters:
        params_dict (Dict[str, pd.DataFrame]): A dictionary where keys are model names and values are DataFrames
            containing the parameter samples for each model.
        n_trial (int, optional): The number of trials for each posterior predictive simulation. Default is 5000.

        Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are model names and values are DataFrames
            containing the posterior predictive data for each model.

        Example:
        >>> fit = CDMs_NSBI_Fit(models=models, data=data, condition='subject_id')
        >>> fitted_parameters = fit.fit_data()
        >>> pp_data = fit.posterior_predictive(fitted_parameters, n_trial=1000)
        >>> print(pp_data['DDM'].head())
        """

        pp_data = {}
        for model_name, model in self.models.items():
            pp_data[model_name] = model.posterior_predictive(
                params_dict[model_name], n_trial=n_trial
            )

        self.pp_data = pp_data
        return pp_data

    def get_plotfit(
        self, pp_data: Dict[str, pd.DataFrame], obs_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, PlotFit]:
        """
        Generates PlotFit objects for each model using the posterior predictive data and observational data.

        Parameters:
        pp_data (Dict[str, pd.DataFrame]): A dictionary where keys are model names and values are DataFrames
            containing the posterior predictive data for each model.
        obs_data (pd.DataFrame, optional): The observational data to be used for generating the PlotFit objects.
            If not provided, the stored observational data in the instance is used.

        Returns:
        Dict[str, PlotFit]: A dictionary where keys are model names and values are PlotFit objects
            for each model.

        The method creates PlotFit objects for each model by pairing the posterior predictive data with the observational data.
        These PlotFit objects can be used to visualize the fit of the model to the data. The resulting PlotFit objects
        are stored in the instance's plot_dict attribute and also returned in a dictionary.

        Example:
        >>> fit = CDMs_NSBI_Fit(models=models, data=data, condition='subject_id')
        >>> fitted_parameters = fit.fit_data()
        >>> pp_data = fit.posterior_predictive(fitted_parameters, n_trial=1000)
        >>> plot_dict = fit.get_plotfit(pp_data)
        >>> CDMs_fit.plot_prediction2(plot_dict, axes=axd, type="rt_correct")
        """

        if obs_data is None:
            obs_data = self.data.copy()

        plot_dict = {}
        for model_name, pp_data_i in pp_data.items():
            fit_plot = PlotFit(obs_data, pp_data_i)

            plot_dict[model_name] = fit_plot

        self.plot_dict = plot_dict
        return plot_dict


def pyCDMs_plot(fit_plot: PlotFit, ax, type="summary", **kwargs):
    match type:
        case "summary":
            fit_plot.summary(show=False, axs=ax, **kwargs)
        case "rt_correct":
            fit_plot.rt_correct(ax=ax, show=False, **kwargs)
        case "er":
            fit_plot.er(ax=ax, show=False, **kwargs)
        case "rt_error":
            fit_plot.rt_error(ax=ax, show=False, **kwargs)
        case "cdf":
            fit_plot.cdf(ax=ax, show=False, **kwargs)
        case "caf":
            fit_plot.caf(ax=ax, show=False, **kwargs)
        case "delta":
            fit_plot.delta(ax=ax, show=False, **kwargs)

