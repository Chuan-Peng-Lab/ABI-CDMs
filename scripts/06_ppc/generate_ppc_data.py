#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

from nsbi_module.NSBI_CDMs import NSBICDM, NSBICDMs
from nsbi_module.utils import FitStore
from nsbi_module.plotting import *
from nsbi_module.utils_pydmc import Ob
from nsbi_module.project_paths import (
    CHECKPOINTS_DIR,
    INTERMEDIATE_DIR,
    SUPPLEMENT_FIGURES_DIR,
    ensure_output_directories,
)



# ─── Model palette ────────────────────────────────────────────────────────────
MODELS = ['DDM', 'SSP', 'DMC', 'DSTP']
MODEL_COLORS = {
    'DDM':  '#b0d97c',
    'SSP':  '#92d28e',
    'DMC':  '#95d8c3',
    'DSTP': '#81cef0',
}
DEFAULT_TASKS      = ['flanker', 'simon', 'stroop']
DEFAULT_TASK_ORDER = ['Flanker', 'Simon', 'Stroop']




m_DDM = NSBICDM("DDM", checkpoint_path=CHECKPOINTS_DIR / "DDM")
m_DMC = NSBICDM("DMC", checkpoint_path=CHECKPOINTS_DIR / "DMC")
m_SSP = NSBICDM("SSP", checkpoint_path=CHECKPOINTS_DIR / "SSP")
m_DSTP = NSBICDM("DSTP", checkpoint_path=CHECKPOINTS_DIR / "DSTP")

models = {
    "DDM": m_DDM,
    "DMC": m_DMC,
    "SSP": m_SSP,
    "DSTP": m_DSTP
}

CDMs_fit = NSBICDMs(models)


# ## Load datasets



df_dict = {}
ensure_output_directories()
with pd.HDFStore(INTERMEDIATE_DIR / "datasets_cross_sectional.h5") as hdfstore:
    for key in hdfstore.keys():
        key = key[1:]
        df_dict[key] = hdfstore[key]
df_dict.keys()


# ## Generate posterior prediction
#
# It will cost 1 hour and 26 minutes to generate the posterior prediction.



def generate_predictions(fit_store, n_samples_posterior=200, n_trials_simulation=2):
    """
    Generates posterior predictive data for all fitted traces in the fit_store.

    Args:
        fit_store: The storage object containing fitted traces.
        n_samples_posterior (int): Number of posterior samples to draw (iterations).
        n_trials_simulation (int): Number of trials to simulate per sample.
    """

    # 1. Identify keys to process
    # Assuming fit_store.key_list_df is available as per your code
    all_keys = fit_store.key_list_df["key"].values
    keys_to_process = [k for k in all_keys if k.endswith("_fitted_trace")]

    print(f"Found {len(keys_to_process)} datasets to process.")

    final_results = {}
    # Outer loop: Iterate through each dataset (file/experiment)
    for key in tqdm(keys_to_process, desc="Datasets"):

        # Define target key
        key_predicted = key + "_predicted"

        # Check existence
        if fit_store.isexist(key_predicted):
            # print(f"[{key_predicted}] exists. Skipping.")
            final_results[key_predicted] = fit_store[key_predicted]
            continue

        # Load data
        fit_results = fit_store[key]

        # Prepare storage for simulation inputs (batches)
        # Structure: { batch_id (0-199): { 'DDM': df_param, ... } }
        batched_inputs = {i: {} for i in range(n_samples_posterior)}

        # Instead of sampling inside the loop 200 times, we sample once.
        for model_name, df_trace in fit_results.items():
            # Clean data
            cols_to_drop = [c for c in ["chain", "draw"] if c in df_trace.columns]
            if cols_to_drop:
                df_trace.drop(columns=cols_to_drop, inplace=True)

            # Efficient Sampling:
            # Group by subject and sample 'n_samples_posterior' rows with replacement.
            # replace=True ensures we can sample 200 times even if chain length < 200 (though unlikely).
            sampled_df = df_trace.groupby("subject_id").sample(n=n_samples_posterior, replace=True).copy()

            # Assign a batch_id (0 to 199) to each sample within each subject
            # cumcount() numbers the items 0, 1, ..., 199 for each group (subject)
            sampled_df["_batch_id"] = sampled_df.groupby("subject_id").cumcount()

            # Distribute into batches
            for batch_id, group in sampled_df.groupby("_batch_id"):
                # Remove the helper column and store
                batched_inputs[batch_id][model_name] = group.drop(columns=["_batch_id"])

        # Container for results
        collected_results = {model: [] for model in fit_results.keys()}

        # Inner Loop: Run Simulation
        # Using leave=False to clear the inner progress bar after completion to keep output clean
        for i in tqdm(range(n_samples_posterior), desc=f"Simulating {key}", leave=False):

            # Get the pre-prepared input for this iteration
            temp_input_dict = batched_inputs[i]

            # Run simulation
            pp_batch = CDMs_fit.posterior_predictive(temp_input_dict, n_trial=n_trials_simulation)

            # Collect results
            for model_name, df_sim in pp_batch.items():
                collected_results[model_name].append(df_sim)

        # Merge results
        final_merged_results = {}
        for model_name, list_of_dfs in collected_results.items():
            final_merged_results[model_name] = pd.concat(list_of_dfs, ignore_index=True)

        # Save to store
        final_results[key_predicted] = final_merged_results
        fit_store[key_predicted] = final_merged_results

    return final_results

fit_store = FitStore(INTERMEDIATE_DIR / "model_fits.h5")
ppd_dict = generate_predictions(fit_store, n_samples_posterior=200, n_trials_simulation=2)
fit_store.close_store()




ppd_dict["clayson2025flanker_fitted_trace_predicted"]["DDM"]


# ### calculate plotting indicators



import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def generate_plotting_data(
    df_dict: Dict[str, pd.DataFrame],
    ppd_dict: Dict[str, Dict[str, pd.DataFrame]],
    Ob_cls: Any,
    ppd_suffix: str = "_fitted_trace_predicted",
    subject_id_col: str = "subject_id",
    rt_col: str = "rt",
    acc_col: str = "accuracy",
    cond_col: str = "congruency"
) -> Dict[str, Any]:
    """
    Processes observed and model-predicted behavioral data using the `Ob` class to generate
    a structured dictionary for plotting metrics (CDF, CAF, Delta plots, Subject Means).

    Args:
        df_dict (Dict[str, pd.DataFrame]): Dictionary of observed datasets.
            Key is dataset name, Value is the trial-level DataFrame.
        ppd_dict (Dict[str, Dict[str, pd.DataFrame]]): Dictionary of posterior predictive data.
            Key is dataset name (usually with suffix), Value is a dict of {ModelName: DataFrame}.
        Ob_cls (class): The core `Ob` class used for calculating behavioral metrics.
            Must support instantiation via `Ob(df)` and have attributes `summary_subject`, `caf`, `delta`.
        ppd_suffix (str, optional): Suffix used in `ppd_dict` keys to match with `df_dict` keys.
            Defaults to "_fitted_trace_predicted".
        subject_id_col (str, optional): Column name for subject ID. Defaults to "subject_id".
        rt_col (str, optional): Column name for Response Time. Defaults to "rt".
        acc_col (str, optional): Column name for Accuracy. Defaults to "accuracy".
        cond_col (str, optional): Column name for Congruency condition. Defaults to "congruency".

    Returns:
        Dict[str, Any]: A nested dictionary structured as:
            {
                "dataset_name": {
                    "observed": { "subject_means": ..., "caf": ..., "delta": ..., "cdf": ... },
                    "models": {
                        "ModelName": { "subject_means": ..., "caf": ..., "delta": ..., "cdf": ... },
                        ...
                    }
                },
                ...
            }
    """

    # --- Internal Helper Function ---
    def _extract_metrics(df: pd.DataFrame, source_label: str) -> Dict[str, pd.DataFrame]:
        """
        Instantiates the Ob class for a single dataframe and extracts key metrics.

        Args:
            df: The raw trial-level dataframe.
            source_label: Label for logging (e.g., 'Observed' or 'Model X').
        """
        # Ensure column mapping matches what Ob expects if necessary
        # (Assuming Ob handles standard column names, but validation could happen here)

        # 1. Instantiate the tool
        try:
            analysis = Ob_cls(df)
        except Exception as e:
            print(f"Error initializing Ob for {source_label}: {e}")
            return {}

        # 2. Extract Subject Means (Goodness of Fit)
        # Ob.summary_subject contains: Subject, Comp, n, n_cor, rt_cor, per_err
        subj_means = analysis.summary_subject.copy()

        # 3. Extract CAF (Conditional Accuracy Function)
        # Ob.caf contains: bin, comp (acc), incomp (acc), effect
        caf_data = analysis.caf.copy()

        # 4. Extract Delta Plot Data
        # Ob.delta contains: bin, mean_comp, mean_incomp, mean_bin, mean_effect
        delta_data = analysis.delta.copy()

        # 5. Extract CDF (Cumulative Distribution Function)
        # As per requirements, CDF data is derived from the delta object
        # (using mean_comp and mean_incomp per bin).
        cdf_data = analysis.delta.copy()
        # We ensure the relevant columns for CDF are present (bin, mean_comp, mean_incomp)
        if not {'mean_comp', 'mean_incomp'}.issubset(cdf_data.columns):
            print(f"Warning: Missing columns for CDF in {source_label}. Available: {cdf_data.columns}")

        return {
            "subject_means": subj_means,
            "caf": caf_data,
            "delta": delta_data,
            "cdf": cdf_data
        }

    # --- Main Processing Loop ---
    plotting_data = {}

    print(f"Starting processing for {len(df_dict)} observed datasets...")

    for dataset_name, df_obs in df_dict.items():
        print(f"Processing dataset: {dataset_name}")

        plotting_data[dataset_name] = {
            "observed": {},
            "models": {}
        }

        # 1. Process Observed Data
        obs_metrics = _extract_metrics(df_obs, source_label=f"Observed-{dataset_name}")
        plotting_data[dataset_name]["observed"] = obs_metrics

        # 2. Process Model Data (if available)
        # Construct the expected key for the predicted data dict
        ppd_key = f"{dataset_name}{ppd_suffix}"

        if ppd_key in ppd_dict:
            model_dict = ppd_dict[ppd_key]

            for model_name, df_model in model_dict.items():
                # Process each model (e.g., DDM, DMC)
                model_metrics = _extract_metrics(df_model, source_label=f"Model-{model_name}")
                plotting_data[dataset_name]["models"][model_name] = model_metrics
        else:
            print(f"Note: No predictive data found for {dataset_name} (Expected key: {ppd_key})")

    print("Data processing complete.")
    return plotting_data




# it will cost 2 mins
plotting_data = generate_plotting_data(
    df_dict=df_dict,
    ppd_dict=ppd_dict,
    Ob_cls=Ob,
    ppd_suffix="_fitted_trace_predicted"
)




with (INTERMEDIATE_DIR / "ppc_data.pkl").open("wb") as output_file:
    pickle.dump(plotting_data, output_file)


# ## Plot PPC

# ### CAF



fig = plot_distribution_curve(
    plotting_data,
    plot_type='caf',
    # figsize=(12, 30),
    model_colors = MODEL_COLORS,
    known_tasks = DEFAULT_TASKS,
    task_order = DEFAULT_TASK_ORDER,
    save_name=SUPPLEMENT_FIGURES_DIR / "ppc_caf_full.svg")


# ### Delta RTs



fig = plot_delta_functions(
    plotting_data,
    structured_layout=True,
    figsize=(3.5, 2.5),
    model_colors = MODEL_COLORS,
    known_tasks = DEFAULT_TASKS,
    task_order = DEFAULT_TASK_ORDER,
    save_name=SUPPLEMENT_FIGURES_DIR / "ppc_delta_full.svg"
)


# ### Selected Examples



selected_data = {key: plotting_data[key] for key in plotting_data.keys()
                 if key.startswith(('eisenberg', 'hedge', 'reymermet'))}




fig = plot_distribution_curve(
    selected_data,
    plot_type='caf',
    # figsize=(12, 30),
    model_colors = MODEL_COLORS,
    known_tasks = DEFAULT_TASKS,
    task_order = DEFAULT_TASK_ORDER,
    save_name=SUPPLEMENT_FIGURES_DIR / "ppc_caf_selected.svg")




fig = plot_delta_functions(
    selected_data,
    structured_layout=True,
    figsize=(3.5, 2.5),
    model_colors = MODEL_COLORS,
    known_tasks = DEFAULT_TASKS,
    task_order = DEFAULT_TASK_ORDER,
    save_name=SUPPLEMENT_FIGURES_DIR / "ppc_delta_selected.svg"
)
