#!/usr/bin/env python
# coding: utf-8



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
from nsbi_module.model_metrics import ModelMetricEvaluator
from nsbi_module.NSBI_CDMs import NSBICDM, NSBICDMs
from nsbi_module.utils import timer, cache_from_file
from nsbi_module.project_paths import CHECKPOINTS_DIR, INTERMEDIATE_DIR, SUPPLEMENT_FIGURES_DIR, ensure_output_directories



ensure_output_directories()
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


# ## validation

# ### simulating data and fitting



@timer
@cache_from_file(INTERMEDIATE_DIR / "model_recovery_cross_fits.pkl")
def cross_fitting(
    models,
    n_prior = 5,
    n_trial = 50,
    n_posterior = 5000
    ):

    model_names = list(models.keys())
    total_iterations = len(model_names) **2 * n_prior

    result = {}
    with tqdm(total=total_iterations) as pbar:
        for sim_model in model_names:

            # Create a list to store results for each simulated dataset
            output_list = []

            # Generate simulated datasets once per simulation model
            prior_samples = models[sim_model].cdms_simulator.sample_prior(n_prior)
            sim_datasets = models[sim_model].simulate_data(
                # n_sim = n_prior,
                n_trial = n_trial,
                params=prior_samples
            )

            # For each simulated dataset, fit with all models
            for dataset_idx, dataset_i in sim_datasets.groupby("subject_id"):

                # Create a dictionary for this dataset
                dataset_result = {
                    "prior": {
                        "param": prior_samples[dataset_idx],
                        "sim_data": dataset_i
                    },
                    "dataset_idx": dataset_idx
                }

                # Fit with each fitting model
                for fit_model in model_names:

                    output_tmpi = {}
                    pbar.set_description(
                        f"Simulating model {sim_model} -> Fitting model {fit_model}"
                    )

                    param_posteriors = models[fit_model].fit_data(
                        dataset_i, n_posterior=n_posterior, return_infdata=False
                    )
                    param_summary_df = models[fit_model].df_summary(param_posteriors)

                    output_tmpi["param_trace"] = param_posteriors
                    output_tmpi["param"] = param_summary_df

                    output_tmpi["sim_data"] = models[fit_model].posterior_predictive(
                        param_summary_df, n_trial=n_trial
                    )

                    dataset_result[fit_model] = output_tmpi

                    pbar.update(1)

                output_list.append(dataset_result)

            result[sim_model] = output_list

    return result




# simulate n_prior prior parameter sets
n_prior = 200
# simulate n_trial trial (n_trial/2 for each congruency condition) for each parameter set
# and simulate n_trial trials for posterior predictive
n_trial = 400

fitting_result = cross_fitting(models, n_prior=n_prior, n_trial=n_trial)
# {
#   'DDM': [
#     {
#       'prior': DataFrame,
#       'DDM': {'param_trace', 'param', 'sim_data'},
#       'DMC': {...},
#       'SSP': {...},
#       'DSTP': {...},
#       'dataset_idx': 0
#     },
#     {
#       'prior': {...},
#       'DDM': {...},
#       'DMC': {...},
#       'SSP': {...},
#       'DSTP': {...},
#       'dataset_idx': 1
#     },
#     # ... more datasets
#   ],
#   'DMC': [
#     {
#       'prior': {...},
#       'DDM': {...},
#       'DMC': {...},
#       'SSP': {...},
#       'DSTP': {...},
#       'dataset_idx': 0
#     },
#     # ... more datasets
#   ],
#   # ... other models
# }
# empirical_df = fitting_result["DMC"][0]["prior"]["sim_data"]
# predicted_df = fitting_result["DMC"][0]["DSTP"]["sim_data"]


# ### preprocessing
#
# It will take 4min to run this.



# empirical_df = fitting_result["DMC"][0]["prior"]["sim_data"]
# predicted_df = fitting_result["DMC"][0]["DSTP"]["sim_data"]
# preprocess_config = {
#         "n_bins_cdf_props": 10,
#         "n_bins_caf_props": 5
# }
# evaluator = ModelMetricEvaluator(
#     empirical_df=empirical_df,
#     predicted_df=predicted_df,
#     config=preprocess_config,
# )
# rmse = evaluator.compute_rmse()
# g_square = evaluator.compute_chi_square_proportions(n_trial=n_trial)
# abic = evaluator.compute_abic_proportions(
#     n_param=5,
#     n_trial=100,
# )




# evaluator._empirical_summary.cdf_props.sort_values(
#             ["condition", "bin"])




def evaluate_model_recovery_metrics(
    empirical_df,
    predicted_df,
    fit_model,
    n_param_dict,
    n_trial,
    preprocess_config={
        "n_bins_caf": 5,
        "n_bins_cdf": 10,
        "n_bins_caf_props": 5,
        "n_bins_cdf_props": 10
    }
):
    evaluator = ModelMetricEvaluator(
        empirical_df=empirical_df,
        predicted_df=predicted_df,
        config=preprocess_config,
    )
    rmse = evaluator.compute_rmse()
    g_square = evaluator.compute_chi_square_proportions(n_trial=n_trial)
    abic = evaluator.compute_abic_proportions(
        n_param=n_param_dict[fit_model],
        n_trial=n_trial,
    )
    # g_square = evaluator.compute_g_square()
    # abic = evaluator.compute_abic(
    #     n_param=n_param_dict[fit_model],
    #     n_trial=n_trial,
    # )
    return {
        "RMSE": rmse["rmse"],
        "g_square": g_square["g_square"],
        "aBIC": abic["aBIC"],
    }


@timer
# @cache_from_file("13model_recovery_result.pkl")
def process_cross_fitting_results(fitting_result, n_trial, n_slice=None, n_jobs=32):
    """
    Process cross fitting results and calculate metrics using parallel computing.

    Parameters:
    fitting_result: dict - The cross fitting results
    n_trial: int - Number of trials used in simulation
    n_slice: int, optional - Slice the data list for testing
    n_jobs: int - Number of parallel jobs (-1 for all CPUs)

    Returns:
    pandas.DataFrame - DataFrame with all metrics
    """

    from joblib import Parallel, delayed

    # Get model names and parameter count dictionary
    model_names = list(fitting_result.keys())
    n_param_dict = {
        "DDM": 4,
        "SSP": 5,
        "DMC": 6,
        "DSTP": 7
    }

    def _process_single_dataset(data_i, sim_model):
        """
        Helper function to process a single dataset against all fit models.
        Processing at this level avoids recalculating 'prior_data' logic multiple times.
        """
        results_buffer = []
        prior_data = data_i["prior"]["sim_data"]

        for fit_model in model_names:
            pred_data_i = data_i[fit_model]["sim_data"]

            metric_result = evaluate_model_recovery_metrics(
                empirical_df=prior_data,
                predicted_df=pred_data_i,
                fit_model=fit_model,
                n_param_dict=n_param_dict,
                n_trial=n_trial,
            )

            results_buffer.append({
                "id": data_i["dataset_idx"],
                "sim_model": sim_model,
                "fit_model": fit_model,
                "aBIC": metric_result["aBIC"],
                "g_square": metric_result["g_square"],
                "RMSE": metric_result["RMSE"],
            })

        return results_buffer

    # 1. Flatten the loops into a task list
    # We loop over sim_model and data_list to prepare tasks for parallel execution
    tasks = []
    for sim_model, data_list in fitting_result.items():
        # Apply slice if provided
        current_data_list = data_list[:n_slice] if n_slice is not None else data_list
        for data_i in current_data_list:
            tasks.append((data_i, sim_model))

    # 2. Execute in parallel
    # Returns a list of lists (one list of results per dataset)
    results_nested = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_dataset)(data, model) for data, model in tasks
    )

    # 3. Flatten the results
    rows = [item for sublist in results_nested for item in sublist]

    return pd.DataFrame(rows)

predicting_restult = process_cross_fitting_results(
    fitting_result,
    n_trial=n_trial,
    # n_slice=20
    )




predicting_restult.head()


# ### plot



def calculate_cross_table(
    df: pd.DataFrame,
    id_col: str = "id",
    sim_col: str = "sim_model",
    fit_col: str = "fit_model",
    sort_models: bool = True
) -> dict:
    """
    Compute cross-tabulation tables for model recovery analysis with consistent ordering.

    For each metric column (e.g., BIC, RMSE), identifies which fitted model achieved the
    best (minimum) value per subject and simulated model. Returns normalized proportions
    with rows (sim_model) and columns (fit_model) sorted consistently across all metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: id, sim_model, fit_model, and one or more numeric metrics.
    id_col : str, default "id"
        Column identifying individual subjects/simulations.
    sim_col : str, default "sim_model"
        Column indicating the true/simulated model.
    fit_col : str, default "fit_model"
        Column indicating the candidate/fitted model.
    sort_models : bool, default True
        If True, sort sim_model and fit_model alphabetically.
        If you need a custom order, set this to False and pre-sort your model labels
        using pandas categorical ordering before calling this function.

    Returns
    -------
    dict of pd.DataFrame
        Each value is a DataFrame with:
            - Index: sim_model (sorted)
            - Columns: fit_model (sorted)
            - Values: proportion of subjects where fit_model was best for that metric.
        Rows sum to 1. All DataFrames share identical row/column order.
    """
    reserved = {id_col, sim_col, fit_col}
    metric_cols = [col for col in df.columns if col not in reserved]

    if not metric_cols:
        raise ValueError("No metric columns found.")

    # Determine consistent model orders
    if sort_models:
        sim_order = sorted(df[sim_col].unique())
        fit_order = sorted(df[fit_col].unique())
    else:
        # Respect categorical order if provided, or use appearance order
        sim_order = pd.unique(df[sim_col])
        fit_order = pd.unique(df[fit_col])

    results = {}

    # Compute number of unique IDs per sim_model (for normalization)
    n_ids_per_sim = df.groupby(sim_col)[id_col].nunique()

    for metric in metric_cols:
        # For each (id, sim_model), find the fit_model with minimum metric
        best_idx = df.groupby([id_col, sim_col])[metric].idxmin()
        print(f"For {metric}, {best_idx.isna().sum()} nan values dropped")
        best_rows = df.loc[best_idx.dropna()]

        # Count best fit_model per sim_model
        counts = best_rows.groupby(sim_col)[fit_col].value_counts()
        cross_tab = counts.unstack(fill_value=0)

        # Reindex to ensure consistent row and column order
        cross_tab = cross_tab.reindex(index=sim_order, columns=fit_order, fill_value=0)

        # Normalize by number of subjects per sim_model
        cross_tab_norm = cross_tab.div(n_ids_per_sim.reindex(sim_order), axis=0)

        # Fill potential NaN (if a sim_model had no data) with 0
        cross_tab_norm = cross_tab_norm.fillna(0.0)

        results[metric] = cross_tab_norm

    return results

def plot_heatmap(df, title="rmse", ax=None, save=True, save_path=SUPPLEMENT_FIGURES_DIR):

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
        created_fig = True
    else:
        created_fig = False

    sns.heatmap(
        df,
        annot=True,
        cmap="coolwarm",
        cbar=False,
        ax=ax
    )

    ax.set_xlabel('Fitting Model')
    ax.set_ylabel('Simulating Model')

    if title:
        ax.set_title(title)

    if save:
        title = title.replace(" ", "_") if title else ""
        plt.savefig(save_path / f"figure_s04_model_recovery_{title}.svg", bbox_inches='tight')




cross_tables = calculate_cross_table(predicting_restult)




plot_heatmap(cross_tables["RMSE"])




fig, axes = plt.subplots(1, 2, figsize=(10, 4))

plot_heatmap(cross_tables["g_square"], ax=axes[0], title="G square", save=False)
plot_heatmap(cross_tables["aBIC"], ax=axes[1], title="aBIC", save=False)

plt.savefig(SUPPLEMENT_FIGURES_DIR / "figure_s04_model_recovery_metrics.svg", bbox_inches='tight')



