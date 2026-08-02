#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import seaborn as sns

from nsbi_module.NSBI_CDMs import NSBICDM, NSBICDMs
from nsbi_module.utils import timer, cache_from_file
from nsbi_module.project_paths import CHECKPOINTS_DIR, INTERMEDIATE_DIR, SUPPLEMENT_FIGURES_DIR, ensure_output_directories

# ## Definition



from tqdm import tqdm
import torch
from nsbi_module.plotting import regplot_with_corr

@timer
def gen_data_and_fitting(
    models, params_fixed_dict, param_key, params_vary, n_trial=200
):
    df_output = pd.DataFrame()
    model_names = list(models.keys())

    # Create progress bar with total length = model_names^2
    total_iterations = len(model_names) * len(model_names)
    pbar = tqdm(total=total_iterations, desc="Model Comparison Progress")

    for sim_model in model_names:
        params_fixed = params_fixed_dict[sim_model]
        param_dict_list = [
            {**params_fixed, param_key: param_value_i} for param_value_i in params_vary
        ]
        sim_param_df = (
            pd.DataFrame(param_dict_list)
            .reset_index(names=["subject_id"])
            .melt(id_vars=["subject_id"], var_name="params", value_name="prior")
        )

        sim_data = models[sim_model].simulate_data(
            n_trial=n_trial, params=param_dict_list
        )

        for fit_model in model_names:
            pbar.set_description(
                f"Simulating model {sim_model} -> Fitting model {fit_model}"
            )

            param_posteriors = models[fit_model].fit_data(
                sim_data, return_infdata=False
            )
            param_summary_df = models[fit_model].df_summary(param_posteriors)
            df_long = param_summary_df.melt(
                id_vars=["subject_id"], var_name="params", value_name="estimates"
            )
            df_long["sim_model"] = sim_model
            df_long["fit_model"] = fit_model

            df_combined = pd.merge(df_long, sim_param_df, on=["subject_id", "params"])

            df_output = pd.concat([df_output, df_combined], axis=0, ignore_index=True)

            # Update progress bar with description
            pbar.update(1)

            torch.cuda.empty_cache()

    pbar.close()
    return df_output

def plot_correlation(df:pd.DataFrame, param = "a", save=True):

    data = df.copy().query("params == @param")
    data.rename(columns={"prior": "x", "estimates": "y"}, inplace=True)
    g = sns.FacetGrid(data, row="fit_model", col="sim_model", height=2, aspect=1.2,sharey=False)
    g.map_dataframe(regplot_with_corr)
    g.tight_layout(pad=0)

    g.set_axis_labels(x_var="", y_var="")
    g.set_titles("")

    for i, fit_model in enumerate(data["fit_model"].unique()):
        g.axes[i, 0].set_ylabel(fit_model, fontsize=14)
    for j, sim_model in enumerate(data["sim_model"].unique()):
        g.axes[0, j].set_title(sim_model, fontsize=14)

    # Add these lines before returning g
    g.figure.text(0.5, -0.02, "Simulating model", ha='center', fontsize=18)
    g.figure.text(-0.05, 0.5, "Fitting model", va='center', rotation='vertical', fontsize=18)

    if save:
        g.savefig(SUPPLEMENT_FIGURES_DIR / f"figure_s10_parameter_mapping_{param}.svg", bbox_inches='tight')

    return g




best_estimates = {
    "DDM": {
        "a": 1.098917,
        "ndt": 0.257500,
        "v_c": 3.614667,
        "v_i": 2.548667
    },

    "DMC": {
        "a": 1.077000,
        "ndt": 0.267500,
        "v_c": 3.198750,
        "alpha": 2.786167,
        "eta": 147.111333,
        "tau": 82.741750
    },

    "SSP": {
        "a": 1.403333,
        "ndt": 0.226167,
        "p": 3.702750,
        "sd_a": 1.414917,
        "r_d": 19.731917
    },

    "DSTP": {
        "a": 1.716833,
        "ndt": 0.187083,
        "vta": 0.951083,
        "vfl": 0.916500,
        "vss": 3.842000,
        "vp2": 10.311083,
        "ass": 1.560833
    }
}




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


# ## a and ter



params_vary = np.linspace(1, 2, 20 + 1)
param_key = "a"
n_trial = 200

a_param_vary_map = cache_from_file(INTERMEDIATE_DIR / f'parameter_mapping_{param_key}.pkl')(gen_data_and_fitting)(models, best_estimates, param_key, params_vary, n_trial)




g = plot_correlation(a_param_vary_map, "a")




params_vary = np.linspace(0.15, 0.3, 15 + 1)
param_key = "ndt"
n_trial = 200

t_param_vary_map = cache_from_file(INTERMEDIATE_DIR / f'parameter_mapping_{param_key}.pkl')(gen_data_and_fitting)(models, best_estimates, param_key, params_vary, n_trial)




g = plot_correlation(t_param_vary_map, "ndt")




