#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np

from nsbi_module.NSBI_CDMs import NSBICDM, NSBICDMs
from nsbi_module.project_paths import CHECKPOINTS_DIR, INTERMEDIATE_DIR, ensure_output_directories
from nsbi_module.utils import FitStore, timer

# ## Load models and helper functions



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




@timer(label="Fitting all datasets and generating posterior predictive data")
def run_fit_and_pp(df_dict, CDMs_fit:NSBICDMs, save_name, sim_pp = False, pp_n_trial = 500, verbose=False):

    fit_store = FitStore(save_name)

    temp_dict = {}

    for key, df in df_dict.items():

        try:
            key_fitted = key + "_fitted"
            if fit_store.isexist(key_fitted):
                fitted_params_sum = fit_store[key_fitted]
            else:
                fitted_params = CDMs_fit.fit_data(df, return_summary=False, batchsize=32, show_progress=True)
                fit_store[key_fitted + "_trace"] = fitted_params
                fitted_params_sum = CDMs_fit.df_summary(fitted_params)
                fit_store[key_fitted] = fitted_params_sum

            temp_dict[key_fitted] = fitted_params_sum

            if sim_pp:
                key_predicted = key + "_predicted"
                if fit_store.isexist(key_predicted):
                    pp_data = fit_store[key_predicted]
                else:
                    pp_data = CDMs_fit.posterior_predictive(temp_dict[key_fitted], n_trial=pp_n_trial)
                    fit_store[key_predicted] = pp_data

                temp_dict[key_predicted] = pp_data

            if verbose:
                print(f"Finished {key}")
        except Exception as e:
            print(f"Error in {key}: {e}")

    fit_store.close_store()
    return temp_dict


# ## Load data



ensure_output_directories()
df_dict = {}
with pd.HDFStore(INTERMEDIATE_DIR / "datasets_cross_sectional.h5") as hdfstore:
    for key in hdfstore.keys():
        key = key[1:]
        df_dict[key] = hdfstore[key]
df_dict.keys()


# ### Fitting all datasets and generating posterior predictive data
#
# Let we fit all the datasets and store the fitted parameters and generate posterior predictive data.
#
# It will take around 1h to fitting the 21 datasets with 7 min for generating posterior predictive data.



fit_store_path = INTERMEDIATE_DIR / "model_fits.h5"
fit_and_pp_dict = run_fit_and_pp(df_dict, CDMs_fit, fit_store_path, sim_pp=False, verbose=True)




fit_and_pp_dict = run_fit_and_pp(df_dict, CDMs_fit, fit_store_path, sim_pp=True,verbose=True)




df_dict.update(fit_and_pp_dict)




df_dict.keys()


# ## Fitting retest datasets
#
# It will take around 40min to fitting the 21 datasets with 3 min for generating posterior predictive data.



df_dict = {}
with pd.HDFStore(INTERMEDIATE_DIR / "datasets_retest.h5") as hdfstore:
    for key in hdfstore.keys():
        dataset_name = key.lstrip("/")
        if dataset_name != "meta_data":
            df = hdfstore[key]
            # Group by session and store each session's data separately
            for session_id, session_df in df.groupby("session"):
                df_dict[f"{dataset_name}_s{int(session_id)}"] = session_df




fit_and_pp_dict = run_fit_and_pp(
    df_dict,
    CDMs_fit,
    save_name=INTERMEDIATE_DIR / "model_fits_retest.h5",
    sim_pp=False,
    verbose=True)




fit_and_pp_dict = run_fit_and_pp(
    df_dict,
    CDMs_fit,
    save_name=INTERMEDIATE_DIR / "model_fits_retest.h5",
    sim_pp=True,
    verbose=True)




fit_and_pp_dict.keys()







# ###
