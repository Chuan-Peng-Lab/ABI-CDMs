#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import sys
from utils_pydmc import Ob
from utils import timer, FitStore, cache_from_file
from model_metrics import ModelMetricEvaluator
from default_settings import PARAMS_KEY_NAME_MAPPING

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## helper functions

# In[ ]:


from sklearn.linear_model import LinearRegression, LogisticRegression
from tqdm.notebook import tqdm
import warnings

# ── Shared analysis utilities ──
from analysis_utils import (
    rename_columns_with_model, concat_dfs_by_subj,
    calculate_indices, compute_model_prediction_indices,
    merge_params_and_indices, get_col_names,
)

def align_and_concatenate_props(props_1, props_2):
    """
    Aligns two property dictionaries component-by-component, then concatenates them.

    This ensures that:
    1. CDF_congruent is aligned with CDF_congruent (e.g., length 4 vs 5 -> truncate to 4)
    2. CAF_congruent is aligned with CAF_congruent (e.g., length 3 vs 5 -> truncate to 3)

    Args:
        props_1 (dict): Dictionary containing distribution properties (e.g., Empirical).
        props_2 (dict): Dictionary containing distribution properties (e.g., Predicted).

    Returns:
        tuple: (aligned_array_1, aligned_array_2) Both are 1D numpy arrays.
    """

    # Define the strict order of concatenation used in the cost function
    keys_order = [
        'cdf_props_congruent', 
        'cdf_props_incongruent',
        'caf_props_congruent', 
        'caf_props_incongruent'
    ]

    aligned_parts_1 = []
    aligned_parts_2 = []

    for key in keys_order:
        # Extract the arrays for this specific component
        arr1 = np.array(props_1[key])
        arr2 = np.array(props_2[key])

        # Find the minimum length for THIS component specificially
        # e.g., CDF might be 4, while CAF might be 3
        common_len = min(len(arr1), len(arr2))

        # Truncate both to the common length and flatten if necessary
        aligned_parts_1.append(arr1[:common_len].reshape(-1))
        aligned_parts_2.append(arr2[:common_len].reshape(-1))

    # Concatenate all parts to create the final arrays for cost calculation
    final_arr_1 = np.concatenate(aligned_parts_1)
    final_arr_2 = np.concatenate(aligned_parts_2)

    return final_arr_1, final_arr_2

# In[11]:


task_indices = indices_dict["clayson2025flanker"]
obs_data_i = task_indices["obs"]["trial_data"]
pp_data_i = task_indices["ppd"]["DDM"]["trial_data"]
# ntrial_by_subj = obs_trial_data.groupby('subject_id')['subject_id'].count()


# In[14]:


obs_data_i_sub_j = obs_data_i[obs_data_i['subject_id'] == 0]
pp_data_i_sub_j = pp_data_i[pp_data_i['subject_id'] == 0]


# In[ ]:


from model_metrics import ModelMetricEvaluator

map_nparams_of_models = {
    "DDM": 4,
    "DMC": 5,
    "SSP": 6,
    "DSTP": 7
}
fit_model = "DDM"

n_trial = obs_data_i_sub_j.shape[0]
preprocess_config = {
    "n_bins_caf": 5,
    "n_bins_cdf": 10,
    "n_bins_caf_props": 5,
    "n_bins_cdf_props": 10
}
evaluator = ModelMetricEvaluator(
    empirical_df=obs_data_i_sub_j,
    predicted_df=pp_data_i_sub_j,
    config=preprocess_config,
)
rmse_i = evaluator.compute_rmse()
abic_i = evaluator.compute_abic_proportions(
    n_param=map_nparams_of_models[fit_model],
    n_trial=n_trial
)


# ## Load data

# In[6]:


store_datasets_path = '21preprocessed_datasets.h5'
store_fits_path = '22fitting_and_prediction.h5'


# ## Merge fitting parameters across models

# ### Calculating behaviour indices and extract fitted parameters
# 
# It will cost 44s to calculate behaviour indices. If calculating the KL divergence, it will cost another 6hour. 
# 
# The output indices_dict has structure as follows:
# ```
# {'clayson2025flanker': {'obs': {'trial_data': 'DataFrame',
#    'subject_indices': 'DataFrame',
#    'subject_caf': 'DataFrame',
#    'subject_summary': 'DataFrame',
#    'subject_delta': 'DataFrame'},
#   'ppd': {'DDM': {'trial_data': 'DataFrame',
#     'subject_indices': 'DataFrame',
#     'subject_caf': 'DataFrame',
#     'subject_summary': 'DataFrame',
#     'subject_delta': 'DataFrame'},
#    'DMC': {'trial_data': 'DataFrame',
#     'subject_indices': 'DataFrame',
#     'subject_caf': 'DataFrame',
#     'subject_summary': 'DataFrame',
#     'subject_delta': 'DataFrame'},
#    'SSP': {'trial_data': 'DataFrame',
#     'subject_indices': 'DataFrame',
#     'subject_caf': 'DataFrame',
#     'subject_summary': 'DataFrame',
#     'subject_delta': 'DataFrame'},
#    'DSTP': {'trial_data': 'DataFrame',
#     'subject_indices': 'DataFrame',
#     'subject_caf': 'DataFrame',
#     'subject_summary': 'DataFrame',
#     'subject_delta': 'DataFrame'}}},
#  'clayson2025stroop':...
# }
# ```

# In[7]:


@timer
def get_indices_dict(store_datasets_path, store_fits_path, n_jobs=16):
    """
    Pre-load all necessary data into memory, then process keys in parallel.
    Avoids passing HDFStore across processes.
    """

    import pandas as pd
    from joblib import Parallel, delayed

    # ———————— Step 1: Pre-load all datasets and fits ————————
    datasets_dict = {}
    fits_predicted_dict = {}
    fits_fitted_dict = {}

    # Load datasets (lightweight read — assume keys are manageable in memory)
    with pd.HDFStore(store_datasets_path, mode='r') as store_datasets:
        for key in store_datasets.keys():
            key_name = key[1:]  # strip leading '/'
            if "meta_data" in key_name:
                continue
            datasets_dict[key_name] = store_datasets[key]

    # Load fits using FitStore (assumes it supports context manager or `with`)
    # If FitStore doesn't support `with`, adapt as needed.
    store_fits = FitStore(store_fits_path)
    fits_keys = store_fits.key_list_df["key"].str.split('_', expand=True).loc[:,0].values
    try:
        for key_name in fits_keys:
            # Predicted PPDs: dict of {model_i: df}
            fits_predicted_dict[key_name] = store_fits[f"{key_name}_predicted"]
            # Fitted params: typically list/dict of subject-wise fitted DataFrames
            fits_fitted_dict[key_name] = store_fits[f"{key_name}_fitted"]
    finally:
        store_fits.close_store()

    # ———————— Step 2: Define pure-data processing function ————————
    def process_key_in_memory(key_name, obs_df, predicted_dict, fitted_list):
        """Pure function: no I/O, only computation."""
        try:
            # Compute indices
            indices_result = {
                "obs": calculate_indices(obs_df),
                "ppd": {
                    model_i: calculate_indices(pp_data_i)
                    for model_i, pp_data_i in predicted_dict.items()
                }
            }
            # Concat fitted dfs
            fitted_df_result = concat_dfs_by_subj(fitted_list)
            return key_name, indices_result, fitted_df_result
        except Exception as e:
            print(f"Error processing key '{key_name}': {e}")
            return key_name, None, None

    # ———————— Step 3: Parallel processing on in-memory data ————————
    tasks = [
        (key_name, datasets_dict[key_name], fits_predicted_dict[key_name], fits_fitted_dict[key_name])
        for key_name in fits_predicted_dict.keys()
    ]

    try:
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_key_in_memory)(key_name, obs_df, pred_dict, fitted_list)
            for key_name, obs_df, pred_dict, fitted_list in tasks
        )
    except Exception as e:
        print(f"Parallel execution failed: {e}")
        raise

    # ———————— Step 4: Assemble results ————————
    indices_dict = {}
    fitted_df_dict = {}
    for key_name, idx_res, fit_res in results:
        if idx_res is not None and fit_res is not None:
            indices_dict[key_name] = idx_res
            fitted_df_dict[key_name] = fit_res
        else:
            print(f"Skipping key '{key_name}' due to errors.")

    return indices_dict, fitted_df_dict

indices_dict, fitted_df_dict = get_indices_dict(store_datasets_path,store_fits_path)


# ### Merge parameters and behaviour indices

# In[19]:



df = merge_params_and_indices(fitted_df_dict,indices_dict)
df = get_col_names(df)
df.head()


# In[20]:


df.to_csv("23subj_indices_across_models_and_tasks.csv", index=False)


# ### Calculating model prediction indices

# In[27]:


# @cache_from_file("23model_prediction_indices_cache.pkl")
def compute_model_prediction_indices_with_cache(indices_dict):
    map_nparams_of_models = {
        "DDM": 4,
        "DMC": 5,
        "SSP": 6,
        "DSTP": 7
    }
    model_prediction_indices = compute_model_prediction_indices(
        indices_dict,
        map_nparams_of_models,
        parallel = True,
        n_jobs = 32
        )

    return model_prediction_indices

model_prediction_indices = compute_model_prediction_indices_with_cache(indices_dict)


# In[38]:


model_prediction_indices = get_col_names(model_prediction_indices)
model_prediction_indices.head()


# In[39]:


model_prediction_indices.to_csv("23model_prediction_indices.csv", index=False)


# ## Reliability datasets preprocessing
# 
# It will cost 2 mins to calculate behaviour indices. 

# In[40]:


store_datasets_path = '21preprocessed_datasets_retest.h5'
store_fits_path = '22fitting_and_prediction_retest.h5'


# In[41]:


@timer
# @cache_from_file("23indices_dict_retest_cache.pkl")
def get_indices_dict(store_datasets_path, store_fits_path):
    """
    Get the indices dictionary and fitted DataFrame dictionary from the HDF5 store.

    Args:
        store_datasets_path (str): Path to the HDF5 store for datasets.
        store_fits_path (str): Path to the HDF5 store for fits.

    Returns:
        tuple: A tuple containing the indices dictionary and fitted DataFrame dictionary.
    """
    indices_dict = {}
    fitted_df_dict = {}
    store_fits = FitStore(store_fits_path)
    store_datasets = pd.HDFStore(store_datasets_path)

    for key in store_datasets.keys():

        key = key.lstrip("/")
        if "meta_data" not in key: 

            df = store_datasets[key]
            # Group by session and store each session's data separately
            for session_id, session_df in df.groupby("session"):
                key_with_session = f"{key}_s{int(session_id)}"
                indices_dict[key_with_session] = {}
                indices_dict[key_with_session]["obs"] = calculate_indices(session_df)
                indices_dict[key_with_session]["ppd"] = {model_i:calculate_indices(pp_data_i) for model_i, pp_data_i in store_fits[f"{key_with_session}_predicted"].items()}

                fitted_df_dict[key_with_session] = concat_dfs_by_subj(store_fits[f"{key_with_session}_fitted"])


    store_datasets.close()
    store_fits.close_store()

    return indices_dict, fitted_df_dict

indices_dict, fitted_df_dict = get_indices_dict(store_datasets_path, store_fits_path)


# ### Extract parameters and behaviour indices

# In[42]:


def get_col_names(df):
    tmp_df = df.copy()
    tmp_df[['task_id', 'session_id']] = tmp_df['task_id'].str.split('_', expand=True)
    tmp_df[['author_year', 'task_name']] = tmp_df['task_id'].str.extract(r'([a-z]+[0-9]{4})([a-z]+)')

    return tmp_df


# In[43]:



subj_indices_across_models_and_tasks = merge_params_and_indices(fitted_df_dict,indices_dict)
subj_indices_across_models_and_tasks = get_col_names(subj_indices_across_models_and_tasks)
subj_indices_across_models_and_tasks.head()


# In[44]:


subj_indices_across_models_and_tasks.to_csv("23subj_indices_across_models_and_tasks_retest.csv", index=False)


# ### Calculating model prediction indices
# 
# It will cost 30s to calculate model prediction indices. If calculating the KL divergence, it will cost another 12 min. 

# In[45]:


map_nparams_of_models = {
    "DDM": 4,
    "DMC": 5,
    "SSP": 6,
    "DSTP": 7
}
model_prediction_indices = compute_model_prediction_indices(
    indices_dict,
    # {kk:indices_dict[kk] for kk in list(indices_dict.keys())[:2]},
    map_nparams_of_models,
    parallel = True,
    n_jobs = 32,
)


# In[46]:


model_prediction_indices = get_col_names(model_prediction_indices)
model_prediction_indices.head()


# In[47]:


model_prediction_indices.to_csv("23model_prediction_indices_retest.csv", index=False)


# In[ ]:





# ### 
