#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sys

from nsbi_module.NSBI_CDMs import NSBICDM, NSBICDMs
from nsbi_module.utils import FitStore, timer, cache_from_file

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Load models and helper functions

# In[2]:


m_DDM = NSBICDM("DDM", checkpoint_path="../../checkpoints/DDM")
m_DMC = NSBICDM("DMC", checkpoint_path="../../checkpoints/DMC")
m_SSP = NSBICDM("SSP", checkpoint_path="../../checkpoints/SSP")
m_DSTP = NSBICDM("DSTP", checkpoint_path="../../checkpoints/DSTP")

models = {
    "DDM": m_DDM,
    "DMC": m_DMC,
    "SSP": m_SSP,
    "DSTP": m_DSTP
}

CDMs_fit = NSBICDMs(models)


# In[3]:


@timer(label="Fitting all datasets and generating posterior predictive data")
def run_fit_and_pp(df_dict, CDMs_fit:NSBICDMs, save_name = "22fitting_and_prediction.h5", sim_pp = False, pp_n_trial = 500, verbose=False):

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

# In[5]:


df_dict = {}
with pd.HDFStore("./21preprocessed_datasets.h5") as hdfstore:
    for key in hdfstore.keys():
        key = key[1:]
        df_dict[key] = hdfstore[key]
df_dict.keys()


# ## Fitting data

# ### Test Fitting precedure. 
# 
# It just cost about 20s to fit one dataset across four models for 16 subjects.

# In[5]:


@timer
def test_run():
    fitted_params = CDMs_fit.fit_data(df_dict["ulrich2015flanker"])
    return fitted_params

fitted_params = test_run()
fitted_params


# ### Test Generate predictive prediction. 
# 
# It cost about 10s to generate posterterior predictive datasets across four models for 5000 trials.

# In[6]:


@timer
def test_run(fitted_params):
    pp_data = CDMs_fit.posterior_predictive(fitted_params)
    return pp_data

pp_data = test_run(fitted_params)
pp_data


# ### Fitting all datasets and generating posterior predictive data
# 
# Let we fit all the datasets and store the fitted parameters and generate posterior predictive data.
# 
# It will take around 1h to fitting the 21 datasets with 7 min for generating posterior predictive data.

# In[6]:


fit_and_pp_dict = run_fit_and_pp(df_dict, CDMs_fit, sim_pp=False, verbose=True)


# In[7]:


# store_tmp = FitStore("22fitting_and_prediction.h5")
# for key in store_tmp.key_list_df.key:
#     if "predected" in key:
#         store_tmp.remove(key)
# store_tmp.close_store()
fit_and_pp_dict = run_fit_and_pp(df_dict, CDMs_fit, sim_pp=True,verbose=True)


# In[8]:


df_dict.update(fit_and_pp_dict)


# In[9]:


df_dict.keys()


# ## Fitting retest datasets
# 
# It will take around 40min to fitting the 21 datasets with 3 min for generating posterior predictive data.

# In[16]:


df_dict = {}
with pd.HDFStore("./21preprocessed_datasets_retest.h5") as hdfstore:
    for key in hdfstore.keys():
        dataset_name = key.lstrip("/")
        if dataset_name != "meta_data":
            df = hdfstore[key]
            # Group by session and store each session's data separately
            for session_id, session_df in df.groupby("session"):
                df_dict[f"{dataset_name}_s{int(session_id)}"] = session_df


# In[17]:


fit_and_pp_dict = run_fit_and_pp(
    df_dict, 
    CDMs_fit, 
    save_name="22fitting_and_prediction_retest.h5",
    sim_pp=False,
    verbose=True)


# In[18]:


fit_and_pp_dict = run_fit_and_pp(
    df_dict, 
    CDMs_fit, 
    save_name="22fitting_and_prediction_retest.h5",
    sim_pp=True,
    verbose=True)


# In[19]:


fit_and_pp_dict.keys()


# In[ ]:





# ### 
