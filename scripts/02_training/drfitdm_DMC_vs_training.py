#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys

import nsbi_module.simulators as simulators
import nsbi_module.default_settings as default_settings
from nsbi_module.NSBI_CDMs import NSBICDM
from nsbi_module.plotting import plot_rt_dists
import time

%load_ext autoreload
%autoreload 2

# ## Generative model (Simulator)
# 
# DMC model with `var_start = TRUE` (starting-point variability via `alpha`).
# Seven free parameters: muc, b, non_dec, sd_non_dec, tau, A, alpha.

# In[2]:


from nsbi_module.dmc_vs_loader import (
    driftdm_dmc_vs_experiment_simulator,
    DRIFTDMC_DMC_VS_CONFIG,
    register as register_vs,
)

# Register the model into the global NSBI registries.
register_vs()

# ── Model config ──────────────────────────────────────────────────────────

driftdm_dmc_vs_config = DRIFTDMC_DMC_VS_CONFIG

# Register the experiment-level simulator.
simulators.TRIAL_SIMULATOR["driftdm_dmc_vs"] = driftdm_dmc_vs_experiment_simulator

# Register the model configuration.
default_settings.MODEL_CONFIG["driftdm_dmc_vs"] = driftdm_dmc_vs_config

# Refresh parameter-name mappings.
default_settings.PARAMS_KEY_NAME_MAPPING = default_settings.get_param_mappings(
    default_settings.MODEL_CONFIG
)

# Initialize the NSBI model.
driftdm_model = NSBICDM("driftdm_dmc_vs")

# ### Prior predictive check

# In[3]:


# Test parameters for a single subject.
test_params = {
    "muc": 4.0,
    "b": 0.6,
    "non_dec": 0.3,
    "sd_non_dec": 0.02,
    "tau": 0.04,
    "A": 0.1,
    "alpha": 4.0,
}

sim_data = driftdm_model.simulate_data(
    n_trial=500,
    params=[test_params],
)

sim_data.head()

# In[4]:


sim_data.groupby(["congruency", "accuracy"])["rt"].describe()

# In[5]:


plot_rt_dists(sim_data, flip=True)

# In[6]:


sim_data10 = driftdm_model.simulate_data(10, n_trial=500)

plot_rt_dists(sim_data10)

# ## Train

# In[7]:


# Print start time
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# In[8]:


history = driftdm_model.run(
    epochs=250,
    batch_size=32,
    num_batches_per_epoch=200,
    keep_optimizer=True
)

# In[9]:


# Print end time
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# ### Training result check

# In[10]:


driftdm_model.plot_trained_result()
