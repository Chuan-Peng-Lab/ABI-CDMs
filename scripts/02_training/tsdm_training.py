#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys

import simulators as simulators
import default_settings as default_settings
from NSBI_CDMs import NSBICDM
from plotting import plot_rt_dists
import time

%load_ext autoreload
%autoreload 2

# ## TSDM (Two-Stage Dual-Mechanism) Generative Model
# 
# The TSDM model decomposes information processing into a **perceptual stage**
# and a **decision stage**, with attention dynamically shifting between
# two processing channels:
#   - **Controlled channel**: task-relevant (target) attribute processing
#   - **Automatic channel**: task-irrelevant (distractor) attribute processing
# 
# Eight free parameters: wc, b, mu_r, sigma_r, r_pc, r_pa, kp, kd.
# Total attention resources are fixed at 2.

# In[2]:


from tsdm_loader import (
    tsdm_experiment_simulator,
    TSDM_CONFIG,
    register as register_tsdm,
)

# Register the model into the global NSBI registries.
register_tsdm()

# ── Model config ──────────────────────────────────────────────────────────
tsdm_config = TSDM_CONFIG

# Register the experiment-level simulator.
simulators.TRIAL_SIMULATOR["tsdm"] = tsdm_experiment_simulator

# Register the model configuration.
default_settings.MODEL_CONFIG["tsdm"] = tsdm_config

# Refresh parameter-name mappings.
default_settings.PARAMS_KEY_NAME_MAPPING = default_settings.get_param_mappings(
    default_settings.MODEL_CONFIG
)

# Initialize the NSBI model.
tsdm_model = NSBICDM("tsdm")

# ### Prior predictive check — single subject

# In[3]:


# Test parameters for a single subject (mid-range plausible values).
test_params = {
    "wc": 1.0,       # balanced initial attention allocation
    "b": 60.0,       # moderate decision boundary (ms)
    "mu_r": 300.0,   # mean non-decision time 300 ms
    "sigma_r": 30.0, # SD of non-decision time 30 ms
    "r_pc": 3.0,     # controlled channel base rate
    "r_pa": 2.5,     # automatic channel base rate
    "kp": 3.0,       # perceptual attention shift rate
    "kd": 4.0,       # incongruent decision shift rate
}

sim_data = tsdm_model.simulate_data(
    n_trial=500,
    params=[test_params],
)

sim_data.head()

# In[4]:


sim_data.groupby(["congruency", "accuracy"])["rt"].describe()

# In[5]:


# plot_rt_dists(sim_data, flip=True)

# ### Prior predictive check — multiple subjects (10 random draws from prior)

# In[6]:


# sim_data10 = tsdm_model.simulate_data(10, n_trial=500)
# plot_rt_dists(sim_data10)

# ## Train

# In[7]:


# Print start time
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# In[ ]:


history = tsdm_model.run(
    epochs=250,
    batch_size=32,
    num_batches_per_epoch=200,
    keep_optimizer=True
)

# In[9]:


# Print end time
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# ### Training result — diagnostics

# In[10]:


tsdm_model.plot_trained_result()
