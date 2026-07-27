#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

from NSBI_CDMs import NSBICDM
from plotting import plot_rt_dists
import time

%load_ext autoreload
%autoreload 2

# ## Generative model (Simulator)

# In[2]:


CDM_model = NSBICDM(model="SSP_fixed_ratio")

# ### prior check
# 
# adapted from flexDDM

# In[3]:


CDM_model.plot_prior_check()

# ### prior predictive check

# In[4]:


params = {
    "a": 0.5 * 2,
    "ndt": 0.30,
    "p": 4,
    "rd_sda_ratio": 5,
}
sim_data = CDM_model.simulate_data(n_trial = 10000, params=[params])
sim_data.groupby(["congruency","accuracy"])["rt"].describe()

# In[5]:


plot_rt_dists(sim_data, flip=True)

# In[6]:


sim_data10 = CDM_model.simulate_data(10, n_trial=500)

plot_rt_dists(sim_data10)

# ## Train

# In[7]:


# print cost time
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# In[8]:


history = CDM_model.run(
    epochs=500,
    batch_size=16,
    num_batches_per_epoch=200, 
    verbose=0,
    # save=False, 
    keep_optimizer=True
)

# In[9]:


# print cost time
import time
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# ### check

# In[10]:


CDM_model.plot_trained_result()
