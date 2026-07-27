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


CDM_model = NSBICDM(model="DMC_fixed_alpha")

# ### prior check
# 
# adapted from flexDDM

# In[3]:


CDM_model.plot_prior_check()

# ### prior predictive check

# In[4]:


params = {
    "a": 0.6*2,     # sqrt(dt)/4 times
    "ndt": 0.3,     # 1/1000 times
    "v_c": 2,     # sqrt(dt)*1000/4 times
    "eta": 158,    # sqrt(dt)*1000/4 times 
    "tau": 30,
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
    # num_batches_per_epoch=10, 
    verbose=0,keep_optimizer=True)

# In[9]:


# print cost time
import time
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# ### check

# In[10]:


CDM_model.plot_trained_result()
