#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

from NSBI_CDMs import NSBICDM
from plotting import plot_rt_dists
import time

# %load_ext autoreload
# %autoreload 2

# ## Generative model (Simulator)

# In[5]:


CDM_model = NSBICDM("DDM")

# ### prior check

# In[ ]:


CDM_model.plot_prior_check()

# ### prior predictive check

# In[6]:


params = {
    "a": 1,    
    "ndt": 0.3,    
    "v_c": 1,    
    "v_i": 0.5  
}
sim_data = CDM_model.simulate_data(n_trial = 10000, params=[params])
sim_data.groupby(["congruency","accuracy"])["rt"].describe()

# In[7]:


plot_rt_dists(sim_data, flip=True)

# In[ ]:


sim_data10 = CDM_model.simulate_data(10, n_trial=500)

plot_rt_dists(sim_data10)

# ## Train

# In[ ]:


# print cost time
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# In[ ]:


history = CDM_model.run(
    epochs=250,
    batch_size=64,
    num_batches_per_epoch=250, 
    verbose=0,
    # save=False, 
    keep_optimizer=True
)

# In[ ]:


# print cost time
import time
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# ### check

# In[3]:


CDM_model.plot_trained_result()
