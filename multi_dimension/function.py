#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math
import torch
import gpytorch
from matplotlib import pyplot as plt


# In[3]:


# function
def rosenbrock(x: torch.Tensor) -> torch.Tensor:
    """N-dimensional Rosenbrock function compatible with PyTorch tensors."""
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.size(1) == 1:  # 1D case (standard quadratic)
        return (x.squeeze() - 1)**2
    return torch.sum(100*(x[:, 1:] - x[:, :-1]**2)**2 + (x[:, :-1] - 1)**2, dim=1)


# In[ ]:


# get_ipython().system('jupyter nbconvert --to script function.ipynb')

