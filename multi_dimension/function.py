#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math
import torch
import gpytorch
from matplotlib import pyplot as plt


# In[ ]:


class FunctionUtils:
    # function
    @staticmethod
    def rosenbrock(x: torch.Tensor) -> torch.Tensor:
        """N-dimensional Rosenbrock function compatible with PyTorch tensors."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.size(1) == 1:  # 1D case (standard quadratic)
            return (x.squeeze() - 1)**2
        return torch.sum(100*(x[:, 1:] - x[:, :-1]**2)**2 + (x[:, :-1] - 1)**2, dim=1)
    
    @staticmethod
    def ackley(x: torch.Tensor, a=20, b=0.2, c=2 * math.pi) -> torch.Tensor:
        """N-dimensional Ackley function compatible with PyTorch tensors."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        dim = x.size(1)
        sum_sq_term = -a * torch.exp(-b * torch.sqrt(torch.sum(x**2, dim=1) / dim))
        cos_term = -torch.exp(torch.sum(torch.cos(c * x), dim=1) / dim)
        return sum_sq_term + cos_term + a + torch.exp(torch.tensor(1.0))
    
    @staticmethod
    def sphere(x: torch.Tensor) -> torch.Tensor:
        """N-dimensional Sphere function compatible with PyTorch tensors."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return torch.sum(x**2, dim=1)
    
    @staticmethod
    def rastrigin(x: torch.Tensor, A=10) -> torch.Tensor:
        """N-dimensional Rastrigin function compatible with PyTorch tensors."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        dim = x.size(1)
        return A * dim + torch.sum(x**2 - A * torch.cos(2 * math.pi * x), dim=1)


# In[ ]:


# get_ipython().system('jupyter nbconvert --to script function.ipynb')

