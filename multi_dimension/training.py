#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import torch
import gpytorch
import nbimporter
from matplotlib import pyplot as plt


# In[ ]:


from data import generate_nd_data
from function import rosenbrock
from nd_model import ExactGPModel


# In[ ]:


def train_gp(X_train, y_train, X_test, y_test, dim: int, train_iter: int = 200, lr = 0.01, optimizer = torch.optim.Adam):
    """
    Trains the GP model and evaluates on the test set.
    
    The evaluation metrics include:
      - MSE: Mean Squared Error.
      - NMSE: Normalized MSE (MSE divided by the variance of y_test).
      - MNLP: Mean Negative Log Predictive likelihood.
      
    Returns:
      model, metrics  where metrics is a dict with keys 'MSE', 'NMSE', and 'MNLP'.
    """
    
    # Ensure target tensors are 1D
    y_train = y_train.squeeze(-1)
    y_test  = y_test.squeeze(-1)
    
    # Initialize model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X_train, y_train, likelihood, dim=dim)
    
    # Training setup
    optimizer = optimizer(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # Training loop
    model.train()
    likelihood.train()
    
    for i in range(train_iter):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        
    # Evaluation
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(X_test))
        mse = torch.mean((pred.mean - y_test) ** 2)
        nmse = mse / torch.var(y_test)
        # MNLP: Mean negative log predictive likelihood.
        mnlp = - pred.log_prob(y_test).mean()

    metrics = {
        'MSE': mse.item(),
        'NMSE': nmse.item(),
        'MNLP': mnlp.item()
    }
    return model, metrics


# In[ ]:


# get_ipython().system('jupyter nbconvert --to script training.ipynb')

