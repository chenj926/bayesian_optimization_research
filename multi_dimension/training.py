#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import torch
import gpytorch
import nbimporter
from matplotlib import pyplot as plt


# In[4]:


from data import generate_nd_data
# from function import FunctionUtils
from nd_model import ExactGPModel


# In[5]:


class GPTrainer:
  def __init__(self, likelihood, optimizer):
        self.likelihood = likelihood
        self.optimizer = optimizer
    
  #  lr = 0.01, optimizer = torch.optim.Adam()
  def train(self, X_train, y_train, X_test, y_test, dim: int, train_iter: int = 200, lr=0.01):
      """
      Trains the GP model and evaluates on the test set.
      
      The evaluation metrics include:
        - MSE: Mean Squared Error.
        - NMSE: Normalized MSE (MSE divided by the variance of y_test).
        - MNLP: Mean Negative Log Predictive likelihood.
        
      Returns:
        model, metrics  where metrics is a dict with keys 'MSE', 'NMSE', and 'MNLP'.
      """
      # Training setup
      model = ExactGPModel(X_train, y_train, self.likelihood(), dim)
      optimizer_handler = self.optimizer(model, lr)
      
      # Training loop
      model.train()
      model.likelihood.train()
      mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
      
      for _ in range(train_iter):
        def closure():
                optimizer_handler.optimizer.zero_grad()
                output = model(X_train)
                loss = -mll(output, y_train)
                loss.backward()
                return loss
        optimizer_handler.optimizer.step(closure)
          
      # Evaluation
      model.eval()
      model.likelihood.eval()
      with torch.no_grad(), gpytorch.settings.fast_pred_var():
          pred = model.likelihood(model(X_test))
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

