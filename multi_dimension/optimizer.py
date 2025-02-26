#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch


# In[3]:


# Update the OptimizerHandler class to use LBFGS instead of Adam
class OptimizerHandler:
    def __init__(self, model, lr=0.01):
        # Replace Adam with LBFGS optimizer
        self.optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, line_search_fn="strong_wolfe")
        self.model = model
        self.mll = None  # Will be set in the train method

    def step(self):
        """Step function that doesn't require closure"""
        # Define closure function internally to avoid exposing it to the user
        def closure():
            self.optimizer.zero_grad()
            output = self.model(self.model.train_inputs[0])
            loss = -self.mll(output, self.model.train_targets)
            loss.backward()
            return loss
            
        # Take the optimization step with closure
        return self.optimizer.step(closure)


# In[ ]:


# get_ipython().system('jupyter nbconvert --to script optimizer.ipynb')

