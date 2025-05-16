#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch


# In[3]:


# Update the OptimizerHandler class to use LBFGS instead of Adam
class OptimizerHandler:
    def __init__(self, model, optimizer_name="LBFGS", lr=0.01):
        self.model = model
        self.optimizer_name = optimizer_name.upper()
        self.lr = lr
        self.mll = None  # Will be set in the train method

        if self.optimizer_name == "LBFGS":
            self.optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, line_search_fn="strong_wolfe")
        elif self.optimizer_name == "ADAM":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer currently: {optimizer_name}. Currently supported: LBFGS, Adam")

    def step(self, closure=None):
        """
        Performs a single optimization step.
        A closure is required for LBFGS and similar optimizers.
        Adam and others might not strictly need it for the step call but
        the training loop often uses it to re-evaluate loss.
        """
        if self.optimizer_name == "LBFGS":
            if closure is None:
                raise ValueError("LBFGS optimizer requires a closure function for the step.")
            return self.optimizer.step(closure)
        
        elif self.optimizer_name == "ADAM":
            if closure is not None: # Adam step can be called after loss.backward()
                loss = closure() # Ensure gradients are computed
            # For Adam, zero_grad is typically called before the closure,
            # and step is called after loss.backward()
            self.optimizer.step()
            # The closure in the training loop will return the loss,
            # but Adam's step itself doesn't return the loss value.
            # We'll return None or the re-evaluated loss if the training loop needs it.
            # For simplicity, the training loop will handle loss tracking.
            return None # Or re-evaluate loss if necessary, but training loop handles it
        else:
            raise ValueError(f"Optimization step not defined for optimizer: {self.optimizer_name}")


# In[ ]:


# get_ipython().system('jupyter nbconvert --to script optimizer.ipynb')

