import math
import torch
import gpytorch
import nbimporter
from matplotlib import pyplot as plt
from data import generate_nd_data
# from function import FunctionUtils
from nd_model import ExactGPModel

class GPTrainer:
  def __init__(self, likelihood_class, optimizer_handler_class): # Pass classes
        self.likelihood_class = likelihood_class
        self.optimizer_handler_class = optimizer_handler_class
    
  #  lr = 0.01, optimizer = torch.optim.Adam()
  def train(self, X_train_norm, y_train_norm, X_test_norm, y_test_norm, 
            dim: int, y_norm_params: dict, train_iter: int = 200, lr=0.01):
      """
      Trains the GP model and evaluates on the test set.
      y_train_norm, y_test_norm are normalized versions.
      y_norm_params contains 'y_mean' and 'y_std' for unscaling.
      Metrics (MSE, NMSE) are reported in the original scale of Y.
      MNLP is reported on the normalized scale (as the likelihood operates on it).
      """
      # Training setup
      likelihood = self.likelihood_class() # Instantiate likelihood
      model = ExactGPModel(X_train_norm, y_train_norm, likelihood, dim)
      optimizer_handler = self.optimizer_handler_class(model, lr)  # Instantiate optimizer handler
      
      # Training loop
      model.train()
      model.likelihood.train()
      mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
      
      for _ in range(train_iter):
        def closure():
                optimizer_handler.optimizer.zero_grad()
                output = model(X_train_norm) # Use normalized X_train
                loss = -mll(output, y_train_norm) # Use normalized y_train
                loss.backward()
                return loss
        optimizer_handler.optimizer.step(closure)
          
      # Evaluation
      model.eval()
      model.likelihood.eval()

      y_mean = y_norm_params['y_mean']
      y_std = y_norm_params['y_std']

      with torch.no_grad(), gpytorch.settings.fast_pred_var():
          # Predictions are on the normalized Y scale
          pred_likelihood_output = model.likelihood(model(X_test_norm))
          pred_mean_norm = pred_likelihood_output.mean

          # Un-normalize predictions and test data for metrics on original scale
          pred_mean_orig = pred_mean_norm * y_std + y_mean
          y_test_orig = y_test_norm * y_std + y_mean

          mse = torch.mean((pred_mean_orig - y_test_orig) ** 2)
          
          # Calculate variance of original y_test for NMSE
          y_test_orig_var = torch.var(y_test_orig)
          if y_test_orig_var.item() == 0: # Avoid division by zero if y_test_orig is constant
              nmse = torch.tensor(float('inf')) if mse.item() > 1e-6 else torch.tensor(0.0)
          else:
              nmse = mse / y_test_orig_var

          # MNLP: Mean negative log predictive likelihood.
          # This should be calculated using the normalized y_test_norm, 
          # as the likelihood is parameterized on the normalized scale.
          mnlp = -pred_likelihood_output.log_prob(y_test_norm).mean()

      metrics = {
          'MSE': mse.item(),
          'NMSE': nmse.item(),
          'MNLP': mnlp.item()
      }
      return model, metrics
