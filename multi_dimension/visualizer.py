#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch, gpytorch
import matplotlib.pyplot as plt


# In[3]:

# normalize back the data when ploting
# Example of creating a Visualizer class to match your object-oriented design
class Visualizer:
    def visualize_1d(self, model, X_train_norm, y_train, X_min, X_max):
        
        # 1) Convert training inputs back to original domain
        X_range = X_max - X_min
        X_train_orig = X_train_norm * X_range + X_min  # shape (N, dim=1)
        
        # 2) Create a test grid in original domain
        test_x_orig = torch.linspace(-3, 3, 100).unsqueeze(1)
        
         # 3) Normalize the test grid
        test_x_norm = (test_x_orig - X_min) / X_range
        
        # 4) Predict using the *normalized* test inputs
        model.eval()
        model.likelihood.eval()
        
        # # Get predictions on a finer grid
        # test_x = torch.linspace(-3, 3, 100).unsqueeze(1)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = model.likelihood(model(test_x_norm))
            mean = predictions.mean
            lower, upper = predictions.confidence_region()
        
        # Plot training data
        plt.figure(figsize=(10, 6))
        plt.scatter(X_train_orig.numpy(), y_train.numpy(), label='Training Data')
        
        # Plot prediction
        plt.plot(test_x_orig.numpy(), mean.numpy(), 'r-', label='Mean')
        plt.fill_between(test_x_orig.squeeze().numpy(), lower.numpy(), upper.numpy(), alpha=0.2, color='r', label='Confidence')
        
        plt.legend()
        plt.title('1D GP Regression')
        plt.show()
    
    def visualize_2d(self, model, X_train_norm, y_train, X_min, X_max):
        model.eval()
        model.likelihood.eval()
        
        # 1) Un‐normalize your training inputs for plotting
        X_range = X_max - X_min
        X_train_orig = X_train_norm * X_range + X_min  # shape (N, 2)

        # 2) Create a 2D mesh in the *original* domain, e.g. [-3, 3]^2
        grid_size = 50
        grid_x = torch.linspace(-3, 3, grid_size)
        grid_y = torch.linspace(-3, 3, grid_size)
        grid_X, grid_Y = torch.meshgrid(grid_x, grid_y)  # shape [50, 50]
        
        # Flatten into (2500, 2) for the model
        grid_points_orig = torch.stack([grid_X.flatten(), grid_Y.flatten()], dim=1)

        # 3) Normalize that grid for GP prediction
        grid_points_norm = (grid_points_orig - X_min) / X_range
            
        # 4) Make predictions in normalized space
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = model.likelihood(model(grid_points_norm))
            mean = predictions.mean.reshape(grid_size, grid_size)
        
        # 5) 5) Plot contour of predictions in original domain
        plt.figure(figsize=(10, 8))
        plt.contourf(grid_X.numpy(), grid_Y.numpy(), mean.numpy(), 50, cmap='viridis')
        plt.colorbar(label='Predicted Value')
        
        # 6) Plot training points (also in original domain)
        plt.scatter(
            X_train_orig[:, 0].numpy(), 
            X_train_orig[:, 1].numpy(), 
            c=y_train.numpy(), cmap='coolwarm', s=50, edgecolors='k', 
            label='Training Data'
        )
        
        # Add labels to each data point
        for i in range(X_train_orig.shape[0]):
            plt.text(
                X_train_orig[i, 0].item() + 0.05,  # Slight offset in x for clarity
                X_train_orig[i, 1].item() + 0.05,  # Slight offset in y for clarity
                f"({X_train_orig[i, 0].item():.2f}, {X_train_orig[i, 1].item():.2f})",
                fontsize=9,
                color='white',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3')
            )
        
        plt.title('2D GP Regression')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.show()


# In[ ]:


# get_ipython().system('jupyter nbconvert --to script visualizer.ipynb')

