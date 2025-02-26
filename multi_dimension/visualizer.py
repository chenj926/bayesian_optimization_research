#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch, gpytorch
import matplotlib.pyplot as plt


# In[3]:

# normalize back the data when ploting
# Example of creating a Visualizer class to match your object-oriented design
class Visualizer:
    def visualize_1d(self, model, X_train, y_train):
        # Get predictions on a finer grid
        test_x = torch.linspace(-3, 3, 100).unsqueeze(1)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = model.likelihood(model(test_x))
            mean = predictions.mean
            lower, upper = predictions.confidence_region()
        
        # Plot training data
        plt.figure(figsize=(10, 6))
        plt.scatter(X_train.numpy(), y_train.numpy(), label='Training Data')
        
        # Plot prediction
        plt.plot(test_x.numpy(), mean.numpy(), 'r-', label='Mean')
        plt.fill_between(test_x.squeeze().numpy(), lower.numpy(), upper.numpy(), alpha=0.2, color='r', label='Confidence')
        
        plt.legend()
        plt.title('1D GP Regression')
        plt.show()
    
    def visualize_2d(self, model, X_train, y_train):
        # Create a 2D grid of points
        grid_size = 50
        grid_x = torch.linspace(-3, 3, grid_size)
        grid_y = torch.linspace(-3, 3, grid_size)
        grid_X, grid_Y = torch.meshgrid(grid_x, grid_y)
        grid_points = torch.stack([grid_X.flatten(), grid_Y.flatten()], dim=1)
        
        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = model.likelihood(model(grid_points))
            mean = predictions.mean.reshape(grid_size, grid_size)
        
        # Plot contour of predictions
        plt.figure(figsize=(10, 8))
        plt.contourf(grid_X.numpy(), grid_Y.numpy(), mean.numpy(), 50, cmap='viridis')
        plt.colorbar(label='Predicted Value')
        
        # Plot training points
        plt.scatter(X_train[:, 0].numpy(), X_train[:, 1].numpy(), c=y_train.numpy(), 
                   cmap='coolwarm', s=50, edgecolors='k', label='Training Data')
        
        # Add labels to each data point
        for i in range(X_train.shape[0]):
            plt.text(
                X_train[i, 0].item() + 0.05,  # Slight offset in x for clarity
                X_train[i, 1].item() + 0.05,  # Slight offset in y for clarity
                f"({X_train[i, 0].item():.2f}, {X_train[i, 1].item():.2f})",
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

