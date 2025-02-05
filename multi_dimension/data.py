import math
import torch
import gpytorch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from function import rosenbrock

def generate_nd_data(dim: int, n_samples: int = 1000, noise_level: float = 1e-6, test_size: float = 0.2, random_state: int = 42):
    """
    Generates n-dimensional data for regression.
    
    Sampling Strategy:
    - For `dim <= 3`: Uses a structured grid-based approach.
    - For `dim > 3`: Uses random uniform sampling.

    Args:
        dim (int): Number of dimensions.
        n_samples (int, optional): Total number of samples. Defaults to 1000.
        noise_level (float, optional): Standard deviation of added Gaussian noise. Defaults to 1.0.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test) - Training and test sets.
    """
    if dim <= 3:  # Low-dimensional case: Use grid-based sampling
        # Compute number of points per dimension to keep the total count close to n_samples
        n_per_dim = round(n_samples ** (1/dim))  
        
        # Generate evenly spaced values in each dimension within the range [-2,2]
        axes = [torch.linspace(-2, 2, n_per_dim) for _ in range(dim)]
        
        # Create a meshgrid of all possible combinations of values across dimensions
        X = torch.stack(torch.meshgrid(*axes)).view(dim, -1).T  # Reshape into (num_samples, dim)
    
    else:  # High-dimensional case: Use random sampling
        # Generate `n_samples` points where each dimension is sampled uniformly from [-2,2]
        X = torch.rand(n_samples, dim) * 4 - 2  

    # Compute the target variable `y` using the Rosenbrock function
    y = rosenbrock(X)  
    
    # Add Gaussian noise to the target variable to simulate real-world variability
    y += torch.randn(X.size(0)) * noise_level  

    # Split the dataset into training (80%) and test (20%) sets
    return train_test_split(X, y, test_size=test_size, random_state=random_state)