import math
import torch
import gpytorch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from function import FunctionUtils

def generate_nd_data(dim: int, n_samples: int = 10, noise_level: float = 1e-6, test_size: float = 0.2, random_state: int = 42):
    """
    Generates n-dimensional data for regression using random uniform sampling
    for all dimensions to avoid too regular grid spacing.
    
    Args:
        dim (int): Number of dimensions.
        n_samples (int, optional): Total number of samples. Defaults to 1000.
        noise_level (float, optional): Standard deviation of added Gaussian noise. Defaults to 1e-6.
        test_size (float, optional): Proportion of the data to use for testing. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test) - Training and test sets.
    """
    
    # Use random uniform sampling for all dimensions
    torch.manual_seed(random_state)
    X = torch.rand(n_samples, dim) * 4 - 2  # Scale to [-2, 2]
    
    # Compute the target variable `y` using the Rosenbrock function
    y = FunctionUtils.rosenbrock(X)  
    # Add Gaussian noise to the target variable to simulate real-world variability
    y += torch.randn(X.size(0)) * noise_level  
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Convert numpy arrays to torch tensors if needed (NOT SURE)
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
    
    return X_train, X_test, y_train, y_test
    
    # if dim <= 3:  # Low-dimensional case: Use grid-based sampling
    #     # Compute number of points per dimension to keep the total count close to n_samples
    #     n_per_dim = round(n_samples ** (1/dim))  
        
    #     # Generate evenly spaced values in each dimension within the range [-2,2]
    #     axes = [torch.linspace(-2, 2, n_per_dim) for _ in range(dim)]
        
    #     # Create a meshgrid of all possible combinations of values across dimensions
    #     """ !!!!!!! use random uniform, avoid meshgrid to seperate too evenly"""
    #     X = torch.stack(torch.meshgrid(*axes)).view(dim, -1).T  # Reshape into (num_samples, dim)
    
    # else:  # High-dimensional case: Use random sampling
    #     # Generate `n_samples` points where each dimension is sampled uniformly from [-2,2]
    #     X = torch.rand(n_samples, dim) * 4 - 2  

    

    # # Split the dataset into training (80%) and test (20%) sets
    # return train_test_split(X, y, test_size=test_size, random_state=random_state)