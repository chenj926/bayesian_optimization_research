import math
import torch
import gpytorch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import time

from function import FunctionUtils

def generate_nd_data(dim: int, n_samples: int = 5, noise_level: float = 1e-6, 
                     test_size: float = 0.2, random_state: int = None,
                     function_name: str = "rosenbrock", domain_range=(-2, 2)):
    """
    Generates n-dimensional data for regression using a specified function.

    Args:
        dim (int): Number of dimensions.
        n_samples (int, optional): Total number of samples. Defaults to 5.
        noise_level (float, optional): Standard deviation of added Gaussian noise. Defaults to 1e-6.
        test_size (float, optional): Proportion of the data to use for testing. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to None.
        function_name (str, optional): Name of the function in FunctionUtils to use. Defaults to "rosenbrock".
        domain_range (tuple, optional): Tuple (min_val, max_val) for the input domain. Defaults to (-2, 2).

    Returns:
        Tuple: (X_train, X_test, y_train, y_test) - Training and test sets.
    """
    # Use random uniform sampling for all dimensions
    current_seed = random_state
    if random_state is None:
        # Generate a seed based on current time if None is passed for more randomness
        # This helps ensure torch gets a new seed if None is explicitly passed
        current_seed = int(time.time() * 1000) % (2**32)
    torch.manual_seed(current_seed)

    domain_min, domain_max = domain_range
    X = torch.rand(n_samples, dim) * (domain_max - domain_min) + domain_min  # Scale to [-2, 2]

    try:
        func_to_call = getattr(FunctionUtils, function_name)
    except AttributeError:
        raise ValueError(f"Function {function_name} not found in FunctionUtils.")
    
    # Compute the target variable `y` using the chosen function
    y = func_to_call(X)  
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