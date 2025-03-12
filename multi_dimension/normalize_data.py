import torch
from sklearn.model_selection import train_test_split
# from function import rosenbrock

# Update the normalization function to use Min-Max normalization
def normalize_data(X_train, X_test):
    """
    Applies min-max normalization to the data.

    Returns:
      X_train_norm, X_test_norm, norm_params

    where norm_params is a dictionary with the computed min and max values.
    """
    # we only normalize x
    X_min = X_train.min(dim=0, keepdim=True)[0]
    X_max = X_train.max(dim=0, keepdim=True)[0]

    # Prevent division by zero by adding a small epsilon where min and max are equal
    X_range = X_max - X_min
    X_range[X_range == 0] = 1e-8  # Avoid division by zero

    # Apply min-max normalization
    X_train_norm = (X_train - X_min) / X_range
    X_test_norm = (X_test - X_min) / X_range
    
    norm_params = {'X_min': X_min, 'X_max': X_max}
    return X_train_norm, X_test_norm, norm_params