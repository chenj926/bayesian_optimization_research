import torch
from sklearn.model_selection import train_test_split
# from function import rosenbrock

# Update the normalization function to use Min-Max normalization
def normalize_data(X_train, X_test, y_train, y_test):
    """
    Applies min-max normalization to the data.

    Returns:
      X_train_norm, X_test_norm, y_train_norm, y_test_norm, norm_params

    where norm_params is a dictionary with the computed min and max values.
    """
    X_min = X_train.min(dim=0, keepdim=True)[0]
    X_max = X_train.max(dim=0, keepdim=True)[0]
    y_min = y_train.min()
    y_max = y_train.max()

    # Prevent division by zero by adding a small epsilon where min and max are equal
    X_range = X_max - X_min
    X_range[X_range == 0] = 1e-8  # Avoid division by zero
    
    y_range = y_max - y_min
    if y_range == 0:
        y_range = 1e-8  # Avoid division by zero

    # Apply min-max normalization
    X_train_norm = (X_train - X_min) / X_range
    X_test_norm = (X_test - X_min) / X_range
    y_train_norm = (y_train - y_min) / y_range
    y_test_norm = (y_test - y_min) / y_range
    
    norm_params = {'X_min': X_min, 'X_max': X_max, 'y_min': y_min, 'y_max': y_max}
    return X_train_norm, X_test_norm, y_train_norm, y_test_norm, norm_params