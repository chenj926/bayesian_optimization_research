import torch
from sklearn.model_selection import train_test_split
# from function import rosenbrock

# Update the normalization function to use Min-Max normalization
def normalize_data(X_train, X_test, y_train, y_test):
    """
    Applies min-max normalization to the X, and standard score normalization to Y.

    Returns:
        X_train_norm, X_test_norm, y_train_norm, y_test_norm,
        x_norm_params (dict with X_min, X_max),
        y_norm_params (dict with y_mean, y_std)
    """
    # we normalize x (Min-Max) -------------------------------------------------
    X_min = X_train.min(dim=0, keepdim=True)[0]
    X_max = X_train.max(dim=0, keepdim=True)[0]
    # Prevent division by zero by adding a small epsilon where min and max are equal
    X_range = X_max - X_min
    X_range[X_range == 0] = 1e-8  # Avoid division by zero
    
    # Apply min-max normalization
    X_train_norm = (X_train - X_min) / X_range
    X_test_norm = (X_test - X_min) / X_range
    # x norm params for inverse norm
    x_norm_params = {'X_min': X_min, 'X_max': X_max}

    # Normalize Y (Standardization: zero mean, unit variance) -------------------
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std.item() == 0:  # Check if std is zero
        y_std = torch.tensor(1e-8)  # Avoid division by zero

    y_train_norm = (y_train - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std  # *Use training mean and std for test set
    # y norm params for inverse norm
    y_norm_params = {'y_mean': y_mean, 'y_std': y_std}
    
    # norm_params = {'X_min': X_min, 'X_max': X_max}
    return X_train_norm, X_test_norm, y_train_norm, y_test_norm, x_norm_params, y_norm_params