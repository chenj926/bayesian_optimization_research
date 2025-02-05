import torch
from sklearn.model_selection import train_test_split
# from function import rosenbrock

def normalize_data(X_train, X_test, y_train, y_test):
    """
    Applies z-score normalization to the data.

    Returns:
      X_train_norm, X_test_norm, y_train_norm, y_test_norm, norm_params

    where norm_params is a dictionary with the computed means and stds.
    """
    X_mean = X_train.mean(dim=0, keepdim=True)
    X_std  = X_train.std(dim=0, keepdim=True)
    y_mean = y_train.mean()
    y_std  = y_train.std()

    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm  = (X_test - X_mean) / X_std
    y_train_norm = (y_train - y_mean) / y_std
    y_test_norm  = (y_test - y_mean) / y_std
    
    # use / ub - lb one

    norm_params = {'X_mean': X_mean, 'X_std': X_std, 'y_mean': y_mean, 'y_std': y_std}
    return X_train_norm, X_test_norm, y_train_norm, y_test_norm, norm_params