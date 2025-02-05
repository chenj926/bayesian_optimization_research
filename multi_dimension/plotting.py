# plotting.py
import torch
import matplotlib.pyplot as plt
import gpytorch
from function import rosenbrock

def visualize_1d(model, X_train, y_train):
    """
    Visualizes the GP regression for a 1D function.
    """
    X_test = torch.linspace(-2, 2, 100).unsqueeze(-1)
    model.eval()
    model.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = model.likelihood(model(X_test))
    
    # True function values
    y_true = rosenbrock(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.plot(X_test.numpy(), y_true.numpy(), 'k--', label="True")
    plt.plot(X_test.numpy(), pred.mean.numpy(), 'b', label="Predicted")
    plt.fill_between(
        X_test.squeeze().numpy(),
        (pred.mean - 2*pred.stddev).numpy(),
        (pred.mean + 2*pred.stddev).numpy(),
        alpha=0.2, color='b'
    )
    plt.plot(X_train.numpy(), y_train.numpy(), 'ro', label="Train Points")
    plt.title("1D Rosenbrock GP Regression")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.show()

def visualize_2d(model, X_train, y_train):
    """
    Visualizes the GP regression for a 2D function using contour plots.
    """
    Ngrid = 50
    x_lin = torch.linspace(-2, 2, Ngrid)
    y_lin = torch.linspace(-2, 2, Ngrid)
    X1, X2 = torch.meshgrid(x_lin, y_lin, indexing='ij')
    X_grid = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=-1)
    
    model.eval()
    model.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = model.likelihood(model(X_grid))
    Z_pred = pred.mean.reshape(Ngrid, Ngrid)
    
    # Compute true function values on the grid
    Z_true = rosenbrock(X_grid).reshape(Ngrid, Ngrid)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    cs1 = axs[0].contourf(X1.numpy(), X2.numpy(), Z_true.numpy(), levels=20, cmap='viridis')
    axs[0].scatter(X_train[:, 0].numpy(), X_train[:, 1].numpy(), color='r', s=15, label="Train Points")
    axs[0].set_title("True Rosenbrock (2D)")
    axs[0].set_xlabel("x1")
    axs[0].set_ylabel("x2")
    fig.colorbar(cs1, ax=axs[0])
    
    cs2 = axs[1].contourf(X1.numpy(), X2.numpy(), Z_pred.numpy(), levels=20, cmap='viridis')
    axs[1].scatter(X_train[:, 0].numpy(), X_train[:, 1].numpy(), color='r', s=15, label="Train Points")
    axs[1].set_title("GP Predicted Mean (2D)")
    axs[1].set_xlabel("x1")
    axs[1].set_ylabel("x2")
    fig.colorbar(cs2, ax=axs[1])
    
    plt.tight_layout()
    plt.show()
