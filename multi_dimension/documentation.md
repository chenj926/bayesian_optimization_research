# Multi-Dimensional Gaussian Process Model Documentation

This project implements a flexible multi-dimensional Gaussian Process (GP) regression model using GPyTorch. It allows for experimentation with different mathematical functions, optimizers, and data parameters.

## Project Structure

-   **`function.py`**: Defines mathematical test functions (e.g., Rosenbrock, Ackley, Sphere, Rastrigin) in the `FunctionUtils` class.
-   **`data.py`**: Contains `generate_nd_data` for creating n-dimensional training and test datasets based on a chosen function from `FunctionUtils`, with configurable noise, sample size, and domain.
-   **`normalize_data.py`**: Provides `normalize_data` to apply Min-Max normalization to input features (X) and standard score normalization to target values (Y). It returns normalized data and normalization parameters for inverse transformation.
-   **`nd_model.py`**: Defines the `ExactGPModel` class, a GPyTorch exact GP model using a constant mean and an RBF kernel with ARD (Automatic Relevance Determination).
-   **`optimizer.py`**: Contains the `OptimizerHandler` class, which manages the instantiation and step logic for different PyTorch optimizers (e.g., L-BFGS, Adam).
-   **`training.py`**: Implements the `GPTrainer` class, responsible for the GP model training loop, hyperparameter optimization (via maximizing marginal log-likelihood), and evaluation (MSE, NMSE, MNLP).
-   **`visualizer.py`**: Provides the `Visualizer` class with methods (`visualize_1d`, `visualize_2d`) to plot GP regression results, including the true function, predicted mean, confidence intervals, and training data.
-   **`main.py`**: Contains the `MainRunner` class, which orchestrates the overall experimental workflow. It handles data generation, normalization, training, and visualization based on a flexible configuration.
-   **`result.ipynb`**: Jupyter notebook serving as the main interface to run experiments. It allows users to define an `experiment_config` dictionary to control dimensions, functions, optimizers, learning rates, iterations, sample sizes, etc.

## Code Features and Implementation Details

### 1. `function.py` - `FunctionUtils`

-   **Purpose**: Provides a collection of benchmark mathematical functions for testing GP models.
-   **Implementation**:
    -   Functions like `rosenbrock`, `ackley`, `sphere`, `rastrigin` are implemented as static methods.
    -   Each function accepts a PyTorch tensor `x` of shape `(N, dim)` (batch of N points, each with `dim` dimensions) or `(dim,)` (single point) and returns a tensor of corresponding function values.
    -   They are designed to work with PyTorch's automatic differentiation.
-   **Syntax Example (`ackley` function)**:
    ```python
    @staticmethod
    def ackley(x: torch.Tensor, a=20, b=0.2, c=2 * math.pi) -> torch.Tensor:
        if x.ndim == 1: # Handle single input point
            x = x.unsqueeze(0)
        dim = x.size(1)
        # Vectorized computation for efficiency
        sum_sq_term = -a * torch.exp(-b * torch.sqrt(torch.sum(x**2, dim=1) / dim))
        cos_term = -torch.exp(torch.sum(torch.cos(c * x), dim=1) / dim)
        return sum_sq_term + cos_term + a + torch.exp(torch.tensor(1.0))
    ```
    -   `x.ndim == 1`: Checks if the input is a 1D tensor (single data point) and reshapes it to `(1, dim)` for consistent batch processing.
    -   `torch.sum(x**2, dim=1)`: Calculates the sum of squares for each point along its dimensions.

### 2. `data.py` - `generate_nd_data`

-   **Purpose**: Generates synthetic datasets for GP regression.
-   **Features**:
    -   Selectable `function_name` (from `FunctionUtils`).
    -   Configurable input `dim` (dimensionality).
    -   Configurable `n_samples`, `noise_level`, `test_size`.
    -   `domain_range` parameter to define the input space bounds.
    -   Uses `torch.rand` for uniform sampling within the domain.
    -   `getattr(FunctionUtils, function_name)` dynamically calls the selected test function.
-   **Difficult Syntax**:
    -   `getattr(object, name)`: This built-in Python function retrieves an attribute (method, in this case) from an object by its string name. It's used here for dynamic function dispatch.

### 3. `optimizer.py` - `OptimizerHandler`

-   **Purpose**: Abstracts the choice and usage of different optimizers.
-   **Features**:
    -   `__init__` takes `model`, `optimizer_name` ("LBFGS" or "ADAM"), and `lr`.
    -   Instantiates `torch.optim.LBFGS` or `torch.optim.Adam`.
-   **Training Step Logic**:
    -   The `training.py` script handles the core training loop.
    -   **L-BFGS**: Requires a `closure` function that re-evaluates the model and computes the loss and gradients. `optimizer.step(closure)` is called.
    -   **Adam**: The typical PyTorch pattern is `optimizer.zero_grad()`, `loss = -mll(...)`, `loss.backward()`, `optimizer.step()`. This is implemented in the training loop.
    ```python
    # In training.py, for Adam:
    # optimizer_handler.optimizer.zero_grad()
    # output = model(X_train_norm)
    # loss = -mll(output, y_train_norm)
    # loss.backward()
    # optimizer_handler.optimizer.step()
    ```

### 4. `training.py` - `GPTrainer`

-   **Purpose**: Manages the GP model training and evaluation.
-   **Features**:
    -   `__init__` accepts `likelihood_class`, `optimizer_handler_class`, `optimizer_name`, and `base_lr`.
    -   `train` method:
        -   Takes normalized training/test data, dimension, normalization parameters, `train_iter`, and `current_lr`.
        -   Instantiates `ExactGPModel`, the specified likelihood, and `OptimizerHandler`.
        -   Uses `gpytorch.mlls.ExactMarginalLogLikelihood` (MLL) as the loss function.
        -   The training loop iteratively calls the optimizer's step method.
        -   Handles different step logic for L-BFGS (uses closure) and Adam (explicit zero_grad, backward, step).
        -   Evaluates the trained model using MSE, NMSE (on original scale), and MNLP (on normalized scale).
-   **Closure for L-BFGS**:
    ```python
    # Inside GPTrainer.train
    def closure():
        optimizer_handler.optimizer.zero_grad()
        output = model(X_train_norm) # Forward pass
        loss = -mll(output, y_train_norm) # Compute loss
        loss.backward() # Compute gradients
        return loss
    # ...
    if self.optimizer_name.upper() == "LBFGS":
        optimizer_handler.optimizer.step(closure)
    ```
    The `closure` is essential for L-BFGS as it may need to re-evaluate the loss and gradients multiple times per optimization step.

### 5. `main.py` - `MainRunner`

-   **Purpose**: Orchestrates experiments based on a configuration dictionary.
-   **Features**:
    -   `__init__` takes all necessary components (data generator, normalizer, trainer class, optimizer handler class, visualizer) and a `config` dictionary.
    -   `run_single_experiment`: Executes one training and evaluation run for a specific configuration.
    -   `run_from_config`:
        -   Parses the `config` dictionary for global and dimension-specific settings.
        -   Parameters controlled: `dims_to_run`, `function_name`, `n_samples`, `optimizer_name`, `lr`, `train_iter`, `noise_level`, `domain_range`, `random_state_data`.
        -   Dynamically fetches the target function for data generation and visualization using `getattr(FunctionUtils, current_function)`.
        -   Calls visualization methods for 1D and 2D cases.

### 6. `result.ipynb` - Experiment Control

-   **Purpose**: User interface for configuring and running experiments.
-   **Implementation**:
    -   A Python dictionary `experiment_config` is defined in a cell.
    -   This dictionary allows specifying:
        -   `dims_to_run`: List of dimensions (e.g., `[1, 2, 3]`).
        -   Global defaults for `function_name`, `n_samples`, `optimizer_name`, `lr`, `train_iter`, etc.
        -   `dim_specific_configs`: A nested dictionary to override global settings for specific dimensions. For example:
            ```python
            "dim_specific_configs": {
                "1": {"function_name": "sphere", "optimizer_name": "ADAM", "lr": 0.05},
                "2": {"function_name": "ackley", "domain_range": (-5, 5)}
            }
            ```
    -   The `MainRunner` is instantiated with this `config` and `runner.run_from_config()` is called to execute all defined experiments.

### 7. `normalize_data.py` - `normalize_data`

-   **X Normalization (Min-Max)**:
    -   `X_min = X_train.min(dim=0, keepdim=True)[0]`
    -   `X_max = X_train.max(dim=0, keepdim=True)[0]`
    -   `X_range = X_max - X_min`
    -   `X_range[X_range == 0] = 1e-8`: Prevents division by zero if a feature is constant.
    -   `X_train_norm = (X_train - X_min) / X_range`
    -   Test data is normalized using parameters from the *training data*.
-   **Y Normalization (Standardization)**:
    -   `y_mean = y_train.mean()`
    -   `y_std = y_train.std()`
    -   `if y_std.item() == 0: y_std = torch.tensor(1e-8)`: Prevents division by zero.
    -   `y_train_norm = (y_train - y_mean) / y_std`
    -   Test data (Y) is normalized using mean and std from the *training data*.

### 8. `visualizer.py` - `Visualizer`

-   **Purpose**: Plots GP regression results.
-   **Features**:
    -   `visualize_1d` and `visualize_2d` now accept `function_to_plot` (a callable) and normalization parameters (`x_norm_params`, `y_norm_params`).
    -   **Un-normalization for Plotting**:
        -   Training points (`X_train_orig`, `y_train_orig`) are plotted in their original scale.
        -   A test grid is created in the *original domain* of X. This grid is then *normalized* before being fed to the GP model for predictions (`test_x_norm_domain`).
        -   The GP model's predictions (mean and confidence bounds), which are on the *normalized Y scale*, are then *un-normalized* back to the original Y scale using `y_mean_val` and `y_std_val` before plotting.
        -   The true function is plotted using `function_to_plot(test_x_orig_domain)`.
    -   `torch.meshgrid`: Used in `visualize_2d` to create a grid of points for contour plotting. `indexing='ij'` ensures matrix indexing.

## Critical Steps for Bayesian Optimization (Future)

The current GP model forms a strong foundation for Bayesian Optimization (BO). Key aspects that will be leveraged:

1.  **GP Model (`ExactGPModel`)**: The GP will serve as the surrogate model to approximate the expensive black-box objective function you want to optimize.
2.  **Prediction Capabilities**: The model's ability to predict mean and variance (uncertainty) at unobserved points (`pred_likelihood_output.mean`, `pred_likelihood_output.variance` or `pred_likelihood_output.stddev`) is crucial.
3.  **Acquisition Functions**: You will need to implement acquisition functions (e.g., Expected Improvement, Probability of Improvement, Upper Confidence Bound). These functions use the GP's mean and variance predictions to decide where to sample next.
4.  **Optimization Loop**: BO is an iterative process:
    a.  Train the GP on initial data.
    b.  Use an acquisition function to find the next point to evaluate.
    c.  Evaluate the true objective function at this new point.
    d.  Add the new point to the dataset and retrain the GP.
    e.  Repeat until a budget is exhausted or convergence.

The parameterization added (choosing functions, optimizers, etc.) will be valuable for testing and comparing BO performance under different scenarios and with different underlying "true" functions.