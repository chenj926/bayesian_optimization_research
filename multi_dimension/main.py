import torch
from data import generate_nd_data
from normalize_data import normalize_data
from training import GPTrainer
from visualizer import Visualizer
import gpytorch
from function import FunctionUtils

class MainRunner:
    def __init__(self, data_generator, normalizer, trainer_class, 
                 optimizer_handler_class, visualizer, config):
        self.data_generator = data_generator
        self.normalizer = normalizer
        self.trainer_class = trainer_class # Store trainer class
        self.optimizer_handler_class = optimizer_handler_class # Store optimizer handler class
        self.visualizer = visualizer
        self.config = config # Store the configuration dictionary

    def run_single_experiment(self, d, function_name, n_samples, 
                              optimizer_name, lr, train_iter, noise_level,
                              domain_range, random_state_data):
        print(f"\n--- Training {d}D model using {function_name} with {optimizer_name} (LR: {lr}, Iter: {train_iter}, Samples: {n_samples}) ---")

        X_train, X_test, y_train, y_test = self.data_generator(
            dim=d, 
            n_samples=n_samples,
            noise_level=noise_level,
            function_name=function_name,
            domain_range=domain_range,
            random_state=random_state_data 
        )

        X_train_norm, X_test_norm, y_train_norm, y_test_norm, x_norm_params, y_norm_params = self.normalizer(
            X_train, X_test, y_train, y_test
        )

        # Instantiate GPTrainer with optimizer choice and base LR
        # Specific LR for this run is passed to train method
        trainer_instance = self.trainer_class(
            gpytorch.likelihoods.GaussianLikelihood, 
            self.optimizer_handler_class,
            optimizer_name=optimizer_name, # Pass optimizer_name to trainer constructor
            base_lr=lr # Can set base_lr here, or pass lr directly to train
        )

        model, metrics = trainer_instance.train(
            X_train_norm, y_train_norm, X_test_norm, y_test_norm, 
            dim=d, 
            y_norm_params=y_norm_params,
            train_iter=train_iter,
            current_lr=lr # Pass the specific lr for this run
        )

        print(f"{d}D Metrics for {function_name}:")
        print(f"  MSE  = {metrics['MSE']:.4f}")
        print(f"  NMSE = {metrics['NMSE']:.4f}")
        print(f"  MNLP = {metrics['MNLP']:.4f}")

        # Get the actual function to plot
        try:
            func_to_plot = getattr(FunctionUtils, function_name)
        except AttributeError:
            print(f"Warning: Function {function_name} not found in FunctionUtils for visualization.")
            func_to_plot = None

        if func_to_plot:
            if d == 1:
                self.visualizer.visualize_1d(model, X_train_norm, y_train,
                                             x_norm_params, y_norm_params, func_to_plot)
            elif d == 2:
                self.visualizer.visualize_2d(model, X_train_norm, y_train,
                                             x_norm_params, y_norm_params, func_to_plot)
        return metrics
    
    def run_from_config(self):
        results = {}
        
        # General settings from config
        dims_to_run = self.config.get("dims_to_run", [1, 2]) # Default to 1D, 2D
        function_name = self.config.get("function_name", "rosenbrock")
        n_samples_global = self.config.get("n_samples", 100)
        optimizer_name_global = self.config.get("optimizer_name", "LBFGS")
        lr_global = self.config.get("lr", 0.01)
        train_iter_global = self.config.get("train_iter", 200)
        noise_level_global = self.config.get("noise_level", 1e-4) # Added noise level
        domain_range_global = self.config.get("domain_range", (-2,2)) # Added domain range
        random_state_data_global = self.config.get("random_state_data", None)

        # Per-dimension overrides if provided
        dim_configs = self.config.get("dim_specific_configs", {})

        for d in dims_to_run:
            # Get specific settings for this dimension, or use global
            current_dim_config = dim_configs.get(str(d), {}) # Use str(d) if keys are strings
            
            current_n_samples = current_dim_config.get("n_samples", n_samples_global)
            # Special handling for n_samples based on dimension if required (as in original logic)
            if d == 1 and "n_samples_1d_viz" in self.config:
                 current_n_samples = self.config.get("n_samples_1d_viz", 50) # Adjusted for better viz
            elif d > 2 and "n_samples_high_dim_factor" in self.config:
                 current_n_samples = max(current_n_samples, self.config.get("n_samples_high_dim_factor",30) * d)

            current_optimizer = current_dim_config.get("optimizer_name", optimizer_name_global)
            current_lr = current_dim_config.get("lr", lr_global)
            current_train_iter = current_dim_config.get("train_iter", train_iter_global)
            current_function = current_dim_config.get("function_name", function_name) # Allow function override per dim
            current_noise = current_dim_config.get("noise_level", noise_level_global)
            current_domain = current_dim_config.get("domain_range", domain_range_global)
            current_random_state_data = current_dim_config.get("random_state_data", random_state_data_global)

            metrics = self.run_single_experiment(
                d=d,
                function_name=current_function,
                n_samples=current_n_samples,
                optimizer_name=current_optimizer,
                lr=current_lr,
                train_iter=current_train_iter,
                noise_level=current_noise,
                domain_range=current_domain,
                random_state_data=current_random_state_data
            )
            if d not in results:
                results[d] = {}
            results[d][current_function] = metrics # Store results per dim and function
            
        print("\n=== Summary Metrics ===")
        for d_val, func_metrics_dict in results.items():
            for func_name_key, m in func_metrics_dict.items():
                 print(f"{d_val}D - {func_name_key} -> MSE: {m['MSE']:.4f}, NMSE: {m['NMSE']:.4f}, MNLP: {m['MNLP']:.4f}")
        return results


        # dims = [1, 2, 3, 4, 5]
        # results = {}
        # for d in dims:
        #     print(f"\n--- Training {d}D model ---")

        #     # Increase n_samples for better training
        #     n_samples_val = max(50, 30 * d) # Example: 50 for 1D, increases for higher D
        #     if d == 1:
        #         n_samples_val = 5 # samples for 1D visualization
            
        #     X_train, X_test, y_train, y_test = self.data_generator(dim=d, n_samples=n_samples_val)
            
        #     # Normalize the data (z-score normalization)
        #     X_train_norm, X_test_norm, y_train_norm, y_test_norm, x_norm_params, y_norm_params = self.normalizer(
        #         X_train, X_test, y_train, y_test # Pass y_train, y_test
        #     )

        #     # Train the GP model and evaluate using MSE, NMSE, MNLP
        #     # Pass y_train_norm, y_test_norm (for evaluation consistency if needed within train)
        #     # and y_norm_params for unscaling metrics
        #     model, metrics = self.trainer.train(
        #         X_train_norm, y_train_norm, X_test_norm, y_test_norm, d, y_norm_params
        #     )

        #     results[d] = metrics # Store metrics
            
        #     print(f"{d}D Metrics:")
        #     print(f"  MSE  = {metrics['MSE']:.4f}")
        #     print(f"  NMSE = {metrics['NMSE']:.4f}")
        #     print(f"  MNLP = {metrics['MNLP']:.4f}")
            
        #     # Visualize only for 1D and 2D cases
        #     if d == 1:
        #         # # Unpack the min and max from norm_params
        #         X_min = x_norm_params['X_min']
        #         X_max = x_norm_params['X_max']
                
        #         # Pass them to the visualizer
        #         self.visualizer.visualize_1d(model, X_train_norm, y_train, # Pass original y_train for plotting
        #                                      X_min, X_max,
        #                                      y_norm_params)
                
        #     elif d == 2:
        #         # Unpack the min and max from norm_params
        #         X_min = x_norm_params['X_min']
        #         X_max = x_norm_params['X_max']
                
        #         self.visualizer.visualize_2d(model, X_train_norm, y_train, # Pass original y_train for plotting
        #                                      X_min, X_max,
        #                                      y_norm_params)
        #     results[d] = metrics
            
        # return results