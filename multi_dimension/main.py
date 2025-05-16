import torch
from data import generate_nd_data
from normalize_data import normalize_data
from training import GPTrainer
from visualizer import Visualizer

class MainRunner:
    def __init__(self, data_generator, normalizer, trainer, visualizer):
        self.data_generator = data_generator
        self.normalizer = normalizer
        self.trainer = trainer
        self.visualizer = visualizer

    def run(self):
        dims = [1, 2, 3, 4, 5]
        results = {}
        for d in dims:
            print(f"\n--- Training {d}D model ---")

            # Increase n_samples for better training
            n_samples_val = max(50, 30 * d) # Example: 50 for 1D, increases for higher D
            if d == 1:
                n_samples_val = 5 # samples for 1D visualization
            
            X_train, X_test, y_train, y_test = self.data_generator(dim=d, n_samples=n_samples_val)
            
            # Normalize the data (z-score normalization)
            X_train_norm, X_test_norm, y_train_norm, y_test_norm, x_norm_params, y_norm_params = self.normalizer(
                X_train, X_test, y_train, y_test # Pass y_train, y_test
            )

            # Train the GP model and evaluate using MSE, NMSE, MNLP
            # Pass y_train_norm, y_test_norm (for evaluation consistency if needed within train)
            # and y_norm_params for unscaling metrics
            model, metrics = self.trainer.train(
                X_train_norm, y_train_norm, X_test_norm, y_test_norm, d, y_norm_params
            )

            results[d] = metrics # Store metrics
            
            print(f"{d}D Metrics:")
            print(f"  MSE  = {metrics['MSE']:.4f}")
            print(f"  NMSE = {metrics['NMSE']:.4f}")
            print(f"  MNLP = {metrics['MNLP']:.4f}")
            
            # Visualize only for 1D and 2D cases
            if d == 1:
                # # Unpack the min and max from norm_params
                X_min = x_norm_params['X_min']
                X_max = x_norm_params['X_max']
                
                # Pass them to the visualizer
                self.visualizer.visualize_1d(model, X_train_norm, y_train, # Pass original y_train for plotting
                                             X_min, X_max,
                                             y_norm_params)
                
            elif d == 2:
                # Unpack the min and max from norm_params
                X_min = x_norm_params['X_min']
                X_max = x_norm_params['X_max']
                
                self.visualizer.visualize_2d(model, X_train_norm, y_train, # Pass original y_train for plotting
                                             X_min, X_max,
                                             y_norm_params)
            results[d] = metrics
            
        return results