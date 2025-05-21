import torch
import gpytorch
import time
import numpy as np
from matplotlib import pyplot as plt

from data import generate_nd_data
from normalize_data import normalize_data
from training import GPTrainer
from optimizer import OptimizerHandler
from nd_model import ExactGPModel
from function import FunctionUtils
from acquisition_function import AcquisitionFunctionUtils
from visualizer import Visualizer # Assuming visualizer will be enhanced

class BayesianOptimizerRunner:
    def __init__(self, config, trainer_class, optimizer_handler_class, visualizer_instance=None):
        self.config = config
        self.objective_function = getattr(FunctionUtils, config["objective_function_name"])
        self.gp_model_class = ExactGPModel
        self.likelihood_class = gpytorch.likelihoods.GaussianLikelihood
        self.trainer_class = trainer_class
        self.optimizer_handler_class = optimizer_handler_class
        self.acquisition_function_utils = AcquisitionFunctionUtils()
        self.normalizer = normalize_data
        self.data_generator = generate_nd_data
        self.visualizer = visualizer_instance if visualizer_instance else Visualizer() # Use provided or new

        self.dim = config["dim"]
        self.domain_range = config["domain_range"]
        self.noise_level = config["noise_level"]
        self.bo_iterations = config["bo_iterations"]
        self.num_initial_samples = config["num_initial_samples"]
        self.initial_sampling_random_state = config.get("initial_sampling_random_state", int(time.time()))
        
        self.acq_func_name = config.get("acq_func_name", "ucb")
        self.kappa_ucb = config.get("kappa_ucb", 1.96)
        # self.xi_ei_pi = config.get("xi_ei_pi", 0.01) # For EI/PI

        self.gp_train_iter = config["gp_train_iter"]
        self.gp_optimizer_name = config["gp_optimizer_name"]
        self.gp_lr = config["gp_lr"]
        self.num_candidate_points = config["num_candidate_points"]
        self.visualization_interval = config.get("visualization_interval", 0) # 0 means no intermediate viz
        self.random_state_bo_loop = config.get("random_state_bo_loop", None)

        if self.random_state_bo_loop is not None:
            torch.manual_seed(self.random_state_bo_loop)
            np.random.seed(self.random_state_bo_loop)

        self.X_observed_orig = None
        self.y_observed_orig = None
        self.best_y_observed = float('inf')
        self.best_x_observed = None
        
        self.history = []


    def _initialize_data(self):
        """Generates initial data points for BO."""
        X_train_init, _, y_train_init, _ = self.data_generator(
            dim=self.dim,
            n_samples=self.num_initial_samples,
            noise_level=self.noise_level, # Initial noise is from true function + noise
            test_size=0.01, # Minimal test set as we only need initial training data
            random_state=self.initial_sampling_random_state,
            function_name=self.config["objective_function_name"],
            domain_range=self.domain_range
        )
        self.X_observed_orig = X_train_init
        self.y_observed_orig = y_train_init

        # Initialize best observed
        min_y_idx = torch.argmin(self.y_observed_orig)
        self.best_y_observed = self.y_observed_orig[min_y_idx].item()
        self.best_x_observed = self.X_observed_orig[min_y_idx]


    def _get_next_sample_point(self):
        """Trains GP and uses acquisition function to find the next sample point."""
        # 1. Normalize currently observed data
        # We use X_observed_orig itself as X_test for normalization purposes
        # as norm params are derived from X_train (which is X_observed_orig here)
        # A single point for X_test_dummy and y_test_dummy is enough
        X_test_dummy = self.X_observed_orig[0:1]
        y_test_dummy = self.y_observed_orig[0:1]

        X_observed_norm, _, y_observed_norm, _, x_norm_params, y_norm_params = self.normalizer(
            self.X_observed_orig, X_test_dummy, self.y_observed_orig, y_test_dummy
        )

        # 2. Train GP model
        trainer_instance = self.trainer_class(
            self.likelihood_class,
            self.optimizer_handler_class,
            optimizer_name=self.gp_optimizer_name,
            base_lr=self.gp_lr
        )
        # Note: The GPTrainer's train method expects X_test_norm and y_test_norm for metrics.
        # For BO's GP fitting step, these metrics are less critical than the posterior.
        # We can pass dummy or a small validation set if needed, or adapt the trainer.
        # For now, let's assume we only care about the model.
        # The current trainer's metrics are on original Y scale after un-normalizing.
        
        gp_model, _ = trainer_instance.train(
            X_train_norm=X_observed_norm,
            y_train_norm=y_observed_norm,
            X_test_norm=X_observed_norm, # Using observed for test metrics for now
            y_test_norm=y_observed_norm, # Using observed for test metrics for now
            dim=self.dim,
            y_norm_params=y_norm_params, # For unscaling metrics if trainer uses it
            train_iter=self.gp_train_iter,
            current_lr=self.gp_lr
        )

        # 3. Generate candidate points over the domain
        domain_min, domain_max = self.domain_range
        if self.dim == 1:
            X_candidates_orig = torch.linspace(domain_min, domain_max, self.num_candidate_points).unsqueeze(-1)
        else:
            # For >1D, random sampling is more practical than a dense grid
            current_seed = int(time.time() * 1000) % (2**32) if self.random_state_bo_loop is None else self.random_state_bo_loop
            torch.manual_seed(current_seed) # ensure reproducibility if random_state_bo_loop is set
            X_candidates_orig = torch.rand(self.num_candidate_points, self.dim) * (domain_max - domain_min) + domain_min
        
        # 4. Normalize candidate points
        X_min_norm_param = x_norm_params['X_min']
        X_range_norm_param = x_norm_params['X_max'] - X_min_norm_param
        X_range_norm_param[X_range_norm_param == 0] = 1e-8 # Avoid division by zero
        X_candidates_norm = (X_candidates_orig - X_min_norm_param) / X_range_norm_param

        # 5. Evaluate acquisition function
        acq_values = None
        if self.acq_func_name == "ucb":
            acq_values = self.acquisition_function_utils.upper_confidence_bound(
                gp_model, trainer_instance.likelihood, X_candidates_norm, kappa=self.kappa_ucb
            )
        # Add EI, PI logic here if needed
        # elif self.acq_func_name == "ei":
        #     # Need y_max_norm for EI
        #     # y_max_orig = torch.min(self.y_observed_orig) # Assuming minimization
        #     # y_max_norm = (y_max_orig - y_norm_params['y_mean']) / y_norm_params['y_std']
        #     # acq_values = self.acquisition_function_utils.expected_improvement(...)
        #     pass
        else:
            raise ValueError(f"Unsupported acquisition function: {self.acq_func_name}")

        # 6. Select point that maximizes acquisition function
        ### ! ------------------------ ! ####
        idx_best_candidate = torch.argmax(acq_values)
        
        x_next_norm = X_candidates_norm[idx_best_candidate].unsqueeze(0) # Keep as (1, dim)

        # 7. Inverse-normalize the selected point
        x_next_orig = x_next_norm * X_range_norm_param + X_min_norm_param
        
        # For visualization purposes, return acquisition values and candidates
        self.current_gp_model = gp_model
        self.current_likelihood = trainer_instance.likelihood
        self.current_x_norm_params = x_norm_params
        self.current_y_norm_params = y_norm_params
        self.current_acq_values_norm = acq_values
        self.current_X_candidates_orig = X_candidates_orig
        self.current_X_candidates_norm = X_candidates_norm


        return x_next_orig.squeeze(0) # Return as (dim,)

    def optimize(self):
        """Main Bayesian Optimization loop."""
        self._initialize_data()
        print(f"Starting Bayesian Optimization for {self.config['objective_function_name']} in {self.dim}D.")
        print(f"Initial best y: {self.best_y_observed:.4f} at X: {self.best_x_observed.numpy()}")

        for i in range(self.bo_iterations):
            print(f"\n--- BO Iteration {i+1}/{self.bo_iterations} ---")
            x_next_orig = self._get_next_sample_point()
            
            # Evaluate objective function at the new point (add noise)
            # Ensure x_next_orig is the correct shape for objective_function
            if x_next_orig.ndim == 1:
                 x_next_orig_eval = x_next_orig.unsqueeze(0) # Make it (1, dim)
            else:
                 x_next_orig_eval = x_next_orig

            y_next_true = self.objective_function(x_next_orig_eval) # (1,)
            
            # Add observation noise
            noise = torch.randn(1) * self.noise_level
            y_next_observed = y_next_true + noise
            y_next_observed = y_next_observed.squeeze() # Make it scalar tensor

            # Add to observed data
            self.X_observed_orig = torch.cat([self.X_observed_orig, x_next_orig.unsqueeze(0)], dim=0)
            self.y_observed_orig = torch.cat([self.y_observed_orig, y_next_observed.unsqueeze(0)], dim=0)
            
            # Update best observed
            if y_next_observed.item() < self.best_y_observed:
                self.best_y_observed = y_next_observed.item()
                self.best_x_observed = x_next_orig
            
            self.history.append({
                'iteration': i + 1,
                'x_next': x_next_orig.tolist(),
                'y_next_observed': y_next_observed.item(),
                'current_best_y': self.best_y_observed,
                'current_best_x': self.best_x_observed.tolist()
            })

            print(f"Sampled X: {x_next_orig.numpy()}, Observed y: {y_next_observed.item():.4f}")
            print(f"Current best y: {self.best_y_observed:.4f} at X: {self.best_x_observed.numpy()}")

            if self.visualizer and self.visualization_interval > 0 and (i + 1) % self.visualization_interval == 0:
                if self.dim == 1:
                    self.visualizer.visualize_bo_1d(
                        model=self.current_gp_model,
                        likelihood=self.current_likelihood,
                        X_observed_orig=self.X_observed_orig,
                        y_observed_orig=self.y_observed_orig,
                        x_next_orig=x_next_orig,
                        acquisition_values_norm=self.current_acq_values_norm,
                        X_candidates_orig=self.current_X_candidates_orig,
                        objective_function=self.objective_function,
                        x_norm_params=self.current_x_norm_params,
                        y_norm_params=self.current_y_norm_params,
                        domain_range=self.domain_range,
                        iteration=i+1
                    )
                # Add 2D visualization call if implemented

        print("\n=== Bayesian Optimization Complete ===")
        print(f"Best y found: {self.best_y_observed:.4f}")
        print(f"At X: {self.best_x_observed.numpy()}")
        return self.X_observed_orig, self.y_observed_orig, self.best_x_observed, self.best_y_observed, self.history