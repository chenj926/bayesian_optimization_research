#!/usr/bin/env python
# coding: utf-8

# In[3]:


# main.py
import torch
from data import generate_nd_data
from normalize_data import normalize_data
from training import GPTrainer
from visualizer import Visualizer


# In[2]:


class MainRunner:
#     def main():
#         # Train models from 1D to 5D
#         dims = [1, 2, 3, 4, 5]
#         results = {}
        
#         for d in dims:
#             print(f"\n--- Training {d}D model ---")
#             # Generate the data (all as torch.Tensors)
#             X_train, X_test, y_train, y_test = generate_nd_data(dim=d, n_samples=10)
#             # Normalize the data (z-score normalization)
#             X_train_norm, X_test_norm, y_train_norm, y_test_norm, norm_params = normalize_data(
#                 X_train, X_test, y_train, y_test
#             )
            
#             # Train the GP model and evaluate using MSE, NMSE, MNLP
#             model, metrics = train_gp(
#                 X_train_norm, y_train_norm, X_test_norm, y_test_norm, 
#                 dim=d, train_iter=300, lr=0.01
#             )
#             results[d] = metrics
            
#             print(f"{d}D Metrics:")
#             print(f"  MSE  = {metrics['MSE']:.4f}")
#             print(f"  NMSE = {metrics['NMSE']:.4f}")
#             print(f"  MNLP = {metrics['MNLP']:.4f}")
            
#             # Visualize only for 1D and 2D cases
#             if d == 1:
#                 visualize_1d(model, X_train_norm, y_train_norm)
#             elif d == 2:
#                 visualize_2d(model, X_train_norm, y_train_norm)
        
#         print("\n=== Summary Metrics for 1D to 5D ===")
#         for d, m in results.items():
#             print(f"{d}D -> MSE: {m['MSE']:.4f}, NMSE: {m['NMSE']:.4f}, MNLP: {m['MNLP']:.4f}")

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
            X_train, X_test, y_train, y_test = self.data_generator(dim=d, n_samples=10)
            # Normalize the data (z-score normalization)
            X_train_norm, X_test_norm, norm_params = self.normalizer(
                X_train, X_test
            )
            # Train the GP model and evaluate using MSE, NMSE, MNLP
            model, metrics = self.trainer.train(
                X_train_norm, y_train, X_test_norm, y_test, d
            )
            
            print(f"{d}D Metrics:")
            print(f"  MSE  = {metrics['MSE']:.4f}")
            print(f"  NMSE = {metrics['NMSE']:.4f}")
            print(f"  MNLP = {metrics['MNLP']:.4f}")
            
            # Visualize only for 1D and 2D cases

            if d == 1:
                # Unpack the min and max from norm_params
                X_min = norm_params['X_min']
                X_max = norm_params['X_max']
                
                # Pass them to the visualizer
                self.visualizer.visualize_1d(model, X_train_norm, y_train, X_min, X_max)
                
            elif d == 2:
                # Unpack the min and max from norm_params
                X_min = norm_params['X_min']
                X_max = norm_params['X_max']
                
                self.visualizer.visualize_2d(model, X_train_norm, y_train, X_min, X_max)
            results[d] = metrics
            
        return results



# if __name__ == "__main__":
#     MainRunner.main()


# In[ ]:


# get_ipython().system('jupyter nbconvert --to script main.ipynb')

