import torch, gpytorch
import matplotlib.pyplot as plt

class Visualizer:
    def visualize_1d(self, model, X_train_norm, y_train_orig, 
                     x_norm_params, y_norm_params, function_to_plot,
                     original_data_range_extension=1.0): # Added function_to_plot
        
        # 1) Un-normalize X_train_norm for plotting (if needed, though scatter takes original X)
        X_min_norm = x_norm_params['X_min']
        X_max_norm = x_norm_params['X_max']
        X_range_norm = X_max_norm - X_min_norm
        X_range_norm[X_range_norm == 0] = 1e-8 # Avoid division by zero

        X_train_orig = X_train_norm * X_range_norm + X_min_norm
        
        # 2) Create a test grid in original X domain, then normalize it for the model
        test_x_min_orig = X_train_orig.min().item() - original_data_range_extension
        test_x_max_orig = X_train_orig.max().item() + original_data_range_extension
        test_x_orig_domain = torch.linspace(test_x_min_orig, test_x_max_orig, 100).unsqueeze(1)
        
        # test_x_orig_domain = torch.linspace(X_train_orig.min()-1, X_train_orig.max()+1, 100).unsqueeze(1) # Wider range
        # test_x_norm_domain = (test_x_orig_domain - X_min_norm) / X_range_norm
        
        # 3) Normalize the test grid for the model
        test_x_norm_domain = (test_x_orig_domain - X_min_norm) / X_range_norm
        
        # 4) Predict using the *normalized* test inputs
        model.eval()
        model.likelihood.eval()
        
        # # Get predictions on a finer grid
        # test_x = torch.linspace(-3, 3, 100).unsqueeze(1)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions_norm = model.likelihood(model(test_x_norm_domain))
            mean_norm = predictions_norm.mean
            lower_norm, upper_norm = predictions_norm.confidence_region()

        # 4) Un-normalize Y predictions
        y_mean_val = y_norm_params['y_mean']
        y_std_val = y_norm_params['y_std']
        
        mean_orig_domain = mean_norm * y_std_val + y_mean_val
        lower_orig_domain = lower_norm * y_std_val + y_mean_val
        upper_orig_domain = upper_norm * y_std_val + y_mean_val

        # True function values in the original domain
        y_true_orig_domain = function_to_plot(test_x_orig_domain)

        # Plot training data (original X, original Y)
        plt.figure(figsize=(12, 7))
        plt.plot(test_x_orig_domain.numpy(), y_true_orig_domain.numpy(), 'k--', label="True Function", zorder=1)
        plt.scatter(X_train_orig.numpy(), y_train_orig.numpy(), label='Training Data', color='blue', s=50, zorder=3)
        
        # Plot prediction (original X, un-normalized Y mean and confidence)
        plt.plot(test_x_orig_domain.numpy(), mean_orig_domain.numpy(), 'r-', label='Mean Prediction', zorder=2)
        plt.fill_between(test_x_orig_domain.squeeze().numpy(), 
                         lower_orig_domain.numpy(), 
                         upper_orig_domain.numpy(), 
                         alpha=0.3, color='red', label='Confidence Region (95%)')
        
        plt.legend()
        plt.title(f'1D GP Regression on {function_to_plot.__name__} (Normalized Y Training)')
        plt.xlabel('X (Original Domain)')
        plt.ylabel('Y (Original Domain)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
    
    def visualize_2d(self, model, X_train_norm, y_train_orig, 
                     x_norm_params, y_norm_params, function_to_plot,
                     original_data_range_extension=0.5):  # Added function_to_plot
        model.eval()
        model.likelihood.eval()
        
        # 1) Un‐normalize X_train_norm for plotting
        X_min_norm = x_norm_params['X_min']
        X_max_norm = x_norm_params['X_max']
        X_range_norm = X_max_norm - X_min_norm
        X_range_norm[X_range_norm == 0] = 1e-8

        X_train_orig = X_train_norm * X_range_norm + X_min_norm

        # 2) Create a 2D mesh in the *original* domain, e.g. [-3, 3]^2
        grid_size = 50
        # Make grid based on range of original training data for better focus
        x1_min_orig, x1_max_orig = X_train_orig[:, 0].min().item()-original_data_range_extension, X_train_orig[:, 0].max().item()+original_data_range_extension
        x2_min_orig, x2_max_orig = X_train_orig[:, 1].min().item()-original_data_range_extension, X_train_orig[:, 1].max().item()+original_data_range_extension

        grid_x1_orig = torch.linspace(x1_min_orig, x1_max_orig, grid_size)
        grid_x2_orig = torch.linspace(x2_min_orig, x2_max_orig, grid_size)
        grid_X1_orig, grid_X2_orig = torch.meshgrid(grid_x1_orig, grid_x2_orig, indexing='ij')
        
        # Flatten
        grid_points_orig = torch.stack([grid_X1_orig.flatten(), grid_X2_orig.flatten()], dim=1)

        # 3) Normalize that grid for GP prediction
        grid_points_norm = (grid_points_orig - X_min_norm) / X_range_norm
            
        # 4) Make predictions in normalized Y space
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions_norm = model.likelihood(model(grid_points_norm))
            mean_norm = predictions_norm.mean
        
        # 5) Un-normalize Y predictions
        y_mean_val = y_norm_params['y_mean']
        y_std_val = y_norm_params['y_std']
        mean_orig = mean_norm * y_std_val + y_mean_val
        mean_orig_reshaped = mean_orig.reshape(grid_size, grid_size)

        # True function values on the grid (original domain)
        Z_true_orig = function_to_plot(grid_points_orig).reshape(grid_size, grid_size)

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # 6) plot
        cp1 = axs[0].contourf(grid_X1_orig.numpy(), grid_X2_orig.numpy(), Z_true_orig.numpy(), 50, cmap='viridis')
        fig.colorbar(cp1, ax=axs[0], label='True Y Value (Original Scale)')
        axs[0].scatter(X_train_orig[:, 0].numpy(), X_train_orig[:, 1].numpy(), color='r', s=25, label="Training Points", edgecolors='k', zorder=2)
        axs[0].set_title(f'True {function_to_plot.__name__} (2D)')
        axs[0].set_xlabel('X1 (Original Domain)')
        axs[0].set_ylabel('X2 (Original Domain)')
        axs[0].legend()
        axs[0].grid(True, linestyle='--', alpha=0.7)

        cp2 = axs[1].contourf(grid_X1_orig.numpy(), grid_X2_orig.numpy(), mean_orig_reshaped.numpy(), 50, cmap='viridis')
        fig.colorbar(cp2, ax=axs[1], label='Predicted Y Value (Original Scale)')
        axs[1].scatter(X_train_orig[:, 0].numpy(), X_train_orig[:, 1].numpy(), color='r', s=25, label="Training Points", edgecolors='k', zorder=2)
        axs[1].set_title(f'GP Predicted Mean for {function_to_plot.__name__} (2D)')
        axs[1].set_xlabel('X1 (Original Domain)')
        axs[1].set_ylabel('X2 (Original Domain)')
        axs[1].legend()
        axs[1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()



        # 6) Plot contour of predictions in original X domain, Y predictions in original domain
        # plt.figure(figsize=(10, 8))
        # plt.contourf(grid_X1_orig.numpy(), grid_X2_orig.numpy(), mean_orig_reshaped.numpy(), 50, cmap='viridis')
        # plt.colorbar(label='Predicted Y Value (Original Scale)')
        
        # # 7) Plot training points (original X, original Y)
        # plt.scatter(
        #     X_train_orig[:, 0].numpy(), 
        #     X_train_orig[:, 1].numpy(), 
        #     c=y_train_orig.numpy(), cmap='coolwarm', s=50, edgecolors='k', 
        #     label='Training Data (Original Y Scale)', zorder=2
        # )
        
        # plt.title('2D GP Regression (Normalized Y Training)')
        # plt.xlabel('X1 (Original Domain)')
        # plt.ylabel('X2 (Original Domain)')
        # plt.legend()
        # plt.grid(True, linestyle='--', alpha=0.7)
        # plt.show()