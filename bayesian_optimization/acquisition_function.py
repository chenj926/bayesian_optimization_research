import torch
import gpytorch

class AcquisitionFunctionUtils:
    @staticmethod
    def upper_confidence_bound(model, likelihood, X_candidates_norm, kappa=1.96):
        """
        Calculates the Upper Confidence Bound (UCB) for candidate points.

        Args:
            model (gpytorch.models.ExactGP): The trained GP model.
            likelihood (gpytorch.likelihoods.GaussianLikelihood): The GP likelihood.
            X_candidates_norm (torch.Tensor): Normalized candidate points (N_candidates, dim).
            kappa (float): Parameter to balance exploration and exploitation.

        Returns:
            torch.Tensor: UCB values for each candidate point.
        """
        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get predictive distribution for candidates
            posterior = model(X_candidates_norm)
            mean_norm = posterior.mean
            stddev_norm = posterior.stddev

        ucb_values = mean_norm + kappa * stddev_norm
        return ucb_values

    # LCB for later, EI, UCTS?
