import numpy as np
class multi_loss():
    def __init__(self):
        pass

    def adaptive_multi_loss_normalization(self,base_losses, comb_losses):
        """
        Implements Adaptive Multi-Losses Normalization.

        Args:
            base_losses (list or np.array): List of base loss values for each batch.
            comb_losses (list or np.array): List of combined loss values for each batch.

        Returns:
            total_loss (float): The final computed total loss.
            w_comb (float): Adjusted weight for the combined loss.
            b_comb (float): Adjusted bias for the combined loss.

        Raises:
            ValueError: If the standard deviation of comb_losses (sigma_comb) is zero.
        """

        # Convert lists to numpy arrays
        base_losses = np.array(base_losses)
        comb_losses = np.array(comb_losses)
        
        # Compute mean and standard deviation
        mu_base = np.mean(base_losses)
        sigma_base = np.std(base_losses, ddof=0)  # Population std deviation
        
        mu_comb = np.mean(comb_losses)
        sigma_comb = np.std(comb_losses, ddof=0)  # Population std deviation

        # Raise an error if sigma_comb is zero
        if sigma_comb == 0:
            raise ValueError("Standard deviation of comb_losses (σ_comb) is zero, cannot normalize.")

        # Normalize the combined loss
        w_comb = sigma_base / sigma_comb
        b_comb = mu_base - w_comb * mu_comb

        # Compute total loss
        total_loss = 0.5 * (np.mean(base_losses) + np.mean(w_comb * comb_losses + b_comb))

        return total_loss, w_comb, b_comb



