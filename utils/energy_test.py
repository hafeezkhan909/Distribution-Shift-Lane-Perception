import numpy as np
import torch
from torch_two_sample import EnergyStatistic


def energy_test(X_src, X_tgt, iterations=1000):
    """
    Energy Statistic test for domain shift detection.
    Replaces MMD for more robust distance measurement in latent spaces.

    Args:
        X_src (np.ndarray): Source domain features, shape (N, D)
        X_tgt (np.ndarray): Target domain features, shape (M, D)
        iterations (int): Number of permutations for p-value. 0 for stat only.

    Returns:
        tuple: (energy_statistic, p_value)
    """

    # 1. Preprocessing: Ensure float32 and Torch Tensors
    X_src_t = torch.from_numpy(X_src.astype(np.float32))
    X_tgt_t = torch.from_numpy(X_tgt.astype(np.float32))

    # 2. Initialize EnergyStatistic
    # Unlike MMD, EnergyStatistic does not require an 'alpha' bandwidth
    energy = EnergyStatistic(len(X_src_t), len(X_tgt_t))

    # 3. Compute Energy Statistic
    # ret_matrix=True returns the pairwise distance matrix needed for p-values
    t_stat, dist_matrix = energy(X_src_t, X_tgt_t, ret_matrix=True)

    if iterations == 0 or iterations is None:
        return t_stat.item(), None

    # 4. Compute p-value via permutation test
    p_value = energy.pval(dist_matrix, n_permutations=iterations)

    return t_stat.item(), p_value


if __name__ == "__main__":
    # Toy data: High-dimensional latent embeddings
    # Source ~ N(0,1), Target ~ N(0.1, 1.2) - a subtle mean and variance shift
    print("Running Energy Statistic Test on Toy Data...")
    X_src = np.random.randn(200, 128)
    X_tgt = (np.random.randn(200, 128) * 1.2) + 0.1

    e_stat, p_val = energy_test(X_src, X_tgt, iterations=500)

    print(f"Energy Statistic: {e_stat:.4f}")
    print(f"P-value: {p_val:.4f}")

    if p_val < 0.05:
        print("Result: Significant domain shift detected.")
    else:
        print("Result: No significant shift detected.")
