# mmd_test.py
import numpy as np
import torch
from scipy.spatial import distance
from torch_two_sample import MMDStatistic


def mmd_test(X_src, X_tgt, iterations=None):
    """
    Paper-style MMD test (using torch_two_sample).
    Matches ShiftTester.multi_dimensional_test() in the paper code.

    Args:
        X_src (np.ndarray): Source domain features, shape (N, D)
        X_tgt (np.ndarray): Target domain features, shape (M, D)

    Returns:
        tuple: (mmd_statistic, p_value)
    """

    # Convert to float32 numpy arrays
    X_src = X_src.astype(np.float32)
    X_tgt = X_tgt.astype(np.float32)

    # Compute median pairwise distance as kernel bandwidth heuristic
    all_dist = distance.cdist(X_src, X_tgt, "euclidean")
    median_dist = np.median(all_dist)
    alpha = 1 / (median_dist + 1e-8)  # same as paper code (not 1/(2σ²))

    # Convert to torch Variables
    X_src_t = torch.autograd.Variable(torch.tensor(X_src))
    X_tgt_t = torch.autograd.Variable(torch.tensor(X_tgt))

    # Create MMD test object
    mmd = MMDStatistic(len(X_src_t), len(X_tgt_t))

    # Compute MMD statistic and kernel matrix
    t_stat, kernel_matrix = mmd(X_src_t, X_tgt_t, alphas=[alpha], ret_matrix=True)

    if iterations == 0:
        # If no permutation test, return statistic and None for p-value
        return t_stat.item(), None

    # Compute p-value via permutation test
    p_value = mmd.pval(kernel_matrix, n_permutations=iterations)

    # Return test statistic
    return t_stat.item(), p_value


if __name__ == "__main__":
    # Toy data: source ~ N(0,1), target ~ N(1,1)
    X_src = np.random.randn(200, 128)
    X_tgt = np.random.randn(200, 128) + 1.0

    mmd_val = mmd_test(X_src, X_tgt)
    print(f"MMD statistic: {mmd_val:.4f}")
