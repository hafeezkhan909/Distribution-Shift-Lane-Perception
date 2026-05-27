import numpy as np
from scipy.stats import ks_2samp


def bks_distance_test(X_src, X_tgt, alpha=0.05):
    """
    BKS Test that returns both a p-value and a 'BKS Distance'.
    """
    n_dimensions = X_src.shape[1]
    ks_stats = []
    p_values = []

    for i in range(n_dimensions):
        # d_stat is the max distance between CDFs for this dimension
        d_stat, p_val = ks_2samp(X_src[:, i], X_tgt[:, i])
        ks_stats.append(d_stat)
        p_values.append(p_val)

    # 1. The BKS Distance (The maximum shift found in any dimension)
    # Range: [0, 1]
    bks_dist = np.max(ks_stats)

    # 2. The Statistical significance
    min_p_val = np.min(p_values)
    corrected_p_val = min(1.0, min_p_val * n_dimensions)

    return bks_dist, corrected_p_val
