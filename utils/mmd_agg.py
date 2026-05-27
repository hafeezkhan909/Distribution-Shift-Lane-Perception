import numpy as np
from scipy.spatial import distance
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from functools import partial

# ==========================================
# JAX Compiled Core Functions (High Speed)
# ==========================================


@jit
def compute_pairwise_sq_dist(X, Y):
    """Computes pairwise squared Euclidean distances efficiently."""
    X_sq = jnp.sum(X**2, axis=1, keepdims=True)
    Y_sq = jnp.sum(Y**2, axis=1)
    return X_sq + Y_sq - 2 * jnp.dot(X, Y.T)


# Tell JIT that n1 and n2 dictate array sizes and will not change
@partial(jit, static_argnames=["n1", "n2"])
def get_mmd_from_K(K, idx_X, idx_Y, n1, n2):
    """Calculates the unbiased MMD estimator from a pooled kernel matrix."""
    # JAX friendly meshgrid-like indexing
    K_XX = K[idx_X[:, None], idx_X]
    K_YY = K[idx_Y[:, None], idx_Y]
    K_XY = K[idx_X[:, None], idx_Y]

    sum_XX = (jnp.sum(K_XX) - jnp.trace(K_XX)) / (n1 * (n1 - 1))
    sum_YY = (jnp.sum(K_YY) - jnp.trace(K_YY)) / (n2 * (n2 - 1))
    sum_XY = jnp.sum(K_XY) / (n1 * n2)
    return sum_XX + sum_YY - 2 * sum_XY


@partial(jit, static_argnames=["n1", "n2"])
def all_alphas_mmd(D_ZZ, idx_X, idx_Y, n1, n2, alphas):
    """Calculates MMD across all kernel bandwidths simultaneously."""

    def single_alpha(alpha):
        K = jnp.exp(-alpha * D_ZZ)
        return get_mmd_from_K(K, idx_X, idx_Y, n1, n2)

    return vmap(single_alpha)(alphas)


@partial(jit, static_argnames=["n1", "n2"])
def run_all_permutations(D_ZZ, perms, n1, n2, alphas):
    """Runs the MMD calculation for all shuffles across all kernels."""

    def single_perm(idx):
        idx_X = idx[:n1]  # JAX now knows n1 is a static integer
        idx_Y = idx[n1:]
        return all_alphas_mmd(D_ZZ, idx_X, idx_Y, n1, n2, alphas)

    return vmap(single_perm)(perms)


# ==========================================
# Backward Compatible Wrapper
# ==========================================


def mmdAgg_test(X_src, X_tgt, iterations=None, seed=42):
    """
    MMDAgg Two-Sample Test using JAX.
    Matches the original mmdAgg_test() signature perfectly.

    Args:
        X_src (np.ndarray): Source domain features, shape (N, D)
        X_tgt (np.ndarray): Target domain features, shape (M, D)
        iterations (int): Number of permutations (B). None/0 disables p-value.
        seed (int): Global seed for deterministic permutations.

    Returns:
        tuple: (base_mmd_statistic, aggregated_p_value)
    """
    # 1. Convert to numpy float32
    X_src = np.asarray(X_src, dtype=np.float32)
    X_tgt = np.asarray(X_tgt, dtype=np.float32)

    n1 = len(X_src)
    n2 = len(X_tgt)

    # 2. Compute median pairwise distance as baseline bandwidth
    all_dist = distance.cdist(X_src, X_tgt, "euclidean")
    median_dist = np.median(all_dist)
    base_alpha = 1 / (median_dist + 1e-8)

    # 3. MMDAgg: Define the Set of Kernels (Lambda)
    # We use a spread of 5 bandwidths around the median to catch all shift types
    multipliers = np.array([0.25, 0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    alphas = jnp.array(base_alpha * multipliers)

    # 4. Pool data and compute distance matrix once
    Z = jnp.vstack([X_src, X_tgt])
    D_ZZ = compute_pairwise_sq_dist(Z, Z)

    # 5. Compute original MMD for all kernels
    orig_idx_X = jnp.arange(n1)
    orig_idx_Y = jnp.arange(n1, n1 + n2)
    original_mmds = all_alphas_mmd(D_ZZ, orig_idx_X, orig_idx_Y, n1, n2, alphas)

    # Return the exact statistic your old code would have returned (multiplier 1.0)
    base_mmd_stat = original_mmds[2].item()

    # Fast exit if no p-value requested
    if iterations is None or iterations == 0:
        return base_mmd_stat, None

    # 6. Generate independent permutations efficiently
    rng_key = random.PRNGKey(seed)
    keys = random.split(rng_key, iterations)
    perms = vmap(lambda k: random.permutation(k, jnp.arange(n1 + n2)))(keys)

    # 7. Run Permutation Test
    # perm_mmds shape -> (iterations, 5)
    perm_mmds = run_all_permutations(D_ZZ, perms, n1, n2, alphas)

    # 8. The Calibration/Aggregation Step
    # We standardize the statistics to find the kernel that detects the largest
    # relative shift, adjusting for the noise floor of each specific kernel.
    mean_mmds = jnp.mean(perm_mmds, axis=0)
    std_mmds = jnp.std(perm_mmds, axis=0) + 1e-8

    std_original = (original_mmds - mean_mmds) / std_mmds
    std_permuted = (perm_mmds - mean_mmds) / std_mmds

    agg_stat_original = jnp.max(std_original)
    agg_stat_permuted = jnp.max(std_permuted, axis=1)

    # Calculate the aggregated p-value
    p_value = jnp.mean(agg_stat_permuted >= agg_stat_original).item()

    return base_mmd_stat, p_value


if __name__ == "__main__":
    np.random.seed(123)  # Seed for reproducible toy data generation

    n_samples = 300
    n_dims = 128
    iterations = 100

    print("==================================================")
    print(" Running MMDAgg JAX Test Suite")
    print("==================================================\n")

    # ---------------------------------------------------------
    # Test 1: Identical Distributions (No Shift)
    # Expected: High p-value (> 0.05). Should NOT reject the null hypothesis.
    # ---------------------------------------------------------
    X_src_1 = np.random.randn(n_samples, n_dims)
    X_tgt_1 = np.random.randn(n_samples, n_dims)

    mmd_val_1, p_val_1 = mmdAgg_test(X_src_1, X_tgt_1, iterations=iterations)
    print("Test 1: Identical Distributions (No Shift)")
    print(f"  -> Base MMD : {mmd_val_1:.6f}")
    print(f"  -> P-Value  : {p_val_1:.4f} (Expected: > 0.05)\n")

    # ---------------------------------------------------------
    # Test 2: Global Mean Shift (Translation)
    # Expected: Low p-value (< 0.05). Every dimension shifts slightly.
    # ---------------------------------------------------------
    X_src_2 = np.random.randn(n_samples, n_dims)
    X_tgt_2 = np.random.randn(n_samples, n_dims) + 0.15  # Subtle global shift

    mmd_val_2, p_val_2 = mmdAgg_test(X_src_2, X_tgt_2, iterations=iterations)
    print("Test 2: Global Mean Shift (Translation)")
    print(f"  -> Base MMD : {mmd_val_2:.6f}")
    print(f"  -> P-Value  : {p_val_2:.4f} (Expected: < 0.05)\n")

    # ---------------------------------------------------------
    # Test 3: Variance Shift (Scaling)
    # Expected: Low p-value (< 0.05). Target is "spread out" more.
    # ---------------------------------------------------------
    X_src_3 = np.random.randn(n_samples, n_dims)
    X_tgt_3 = np.random.randn(n_samples, n_dims) * 1.2  # 20% more variance

    mmd_val_3, p_val_3 = mmdAgg_test(X_src_3, X_tgt_3, iterations=iterations)
    print("Test 3: Variance Shift (Scaling)")
    print(f"  -> Base MMD : {mmd_val_3:.6f}")
    print(f"  -> P-Value  : {p_val_3:.4f} (Expected: < 0.05)\n")

    # ---------------------------------------------------------
    # Test 4: Single Dimension "Malignant" Shift
    # Expected: Low p-value (< 0.05). 127 dimensions are identical, 1 shifts.
    # This is where single-kernel MMD often fails but MMDAgg succeeds.
    # ---------------------------------------------------------
    X_src_4 = np.random.randn(n_samples, n_dims)
    X_tgt_4 = np.random.randn(n_samples, n_dims)
    X_tgt_4[:, 42] += 2.0  # Massive shift in just ONE dimension

    mmd_val_4, p_val_4 = mmdAgg_test(X_src_4, X_tgt_4, iterations=iterations)
    print("Test 4: Single Dimension 'Malignant' Shift")
    print(f"  -> Base MMD : {mmd_val_4:.6f}")
    print(f"  -> P-Value  : {p_val_4:.4f} (Expected: < 0.05)\n")

    print("==================================================")
    print(" Tests Complete.")
    print("==================================================")
