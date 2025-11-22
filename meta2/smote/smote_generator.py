"""SMOTE implementation for a single target activity.

Provides a lightweight SMOTE-style augmentation without external deps.

Function: generate_smote_samples(X, y, activity_label, K, k_neighbors=5)

Parameters:
  X : np.ndarray (n_samples, n_features)
  y : np.ndarray (n_samples,) labels (ints)
  activity_label : int -> the minority class to augment
  K : int -> number of synthetic samples to create
  k_neighbors : int -> neighbors used to interpolate (default 5)

Returns:
  X_aug, y_aug : arrays with the new synthetic samples appended.

Behavior:
  - If fewer than 2 samples of the class exist, raises ValueError.
  - For each synthetic sample: pick random minority instance i and one
    random neighbor j among its k nearest minority neighbors; create
    new = X_i + rand(0,1)*(X_j - X_i).
  - Feature space assumed numeric & continuous.

Note: This is a simplified adaptation; it does not guard against creating
samples that cross decision boundaries or leave the data manifold. For
high-dimensional / sensitive domains, validate outputs.
"""

from __future__ import annotations
import numpy as np

def _pairwise_distances(A: np.ndarray) -> np.ndarray:
    """Compute full pairwise Euclidean distance matrix for rows in A."""
    # (a-b)^2 = a^2 + b^2 - 2ab
    sq = np.sum(A*A, axis=1, keepdims=True)
    d2 = sq + sq.T - 2 * (A @ A.T)
    # Correct potential tiny negatives from floating point
    np.maximum(d2, 0, out=d2)
    return np.sqrt(d2)

def generate_smote_samples(
    X: np.ndarray,
    y: np.ndarray,
    activity_label: int,
    K: int,
    k_neighbors: int = 5,
    random_state: int | None = None
):
    """Generate K synthetic samples for the given activity using SMOTE.

    Returns (X_aug, y_aug).
    """
    if K <= 0:
        return X.copy(), y.copy()
    rng = np.random.default_rng(random_state)

    # Filter minority instances
    mask = (y == activity_label)
    X_min = X[mask]
    n_min = X_min.shape[0]
    if n_min < 2:
        raise ValueError("Need at least 2 minority samples for SMOTE.")

    k = min(k_neighbors, n_min - 1)
    if k < 1:
        raise ValueError("Not enough samples for the requested k_neighbors.")

    # Distances among minority samples
    D = _pairwise_distances(X_min)
    # For each sample, get indices of k nearest (exclude self)
    neigh_indices = []
    for i in range(n_min):
        # argsort distances; skip itself at position 0
        idx = np.argsort(D[i])[1:k+1]
        neigh_indices.append(idx)

    synth = []
    for _ in range(K):
        i = rng.integers(0, n_min)
        neighs = neigh_indices[i]
        j = rng.choice(neighs)
        xi = X_min[i]
        xj = X_min[j]
        gap = rng.random()  # interpolation factor
        x_new = xi + gap * (xj - xi)
        synth.append(x_new)

    X_new = np.vstack([X] + ([np.vstack(synth)] if synth else []))
    y_new = np.concatenate([y, np.full(len(synth), activity_label, dtype=y.dtype)])
    return X_new, y_new

if __name__ == '__main__':
    # Minimal self-test: creates 5 synthetic samples for class 3 if present.
    try:
        X = np.loadtxt('meta2_features.csv', delimiter=',')
        if X.ndim == 1: X = X.reshape(1,-1)
        y = X[:, -1].astype(int)
        X_only = X[:, :-1]
        label = 3
        if np.sum(y == label) >= 2:
            X_aug, y_aug = generate_smote_samples(X_only, y, label, K=5, random_state=42)
            print(f"Synthetic samples generated for activity {label}: {X_aug.shape[0]-X_only.shape[0]}")
        else:
            print(f"Not enough samples of activity {label} for SMOTE test.")
    except Exception as e:
        print(f"Self-test failed: {e}")
