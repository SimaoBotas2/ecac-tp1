"""SMOTE implementation for a single target activity.

Provides a lightweight SMOTE-style augmentation without external deps and can
optionally propagate participant IDs plus progress callbacks for UI feedback.

Function: ``generate_smote_samples(X, y, activity_label, K, ...)``

Parameters
----------
X : np.ndarray
        Feature matrix (n_samples, n_features).
y : np.ndarray
        Label vector (n_samples,).
activity_label : int
        Class to augment.
K : int
        Number of synthetic samples to create.
k_neighbors : int, optional
        Neighbours used during interpolation (default 5).
participants : np.ndarray | None, optional
        Participant IDs aligned with ``y``. When supplied, return value includes the
        augmented participant vector; generated samples inherit the participant of
        the seed window.
progress_callback : Callable[[int, int], None] | None, optional
        Callback executed once per synthetic sample with ``(current, total)``.

Returns
-------
Tuple[np.ndarray, np.ndarray, np.ndarray | None]
        ``(X_aug, y_aug, participants_aug_or_None)``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

def _pairwise_distances(A: np.ndarray) -> np.ndarray: #versao super otimizada de distancia euclidiana
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
    random_state: int | None = None,
    participants: np.ndarray | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Generate K synthetic samples for the given activity using SMOTE."""
    if K <= 0:
        participants_copy = (
            participants.copy() if participants is not None else None
        )
        return X.copy(), y.copy(), participants_copy
    rng = np.random.default_rng(random_state)

    if participants is not None and participants.shape[0] != y.shape[0]:
        raise ValueError("Participants array must align with labels array.")

    # Filter minority instances
    mask = (y == activity_label)
    X_min = X[mask]
    n_min = X_min.shape[0]
    if n_min < 2:
        raise ValueError("Need at least 2 minority samples for SMOTE.")

    participants_min = participants[mask] if participants is not None else None

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
    synth_participants: list[int] = []
    for idx in range(K):
        if progress_callback is not None:
            progress_callback(idx, K)
        i = rng.integers(0, n_min)
        neighs = neigh_indices[i]
        j = rng.choice(neighs)
        xi = X_min[i]
        xj = X_min[j]
        gap = rng.random()  # interpolation factor
        x_new = xi + gap * (xj - xi)
        synth.append(x_new)
        if participants_min is not None:
            synth_participants.append(int(participants_min[i]))

    X_new = np.vstack([X] + ([np.vstack(synth)] if synth else []))
    y_new = np.concatenate([y, np.full(len(synth), activity_label, dtype=y.dtype)])
    participants_new = None
    if participants is not None:
        new_parts = (
            np.array(synth_participants, dtype=participants.dtype)
            if synth_participants
            else np.empty(0, dtype=participants.dtype)
        )
        participants_new = np.concatenate([participants, new_parts])
    return X_new, y_new, participants_new

if __name__ == '__main__':
    # Minimal self-test: creates 5 synthetic samples for class 3 if present.
    try:
        X = np.loadtxt('meta2_features.csv', delimiter=',')
        if X.ndim == 1: X = X.reshape(1,-1)
        y = X[:, -1].astype(int)
        X_only = X[:, :-1]
        label = 3
        if np.sum(y == label) >= 2:
            X_aug, y_aug, _ = generate_smote_samples(
                X_only, y, label, K=5, random_state=42
            )
            print(f"Synthetic samples generated for activity {label}: {X_aug.shape[0]-X_only.shape[0]}")
        else:
            print(f"Not enough samples of activity {label} for SMOTE test.")
    except Exception as e:
        print(f"Self-test failed: {e}")
