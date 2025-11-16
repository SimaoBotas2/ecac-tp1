"""Meta2 balance analysis + SMOTE helper.

This module merges the previous balance analysis and the SMOTE generator so
`mainActivity.py` can simply call `run_meta2(...)` to perform the full
operation: ensure `meta2_features.csv` exists (via `meta2_prepare`), show
distribution for activities 1..7, optionally augment a chosen activity with
SMOTE, and save an augmented CSV.

Functions:
 - analyze(path, activities): print simple class distribution for activities
 - generate_smote_samples(...): lightweight SMOTE implementation
 - run_meta2(...): high-level entrypoint used by `mainActivity.py`
"""

from collections import Counter
import numpy as np
import os
from typing import Tuple

FILE = 'meta2_features.csv'


def analyze(path: str = FILE, activities=range(1, 8)):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.loadtxt(path, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    labels = data[:, -1].astype(int)
    counts = Counter(labels)
    total = sum(counts[a] for a in activities if a in counts)
    print("Class distribution (activities 1-7):")
    for a in activities:
        c = counts.get(a, 0)
        pct = (c / total * 100) if total else 0
        print(f"  Activity {a}: {c} samples ({pct:.2f}%)")
    if total == 0:
        print("No samples found for requested activities.")
        return counts
    max_c = max(counts.get(a, 0) for a in activities)
    min_c = min(counts.get(a, 0) for a in activities)
    ratio = max_c / (min_c or 1)
    print(f"\nImbalance ratio (max/min): {ratio:.2f}")
    if ratio <= 1.5:
        print("Dataset roughly balanced (<=1.5x).")
    elif ratio <= 3:
        print("Moderate imbalance; consider light augmentation / class-weighting.")
    else:
        print("Strong imbalance; SMOTE or re-sampling recommended.")
    return counts


def _pairwise_distances(A: np.ndarray) -> np.ndarray:
    sq = np.sum(A * A, axis=1, keepdims=True)
    d2 = sq + sq.T - 2 * (A @ A.T)
    np.maximum(d2, 0, out=d2)
    return np.sqrt(d2)


def generate_smote_samples(
    X: np.ndarray,
    y: np.ndarray,
    activity_label: int,
    K: int,
    k_neighbors: int = 5,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate K synthetic samples for the given activity using SMOTE.

    Returns (X_aug, y_aug).
    """
    if K <= 0:
        return X.copy(), y.copy()
    rng = np.random.default_rng(random_state)

    mask = (y == activity_label)
    X_min = X[mask]
    n_min = X_min.shape[0]
    if n_min < 2:
        raise ValueError("Need at least 2 minority samples for SMOTE.")

    k = min(k_neighbors, n_min - 1)
    if k < 1:
        raise ValueError("Not enough samples for the requested k_neighbors.")

    D = _pairwise_distances(X_min)
    neigh_indices = []
    for i in range(n_min):
        idx = np.argsort(D[i])[1 : k + 1]
        neigh_indices.append(idx)

    synth = []
    for _ in range(K):
        i = rng.integers(0, n_min)
        neighs = neigh_indices[i]
        j = rng.choice(neighs)
        xi = X_min[i]
        xj = X_min[j]
        gap = rng.random()
        x_new = xi + gap * (xj - xi)
        synth.append(x_new)

    X_new = np.vstack([X] + ([np.vstack(synth)] if synth else []))
    y_new = np.concatenate([y, np.full(len(synth), activity_label, dtype=y.dtype)])
    return X_new, y_new


def run(
    atividade_para_augment: int = 3,
    K: int = 50,
    k_neighbors: int = 5,
    random_state: int | None = 42,
    meta2_path: str = FILE,
    out_path: str = 'meta2_features_aug.csv',
    ensure_meta2: bool = True,
) -> str:
    """High-level operation: ensure meta2 exists, analyze and optionally SMOTE.

    Returns path to saved output (either original meta2 or augmented CSV).
    """
    if ensure_meta2:
        try:
            import meta2_prepare_INUTIL

            if hasattr(meta2_prepare_INUTIL, 'build_meta2'):
                meta2_prepare_INUTIL.build_meta2()
                print(f"[meta2_balance] Ensured '{meta2_path}' exists via meta2_prepare.build_meta2().")
        except Exception as e:
            print(f"[meta2_balance] Warning: could not run meta2_prepare.build_meta2(): {e}")

    if not os.path.exists(meta2_path):
        raise FileNotFoundError(meta2_path)

    data = np.loadtxt(meta2_path, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    y = data[:, -1].astype(int)
    X = data[:, :-1]

    counts = analyze(meta2_path)

    if np.sum(y == atividade_para_augment) < 2:
        print(f"[meta2_balance] Not enough samples of activity {atividade_para_augment} for SMOTE.")
        return meta2_path

    try:
        X_aug, y_aug = generate_smote_samples(
            X, y, atividade_para_augment, K=K, k_neighbors=k_neighbors, random_state=random_state
        )
        augmented = np.column_stack([X_aug, y_aug])
        np.savetxt(out_path, augmented, delimiter=',', fmt='%.6f')
        print(f"[meta2_balance] Saved augmented dataset to '{out_path}' (+{K} samples for activity {atividade_para_augment}).")
        return out_path
    except Exception as e:
        print(f"[meta2_balance] Error during SMOTE augmentation: {e}")
        return meta2_path


if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        print(f"[meta2_balance] Failed: {e}")


def generate_and_visualize_samples_for_participant(
    participante: int,
    activity: int = 4,
    K: int = 3,
    sensors=(1, 2, 3, 4, 5),
    out_plot: str | None = None,
    random_state: int | None = 42,
    allowed_activities=tuple(range(1, 8)),
):
    """Generate K SMOTE samples for `activity` using ONLY data from `participante`.

    Also creates a 2D scatter plot using the first two features, colors points by
    activity and highlights the synthetic samples.

    Saves plot to `out_plot` (if provided) or `meta2_part{participant}_act{activity}.png`.
    """
    try:
        import data_treatment
        import feature_extractor as fe
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"Missing dependency for generation/plot: {e}")

    # 1) Load raw data for the specific participant and sensors
    dados = data_treatment.get_data(participante, list(sensors)) # type: ignore
    if getattr(dados, 'size', 0) == 0:
        raise FileNotFoundError(f"No data found for participant {participante} with sensors {sensors}")

    activities = dados[:, 11].astype(int)
    accel_data = dados[:, 1:4].astype(float)
    gyro_data = dados[:, 4:7].astype(float)
    mag_data = dados[:, 7:10].astype(float)

    sr = fe.sampling_rate_calculator(dados)

    # 2) Extract features for this participant only
    X, y, _ = fe.extract_features_4_2(accel_data, gyro_data, mag_data, activities, sampling_rate=sr)
    if getattr(X, 'size', 0) == 0:
        raise RuntimeError("Feature extraction returned no windows for this participant.")

    # 2.1) Filter to allowed activities (e.g., 1..7)
    allowed_set = set(allowed_activities)
    mask_allowed = np.isin(y, list(allowed_set))
    X = X[mask_allowed]
    y = y[mask_allowed]
    if X.size == 0:
        raise RuntimeError(f"No windows remain after filtering allowed activities {sorted(allowed_set)}.")

    # 3) Check first two features exist
    if X.shape[1] < 2:
        raise RuntimeError("Not enough features to plot first two dimensions.")

    # 4) Ensure there are enough samples of the target activity for SMOTE
    n_target = int((y == activity).sum())
    print(f"[meta2_balance] Participant {participante} - activity {activity}: {n_target} samples available (after filtering).")
    print(f"[meta2_balance] Activities kept: {sorted(np.unique(y))}")
    if n_target < 2:
        raise ValueError(f"Need at least 2 samples of activity {activity} for SMOTE (found {n_target}).")

    # 5) Apply SMOTE (using only this participant's X,y)
    X_aug, y_aug = generate_smote_samples(X, y, activity, K=K, k_neighbors=5, random_state=random_state)
    n_synth = X_aug.shape[0] - X.shape[0]
    print(f"[meta2_balance] Generated {n_synth} synthetic samples for activity {activity} (K={K}).")

    # 6) Plot first two features: originals colored by activity, synthetic highlighted
    synth_idx = np.arange(X.shape[0], X_aug.shape[0])

    unique_acts = np.unique(y_aug)
    import matplotlib.pyplot as plt  # ensure plt is available here
    cmap = plt.get_cmap('tab10')
    color_map = {act: cmap(i % 10) for i, act in enumerate(unique_acts)}

    fig, ax = plt.subplots(figsize=(7, 6))

    # plot original points by activity
    for act in np.unique(y):
        mask = (y == act)
        ax.scatter(X[mask, 0], X[mask, 1], label=f"act {act}", alpha=0.8, s=40, color=color_map.get(act))

    # plot synthetic points (only those with target activity)
    if n_synth > 0:
        synth_mask = (y_aug[synth_idx] == activity)
        if synth_mask.any():
            sx = X_aug[synth_idx][synth_mask, 0]
            sy = X_aug[synth_idx][synth_mask, 1]
            ax.scatter(sx, sy, facecolors='none', edgecolors='k', s=120, linewidths=1.5, label=f"synthetic act {activity}")

    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')
    ax.set_title(f'Participant {participante} - Activity {activity} (SMOTE +{n_synth})')
    ax.legend()
    ax.grid(True)

    if out_plot is None:
        out_plot = f"meta2_part{participante}_act{activity}_plot.png"

    fig.tight_layout()
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)

    print(f"[meta2_balance] Plot saved to '{out_plot}'")
    return {'plot': out_plot, 'n_synthetic': n_synth, 'n_original': X.shape[0]}