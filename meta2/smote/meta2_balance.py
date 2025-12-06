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
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.progress import progress_bar
from meta2.splits.meta2_prepare import build_meta2

ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"
FILE = DATA_PROCESSED / 'meta2_features.csv'


def _load_meta2_arrays(path: str | Path):
    data = np.loadtxt(path, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError("meta2 dataset must include participant and label columns.")
    labels = data[:, -1].astype(int)
    participants = data[:, -2].astype(int)
    features = data[:, :-2] if data.shape[1] > 2 else np.empty((data.shape[0], 0))
    return features, labels, participants


def analyze(path: str = FILE, activities=range(1, 8)):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    _, labels, participants = _load_meta2_arrays(path)
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
    part_counts = Counter(participants)
    print("\nSamples per participant:")
    for pid, count in sorted(part_counts.items()):
        print(f"  Participant {pid}: {count} windows")
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
    participants: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Generate K synthetic samples for the given activity using SMOTE.

    Returns (X_aug, y_aug, participants_aug). If `participants` is None,
    the third value is None as well.
    """
    if K <= 0:
        return X.copy(), y.copy(), participants.copy() if participants is not None else None
    rng = np.random.default_rng(random_state)

    if participants is not None and len(participants) != len(y):
        raise ValueError("Participants array must align with y labels.")

    mask = (y == activity_label)
    X_min = X[mask]
    n_min = X_min.shape[0]
    if n_min < 2:
        raise ValueError("Need at least 2 minority samples for SMOTE.")

    participants_min = participants[mask] if participants is not None else None

    k = min(k_neighbors, n_min - 1)
    if k < 1:
        raise ValueError("Not enough samples for the requested k_neighbors.")

    D = _pairwise_distances(X_min)
    neigh_indices = []
    for i in range(n_min):
        idx = np.argsort(D[i])[1 : k + 1]
        neigh_indices.append(idx)

    synth = []
    synth_participants: list[int] = []
    for smote_idx in range(K):
        progress_bar(smote_idx, K, label="Generating SMOTE samples")
        
        i = rng.integers(0, n_min)
        neighs = neigh_indices[i]
        j = rng.choice(neighs)
        xi = X_min[i]
        xj = X_min[j]
        gap = rng.random()
        x_new = xi + gap * (xj - xi)
        synth.append(x_new)
        if participants_min is not None:
            synth_participants.append(int(participants_min[i]))

    X_new = np.vstack([X] + ([np.vstack(synth)] if synth else []))
    y_new = np.concatenate([y, np.full(len(synth), activity_label, dtype=y.dtype)])
    participants_new = None
    if participants is not None:
        if synth_participants:
            new_parts = np.array(synth_participants, dtype=participants.dtype)
        else:
            new_parts = np.empty(0, dtype=participants.dtype)
        participants_new = np.concatenate([participants, new_parts])
    return X_new, y_new, participants_new


def run(
    atividade_para_augment: int = 3,
    K: int = 50,
    k_neighbors: int = 5,
    random_state: int | None = 42,
    meta2_path: str | Path = FILE,
    out_path: str | Path = DATA_PROCESSED / 'meta2_features_aug.csv',
    ensure_meta2: bool = True,
    participant_filter: int | None = None,
) -> str:
    """High-level operation: ensure meta2 exists, analyze and optionally SMOTE.

    Returns path to saved output (either original meta2 or augmented CSV).
    """
    build_meta2()
    
    if not os.path.exists(meta2_path):
        raise FileNotFoundError(meta2_path)

    X_full, y_full, participants = _load_meta2_arrays(meta2_path)

    mask = np.ones_like(y_full, dtype=bool)
    if participant_filter is not None:
        mask &= (participants == participant_filter)
        if not np.any(mask):
            raise ValueError(f"Participant {participant_filter} not found in dataset {meta2_path}.")

    X = X_full[mask]
    y = y_full[mask]
    participants_subset = participants[mask]

    counts = analyze(meta2_path)

    if np.sum(y == atividade_para_augment) < 2:
        print(f"[meta2_balance] Not enough samples of activity {atividade_para_augment} for SMOTE.")
        return meta2_path

    try:
        X_aug, y_aug, participants_aug = generate_smote_samples(
            X,
            y,
            atividade_para_augment,
            K=K,
            k_neighbors=k_neighbors,
            random_state=random_state,
            participants=participants_subset,
        )
        if participants_aug is None:
            raise RuntimeError("SMOTE did not return participant IDs; ensure participants array is provided.")

        base = np.column_stack([X_full, participants, y_full])
        n_original_subset = X.shape[0]
        synth_X = X_aug[n_original_subset:]
        synth_y = y_aug[n_original_subset:]
        synth_participants = participants_aug[n_original_subset:]

        if synth_X.size == 0:
            augmented = base
            print(f"[meta2_balance] No synthetic samples generated (K={K}).")
        else:
            synth_rows = np.column_stack([synth_X, synth_participants, synth_y])
            augmented = np.vstack([base, synth_rows])
        np.savetxt(out_path, augmented, delimiter=',', fmt='%.10e')
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
    force_recompute: bool = False,
    features_dir: str | Path | None = None,
):
    """Generate K SMOTE samples for `activity` using ONLY data from `participante`.

    Also creates a 2D scatter plot using the first two features, colors points by
    activity and highlights the synthetic samples.

    Saves plot to `out_plot` (if provided) or `meta2_part{participant}_act{activity}.png`.
    """
    from meta1.preprocessing import data_treatment
    from meta1.features import feature_extractor as fe
    import matplotlib.pyplot as plt

    allowed_set = set(allowed_activities)
    features_dir = Path(features_dir) if features_dir else DATA_PROCESSED

    X = None
    y = None

    if not force_recompute:
        try:
            X_all = np.loadtxt(features_dir / "features_X.csv", delimiter=',')
            if X_all.ndim == 1:
                X_all = X_all.reshape(1, -1)
            y_all = np.loadtxt(features_dir / "features_y.csv", delimiter=',').astype(int)
            participants_all = np.loadtxt(features_dir / "features_participant.csv", delimiter=',').astype(int)
            if participants_all.ndim > 1:
                participants_all = participants_all.ravel()
            mask = (participants_all == participante) & np.isin(y_all, list(allowed_set))
            if not np.any(mask):
                raise ValueError(
                    f"No precomputed features found for participant {participante} with allowed activities {sorted(allowed_set)}."
                )
            X = X_all[mask]
            y = y_all[mask]
            print(f"[meta2_balance] Using precomputed features from '{features_dir}'.")
        except Exception as e:
            print(f"[meta2_balance] Falling back to on-the-fly extraction (force_recompute=True): {e}")
            force_recompute = True

    if force_recompute or X is None or y is None:
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

        mask_allowed = np.isin(y, list(allowed_set))
        X = X[mask_allowed]
        y = y[mask_allowed]
        if X.size == 0:
            raise RuntimeError(f"No windows remain after filtering allowed activities {sorted(allowed_set)}.")
        print(f"[meta2_balance] Features recomputed for participant {participante} (force_recompute=True).")

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
    X_aug, y_aug, _ = generate_smote_samples(
        X, y, activity, K=K, k_neighbors=5, random_state=random_state, participants=np.full(len(y), participante)
    )
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