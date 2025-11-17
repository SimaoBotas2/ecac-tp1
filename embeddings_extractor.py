"""Embeddings extractor using ssl-wearables harnet models.

- Uses only accelerometer (x,y,z)
- 5-second windows with 50% overlap (same as features flow)
- Resamples each window to 30 Hz (length=150) and passes through harnet5
- Returns an embeddings dataset of shape [n_segments, n_embeddings]

Notes:
- Requires torch and internet access on first run to fetch model via torch.hub.
- Filters labels to allowed_activities (default 1..7).
"""
from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np
import torch

# Torch hub repo
_REPO = 'OxWearables/ssl-wearables'


def _load_harnet5(device: str = 'cpu') -> torch.nn.Module:
    """Load the harnet5 model and return its feature extractor module.

    Assumes model exposes a 'feature_extractor' attribute as per the project docs.
    """
    model = torch.hub.load(_REPO, 'harnet5', class_num=5, pretrained=True)  # class_num arbitrary here
    model.eval()
    model.to(device)
    # Prefer documented attribute
    if hasattr(model, 'feature_extractor'):
        feature_extractor = getattr(model, 'feature_extractor')
        feature_extractor.eval()
        feature_extractor.to(device)
        return feature_extractor
    # Fallback: try to remove classifier if present
    # This may fail if the internal API is different; prefer feature_extractor above.
    children = list(model.children())
    if len(children) >= 2:
        feature_extractor = torch.nn.Sequential(*children[:-1])
        feature_extractor.eval()
        feature_extractor.to(device)
        return feature_extractor
    raise AttributeError("Could not locate feature_extractor in harnet5 model.")


def _resample_to_30hz(window_xyz: np.ndarray, orig_sr: int | float, target_sr: int = 30) -> np.ndarray:
    """Linearly resample a (T,3) window from orig_sr to target_sr for 5 seconds.

    Returns shape (3, 150) suitable for (C,T) with C=3.
    """
    T = window_xyz.shape[0]
    duration = T / float(orig_sr)
    target_len = int(round(duration * target_sr))
    # For 5 seconds at 30Hz, expect 150
    t_src = np.linspace(0.0, duration, num=T, endpoint=False)
    t_dst = np.linspace(0.0, duration, num=target_len, endpoint=False)
    out = np.empty((3, target_len), dtype=np.float32)
    for ch in range(3):
        out[ch] = np.interp(t_dst, t_src, window_xyz[:, ch].astype(np.float32))
    return out


def extract_embeddings_dataset(
    accel_data: np.ndarray,
    activities: np.ndarray,
    sampling_rate: int | float,
    window_size_sec: int | float = 5,
    overlap: float = 0.5,
    allowed_activities: Iterable[int] = range(1, 8),
    batch_size: int = 64,
    device: str = 'cpu',
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract embeddings for 5s windows (resampled to 30Hz) using harnet5.

    Parameters
    - accel_data: array (N,3)
    - activities: array (N,) integer labels aligned with accel_data
    - sampling_rate: original sampling rate (e.g., 50)
    - allowed_activities: only keep labels in this set (default 1..7)

    Returns
    - X_emb: np.ndarray (n_windows, n_embeddings)
    - y_win: np.ndarray (n_windows,) labels per window
    """
    accel_data = np.asarray(accel_data, dtype=np.float32)
    y_all = np.asarray(activities).astype(int)
    sr = float(sampling_rate)

    win_len = int(round(window_size_sec * sr))
    step = int(round(win_len * (1.0 - overlap)))
    if step <= 0:
        step = 1

    # Collect resampled windows and labels
    resampled_batches: list[np.ndarray] = []
    labels: list[int] = []

    # Slide over windows
    for start in range(0, accel_data.shape[0] - win_len + 1, step):
        end = start + win_len
        label_win = y_all[start:end]
        uniq = np.unique(label_win)
        if len(uniq) != 1:
            continue  # skip mixed-activity windows
        label = int(uniq[0])
        if label not in set(allowed_activities):
            continue
        win = accel_data[start:end, :]  # (T,3)
        rs = _resample_to_30hz(win, orig_sr=sr, target_sr=30)  # (3,150)
        resampled_batches.append(rs)
        labels.append(label)

    if not resampled_batches:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=int)

    # Stack to (N,3,150)
    X_rs = np.stack(resampled_batches, axis=0)
    y_win = np.asarray(labels, dtype=int)

    # Load feature extractor
    feat_extractor = _load_harnet5(device=device)

    # Infer embedding dimension with a single forward
    with torch.no_grad():
        probe = torch.from_numpy(X_rs[:1]).to(device)  # (1,3,150)
        emb_probe = feat_extractor(probe)
        emb_dim = int(np.prod(list(emb_probe.shape)[1:]))

    # Forward in batches
    X_emb = np.empty((X_rs.shape[0], emb_dim), dtype=np.float32)
    idx = 0
    with torch.no_grad():
        for i in range(0, X_rs.shape[0], batch_size):
            batch = torch.from_numpy(X_rs[i:i+batch_size]).to(device)
            feats = feat_extractor(batch)
            feats = torch.flatten(feats, start_dim=1)
            n = feats.shape[0]
            X_emb[idx:idx+n] = feats.cpu().numpy().astype(np.float32)
            idx += n

    return X_emb, y_win


def save_embeddings_to_csv(X_emb: np.ndarray, y: np.ndarray, base_path: str = '.') -> Tuple[str, str]:
    """Save embeddings and labels to CSV files; returns paths."""
    import os
    x_path = os.path.join(base_path, 'embeddings_X.csv')
    y_path = os.path.join(base_path, 'embeddings_y.csv')
    if X_emb.size > 0:
        np.savetxt(x_path, X_emb, delimiter=',', fmt='%.6f')
        np.savetxt(y_path, y, delimiter=',', fmt='%d')
    return x_path, y_path
