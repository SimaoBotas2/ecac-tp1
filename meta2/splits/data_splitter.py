"""Dataset splitting utilities for Meta 2.

Creates train/val/test splits for both features and embeddings datasets
using two strategies:
 - within-subject: 60/20/20 split inside each participant
 - between-subject: 9/3/3 participants in train/val/test

Outputs are saved as CSVs in data/processed/splits with columns
[features..., participant_id, label].
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.progress import progress_bar

ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"
SPLITS_DIR = DATA_PROCESSED / "splits"
ALLOWED_ACTIVITIES = tuple(range(1, 8))

FEATURE_FILES = {
    "X": DATA_PROCESSED / "features_X.csv",
    "y": DATA_PROCESSED / "features_y.csv",
    "p": DATA_PROCESSED / "features_participant.csv",
}

EMBED_FILES = {
    "X": DATA_PROCESSED / "embeddings_X.csv",
    "y": DATA_PROCESSED / "embeddings_y.csv",
    "p": DATA_PROCESSED / "embeddings_participant.csv",
}

_SPLIT_NAMES = ("train", "val", "test")


def _ensure_matrix(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _ensure_vector(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr).astype(float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.astype(int)


def _load_dataset(kind: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = FEATURE_FILES if kind == "features" else EMBED_FILES
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Ficheiros em falta para '{kind}': {missing}")
    X = _ensure_matrix(np.loadtxt(files["X"], delimiter=","))
    y = _ensure_vector(np.loadtxt(files["y"], delimiter=","))
    participants = _ensure_vector(np.loadtxt(files["p"], delimiter=","))
    if X.shape[0] != y.shape[0] or X.shape[0] != participants.shape[0]:
        raise ValueError(f"Dimensões inconsistentes no dataset '{kind}'.")

    mask = np.isin(y, ALLOWED_ACTIVITIES)
    if not mask.all():
        removed = int((~mask).sum())
        print(f"[SPLITS][{kind}] Ignorados {removed} registos fora das atividades 1-7.")
        X = X[mask]
        y = y[mask]
        participants = participants[mask]
    return X, y, participants


def _combine_rows(X: np.ndarray, participants: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    if idx.size == 0:
        return np.empty((0, X.shape[1] + 2))
    return np.column_stack((X[idx], participants[idx], y[idx]))


def _finalize(chunks: list[np.ndarray], n_features: int) -> np.ndarray:
    if not chunks:
        return np.empty((0, n_features + 2))
    return np.vstack(chunks)


def _save_splits(kind: str, strategy: str, splits: Dict[str, np.ndarray]) -> None:
    out_dir = SPLITS_DIR / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, data in splits.items():
        path = out_dir / f"{kind}_{strategy}_{split_name}.csv"
        np.savetxt(path, data, delimiter=",", fmt="%.10e")


def _print_summary(kind: str, strategy: str, splits: Dict[str, np.ndarray]) -> None:
    counts = ", ".join(f"{name}:{data.shape[0]}" for name, data in splits.items())
    print(f"[SPLITS][{kind}] {strategy} -> {counts}")


def _partition_counts(total: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    ratios = np.asarray(ratios, dtype=float)
    if ratios.size != 3 or np.isclose(ratios.sum(), 0):
        raise ValueError("Ratios devem ter três valores positivos.")
    ratios = ratios / ratios.sum()
    raw = ratios * total
    frac = raw - np.floor(raw)
    counts = np.floor(raw).astype(int)
    remainder = total - counts.sum()
    if remainder > 0:
        order = np.argsort(-frac)
        for idx in order[:remainder]:
            counts[idx] += 1
    counts[2] = total - counts[0] - counts[1]
    return int(counts[0]), int(counts[1]), int(counts[2])


def split_within_subject( # Pega num participante da shuffla os indices e faz o split iterativamente por todos os participantes
    kind: str,
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    random_state: int | None = 42,
    save: bool = True,
) -> Dict[str, np.ndarray]:
    """Performs 60/20/20 TVT split within each participant and saves CSVs."""
    X, y, participants = _load_dataset(kind)
    rng = np.random.default_rng(random_state)
    buffers = {name: [] for name in _SPLIT_NAMES}
    unique_pids = np.unique(participants)
    
    for pidx, pid in enumerate(unique_pids):
        progress_bar(pidx, len(unique_pids), label=f"Within-subject split ({kind})")
        
        idx = np.where(participants == pid)[0]
        if idx.size == 0:
            continue
        rng.shuffle(idx)
        n_train, n_val, _ = _partition_counts(idx.size, ratios)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
        buffers["train"].append(_combine_rows(X, participants, y, train_idx))
        buffers["val"].append(_combine_rows(X, participants, y, val_idx))
        buffers["test"].append(_combine_rows(X, participants, y, test_idx))
    splits = {name: _finalize(buffers[name], X.shape[1]) for name in _SPLIT_NAMES}
    if save:
        _save_splits(kind, "within", splits)
    _print_summary(kind, "within", splits)
    return splits


def split_between_subject( # Pega nos participantes mete numa pool e sorteia 9 para treino, 3 para val e 3 para teste
    kind: str,
    train_subjects: int = 9,
    val_subjects: int = 3,
    test_subjects: int = 3,
    random_state: int | None = 42,
    participant_groups: Dict[str, list[int]] | None = None,
    save: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, list[int]]]:
    """Splits by assigning participants to train/val/test groups."""
    X, y, participants = _load_dataset(kind)
    unique_participants = np.unique(participants)
    required = train_subjects + val_subjects + test_subjects
    if participant_groups is None:
        if unique_participants.size < required:
            raise ValueError(
                f"Dataset '{kind}' necessita de pelo menos {required} participantes para cumprir o split 9/3/3."
            )
        rng = np.random.default_rng(random_state)
        shuffled = unique_participants.copy()
        rng.shuffle(shuffled)
        train_ids = shuffled[:train_subjects].tolist()
        val_ids = shuffled[train_subjects:train_subjects + val_subjects].tolist()
        test_ids = shuffled[train_subjects + val_subjects:train_subjects + val_subjects + test_subjects].tolist()
        groups = {"train": train_ids, "val": val_ids, "test": test_ids}
    else:
        groups = {
            split: list(map(int, participant_groups.get(split, [])))
            for split in _SPLIT_NAMES
        }
        dataset_participants = set(unique_participants.tolist())
        missing = {
            split: [pid for pid in groups[split] if pid not in dataset_participants]
            for split in _SPLIT_NAMES
        }
        missing_filtered = {k: v for k, v in missing.items() if v}
        if missing_filtered:
            raise ValueError(
                f"Dataset '{kind}' não contém todos os participantes necessários: {missing_filtered}"
            )
    splits = {}
    for split, ids in groups.items():
        mask = np.isin(participants, ids)
        idx = np.where(mask)[0]
        if idx.size == 0:
            raise ValueError(f"Split '{split}' ficou vazio para dataset '{kind}'.")
        splits[split] = _combine_rows(X, participants, y, idx)
    if save:
        _save_splits(kind, "between", splits)
    _print_summary(kind, "between", splits)
    return splits, groups


def load_saved_splits(kind: str, strategy: str) -> Dict[str, np.ndarray]:
    """Load previously saved CSV splits for a given dataset kind and strategy."""
    strategy = strategy.lower()
    if strategy not in {"within", "between"}:
        raise ValueError("strategy deve ser 'within' ou 'between'.")

    splits = {}
    base_dir = SPLITS_DIR / kind
    for split_name in _SPLIT_NAMES:
        path = base_dir / f"{kind}_{strategy}_{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Ficheiro de split não encontrado: {path}")
        data = _ensure_matrix(np.loadtxt(path, delimiter=","))
        splits[split_name] = data
    return splits