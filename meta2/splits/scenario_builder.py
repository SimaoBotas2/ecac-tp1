"""Scenario preparation utilities for Meta 2 splits.

For every split (train/val/test) produced by `data_splitter`, this module
creates three downstream datasets per kind (features/embeddings):

1. `all`    – StandardScaler-normalised features (fit on train only).
2. `pca`    – PCA projection keeping >= target variance (fit on train only).
3. `relief` – ReliefF top-k feature subset (fit on train only).

All derived datasets are persisted to `data/processed/scenarios/<kind>/<strategy>/`
so they can be re-used later without re-computing transformations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, cast

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from meta2.splits import data_splitter
from meta1.features.feature_extractor import pca_analysis
from meta1.features import feature_selection as fs

ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"
SCENARIOS_DIR = DATA_PROCESSED / "scenarios"
SPLIT_NAMES = ("train", "val", "test")


@dataclass
class SplitView:
    """Container holding features, labels and participant ids for one split."""

    X: np.ndarray
    y: np.ndarray
    participants: np.ndarray


@dataclass
class ScenarioResult:
    """Holds the transformed splits and metadata required to reuse them."""

    split_data: Dict[str, SplitView]
    metadata: Dict[str, np.ndarray]


def _matrix_to_view(matrix: np.ndarray) -> SplitView:
    if matrix.size == 0:
        return SplitView(
            X=np.empty((0, 0)),
            y=np.empty((0,), dtype=int),
            participants=np.empty((0,), dtype=int),
        )
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[1] < 2:
        raise ValueError("Split matrix must contain participant and label columns.")
    X = matrix[:, :-2].astype(float)
    participants = matrix[:, -2].astype(int)
    y = matrix[:, -1].astype(int)
    return SplitView(X=X, y=y, participants=participants)


def _normalise_views(views: Dict[str, SplitView]) -> tuple[Dict[str, SplitView], StandardScaler]:
    train_view = views.get("train")
    if train_view is None or train_view.X.size == 0:
        raise ValueError("Split 'train' precisa de conter dados para ajustar o scaler.")
    scaler = StandardScaler()
    scaler.fit(train_view.X)

    normalised = {
        name: SplitView(
            X=scaler.transform(view.X),
            y=view.y.copy(),
            participants=view.participants.copy(),
        )
        for name, view in views.items()
    }
    return normalised, scaler


def _fit_pca_with_helper(train_X: np.ndarray, target_variance: float) -> tuple[PCA, StandardScaler, np.ndarray]:
    if train_X.size == 0:
        raise ValueError("Não é possível ajustar PCA sem dados de treino.")
    X_pca, pca_model, scaler = pca_analysis(train_X, target_variance=target_variance)
    if not isinstance(pca_model, PCA):
        raise TypeError("pca_analysis não devolveu uma instância PCA esperada.")
    if not isinstance(scaler, StandardScaler):
        raise TypeError("pca_analysis não devolveu uma instância StandardScaler esperada.")
    return pca_model, scaler, X_pca


def _fit_relief_with_helper(
    train_X: np.ndarray,
    train_y: np.ndarray,
    top_k: int,
    max_neighbors: int,
    max_samples: int | None = None,
    random_state: int | None = 42,
) -> np.ndarray:
    if train_X.size == 0:
        raise ValueError("Não é possível ajustar ReliefF sem dados de treino.")
    if train_X.shape[1] == 0:
        raise ValueError("Dataset sem features para ReliefF.")
    k = min(top_k, train_X.shape[1])
    X_work = train_X
    y_work = train_y
    if max_samples is not None and train_X.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        subset_idx = rng.choice(train_X.shape[0], size=max_samples, replace=False)
        X_work = train_X[subset_idx]
        y_work = train_y[subset_idx]
    n_neighbors = min(max_neighbors, max(1, X_work.shape[0] - 1))
    top_idx, _ = fs.reliefF_selection(X_work, y_work, top_n=k, n_neighbors=n_neighbors)
    return np.sort(np.asarray(top_idx, dtype=int))


def _subset_views(views: Dict[str, SplitView], idx: np.ndarray) -> Dict[str, SplitView]:
    subset = {}
    for name, view in views.items():
        subset[name] = SplitView(
            X=view.X[:, idx],
            y=view.y.copy(),
            participants=view.participants.copy(),
        )
    return subset


def _build_pca_views(
    raw_views: Dict[str, SplitView],
    scaler: StandardScaler,
    pca_model: PCA,
    train_projection: np.ndarray,
) -> Dict[str, SplitView]:
    projections: Dict[str, SplitView] = {}
    for name, view in raw_views.items():
        if name == "train":
            X_proj = train_projection
        else:
            X_scaled = scaler.transform(view.X)
            X_proj = pca_model.transform(X_scaled)
        projections[name] = SplitView(
            X=X_proj,
            y=view.y.copy(),
            participants=view.participants.copy(),
        )
    return projections


def _persist_scenario(
    kind: str,
    strategy: str,
    scenario: str,
    result: ScenarioResult,
) -> Path:
    out_dir = SCENARIOS_DIR / kind / strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{scenario}.npz"
    payload = {}
    for split_name, view in result.split_data.items():
        payload[f"{split_name}_X"] = view.X
        payload[f"{split_name}_y"] = view.y
        payload[f"{split_name}_participants"] = view.participants
    for meta_key, meta_val in result.metadata.items():
        payload[f"meta_{meta_key}"] = meta_val
    np.savez_compressed(path, **payload)
    return path


def prepare_scenarios(
    kind: str,
    strategy: str,
    splits: Dict[str, np.ndarray] | None = None,
    target_variance: float = 0.9,
    relief_top_k: int = 15,
    relief_neighbors: int = 10,
    relief_max_samples: int | None = 5000,
    save: bool = True,
) -> Dict[str, ScenarioResult]:
    """Compute normalised/PCA/ReliefF datasets for a given split strategy."""
    if splits is None:
        splits = data_splitter.load_saved_splits(kind, strategy)
    views = {name: _matrix_to_view(matrix) for name, matrix in splits.items()}

    normalised, scaler = _normalise_views(views)
    scenarios: Dict[str, ScenarioResult] = {}

    scaler_mean = cast(np.ndarray, scaler.mean_)
    scaler_scale = cast(np.ndarray, scaler.scale_)
    scaler_var = cast(np.ndarray, scaler.var_)

    scenarios["all"] = ScenarioResult(
        split_data=normalised,
        metadata={
            "scaler_mean": scaler_mean.astype(np.float32),
            "scaler_scale": scaler_scale.astype(np.float32),
            "scaler_var": scaler_var.astype(np.float32),
        },
    )

    train_raw_view = views.get("train")
    if train_raw_view is None:
        raise ValueError("Split 'train' necessário para ajustar PCA.")
    pca_model, pca_scaler, train_projection = _fit_pca_with_helper(train_raw_view.X, target_variance)
    pca_scaler_mean = cast(np.ndarray, pca_scaler.mean_)
    pca_scaler_scale = cast(np.ndarray, pca_scaler.scale_)
    pca_views = _build_pca_views(views, pca_scaler, pca_model, train_projection)
    scenarios["pca"] = ScenarioResult(
        split_data=pca_views,
        metadata={
            "pca_components": pca_model.components_.astype(np.float32),
            "pca_mean": pca_model.mean_.astype(np.float32),
            "pca_explained_variance_ratio": pca_model.explained_variance_ratio_.astype(np.float32),
            "pca_n_components": np.asarray([pca_model.n_components_], dtype=np.int32),
            "pca_scaler_mean": pca_scaler_mean.astype(np.float32),
            "pca_scaler_scale": pca_scaler_scale.astype(np.float32),
        },
    )

    relief_idx = _fit_relief_with_helper(
        normalised["train"].X,
        normalised["train"].y,
        relief_top_k,
        relief_neighbors,
        max_samples=relief_max_samples,
    )
    relief_views = _subset_views(normalised, relief_idx)
    scenarios["relief"] = ScenarioResult(
        split_data=relief_views,
        metadata={"relief_indices": relief_idx.astype(np.int32)},
    )

    if save:
        for scenario_name, result in scenarios.items():
            _persist_scenario(kind, strategy, scenario_name, result)

    return scenarios


def load_scenario(kind: str, strategy: str, scenario: str) -> ScenarioResult:
    """Load a previously persisted scenario dataset."""
    path = SCENARIOS_DIR / kind / strategy / f"{scenario}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Scenario file não encontrado: {path}")
    with np.load(path, allow_pickle=False) as data:
        splits: Dict[str, SplitView] = {}
        for split_name in SPLIT_NAMES:
            key_X = f"{split_name}_X"
            if key_X not in data:
                continue
            splits[split_name] = SplitView(
                X=data[key_X],
                y=data[f"{split_name}_y"],
                participants=data[f"{split_name}_participants"],
            )
        metadata = {
            key.replace("meta_", ""): data[key]
            for key in data.files
            if key.startswith("meta_")
        }
    return ScenarioResult(split_data=splits, metadata=metadata)