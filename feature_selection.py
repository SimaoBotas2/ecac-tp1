import numpy as np
from skfeature.function.similarity_based import fisher_score
from skrebate import ReliefF

def fisher_score_selection(X, y, top_n=10):
    scores = fisher_score.fisher_score(X, y)
    sorted_idx = np.argsort(scores)[::-1]
    top_idx = sorted_idx[:top_n]
    return top_idx, scores

def reliefF_selection(X, y, top_n=10, n_neighbors=10):
    relief = ReliefF(n_neighbors=n_neighbors, n_features_to_select=top_n)
    relief.fit(X, y)
    top_idx = relief.top_features_[:top_n]
    scores = relief.feature_importances_
    return top_idx, scores

def get_features_by_indices(X, indices):
    return X[:, indices]
