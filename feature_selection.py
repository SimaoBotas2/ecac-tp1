import numpy as np
from sklearn.feature_selection import f_classif
from skrebate import ReliefF

def fisher_score_selection(X, y, top_n=10):
    F, _ = f_classif(X, y) 
    top_idx = np.argsort(F)[::-1][:top_n]
    return top_idx, F

def reliefF_selection(X, y, top_n=10, n_neighbors=10):
    relief = ReliefF(n_neighbors=n_neighbors, n_features_to_select=top_n)
    relief.fit(X, y)
    top_idx = relief.top_features_[:top_n]
    scores = relief.feature_importances_
    return top_idx, scores

def get_features_by_indices(X, indices):
    return X[:, indices]
