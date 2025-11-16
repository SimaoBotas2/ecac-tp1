import numpy as np
from sklearn.feature_selection import f_classif
from skrebate import ReliefF

#Nota:
    #As docstrings deste documento foram escritas pelos autores e refinadas com ajuda de LLMs.


def fisher_score_selection(X, y, top_n=10):
    """
    Seleciona as top_n features com base no Fisher Score (ANOVA F-test).

    Parâmetros
    ----------
    X : array-like
        Matriz de features (amostras x variáveis).
    y : array-like
        Rótulos das classes.
    top_n : int, opcional
        Número de features a selecionar (padrão = 10).

    Retorna
    -------
    top_idx : np.ndarray
        Índices das features mais relevantes.
    F : np.ndarray
        Scores F de todas as features.
    """
     
    F, _ = f_classif(X, y) 
    top_idx = np.argsort(F)[::-1][:top_n]
    return top_idx, F

def reliefF_selection(X, y, top_n=10, n_neighbors=10):
    """
    Seleciona as top_n features usando o algoritmo ReliefF.

    Parâmetros
    ----------
    X : array-like
        Matriz de features.
    y : array-like
        Rótulos das classes.
    top_n : int, opcional
        Número de features a selecionar (padrão = 10).
    n_neighbors : int, opcional
        Número de vizinhos considerados (padrão = 10).

    Retorna
    -------
    top_idx : np.ndarray
        Índices das features selecionadas.
    scores : np.ndarray
        Importância atribuída a cada feature.
    """
    relief = ReliefF(n_neighbors=n_neighbors, n_features_to_select=top_n)
    relief.fit(X, y)
    top_idx = relief.top_features_[:top_n]
    scores = relief.feature_importances_
    return top_idx, scores

def print_selection(name, indices, scores, feature_names):
    print(f"\n{name} - Top {len(indices)} features:")
    for idx, score in zip(indices, scores[indices]):
        fname = feature_names[idx] if idx < len(feature_names) else f'f{idx}'
        print(f"  idx={idx:3d}  name={fname:40s}  weight={score:.6f}")