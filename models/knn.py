# ======================== kNN MANUAL ================================
# ECAC 2025 – META 2 – Ponto 4.1
# Implementação manual de k-Nearest Neighbors

import numpy as np
from collections import Counter
import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.progress import progress_bar, progress_with_time


class KNNClassifier:
    """Classificador k-Nearest Neighbors implementado manualmente."""
    
    def __init__(self, k=5, distance_metric='euclidean'):
        """
        Parâmetros:
        -----------
        k : int
            Número de vizinhos mais próximos.
        distance_metric : str
            'euclidean' ou 'manhattan'.
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        self.is_fitted = False
    
    def fit(self, X_train, y_train):
        """Armazena dados de treino."""
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.is_fitted = True
        return self
    
    def _distance(self, x1, x2):
        """Calcula distância euclidiana ou manhattan."""
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        else:
            return np.sum(np.abs(x1 - x2))
    
    def predict(self, X_test):
        """Prediz classes para X_test."""
        if not self.is_fitted:
            raise RuntimeError("Modelo não está treinado.")
        
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        predictions = []
        start_time = time.time()
        
        for idx, x in enumerate(X_test):
            progress_with_time(idx, len(X_test), start_time, label="Predicting")
            
            distances = np.array([self._distance(x, xt) for xt in self.X_train])
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            pred = Counter(k_labels).most_common(1)[0][0]
            predictions.append(pred)
        
        # Mostrar a última iteração
        progress_with_time(len(X_test), len(X_test), start_time, label="Predicting")
        
        return np.array(predictions)
    
    def score(self, X_test, y_test):
        """Retorna acurácia."""
        return np.mean(self.predict(X_test) == y_test)


# ======================== MÉTRICAS ================================

def confusion_matrix(y_true, y_pred, classes=None):
    """Calcula matriz de confusão para classes específicas."""
    if classes is None:
        classes = np.arange(1, 8)  # Default: atividades 1-7
    
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    for true_val, pred_val in zip(y_true, y_pred):
        if true_val in class_to_idx and pred_val in class_to_idx:
            true_idx = class_to_idx[true_val]
            pred_idx = class_to_idx[pred_val]
            cm[true_idx, pred_idx] += 1
    
    return cm, classes


def print_metrics(y_true, y_pred):
    """Imprime relatório de classificação com métricas por classe."""
    classes = np.arange(1, 8)
    cm, _ = confusion_matrix(y_true, y_pred, classes=classes)
    
    # Acurácia
    acc = np.mean(y_true == y_pred)
    
    print("\n=== CLASSIFICATION REPORT ===")
    print(f"Accuracy: {acc:.4f}\n")
    
    # Matriz de confusão
    print("Confusion Matrix (rows=true, cols=predicted):")
    print("     ", "  ".join(f"A{c}" for c in classes))
    for i, true_class in enumerate(classes):
        row_str = "  ".join(f"{cm[i, j]:4d}" for j in range(len(classes)))
        print(f"A{true_class}  {row_str}")
    
    # Métricas por classe
    print("\nPer-class metrics:")
    print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 52)
    
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        print(f"A{c:<7} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {int(support):<10}")
    
    # Macro averages
    precisions = []
    recalls = []
    f1s = []
    
    for i in range(len(classes)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    
    print("-" * 52)
    print(f"{'Macro Avg':<7} {np.mean(precisions):<12.4f} {np.mean(recalls):<12.4f} {np.mean(f1s):<12.4f}")
    print(f"{'Weighted':<7} {np.mean(precisions):<12.4f} {np.mean(recalls):<12.4f} {np.mean(f1s):<12.4f}")

