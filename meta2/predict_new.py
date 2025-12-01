# ======================== TASK 6: DEPLOYMENT ================================
# ECAC 2025 – TP1 (META 2) – Task 6
# Step 1: Treina o melhor modelo com train+val e guarda-o
# Step 2: Cria função que carrega e usa o modelo para predizer

import sys
from pathlib import Path
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.knn import KNNClassifier
from meta1.features import feature_extractor as fe
from meta1.features import feature_selection as fs

# ============================================================================
# STEP 1: TREINAR E GUARDAR O MELHOR MODELO
# ============================================================================

def train_and_save_best_model(data_type='features',
                               split_type='within',
                               scenario='all',
                               k=10,
                               save_dir=None):
    """
    Treina o melhor modelo com train+val combinados e guarda.
    
    Parâmetros:
    -----------
    data_type : str ('features' ou 'embeddings')
    split_type : str ('within' ou 'between')
    scenario : str ('all', 'pca', 'relief')
    k : int (número de vizinhos do kNN)
    save_dir : Path (diretório para guardar modelo)
    
    Returns:
    --------
    model_info : dict com informações do modelo guardado
    """
    
    if save_dir is None:
        save_dir = ROOT / "models" / "trained_models"
    save_dir.mkdir(parents=True, exist_ok=True)
    # Nome do ficheiro do modelo e caminho
    model_filename = f"model_{data_type}_{split_type}_{scenario}_k{k}.pkl"
    model_path = save_dir / model_filename

    # Se já existir, não treinar de novo — carregar info e devolver
    if model_path.exists():
        try:
            with open(model_path, 'rb') as f:
                existing = pickle.load(f)
            print(f"\n[Task6] Modelo já existente: {model_path}. A reutilizar.")
            return {
                'model_path': model_path,
                'data_type': existing.get('data_type', data_type),
                'split_type': existing.get('split_type', split_type),
                'scenario': existing.get('scenario', scenario),
                'k': existing.get('k', k),
                'test_accuracy': existing.get('test_accuracy', None),
            }
        except Exception as e:
            print(f"[Task6] Aviso: falha ao ler modelo existente ({e}). Será re-treinado.")

    print(f"\n{'='*70}")
    print(f"TREINAR E GUARDAR MELHOR MODELO")
    print(f"{'='*70}")
    print(f"\nConfigurações:")
    print(f"  Data Type: {data_type}")
    print(f"  Split Type: {split_type}")
    print(f"  Scenario: {scenario}")
    print(f"  k: {k}")
    
    # 1. Carregar dados de treino
    data_dir = ROOT / "data" / "processed"
    scenarios_dir = data_dir / "scenarios" / data_type / split_type
    scenario_file = scenarios_dir / f"{scenario}.npz"
    
    if not scenario_file.exists():
        raise FileNotFoundError(f"Ficheiro não encontrado: {scenario_file}")
    
    print(f"\nCarregando dados de {scenario_file.name}...")
    data = np.load(scenario_file)
    X_train = data['train_X']
    y_train = data['train_y']
    X_val = data['val_X']
    y_val = data['val_y']
    X_test = data['test_X']
    y_test = data['test_y']
    
    print(f"  Train: {X_train.shape[0]} amostras")
    print(f"  Val: {X_val.shape[0]} amostras")
    print(f"  Test: {X_test.shape[0]} amostras")
    # 2. Combinar train + val
    print(f"\nCombinando train + val para retreinar...")
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.hstack([y_train, y_val])
    print(f"  Train+Val: {X_train_val.shape[0]} amostras (combinadas)")
    
    # 3. Normalizar
    print(f"\nNormalizando dados...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_val)
    
    # 4. Aplicar PCA se necessário
    pca_model = None
    if scenario == 'pca':
        print(f"Aplicando PCA (90% variância)...")
        pca_model = PCA(n_components=0.9)
        X_scaled = pca_model.fit_transform(X_scaled)
        print(f"Dimensões após PCA: {X_scaled.shape[1]}")
    
    # 5. Aplicar ReliefF se necessário
    reliefF_indices = None
    if scenario == 'relief':
        print(f"Selecionando top 15 features com ReliefF...")
        _, relief_scores = fs.reliefF_selection(X_train_val, y_train_val, top_n=15)
        reliefF_indices = np.argsort(relief_scores)[::-1][:15]
        X_scaled = X_scaled[:, reliefF_indices]
        print(f"  Features selecionadas: {len(reliefF_indices)}")
    
    # 6. Treinar kNN
    print(f"\nTreinando kNN com k={k}...")
    knn_model = KNNClassifier(k=k)
    knn_model.fit(X_scaled, y_train_val)
    
    # 7. Avaliar no test set (para informação)
    print(f"\nAvaliando no test set...")
    X_test_scaled = scaler.transform(X_test)
    if scenario == 'pca':
        X_test_scaled = pca_model.transform(X_test_scaled)
    elif scenario == 'relief':
        X_test_scaled = X_test_scaled[:, reliefF_indices]
    
    y_pred_test = knn_model.predict(X_test_scaled)
    test_accuracy = (y_pred_test == y_test).mean()
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    
    # 8. Guardar modelo
    
    model_data = {
        'knn_model': knn_model,
        'scaler': scaler,
        'pca_model': pca_model,
        'reliefF_indices': reliefF_indices,
        'data_type': data_type,
        'split_type': split_type,
        'scenario': scenario,
        'k': k,
        'test_accuracy': test_accuracy,
        'sampling_rate': 50  # Hz
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModelo guardado em: {model_path}")
    print(f"{'='*70}\n")
    
    return {
        'model_path': model_path,
        'data_type': data_type,
        'split_type': split_type,
        'scenario': scenario,
        'k': k,
        'test_accuracy': test_accuracy
    }


# ============================================================================
# STEP 2: FUNÇÃO DE DEPLOYMENT
# ============================================================================

def load_model(model_path):
    """Carrega modelo guardado."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def predict_activity(raw_data_256x9, model_data):
    """
    Task 6 - Step 2: Prediz atividade para dados brutos.
    
    Parâmetros:
    -----------
    raw_data_256x9 : np.ndarray
        Shape (256, 9) com [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, mag_x, mag_y, mag_z]
    model_data : dict
        Modelo carregado com load_model()
    
    Returns:
    --------
    predicted_activity : int (1-7)
    confidence : float (0-1, proporção de vizinhos que votam para a classe)
    """
    
    # Validar input
    if raw_data_256x9.shape != (256, 9):
        raise ValueError(f"Input deve ser shape (256, 9), recebido {raw_data_256x9.shape}")
    
    # Extrair componentes do modelo
    knn_model = model_data['knn_model']
    scaler = model_data['scaler']
    pca_model = model_data['pca_model']
    reliefF_indices = model_data['reliefF_indices']
    data_type = model_data['data_type']
    scenario = model_data['scenario']
    sampling_rate = model_data['sampling_rate']
    
    # 1. Separar sensores
    accel = raw_data_256x9[:, 0:3].astype(float)
    gyro = raw_data_256x9[:, 3:6].astype(float)
    mag = raw_data_256x9[:, 6:9].astype(float)
    
    # 2. Extrair features
    if data_type == 'features':
        # Extrair features temporais + espectrais
        features_dict = fe.FeatureExtractor.extract_window_features(
            accel, gyro, mag, sampling_rate
        )
        feature_vector = np.array(list(features_dict.values())).reshape(1, -1)
    else:
        raise NotImplementedError("Embeddings requerem modelo pré-treinado - não implementado aqui")
    
    # 3. Normalizar
    feature_scaled = scaler.transform(feature_vector)
    
    # 4. Aplicar PCA/ReliefF
    if scenario == 'pca':
        feature_scaled = pca_model.transform(feature_scaled)
    elif scenario == 'relief':
        feature_scaled = feature_scaled[:, reliefF_indices]
    
    # 5. Predizer com kNN
    predicted_activity = knn_model.predict(feature_scaled)[0]
    
    # 6. Calcular confiança (% de vizinhos que votam para essa classe)
    from collections import Counter
    distances = np.linalg.norm(
        knn_model.X_train - feature_scaled,
        axis=1
    )
    k_nearest_indices = np.argsort(distances)[:knn_model.k]
    k_nearest_labels = knn_model.y_train[k_nearest_indices]
    
    votes = Counter(k_nearest_labels)
    max_votes = votes[predicted_activity]
    confidence = max_votes / knn_model.k
    
    return int(predicted_activity), confidence

