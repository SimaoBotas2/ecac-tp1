#DEPLOYMENT - Predição com kNN Model

import sys
from pathlib import Path
import numpy as np
import pickle

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.knn import KNNClassifier
from meta2.pipeline_predict import process_csv_file, select_random_csvs


def evaluate_model(
    data_type='features',
    split_type='within',
    scenario='all',
    k=10,
    X_test_custom=None,
    y_test_custom=None,
    verbose=True,
):
    """
    Carrega/treina modelo e avalia em dados customizados.
    
    Parâmetros:
    -----------
    data_type : str
        Tipo de dados ('features' ou 'embeddings')
    split_type : str
        Tipo de split ('within' ou 'between')
    scenario : str
        Cenário ('all', 'pca', 'relief')
    k : int
        Número de vizinhos
    X_test_custom : array ou None
        Features de teste customizadas
    y_test_custom : array ou None
        Etiquetas de teste customizadas
    verbose : bool
        Se True, imprime informações
    
    Returns:
    --------
    dict com:
        - test_accuracy: acurácia no teste
        - y_test: etiquetas reais
        - y_pred: etiquetas preditas
    """
    data_dir = ROOT / "data" / "processed"
    scenarios_dir = data_dir / "scenarios" / data_type / split_type
    scenario_file = scenarios_dir / f"{scenario}.npz"

    if not scenario_file.exists():
        raise FileNotFoundError(f"Cenário não encontrado: {scenario_file}")

    model_dir = ROOT / "models" / "trained_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"model_eval_{data_type}_{split_type}_{scenario}_k{k}.pkl"

    data = np.load(scenario_file)
    X_train = data['train_X']
    y_train = data['train_y']
    X_val = data['val_X']
    y_val = data['val_y']
    X_test = X_test_custom if X_test_custom is not None else data['test_X']
    y_test = y_test_custom if y_test_custom is not None else data['test_y']

    if verbose:
        print(f"\n[Evaluation] Cenário: {scenario} ({data_type}/{split_type}, k={k})")
        print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]} samples")
        print(f"  Features: {X_train.shape[1]}\n")

    if model_file.exists():
        if verbose:
            print(f"Carregando modelo guardado: {model_file.name}")
        with open(model_file, 'rb') as f:
            knn = pickle.load(f)
    else:
        if verbose:
            print(f"Modelo não encontrado. Treinando kNN com k={k}...")
        
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.hstack([y_train, y_val])
        
        if verbose:
            print(f"  Treinando com {X_train_val.shape[0]} samples (train + val)")
        
        knn = KNNClassifier(k=k)
        knn.fit(X_train_val, y_train_val)
        
        if verbose:
            print(f"Guardando modelo: {model_file.name}")
        with open(model_file, 'wb') as f:
            pickle.dump(knn, f)

    if verbose:
        print("Avaliando...\n")
    y_pred_test = knn.predict(X_test)
    acc_test = (y_pred_test == y_test).mean()

    if verbose:
        print(f"Test Accuracy: {acc_test:.4f}\n")
        print("Atividade Real vs Predicted:")
        print(f"{'Real':<8} {'Predicted':<12} {'Status':<10}")
        print("-" * 30)
        for real, pred in zip(y_test, y_pred_test):
            status = "Correto" if real == pred else "Errado"
            print(f"A{real:<7} A{pred:<11} {status}")

    return {
        'test_accuracy': acc_test,
        'y_test': y_test,
        'y_pred': y_pred_test,
    }


def predict_from_array(
    raw_data,
    activity_label,
    data_type='features',
    split_type='within',
    scenario='all',
    k=10,
    verbose=True,
):
    """
    Prediz atividade a partir de um array raw de sensores.
    
    Parâmetros:
    -----------
    raw_data : np.ndarray
        Array de shape (256, 9) com [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, mag_x, mag_y, mag_z]
    activity_label : int
        Etiqueta real da atividade (1-7)
    data_type : str
        Tipo de dados ('features' ou 'embeddings')
    split_type : str
        Tipo de split ('within' ou 'between')
    scenario : str
        Cenário ('all', 'pca', 'relief')
    k : int
        Número de vizinhos
    verbose : bool
        Se True, imprime resultados
    
    Returns:
    --------
    dict com:
        - activity_predicted: atividade predita
        - activity_real: atividade real
        - accuracy: se acertou (1.0 ou 0.0)
        - raw_shape: shape do input
        - is_correct: bool se a predição foi correta
    """
    raw_data = np.asarray(raw_data, dtype=np.float32)
    
    if raw_data.shape != (256, 9):
        raise ValueError(f"Shape esperado: (256, 9), recebido: {raw_data.shape}")
    
    if verbose:
        print(f"\n[Predict] Array recebido com shape {raw_data.shape}")
        print(f"  Atividade real: A{activity_label}\n")
    
    # Primeiras 250 linhas para feature extraction
    feature_window = raw_data[:250, :]
    
    accel_data = feature_window[:, 0:3]
    gyro_data = feature_window[:, 3:6]
    mag_data = feature_window[:, 6:9]
    
    # Atividades para extract_features_4_2
    activities_for_extract = np.full(250, activity_label, dtype=int)
    
    try:
        from meta1.features import feature_extractor as fe
        X_features_list, _, _ = fe.extract_features_4_2(
            accel_data, gyro_data, mag_data, activities_for_extract,
            sampling_rate=50, participant_ids=None
        )
        if len(X_features_list) == 0:
            raise ValueError("Nenhuma janela válida extraída")
        X_features = np.array(X_features_list[0]).reshape(1, -1)
    except Exception as e:
        raise RuntimeError(f"Erro ao extrair features: {e}")
    
    # Carregar scaler e escalar
    scenario_file = ROOT / "data" / "processed" / "scenarios" / data_type / split_type / f"{scenario}.npz"
    
    if not scenario_file.exists():
        raise FileNotFoundError(f"Cenário não encontrado: {scenario_file}")
    
    data = np.load(scenario_file)
    meta_mean = data['meta_scaler_mean'].astype(np.float64)
    meta_scale = data['meta_scaler_scale'].astype(np.float64)
    X_scaled = (X_features - meta_mean) / meta_scale
    
    # Avaliar modelo
    eval_results = evaluate_model(
        data_type=data_type,
        split_type=split_type,
        scenario=scenario,
        k=k,
        X_test_custom=X_scaled,
        y_test_custom=np.array([activity_label]),
        verbose=verbose,
    )
    
    accuracy = eval_results['test_accuracy']
    y_pred = eval_results['y_pred'][0]
    
    if verbose:
        print(f"[Predição] A{y_pred} (Real: A{activity_label})")
    
    return {
        'activity_predicted': int(y_pred),
        'activity_real': int(activity_label),
        'accuracy': float(accuracy),
        'raw_shape': raw_data.shape,
        'is_correct': accuracy == 1.0,
    }


def evaluate_multiple_csvs(
    num_csvs=3,
    part_folder=None,
    data_type='features',
    split_type='within',
    scenario='all',
    k=10,
    random_participant=False,
    random_device=False,
    participants=None,
    devices=None,
):
    """
    Testa o modelo em múltiplos CSV, apenas para testar que a função de predicting funciona corretamente.
    
    Parâmetros:
    -----------
    num_csvs : int
        Número de CSVs a testar.
    part_folder : str ou None
        Pasta específica (ex: 'part7'). Se None e random_participant=False, usa 'part7'.
    data_type : str
        Tipo de dados ('features' ou 'embeddings')
    split_type : str
        Tipo de split ('within' ou 'between')
    scenario : str
        Cenário ('all', 'pca', 'relief')
    k : int
        Número de vizinhos
    random_participant : bool
        Se True, escolhe participante aleatório
    random_device : bool
        Se True, escolhe device aleatório
    participants : list ou None
        IDs de participantes disponíveis
    devices : list ou None
        IDs de devices disponíveis
    
    Returns:
    --------
    list de dicts com resultados de cada teste
    """
    results = []
    
    try:
        selected_csvs = select_random_csvs(
            num_csvs=num_csvs,
            part_folder=part_folder,
            participants=participants,
            devices=devices,
            random_participant=random_participant,
            random_device=random_device,
        )
        
        for csv_idx, csv_path in enumerate(selected_csvs, 1):
            print(f"\n[Test {csv_idx}/{len(selected_csvs)}] CSV: {csv_path.name}")
            print("=" * 60)
            
            try:
                csv_data = process_csv_file(csv_path, scenario=scenario)
                X_test_scaled = csv_data['X_test_scaled']
                y_test_csv = csv_data['y_test']
                
                print(f"  Raw shape: {csv_data['raw_shape']}")
                print(f"  Linhas selecionadas: [{csv_data['window_start']}:{csv_data['window_end']}]")
                print(f"  Atividade confirmada: A{csv_data['activity']}")
                print(f"  Janelas válidas encontradas: {csv_data['valid_windows_count']}")
                print(f"  Features shape: {csv_data['features_shape']}")
                print(f"  Scaled shape: {X_test_scaled.shape}\n")
                
                eval_results = evaluate_model(
                    data_type=data_type,
                    split_type=split_type,
                    scenario=scenario,
                    k=k,
                    X_test_custom=X_test_scaled,
                    y_test_custom=y_test_csv,
                )
                
                results.append({
                    'csv_name': csv_path.name,
                    'window_range': f"[{csv_data['window_start']}:{csv_data['window_end']}]",
                    'activity': csv_data['activity'],
                    'accuracy': eval_results['test_accuracy'],
                    'y_test': eval_results['y_test'],
                    'y_pred': eval_results['y_pred'],
                })
                
            except Exception as e:
                print(f"  [ERRO] Falha ao processar {csv_path.name}: {e}")
        
        print("\n" + "=" * 60)
        print("RESUMO DOS TESTES")
        print("=" * 60)
        for result in results:
            print(f"{result['csv_name']} A{result['activity']} {result['window_range']}: {result['accuracy']:.4f}")
        
        if results:
            avg_accuracy = np.mean([r['accuracy'] for r in results])
            print(f"\nAcurácia média: {avg_accuracy:.4f}")
        
        return results
    
    except Exception as e:
        print(f"[ERRO] Falha ao avaliar múltiplos CSVs: {e}")
        return []