# ======================== META 2 ================================
# ECAC 2025 – TP1 (META 2)
# Autores: Simão Tomás Botas Carvalho nº 2021223055
#          Martim Costa Duarte nº 2021275991
#          Estrutura do projeto refinada com LLM (CHAT-GPT 5)

import sys
from pathlib import Path
# GARANTE QUE O ROOT DO PROJETO ESTÁ NO PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import numpy as np
from meta1.preprocessing import data_treatment
from meta1.features import feature_extractor as fe
from meta2.smote import meta2_balance
from meta2.embeddings.embeddings_extractor import (
    extract_embeddings_dataset,
    save_embeddings_to_csv,
)
from meta2.splits import data_splitter
from meta2.splits import scenario_builder
from models.knn import KNNClassifier, print_metrics
from utils.progress import print_section


# ======================================================================
# PATHS DO PROJETO
# ======================================================================

DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
META2_FEATURES_FILE = DATA_PROCESSED / "meta2_features.csv"
FEATURES_X_FILE = DATA_PROCESSED / "features_X.csv"
FEATURES_Y_FILE = DATA_PROCESSED / "features_y.csv"
FEATURES_PART_FILE = DATA_PROCESSED / "features_participant.csv"

# Flag para voltar a extrair features específicas (default False)
FORCE_SPECIFIC_FEATURE_RECOMPUTE = False

print_section("META 2 - Preparar Dados")

# ======================================================================
# TODO 0 — DEFINIR PARÂMETROS BASE
# ======================================================================

participant_selected = 3
sensors_selected = [1, 2, 3, 4, 5]
all_participants = list(range(15))
SPLIT_RANDOM_STATE = 42

# ======================================================================
# 1 — CARREGAR DADOS E FILTRAR APENAS ATIVIDADES 1–7
# ======================================================================

print("\n--- Carregar dados ---")

dados = data_treatment.get_data(participant_selected, sensors_selected)  # type: ignore[arg-type]
activities = dados[:, 11].astype(int)

mask_1_to_7 = activities <= 7
dados = dados[mask_1_to_7]
activities = activities[mask_1_to_7]

print(f"Após filtrar atividades 1-7: {dados.shape}")

# ======================================================================
# 1.1 — ANALISAR BALANCEAMENTO DO DATASET
# ======================================================================

print("\n=== 1.1 - Balanceamento das atividades ===")
if META2_FEATURES_FILE.exists():
    try:
        meta2_balance.analyze(path=str(META2_FEATURES_FILE))
    except Exception as e:
        print(f"[META2][1.1] Erro ao analisar balanceamento via ficheiro: {e}")
else:
    print(f"[META2][1.1] Aviso: ficheiro '{META2_FEATURES_FILE.name}' não encontrado. A usar dados em memória.")
    unique, counts = np.unique(activities, return_counts=True)
    total = counts.sum()
    for act, count in zip(unique, counts):
        pct = (count / total) * 100 if total else 0
        print(f"  Activity {act}: {count} samples ({pct:.2f}%)")
    if counts.size:
        ratio = counts.max() / max(counts.min(), 1)
        print(f"  Imbalance ratio (max/min): {ratio:.2f}")
    else:
        print("  Sem amostras após filtragem.")

# ======================================================================
#  1.2 — SMOTE
#  1.3 — VISUALIZAR SÍNTESE DA ATIVIDADE 4 DO PARTICIPANTE 3
# ======================================================================
"""
print("\n=== 1.2 e 1.3 - Gerar e Visualizar Amostras Sintéticas ===")

activity_for_aug = 4
K_aug = 3

force_plot_recompute = FORCE_SPECIFIC_FEATURE_RECOMPUTE
if not force_plot_recompute:
    try:
        features_X = np.loadtxt(FEATURES_X_FILE, delimiter=",")
        if features_X.ndim == 1:
            features_X = features_X.reshape(1, -1)
        features_y = np.loadtxt(FEATURES_Y_FILE, delimiter=",").astype(int)
        features_part = np.loadtxt(FEATURES_PART_FILE, delimiter=",").astype(int)
        if features_part.ndim > 1:
            features_part = features_part.ravel()

        mask_participant = (features_part == participant_selected)
        mask_allowed = np.isin(features_y, list(range(1, 8)))
        combined_mask = mask_participant & mask_allowed
        total_participant_windows = int(combined_mask.sum())
        total_activity_windows = int((combined_mask & (features_y == activity_for_aug)).sum())

        print(
            f"[META2][1.2/1.3] Features pré-computadas encontradas: {total_participant_windows} janelas do participante {participant_selected}."
        )
        print(
            f"[META2][1.2/1.3] Atividade {activity_for_aug}: {total_activity_windows} janelas disponíveis (pré-extraídas)."
        )
        if total_participant_windows == 0:
            raise ValueError("Participante sem janelas no ficheiro pré-computado.")
    except Exception as e:
        print(f"[META2][1.2/1.3] Aviso: falha ao usar features pré-computadas ({e}). A re-extrair do bruto...")
        force_plot_recompute = True


try:
    meta2_balance.generate_and_visualize_samples_for_participant(
        participant_selected,
        activity=activity_for_aug,
        K=K_aug,
        sensors=sensors_selected,
        force_recompute=force_plot_recompute,
        features_dir=DATA_PROCESSED,
    )
except Exception as e:
    print(f"[META2] Erro em SMOTE/visualização: {e}")
    print("TODO 1.2/1.3: verificar meta2_balance.py")
"""

# ======================================================================
# TODO 2 — EXTRAÇÃO DE EMBEDDINGS
# ======================================================================
"""
print_section("2.1 - Extrair Embeddings")

try:
    dados_all, participants_all = data_treatment.get_data(  # type: ignore[arg-type]
        all_participants, sensors_selected, return_participants=True
    )
    accel_data = dados_all[:, 1:4].astype(float)
    activities_all = dados_all[:, 11].astype(int)
    sr = fe.sampling_rate_calculator(dados_all)

    embeddings_X, embeddings_y, embeddings_part = extract_embeddings_dataset(
        accel_data=accel_data,
        activities=activities_all,
        sampling_rate=sr,
        window_size_sec=5,
        overlap=0.5,
        allowed_activities=range(1, 8),
        batch_size=64,
        device="cpu",
        participant_ids=participants_all,
    )

    x_path, y_path, part_path = save_embeddings_to_csv(
        embeddings_X,
        embeddings_y,
        base_path=DATA_PROCESSED,
        participants=embeddings_part,
    )
    print(f"Embeddings guardados → {x_path} | {y_path} | {part_path}")

except Exception as e:
    print(f"[EMBEDDINGS] Erro: {e}")
    print("TODO 2.1: completar embeddings_extractor.py")
"""

# ======================================================================
# TODO 3 — DATA SPLITTING (WITHIN + BETWEEN SUBJECT)
# ======================================================================
"""
Task 3.1 - TVT 60-20-20 within subject
Task 3.2 - 9 train / 3 val / 3 test between subjects
Task 3.3 - Discuss differences (written)
Task 3.4 - Prepare datasets:
      a) full
      b) PCA → 90%
      c) ReliefF → top 15


print_section("3.1 / 3.2 - Data Splitting (Features & Embeddings)")
try:
    within_features = data_splitter.split_within_subject("features", random_state=SPLIT_RANDOM_STATE)
    within_embeddings = data_splitter.split_within_subject("embeddings", random_state=SPLIT_RANDOM_STATE)
    between_features, participant_groups = data_splitter.split_between_subject(
        "features", random_state=SPLIT_RANDOM_STATE
    )
    between_embeddings, _ = data_splitter.split_between_subject(
        "embeddings", participant_groups=participant_groups
    )
    print("[META2][3.1 e 3.2] Splits guardados em data/processed/splits.")

    scenario_builder.prepare_scenarios("features", "within", splits=within_features)
    scenario_builder.prepare_scenarios("embeddings", "within", splits=within_embeddings)
    scenario_builder.prepare_scenarios("features", "between", splits=between_features)
    scenario_builder.prepare_scenarios("embeddings", "between", splits=between_embeddings)
    print("[META2][3.4] Cenários guardados em data/processed/scenarios.")
except Exception as e:
    print(f"[META2][3.x] Erro ao gerar splits: {e}")
"""

# ======================================================================
# 4 — MODEL LEARNING (kNN)
# ======================================================================
"""
Task 4.1 - Implement manual kNN
Task 4.2 - Implement metrics:
      - confusion matrix
      - accuracy
      - precision
      - recall
      - F1
"""

print("\n=== 4.1 / 4.2 - kNN Training & Evaluation ===\n")

try:
    # Ficheiro para guardar resultados
    results_dir = DATA_PROCESSED / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "knn_results.txt"
    
    # Abrir ficheiro para escrita
    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("META 2 - kNN Classification Results\n")
        f.write("=" * 70 + "\n\n")
        
        # Carregar splits já criados (within-subject)
        splits_dir = DATA_PROCESSED / "splits" / "embeddings+0"
        
        X_train = np.loadtxt(splits_dir / "embeddings_within_train.csv", delimiter=",")
        y_train = np.loadtxt(splits_dir / "embeddings_within_train.csv", delimiter=",", usecols=-1).astype(int)
        
        X_val = np.loadtxt(splits_dir / "embeddings_within_val.csv", delimiter=",")
        y_val = np.loadtxt(splits_dir / "embeddings_within_val.csv", delimiter=",", usecols=-1).astype(int)
        
        X_test = np.loadtxt(splits_dir / "embeddings_within_test.csv", delimiter=",")
        y_test = np.loadtxt(splits_dir / "embeddings_within_test.csv", delimiter=",", usecols=-1).astype(int)
        
        # Remove a última coluna (labels) dos dados de entrada
        X_train = X_train[:, :-1]
        X_val = X_val[:, :-1]
        X_test = X_test[:, :-1]
        
        msg = f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}\n"
        msg += f"Features por amostra: {X_train.shape[1]}\n\n"
        print(msg)
        f.write(msg)
        
        # Treinar kNN
        knn = KNNClassifier(k=5)
        knn.fit(X_train, y_train)
        
        # Calcular métricas
        train_acc = knn.score(X_train, y_train)
        val_acc = knn.score(X_val, y_val)
        test_acc = knn.score(X_test, y_test)
        
        # Resultados
        results_msg = f"\nTrain Accuracy: {train_acc:.4f}\n"
        results_msg += f"Val Accuracy: {val_acc:.4f}\n"
        results_msg += f"Test Accuracy: {test_acc:.4f}\n"
        print(results_msg)
        f.write(results_msg)
        
        # Matriz de confusão e métricas detalhadas
        y_pred_train = knn.predict(X_train)
        y_pred_val = knn.predict(X_val)
        y_pred_test = knn.predict(X_test)
        
        from models.knn import confusion_matrix
        
        cm_test, classes = confusion_matrix(y_test, y_pred_test)
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("TEST SET - Detailed Metrics\n")
        f.write("=" * 70 + "\n\n")
        
        # Escrever matriz de confusão
        f.write("Confusion Matrix (rows=true, cols=predicted):\n")
        f.write("     " + "  ".join(f"A{c}" for c in classes) + "\n")
        for i, true_class in enumerate(classes):
            row_str = "  ".join(f"{cm_test[i, j]:4d}" for j in range(len(classes)))
            f.write(f"A{true_class}  {row_str}\n")
        
        # Métricas por classe
        f.write("\nPer-class metrics:\n")
        f.write(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-" * 52 + "\n")
        
        for i, c in enumerate(classes):
            tp = cm_test[i, i]
            fp = cm_test[:, i].sum() - tp
            fn = cm_test[i, :].sum() - tp
            support = cm_test[i, :].sum()
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
            f.write(f"A{c:<7} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {int(support):<10}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Resultados guardados em: {results_file}\n")
        f.write("=" * 70 + "\n")
    
    # Também imprimir no terminal
    print_metrics(y_test, y_pred_test)
    print(f"\n✓ Resultados guardados em: {results_file}")

except Exception as e:
    print(f"[META2][4.1/4.2] Erro ao usar splits: {e}")
    import traceback
    traceback.print_exc()


# ======================================================================
# TODO 5 — EVALUATION PIPELINE
# ======================================================================
"""
Task 5.1 - Hyperparameter tuning (k)
Task 5.2 - Report confusion matrices, compare
Task 5.3 - Hypothesis testing (repeat splits for distributions)
"""


# ======================================================================
# TODO 6 — DEPLOYMENT FUNCTION
# ======================================================================
"""
Task 6:
Create pipeline function:

predict(segment_256x9):
    - normalize
    - split into windows
    - extract features OR embeddings
    - apply PCA / ReliefF if required
    - run classifier
    - return predicted activity
"""


# ======================================================================
# TODO 7 — GO FURTHER (BONUS)
# ======================================================================
"""
Task 7:
Extra improvements:
    - Deep neural models
    - LSTM/Transformer
    - Augmentation improvements
    - Ensembling models
"""


print("\n=== META 2 - TODOS DEFINIDOS ===\n")
