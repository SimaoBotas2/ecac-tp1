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


# ======================================================================
# PATHS DO PROJETO
# ======================================================================

DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
META2_FEATURES_FILE = DATA_PROCESSED / "meta2_features.csv"

print("\n=== META 2 – Preparar Dados ===")

# ======================================================================
# TODO 0 — DEFINIR PARÂMETROS BASE
# ======================================================================

participant_selected = 3
sensors_selected = [1, 2, 3, 4, 5]

# ======================================================================
# TODO 1 — CARREGAR DADOS E FILTRAR APENAS ATIVIDADES 1–7
# ======================================================================

print("\n--- Carregar dados ---")

dados = data_treatment.get_data(participant_selected, sensors_selected)
activities = dados[:, 11].astype(int)

mask_1_to_7 = activities <= 7
dados = dados[mask_1_to_7]
activities = activities[mask_1_to_7]

print(f"Após filtrar atividades 1-7: {dados.shape}")

# ======================================================================
# TODO 1.1 — ANALISAR BALANCEAMENTO DO DATASET
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
# TODO 1.2 — SMOTE (meta2_balance.py)
# TODO 1.3 — VISUALIZAR SÍNTESE DA ATIVIDADE 4 DO PARTICIPANTE 3
# ======================================================================

print("\n=== 1.2 e 1.3 - Gerar e Visualizar Amostras Sintéticas ===")

activity_for_aug = 4
K_aug = 3

try:
    meta2_balance.generate_and_visualize_samples_for_participant(
        participant_selected,
        activity=activity_for_aug,
        K=K_aug,
        sensors=sensors_selected,
    )
except Exception as e:
    print(f"[META2] Erro em SMOTE/visualização: {e}")
    print("TODO 1.2/1.3: verificar meta2_balance.py")


# ======================================================================
# TODO 2 — EXTRAÇÃO DE EMBEDDINGS
# ======================================================================

print("\n=== 2.1 - Extrair Embeddings ===")

try:
    accel_data = dados[:, 1:4].astype(float)
    sr = fe.sampling_rate_calculator(dados)

    embeddings_X, embeddings_y = extract_embeddings_dataset(
        accel_data=accel_data,
        activities=activities,
        sampling_rate=sr,
        window_size_sec=5,
        overlap=0.5,
        allowed_activities=range(1, 8),
        batch_size=64,
        device="cpu",
    )

    x_path, y_path = save_embeddings_to_csv(embeddings_X, embeddings_y)
    print(f"Embeddings guardados → {x_path} | {y_path}")

except Exception as e:
    print(f"[EMBEDDINGS] Erro: {e}")
    print("TODO 2.1: completar embeddings_extractor.py")


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
"""


# ======================================================================
# TODO 4 — MODEL LEARNING (kNN)
# ======================================================================
"""
Task 4.1 – Implement manual kNN
Task 4.2 – Implement metrics:
      - confusion matrix
      - accuracy
      - precision
      - recall
      - F1
"""


# ======================================================================
# TODO 5 — EVALUATION PIPELINE
# ======================================================================
"""
Task 5.1 – Hyperparameter tuning (k)
Task 5.2 – Report confusion matrices, compare
Task 5.3 – Hypothesis testing (repeat splits for distributions)
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


print("\n=== META 2 – TODOS DEFINIDOS ===\n")
