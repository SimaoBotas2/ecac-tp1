# ======================== META 2 ================================
# ECAC 2025 – TP1 (META 2)
# Autores: Simão Tomás Botas Carvalho nº 2021223055
#          Martim Costa Duarte nº 2021275991
#          Estrutura do projeto refinada com LLM (CHAT-GPT 5)

import sys
from pathlib import Path
<<<<<<< HEAD
=======
# GARANTE QUE O ROOT DO PROJETO ESTÁ NO PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
>>>>>>> nigga
import numpy as np
from meta1.preprocessing import data_treatment
from meta1.features import feature_extractor as fe
from meta2.smote import meta2_balance
from meta2.embeddings.embeddings_extractor import (
    extract_embeddings_dataset,
    save_embeddings_to_csv,
)
<<<<<<< HEAD

# GARANTE QUE O ROOT DO PROJETO ESTÁ NO PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
=======
from meta2.splits import data_splitter

>>>>>>> nigga

# ======================================================================
# PATHS DO PROJETO
# ======================================================================

DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
<<<<<<< HEAD

print("\n=== META 2 – Preparar Dados ===")
=======
META2_FEATURES_FILE = DATA_PROCESSED / "meta2_features.csv"
FEATURES_X_FILE = DATA_PROCESSED / "features_X.csv"
FEATURES_Y_FILE = DATA_PROCESSED / "features_y.csv"
FEATURES_PART_FILE = DATA_PROCESSED / "features_participant.csv"

# Flag para voltar a extrair features específicas (default False)
FORCE_SPECIFIC_FEATURE_RECOMPUTE = False

print("\n=== META 2 - Preparar Dados ===")
>>>>>>> nigga

# ======================================================================
# TODO 0 — DEFINIR PARÂMETROS BASE
# ======================================================================

participant_selected = 3
sensors_selected = [1, 2, 3, 4, 5]
<<<<<<< HEAD
=======
all_participants = list(range(15))
SPLIT_RANDOM_STATE = 42
>>>>>>> nigga

# ======================================================================
# TODO 1 — CARREGAR DADOS E FILTRAR APENAS ATIVIDADES 1–7
# ======================================================================

print("\n--- Carregar dados ---")

<<<<<<< HEAD
dados = data_treatment.get_data(participant_selected, sensors_selected)
=======
dados = data_treatment.get_data(participant_selected, sensors_selected)  # type: ignore[arg-type]
>>>>>>> nigga
activities = dados[:, 11].astype(int)

mask_1_to_7 = activities <= 7
dados = dados[mask_1_to_7]
activities = activities[mask_1_to_7]

<<<<<<< HEAD
print(f"Após filtrar atividades 1–7: {dados.shape}")
=======
print(f"Após filtrar atividades 1-7: {dados.shape}")
>>>>>>> nigga

# ======================================================================
# TODO 1.1 — ANALISAR BALANCEAMENTO DO DATASET
# ======================================================================
<<<<<<< HEAD
"""
→ Contar amostras por atividade
→ Verificar desbalanceamento
→ Plot opcional
"""

# ======================================================================
# TODO 1.2 — SMOTE (meta2_balance.py)
# ======================================================================
"""
Implementação esperada:
- selecionar janelas/segmentos da atividade A
- interpolar entre vizinhos próximos
"""

# ======================================================================
# TODO 1.3 — VISUALIZAR SÍNTESE DA ATIVIDADE 4 DO PARTICIPANTE 3
# ======================================================================

print("\n=== 1.3 – Gerar e Visualizar Amostras Sintéticas ===")
=======

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
>>>>>>> nigga

activity_for_aug = 4
K_aug = 3

<<<<<<< HEAD
=======
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


>>>>>>> nigga
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


# ======================================================================
# TODO 2 — EXTRAÇÃO DE EMBEDDINGS
# ======================================================================

<<<<<<< HEAD
print("\n=== 2.1 – Extrair Embeddings ===")

try:
    accel_data = dados[:, 1:4].astype(float)
    sr = fe.sampling_rate_calculator(dados)

    embeddings_X, embeddings_y = extract_embeddings_dataset(
        accel_data=accel_data,
        activities=activities,
=======
print("\n=== 2.1 - Extrair Embeddings ===")

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
>>>>>>> nigga
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


# ======================================================================
# TODO 3 — DATA SPLITTING (WITHIN + BETWEEN SUBJECT)
# ======================================================================
"""
Task 3.1 – TVT 60-20-20 within subject
Task 3.2 – 9 train / 3 val / 3 test between subjects
Task 3.3 – Discuss differences (written)
Task 3.4 – Prepare datasets:
      a) full
      b) PCA → 90%
      c) ReliefF → top 15
"""


# ======================================================================
# TODO 4 — MODEL LEARNING (kNN)
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
