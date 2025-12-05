# ECAC 2025 – TP1 (META 2)
# Autores: Simão Tomás Botas Carvalho nº 2021223055
#          Martim Costa Duarte nº 2021275991

import sys
from pathlib import Path
# GARANTE QUE O ROOT DO PROJETO ESTÁ NO PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from meta1.preprocessing import data_treatment
from meta1.features import feature_extractor as fe
from meta2.smote import meta2_balance
from meta2.embeddings.embeddings_extractor import (
    extract_embeddings_dataset,
    save_embeddings_to_csv,
)
from meta2.splits import data_splitter, scenario_builder
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

#print_section("META 2 - Preparar Dados")

# ======================================================================
# 0 — DEFINIR PARÂMETROS BASE
# ======================================================================
"""
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
"""
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
        print("Sem amostras após filtragem.")


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
# 2 — EXTRAÇÃO DE EMBEDDINGS
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
# 3 — DATA SPLITTING (WITHIN + BETWEEN SUBJECT)
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
# 4 — EXEMPLO: TREINAR UM ÚNICO MODELO kNN
# ======================================================================
"""
Exemplo de treino de um modelo kNN individual.
Altere os parâmetros abaixo para testar diferentes cenários.
"""

DATA_TYPE = "features"      # "features" ou "embeddings"
SPLIT_TYPE = "within"       # "within" ou "between"
SCENARIO = "all"            # "all", "pca" ou "relief"
K_VALUE = 3                # Número de vizinhos

DEBUG_SINGLE_MODEL = False  # Muda para True para executar este exemplo

if DEBUG_SINGLE_MODEL:
    print_section(f"4 - DEBUG: Modelo kNN | {DATA_TYPE} | {SPLIT_TYPE} | {SCENARIO} | k={K_VALUE}")
    
    from models.knn import confusion_matrix
    
    scenario_file = DATA_PROCESSED / "scenarios" / DATA_TYPE / SPLIT_TYPE / f"{SCENARIO}.npz"
    
    if scenario_file.exists():
        data = np.load(scenario_file)
        X_train = data['train_X']
        y_train = data['train_y']
        X_val = data['val_X']
        y_val = data['val_y']
        X_test = data['test_X']
        y_test = data['test_y']
        
        # Treinar
        knn = KNNClassifier(k=K_VALUE)
        knn.fit(X_train, y_train)
        
        # Avaliar
        y_pred_test = knn.predict(X_test)
        test_acc = (y_pred_test == y_test).mean()
        
        cm_test, classes = confusion_matrix(y_test, y_pred_test)
        
        print(f"Test Accuracy: {test_acc:.4f}\n")
        print("Confusion Matrix:")
        print("     " + "  ".join(f"A{c}" for c in classes))
        for i, true_class in enumerate(classes):
            row_str = "  ".join(f"{cm_test[i, j]:4d}" for j in range(len(classes)))
            print(f"A{true_class}  {row_str}")
    else:
        print(f"[ERRO] Ficheiro não encontrado: {scenario_file}")

# ======================================================================
# 5 — EVALUATION PIPELINE
# ======================================================================
# Treina todos os modelos em train + validation
# Seleciona melhor k por validation accuracy
# Avalia no test set (para within e between)

from models.evaluation import run_evaluation, run_with_params_cli

try:
    # Escolha: usar toda a pipeline (validação + retrain) OU apenas test com parâmetros dados
    USE_SINGLE_RETRAIN = False  # mudar para True para chamar apenas um cenario com um k
    #o cenario é definido na secção 4 acima
    if USE_SINGLE_RETRAIN:
        print("5 - Teste único (sem validação)")
        # Usa os parâmetros definidos acima na Secção 4
        evaluation_results = run_with_params_cli(DATA_PROCESSED, DATA_TYPE, SPLIT_TYPE, SCENARIO, K_VALUE)
    elif run_evaluation:
        print("5 - Evaluation pipeline completa")
        evaluation_results = run_evaluation(DATA_PROCESSED)
    else:
        print("[META2][5] Aviso: run_evaluation não pôde ser importado")
    
except (OSError, ValueError, RuntimeError, ImportError) as e:
    print(f"[META2][5] Erro na avaliação: {e}")
    import traceback
    traceback.print_exc()

# ======================================================================
# 6 — DEPLOYMENT FUNCTION
# ======================================================================
"""
from meta2.predict_new import evaluate_multiple_csvs, predict_from_array

# Task 6: Testar modelo com múltiplos CSVs
print_section("6 - Deployment com Dados CSV")

try:
    raw_array = np.random.randn(256, 9).astype(np.float32)  # Shape: (256 linhas, 9 colunas)
    result = predict_from_array(
        raw_data=raw_array,
        activity_label=4,  # label real da atividade (1-7)
        data_type='features',
        split_type='within',
        scenario='all',
        k=3,
        verbose=True,
    )
    print(f"Predição: A{result['activity_predicted']} (Real: A{result['activity_real']})")
    print(f"Correto: {result['is_correct']} (Acurácia: {result['accuracy']:.1%})")
    

    # Opção 2: Testar com participante fixo (part7)
    # results = evaluate_multiple_csvs(
    #     num_csvs=10,
    #     part_folder='part7',
    #     data_type='features',
    #     split_type='within',
    #     scenario='all',
    #     k=10,
    # )
    
    #Opção 3: Testar com participante ALEATÓRIO e devices ALEATÓRIOS
    results = evaluate_multiple_csvs(
         num_csvs=15,
         random_participant=True,  # Escolhe participante aleatório (0-14)
         random_device=True,       # Escolhe devices aleatórios (1-5)
         data_type='features',
         split_type='within',
         scenario='all',
         k=10,
    )

except Exception as e:
    print(f"[Task 6] Erro ao processar CSVs: {e}")
    import traceback
    traceback.print_exc()
"""


# ======================================================================
# TODO 7 — GO FURTHER (BONUS)
# ======================================================================
# Task 7:
# Extra improvements:
#     - Deep neural models
#     - LSTM/Transformer
#     - Augmentation improvements
#     - Ensembling models

