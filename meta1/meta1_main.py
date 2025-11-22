# ======================== META 1 ================================
# ECAC 2025 – META 1
# Autores: Simão Tomás Botas Carvalho nº 2021223055
#          Martim Costa Duarte nº 2021275991
#          Estrutura do projeto refinada com LLM (CHAT-GPT 5)

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# IMPORTS AJUSTADOS À ESTRUTURA NOVA
from meta1.preprocessing import data_treatment, statistic_significance
from meta1.outliers import boxplot, dbscan, k_means, zscore as z
from meta1.features import feature_extractor as fe, feature_selection as fs
from utils.config import DEBUG

# Todos os participantes para 3.1 (como exige o enunciado)
all_participants = list(range(15))
all_sensors = [1, 2, 3, 4, 5]

# Participante para features / Kmeans / DBSCAN
participant_selected = 3
sensors_selected = all_sensors

# ======================================================================
# 1) CARREGAR DADOS
# ======================================================================

print("\n=== 1. CARREGAR DADOS ===")

# Para 3.1: TODOS os participantes e TODOS os sensores
dados_all = data_treatment.get_data(all_participants, all_sensors)
activities_all = dados_all[:, 11].astype(int)
modules_all = data_treatment.calculate_modules(dados_all[:, 1:10])

# Para restante análise: apenas um participante (3) com todos os sensores
dados = data_treatment.get_data(participant_selected, sensors_selected)
activities = dados[:, 11].astype(int)
modules = data_treatment.calculate_modules(dados[:, 1:10])

print("Dados carregados.")

# ======================================================================
# 3.1) BOXPLOTS POR ATIVIDADE PARA TODOS OS PARTICIPANTES
# ======================================================================

print("\n=== 3.1 BOXPLOTS POR ATIVIDADE (TODOS OS PARTICIPANTES) ===")

labels = ["Aceleração", "Giroscópio", "Magnetómetro"]

sensor_info_all = {
    label: modules_all[:, i]
    for i, label in enumerate(labels)
}

for label, data_sensor in sensor_info_all.items():
    print(f"\n--- Boxplot: {label} ---")
    boxplot.create_boxplot_per_activity(data_sensor, activities_all, label)

# ======================================================================
# 3.3) Z-SCORE
# ======================================================================

print("\n=== 3.3 / 3.4 – Z-SCORE OUTLIERS ===")

k_values = [3, 3.5, 4]

sensor_info_single = {
    label: modules[:, i]
    for i, label in enumerate(labels)
}

z.plot_outliers_zScore(sensor_info_single, activities, k_values)

# ======================================================================
# 3.6 / 3.7) K-MEANS & OUTLIERS
# ======================================================================

print("\n=== 3.6 / 3.7 – K-MEANS ===")

atividades_selecionadas = list(range(1, 17))
modules_norm = data_treatment.normalize_range(modules, 0, 1)

k = 4
mods_filt, clusters, centroids, distances, labels_filt = k_means.k_means_manual(
    modules_norm, activities, atividades_selecionadas, k
)

outliers_kmeans = k_means.detect_outliers_kmeans(distances, threshold_std=2)

k_means.print_outlier_density_per_activity(
    labels_filt, outliers_kmeans, atividades_selecionadas
)

k_means.plot_kmeans_results_3d(
    mods_filt, clusters, centroids, outliers_kmeans, labels_filt, atividades_selecionadas
)

k_means.plot_kmeans_outliers(distances, labels_filt, outliers_kmeans)

# ======================================================================
# 3.7.1) DBSCAN (BÓNUS) – comentado se não quiseres correr sempre
# ======================================================================
"""
print("\n=== 3.7.1 – DBSCAN (BÓNUS) ===")

eps = 0.04
db_data, db_clusters, db_labels = dbscan.dbscan_cluster(
    modules_norm, activities, atividades_selecionadas, eps
)

dbscan.plot_dbscan_results_3d(db_data, db_clusters, db_labels, atividades_selecionadas)
dbscan.plot_dbscan_outliers(db_data, db_clusters, db_labels)
"""

# ======================================================================
# 4.1) SIGNIFICÂNCIA ESTATÍSTICA
# ======================================================================

print("\n=== 4.1 – SIGNIFICÂNCIA ESTATÍSTICA ===")

modules_norm2 = data_treatment.normalize_range(modules)
statistic_significance.analyze_statistical_significance(modules_norm2, activities)

# ======================================================================
# 4.2) EXTRAÇÃO DE FEATURES TEMPORAIS + ESPECTRAIS
# ======================================================================

print("\n=== 4.2 – EXTRAÇÃO DE FEATURES ===")

accel = dados[:, 1:4].astype(float)
gyro = dados[:, 4:7].astype(float)
mag  = dados[:, 7:10].astype(float)

sr = fe.sampling_rate_calculator(dados)

Xfeat, yfeat, winfo = fe.extract_features_4_2(accel, gyro, mag, activities, sr)

np.savetxt(DATA_PROCESSED / "features_X.csv", Xfeat, delimiter=",")
np.savetxt(DATA_PROCESSED / "features_y.csv", yfeat, delimiter=",")

print(f"Features extraídas. Shape: {Xfeat.shape}")

# ======================================================================
# 4.3) PCA
# ======================================================================

print("\n=== 4.3 – PCA ===")

X_features = np.loadtxt(DATA_PROCESSED / "features_X.csv", delimiter=",")
X_pca, pca_model, scaler = fe.pca_analysis(X_features, target_variance=0.75)

np.savetxt(DATA_PROCESSED / "features_X_pca.csv", X_pca, delimiter=",")

# ======================================================================
# 4.5 / 4.6) FISHER SCORE & RELIEFF
# ======================================================================

print("\n=== 4.5 / 4.6 – FISHER vs RELIEFF ===")

top10_fisher, fisher_scores = fs.fisher_score_selection(X_features, yfeat, top_n=10)
top10_relief, relief_scores = fs.reliefF_selection(X_features, yfeat, top_n=10)

# reconstruir nome das features (primeira janela)
try:
    window_size = 5 * sr
    feat_dict = fe.FeatureExtractor.extract_window_features(
        accel[:window_size], gyro[:window_size], mag[:window_size], sr
    )
    feature_names = list(feat_dict.keys())
except Exception:
    feature_names = [f"f{i}" for i in range(X_features.shape[1])]

fs.print_selection("Fisher Score", np.array(top10_fisher), fisher_scores, feature_names)
fs.print_selection("ReliefF", np.array(top10_relief), relief_scores, feature_names)

print("\nFeatures em comum:", set(top10_fisher) & set(top10_relief))

print("\n=== META 1 COMPLETA ===\n")
