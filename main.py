import data_treatment
import boxplot
import numpy as np
import k_means
import statistic_significance

# Trabalho Realizado por:
# Martim Alves Rodrigues da Costa Duarte nº 2021275991
# Simão Tomás Botas Carvalho nº 2021223055

all_participants = list(range(0, 15))
all_sensors = [1, 2, 3, 4, 5]

# 1 Get dados em np array
dados = data_treatment.get_data(1, sensor=1)

# calcular o módulo dos sensores
modules = boxplot.calculate_modules(dados[:, 1:10])

# 2. Extrair atividades (coluna 12)
activities = dados[:, 11].astype(int)  # índice 11 = coluna 12

# 3. Criar boxplots para cada sensor e detetar outliers
labels = ["Aceleração", "Giroscópio", "Magnetômetro"]

sensor_info = {
    label: modules[:, i]
    for i, label in enumerate(labels)
}


for label, data in sensor_info.items():
    boxplot.create_boxplot_per_activity(data, activities, label)

"""

# 3.4 Outliers por sensor e atividade
k_values = [1, 3]
data_treatment.plot_outliers_per_activity(sensor_info, activities, k_values)

"""
# 3.6 K-means manual por atividade
k = 4  # número de clusters
atividades_selecionadas = [1,2]
modules_filtrados, clusters, centroids, distances, labels_filtrados = k_means.k_means_manual(
    data=modules,
    labels=activities,
    atividades=atividades_selecionadas,
    k=k
)

outliers = k_means.detect_outliers_kmeans(distances, threshold_std=2)

if k_means.DEBUG:
    print(f"Encontrados {np.sum(outliers)} outliers com k={k} nas atividades {atividades_selecionadas}")

# 3.7 Plot 3D K-means por atividade
k_means.plot_kmeans_results_3d(
    data=modules_filtrados,
    clusters=clusters,
    centroids=centroids,
    outliers=outliers,
    labels=labels_filtrados,
    atividades=atividades_selecionadas,
    title="K-means por Atividade"
)

# 4.1 Análise de significância estatística
#statistic_significance.analyze_statistical_significance(modules, activities)
