import data_treatment
import boxplot
import numpy as np
import k_means


# 1 Get dados em np array
dados = data_treatment.get_data(sensor=[1,2,3])

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


#3.4
k_values = [1,3]
data_treatment.plot_outliers_per_activity(sensor_info,activities,k_values)

# 3.6 K-means manual
k = 4  # número de clusters
clusters, centroids, distances = k_means.k_means_manual(modules, k)
outliers = k_means.detect_outliers_kmeans(distances, threshold_std=2)

if k_means.DEBUG:
    print(f"Encontrados {np.sum(outliers)} outliers com k={k}")

# 3.7 Plot 3D
k_means.plot_kmeans_results_3d(modules, clusters, centroids, outliers, title=f"K-means com k={k}")

