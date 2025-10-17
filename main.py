import data_treatment
import boxplot
import numpy as np
import k_means
import statistic_significance
import feature_extractor
from config import DEBUG
#Trabalho Realizado por:
  #Martim Alves Rodrigues da Costa Duarte nº 2021275991
  #Simão Tomás Botas Carvalho nº 2021223055

if DEBUG:
    print("DEBUG está ativo")


#usar estes arrays para chamar a função abaixo
#CUIDADO
all_participants = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
all_sensors = [1,2,3,4,5]

# 1 Get dados em np array
dados = data_treatment.get_data(participante=list(range(0, 15)), sensor=[1, 2, 3])  # type: ignore

# calcular o módulo dos sensores
modules = boxplot.calculate_modules(dados[:, 1:10])

# 2. Extrair atividades (coluna 12)
activities = dados[:, 11].astype(int)  # índice 11 = coluna 12

"""# 3. Criar boxplots para cada sensor e detetar outliers
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

# 4.1 Análise de significância estatística
# F (ANOVA) = diferenças de MÉDIAS (maior = mais diferente)
# H (Kruskal) = diferenças de DISTRIBUIÇÕES (maior = mais diferente)
# p < 0.05 = significativo
statistic_significance.analyze_statistical_significance(modules, activities)
"""

# 4.2 Extração de features temporais e espectrais
# 4.2 - Extração de features
# Preparar dados dos sensores
accel_data = dados[:, 1:4].astype(float)  # Colunas 2-4
gyro_data = dados[:, 4:7].astype(float)   # Colunas 5-7
mag_data = dados[:, 7:10].astype(float)   # Colunas 8-10

# Executar extração de features
X_features, y_labels, window_info = feature_extractor.extract_features_4_2(
    accel_data, gyro_data, mag_data, activities, sampling_rate=50
)

# Verificar primeiras features
if len(X_features) > 0:
    print(f"\nPrimeira janela - {X_features.shape[1]} features:")
    print(f"Labels: {y_labels[:10]}...")  # Primeiros 10 labels
    
    # Salvar features para usar nos próximos pontos
    np.save('features_X.npy', X_features)
    np.save('features_y.npy', y_labels)
    print("Features guardadas em 'features_X.npy' e 'features_y.npy'")