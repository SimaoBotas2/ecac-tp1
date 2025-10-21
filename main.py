import data_treatment
import boxplot
import dbscan
import numpy as np
import k_means
import statistic_significance
import feature_extractor as fe
import feature_selection as fs
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
dados = data_treatment.get_data(all_participants,all_sensors) #type: ignore

print("Fim de get data")

# calcular o módulo dos sensores
modules = data_treatment.calculate_modules(dados[:, 1:10])
#modules = data_treatment.normalize_range(modules,0,1) #desconmentar ou comentar de acordo com o que se quer
normalizado = False  #mudar aqui se normalizarmos


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



# 3.4 Outliers por sensor e atividade usando o z score
k_values = [3,3.5,4]
data_treatment.plot_outliers_zScore(sensor_info, activities, k_values)




# para k-means e dbscan
atividades_selecionadas = 5 #mudar aqui o número da atividade a ver, também aceita array

# 3.6 K-means manual por atividade
k = 4  # número de clusters
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
)

#3.7.1 Dbscan (bónus)
if normalizado == True:
    eps = 0.04 #valor que encontrei melhor com os valores normalizados
else:
    eps = 3.2

dbscan_data, dbscan_clusters, dbscan_labels = dbscan.dbscan_cluster(modules,activities,atividades_selecionadas,eps)

dbscan.plot_dbscan_results_3d(dbscan_data,dbscan_clusters,dbscan_labels,atividades_selecionadas)


# 4.1 Análise de significância estatística
# F (ANOVA) = diferenças de MÉDIAS (maior = mais diferente)
# H (Kruskal) = diferenças de DISTRIBUIÇÕES (maior = mais diferente)
# p < 0.05 = significativo
statistic_significance.analyze_statistical_significance(modules, activities)


# 4.2 Extração de features temporais e espectrais

# Preparar dados dos sensores
accel_data = dados[:, 1:4].astype(float)  # Colunas 2-4
gyro_data = dados[:, 4:7].astype(float)   # Colunas 5-7
mag_data = dados[:, 7:10].astype(float)   # Colunas 8-10

sr = fe.sampling_rate_calculator(dados)

# Executar extração de features
X_features, y_labels, window_info = fe.extract_features_4_2(
    accel_data, gyro_data, mag_data, activities, sampling_rate=sr
)

# Verificar primeiras features
if len(X_features) > 0:
    print(f"\nPrimeira janela - {X_features.shape[1]} features:")
    print(f"Labels: {y_labels[:10]}...")  # Primeiros 10 labels
    
    # Salvar features para usar nos próximos pontos
    np.savetxt('features_X.csv', X_features, delimiter=',', fmt='%.6f')
    np.savetxt('features_y.csv', y_labels, delimiter=',', fmt='%d')
    print("Features guardadas em 'features_X.csv' e 'features_y.csv'")
    with open('features_info.txt', 'w') as f:
      f.write(f"Total janelas: {X_features.shape[0]}\n")
      f.write(f"Total features por janela: {X_features.shape[1]}\n")
      f.write(f"Sampling rate: {sr} Hz\n")
      f.write(f"Window size: {5 * sr} amostras\n")
    
    print("Metadados guardados em 'features_info.txt'")



# 4.3 Análise PCA
X_features = np.loadtxt('features_X.csv', delimiter=',')
X_pca, pca_model, scaler = fe.pca_analysis(X_features, target_variance=0.75)

#Guardar resultados PCA
np.savetxt('features_X_pca.csv', X_pca, delimiter=',', fmt='%.6f')
print("Features PCA guardadas em 'features_X_pca.csv'")

#4.5

#fisher
top10_fisher, fisher_scores = fs.fisher_score_selection(X_features, y_labels, top_n=10)
print("Top 10 Fisher Score:", top10_fisher)
print("Scores:", fisher_scores[top10_fisher])

# ReliefF
top10_relief, relief_scores = fs.reliefF_selection(X_features, y_labels, top_n=10, n_neighbors=10)
print("Top 10 ReliefF:", top10_relief)
print("Pesos:", relief_scores[top10_relief])


# Comparar resultados
common = set(top10_fisher).intersection(set(top10_relief))

if common:
    print(f"Features em comum entre Fisher e ReliefF: {common}")
else:
    print("Não existem features em comum entre Fisher e ReliefF.")
