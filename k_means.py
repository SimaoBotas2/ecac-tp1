import numpy as np
import matplotlib.pyplot as plt

DEBUG = True

#3.6
def k_means_manual(data, k, max_iters=100):
    """
    Performs K-means clustering on the given dataset using a manual implementation.
    Args:
        data (np.ndarray): The input data array of shape (n_samples, n_features).
        k (int): The number of clusters to form.
        max_iters (int, optional): Maximum number of iterations for the algorithm. Default is 100.
    Returns:
        clusters (np.ndarray): Array of shape (n_samples,) with the cluster index assigned to each sample.
        centroids (np.ndarray): Array of shape (k, n_features) with the final centroid positions.
        final_distances (np.ndarray): Array of shape (n_samples,) with the distance of each sample to its assigned centroid.
    Notes:
        - Centroids are initialized randomly from the data points.
        - The algorithm iteratively assigns points to the nearest centroid and updates centroids until convergence or max_iters.
        - If a cluster becomes empty, its centroid remains unchanged.
        - The function also computes the final distances of each sample to its assigned centroid for outlier detection.
    """

    # Inicializar os centroides
    n_samples = data.shape[0]  #calcular o numero de samples que existem no dataset
    indices = np.random.choice(n_samples, k, replace=False) #escolher indices random para serem os primeiros centroides
    centroids = data[indices] # Fazer os centroides com os indices

    for _ in range(max_iters):
        distances = np.zeros((n_samples, k)) #Cria uma matriz para guardar o valor da distancia para cada cluster

        for i in range(k):
            # distância = sqrt((x1-x2)² + (y1-y2)² + (z1-z2)²)
            distances[:, i] = np.linalg.norm(data- centroids[i], axis=1) # Calcula a norma de um vetor (distância entre os pontos e os centroides)
    
        clusters = np.argmin(distances, axis=1) # Diz a que grupo pertence cada ponto (Ex: Linha (ponto 1) vai pelas colunas (clusters) e retorna o indice (cluster) da distancia mais pequena dessa coluna)

        # Recalcular centroides
        new_centriodes = np.zeros_like(clusters) #criar uma estrutura igual a anterioe para guardar os clusters novos
        for i in range(k):
            if np.sum(clusters == i) > 0: #verificar se o cluster tem pelo menos um ponto associado
                new_centriodes[i] = np.mean(data[clusters == i], axis = 0) #faz a media dos pontos do cluster para alinhar o novo cluster
            else:
                new_centriodes[i] = clusters[i] # mantem um centroide nulo se o cluester ja estiver vazio

        # Verificar a convergencia dos centroides
        if np.allclose(centroids, new_centriodes): # Funcao verifica a igualdade de dois arrays dentro de uma pequena margem de erro
            if DEBUG:
                print(f"K-means convergiu na iteracao {i}")
            break

        centroids = new_centriodes

    # Calcular as distancias finais para detecao de outliers
    final_distances = np.zeros(n_samples)
    for i in range(n_samples):
        final_distances = np.linalg.norm(data[i] - centroids[clusters[i]])
    
    return clusters, centroids, final_distances

    
def detect_outliers_kmeans(distances, threshold_std=2):
    """
    Detects outliers in a set of distances using the k-means method.
    Parameters:
        distances (array-like): Array of distances to cluster centers.
        threshold_std (float, optional): Number of standard deviations above the mean to use as the outlier threshold. Default is 2.
    Returns:
        numpy.ndarray: Boolean array indicating which distances are considered outliers.
    """

    mean_distances = np.mean(distances)
    std = np.std(distances)
    threshold = mean_distances + std * threshold_std

    outliers = distances > threshold
    return outliers




