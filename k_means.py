import numpy as np
import matplotlib.pyplot as plt

DEBUG = True
#note : String doc made by copilot and slightly modified by me

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
        new_centroids = np.zeros_like(centroids) #criar uma estrutura igual a anterior para guardar os clusters novos
        for i in range(k):
            if np.sum(clusters == i) > 0: #verificar se o cluster tem pelo menos um ponto associado
                cluster_points = data[clusters == i] #pegar os pontos que pertencem ao cluster
                new_centroids[i] = np.mean(cluster_points, axis = 0) #faz a media dos pontos do cluster para alinhar o novo cluster
            else:
                new_centroids[i] = clusters[i] # mantem um centroide nulo se o cluester ja estiver vazio

        # Verificar a convergencia dos centroides
        if np.allclose(centroids, new_centroids): # Funcao verifica a igualdade de dois arrays dentro de uma pequena margem de erro
            if DEBUG:
                print(f"K-means convergiu na iteracao {i}")
            break

        centroids = new_centroids

    # Calcular as distancias finais para detecao de outliers
    final_distances = np.zeros(n_samples)
    for i in range(n_samples):
        final_distances[i] = np.linalg.norm(data[i] - centroids[clusters[i]])
    
    return clusters, centroids, final_distances

# 3.7   
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

def plot_kmeans_results_3d(data, clusters, centroids, outliers, title="K-means Clustering"):
    """
    Plots the results of K-means clustering in a 3D scatter plot, highlighting clusters, centroids, and outliers.
    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_samples, 3) containing the data points to plot.
    clusters : np.ndarray
        Array of shape (n_samples,) with cluster labels assigned to each data point.
    centroids : np.ndarray
        Array of shape (n_clusters, 3) containing the coordinates of cluster centroids.
    outliers : np.ndarray
        Boolean array of shape (n_samples,) indicating which data points are considered outliers.
    title : str, optional
        Title for the plot (default is "K-means Clustering").
    Notes
    -----
    - Normal (non-outlier) points are colored by cluster.
    - Outliers are marked with black 'x'.
    - Centroids are shown as large yellow stars.
    - The axes are labeled according to the features: acceleration, gyroscope, and magnetometer.
    """

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot pontos normais
    normal_points = data[~outliers]  #o "~" é um negativo ou seja todos os q nao sao outliers
    normal_clusters = clusters[~outliers]
    
    # Plot outliers
    outlier_points = data[outliers]
    
    # Cores para clusters
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot cada cluster
    for i in range(len(centroids)):
        cluster_points = normal_points[normal_clusters == i]
        if len(cluster_points) > 0:
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                      c=colors[i % len(colors)], label=f'Cluster {i}', alpha=0.6, s=20)
    
    # Plot outliers
    if len(outlier_points) > 0:
        ax.scatter(outlier_points[:, 0], outlier_points[:, 1], outlier_points[:, 2], 
                  c='black', marker='x', s=50, label='Outliers', linewidth=2)
    
    # Plot centróides
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
              c='yellow', marker='*', s=200, label='Centróides', edgecolors='black')
    
    ax.set_xlabel('Módulo Aceleração')
    ax.set_ylabel('Módulo Giroscópio')
    ax.set_zlabel('Módulo Magnetómetro')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

