import numpy as np
import matplotlib.pyplot as plt

DEBUG = True

#Note:
    #The docstrings in this document were written by us and refined by AI


def k_means_manual(data, labels, atividades, k, max_iters=100):
    """
    Executa o agrupamento K-means apenas para as amostras correspondentes às atividades indicadas.

    Parâmetros
    ----------
    data : np.ndarray
        Array com formato (n_amostras, n_características) contendo os dados de entrada.
    labels : np.ndarray
        Array com formato (n_amostras,) contendo os rótulos de atividade de cada amostra.
    atividades : int ou list[int]
        Atividade ou lista de atividades sobre as quais o K-means será aplicado.
    k : int
        Número de clusters a formar.
    max_iters : int, opcional
        Número máximo de iterações do algoritmo (por defeito = 100).

    Retorna
    -------
    data_filtrada : np.ndarray
        Dados correspondentes apenas às atividades selecionadas.
    clusters : np.ndarray
        Índice do cluster atribuído a cada ponto.
    centroids : np.ndarray
        Coordenadas finais dos centróides.
    final_distances : np.ndarray
        Distância de cada ponto ao seu centróide atribuído.
    """

    if isinstance(atividades, int):
        atividades = [atividades]

    mask = np.isin(labels, atividades)
    data_filtrada = data[mask]

    n_samples = data_filtrada.shape[0]
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = data_filtrada[indices]

    for _ in range(max_iters):
        distances = np.linalg.norm(data_filtrada[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        clusters = np.argmin(distances, axis=1)

        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            if np.sum(clusters == i) > 0:
                cluster_points = data_filtrada[clusters == i]
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[i] = centroids[i]

        if np.allclose(centroids, new_centroids):
            if DEBUG:
                print(f"K-means convergiu na iteracao {_}")
            break

        centroids = new_centroids

    final_distances = np.linalg.norm(data_filtrada - centroids[clusters], axis=1)
    return data_filtrada, clusters, centroids, final_distances


def detect_outliers_kmeans(distances, threshold_std=2):
    """
    Deteta outliers num conjunto de distâncias utilizando o método K-means.

    Parâmetros
    ----------
    distances : array-like
        Array com as distâncias aos centros dos clusters.
    threshold_std : float, opcional
        Número de desvios padrão acima da média a usar como limite para considerar um outlier (por defeito = 2).

    Retorna
    -------
    numpy.ndarray
        Array booleano que indica quais as distâncias consideradas outliers.
    """
    mean_distances = np.mean(distances)
    std = np.std(distances)
    threshold = mean_distances + std * threshold_std
    outliers = distances > threshold
    return outliers


def plot_kmeans_results_3d(data, clusters, centroids, outliers, title="K-means por Atividade"):
    """
    Plota os resultados do agrupamento K-means num gráfico 3D, destacando clusters, centróides e outliers.

    Parâmetros
    ----------
    data : np.ndarray
        Array de forma (n_amostras, 3) contendo os pontos de dados a serem plotados.
    clusters : np.ndarray
        Array de forma (n_amostras,) com os rótulos de cluster atribuídos a cada ponto.
    centroids : np.ndarray
        Array de forma (n_clusters, 3) contendo as coordenadas dos centróides.
    outliers : np.ndarray
        Array booleano de forma (n_amostras,) que indica quais pontos são outliers.
    title : str, opcional
        Título do gráfico.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    normal_points = data[~outliers]
    normal_clusters = clusters[~outliers]
    outlier_points = data[outliers]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    for i in range(len(centroids)):
        cluster_points = normal_points[normal_clusters == i]
        if len(cluster_points) > 0:
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                       c=colors[i % len(colors)], label=f'Cluster {i}', alpha=0.6, s=20)

    if len(outlier_points) > 0:
        ax.scatter(outlier_points[:, 0], outlier_points[:, 1], outlier_points[:, 2],
                   c='black', marker='x', s=50, label='Outliers', linewidth=2)

    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
               c='yellow', marker='*', s=200, label='Centróides', edgecolors='black')

    ax.set_xlabel('Módulo Aceleração')
    ax.set_ylabel('Módulo Giroscópio')
    ax.set_zlabel('Módulo Magnetómetro')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
