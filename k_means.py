import numpy as np
import matplotlib.pyplot as plt
from config import DEBUG

#note : docstring made by copilot and slightly modified by me

#Note:
    #The docstrings in this document were written by us and refined by AI


def k_means_manual(data, labels, atividades, k, max_iters=100):
    """
    Executa o agrupamento K-means apenas para as amostras correspondentes às atividades indicadas.
    """
    if isinstance(atividades, int):
        atividades = [atividades]

    mask = np.isin(labels, atividades)
    data_filtrada = data[mask]
    labels_filtrados = labels[mask]

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
    return data_filtrada, clusters, centroids, final_distances, labels_filtrados



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


def plot_kmeans_results_3d(data, clusters, centroids, outliers, labels, atividades, title="K-means por Atividade"):
    """
    Plota os resultados do K-means num gráfico 3D, gerando um plot separado para cada atividade.
    """
    if isinstance(atividades, int):
        atividades = [atividades]

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    for atividade in atividades:
        mask = labels == atividade
        if np.sum(mask) == 0:
            continue

        data_atividade = data[mask]
        clusters_atividade = clusters[mask]
        outliers_atividade = outliers[mask]

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        normal_points = data_atividade[~outliers_atividade]
        normal_clusters = clusters_atividade[~outliers_atividade]
        outlier_points = data_atividade[outliers_atividade]

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
        ax.set_title(f"{title} - Atividade {atividade}")
        ax.legend()
        plt.tight_layout()
        plt.show()

