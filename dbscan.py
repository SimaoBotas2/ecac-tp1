import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from config import DEBUG



#Note:
    #The docstrings in this document were written by us and refined by AI


def dbscan_cluster(data, labels, atividades, eps=0.5, min_samples=5):
    """
    Executa o agrupamento DBSCAN apenas para as amostras correspondentes às atividades indicadas.
    """
    if isinstance(atividades, int):
        atividades = [atividades]

    mask = np.isin(labels, atividades)
    data_filtrada = data[mask]
    labels_filtrados = labels[mask]

    db = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = db.fit_predict(data_filtrada)

    if DEBUG:
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_outliers = np.sum(clusters == -1)
        print("DBSCAN encontrou", n_clusters, "clusters e", n_outliers, "outliers")

    return data_filtrada, clusters, labels_filtrados


def plot_dbscan_results_3d(data, clusters, labels, atividades, title="DBSCAN por Atividade"):
    """
    Plota os resultados do DBSCAN num gráfico 3D, gerando um plot separado para cada atividade.
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

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i, c in enumerate(set(clusters_atividade)):
            if c == -1:
                pontos = data_atividade[clusters_atividade == c]
                ax.scatter(pontos[:, 0], pontos[:, 1], pontos[:, 2],
                           c='black', marker='x', s=50, label='Outliers', linewidth=2)
            else:
                pontos = data_atividade[clusters_atividade == c]
                ax.scatter(pontos[:, 0], pontos[:, 1], pontos[:, 2],
                           c=colors[i % len(colors)], label='Cluster ' + str(c), alpha=0.6, s=20)

        ax.set_xlabel('Módulo Aceleração')
        ax.set_ylabel('Módulo Giroscópio')
        ax.set_zlabel('Módulo Magnetómetro')
        ax.set_title(title + " - Atividade " + str(atividade))
        ax.legend()
        plt.tight_layout()
        plt.show()
