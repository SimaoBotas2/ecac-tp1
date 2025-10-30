import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from math import ceil
from config import DEBUG


#Nota:
    #As docstrings deste documento foram escritas pelos autores e refinadas com ajuda de LLMs.


def dbscan_cluster(data, labels, atividades, eps=0.5, min_samples=5):
    """
    Aplica DBSCAN a um subconjunto de dados filtrado por atividades específicas.

    Parâmetros
    ----------
    data : np.ndarray
        Dados de entrada (amostras × features).
    labels : np.ndarray
        Rótulos das atividades correspondentes a cada amostra.
    atividades : int ou lista de int
        Atividades a incluir na filtragem.
    eps : float, opcional
        Distância máxima entre pontos vizinhos (default=0.5).
    min_samples : int, opcional
        Número mínimo de pontos para formar um cluster (default=5).

    Retorna
    -------
    data_filtrada : np.ndarray
        Subconjunto de dados correspondente às atividades selecionadas.
    clusters : np.ndarray
        Labels atribuídos pelo DBSCAN (-1 para outliers).
    labels_filtrados : np.ndarray
        Labels originais das amostras filtradas.
    """


    #transformar em array para iterar
    if isinstance(atividades, int):
        atividades = [atividades]

    #criar mascara para as atividades selecionadas
    mask = np.isin(labels, atividades)
    data_filtrada = data[mask]
    labels_filtrados = labels[mask]

    #Usar a implementação do sklearn
    db = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = db.fit_predict(data_filtrada)

    if DEBUG:
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_outliers = np.sum(clusters == -1)
        print("DBSCAN encontrou", n_clusters, "clusters e", n_outliers, "outliers")

    return data_filtrada, clusters, labels_filtrados


def plot_dbscan_results_3d(data, clusters, labels, atividades, title="DBSCAN por Atividade"):
    """
    Plota os resultados do DBSCAN em 3D, criando um gráfico separado para cada atividade.

    Parâmetros
    ----------
    data : np.ndarray
        Dados de entrada (amostras × 3 features), normalmente os módulos dos sensores.
    clusters : np.ndarray
        Labels atribuídos pelo DBSCAN (-1 indica outliers).
    labels : np.ndarray
        Rótulos originais das atividades.
    atividades : int ou lista de int
        Atividades a incluir nos plots.
    title : str, opcional
        Título base do gráfico (default="DBSCAN por Atividade").

    Comportamento
    ------------
    - Cada atividade gera um gráfico 3D separado.
    - Cada cluster recebe uma cor distinta; outliers são marcados a preto.
    - Eixos representam os módulos do acelerómetro, giroscópio e magnetómetro.
    """

    if isinstance(atividades, int):
        atividades = [atividades]

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    for atividade in atividades:
        #criar mascara das atividades selecionadas
        mask = labels == atividade
        if np.sum(mask) == 0:
            continue

        #aplicar a mascara
        data_atividade = data[mask]
        clusters_atividade = clusters[mask]

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i, c in enumerate(set(clusters_atividade)): #usar enumerate para garantir a ordem dos clusters (visualização melhor)
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

def plot_dbscan_outliers(data, clusters, activities):
    """
    Plota os resultados do DBSCAN em 2D, mostrando a distância de cada ponto
    (ou índice) com subplots separados por atividade, destacando outliers em vermelho.

    Parâmetros
    ----------
    data : np.ndarray
        Dados de entrada (amostras × features). Se for multidimensional (>1D),
        o eixo x representará o índice da amostra.
    clusters : np.ndarray
        Labels atribuídos pelo DBSCAN (-1 indica outliers).
    activities : np.ndarray
        Labels das atividades correspondentes a cada amostra.
    """
    activities = np.asarray(activities)
    unique_activities = np.unique(activities)
    n_activities = len(unique_activities)
    title = "Outliers DBSCAN por Atividade"

    ncols = ceil(n_activities / 2)
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for idx, activity in enumerate(unique_activities):
        ax = axes[idx]
        mask = activities == activity
        activity_clusters = clusters[mask]
        activity_data = data[mask]

        # Determina outliers
        outliers = activity_clusters == -1
        normals = ~outliers

        # Se os dados forem multidimensionais, projetamos em 1D só pra visualização
        if activity_data.ndim > 1:
            y_vals = np.linalg.norm(activity_data, axis=1)
        else:
            y_vals = activity_data

        # Plot dos pontos
        ax.scatter(np.where(mask)[0][normals], y_vals[normals],
                   color='blue', alpha=0.6, label='Normal')
        ax.scatter(np.where(mask)[0][outliers], y_vals[outliers],
                   color='red', alpha=0.8, label='Outlier')

        ax.set_title(f"Atividade {activity}")
        ax.set_xlabel("Índice da amostra")
        if idx % ncols == 0:
            ax.set_ylabel("Magnitude / Distância (proxy)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for j in range(n_activities, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
