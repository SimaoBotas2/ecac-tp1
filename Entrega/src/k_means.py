import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from config import DEBUG

#Nota:
    #As docstrings deste documento foram escritas pelos autores e refinadas com ajuda de LLMs.

def k_means_manual(data, labels, atividades, k, max_iters=100):
    """
    Implementação manual do algoritmo k-means filtrando por atividades específicas.
    
    Parameters:
    - data: array numpy com shape (n_amostras, n_features)
    - labels: array com labels/classes de cada amostra
    - atividades: lista ou int com atividades a considerar
    - k: número de clusters
    - max_iters: número máximo de iterações

    Returns:
    - data_filtrada: dados apenas das atividades selecionadas
    - clusters: labels dos clusters atribuídos
    - centroids: centróides finais
    - final_distances: distâncias de cada ponto ao seu centróide
    - labels_filtrados: labels originais filtrados
    """

    # Se atividades for um int, transforma numa lista
    if isinstance(atividades, int):
        atividades = [atividades]

    # Criar máscara para filtrar apenas as amostras das atividades selecionadas
    mask = np.isin(labels, atividades)
    data_filtrada = data[mask]        # dados filtrados
    labels_filtrados = labels[mask]   # labels filtrados

    # calcular o numero de samples que existem no dataset filtrado
    n_samples = data_filtrada.shape[0]

    # escolher indices random para serem os primeiros centroides
    indices = np.random.choice(n_samples, k, replace=False)
    # Fazer os centroides com os indices
    centroids = data_filtrada[indices]

    for _ in range(max_iters):
        # Cria uma matriz para guardar o valor da distancia para cada cluster
        distances = np.linalg.norm(data_filtrada[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)

        # Diz a que grupo pertence cada ponto (Ex: Linha (ponto 1) vai pelas colunas (clusters) e retorna o indice (cluster) da distancia mais pequena dessa coluna)
        clusters = np.argmin(distances, axis=1)

        # Recalcular centroides
        new_centroids = np.zeros_like(centroids) # criar uma estrutura igual a anterioe para guardar os clusters novos
        for i in range(k):
            if np.sum(clusters == i) > 0: # verificar se o cluster tem pelo menos um ponto associado
                cluster_points = data_filtrada[clusters == i]
                # faz a media dos pontos do cluster para alinhar o novo cluster
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # mantem um centroide nulo se o cluester ja estiver vazio
                new_centroids[i] = centroids[i]

        # Funcao verifica a igualdade de dois arrays dentro de uma pequena margem de erro
        if np.allclose(centroids, new_centroids):
            if DEBUG:
                print(f"K-means convergiu na iteracao {_}")
            break

        # Atualiza os centroides
        centroids = new_centroids

    # Calcular as distancias finais para detecao de outliers
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

#teste
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

def plot_kmeans_outliers(distances, activities, outliers):
    """
    Plota as distâncias de cada ponto ao seu centróide, com subplots separados por atividade,
    destacando outliers em vermelho e pontos normais em azul.

    Parâmetros
    ----------
    distances : np.ndarray
        Distâncias finais dos pontos aos seus centróides (1D).
    activities : np.ndarray
        Labels das atividades correspondentes a cada ponto.
    outliers : np.ndarray
        Boolean array indicando quais pontos são outliers.
    """
    activities = np.asarray(activities)
    unique_activities = np.unique(activities)
    n_activities = len(unique_activities)
    title = "Outliers K-means por Atividade"

    # grade com 2 linhas e colunas suficientes
    ncols = ceil(n_activities / 2)
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8), sharey=True)
    axes = np.atleast_1d(axes).flatten()  # garante vetor 1D

    for idx, activity in enumerate(unique_activities):
        ax = axes[idx]
        mask = activities == activity
        activity_distances = distances[mask]
        activity_outliers = outliers[mask]

        # Pontos normais
        ax.scatter(np.where(mask)[0][~activity_outliers], activity_distances[~activity_outliers],
                   color='blue', alpha=0.6, label='Normal')
        # Outliers
        ax.scatter(np.where(mask)[0][activity_outliers], activity_distances[activity_outliers],
                   color='red', alpha=0.8, label='Outlier')

        ax.set_title(f"Atividade {activity}")
        ax.set_xlabel("Índice da amostra")
        if idx % ncols == 0:  # só na primeira coluna
            ax.set_ylabel("Distância ao centróide")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for j in range(n_activities, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


def print_outlier_density_per_activity(labels, outliers, atividades=None):
    """
    Imprime a densidade de outliers (em %) por atividade para os resultados do K-means.

    Parâmetros
    ----------
    labels : np.ndarray
        Labels das atividades correspondentes a cada ponto (após filtro aplicado ao K-means).
    outliers : np.ndarray
        Array booleano com a marcação de outliers (True = outlier) para cada ponto.
    atividades : int | list[int] | None
        Subconjunto de atividades a considerar. Se None, usa todas presentes em `labels`.

    Saída
    -----
    Apenas imprime um resumo por atividade no formato:
      A{atividade}: X outliers em N pontos (Y%)
    """
    labels = np.asarray(labels).astype(int)
    outliers = np.asarray(outliers).astype(bool)

    if atividades is None:
        atividades = np.unique(labels)
    elif isinstance(atividades, int):
        atividades = [atividades]

    print("Outliers detectados e densidade (%) do K-means:")
    for atividade in atividades:
        mask = labels == atividade
        n_total = int(np.sum(mask))
        n_outliers = int(np.sum(outliers[mask]))
        density = (n_outliers / n_total) * 100 if n_total > 0 else 0.0
        print(f"A{int(atividade)}: {n_outliers} outliers em {n_total} pontos ({density:.2f}%)")
