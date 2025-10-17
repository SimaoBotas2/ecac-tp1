import numpy as np
import matplotlib.pyplot as plt
from config import DEBUG

#note : docstring made by copilot and slightly modified by me

#3.6
def k_means_manual(data, k, max_iters=100):
    """
    Executa o agrupamento K-means no conjunto de dados fornecido através de uma implementação manual.

    Parâmetros
    ----------
    data : np.ndarray
        Array de entrada com formato (n_amostras, n_características).
    k : int
        Número de clusters a formar.
    max_iters : int, opcional
        Número máximo de iterações do algoritmo. O valor por defeito é 100.

    Retorna
    -------
    clusters : np.ndarray
        Array com formato (n_amostras,) que indica o índice do cluster atribuído a cada amostra.
    centroids : np.ndarray
        Array com formato (k, n_características) contendo as posições finais dos centróides.
    final_distances : np.ndarray
        Array com formato (n_amostras,) contendo a distância de cada amostra ao seu centróide atribuído.

    Notas
    -----
    - Os centróides são inicializados aleatoriamente a partir dos pontos de dados.
    - O algoritmo atribui iterativamente os pontos ao centróide mais próximo e atualiza os centróides até convergir ou atingir o número máximo de iterações.
    - Se um cluster ficar vazio, o seu centróide permanece inalterado.
    - A função também calcula as distâncias finais de cada amostra ao respetivo centróide, úteis para deteção de outliers.
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
    Deteta outliers num conjunto de distâncias utilizando o método k-means.

    Parâmetros
    ----------
    distances : array-like
        Array com as distâncias aos centros dos clusters.
    threshold_std : float, opcional
        Número de desvios padrão acima da média a usar como limite para considerar um outlier. O valor por defeito é 2.

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

def plot_kmeans_results_3d(data, clusters, centroids, outliers, title="K-means Clustering"):
    """
    Plota os resultados do agrupamento K-means num gráfico 3D, destacando clusters, centróides e outliers.

    Parâmetros
    ----------
    data : np.ndarray
        Array de forma (n_amostras, 3) contendo os pontos de dados a serem plotados.
    clusters : np.ndarray
        Array de forma (n_amostras,) com os rótulos de cluster atribuídos a cada ponto.
    centroids : np.ndarray
        Array de forma (n_clusters, 3) contendo as coordenadas dos centróides dos clusters.
    outliers : np.ndarray
        Array booleano de forma (n_amostras,) que indica quais pontos de dados são considerados outliers.
    title : str, opcional
        Título do gráfico (por defeito é "Agrupamento K-means").

    Notas
    -----
    - Os pontos normais (não outliers) são coloridos de acordo com o cluster.
    - Os outliers são marcados com um 'x' preto.
    - Os centróides são mostrados como estrelas amarelas grandes.
    - Os eixos são rotulados conforme as variáveis: aceleração, giroscópio e magnetómetro.
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

