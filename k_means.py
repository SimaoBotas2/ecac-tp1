import numpy as np
import matplotlib.pyplot as plt

DEBUG = True

def k_means_manual(data, k, max_iters=100):
    """
    Implementação manual do algoritmo k-means
    
    Parameters:
    - data: array numpy com shape (n_amostras, n_features)
    - k: número de clusters
    - max_iters: número máximo de iterações
    
    Returns:
    - clusters: array com labels dos clusters para cada ponto
    - centroids: array com os centróides finais
    - distances: distâncias de cada ponto ao seu centróide
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

    

        
    


