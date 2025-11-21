import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from math import ceil
from utils.config import DEBUG

#Trabalho Realizado por:
  #Martim Alves Rodrigues da Costa Duarte nº 2021275991
  #Simão Tomás Botas Carvalho nº 2021223055

#Nota:
    #As docstrings deste documento foram escritas pelos autores e refinadas com ajuda de LLMs.
def dbscan_cluster(data, labels, atividades, eps=0.5, min_samples=20):
    """
    Aplica o algoritmo DBSCAN a um subconjunto de dados filtrado por atividades específicas
    e imprime o número e a densidade (%) de outliers por atividade.

    Parâmetros
    ----------
    data : np.ndarray
        Matriz de dados (amostras × features), já normalizada externamente.
    labels : np.ndarray
        Vetor de rótulos das atividades correspondentes a cada amostra.
    atividades : int ou list[int]
        Atividade(s) a incluir na filtragem.
    eps : float, opcional
        Distância máxima entre pontos vizinhos (default = 0.5).
    min_samples : int, opcional
        Número mínimo de pontos para formar um cluster (default = 5).

    Retorna
    -------
    data_filtrada : np.ndarray
        Subconjunto de dados correspondente às atividades selecionadas.
    clusters : np.ndarray
        Rótulos atribuídos pelo DBSCAN (-1 indica outliers).
    labels_filtrados : np.ndarray
        Rótulos originais das amostras filtradas.
    """

    # Garantir lista
    atividades = np.atleast_1d(atividades)

    # Filtrar dados
    mask = np.isin(labels, atividades)
    data_filtrada = data[mask]
    labels_filtrados = labels[mask]

    # Aplicar DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples,algorithm= 'ball_tree',n_jobs=-1)
    clusters = db.fit_predict(data_filtrada)

    unique_acts = np.unique(labels_filtrados)
    print("\nDensidade de outliers por atividade:")
    for act in unique_acts:
        mask_act = labels_filtrados == act
        clusters_act = clusters[mask_act]
        n_outliers_act = np.sum(clusters_act == -1)
        n_total_act = len(clusters_act)
        dens_act = (n_outliers_act / n_total_act) * 100 if n_total_act > 0 else 0
        print(f"  Atividade {int(act)}: {n_outliers_act} outliers em {n_total_act} pontos "
              f"({dens_act:.2f}%)")

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
    activities = np.asarray(activities)
    unique_activities = np.unique(activities)

    x_vals = []
    y_vals = []
    colors = []

    for act in unique_activities:
        mask = (activities == act)
        d_act = data[mask]
        c_act = clusters[mask]

        out = (c_act == -1)

        if d_act.ndim > 1:
            y = np.linalg.norm(d_act, axis=1)
        else:
            y = d_act

        x_vals.append(np.full_like(y, act))
        y_vals.append(y)
        colors.append(np.where(out, 'red', 'blue'))

    x_vals = np.concatenate(x_vals)
    y_vals = np.concatenate(y_vals)
    colors = np.concatenate(colors)

    plt.figure(figsize=(14, 6))
    plt.scatter(x_vals, y_vals, c=colors, alpha=0.6, s=20)

    plt.xticks(unique_activities, [f"A{int(a)}" for a in unique_activities])
    plt.xlabel("Atividade")
    plt.ylabel("Magnitude / Distância")
    plt.title("Outliers DBSCAN por Atividade")
    plt.grid(True, alpha=0.3)
    plt.show()

