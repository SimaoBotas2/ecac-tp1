import csv
import os
import numpy as np
import math
import matplotlib.pyplot as plt

def get_data(participante=0):
    nome_pasta = os.path.join("dataset", "part" + str(participante))

    dados = []
    
    # ainda n sei se é só um sensor ou nao
    for i in range(1, 6):
        arquivo = os.path.join(nome_pasta, "part" + str(participante) + "dev" + str(i) + ".csv")
        with open(arquivo, newline="", encoding="utf-8") as csvfile:
            reading = csv.reader(csvfile, delimiter=",")
            data = list(reading)
            dados.extend(data)  

    dados_np = np.array(dados)
    #print(dados_np)
    return dados_np


def calculate_zscores(array, k):
    mean = np.mean(array)
    std = np.std(array)

    z_scores = (array - mean) / std
    outliers = array[np.abs(z_scores) > k]

    return outliers


def plot_outliers(sensor_info, k_values):
    """
    Cria subplots para cada sensor (cada entrada no dicionário sensor_info),
    mostrando os outliers (vermelho) e valores normais (azul),
    para vários valores de k.
    """
    for label, sensor_data in sensor_info.items():
        n_k = len(k_values)
        fig, axes = plt.subplots(1, n_k, figsize=(5 * n_k, 4), sharey=True)
        if n_k == 1:
            axes = [axes]

        for i in range(n_k):
            ax = axes[i]
            k = k_values[i]


            # aqui deve dar pra melhorar performance
            outliers = calculate_zscores(sensor_data, k)
            is_outlier = np.isin(sensor_data, outliers)

            ax.scatter(
                np.arange(len(sensor_data)),
                sensor_data,
                c=np.where(is_outlier, 'red', 'blue'),
                alpha=0.6,
                s=15
            )

            ax.set_title(f"{label} (k={k})")
            ax.set_xlabel("Amostras")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("Módulo")
        plt.tight_layout()
        plt.show()


