import csv
import os
import numpy as np
import math
import matplotlib.pyplot as plt

def get_data(participante=0):
    nome_pasta = os.path.join("dataset", "part" + str(participante))

    dados = []

    for i in range(1, 6):
            arquivo = os.path.join(nome_pasta, "part" + str(participante) + "dev" + str(i) + ".csv")
            with open(arquivo, newline="", encoding="utf-8") as csvfile:
                reading = csv.reader(csvfile, delimiter=",")
                data = list(reading)
                dados.extend(data)  

    dados_np = np.array(dados)
    #print(dados_np)
    return dados_np

def z_score(array, k = 1):
    media = np.mean(array)
    std = np.std(array)
    z_scores = (array - media) / std 

    outliers = z_scores[np.abs(z_scores) > k]

    return outliers


def plot_outliers(modules, labels, k_values):
    """
    Plota os módulos dos sensores em subplots, mostrando os outliers
    em vermelho para diferentes valores de k.
    
    Parameters:
    - modules: np.array (n_amostras x n_sensores)
    - labels: lista com nomes dos sensores
    - k_values: lista com valores de k para o Z-score
    """
    for i in range(modules.shape[1]):
        sensor_data = modules[:, i]
        
        n_k = len(k_values)
        fig, axes = plt.subplots(1, n_k, figsize=(5*n_k, 4), sharey=True)
        if n_k == 1:
            axes = [axes]
        
        for ax, k in zip(axes, k_values):
            z_scores = (sensor_data - np.mean(sensor_data)) / np.std(sensor_data)
            is_outlier = np.abs(z_scores) > k
            
            ax.scatter(np.arange(len(sensor_data)), sensor_data, 
                       c=np.where(is_outlier, 'red', 'blue'), alpha=0.6, s=15)
            ax.set_title(f"{labels[i]} - k={k}")
            ax.set_xlabel("Amostras")
            ax.grid(True, alpha=0.3)
        
        axes[0].set_ylabel("Módulo")
        plt.tight_layout()
        plt.show()


