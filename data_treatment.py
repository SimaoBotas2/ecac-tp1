import csv
import os
import numpy as np
import math
import matplotlib.pyplot as plt

def get_data(participante=0):

    # mudar isto para modular, participante, sensor, atividade
    nome_pasta = os.path.join("dataset", "part" + str(participante))

    dados = []
    # ainda n sei se é só um sensor ou nao
    # alterar aqui para um for ou o i só para 1 para o numero de sensores
    i = 1
    #for i in range(1,6):
    arquivo = os.path.join(nome_pasta, "part" + str(participante) + "dev" + str(i) + ".csv")
    with open(arquivo, newline="", encoding="utf-8") as csvfile:
            reading = csv.reader(csvfile, delimiter=",")
            data = list(reading)
            dados.extend(data)  

    dados_np = np.array(dados)
    return dados_np


def calculate_zscores(array, k):
    mean = np.mean(array)
    std = np.std(array)

    z_scores = (array - mean) / std
    outliers = array[np.abs(z_scores) > k]

    return outliers


def plot_outliers_per_activity(sensor_info, activities, k_values):
    unique_activities = np.unique(activities)
    
    for label, sensor_data in sensor_info.items():
        n_k = len(k_values)
        fig, axes = plt.subplots(1, n_k, figsize=(5 * n_k, 4), sharey=True)
        if n_k == 1:
            axes = [axes]

        for i in range(n_k):
            ax = axes[i]
            k = k_values[i]

            x_positions = []
            y_values = []
            colors = []

            for j, activity in enumerate(unique_activities):
                mask = activities == activity
                activity_data = sensor_data[mask]

                outliers = calculate_zscores(activity_data, k)
                is_outlier = np.isin(activity_data, outliers)

                x_activity = np.full_like(activity_data, j)
                x_positions.append(x_activity)
                y_values.append(activity_data)
                colors.append(np.where(is_outlier, 'red', 'blue'))

            x_positions = np.concatenate(x_positions)
            y_values = np.concatenate(y_values)
            colors = np.concatenate(colors)

            ax.scatter(x_positions, y_values, c=colors, alpha=0.6, s=15)

            ax.set_xticks(range(len(unique_activities)))
            ax.set_xticklabels([f"A{int(a)}" for a in unique_activities])
            ax.set_xlabel("Atividades")
            ax.set_title(f"{label} (k={k})")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("Módulo")
        plt.tight_layout()
        plt.show()


