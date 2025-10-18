import csv
import os
import numpy as np
import math
import matplotlib.pyplot as plt

#Note:
    #The docstrings in this document were written by us and refined by AI


import numpy as np

def normalize_range(X, new_min=0, new_max=1):
    """
    Normaliza um array NumPy para o intervalo [new_min, new_max].

    Aplica normalização min–max a cada coluna do array, escalando os valores
    para o intervalo definido. Colunas com valores constantes são mantidas.

    Parâmetros
    ----------
    X : np.ndarray
        Array de entrada (amostras × variáveis).
    new_min : float, opcional
        Valor mínimo desejado. Por defeito é 0.
    new_max : float, opcional
        Valor máximo desejado. Por defeito é 1.

    Retorna
    -------
    np.ndarray
        Array normalizado no intervalo [new_min, new_max].
    """
    X = np.asarray(X, dtype=float)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    denom = np.where(X_max - X_min == 0, 1, X_max - X_min)
    X_scaled = (X - X_min) / denom * (new_max - new_min) + new_min
    return X_scaled



def get_data(participante=0, sensor=1):
    """
    Lê dados CSV de um ou mais participantes, com opção de filtrar por sensor.

    Parâmetros
    ----------
    participante : int ou list[int], opcional
        Participante(s) a carregar. O valor por defeito é 0.
    sensor : int ou list[int], opcional
        Sensor(es) a carregar. O valor por defeito é 1.

    Retorna
    -------
    numpy.ndarray
        Array NumPy com os dados combinados dos parâmetros selecionados.
"""
    #Verificação dos parametros de entrada e troca para uma lista para iteração (poupar código)

    if not isinstance(participante,list) :
            participante = [participante]

    if not isinstance(sensor,list):
            sensor = [sensor]

    dados = []

    for p in participante:
        nome_pasta = os.path.join("dataset", "part" + str(p))
        for s in sensor:
            arquivo = os.path.join(nome_pasta, "part" + str(p) + "dev" + str(s) + ".csv")
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
    """
    Plota os dados de cada sensor, destacando os outliers por atividade usando o Z-score.

    Parâmetros
    ----------
    sensor_info : dict
        {nome_sensor: np.array} contendo os dados de cada sensor.
    activities : array
        Rótulos das atividades para cada amostra.
    k_values : list
        Valores limite de Z-score; cada valor de k gera um subplot separado.
    
    Notas
    -----
    - Os outliers são mostrados a vermelho e os pontos normais a azul.
    - O eixo X representa as atividades.
    - O Z-score é calculado separadamente para cada atividade.
"""

    activities = np.array(activities)
    unique_activities = np.unique(activities)
    
    for label, sensor_data in sensor_info.items():
        print(f"\nSensor: {label}")
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

            print(f"\nOutliers para K = {k}:")
            for j, activity in enumerate(unique_activities):
                mask = activities == activity
                activity_data = sensor_data[mask]

                outliers = calculate_zscores(activity_data, k)
                is_outlier = np.isin(activity_data, outliers)

                # Calcular densidade de outliers
                # TODO talvez guardar a densidade calculada em cada para se poder comparar com o outro método
                density = (len(outliers) / len(activity_data)) * 100
                print(f"{len(outliers)} outliers detectados na atividade {int(activity)} ({density:.2f}%)")

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

