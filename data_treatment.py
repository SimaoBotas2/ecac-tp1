import csv
import os
import numpy as np
import math
import matplotlib.pyplot as plt

#Note:
    #The docstrings in this document were written by us and refined by AI


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

def calculate_module(data, columns):
    xyz_data = data[:, columns].astype(float)
    return np.sqrt(np.sum(xyz_data**2, axis=1))

def calculate_modules(data):
    n_sensors = data.shape[1] // 3
    modules = np.zeros((data.shape[0], n_sensors))
    for i, x in enumerate(range(0, data.shape[1], 3)):
        modules[:,  i] = calculate_module(data, [x, x+1, x+2])
    return modules

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


def plot_outliers_zScore (sensor_info, activities, k_values):
    """
    Plota dados de cada sensor, destacando outliers por atividade via Z-score.

    Parâmetros
    ----------
    sensor_info : dict
        {nome_sensor: np.ndarray} com os dados de cada sensor (1D).
    activities : array-like
        Rótulos das atividades.
    k_values : list[float]
        Limiares de Z-score (cada valor gera um subplot).

    Notas
    -----
    - Outliers a vermelho, pontos normais a azul.
    - Cálculo de Z-score otimizado por atividade.
    """
    activities = np.asarray(activities)
    unique_activities = np.unique(activities)

    for label, sensor_data in sensor_info.items():
        print(f"\nSensor: {label}")
        n_k = len(k_values)
        fig, axes = plt.subplots(1, n_k, figsize=(5 * n_k, 4), sharey=True)
        axes = np.atleast_1d(axes)


        zscores_by_activity = {}
        for activity in unique_activities:
            # Seleciona apenas as amostras correspondentes a esta atividade
            mask = (activities == activity)
            data_act = sensor_data[mask]

            # Calcula o z-score: mede o quão distante cada valor está da média (em desvios padrão)
            mean = np.mean(data_act)
            std = np.std(data_act)
            z = (data_act - mean) / std

            # Guarda tudo num dicionário para acesso rápido mais tarde
            zscores_by_activity[activity] = {
                "mask": mask,
                "data": data_act,
                "zscore": z
            }

        for i, k in enumerate(k_values):
            ax = axes[i]
            x_positions, y_values, colors = [], [], []

            print(f"\nOutliers para K = {k}:")
            for j, activity in enumerate(unique_activities):
                mask, data_act, z = zscores_by_activity[activity]

                is_outlier = np.abs(z) > k #se está a distancia maior que k, é outlier
                n_outliers = np.sum(is_outlier)
                density = (n_outliers / len(data_act)) * 100 #cálculo da densidade
                print(f"{n_outliers} outliers na atividade {int(activity)} ({density:.2f}%)")

                x_positions.append(np.full_like(data_act, j))
                y_values.append(data_act)
                colors.append(np.where(is_outlier, 'red', 'blue'))

            ax.scatter(
                np.concatenate(x_positions),
                np.concatenate(y_values),
                c=np.concatenate(colors),
                alpha=0.6, s=15
            )

            ax.set_xticks(range(len(unique_activities)))
            ax.set_xticklabels([f"A{int(a)}" for a in unique_activities])
            ax.set_xlabel("Atividades")
            ax.set_title(f"{label} (k={k})")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("Módulo")
        plt.tight_layout()
        plt.show()
