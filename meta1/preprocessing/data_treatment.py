import csv
import os
import numpy as np
import matplotlib.pyplot as plt

#Trabalho Realizado por:
  #Martim Alves Rodrigues da Costa Duarte nº 2021275991
  #Simão Tomás Botas Carvalho nº 2021223055

#Nota:
    #As docstrings deste documento foram escritas pelos autores e refinadas com ajuda de LLMs.


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

def get_data(participante=0, sensor=1, return_participants=False):
    """
    Lê dados CSV de um ou mais participantes, com opção de filtrar por sensor.
    Agora assume que os dados estão em: data/raw/dataset/partX/partXdevY.csv
<<<<<<< HEAD
    """

    if not isinstance(participante, list):
        participante = [participante]

=======

    Se return_participants=True, devolve também um vetor com o ID do participante
    correspondente a cada linha da matriz devolvida.
    """

    if not isinstance(participante, list):
        participante = [participante]

>>>>>>> nigga
    if not isinstance(sensor, list):
        sensor = [sensor]

    dados = []
    participant_ids = []

    # BASE_DIR = raiz do projeto = 2 níveis acima deste ficheiro (meta1/preprocessing)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    dataset_root = os.path.join(base_dir, "data", "raw", "dataset")

    # BASE_DIR = raiz do projeto = 2 níveis acima deste ficheiro (meta1/preprocessing)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    dataset_root = os.path.join(base_dir, "data", "raw", "dataset")

    for p in participante:
        nome_pasta = os.path.join(dataset_root, "part" + str(p))
        for s in sensor:
            arquivo = os.path.join(nome_pasta, f"part{p}dev{s}.csv")
            with open(arquivo, newline="", encoding="utf-8") as csvfile:
                reading = csv.reader(csvfile, delimiter=",")
                data = list(reading)
                dados.extend(data)
<<<<<<< HEAD

    dados_np = np.array(dados)
=======
                participant_ids.extend([p] * len(data))

    dados_np = np.array(dados)
    if return_participants:
        return dados_np, np.array(participant_ids, dtype=int)
>>>>>>> nigga
    return dados_np