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
