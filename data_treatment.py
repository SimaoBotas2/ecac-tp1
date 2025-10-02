import csv
import os
import numpy as np
import math

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

def calculate_media(array):
    sum = 0
    for i in array:
        sum += i
    media = sum/len(array)
    return media

def calculate_desvio(array):
    sum = 0
    for i in array:
        sum += i**2
    media2 = sum/len(array)
    media = calculate_media(array)

    variancia = media2 - (media**2)
    desvio = math.sqrt(variancia)

    return desvio
