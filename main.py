import csv
import os
import numpy as np

participante = 0
nome_pasta = os.path.join("dataset", "part" + str(participante))


for i in range(1,6):
    arquivo = os.path.join(nome_pasta, "part" + str(participante) + "dev" + str(i) + ".csv")
with open(arquivo, newline="", encoding="utf-8") as csvfile:
    reading = csv.reader(csvfile, delimiter="\t")
    data = list(reading)
