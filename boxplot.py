from sys import modules
import matplotlib.pyplot as plt
import numpy as np


ylabel = ["|aceleracao|", "giroscopio", "magnetometro"]

def calculate_module(data, columns):
    xyz_data = data[:, columns].astype(float)
    return np.sqrt(np.sum(xyz_data**2, axis=1))

def calculate_modules(data):
    n_sensors = data.shape[1] // 3
    modules = np.zeros((data.shape[0], n_sensors))
    for i, x in enumerate(range(0, data.shape[1], 3)):
        modules[:,  i] = calculate_module(data, [x, x+1, x+2])
    return modules
        


def create_boxplot(data, ylabel, xlabel, title="Boxplot"):
    plt.boxplot(data, labels = xlabel)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()