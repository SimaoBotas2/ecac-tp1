import matplotlib.pyplot as plt


ylabel = ["|aceleracao|", "giroscopio", "magnetometro"]

def calculate_module(data):
    module = []
    for row in data:
        x = float(row[1])
        y = float(row[2])
        z = float(row[3])
        mod = (x**2 + y**2 + z**2)**0.5
        module.append(mod)
    return module

def create_boxplot(data, ylabel, title="Boxplot", xlabel="Category label"):
    plt.boxplot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()