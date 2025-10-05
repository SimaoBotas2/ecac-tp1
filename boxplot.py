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
        
def create_boxplot_per_activity(modules, activities, sensor_name):
    """
    Cria UM boxplot para UM sensor específico, mostrando todas as atividades
    
    Parameters:
    - modules: array com módulos [acel, giro, mag]
    - activities: coluna 12 com rótulos das atividades  
    - sensor_name: nome para o título
    """
    unique_activities = np.unique(activities)
    
    # Preparar dados: uma lista por atividade
    boxplot_data = []
    activity_labels = []
    
    for activity in unique_activities:
        mask = activities == activity
        activity_data = modules[mask]
        boxplot_data.append(activity_data)
        activity_labels.append(f"A{int(activity)}")


    # Criar o boxplot
    plt.figure(figsize=(14, 6))
    bp = plt.boxplot(boxplot_data, labels=activity_labels)
    plt.title(f'Módulo de {sensor_name} por Atividade')
    plt.xlabel('Atividades')
    plt.ylabel(f'Módulo de {sensor_name}')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    #Outliers e densidade

    outlier_print = 1 # MUDAR PARA 0 PRA PRINTAR A DENSIDADE

    if outlier_print == 0:
        print("Outliers detectados e densidade (%):")
        for i, outlier in enumerate(bp["fliers"]):
            outliers = outlier.get_ydata()
            n_outliers = len(outliers)
            n_total = len(boxplot_data[i])
            density = (n_outliers / n_total) * 100 if n_total > 0 else 0
            print(f"{activity_labels[i]}: {n_outliers} outliers em {n_total} pontos "
                    f"({density:.2f}%)")
