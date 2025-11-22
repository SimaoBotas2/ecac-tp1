from sys import modules
import matplotlib.pyplot as plt
import numpy as np

#Trabalho Realizado por:
  #Martim Alves Rodrigues da Costa Duarte nº 2021275991
  #Simão Tomás Botas Carvalho nº 2021223055


#Nota:
    #As docstrings deste documento foram escritas pelos autores e refinadas com ajuda de LLMs.


ylabel = ["|aceleracao|", "giroscopio", "magnetometro"]

        
def create_boxplot_per_activity(modules, activities, sensor_name):
    """
    Cria UM boxplot para UM sensor específico, mostrando todas as atividades
    
    Parameters:
    - modules: array com módulos [acel, giro, mag]
    - activities: coluna 12 com rótulos dasg atividades  
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
    bp = plt.boxplot(boxplot_data, patch_artist=True)
    plt.title(f'Módulo de {sensor_name} por Atividade')
    plt.xlabel('Atividades')
    plt.ylabel(f'Módulo de {sensor_name}')
    plt.grid(True, alpha=0.3)
    plt.xticks(ticks=range(1, len(activity_labels)+1), labels=activity_labels, rotation=45)
    plt.tight_layout()
    plt.show()
    
    #Outliers e densidade

    outlier_print = 0 # MUDAR PARA 0 PRA PRINTAR A DENSIDADE

    if outlier_print == 0:
        print(f'Outliers detectados e densidade (%) do {sensor_name}:')
        for i, outlier in enumerate(bp["fliers"]):
            outliers = outlier.get_ydata()
            n_outliers = len(outliers)
            n_total = len(boxplot_data[i])
            density = (n_outliers / n_total) * 100 if n_total > 0 else 0
            print(f"{activity_labels[i]}: {n_outliers} outliers em {n_total} pontos "
                    f"({density:.2f}%)")
