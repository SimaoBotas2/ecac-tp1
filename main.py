import data_treatment
import boxplot
import numpy as np


# 1 Get dados em np array
dados = data_treatment.get_data(sensor=[1,2,3])

# calcular o módulo dos sensores
modules = boxplot.calculate_modules(dados[:, 1:10])

# 2. Extrair atividades (coluna 12)
activities = dados[:, 11].astype(int)  # índice 11 = coluna 12

# 3. Criar boxplots para cada sensor e detetar outliers
labels = ["Aceleração", "Giroscópio", "Magnetômetro"]

sensor_info = {
    label: modules[:, i]
    for i, label in enumerate(labels)
}

for label, data in sensor_info.items():
  boxplot.create_boxplot_per_activity(data, activities, label)


#3.4
k_values = [1,3]
data_treatment.plot_outliers_per_activity(sensor_info,activities,k_values)

