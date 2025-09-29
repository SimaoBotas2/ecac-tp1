import read_data
import boxplot


# 1 Get dados em np array
dados = read_data.get_data(participante=1)

# calcular o módulo dos sensores
modules = boxplot.calculate_modules(dados[:, 1:10])

# 2. Extrair atividades (coluna 12)
activities = dados[:, 11].astype(int)  # índice 11 = coluna 12

# 3. Criar boxplots para cada sensor e detetar outliers
labels = ["Aceleração", "Giroscópio", "Magnetômetro"]
for x in range(modules.shape[1]):
    boxplot.create_boxplot_per_activity(modules[:, x], activities, labels[x])

