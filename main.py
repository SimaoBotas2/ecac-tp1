from read_data import get_data
import boxplot

dados = get_data(participante=1)

print(dados[0])

# calcular o módulo dos sensores
modules = boxplot.calculate_modules(dados[:, 1:10])

print(modules)

