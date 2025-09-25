import csv

participante = 1

nome_pasta = 'dataset/part' + str(participante)

#print(nome_pasta)

for i in range(1,5):
    with open(nome_pasta + 'dev' + str(i) + '.csv') as csvfile:
        reading = csv(('part' + str(participante) + 'dev' + str(i)), delimiter='\t')
        for row in reading:
            print(row)


