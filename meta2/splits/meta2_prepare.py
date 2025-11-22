"""Gerar ficheiro combinado de features apenas para atividades 1 a 7.

Este script lê `features_X.csv` e `features_y.csv` (gerados na meta 1),
filtra as janelas cuja atividade (label) está entre 1 e 7 (inclusive) e
cria um único ficheiro `meta2_features.csv` contendo todas as features
e, na última coluna, o label da atividade.

Formato de saída:
    Cada linha = [f0, f1, ..., fN-1, atividade]

Notas:
 - Não há cabeçalho (segue o padrão dos ficheiros originais).
 - Mantém a ordem original das features.
 - Caso seja necessário adicionar nomes de features, pode-se reconstruir
   tal como feito em `mainActivity.py` usando um exemplo de janela.
"""

from typing import Iterable
import numpy as np
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"

INPUT_X = DATA_PROCESSED / 'features_X.csv'
INPUT_Y = DATA_PROCESSED / 'features_y.csv'
INPUT_PART = DATA_PROCESSED / 'features_participant.csv'
OUTPUT_FILE = DATA_PROCESSED / 'meta2_features.csv'

def build_meta2(
    input_X: str = INPUT_X,
    input_y: str = INPUT_Y,
    output: str = OUTPUT_FILE,
    activities: Iterable[int] = range(1, 8)
):
    if not (os.path.exists(input_X) and os.path.exists(input_y) and os.path.exists(INPUT_PART)):
        raise FileNotFoundError("Ficheiros de entrada 'features_X.csv', 'features_y.csv' ou 'features_participant.csv' não encontrados.")

    # Ler matrizes de features e labels
    X = np.loadtxt(input_X, delimiter=',')
    y = np.loadtxt(input_y, delimiter=',')
    participants = np.loadtxt(INPUT_PART, delimiter=',')

    # Garantir que y é inteiro
    y = y.astype(int)

    activities = list(activities)
    mask = np.isin(y, activities)

    X_filtered = X[mask]
    y_filtered = y[mask]
    participants_filtered = participants[mask]

    combined = np.column_stack((X_filtered, participants_filtered, y_filtered))

    # Guardar (todas as features + coluna participante + label no fim)
    np.savetxt(output, combined, delimiter=',', fmt='%.10e')

    print(
        f"Gerado '{output}' com {combined.shape[0]} janelas, {X_filtered.shape[1]} features, coluna participante e label."
    )
    print(f"Atividades incluídas: {sorted(set(y_filtered))}")

if __name__ == '__main__':
    build_meta2()
