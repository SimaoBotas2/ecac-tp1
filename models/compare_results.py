# ======================== COMPARE kNN RESULTS ================================
# Comparar diferentes valores de k do kNN
# Analisa os resultados guardados em data/processed/results/

import sys
from pathlib import Path
import re
import pandas as pd

# Setup path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

RESULTS_DIR = ROOT / "data" / "processed" / "results"


def extract_metrics_from_file(file_path):
    """Extrai métricas (Test Accuracy) de um ficheiro de resultados."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Procurar por Test Accuracy
        match = re.search(r'Test Accuracy:\s+([\d.]+)', content)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"Erro ao ler {file_path.name}: {e}")
    
    return None


def parse_filename(filename):
    """Extrai informações do nome do ficheiro."""
    # Formato: knn_results_{data_type}_{scenario_split}_{scenario}_k{k}.txt
    pattern = r'knn_results_(\w+)_(\w+)_(\w+)_k(\d+)\.txt'
    match = re.match(pattern, filename)
    
    if match:
        data_type, split_type, scenario, k_value = match.groups()
        return {
            'data_type': data_type,
            'split_type': split_type,
            'scenario': scenario,
            'k': int(k_value)
        }
    return None


def compare_k_values():
    """Compara resultados para diferentes valores de k."""
    
    if not RESULTS_DIR.exists():
        print(f"Diretório de resultados não encontrado: {RESULTS_DIR}")
        return
    
    # Recolher todos os ficheiros de resultados
    result_files = sorted(RESULTS_DIR.glob("knn_results_*.txt"))
    
    if not result_files:
        print(f"Nenhum ficheiro de resultados encontrado em {RESULTS_DIR}")
        return
    
    print(f"Encontrados {len(result_files)} ficheiros de resultados\n")
    
    # Estrutura para guardar dados
    comparison_data = []
    
    for file_path in result_files:
        info = parse_filename(file_path.name)
        accuracy = extract_metrics_from_file(file_path)
        
        if info and accuracy is not None:
            comparison_data.append({
                'Data Type': info['data_type'].upper(),
                'Split': info['split_type'].upper(),
                'Scenario': info['scenario'].upper(),
                'k': info['k'],
                'Test Accuracy': accuracy
            })
    
    if not comparison_data:
        print("Nenhuma métrica foi extraída")
        return
    
    # Converter para DataFrame
    df = pd.DataFrame(comparison_data)
    
    print("=" * 80)
    print("COMPARAÇÃO DE RESULTADOS - DIFERENTES VALORES DE k")
    print("=" * 80 + "\n")
    
    # 1. Ordenar por Test Accuracy (melhor no topo)
    print("RANKING GERAL (Ordenado por Test Accuracy):\n")
    df_sorted = df.sort_values('Test Accuracy', ascending=False)
    print(df_sorted.to_string(index=False))
    print("\n")
    
    # 2. Comparação por tipo de dados
    print("=" * 80)
    print("COMPARAÇÃO POR TIPO DE DADOS:\n")
    
    for data_type in df['Data Type'].unique():
        df_data = df[df['Data Type'] == data_type]
        print(f"--- {data_type} ---")
        df_data_sorted = df_data.sort_values('Test Accuracy', ascending=False)
        print(df_data_sorted[['Split', 'Scenario', 'k', 'Test Accuracy']].to_string(index=False))
        print()
    
    # 3. Comparação por valor de k
    print("=" * 80)
    print("COMPARAÇÃO POR VALOR DE k:\n")
    
    for k in sorted(df['k'].unique()):
        df_k = df[df['k'] == k]
        avg_acc = df_k['Test Accuracy'].mean()
        max_acc = df_k['Test Accuracy'].max()
        min_acc = df_k['Test Accuracy'].min()
        
        print(f"k = {k}:")
        print(f"  Média: {avg_acc:.4f}")
        print(f"  Máximo: {max_acc:.4f}")
        print(f"  Mínimo: {min_acc:.4f}")
        print(f"  Cenários: {len(df_k)}")
        print()
    
    # 4. Melhor combinação
    print("=" * 80)
    print("MELHOR RESULTADO:\n")
    best_row = df_sorted.iloc[0]
    print(f"Data Type: {best_row['Data Type']}")
    print(f"Split: {best_row['Split']}")
    print(f"Scenario: {best_row['Scenario']}")
    print(f"k: {best_row['k']}")
    print(f"Test Accuracy: {best_row['Test Accuracy']:.4f}")
    print()
    
    # 5. Recomendações
    print("=" * 80)
    print("RECOMENDAÇÕES:\n")
    
    # Melhor k geral
    k_stats = df.groupby('k')['Test Accuracy'].agg(['mean', 'std', 'count'])
    best_k = k_stats['mean'].idxmax()
    print(f"• Melhor valor de k (média): k={best_k} (média: {k_stats.loc[best_k, 'mean']:.4f})")
    
    # Melhor combinação data_type + split
    combo_stats = df.groupby(['Data Type', 'Split'])['Test Accuracy'].mean().sort_values(ascending=False)
    best_combo = combo_stats.index[0]
    print(f"• Melhor combinação Data-Split: {best_combo[0]} + {best_combo[1]} (média: {combo_stats.iloc[0]:.4f})")
    
    # Melhor scenario
    scenario_stats = df.groupby('Scenario')['Test Accuracy'].mean().sort_values(ascending=False)
    best_scenario = scenario_stats.index[0]
    print(f"• Melhor scenario: {best_scenario} (média: {scenario_stats.iloc[0]:.4f})")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    compare_k_values()
