# ======================== COMPARE kNN RESULTS ================================
# Comparar diferentes valores de k do kNN
# Agora lê resultados de JSON em data/results (ou fallback em data/processed/results/evaluation)

import sys
from pathlib import Path
import json
import pandas as pd

# Setup path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Preferir data/results; se não existir, usar data/processed/results/evaluation
JSON_PRIMARY = ROOT / "data" / "processed" / "results" / "evaluation" / "evaluation_results.json"


def _infer_fields_from_key(key: str):
    """Inferir Data Type, Split e Scenario a partir da chave do JSON.
    Ex.: 'features_within_all', 'embeddings_between_relief', 'embeddings_within_pca'.
    """
    k = key.lower()
    data_type = 'FEATURES' if 'feature' in k else ('EMBEDDINGS' if 'embedding' in k else 'UNKNOWN')
    split = 'WITHIN' if 'within' in k else ('BETWEEN' if 'between' in k else 'UNKNOWN')
    scenario = 'ALL'
    if 'pca' in k:
        scenario = 'PCA'
    elif 'relief' in k:
        scenario = 'RELIEF'
    elif 'all' in k:
        scenario = 'ALL'
    return data_type, split, scenario

def load_results_json():
    """Carrega resultados do JSON e devolve um DataFrame normalizado.
    Suporta o formato atual: dict com chaves por cenário, contendo 'k_accuracies', 'best_k', etc.
    Também suporta formatos anteriores: lista de dicts ou dict com chave 'results'.
    """
    json_path = JSON_PRIMARY if JSON_PRIMARY.exists() else None
    if not json_path or not json_path.exists():
        print(f"Não encontrei o ficheiro JSON de resultados em: {JSON_PRIMARY}")
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Erro ao ler JSON: {json_path} -> {e}")
        return None

    rows = []

    # Formato atual: dict de cenários (como 'features_within_all': {...})
    if isinstance(data, dict) and 'results' not in data:
        for key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            data_type, split, scenario = _infer_fields_from_key(key)
            k_map = entry.get('k_accuracies') or {}
            if k_map and isinstance(k_map, dict):
                for k_str, acc in k_map.items():
                    try:
                        k_val = int(k_str)
                        acc_val = float(acc)
                    except (TypeError, ValueError):
                        continue
                    rows.append({
                        'Data Type': data_type,
                        'Split': split,
                        'Scenario': scenario,
                        'k': k_val,
                        'Val Accuracy': acc_val,
                    })
            else:
                bk = entry.get('best_k')
                vacc = entry.get('val_acc_at_best_k')
                try:
                    k_val = int(bk)
                    acc_val = float(vacc)
                except (TypeError, ValueError):
                    continue
                rows.append({
                    'Data Type': data_type,
                    'Split': split,
                    'Scenario': scenario,
                    'k': k_val,
                    'Val Accuracy': acc_val,
                })
    else:
        # Formatos anteriores
        if isinstance(data, dict) and 'results' in data:
            records = data['results']
        elif isinstance(data, list):
            records = data
        else:
            records = [data]
        for r in records:
            data_type = r.get('data_type') or r.get('dataType') or r.get('type')
            split_type = r.get('split_type') or r.get('splitType') or r.get('split')
            scenario = r.get('scenario') or r.get('scenario_name') or r.get('name')
            k = r.get('k') or r.get('best_k')
            val_acc = r.get('val_accuracy') or r.get('valAccuracy') or r.get('validation_accuracy')
            if data_type is None or split_type is None or scenario is None or k is None or val_acc is None:
                continue
            try:
                k = int(k)
                val_acc = float(val_acc)
            except (TypeError, ValueError):
                continue
            rows.append({
                'Data Type': str(data_type).upper(),
                'Split': str(split_type).upper(),
                'Scenario': str(scenario).upper(),
                'k': k,
                'Val Accuracy': val_acc,
            })

    if not rows:
        print("JSON não contém entradas válidas para análise.")
        return None

    df = pd.DataFrame(rows)
    df['Data Type'] = df['Data Type'].astype(str).str.upper()
    df['Split'] = df['Split'].astype(str).str.upper()
    df['Scenario'] = df['Scenario'].astype(str).str.upper()
    return df


def compare_k_values():
    """Compara resultados para diferentes valores de k e melhor split usando JSON."""
    df = load_results_json()
    if df is None or df.empty:
        return

    print("=" * 80)
    print("COMPARAÇÃO DE RESULTADOS (a partir de JSON) - DIFERENTES VALORES DE k")
    print("NOTA: Usando VALIDATION Accuracy para seleção de k (sem data leakage)")
    print("=" * 80 + "\n")

    # Ranking geral
    df_sorted = df.sort_values('Val Accuracy', ascending=False)
    print("RANKING GERAL (Ordenado por Val Accuracy):\n")
    print(df_sorted.to_string(index=False))
    print()

    # Melhor k (média de Val Accuracy)
    k_stats = df.groupby('k')['Val Accuracy'].agg(['mean', 'std', 'count'])
    best_k = k_stats['mean'].idxmax()
    print("=" * 80)
    print("SELEÇÃO DE HIPERPARÂMETRO k (com base em Val Accuracy média):\n")
    print(f"• Melhor k: {best_k} (média: {k_stats.loc[best_k, 'mean']:.4f}, n={int(k_stats.loc[best_k, 'count'])})")
    print()

    # Melhor modelo de splitting (média de Val Accuracy por Split)
    split_stats = df.groupby('Split')['Val Accuracy'].mean().sort_values(ascending=False)
    best_split = split_stats.index[0]
    print("=" * 80)
    print("MELHOR MODELO DE SPLITTING (Val Accuracy média):\n")
    print(f"• Split: {best_split} (média: {split_stats.iloc[0]:.4f})")
    print()

    # Melhor combinação (Data Type + Split)
    combo_stats = df.groupby(['Data Type', 'Split'])['Val Accuracy'].mean().sort_values(ascending=False)
    best_combo = combo_stats.index[0]
    print("=" * 80)
    print("MELHOR COMBINAÇÃO DATA+SPLIT (Val Accuracy média):\n")
    print(f"• {best_combo[0]} + {best_combo[1]} (média: {combo_stats.iloc[0]:.4f})")
    print()

    # Melhor cenário individual
    best_row = df_sorted.iloc[0]
    print("=" * 80)
    print("MELHOR CENÁRIO INDIVIDUAL (maior Val Accuracy):\n")
    print(f"• Data Type: {best_row['Data Type']} | Split: {best_row['Split']} | Scenario: {best_row['Scenario']} | k: {best_row['k']} | Val Acc: {best_row['Val Accuracy']:.4f}")
    print()



def analyze_class_balance():
    """Analisa o equilíbrio por atividade (classe) usando recalls por classe.
    Calcula, para cada cenário, a média e o desvio padrão dos recalls por classe.
    Um bom modelo tem média alta e desvio baixo (menos desbalanceado).
    """
    json_path = JSON_PRIMARY
    if not json_path.exists():
        print(f"Não encontrei o ficheiro JSON de resultados em: {JSON_PRIMARY}")
        return
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Erro ao ler JSON: {json_path} -> {e}")
        return

    rows = []
    if isinstance(data, dict):
        for key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            data_type, split, scenario = _infer_fields_from_key(key)
            per_class = entry.get('per_class_metrics')
            if not isinstance(per_class, dict):
                continue
            # usar recall como taxa de acerto por classe
            recalls = []
            for _, metrics in per_class.items():
                try:
                    r = float(metrics.get('recall'))
                except (TypeError, ValueError):
                    continue
                recalls.append(r)
            if not recalls:
                continue
            mean_recall = sum(recalls) / len(recalls)
            # desvio padrão simples
            var = sum((r - mean_recall) ** 2 for r in recalls) / len(recalls)
            std_recall = var ** 0.5
            rows.append({
                'Data Type': data_type,
                'Split': split,
                'Scenario': scenario,
                'Mean Recall': mean_recall,
                'Std Recall': std_recall,
            })

    if not rows:
        print("Sem métricas por classe para analisar.")
        return

    df = pd.DataFrame(rows)
    df['Data Type'] = df['Data Type'].astype(str).str.upper()
    df['Split'] = df['Split'].astype(str).str.upper()
    df['Scenario'] = df['Scenario'].astype(str).str.upper()

    print("=" * 80)
    print("ANÁLISE DE BALANCEAMENTO POR ATIVIDADE (recall por classe)\n")
    # ordenar por mean alto e std baixo
    df_sorted = df.sort_values(by=['Mean Recall', 'Std Recall'], ascending=[False, True])
    print(df_sorted.to_string(index=False, formatters={'Mean Recall': lambda x: f"{x:.4f}", 'Std Recall': lambda x: f"{x:.4f}"}))
    print()

    best = df_sorted.iloc[0]
    print("Melhor equilíbrio:")
    print(f"• {best['Data Type']} | {best['Split']} | {best['Scenario']} -> Mean Recall={best['Mean Recall']:.4f}, Std={best['Std Recall']:.4f}")



if __name__ == "__main__":
    compare_k_values()
    analyze_class_balance()