# ======================== EVALUATION - SIMPLE ================================
# ECAC 2025 – META 2 – Task 5
# Treina todos os modelos em train + validation
# Seleciona melhor k por validation accuracy
# Avalia no test set

import numpy as np
from pathlib import Path
import sys
import json
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.knn import KNNClassifier, confusion_matrix
from utils.progress import print_section, progress_with_time



class SimpleEvaluation:
    """
    Avalia todos os cenários em 3 fases:
    1. Treina kNN com vários k em train, valida em val → guarda melhor k
    2. Análise do melhor k por scenario
    3. Retrain com train + val combinados usando melhor k, avalia em test
    """
    
    def __init__(self, data_processed_path):
        self.data_processed = Path(data_processed_path)
        self.k_values = [3, 5, 7]
        self.results = {}
        self.best_ks = {}  # Guarda melhor k para cada scenario
        
    def load_scenario(self, data_type, split_type, scenario):
        """Load scenario data from .npz file."""
        scenario_path = self.data_processed / "scenarios" / data_type / split_type / f"{scenario}.npz"
        
        if not scenario_path.exists():
            raise FileNotFoundError(f"Ficheiro não encontrado: {scenario_path}")
        
        data = np.load(scenario_path)
        return {
            'X_train': data['train_X'],
            'y_train': data['train_y'],
            'X_val': data['val_X'],
            'y_val': data['val_y'],
            'X_test': data['test_X'],
            'y_test': data['test_y'],
        }
    
    def evaluate_scenario(self, data_type, split_type, scenario):
        """
        FASE 1: Testa vários k em train, valida em val
        Guarda os resultados de validação para cada k
        """
        print(f"\n  [{data_type:12} | {split_type:8} | {scenario:7}]")
        
        # Load data
        scenario_data = self.load_scenario(data_type, split_type, scenario)
        X_train = scenario_data['X_train']
        y_train = scenario_data['y_train']
        X_val = scenario_data['X_val']
        y_val = scenario_data['y_val']
        
        print(f"    Carregado: X_train={X_train.shape}, X_val={X_val.shape}")
        print(f"    Testando k={self.k_values}...")
        
        # Teste vários k em train e valida em val
        best_k = None
        best_val_acc = -1
        k_accuracies = {}
        
        start_time = time.time()
        for idx, k in enumerate(self.k_values):
            print(f"\n      [k={k}] Treinando...", flush=True)
            iter_start = time.time()
            
            knn = KNNClassifier(k=k)
            knn.fit(X_train, y_train)
            fit_time = time.time() - iter_start
            print(f"      [k={k}] Fit OK ({fit_time:.2f}s) - Validando...")
            
            val_acc = knn.score(X_val, y_val)
            val_time = time.time() - iter_start - fit_time
            k_accuracies[k] = val_acc
            print(f"      [k={k}] Val Acc={val_acc:.4f} ({val_time:.2f}s)", flush=True)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_k = k
            
            # Progress bar para seleção de k
            progress_with_time(idx + 1, len(self.k_values), start_time, label="K-selection")
        
        # Store para Phase 2 e 3
        key = f"{data_type}_{split_type}_{scenario}"
        self.best_ks[key] = {
            'best_k': best_k,
            'val_acc': best_val_acc,
            'k_accuracies': k_accuracies,
        }
        
        print(f"    MELHOR K: {best_k:2d} com Val Acc={best_val_acc:.4f}")
        
        return best_k, best_val_acc, k_accuracies
    
    def analyze_best_ks(self):
        """
        FASE 2: Analisa o melhor k para cada scenario
        """
        print_section("5.1 - Analise do Melhor k")
        print()
        print(f"{'Data Type':<15} {'Split':<10} {'Scenario':<10} {'Best k':<8} {'Val Acc':<10}")
        print("-" * 65)
        
        for key in sorted(self.best_ks.keys()):
            parts = key.split('_')
            data_type, split_type, scenario = parts[0], parts[1], parts[2]
            result = self.best_ks[key]
            
            print(f"{data_type:<15} {split_type:<10} {scenario:<10} {result['best_k']:<8} "
                  f"{result['val_acc']:<10.4f}")
        
        print()
    
    def retrain_and_test(self):
        """
        FASE 3: Retreina cada modelo com train+val combinados e testa em test
        """
        print_section("5.2 - Fase 3: Retrain com Train+Val e Teste")
        print()
        
        data_types = ["features", "embeddings"]
        split_types = ["within", "between"]
        scenarios = ["all", "pca", "relief"]
        
        for split_type in split_types:
            print(f"\n=== {split_type.upper()} SUBJECT ===\n")
            
            for data_type in data_types:
                print(f"  Data Type: {data_type.upper()}")
                
                for scenario in scenarios:
                    key = f"{data_type}_{split_type}_{scenario}"
                    best_k = self.best_ks[key]['best_k']
                    
                    try:
                        self._retrain_and_test_single(data_type, split_type, scenario, best_k)
                    except (OSError, ValueError, RuntimeError) as e:
                        print(f"    [ERRO] {e}")
    
    def _retrain_and_test_single(self, data_type, split_type, scenario, best_k):
        """
        Retreina um modelo específico com o melhor k e testa em test
        """
        print(f"\n  [{data_type:12} | {split_type:8} | {scenario:7}] k={best_k}")
        
        # Load data
        scenario_data = self.load_scenario(data_type, split_type, scenario)
        X_train = scenario_data['X_train']
        y_train = scenario_data['y_train']
        X_val = scenario_data['X_val']
        y_val = scenario_data['y_val']
        X_test = scenario_data['X_test']
        y_test = scenario_data['y_test']
        
        # Combinar train + val (80% dos dados)
        X_trainval = np.vstack([X_train, X_val])
        y_trainval = np.hstack([y_train, y_val])
        
        print(f"    Combinado: X_trainval={X_trainval.shape}, X_test={X_test.shape}")
        print(f"    Treinando modelo com k={best_k}...", end="", flush=True)
        
        # Treinar modelo final com melhor k
        knn_final = KNNClassifier(k=best_k)
        knn_final.fit(X_trainval, y_trainval)
        
        print(" OK")
        
        print(f"    Testando em {len(X_test)} amostras...", end="", flush=True)
        
        # Avaliar em test
        y_pred_test = knn_final.predict(X_test)
        test_acc = (y_pred_test == y_test).mean()
        
        print(f" OK - Accuracy={test_acc:.4f}")
        
        # Confusion matrix
        cm_test, classes = confusion_matrix(y_test, y_pred_test)
        
        # Compute per-class metrics
        per_class_metrics = self._compute_per_class_metrics(cm_test, classes)
        
        # Store result
        result = {
            'best_k': best_k,
            'val_acc_at_best_k': self.best_ks[f"{data_type}_{split_type}_{scenario}"]['val_acc'],
            'k_accuracies': self.best_ks[f"{data_type}_{split_type}_{scenario}"]['k_accuracies'],
            'test_acc': test_acc,
            'confusion_matrix': cm_test.tolist(),
            'classes': classes.tolist(),
            'per_class_metrics': per_class_metrics,
            'y_test': y_test.tolist(),
            'y_pred': y_pred_test.tolist(),
        }
        
        key = f"{data_type}_{split_type}_{scenario}"
        self.results[key] = result
        
        print(f"    RESULT: Test Acc={test_acc:.4f}")
    
    def _compute_per_class_metrics(self, cm, classes):
        """Compute precision, recall, F1 for each class."""
        metrics = {}
        
        for i, activity_class in enumerate(classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            support = cm[i, :].sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[int(activity_class)] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(support)
            }
        
        return metrics
    
    def run_all(self):
        """Run evaluation pipeline in 3 phases."""
        # FASE 1: Testar vários k em train, validar em val
        print_section("5.0 - Fase 1: Seleção do Melhor k")
        print()
        
        data_types = ["features", "embeddings"]
        split_types = ["within", "between"]
        scenarios = ["all", "pca", "relief"]
        
        for split_type in split_types:
            print(f"\n=== {split_type.upper()} SUBJECT ===\n")
            
            for data_type in data_types:
                print(f"  Data Type: {data_type.upper()}")
                
                for scenario in scenarios:
                    try:
                        self.evaluate_scenario(data_type, split_type, scenario)
                    except (OSError, ValueError, RuntimeError) as e:
                        print(f"    [ERRO] {e}")
        
        # FASE 2: Análise do melhor k
        self.analyze_best_ks()
        
        # FASE 3: Retrain com train+val e teste
        self.retrain_and_test()
        
        return self.results

    def run_best_only(self, data_type: str, split_type: str, scenario: str):
        """Avalia apenas um cenário indicado, escolhe o melhor k por validação,
        retreina com train+val e testa no conjunto de teste.

        Retorna os resultados desse cenário em self.results.
        """
        # Fase 1: escolher melhor k apenas para este cenário
        try:
            best_k, best_val_acc, k_accuracies = self.evaluate_scenario(data_type, split_type, scenario)
        except (OSError, ValueError, RuntimeError) as e:
            raise RuntimeError(f"Falha ao avaliar cenário {data_type}_{split_type}_{scenario}: {e}")

        # Guardar entrada best_ks (para consistência com retrain)
        key = f"{data_type}_{split_type}_{scenario}"
        self.best_ks[key] = {
            'best_k': best_k,
            'val_acc': best_val_acc,
            'k_accuracies': k_accuracies,
        }

        # Fase 3: retrain + test apenas para este cenário
        self._retrain_and_test_single(data_type, split_type, scenario, best_k)
        return self.results

    def run_with_params(self, data_type: str, split_type: str, scenario: str, k: int):
        """Retreina e testa apenas um cenário com parâmetros fornecidos pelo utilizador,
        SEM validação prévia. Usa k indicado para treinar em train+val e avaliar em test.
        """
        # Preparar entrada best_ks mínima para consistência do formato de saída
        key = f"{data_type}_{split_type}_{scenario}"
        self.best_ks[key] = {
            'best_k': int(k),
            'val_acc': None,
            'k_accuracies': {},
        }
        # Executar retrain + test diretamente
        self._retrain_and_test_single(data_type, split_type, scenario, int(k))
        return self.results
    
    def print_summary(self):
        """Print summary of final results."""
        print_section("5.3 - Resumo Final dos Resultados")
        print()
        
        # Summary table
        print(f"{'Data Type':<15} {'Split':<10} {'Scenario':<10} {'Best k':<8} {'Val Acc':<10} {'Test Acc':<10}")
        print("-" * 80)
        
        for key, result in sorted(self.results.items()):
            parts = key.split('_')
            data_type, split_type, scenario = parts[0], parts[1], parts[2]
            val_acc = result.get('val_acc_at_best_k')
            val_acc_str = f"{val_acc:.4f}" if isinstance(val_acc, (int, float)) else "-"
            print(f"{data_type:<15} {split_type:<10} {scenario:<10} {result['best_k']:<8} "
                  f"{val_acc_str:<10} {result['test_acc']:<10.4f}")
        
        print()
    
    def print_confusion_matrices(self):
        """Print confusion matrices for each scenario."""
        print_section("5.4 - Confusion Matrices")
        print()
        
        for key, result in sorted(self.results.items()):
            parts = key.split('_')
            data_type, split_type, scenario = parts[0], parts[1], parts[2]
            
            cm = np.array(result['confusion_matrix'])
            classes = np.array(result['classes'])
            
            print(f"\n{data_type.upper()} | {split_type.upper()} | {scenario.upper()}")
            print(f"k={result['best_k']}, Test Accuracy: {result['test_acc']:.4f}\n")
            
            # Print matrix
            print("     " + "  ".join(f"A{c}" for c in classes))
            for i, true_class in enumerate(classes):
                row_str = "  ".join(f"{cm[i, j]:4d}" for j in range(len(classes)))
                print(f"A{true_class}  {row_str}")
            
            # Print per-class metrics
            print(f"\n{'Activity':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
            print("-" * 60)
            
            metrics = result['per_class_metrics']
            for activity, m in sorted(metrics.items()):
                print(f"A{activity:<11} {m['precision']:<12.4f} {m['recall']:<12.4f} "
                      f"{m['f1']:<12.4f} {m['support']:<10}")
            
            print()
    
    def save_results(self, output_dir=None):
        """Save results to JSON file."""
        if output_dir is None:
            output_dir = self.data_processed / "results" / "evaluation"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "evaluation_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n[OK] Resultados guardados em: {output_file}\n")


def run_evaluation(data_processed_path):
    """Run complete evaluation pipeline."""
    evaluator = SimpleEvaluation(data_processed_path)
    evaluator.run_all()
    evaluator.print_summary()
    evaluator.print_confusion_matrices()
    evaluator.save_results()
    return evaluator.results

def run_best_model(data_processed_path, data_type, split_type, scenario):
    """Run pipeline apenas para um cenário específico indicado pelo utilizador."""
    evaluator = SimpleEvaluation(data_processed_path)
    evaluator.run_best_only(data_type, split_type, scenario)
    evaluator.print_summary()
    evaluator.print_confusion_matrices()
    evaluator.save_results()
    return evaluator.results

def run_with_params_cli(data_processed_path, data_type, split_type, scenario, k):
    """CLI helper: retreina + testa um cenário específico com k fornecido."""
    evaluator = SimpleEvaluation(data_processed_path)
    evaluator.run_with_params(data_type, split_type, scenario, int(k))
    evaluator.print_summary()
    evaluator.print_confusion_matrices()
    evaluator.save_results()
    return evaluator.results


if __name__ == "__main__":
    # CLI:
    #  - Sem args: corre pipeline completo
    #  - 3 args: data_type split_type scenario -> escolhe melhor k por validação e retreina
    #  - 4 args: data_type split_type scenario k -> retreina diretamente com k indicado, SEM validação
    DATA_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
    if len(sys.argv) == 5:
        dt, sp, sc, k = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
        run_with_params_cli(DATA_PROCESSED, dt, sp, sc, k)
    elif len(sys.argv) == 4:
        dt, sp, sc = sys.argv[1], sys.argv[2], sys.argv[3]
        run_best_model(DATA_PROCESSED, dt, sp, sc)
    else:
        run_evaluation(DATA_PROCESSED)
