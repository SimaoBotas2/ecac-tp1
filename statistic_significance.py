import numpy as np
from scipy import stats

def analyze_statistical_significance(modules, activities):
    """
    4.1 - Analyzes statistical significance between activities
    """
    
    unique_activities = np.unique(activities)
    sensor_names = ['Aceleração', 'Giroscópio', 'Magnetómetro']
    
    print("=== 4.1 - SIGNIFICÂNCIA ESTATÍSTICA ===")
    
    for sensor_idx, sensor_name in enumerate(sensor_names):
        print(f"\n--- {sensor_name} ---")
        
        # Preparar dados por atividade
        activity_data = []
        for activity in unique_activities:
            mask = activities == activity
            activity_data.append(modules[mask, sensor_idx])
        
        # Teste de normalidade (Kolmogorov-Smirnov)
        print("Teste de normalidade (K-S):")
        for i, activity in enumerate(unique_activities):
            stat, p_value = stats.kstest(activity_data[i], 'norm')
            normal = "Normal" if p_value > 0.05 else "Não-normal"
            print(f"  Atividade {activity}: p={p_value:.4f} ({normal})")
        
        # Teste ANOVA ou Kruskal-Wallis dependendo da normalidade
        # (Vamos simplificar e fazer ambos para comparar)
        print("\nComparação entre TODAS as atividades:")
        
        # ANOVA (para dados normais)
        f_stat, p_anova = stats.f_oneway(*activity_data)
        print(f"  ANOVA: F={f_stat:.4f}, p={p_anova:.4f}")
        
        # Kruskal-Wallis (para dados não-normais)
        h_stat, p_kruskal = stats.kruskal(*activity_data)
        print(f"  Kruskal-Wallis: H={h_stat:.4f}, p={p_kruskal:.4f}")
        
        # Interpretação
        significant = "SIGNIFICATIVO" if p_anova < 0.05 else "NÃO SIGNIFICATIVO"
        print(f"  Resultado: {significant}")