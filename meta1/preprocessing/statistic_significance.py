import numpy as np
from scipy import stats

#Trabalho Realizado por:
  #Martim Alves Rodrigues da Costa Duarte nº 2021275991
  #Simão Tomás Botas Carvalho nº 2021223055

def _format_p(p_value: float) -> str:
    """Formata p-values em notação científica; para underflow (0.0) mostra limite inferior."""
    try:
        p = float(p_value)
    except Exception:
        return str(p_value)
    if p <= 0.0:
        # Mostrar limite inferior representável (valor subnormal mínimo não é prático; usar tiny)
        return f"< {np.finfo(float).tiny:.0e}"
    return f"{p:.16e}"


def analyze_statistical_significance(modules, activities):
    """
    Analisa a significância estatística entre atividades usando sensores.

    Parâmetros
    ----------
    modules : np.ndarray
        Dados de sensores (amostras x 3 sensores: Aceleração, Giroscópio, Magnetómetro).
    activities : np.ndarray
        Rótulos das atividades correspondentes a cada amostra.

    Retorna
    -------
    None
        Imprime resultados dos testes de normalidade (Kolmogorov-Smirnov), ANOVA e Kruskal-Wallis
        para cada sensor, indicando se há diferença estatisticamente significativa entre atividades.
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
            stat, p_value = stats.kstest(
            activity_data[i], 
            'norm', 
            args=(np.mean(activity_data[i]), np.std(activity_data[i]))
            )
            normal = "Normal" if p_value > 0.05 else "Não-normal"
            print(f"  Atividade {activity}: p={_format_p(p_value)} ({normal})")
        
        # Teste ANOVA ou Kruskal-Wallis dependendo da normalidade
        # (Vamos simplificar e fazer ambos para comparar)
        print("\nComparação entre TODAS as atividades:")

        normal_all = all(stats.kstest(
            data, 
            'norm', 
            args=(np.mean(data), np.std(data))
            ).pvalue > 0.05 for data in activity_data)

        p_used = None
        if normal_all:
            # ANOVA (para dados normais)
            f_stat, p_anova = stats.f_oneway(*activity_data)
            print(f"  ANOVA: F={f_stat:.4f}, p={_format_p(p_anova)}")
            p_used = p_anova
        else:
            # Kruskal-Wallis (para dados não-normais)
            h_stat, p_kruskal = stats.kruskal(*activity_data)
            print(f"  Kruskal-Wallis: H={h_stat:.4f}, p={_format_p(p_kruskal)}")
            p_used = p_kruskal
        
        # Interpretação
        significant = "SIGNIFICATIVO" if (p_used is not None and p_used < 0.05) else "NÃO SIGNIFICATIVO"
        print(f"  Resultado: {significant}")