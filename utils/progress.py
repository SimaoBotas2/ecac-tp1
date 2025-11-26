# ======================== PROGRESS BAR ================================
# Utilitário para mostrar progresso de execução
# Funções neste ficheiro foram geradas por LLMs.

import sys
import time


def progress_bar(current, total, label="", width=30):
    """
    Imprime uma barra de progresso simples.
    
    Parâmetros:
    -----------
    current : int
        Iteração atual (0-based ou 1-based, depende de quem chama).
    total : int
        Total de iterações.
    label : str
        Rótulo opcional.
    width : int
        Largura da barra em caracteres.
    """
    if total <= 0:
        return
    
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    
    pct_str = f"{percent*100:.1f}%"
    prog_str = f"\r{label} [{bar}] {pct_str} ({current}/{total})"
    
    sys.stdout.write(prog_str)
    sys.stdout.flush()
    
    if current >= total - 1:
        sys.stdout.write("\n")
        sys.stdout.flush()


def progress_with_time(current, total, start_time, label="", width=30):
    """
    Imprime barra de progresso com tempo decorrido e ETA.
    
    Parâmetros:
    -----------
    current : int
        Iteração atual.
    total : int
        Total de iterações.
    start_time : float
        Tempo de início (use time.time()).
    label : str
        Rótulo.
    width : int
        Largura da barra.
    """
    if total <= 0:
        return
    
    elapsed = time.time() - start_time
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    
    # Estimar ETA
    if current > 0:
        rate = elapsed / current
        eta = (total - current) * rate
        eta_str = f"{int(eta)}s"
    else:
        eta_str = "?"
    
    pct_str = f"{percent*100:.1f}%"
    time_str = f"{int(elapsed)}s"
    
    prog_str = f"\r{label} [{bar}] {pct_str} ({current}/{total}) | {time_str} | ETA: {eta_str}"
    
    sys.stdout.write(prog_str)
    sys.stdout.flush()
    
    if current >= total - 1:
        sys.stdout.write("\n")
        sys.stdout.flush()


def print_section(title):
    """Imprime um título de secção formatado."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
