import numpy as np
import matplotlib.pyplot as plt
def zscore(data, k):
    """
    Identifica outliers num array com base no Z-Score.

    Parâmetros
    ----------
    data : array-like
        Amostras numéricas (1D).
    k : float
        Limite de Z-Score (ex: 3, 3.5, 4).

    Retorna
    -------
    is_outlier : np.ndarray (bool)
        Máscara indicando quais amostras são outliers.
    z : np.ndarray (float)
        Valores de Z-Score correspondentes.
    """
    data = np.asarray(data, dtype=float)
    mean = np.mean(data)
    std = np.std(data)

    if std == 0:
        z = np.zeros_like(data)
    else:
        z = (data - mean) / std

    is_outlier = np.abs(z) > k
    return is_outlier, z


def plot_outliers_zScore(sensor_info, activities, k_values):
    """
    Plota dados de sensores, destacando outliers via Z-Score para vários limiares k.

    Parâmetros
    ----------
    sensor_info : dict
        {nome_sensor: np.ndarray} com dados (1D ou Nx3).
    activities : array-like
        Rótulos das atividades.
    k_values : list[float]
        Valores de k usados no cálculo do Z-Score.
    """
    activities = np.asarray(activities)
    unique_activities = np.unique(activities)

    for label, sensor_data in sensor_info.items():
        print(f"\nSensor: {label}")
        sensor_data = np.asarray(sensor_data, dtype=float)

        # Se o sensor for 3D, calcula o módulo
        if sensor_data.ndim == 2 and sensor_data.shape[1] == 3:
            sensor_data = np.linalg.norm(sensor_data, axis=1)

        n_k = len(k_values)
        fig, axes = plt.subplots(1, n_k, figsize=(5 * n_k, 4), sharey=True)
        axes = np.atleast_1d(axes)

        for i, k in enumerate(k_values):
            ax = axes[i]
            x_positions, y_values, colors = [], [], []

            print(f"\nOutliers para K = {k}:")
            for j, activity in enumerate(unique_activities):
                mask = (activities == activity)
                data_act = sensor_data[mask]

                is_outlier, _ = zscore(data_act, k)

                n_outliers = np.sum(is_outlier)
                density = (n_outliers / len(data_act)) * 100
                print(f"{n_outliers} outliers na atividade {int(activity)} ({density:.2f}%)")

                x_positions.append(np.full_like(data_act, j))
                y_values.append(data_act)
                colors.append(np.where(is_outlier, 'red', 'blue'))

            ax.scatter(
                np.concatenate(x_positions),
                np.concatenate(y_values),
                c=np.concatenate(colors),
                alpha=0.6, s=15
            )

            ax.set_xticks(range(len(unique_activities)))
            ax.set_xticklabels([f"A{int(a)}" for a in unique_activities])
            ax.set_xlabel("Atividades")
            ax.set_title(f"{label} (k={k})")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("Módulo")
        plt.tight_layout()
        plt.show()
