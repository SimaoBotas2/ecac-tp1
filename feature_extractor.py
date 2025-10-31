import numpy as np
import scipy.stats as stats
import scipy.fft as fft
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import DEBUG

#Trabalho Realizado por:
  #Martim Alves Rodrigues da Costa Duarte nº 2021275991
  #Simão Tomás Botas Carvalho nº 2021223055


def sampling_rate_calculator(dados):
    timestamps = dados[:, 10].astype(float)
    timestamps_sec = timestamps / 1000.0

    time_diffs = np.diff(timestamps_sec)

    # Filtrar diferenças razoáveis
    time_diffs_clean = time_diffs[(time_diffs > 0.005) & (time_diffs < 0.5)]

    if len(time_diffs_clean) > 0:
        sr_est = 1.0 / np.median(time_diffs_clean)
        # Arredondar para valor típico
        rates = [25, 50, 100, 200]
        detected = int(min(rates, key=lambda x: abs(x - sr_est)))
        if DEBUG: print(f"Taxa de amostragem detectada: {detected} Hz (estimada {sr_est:.2f} Hz)")
        return detected
    else:
        detected = 50
        if DEBUG: print(f"Taxa de amostragem detectada: {detected} Hz (valor padrão)")
        return detected

class FeatureExtractor:

    @staticmethod
    def temporal_features(signal):
        """Extrai as 13 features temporais identificadas no artigo"""

        features = {}

        # Features estatísticas básicas
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['var'] = np.var(signal)
        features['rms'] = np.sqrt(np.mean(signal**2))
        features['max'] = np.max(signal)
        features['min'] = np.min(signal)
        features['range'] = features['max'] - features['min']
        
        # 2. Estatísticas de forma (usando scipy.stats)
        features['skewness'] = stats.skew(signal)
        features['kurtosis'] = stats.kurtosis(signal)
        
        # 3. Outras features
        features['mad'] = np.mean(np.abs(signal - features['mean']))  # Mean Absolute Deviation
        features['energy'] = np.sum(signal**2)
        
        # 4. Entropia (aproximada)
        hist, _ = np.histogram(signal, bins=10, density=True)
        hist = hist[hist > 0]
        features['entropy'] = -np.sum(hist * np.log(hist))
        
        # 5. Área sob a curva (integral numérica)
        features['auc'] = np.trapz(np.abs(signal))
        
        return features
    
    @staticmethod
    def spectral_features(signal, sampling_rate=50):
        """Extrai as 7 features espetrais de um sinal 1D"""
        features = {}

        # Transformada de Fourier (scipy.fft)
        fft_vals = np.abs(fft.fft(signal))  # type: ignore
        freqs = fft.fftfreq(len(signal), 1/sampling_rate)
        
        # Apenas frequências positivas
        positive_idx = freqs > 0
        fft_vals = fft_vals[positive_idx]
        freqs = freqs[positive_idx]
        
        if len(fft_vals) == 0:
            return features
        
        # 1. Centroide espectral
        features['spectral_centroid'] = np.sum(freqs * fft_vals) / np.sum(fft_vals)
        
        # 2. Largura de banda espectral
        centroid = features['spectral_centroid']
        features['spectral_bandwidth'] = np.sqrt(
            np.sum((freqs - centroid)**2 * fft_vals) / np.sum(fft_vals)
        )
        
        # 3. Energia em bandas de frequência (0-5Hz, 5-10Hz, 10-15Hz, 15-20Hz)
        bands = [(0, 5), (5, 10), (10, 15), (15, 20)]
        for i, (low, high) in enumerate(bands):
            band_mask = (freqs >= low) & (freqs < high)
            features[f'energy_band_{i}'] = np.sum(fft_vals[band_mask])
        
        # 4. Planicidade espectral (scipy.stats)
        fft_pos = fft_vals[fft_vals > 0]
        if len(fft_pos) > 0:
            features['spectral_flatness'] = stats.gmean(fft_pos) / np.mean(fft_pos)
        else:
            features['spectral_flatness'] = 0
        
        # 5. Entropia espectral
        spectral_pdf = fft_vals / np.sum(fft_vals)
        spectral_pdf = spectral_pdf[spectral_pdf > 0]
        features['spectral_entropy'] = -np.sum(spectral_pdf * np.log(spectral_pdf))
        
        return features
    
    @staticmethod
    def extract_window_features(accel_window, gyro_window, mag_window, sampling_rate=50):
        """
        Extrai TODAS as features para uma janela temporal
        """
        all_features = {}
        
        # Para CADA sensor e CADA eixo
        sensors = {
            'accel': accel_window,
            'gyro': gyro_window, 
            'mag': mag_window
        }
        
        for sensor_name, sensor_data in sensors.items():
            # Features para cada eixo (X, Y, Z)
            for axis in range(3):
                signal = sensor_data[:, axis]
                
                # Features temporais (13 features por eixo)
                temp_feat = FeatureExtractor.temporal_features(signal)
                for key, val in temp_feat.items():
                    all_features[f'{sensor_name}_axis{axis}_{key}'] = val
                
                # Features espectrais (7 features por eixo)
                spec_feat = FeatureExtractor.spectral_features(signal, sampling_rate)
                for key, val in spec_feat.items():
                    all_features[f'{sensor_name}_axis{axis}_{key}'] = val
            
            # Features do MÓDULO também (magnitude vectorial)
            magnitude = np.linalg.norm(sensor_data, axis=1)
            temp_mag = FeatureExtractor.temporal_features(magnitude)
            spec_mag = FeatureExtractor.spectral_features(magnitude, sampling_rate)
            
            for key, val in temp_mag.items():
                all_features[f'{sensor_name}_mag_{key}'] = val
            for key, val in spec_mag.items():
                all_features[f'{sensor_name}_mag_{key}'] = val
        
        return all_features
    
def extract_features_4_2(accel_data, gyro_data, mag_data, activities, sampling_rate=50):
    """
    Implementa o ponto 4.2:
    - Janelas de 5 segundos com 50% overlap
    - Verifica atividades mistas
    - Extrai features para cada janela válida
    """
    
    print("=== 4.2 - EXTRAÇÃO DE FEATURES TEMPORAIS E ESPETRAIS ===")
    
    # 1. Parâmetros de janelamento
    window_size = 5 * sampling_rate  # 5 segundos
    overlap = 0.5
    step_size = int(window_size * (1 - overlap))
    
    print(f"Window size: {window_size} amostras ({window_size/sampling_rate} segundos)")
    print(f"Step size: {step_size} amostras")
    print(f"Sampling rate: {sampling_rate} Hz")
    
    # 2. Listas para guardar resultados
    features_list = []
    labels_list = []
    window_info = []  # Para debug
    
    # 3. Processar cada janela
    total_windows = 0
    valid_windows = 0
    
    for start_idx in range(0, len(accel_data) - window_size + 1, step_size):
        end_idx = start_idx + window_size
        total_windows += 1
        
        # Extrair janelas
        accel_window = accel_data[start_idx:end_idx]
        gyro_window = gyro_data[start_idx:end_idx]
        mag_window = mag_data[start_idx:end_idx]
        activity_window = activities[start_idx:end_idx]
        
        # Verificar se a janela contém APENAS UMA atividade
        unique_activities = np.unique(activity_window)
        
        if len(unique_activities) == 1:
            # JANELA VÁLIDA - extrair features
            try:
                features = FeatureExtractor.extract_window_features(
                    accel_window, gyro_window, mag_window, sampling_rate
                )
                
                # Converter dicionário para array
                feature_vector = list(features.values())
                features_list.append(feature_vector)
                labels_list.append(unique_activities[0])
                
                valid_windows += 1
                window_info.append({
                    'start': start_idx,
                    'end': end_idx,
                    'activity': unique_activities[0],
                    'n_features': len(feature_vector)
                })
                
            except Exception as e:
                print(f"Erro na janela {start_idx}-{end_idx}: {e}")
                continue
        
        # Progresso a cada 1000 janelas
        if (total_windows % 1000 == 0) and DEBUG:
            print(f"Processadas {total_windows} janelas...")
    
    # 4. Converter para arrays numpy
    X = np.array(features_list)  # Features matrix
    y = np.array(labels_list)    # Labels vector
    
    # 5. Estatísticas finais
    print(f"\n=== RESULTADOS ===")
    print(f"Total de janelas processadas: {total_windows}")
    print(f"Janelas válidas (uma atividade): {valid_windows}")
    print(f"Taxa de sucesso: {valid_windows/total_windows*100:.1f}%")
    print(f"Shape final - X: {X.shape}, y: {y.shape}")
    
    # Distribuição das atividades nas janelas válidas
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nDistribuição por atividade:")
    for activity, count in zip(unique, counts):
        print(f"  Atividade {activity}: {count} janelas")
    
    return X, y, window_info

#4.3
def pca_analysis(X_features, target_variance=0.75):
    """Faz análise PCA completa""" # Principal Component Analysis
    
    # Normalizar
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_features) # Normalização das features antes do PCA é crucial para garantir que a variância de cada feature seja considerada igualmente no cálculo das componentes principais.
    
    # PCA completo para análise
    pca_full = PCA()
    pca_full.fit(X_normalized) # Encontra a melhor combinação linear das features que captura a maior parte da variância nos dados.
    
    # Encontrar componentes para variância alvo
    variancia_acumulada = np.cumsum(pca_full.explained_variance_ratio_) #soma a variância explicada por cada componente principal de forma acumulativa.
    n_components = np.argmax(variancia_acumulada >= target_variance) + 1 # verifica o primeiro índice onde a variância acumulada atinge ou excede o valor alvo (75% neste caso).
    
    # PCA final com componentes certos
    pca_final = PCA(n_components=n_components)
    X_pca = pca_final.fit_transform(X_normalized) # Reduz os dados normalizados para o número ótimo de componentes principais.
    
    # Resultados
    print(f"PCA: {X_features.shape[1]} → {n_components} componentes")
    print(f"Variância explicada: {variancia_acumulada[n_components-1]*100:.2f}%")
    
    return X_pca, pca_final, scaler