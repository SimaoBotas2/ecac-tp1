
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from meta1.features import feature_extractor as fe


def process_csv_file(csv_path, scenario='all', window_size=256):
    """
    Extrai features de janela aleatória com atividade uniforme.
    
    Parâmetros:
    -----------
    csv_path : str ou Path
        Caminho para o ficheiro CSV
    scenario : str
        Nome do cenário (para carregar scaler)
    window_size : int
        Tamanho da janela (default 256)
    
    Returns:
    --------
    dict com:
        - X_test_scaled: features escaladas (1, 252)
        - y_test: etiqueta da atividade
        - raw_shape, features_shape, csv_name, window_start, window_end, activity, valid_windows_count
    """
    csv_path = Path(csv_path) if isinstance(csv_path, str) else csv_path
    csv_data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    
    total_rows = csv_data.shape[0]
    if total_rows < window_size:
        raise ValueError(f"CSV tem apenas {total_rows} linhas, precisa de {window_size}")
    
    all_labels = csv_data[:, 11].astype(int)
    valid_windows = []
    
    for start_idx in range(total_rows - window_size + 1):
        end_idx = start_idx + window_size
        if np.all(all_labels[start_idx:end_idx] == all_labels[start_idx]):
            valid_windows.append(start_idx)
    
    if not valid_windows:
        raise ValueError(f"Nenhuma janela uniforme em {csv_path.name}")
    
    start_idx = np.random.choice(valid_windows)
    end_idx = start_idx + window_size
    raw_window = csv_data[start_idx:end_idx, 1:10].astype(np.float32)
    feature_window = raw_window[:250]
    
    accel_data = feature_window[:, 0:3]
    gyro_data = feature_window[:, 3:6]
    mag_data = feature_window[:, 6:9]
    
    y_test_csv = csv_data[start_idx:end_idx, 11].astype(int)
    activity_label = y_test_csv[0]
    activities_for_extract = np.full(250, activity_label, dtype=int)
    
    try:
        X_features_list, _, _ = fe.extract_features_4_2(
            accel_data, gyro_data, mag_data, activities_for_extract, 
            sampling_rate=50, participant_ids=None
        )
        if len(X_features_list) == 0:
            raise ValueError("Nenhuma janela válida extraída")
        X_test_csv_features = np.array(X_features_list[0]).reshape(1, -1)
    except Exception as e:
        raise RuntimeError(f"Erro ao extrair features: {e}") from e
    
    scenario_file = ROOT / "data" / "processed" / "scenarios" / 'features' / 'within' / f'{scenario}.npz'
    
    if not scenario_file.exists():
        raise FileNotFoundError(f"Cenário não encontrado: {scenario_file}")
    
    data = np.load(scenario_file)
    meta_mean = data['meta_scaler_mean'].astype(np.float64)
    meta_scale = data['meta_scaler_scale'].astype(np.float64)
    X_test_scaled = (X_test_csv_features - meta_mean) / meta_scale
    
    return {
        'X_test_scaled': X_test_scaled,
        'y_test': np.array([activity_label]),
        'raw_shape': feature_window.shape,
        'features_shape': X_test_csv_features.shape,
        'csv_name': csv_path.name,
        'window_start': start_idx,
        'window_end': start_idx + 250,
        'activity': activity_label,
        'valid_windows_count': len(valid_windows),
    }


def select_random_csvs(
    num_csvs=3,
    part_folder=None,
    participants=None,
    devices=None,
    random_participant=False,
    random_device=False,
):
    """
    Seleciona N arquivos CSV aleatórios.
    
    Parâmetros:
    -----------
    num_csvs : int
        Número de CSVs a selecionar.
    part_folder : str
        Pasta específica (ex: 'part7'). Se None e random_participant=False, usa 'part7'.
    participants : list ou None
        Lista de IDs de participantes (ex: [0, 1, 5, 7]). Se None, descobre automaticamente.
    devices : list ou None
        Lista de IDs de devices (ex: [1, 2, 3, 4, 5]). Se None, descobre automaticamente.
    random_participant : bool
        Se True, escolhe participante aleatório.
    random_device : bool
        Se True, escolhe device aleatório dentro do participante.
    
    Returns:
    --------
    list de Path aos CSVs selecionados.
    """
    dataset_dir = ROOT / 'data' / 'raw' / 'dataset'
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {dataset_dir}")
    
    # Descobrir participantes disponíveis
    if participants is None:
        part_folders = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith('part')])
        participants = [int(p.replace('part', '')) for p in part_folders]
    
    if not participants:
        raise FileNotFoundError(f"Nenhum participante encontrado em {dataset_dir}")
    
    # Selecionar participante
    if random_participant:
        selected_participant = np.random.choice(participants)
        part_folder = f'part{selected_participant}'
        print(f"Participante aleatório selecionado: P{selected_participant}")
    elif part_folder is None:
        part_folder = 'part7'  # Default
    
    participant_dir = dataset_dir / part_folder
    
    if not participant_dir.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {participant_dir}")
    
    # Descobrir devices disponíveis para este participante
    if devices is None:
        csv_files = list(participant_dir.glob('*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"Nenhum CSV encontrado em {participant_dir}")
        
        # Extrair IDs de devices dos nomes (partXdevY.csv → Y)
        devices = []
        for csv_file in csv_files:
            # Ex: part7dev1.csv → device 1
            try:
                dev_id = int(csv_file.stem.split('dev')[-1])
                devices.append(dev_id)
            except ValueError:
                pass
        devices = sorted(list(set(devices)))
    
    if not devices:
        raise FileNotFoundError(f"Nenhum device encontrado em {participant_dir}")
    
    # Selecionar devices
    if random_device:
        num_tests = min(num_csvs, len(devices))
        selected_devices = np.random.choice(devices, size=num_tests, replace=False)
        print(f"Devices aleatórios selecionados: {list(selected_devices)}")
    else:
        num_tests = min(num_csvs, len(devices))
        selected_devices = devices[:num_tests]
    
    # Construir paths dos CSVs
    selected_csvs = []
    for dev_id in selected_devices:
        csv_file = participant_dir / f"{part_folder}dev{dev_id}.csv"
        if csv_file.exists():
            selected_csvs.append(csv_file)
    
    if not selected_csvs:
        raise FileNotFoundError(f"Nenhum CSV encontrado para {part_folder} com devices {selected_devices}")
    
    return selected_csvs
