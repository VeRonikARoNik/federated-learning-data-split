import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# ==========================================
# KONFIGURACJA
# ==========================================
BASE_PATH = r'C:\Users\Veronika\Desktop\data\datasets'
SPLITS_DIR = os.path.join(BASE_PATH, 'splits_custom')
OUTPUT_FILE = os.path.join(BASE_PATH, 'dashboard_data.json')

# Konfiguracja datasetow (dla opisu)
DATASET_INFO = {
    'rt_iot2022': {
        'fullName': 'Real-Time IoT Intrusion Detection Dataset 2022',
        'target': 'Attack_type',
        'type': 'classification',
        'domain': {
            'pl': 'Cyberbezpieczenstwo IoT',
            'en': 'IoT Cybersecurity'
        },
        'description': {
            'pl': 'Zbior danych do wykrywania atakow sieciowych w urzadzeniach IoT.',
            'en': 'Dataset for detecting network attacks in IoT devices.'
        }
    },
    'letter_recognition': {
        'fullName': 'Letter Recognition Database',
        'target': 'letter',
        'type': 'classification',
        'domain': {
            'pl': 'Rozpoznawanie obrazow',
            'en': 'Image Recognition'
        },
        'description': {
            'pl': 'Klasyczny zbior danych do rozpoznawania 26 liter alfabetu lacinskiego.',
            'en': 'Classic dataset for recognizing 26 letters of the Latin alphabet.'
        }
    },
    'electric_power': {
        'fullName': 'Individual Household Electric Power Consumption',
        'target': 'Global_active_power',
        'type': 'timeseries',
        'domain': {
            'pl': 'Energia / Smart Grid',
            'en': 'Energy / Smart Grid'
        },
        'description': {
            'pl': 'Pomiary zuzycia energii elektrycznej w gospodarstwie domowym.',
            'en': 'Electric power consumption measurements in household.'
        }
    },
    'mimic_iv_ed': {
        'fullName': 'MIMIC-IV Emergency Department Demo',
        'target': 'disposition',
        'type': 'classification',
        'domain': {
            'pl': 'Medycyna / SOR',
            'en': 'Healthcare / Emergency'
        },
        'description': {
            'pl': 'Dane z Oddzialu Ratunkowego. Przewidywanie hospitalizacji.',
            'en': 'Emergency Department data. Predicting hospitalization.'
        }
    }
}

# ==========================================
# FUNCTION: Analyze single dataset
# ==========================================
def analyze_dataset(dataset_name):
    """Analizuje dataset i zwraca statystyki."""
    
    dataset_dir = os.path.join(SPLITS_DIR, dataset_name)
    
    if not os.path.exists(dataset_dir):
        print(f'  [SKIP] {dataset_name} - folder nie istnieje')
        return None
    
    print(f'  Analizuje {dataset_name}...')
    
    result = {
        'name': dataset_name,
        'info': DATASET_INFO.get(dataset_name, {}),
        'clients': {}
    }
    
    # Sprawdz dostepne konfiguracje klientow
    for num_clients in [2, 3, 4, 5]:
        clients_dir = os.path.join(dataset_dir, f'{num_clients}_clients')
        
        if not os.path.exists(clients_dir):
            continue
        
        client_config = {
            'numClients': num_clients,
            'clientsData': []
        }
        
        # Analizuj kazdego klienta
        for client_id in range(1, num_clients + 1):
            client_folder = os.path.join(clients_dir, f'client_{client_id}')
            
            if not os.path.exists(client_folder):
                continue
            
            client_stats = {
                'clientId': client_id,
                'folds': [],
                'trainTotal': 0,
                'valRows': 0,
                'testRows': 0,
                'features': 0,
                'classDistribution': {}
            }
            
            # Wczytaj foldy treningowe
            for fold_num in range(1, 11):
                fold_file = os.path.join(
                    client_folder,
                    f'client_{client_id}_train_fold_{fold_num}.csv'
                )
                if os.path.exists(fold_file):
                    df_fold = pd.read_csv(fold_file)
                    fold_size = len(df_fold)
                    client_stats['folds'].append(fold_size)
                    
                    # Pobierz liczbe cech z pierwszego folda
                    if fold_num == 1:
                        client_stats['features'] = len(df_fold.columns) - 1
            
            client_stats['trainTotal'] = sum(client_stats['folds'])
            
            # Wczytaj walidacje
            val_file = os.path.join(client_folder, f'client_{client_id}_val.csv')
            if os.path.exists(val_file):
                df_val = pd.read_csv(val_file)
                client_stats['valRows'] = len(df_val)
                
                # Rozklad klas (dla klasyfikacji)
                target_col = df_val.columns[-1]
                if df_val[target_col].dtype == 'object' or df_val[target_col].nunique() < 50:
                    class_dist = df_val[target_col].value_counts().to_dict()
                    # Konwertuj klucze na stringi (dla JSON)
                    client_stats['classDistribution'] = {
                        str(k): int(v) for k, v in class_dist.items()
                    }
            
            # Wczytaj test
            test_file = os.path.join(client_folder, f'client_{client_id}_test.csv')
            if os.path.exists(test_file):
                df_test = pd.read_csv(test_file)
                client_stats['testRows'] = len(df_test)
            
            client_config['clientsData'].append(client_stats)
        
        # Oblicz statystyki zbiorcze dla konfiguracji
        if client_config['clientsData']:
            first_client = client_config['clientsData'][0]
            total = first_client['trainTotal'] + first_client['valRows'] + first_client['testRows']
            
            client_config['summary'] = {
                'trainRows': first_client['trainTotal'],
                'valRows': first_client['valRows'],
                'testRows': first_client['testRows'],
                'totalRows': total,
                'features': first_client['features'],
                'numFolds': len(first_client['folds']),
                'avgFoldSize': int(np.mean(first_client['folds'])) if first_client['folds'] else 0,
                'trainPercent': round(100 * first_client['trainTotal'] / total, 2) if total > 0 else 0,
                'valPercent': round(100 * first_client['valRows'] / total, 2) if total > 0 else 0,
                'testPercent': round(100 * first_client['testRows'] / total, 2) if total > 0 else 0,
                'classDistribution': first_client['classDistribution']
            }
        
        result['clients'][str(num_clients)] = client_config
    
    return result

# ==========================================
# FUNCTION: Generate dashboard JSON
# ==========================================
def generate_dashboard_json():
    """Generuje plik JSON dla dashboardow."""
    
    print('=' * 60)
    print('GENEROWANIE DASHBOARD_DATA.JSON')
    print('=' * 60)
    
    # Znajdz wszystkie dostepne datasety
    available_datasets = []
    if os.path.exists(SPLITS_DIR):
        for item in os.listdir(SPLITS_DIR):
            item_path = os.path.join(SPLITS_DIR, item)
            if os.path.isdir(item_path):
                available_datasets.append(item)
    
    print(f'\nZnalezione datasety: {available_datasets}')
    
    # Analizuj kazdy dataset
    dashboard_data = {
        'generatedAt': pd.Timestamp.now().isoformat(),
        'basePath': SPLITS_DIR,
        'datasets': {}
    }
    
    print('\nAnalizowanie datasetow:')
    for dataset_name in available_datasets:
        result = analyze_dataset(dataset_name)
        if result:
            dashboard_data['datasets'][dataset_name] = result
    
    # Zapisz JSON
    print(f'\nZapisywanie do: {OUTPUT_FILE}')
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
    
    print('\n' + '=' * 60)
    print('PODSUMOWANIE')
    print('=' * 60)
    
    for dataset_name, data in dashboard_data['datasets'].items():
        print(f'\n{dataset_name}:')
        for num_clients, config in data['clients'].items():
            summary = config.get('summary', {})
            print(f'  {num_clients} klientow:')
            print(f'    Train: {summary.get("trainRows", 0):,} ({summary.get("trainPercent", 0)}%)')
            print(f'    Val:   {summary.get("valRows", 0):,} ({summary.get("valPercent", 0)}%)')
            print(f'    Test:  {summary.get("testRows", 0):,} ({summary.get("testPercent", 0)}%)')
            print(f'    Folds: {summary.get("numFolds", 0)}, avg size: {summary.get("avgFoldSize", 0):,}')
    
    print(f'\nPlik zapisany: {OUTPUT_FILE}')
    print(f'Rozmiar: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB')
    
    return dashboard_data

# ==========================================
# MAIN
# ==========================================
if __name__ == '__main__':
    generate_dashboard_json()
