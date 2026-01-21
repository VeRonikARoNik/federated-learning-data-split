import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
from pathlib import Path

# ==========================================
# KONFIGURACJA
# ==========================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Sciezki
BASE_PATH = r'C:\Users\Veronika\Desktop\data\datasets'

DATASETS = {
    'rt_iot2022': {
        'file': r'rt-iot2022\RT_IOT2022',
        'target': 'Attack_type',
        'index_col': 0,
        'type': 'classification'
    },
    'letter_recognition': {
        'file': r'letter+recognition\letter-recognition.data',
        'target': 'letter',
        'header': None,
        'type': 'classification'
    },
    'electric_power': {
        'file': r'individual+household+electric+power+consumption\household_power_consumption.txt',
        'target': 'Global_active_power',
        'separator': ';',
        'type': 'timeseries'
    },
    'mimic_iv_ed': {
        'file': r'mimic-iv-ed-demo-2.2\mimic-iv-ed-demo-2.2\ed',
        'target': 'disposition',
        'type': 'classification',  # Zmienione na classification (binarna klasyfikacja)
        'special': 'mimic_ed'
    }
}

OUTPUT_DIR = os.path.join(BASE_PATH, 'splits_custom')

# ==========================================
# HELPER FUNCTION: Load dataset
# ==========================================
def load_dataset(dataset_name):
    
    config = DATASETS[dataset_name]
    print('\n' + '='*60)
    print('Loading ' + dataset_name + '...')
    print('='*60)
    
    file_path = os.path.join(BASE_PATH, config['file'])
    
    if dataset_name == 'rt_iot2022':
        df = pd.read_csv(file_path, index_col=config['index_col'])
    
    elif dataset_name == 'letter_recognition':
        df = pd.read_csv(
            file_path,
            header=config['header'],
            sep=','
        )
        col_names = ['letter'] + [f'feature_{i}' for i in range(1, 17)]
        df.columns = col_names
    
    elif dataset_name == 'electric_power':
        df = pd.read_csv(
            file_path,
            sep=config['separator'],
            na_values=['?'],
            low_memory=False
        )
        # Convert target to numeric
        df[config['target']] = pd.to_numeric(df[config['target']], errors='coerce')
        # Drop rows where target is NaN
        df = df.dropna(subset=[config['target']])
        print(f'After cleaning: {len(df)} rows')
    
    elif dataset_name == 'mimic_iv_ed':
        # Load and merge MIMIC-IV-ED tables
        ed_path = file_path
        edstays = pd.read_csv(os.path.join(ed_path, 'edstays.csv', 'edstays.csv'))
        vitalsign = pd.read_csv(os.path.join(ed_path, 'vitalsign.csv', 'vitalsign.csv'))
        triage = pd.read_csv(os.path.join(ed_path, 'triage.csv', 'triage.csv'))
        
        # Convert all vital sign columns to numeric (fix for string values like '64')
        numeric_cols = ['heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'temperature', 'pain']
        for col in numeric_cols:
            if col in vitalsign.columns:
                vitalsign[col] = pd.to_numeric(vitalsign[col], errors='coerce')
        
        # Aggregate vital signs per stay (mean, std, min, max)
        vs_agg = vitalsign.groupby('stay_id').agg({
            'heartrate': ['mean', 'std', 'min', 'max', 'count'],
            'resprate': ['mean', 'std', 'min', 'max'],
            'o2sat': ['mean', 'std', 'min', 'max'],
            'sbp': ['mean', 'std', 'min', 'max'],
            'dbp': ['mean', 'std', 'min', 'max'],
            'temperature': ['mean', 'std'],
            'pain': ['mean', 'max']
        })
        vs_agg.columns = ['_'.join(col).strip() for col in vs_agg.columns.values]
        vs_agg = vs_agg.reset_index()
        
        # Merge with edstays
        df = edstays.merge(vs_agg, on='stay_id', how='inner')
        
        # Add triage data
        triage_subset = triage[['stay_id', 'acuity']].copy()
        df = df.merge(triage_subset, on='stay_id', how='left')
        
        # Encode categorical variables
        df['gender_encoded'] = (df['gender'] == 'M').astype(int)
        df['transport_ambulance'] = (df['arrival_transport'] == 'AMBULANCE').astype(int)
        
        # Keep only ADMITTED and HOME for binary classification
        df = df[df['disposition'].isin(['ADMITTED', 'HOME'])].copy()
        df['disposition'] = (df['disposition'] == 'ADMITTED').astype(int)
        
        # Drop non-feature columns
        drop_cols = ['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 
                     'gender', 'race', 'arrival_transport']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        # Fill NaN with median
        for col in df.columns:
            if col != config['target'] and df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
        
        print(f'After merging and cleaning: {len(df)} stays')
        print(f'Features: {len(df.columns) - 1}')
        print(f'ADMITTED: {df[config["target"]].sum()}, HOME: {(~df[config["target"]].astype(bool)).sum()}')
        if len(df) > 100000:
            df = df.head(100000)
        print(f'After cleaning: {len(df)} rows')
    
    print('Shape: ' + str(df.shape))
    print('Columns: ' + str(df.columns.tolist()[:5]) + '... (' + str(df.shape[1]) + ' total)')
    
    return df, config

# ==========================================
# FUNCTION: Stratified split for classification
# ==========================================
def stratified_split_classification(df, config):
    """
    Podział ze stratyfikacją dla danych klasyfikacyjnych.
    Zachowuje proporcje klas w train/val/test.
    """
    target_col = config['target']
    
    # Sprawdź minimalną liczbę próbek na klasę
    class_counts = df[target_col].value_counts()
    min_class_count = class_counts.min()
    
    print(f'\n  Class distribution in full dataset:')
    print(f'    Total classes: {len(class_counts)}')
    print(f'    Min samples per class: {min_class_count}')
    print(f'    Top 5 classes: {dict(class_counts.head())}')
    
    # Jeśli są klasy z < 10 próbkami, usuń je (nie da się stratyfikować)
    if min_class_count < 10:
        print(f'\n  WARNING: Removing classes with < 10 samples for stratification')
        valid_classes = class_counts[class_counts >= 10].index
        df = df[df[target_col].isin(valid_classes)].copy()
        print(f'    Remaining: {len(df)} rows, {len(valid_classes)} classes')
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Pierwszy podział: 80% train, 20% temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )
    
    # Drugi podział: 50% val, 50% test (z temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )
    
    # Złóż z powrotem DataFrames
    train_data = pd.concat([X_train, y_train], axis=1)
    val_data = pd.concat([X_val, y_val], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    return train_data, val_data, test_data

# ==========================================
# FUNCTION: Sequential split for timeseries
# ==========================================
def sequential_split_timeseries(df, config):
    """
    Podział sekwencyjny dla danych czasowych.
    Zachowuje kolejność czasową.
    """
    total_rows = len(df)
    
    train_size = int(total_rows * 0.8)
    val_size = int(total_rows * 0.1)
    
    train_data = df.iloc[:train_size].copy()
    val_data = df.iloc[train_size:train_size+val_size].copy()
    test_data = df.iloc[train_size+val_size:].copy()
    
    return train_data, val_data, test_data

# ==========================================
# FUNCTION: Create stratified 10-fold CV
# ==========================================
def create_stratified_folds(train_data, config):
    """
    Tworzy 10 stratyfikowanych foldów z danych treningowych.
    Każdy fold ma podobny rozkład klas.
    """
    target_col = config['target']
    
    if config['type'] == 'classification':
        X = train_data.drop(columns=[target_col])
        y = train_data[target_col]
        
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
        
        folds = []
        fold_class_distributions = []
        
        for fold_idx, (_, fold_indices) in enumerate(skf.split(X, y)):
            fold_data = train_data.iloc[fold_indices].copy()
            folds.append(fold_data)
            
            # Zapisz rozkład klas dla weryfikacji
            fold_dist = fold_data[target_col].value_counts(normalize=True)
            fold_class_distributions.append(fold_dist)
        
        # Pokaż rozkład klas w foldach
        print(f'\n  Stratified fold class distribution (top 3 classes):')
        for i, dist in enumerate(fold_class_distributions[:3]):  # Pokaż tylko 3 pierwsze foldy
            top3 = dict(dist.head(3))
            print(f'    Fold {i+1}: {top3}')
        print(f'    ... (remaining folds have similar distribution)')
        
    else:
        # Dla timeseries - zwykły podział sekwencyjny
        fold_size = len(train_data) // 10
        folds = []
        for i in range(10):
            start_idx = i * fold_size
            if i == 9:
                fold_data = train_data.iloc[start_idx:].copy()
            else:
                fold_data = train_data.iloc[start_idx:start_idx + fold_size].copy()
            folds.append(fold_data)
    
    return folds

# ==========================================
# FUNCTION: Split dataset with stratification
# ==========================================
def split_dataset(dataset_name, df, config):
    """
    Główna funkcja podziału danych.
    - Dla klasyfikacji: stratyfikowany podział
    - Dla timeseries: sekwencyjny podział
    """
    print('\nSplitting ' + dataset_name + ' with STRATIFICATION...')
    
    if config['type'] == 'classification':
        train_data, val_data, test_data = stratified_split_classification(df, config)
    else:
        train_data, val_data, test_data = sequential_split_timeseries(df, config)
    
    total_rows = len(train_data) + len(val_data) + len(test_data)
    
    train_pct = len(train_data) / total_rows * 100
    val_pct = len(val_data) / total_rows * 100
    test_pct = len(test_data) / total_rows * 100
    
    print(f'\n  Split results:')
    print(f'    Train: {len(train_data)} rows ({train_pct:.2f}%)')
    print(f'    Val:   {len(val_data)} rows ({val_pct:.2f}%)')
    print(f'    Test:  {len(test_data)} rows ({test_pct:.2f}%)')
    
    # Weryfikacja proporcji klas
    if config['type'] == 'classification':
        target_col = config['target']
        print(f'\n  Class distribution verification (top 3 classes):')
        
        train_dist = train_data[target_col].value_counts(normalize=True).head(3)
        val_dist = val_data[target_col].value_counts(normalize=True).head(3)
        test_dist = test_data[target_col].value_counts(normalize=True).head(3)
        
        print(f'    Train: {dict(train_dist)}')
        print(f'    Val:   {dict(val_dist)}')
        print(f'    Test:  {dict(test_dist)}')
    
    # Utwórz stratyfikowane foldy
    print('\n  Creating 10 STRATIFIED folds for Cross-Validation...')
    folds = create_stratified_folds(train_data, config)
    
    print(f'\n  Fold sizes:')
    for i, fold in enumerate(folds):
        print(f'    Fold {i+1}: {len(fold)} rows')
    print(f'    Total: {sum(len(f) for f in folds)} rows')
    
    return folds, val_data, test_data

# ==========================================
# FUNCTION: Save federated split with CV structure
# ==========================================
def save_federated_split(dataset_name, num_clients, folds, val_data, test_data, output_dir):
    """
    Zapisuje dane dla każdego klienta z pełną strukturą CV.
    
    Struktura dla każdego klienta:
    - 10 plików treningowych (jeden na każdy fold)
    - 1 plik walidacyjny (wspólny)
    - 1 plik testowy (wspólny)
    
    Dodatkowo: plik z instrukcją jak używać foldów do CV
    """
    dataset_dir = os.path.join(output_dir, dataset_name)
    client_dir = os.path.join(dataset_dir, str(num_clients) + '_clients')
    os.makedirs(client_dir, exist_ok=True)
    
    print('\n  Saving ' + str(num_clients) + ' clients...')
    
    for client_id in range(1, num_clients + 1):
        client_folder = os.path.join(client_dir, 'client_' + str(client_id))
        os.makedirs(client_folder, exist_ok=True)
        
        # Rotacja foldów dla różnych klientów
        start_fold = ((client_id - 1) * (len(folds) // num_clients)) % len(folds)
        
        # Zapisz foldy treningowe
        for fold_num in range(1, 11):
            fold_idx = (start_fold + fold_num - 1) % len(folds)
            fold_data = folds[fold_idx]
            
            filename = os.path.join(
                client_folder,
                'client_' + str(client_id) + '_train_fold_' + str(fold_num) + '.csv'
            )
            fold_data.to_csv(filename, index=False)
        
        # Zapisz val i test
        val_file = os.path.join(client_folder, 'client_' + str(client_id) + '_val.csv')
        test_file = os.path.join(client_folder, 'client_' + str(client_id) + '_test.csv')
        
        val_data.to_csv(val_file, index=False)
        test_data.to_csv(test_file, index=False)
    
    # Zapisz instrukcję CV
    cv_instructions = """
# 10-Fold Cross-Validation Instructions

## Jak używać foldów do Cross-Validation:

Dla każdej iteracji i (i = 1, 2, ..., 10):
- **Dane treningowe**: Połącz wszystkie foldy OPRÓCZ fold_i
- **Dane walidacyjne lokalne**: Użyj fold_i
- **Dane walidacyjne globalne**: Użyj pliku _val.csv
- **Dane testowe**: Użyj pliku _test.csv (tylko na końcu!)

## Przykład w Pythonie:

```python
import pandas as pd

def load_cv_fold(client_folder, client_id, cv_iteration):
    '''
    Ładuje dane dla jednej iteracji CV.
    cv_iteration: 1-10
    '''
    all_folds = []
    val_fold = None
    
    for fold_num in range(1, 11):
        fold_file = f'{client_folder}/client_{client_id}_train_fold_{fold_num}.csv'
        fold_data = pd.read_csv(fold_file)
        
        if fold_num == cv_iteration:
            val_fold = fold_data  # Ten fold jako walidacja lokalna
        else:
            all_folds.append(fold_data)  # Pozostałe jako trening
    
    train_data = pd.concat(all_folds, ignore_index=True)
    return train_data, val_fold

# Użycie:
for cv_iter in range(1, 11):
    train, local_val = load_cv_fold('client_1', 1, cv_iter)
    # Trenuj model na train, waliduj na local_val
```

## Struktura plików:
- client_X_train_fold_1.csv ... client_X_train_fold_10.csv (foldy treningowe)
- client_X_val.csv (walidacja globalna - wspólna dla wszystkich klientów)
- client_X_test.csv (test - wspólny dla wszystkich klientów)
"""
    
    cv_file = os.path.join(client_dir, 'CV_INSTRUCTIONS.md')
    with open(cv_file, 'w', encoding='utf-8') as f:
        f.write(cv_instructions)
    
    print('    ' + str(num_clients) + ' clients saved to ' + client_dir)

# ==========================================
# MAIN: Process all datasets
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print('\n' + '='*60)
    print('PROCESSING ALL DATASETS WITH STRATIFIED SPLIT')
    print('='*60)
    
    results = {}
    
    for dataset_name in DATASETS.keys():
        try:
            df, config = load_dataset(dataset_name)
            
            folds, val_data, test_data = split_dataset(dataset_name, df, config)
            
            print('\nSaving federated configurations...')
            for num_clients in [2, 3, 4, 5]:
                save_federated_split(
                    dataset_name,
                    num_clients,
                    folds,
                    val_data,
                    test_data,
                    OUTPUT_DIR
                )
            
            results[dataset_name] = {
                'total_rows': sum(len(f) for f in folds) + len(val_data) + len(test_data),
                'features': folds[0].shape[1] - 1,
                'train_rows': sum(len(f) for f in folds),
                'val_rows': len(val_data),
                'test_rows': len(test_data),
                'folds': len(folds),
                'type': config['type']
            }
            
        except Exception as e:
            print('Error processing ' + dataset_name + ': ' + str(e))
            import traceback
            traceback.print_exc()
    
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    
    for dataset_name, stats in results.items():
        total = stats['train_rows'] + stats['val_rows'] + stats['test_rows']
        print('\n' + dataset_name + f" ({stats['type']}):")
        print('  Total rows: ' + str(stats['total_rows']))
        print('  Features: ' + str(stats['features']))
        print(f"  Train: {stats['train_rows']} ({stats['train_rows']/total*100:.2f}%)")
        print(f"  Val: {stats['val_rows']} ({stats['val_rows']/total*100:.2f}%)")
        print(f"  Test: {stats['test_rows']} ({stats['test_rows']/total*100:.2f}%)")
        print('  Folds: ' + str(stats['folds']) + ' (STRATIFIED)' if stats['type'] == 'classification' else '')
        print('  Location: ' + OUTPUT_DIR + '/' + dataset_name + '/')
    
    print('\n' + '='*60)
    print('ALL DATASETS PROCESSED WITH STRATIFICATION')
    print('='*60)

# ==========================================
# RUN
# ==========================================
if __name__ == '__main__':
    main()
