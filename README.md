# Federated Learning Dataset Splitter

> Stratyfikowany podzial danych dla uczenia federacyjnego z 10-Fold Cross-Validation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Spis tresci

- [O projekcie](#o-projekcie)
- [Datasety](#datasety)
- [Metodologia podzialu](#metodologia-podzialu)
- [Struktura projektu](#struktura-projektu)
- [Instalacja](#instalacja)
- [Uzycie](#uzycie)
- [Dashboardy](#dashboardy)
- [Dokumentacja matematyczna](#dokumentacja-matematyczna)
- [Autor](#autor)

---

## O projekcie

Projekt implementuje **stratyfikowany podzial danych** dla scenariuszy **Federated Learning** z pelna obsluga **10-Fold Cross-Validation**. 

### Kluczowe cechy:

- **Stratyfikacja** - zachowanie proporcji klas we wszystkich zbiorach
- **Podzial 80/10/10** - Train / Validation / Test
- **10 foldow CV** - kazdy fold z identycznym rozkladem klas
- **2-5 klientow** - konfiguracje dla uczenia federacyjnego
- **Rotacja foldow** - kazdy klient ma te same dane w innej kolejnosci
- **Wspoldzielone Val/Test** - identyczne zbiory walidacyjne i testowe dla wszystkich klientow

---

## Datasety

| Dataset | Typ | Wierszy | Cech | Klas | Domena |
|---------|-----|---------|------|------|--------|
| **RT-IoT2022** | Klasyfikacja | 123,117 | 83 | 12 | Cyberbezpieczenstwo IoT |
| **Letter Recognition** | Klasyfikacja | 20,000 | 16 | 26 | Rozpoznawanie obrazow |
| **Electric Power** | Szereg czasowy | 2,049,279 | 8 | - | Energia / Smart Grid |
| **MIMIC-IV-ED** | Klasyfikacja | 196 | 28 | 2 | Medycyna / SOR |

---

## Metodologia podzialu

### Formula podzialu

```
D = D_train ∪ D_val ∪ D_test

gdzie:
  |D_train| = 0.80 x |D|    (80% - dane treningowe)
  |D_val|   = 0.10 x |D|    (10% - dane walidacyjne)
  |D_test|  = 0.10 x |D|    (10% - dane testowe)
```

### Stratyfikowana Cross-Validation

```
D_train = F1 ∪ F2 ∪ F3 ∪ ... ∪ F10

gdzie:
  |Fi| = |D_train| / 10
  p(c | Fi) = p(c | D)  - kazdy fold ma identyczny rozklad klas
```

### Rotacja foldow miedzy klientami

```
start_fold(k) = ((k - 1) x floor(10 / K)) MOD 10

Przyklad dla K=2 klientow:
  Client 1: [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10]
  Client 2: [F6, F7, F8, F9, F10, F1, F2, F3, F4, F5]
```

---

## Struktura projektu

```
federated-learning-data-split/

├── README.md                             # Ten plik
├── requirements.txt                      # Zaleznosci Python
├── .gitignore                            # Ignorowane pliki
├── LICENSE                               # Licencja MIT
├── Opis_Matematyczny_Podzialu_Danych.pdf # Dokumentacja matematyczna

├── scripts/
│   ├── split_all_datasets_fixed.py      # Glowny skrypt podzialu
│   └── generate_dashboard_json.py       # Generator danych dla dashboardow

└── datasets/
    ├── dashboard-before-split.html      # Dashboard: przeglad datasetow
    ├── dashboard-after-split.html       # Dashboard: analiza podzialu
    └── dashboard_data.json              # Dane dla dashboardow
```

### Dane (Google Drive)

Pliki CSV z podzialem danych sa dostepne na Google Drive:

**[Pobierz dane z Google Drive](https://drive.google.com/drive/u/1/folders/16TtUzFpqpm-sb2snrRUe7lO_qQqNAlfv)**

Struktura danych na dysku:
```
splits_custom/
├── rt_iot2022/
│   ├── 2_clients/
│   │   ├── client_1/
│   │   │   ├── client_1_train_fold_1.csv ... client_1_train_fold_10.csv
│   │   │   ├── client_1_val.csv
│   │   │   └── client_1_test.csv
│   │   └── client_2/
│   ├── 3_clients/
│   ├── 4_clients/
│   └── 5_clients/
├── letter_recognition/
├── electric_power/
└── mimic_iv_ed/
```

---

## Instalacja

### 1. Klonowanie repozytorium

```bash
git clone https://github.com/VeRonikARoNik/federated-learning-data-split.git
cd federated-learning-data-split
```

### 2. Instalacja zaleznosci

```bash
pip install -r requirements.txt
```

### 3. Pobranie danych

Pobierz dane z Google Drive i umiesc w folderze `data/`:

**[Google Drive - splits_custom](https://drive.google.com/drive/u/1/folders/16TtUzFpqpm-sb2snrRUe7lO_qQqNAlfv)**

---

## Uzycie

### Podzial danych

```bash
python scripts/split_all_datasets_fixed.py
```

### Generowanie JSON dla dashboardow

```bash
python scripts/generate_dashboard_json.py
```

### Uzycie foldow do Cross-Validation

```python
import pandas as pd

def load_cv_fold(client_folder, client_id, cv_iteration):
    """
    Laduje dane dla jednej iteracji CV.
    
    Args:
        client_folder: sciezka do folderu klienta
        client_id: numer klienta (1-5)
        cv_iteration: numer iteracji CV (1-10)
    
    Returns:
        train_data: DataFrame z danymi treningowymi (9 foldow)
        val_fold: DataFrame z foldem walidacyjnym (1 fold)
    """
    all_folds = []
    val_fold = None
    
    for fold_num in range(1, 11):
        fold_file = f'{client_folder}/client_{client_id}_train_fold_{fold_num}.csv'
        fold_data = pd.read_csv(fold_file)
        
        if fold_num == cv_iteration:
            val_fold = fold_data  # Ten fold jako walidacja lokalna
        else:
            all_folds.append(fold_data)  # Pozostale jako trening
    
    train_data = pd.concat(all_folds, ignore_index=True)
    return train_data, val_fold


# Przyklad uzycia:
for cv_iter in range(1, 11):
    train, local_val = load_cv_fold('data/splits_custom/rt_iot2022/2_clients/client_1', 1, cv_iter)
    print(f'CV {cv_iter}: Train={len(train)}, Val={len(local_val)}')
    # Trenuj model na train, waliduj na local_val
```

---

## Dashboardy

Projekt zawiera dwa interaktywne dashboardy HTML:

### 1. Dataset Explorer (dashboard-before-split.html)

Przeglad oryginalnych datasetow przed podzialem:
- Statystyki ogolne (wiersze, cechy, klasy)
- Rozklad klas (wykres slupkowy)
- Informacje o domenie i typie zadania

### 2. Client Split Analysis (dashboard-after-split.html)

Szczegolowa analiza podzialu federacyjnego:
- Wybor datasetu i liczby klientow
- Wizualizacja podzialu Train/Val/Test
- Rozmiary foldow dla kazdego klienta
- Informacje o rotacji i wspoldzieleniu danych

**Uruchomienie:**
```bash
# Otworz w przegladarce
open datasets/dashboard-before-split.html
open datasets/dashboard-after-split.html
```

Dashboardy wymagaja pliku `dashboard_data.json` w tym samym folderze.

---

## Dokumentacja matematyczna

Pelny opis matematyczny metodologii znajduje sie w pliku:

**[Opis_Matematyczny_Podzialu_Danych.pdf](Opis_Matematyczny_Podzialu_Danych.pdf)**

Zawiera:
- Definicje symboli matematycznych
- Formuly podzialu stratyfikowanego
- Algorytm w pseudokodzie
- Schemat blokowy
- Przyklady obliczeniowe

---

## Wymagania

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

---

## Licencja

Ten projekt jest udostepniony na licencji MIT - zobacz plik [LICENSE](LICENSE) po szczegoly.

---

## Autor

**Veronika**

GitHub: [@VeRonikARoNik](https://github.com/VeRonikARoNik)

---

## Podziekowania

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) - Letter Recognition, Electric Power, RT-IoT2022
- [PhysioNet](https://physionet.org/) - MIMIC-IV-ED

---

<p align="center">
  <i>Jesli projekt byl pomocny, zostaw gwiazdke na GitHubie!</i>
</p>
