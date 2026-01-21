# Federated Learning Dataset Splitter

> Stratyfikowany podział danych dla uczenia federacyjnego z 10-Fold Cross-Validation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Spis treści

- [O projekcie](#o-projekcie)
- [Datasety](#datasety)
- [Metodologia podziału](#metodologia-podziału)
- [Struktura projektu](#struktura-projektu)
- [Instalacja](#instalacja)
- [Użycie](#użycie)
- [Dashboardy](#dashboardy)
- [Dokumentacja matematyczna](#dokumentacja-matematyczna)
- [Autor](#autor)

---

## O projekcie

Projekt implementuje **stratyfikowany podział danych** dla scenariuszy **Federated Learning** z pełną obsługą **10-Fold Cross-Validation**. 

### Kluczowe cechy:

- **Stratyfikacja** — zachowanie proporcji klas we wszystkich zbiorach
- **Podział 80/10/10** — Train / Validation / Test
- **10 foldów CV** — każdy fold z identycznym rozkładem klas
- **2-5 klientów** — konfiguracje dla uczenia federacyjnego
- **Rotacja foldów** — każdy klient ma te same dane w innej kolejności
- **Współdzielone Val/Test** — identyczne zbiory walidacyjne i testowe dla wszystkich klientów

---

## Datasety

| Dataset | Typ | Wierszy | Cech | Klas | Domena |
|---------|-----|---------|------|------|--------|
| **RT-IoT2022** | Klasyfikacja | 123,117 | 83 | 12 | Cyberbezpieczeństwo IoT |
| **Letter Recognition** | Klasyfikacja | 20,000 | 16 | 26 | Rozpoznawanie obrazów |
| **Electric Power** | Szereg czasowy | 2,049,279 | 8 | - | Energia / Smart Grid |
| **MIMIC-IV-ED** | Klasyfikacja | 196 | 28 | 2 | Medycyna / SOR |

---

## Metodologia podziału

### Formuła podziału

```
D = D_train ∪ D_val ∪ D_test

gdzie:
  |D_train| = 0.80 × |D|    (80% — dane treningowe)
  |D_val|   = 0.10 × |D|    (10% — dane walidacyjne)
  |D_test|  = 0.10 × |D|    (10% — dane testowe)
```

### Stratyfikowana Cross-Validation

```
D_train = F₁ ∪ F₂ ∪ F₃ ∪ ... ∪ F₁₀

gdzie:
  |Fᵢ| = |D_train| / 10
  p(c | Fᵢ) = p(c | D)  — każdy fold ma identyczny rozkład klas
```

### Rotacja foldów między klientami

```
start_fold(k) = ((k - 1) × floor(10 / K)) MOD 10

Przykład dla K=2 klientów:
  Client 1: [F₁, F₂, F₃, F₄, F₅, F₆, F₇, F₈, F₉, F₁₀]
  Client 2: [F₆, F₇, F₈, F₉, F₁₀, F₁, F₂, F₃, F₄, F₅]
```

---

## Struktura projektu

```
federated-learning-data-split/

├── README.md                          # Ten plik
├── requirements.txt                   # Zależności Python
├── .gitignore                         # Ignorowane pliki
├── LICENSE                            # Licencja MIT

├── scripts/
│   ├── split_all_datasets_fixed.py      # Główny skrypt podziału
│   └── generate_dashboard_json.py       # Generator danych dla dashboardów

├── datasets/
│   ├── dashboard-before-split.html      # Dashboard: przegląd datasetów
│   ├── dashboard-after-split.html       # Dashboard: analiza podziału
│   └── dashboard_data.json              # Dane dla dashboardów

└── Opis_Matematyczny_Podzialu_Danych.pdf  # Dokumentacja matematyczna
```

### Dane (Google Drive)

Pliki CSV z podziałem danych są dostępne na Google Drive:

> **[Pobierz dane z Google Drive](https://drive.google.com/drive/u/1/folders/16TtUzFpqpm-sb2snrRUe7lO_qQqNAlfv)**

Struktura danych:
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

### 2. Instalacja zależności

```bash
pip install -r requirements.txt
```

### 3. Pobranie danych

Pobierz dane z Google Drive i umieść w folderze `data/`:

> **[Google Drive - splits_custom](https://drive.google.com/drive/u/1/folders/16TtUzFpqpm-sb2snrRUe7lO_qQqNAlfv)**

---

## Użycie

### Podział danych

```bash
python scripts/split_all_datasets_fixed.py
```

### Generowanie JSON dla dashboardów

```bash
python scripts/generate_dashboard_json.py
```

### Użycie foldów do Cross-Validation

```python
import pandas as pd

def load_cv_fold(client_folder, client_id, cv_iteration):
    """
    Ładuje dane dla jednej iteracji CV.
    
    Args:
        client_folder: ścieżka do folderu klienta
        client_id: numer klienta (1-5)
        cv_iteration: numer iteracji CV (1-10)
    
    Returns:
        train_data: DataFrame z danymi treningowymi (9 foldów)
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
            all_folds.append(fold_data)  # Pozostałe jako trening
    
    train_data = pd.concat(all_folds, ignore_index=True)
    return train_data, val_fold


# Przykład użycia:
for cv_iter in range(1, 11):
    train, local_val = load_cv_fold('data/splits_custom/rt_iot2022/2_clients/client_1', 1, cv_iter)
    print(f'CV {cv_iter}: Train={len(train)}, Val={len(local_val)}')
    # Trenuj model na train, waliduj na local_val
```

---

## Dashboardy

Projekt zawiera dwa interaktywne dashboardy HTML:

### 1. Dataset Explorer (`dashboard-before-split.html`)

Przegląd oryginalnych datasetów przed podziałem:
- Statystyki ogólne (wiersze, cechy, klasy)
- Rozkład klas (wykres słupkowy)
- Informacje o domenie i typie zadania

### 2. Client Split Analysis (`dashboard-after-split.html`)

Szczegółowa analiza podziału federacyjnego:
- Wybór datasetu i liczby klientów
- Wizualizacja podziału Train/Val/Test
- Rozmiary foldów dla każdego klienta
- Informacje o rotacji i współdzieleniu danych

**Uruchomienie:**
```bash
# Otwórz w przeglądarce
open dashboards/dashboard-before-split.html
open dashboards/dashboard-after-split.html
```

> Dashboardy wymagają pliku `dashboard_data.json` w tym samym folderze.

---

## Dokumentacja matematyczna

Pełny opis matematyczny metodologii znajduje się w pliku:

**[docs/Opis_Matematyczny_Podzialu_Danych.pdf](docs/Opis_Matematyczny_Podzialu_Danych.pdf)**

Zawiera:
- Definicje symboli matematycznych
- Formuły podziału stratyfikowanego
- Algorytm w pseudokodzie
- Schemat blokowy
- Przykłady obliczeniowe

---

## Wymagania

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

---

## Licencja

Ten projekt jest udostępniony na licencji MIT — zobacz plik [LICENSE](LICENSE) po szczegóły.

---

## Autor

**Veronika**

GitHub: [@VeRonikARoNik](https://github.com/VeRonikARoNik)

---

## Podziękowania

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) — Letter Recognition, Electric Power
- [PhysioNet](https://physionet.org/) — MIMIC-IV-ED
- [Kaggle](https://kaggle.com/) — RT-IoT2022

---

<p align="center">
  <i>Jeśli projekt był pomocny, zostaw gwiazdke na GitHubie!</i>
</p>
