# ECM_2526_FinalProject
Projet final dans le cadre du cours DDEFI

Présentation de l'arbo Le nom de chaque personne ainsi que son compte Github et Kaggle

Les membres de ce projet sont : 
  - Julien PATRAT (julienpatrat)
  - Lucas JENOT (lujenot-prog)
  - Rayan BOUKHEZZA (Rayanboukhezza)
  - Benoît MEUNIER (Benoit1020)

## Setup (Python)
```bash
cd Script
python -m venv .venv
source .venv/bin/activate  # mac/linux
pip install -r requirements.txt

# Bankruptcy Prediction — Polish & USA pipelines

Ce dépôt contient un projet de **prédiction de faillite d’entreprises** (classification binaire), avec deux datasets :
- 🇵🇱 **Polish Companies Bankruptcy (UCI)** : variables `Attr1..Attr64`
- 🇺🇸 **US Company-Year dataset** : variables `X1..X18` + `status_label`

Objectifs :
1) Construire une pipeline reproductible (préparation → split → entraînement → scoring → calibration → buckets)
2) Comparer des modèles (baseline logreg vs modèle plus performant)
3) Produire un score de risque **R** (probabilité calibrée) + des **risk buckets**
4) Alimenter un mini-site de démonstration (lookup par identifiant d’entreprise)

---

## Structure du dépôt

- `Script/src/data/`  
  Scripts de préparation et de split (Polish/USA)
- `Script/src/models/`  
  Entraînement, calibration, scoring, risk buckets, génération des lookups JSON
- `Script/src/evaluation/`  
  Évaluation / tuning de seuils
- `reports/`  
  Rapports JSON/CSV (audit dataset, métriques, buckets, comparaison modèles)
- `Data/source/`  
  Données brutes (non versionnées si trop volumineuses)
- `Data/processed/`  
  Données transformées (générées, ignorées par git)

⚠️ Les dossiers `Data/processed/` et `models/` contiennent des artefacts générés et ne sont pas versionnés (voir `.gitignore`).

---

## Datasets

### 🇵🇱 Polish Companies Bankruptcy (UCI)
- Format : 5 fichiers (1stYear..5thYear).  
- Dans ce projet, on utilise principalement `1stYear`.
- Features : `Attr1..Attr64` (ratios financiers)
- Target : `class` (1 = faillite, 0 = non-faillite)

Documentation variables :
- `reports/polish_feature_dictionary.md`

### 🇺🇸 USA Company-Year dataset
- Format : une ligne = une entreprise pour une année fiscale (`company_name`, `fyear`)
- Features : `X1..X18`
- Target : `status_label` (mappé en `class`)

Documentation variables :
- `reports/us_feature_dictionary.md`

---

## Reproduire la pipeline (commandes)

### 🇵🇱 Pipeline Polish (Year 1)
1) Placer le zip UCI dans :
`Data/source/polish+companies+bankruptcy+data.zip`

2) Lancer :
```bash
python Script/src/data/prepare_year1.py
python Script/src/data/split_year1.py
python Script/src/models/train_logreg_baseline.py
python Script/src/models/calibrate_logreg.py
python Script/src/models/train_hgb.py
python Script/src/models/score_all_year1_hgb.py
python Script/src/models/risk_buckets.py
python Script/src/evaluation/threshold_tuning.py
