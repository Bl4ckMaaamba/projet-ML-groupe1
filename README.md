# ML III — Détection d'anomalies (AI4I 2020)

Projet de groupe pour le cours **Machine Learning III (Unsupervised Learning)** à Albert School.

- **Équipe** : Noah, Isaac, Diego
- **Dataset** : AI4I 2020 Predictive Maintenance (`ai4i2020.csv`, 10 000 obs)
- **Énoncé** : `1773592112_Session5_Advanced_Anomaly_Detection_Assessment.pdf`
- **Rendu final** : `notebooks/final_report.ipynb`

## Démarrage rapide

```bash
# 1. Se placer dans le dossier
cd ML-projet-cours

# 2. Créer un environnement virtuel (Python 3.11+)
python3 -m venv .venv
source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Activer nbstripout (clean les outputs des notebooks dans les commits)
nbstripout --install

# 5. Lancer Jupyter
jupyter lab
```

## Workflow Git

Chacun travaille sur sa branche perso, créée depuis `main` :

```bash
git checkout main
git pull
git checkout -b noah/eda      # Noah uniquement
git checkout -b isaac/models  # Isaac uniquement
git checkout -b diego/eval    # Diego uniquement
```

Commits réguliers (≥ 6 par personne), conventionnels (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`).
Merge sur `main` en phase 3 (assemblage).

## Documentation

- `PLAN.md` — plan d'ensemble, stratégie 20/20, phasage temporel
- `TASKS_NOAH.md` — checklist Noah (EDA + preprocessing)
- `TASKS_ISAAC.md` — checklist Isaac (4 modèles + tuning)
- `TASKS_DIEGO.md` — checklist Diego (viz + analyse coût + reco business)

## Structure

```
ML-projet-cours/
├── ai4i2020.csv
├── 1773592112_Session5_*.pdf
├── PLAN.md
├── TASKS_*.md
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Noah
│   ├── models.py           # Isaac
│   ├── evaluation.py       # Diego
│   └── visualization.py    # Diego
├── notebooks/
│   ├── 01_eda.ipynb        # Noah (sandbox)
│   ├── 02_models.ipynb     # Isaac (sandbox)
│   ├── 03_evaluation.ipynb # Diego (sandbox)
│   └── final_report.ipynb  # rendu final
└── outputs/
    ├── figures/            # PNG exportés
    └── results/            # CSV des prédictions
```
