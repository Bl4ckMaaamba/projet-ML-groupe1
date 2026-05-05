# ML III — Détection d'anomalies (AI4I 2020)

Projet de groupe pour le cours **Machine Learning III (Unsupervised Learning)** à Albert School.

- **Équipe** : Noah Soulisse, Isaac, Diego Guenancia
- **Dataset** : AI4I 2020 Predictive Maintenance (`ai4i2020.csv`, 10 000 obs)
- **Rendu** : `final_report.ipynb` (à la racine)

## Démarrage

```bash
git clone https://github.com/Bl4ckMaaamba/projet-ML-groupe1.git
cd projet-ML-groupe1
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab final_report.ipynb
```

Puis `Restart & Run All` dans Jupyter — le notebook s'exécute de bout en bout en ~2 minutes (entraînement des 4 modèles + sensitivity analysis).

## Contenu du notebook

`final_report.ipynb` est **autonome** : tous les helpers (preprocessing, modèles, évaluation, visualisation) sont définis inline. Il couvre les 4 parties de l'énoncé :

1. **EDA & Preprocessing** — distributions, corrélations, justification des choix de scaling et d'encoding
2. **Modeling & Tuning** — 4 modèles (Isolation Forest, One-Class SVM, LOF, Elliptic Envelope), hyperparamètres calibrés et justifiés
3. **Visualisation & Deep Dive** — projection PCA, superposition des décisions des 4 modèles, deep dive mathématique sur 4 cas réels de désaccord
4. **Cost Analysis & Recommandation** — analyse coût (FP €500, FN €15 000), sensitivity analysis sur `nu`, recommandation business chiffrée

## Résultat clé

**Modèle recommandé** : One-Class SVM avec `nu = 0.25`.
**Pertes attendues annuelles** : €2.32 M (vs. €5.09 M sans détection, **−54.3 %**).
**Recall** : 76 % sur les pannes réelles, au prix de ~6 fausses alertes par jour.

## Structure du repo

```
.
├── README.md
├── requirements.txt
├── ai4i2020.csv
└── final_report.ipynb
```

Le notebook est segmenté par les **4 parties de l'énoncé** (bannières "Part 1", "Part 2", "Part 3", "Part 4") pour faciliter la correction.
