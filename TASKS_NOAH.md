# Tâches — Noah

**Rôle** : Data Engineer
**Parts énoncé couvertes** : Part 1 (EDA & Preprocessing)
**Branche Git** : `noah/eda`

---

## Ce que tu produis

| Livrable                          | Chemin                          |
|-----------------------------------|---------------------------------|
| Module de preprocessing           | `src/preprocessing.py`          |
| Notebook d'exploration            | `notebooks/01_eda.ipynb`        |
| Section "EDA + Preprocessing" dans le rapport final | dans `notebooks/final_report.ipynb` (phase 4) |

---

## Ce dont les autres ont besoin de toi

- **Isaac** attend de toi : un `X_train` propre (numériques scalées, `Type` encodée) + le **taux réel d'anomalies** dans `Machine failure` pour qu'il calibre ses grilles d'hyperparamètres autour. Tu lui donnes ça à la fin de la Phase 1.
- **Diego** attend de toi : la même `X_train`, et idéalement les **noms de colonnes après encoding** (pour ses analyses de feature importance).

---

## Checklist détaillée

### Setup (Phase 0 — 15 min)
- [ ] `git init` à la racine, créer ta branche : `git checkout -b noah/eda`
- [ ] Vérifier la présence de `ai4i2020.csv` à la racine
- [ ] Créer le squelette : `src/__init__.py`, `src/preprocessing.py`, `notebooks/01_eda.ipynb`
- [ ] **Premier commit** : `chore: init project structure`

### EDA — partie 1 (Phase 1 — 45 min)
- [ ] Charger le CSV avec pandas
- [ ] `df.info()`, `df.describe()`, `df.isna().sum()`
- [ ] Identifier les colonnes à drop pendant le training :
  - `UDI` (id pur)
  - `Product ID` (id texte)
  - `Machine failure` (la target — on la garde dans une variable séparée pour la Part 4 mais on ne la touche plus)
  - `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (sous-types de pannes — fuites de la target)
- [ ] Distributions des 5 features numériques :
  - Air temperature [K]
  - Process temperature [K]
  - Rotational speed [rpm]
  - Torque [Nm]
  - Tool wear [min]
- [ ] Histogrammes + boxplots pour chacune → repérer skew, outliers visuels
- [ ] Distribution de `Type` (M / L / H — quelle proportion ?)
- [ ] **Commit** : `feat: load AI4I dataset and check schema`

### EDA — partie 2 (Phase 2 — 1h)
- [ ] Matrice de corrélation + heatmap (seaborn)
  - Tu devrais voir une forte corrélation Air temp ↔ Process temp → noter
  - Et probablement Rotational speed ↔ Torque (relation physique inverse)
- [ ] Calculer le **taux réel** de `Machine failure` dans le dataset
  - `df["Machine failure"].mean()` → tu devrais trouver ~3.4 %
  - **Important** : ce chiffre te sert UNIQUEMENT à informer la grille de tuning de Isaac. On ne l'utilise pas comme label dans le training.
  - **Mentionner explicitement dans le notebook** : "Nous lisons ce taux pour calibrer le paramètre `contamination`, mais nous n'utilisons jamais la colonne comme cible pendant l'entraînement."
- [ ] Identifier les outliers visuels (ex : Tool wear très élevés, Torque extrêmes)
- [ ] **Commit** : `feat: distribution analysis + correlation matrix`

### Justifications théoriques (Phase 2 — 30 min)
- [ ] Section markdown : "Pourquoi standardiser les features ?"
  - **LOF** : utilise des distances euclidiennes entre voisins. Si Rotational speed (~1500) écrase numériquement Air temperature (~300), la distance est dominée par une seule feature. Standardiser remet tout sur la même échelle.
  - **Elliptic Envelope** : repose sur la distance de Mahalanobis avec la matrice de covariance. Théoriquement invariante à l'échelle, mais en pratique le calcul de la covariance robuste (MCD) peut être numériquement instable sans scaling → on scale par sécurité.
  - **One-Class SVM** : le kernel RBF dépend de distances → idem LOF.
  - **Isolation Forest** : découpe sur des seuils axe par axe (split aléatoire). Aucune distance, aucune sensibilité à l'échelle. **Pas besoin de scaler** pour IF, mais pas non plus de raison de ne pas le faire si on garde un seul `X_processed` partagé. **Décision** : on scale pour homogénéité, en notant explicitement que IF n'en a pas besoin.
- [ ] Section markdown : "Comment traiter `Type` (M/L/H) pour les algos distance-based ?"
  - Option ordinale : suppose un ordre Light < Medium < High → mais est-ce vraiment ordonné ? Voir si la doc du dataset le suggère (les types correspondent à des qualités de produit).
  - Option one-hot : transforme en 3 colonnes binaires. Plus sûr, pas d'hypothèse d'ordre. Mais ajoute des dimensions.
  - **Décision recommandée** : one-hot. Justifier avec : "On évite d'imposer un ordre arbitraire qui biaiserait les distances euclidiennes."
- [ ] **Commit** : `docs: add scaling and encoding rationale`

### Module `preprocessing.py` (Phase 2 — 30 min)
- [ ] Implémenter :

```python
"""Data loading and preprocessing for AI4I 2020 anomaly detection."""

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]
LABEL_COLUMNS = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
ID_COLUMNS = ["UDI", "Product ID"]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the AI4I 2020 dataset.

    Args:
        path: Path to ai4i2020.csv.

    Returns:
        DataFrame with all columns.
    """
    return pd.read_csv(path)


def split_features_and_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix from the held-out label.

    The Machine failure column must not be used during training. We keep it
    aside for the final Part 4 evaluation only.

    Args:
        df: Raw dataset.

    Returns:
        (X, y) where X excludes IDs, labels and failure subtypes,
        and y is the Machine failure series.
    """
    y = df["Machine failure"].copy()
    drop_cols = ID_COLUMNS + LABEL_COLUMNS
    X = df.drop(columns=drop_cols)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build the sklearn preprocessing pipeline.

    Numeric features are standardized; Type is one-hot encoded.

    Returns:
        Fitted-ready ColumnTransformer.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
```

- [ ] **Commit** : `feat: implement preprocessing pipeline`

### Sync 1 (fin de Phase 1) — IMPORTANT
Tu envoies à Isaac :
- Le taux d'anomalies observé : `df["Machine failure"].mean()` → fourchette à tester pour `contamination` / `nu`
- Confirmation que `X_processed` aura N colonnes (5 numériques + 2 ou 3 one-hot pour Type, selon `drop="first"` ou pas)

### Phase 3 (avec l'équipe — 30 min)
- [ ] Aider à l'assemblage du `final_report.ipynb`
- [ ] Copier ta section EDA dans le notebook final
- [ ] Écrire la **section intro** du rapport (contexte business, dataset, démarche)

### Phase 4 (relecture — 30 min)
- [ ] Relire la section de Isaac (challenger les hyperparams)
- [ ] Relire la section de Diego (vérifier la cohérence avec ton EDA)
- [ ] **Commit final** : `docs: final EDA section in report`

---

## Tes commits cibles (≥ 6)

```
chore: init project structure
feat: load AI4I dataset and check schema
feat: distribution analysis + correlation matrix
docs: add scaling and encoding rationale
feat: implement preprocessing pipeline
docs: final EDA section in report
```

---

## Anti-patterns à éviter

- **Ne pas** committer `ai4i2020.csv` dans `.gitignore` (on veut le garder versionné, c'est petit, 522 Ko)
- **Ne pas** dropper `Machine failure` SANS la sauvegarder de côté — Diego en a besoin pour la Part 4
- **Ne pas** mélanger features et labels dans la même DataFrame en sortie — toujours `(X, y)`
- **Ne pas** oublier les sous-types de pannes (`TWF`, `HDF`, etc.) dans les colonnes à drop — c'est de la fuite de target

---

## Coup de pouce 20/20

Ce qui distingue un EDA "bon" d'un EDA "excellent" :
- **Conclure chaque sous-section par 1-2 phrases de takeaway** ("→ Cette corrélation forte suggère qu'on pourrait considérer une feature dérivée température différentielle")
- **Quantifier les observations** ("3.4 % d'anomalies", pas "un peu d'anomalies")
- **Lier ce que tu observes à ce que feront Isaac et Diego** ("→ La distribution non-gaussienne de Tool wear nous fait douter de l'hypothèse de Elliptic Envelope. À noter pour la critique du modèle.")
