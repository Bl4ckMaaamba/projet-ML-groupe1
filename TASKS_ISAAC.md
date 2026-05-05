# Tâches — Isaac

**Rôle** : ML Engineer
**Parts énoncé couvertes** : Part 2 (Modeling & Hyperparameter Tuning)
**Branche Git** : `isaac/models`

---

## Ce que tu produis

| Livrable                          | Chemin                          |
|-----------------------------------|---------------------------------|
| Module avec les 4 modèles         | `src/models.py`                 |
| Notebook d'entraînement et tuning | `notebooks/02_models.ipynb`     |
| Section "Modeling" du rapport     | dans `notebooks/final_report.ipynb` (phase 4) |
| Prédictions exportées             | `outputs/results/preds_*.csv`   |

---

## Ce dont tu as besoin des autres

- **De Noah** (fin Phase 1) :
  - `X_processed` : la matrice scalée + encodée (numpy array ou DataFrame, 10 000 lignes × ~7 colonnes)
  - Le **taux d'anomalies réel** dans le dataset (~3.4 %) → tu calibres tes grilles autour
  - La fonction `build_preprocessor()` à importer

## Ce que les autres attendent de toi

- **Diego** attend (fin Phase 2) :
  - Un `models_dict = {"isolation_forest": model, "ocsvm": model, "lof": model, "elliptic": model}`
  - Pour chaque modèle, un array `y_pred` (-1 / 1) sur tout `X_processed`, sauvegardé en CSV
  - Pour LOF : prédictions via `fit_predict` (LOF n'a pas de `predict` standard sauf si `novelty=True`)

---

## Checklist détaillée

### Setup (Phase 0 — 15 min)
- [ ] Branche : `git checkout -b isaac/models` (depuis `main` une fois le scaffolding poussé par Noah)
- [ ] Créer le squelette `src/models.py` et `notebooks/02_models.ipynb`
- [ ] **Premier commit** : `chore: scaffold models module`

### Phase 1 (45 min) — Préparer pendant que Noah finit son EDA
- [ ] Lire la doc sklearn des 4 modèles (rafraîchir les paramètres)
- [ ] Préparer la structure du module avec les 4 fonctions de training (vides pour l'instant)
- [ ] **Commit** : `feat: scaffold four anomaly detection trainers`

### Phase 2 — Implémentation et tuning (1h30)

#### 1. Isolation Forest (~20 min)
- [ ] Implémenter `train_isolation_forest`
- [ ] Tuner :
  - `contamination` : tester `[0.02, 0.034, 0.05, 0.10]` (fourchette autour du taux réel donné par Noah)
  - `n_estimators` : tester `[100, 200, 300]`
  - `max_samples` : `"auto"` (256) vs `0.5` (la moitié)
  - `random_state=42` toujours, pour la reproductibilité
- [ ] **Justification** (markdown dans le notebook) :
  - "On choisit `contamination=0.034` car cohérent avec le taux observé par l'EDA. Test des valeurs voisines pour vérifier la stabilité."
  - "n_estimators=200 retenu : compromis entre temps de calcul et stabilité du score."
- [ ] **Commit** : `feat: tune Isolation Forest`

#### 2. One-Class SVM (~25 min)
- [ ] Implémenter `train_one_class_svm`
- [ ] Tuner :
  - `nu` : équivalent à `contamination` ici. Tester `[0.02, 0.034, 0.05, 0.10]`
  - `kernel="rbf"` (défaut, le plus expressif pour des frontières non linéaires)
  - `gamma` : tester `"scale"` (défaut) vs `"auto"` vs valeurs fixes `[0.01, 0.1, 1.0]`
- [ ] **Attention** : OC-SVM est **lent** sur 10k points. Si tu galères, sous-échantillonne pour le tuning (3k points) puis ré-entraîne sur tout avec les meilleurs params.
- [ ] **Justification** :
  - "RBF kernel : on n'a pas de raison de penser à une frontière linéaire, et l'EDA montre des distributions non triviales."
  - "gamma='scale' = `1 / (n_features * X.var())` → adaptatif aux données scalées de Noah."
- [ ] **Commit** : `feat: implement One-Class SVM with tuned nu and gamma`

#### 3. Local Outlier Factor (~20 min)
- [ ] Implémenter `train_lof`
- [ ] Tuner :
  - `n_neighbors` : tester `[10, 20, 35, 50]`
    - Trop petit : sensible au bruit local
    - Trop grand : lisse les vraies anomalies, devient une moyenne globale
    - Règle empirique : ~√N → ici √10000 = 100, mais souvent on prend 20-50 en pratique
  - `contamination` : `[0.02, 0.034, 0.05, 0.10]`
  - `novelty=False` (on fait du fit_predict sur les données d'entraînement)
- [ ] **Important** : LOF avec `novelty=False` n'a pas de `predict()`, seulement `fit_predict()`. Si tu veux pouvoir prédire sur de nouveaux points, il faut `novelty=True`. Pour le projet, `fit_predict` sur `X_processed` suffit.
- [ ] **Justification** :
  - "n_neighbors=35 : compromis. À 20 trop de bruit local. À 50 on commence à perdre les anomalies les plus subtiles."
- [ ] **Commit** : `feat: tune LOF n_neighbors via stability check`

#### 4. Elliptic Envelope (~15 min)
- [ ] Implémenter `train_elliptic_envelope`
- [ ] Tuner :
  - `contamination` : `[0.02, 0.034, 0.05, 0.10]`
  - `support_fraction` : `None` (auto = `(n_samples + n_features + 1) / 2 / n_samples`) vs `0.75` vs `0.9`
- [ ] **CAVEAT théorique à mentionner** :
  - Elliptic Envelope suppose que les données sont **gaussiennes multivariées**.
  - Demande à Noah si ses histogrammes confirment cette hypothèse. Si non (skew, multimodal), c'est une **limite à mentionner explicitement** dans le rapport — ça montre la rigueur critique attendue dans la grille (40 % insight).
- [ ] **Justification** :
  - "Hypothèse gaussienne partiellement violée (cf. EDA, Tool wear non gaussien). On garde le modèle pour comparaison mais on s'attend à des performances dégradées sur les anomalies non-elliptiques."
- [ ] **Commit** : `feat: implement Elliptic Envelope with covariance robust estimation`

### Génération des prédictions (Phase 2 fin — 15 min)
- [ ] Pour chaque modèle, exporter les prédictions dans `outputs/results/preds_{name}.csv`
- [ ] Format suggéré : 1 colonne `prediction` (-1 = anomaly, 1 = normal), avec l'index de la ligne d'origine
- [ ] Construire le `models_dict` exporté pour Diego
- [ ] **Commit** : `feat: export predictions for downstream evaluation`

### Phase 3 (avec Diego — 30 min)
- [ ] Vérifier avec Diego que les prédictions sont au bon format
- [ ] Si Diego trouve un comportement bizarre sur un modèle (ex : 100 % anomalies), debugger ensemble

### Phase 4 (relecture — 30 min)
- [ ] Copier ta section "Modeling" dans `final_report.ipynb`
- [ ] Markdown bien structuré : un sous-chapitre par modèle, justification + résultats
- [ ] **Commit final** : `docs: final modeling section in report`

---

## Squelette de `models.py` (pour démarrer)

```python
"""Anomaly detection models for AI4I 2020 dataset.

All trainers accept a preprocessed feature matrix X (scaled + encoded)
and return a fitted estimator. Predictions follow sklearn convention:
-1 for anomaly, 1 for inlier.
"""

from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope


def train_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.034,
    n_estimators: int = 200,
    max_samples: str | float = "auto",
    random_state: int = 42,
) -> IsolationForest:
    """Train an Isolation Forest.

    Args:
        X: Preprocessed feature matrix.
        contamination: Expected proportion of anomalies.
        n_estimators: Number of trees in the forest.
        max_samples: Number of samples to draw to train each tree.
        random_state: Reproducibility seed.

    Returns:
        Fitted IsolationForest.
    """
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def train_one_class_svm(
    X: np.ndarray,
    nu: float = 0.034,
    kernel: str = "rbf",
    gamma: str | float = "scale",
) -> OneClassSVM:
    """Train a One-Class SVM.

    Args:
        X: Preprocessed feature matrix.
        nu: Upper bound on the fraction of training errors / lower bound
            on the fraction of support vectors. Plays a role similar to
            contamination.
        kernel: Kernel type.
        gamma: Kernel coefficient for rbf/poly/sigmoid.

    Returns:
        Fitted OneClassSVM.
    """
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X)
    return model


def train_lof(
    X: np.ndarray,
    n_neighbors: int = 35,
    contamination: float = 0.034,
) -> LocalOutlierFactor:
    """Train a Local Outlier Factor (in fit_predict mode).

    Args:
        X: Preprocessed feature matrix.
        n_neighbors: Size of the neighborhood for density estimation.
        contamination: Expected proportion of outliers.

    Returns:
        Fitted LocalOutlierFactor (use fit_predict to get labels).
    """
    return LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        n_jobs=-1,
    )


def train_elliptic_envelope(
    X: np.ndarray,
    contamination: float = 0.034,
    support_fraction: float | None = None,
    random_state: int = 42,
) -> EllipticEnvelope:
    """Train a robust covariance estimator (Elliptic Envelope).

    Assumes the inliers follow a multivariate Gaussian distribution.

    Args:
        X: Preprocessed feature matrix.
        contamination: Expected proportion of anomalies.
        support_fraction: Proportion of points used in MCD. None = auto.
        random_state: Reproducibility seed.

    Returns:
        Fitted EllipticEnvelope.
    """
    model = EllipticEnvelope(
        contamination=contamination,
        support_fraction=support_fraction,
        random_state=random_state,
    )
    model.fit(X)
    return model
```

---

## Tes commits cibles (≥ 6)

```
chore: scaffold models module
feat: scaffold four anomaly detection trainers
feat: tune Isolation Forest
feat: implement One-Class SVM with tuned nu and gamma
feat: tune LOF n_neighbors via stability check
feat: implement Elliptic Envelope with covariance robust estimation
feat: export predictions for downstream evaluation
docs: final modeling section in report
```

---

## Anti-patterns à éviter

- **Ne pas** garder les hyperparamètres par défaut. L'énoncé est explicite : "Do not keep default hyperparameters". Tu dois tuner et **justifier**.
- **Ne pas** confondre `LOF.fit_predict` et `LOF.predict` (le 2e n'existe que si `novelty=True`)
- **Ne pas** oublier que OC-SVM est lent sur 10k points → si trop long, sous-échantillonner pour le tuning
- **Ne pas** oublier `random_state=42` partout où c'est possible (reproductibilité = critère de code quality)
- **Ne pas** négliger la justification : un hyperparam sans markdown qui explique = -2 points

---

## Coup de pouce 20/20

Ce qui distingue un tuning "fait" d'un tuning "rigoureux" :
- **Mesurer la stabilité** : pour chaque modèle, faire un bootstrap (5 sous-échantillons à 80 %), regarder la variance du % d'anomalies prédites. Un modèle stable = un bon modèle.
- **Comparer 2-3 configs par modèle** dans un petit tableau (pas juste annoncer la meilleure). Ça montre qu'on a vraiment exploré.
- **Mentionner les limites théoriques** de chaque modèle (Elliptic Envelope ↔ gaussianité, OC-SVM ↔ lent, LOF ↔ choix de k délicat). Le rapport doit montrer que tu connais les algos, pas juste que tu les as appelés.
- **Sauvegarder un seed partout** : sinon Diego n'aura pas les mêmes résultats que toi.
