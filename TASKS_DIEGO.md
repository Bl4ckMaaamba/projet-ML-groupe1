# Tâches — Diego

**Rôle** : Analyst & Business
**Parts énoncé couvertes** : Part 3 (Comparaison & Visualisations) + Part 4 (Conclusion managériale)
**Branche Git** : `diego/eval`

> **Important** : la Part 4 pèse le plus dans la note d'insight critique (40 % du total). C'est ta partie qui peut faire passer l'équipe de 16 à 20.

---

## Ce que tu produis

| Livrable                          | Chemin                          |
|-----------------------------------|---------------------------------|
| Module visualisations             | `src/visualization.py`          |
| Module évaluation et coût         | `src/evaluation.py`             |
| Notebook viz + analyse business   | `notebooks/03_evaluation.ipynb` |
| Sections "Visualisations", "Coût", "Recommandation" du rapport | dans `notebooks/final_report.ipynb` (phase 4) |

---

## Ce dont tu as besoin des autres

- **De Noah** (fin Phase 1) :
  - `X_processed` (matrice scalée + encodée)
  - La série `y_true = df["Machine failure"]` mise de côté pour la Part 4
  - Les noms de colonnes après preprocessing
- **De Isaac** (fin Phase 2) :
  - `models_dict` avec les 4 modèles entraînés
  - Les fichiers `outputs/results/preds_*.csv` avec les prédictions de chaque modèle

---

## Checklist détaillée

### Setup (Phase 0 — 15 min)
- [ ] Branche : `git checkout -b diego/eval`
- [ ] Créer `src/visualization.py`, `src/evaluation.py`, `notebooks/03_evaluation.ipynb`, dossier `outputs/figures/`
- [ ] **Premier commit** : `chore: scaffold viz and eval modules`

### Phase 1 (45 min) — Préparer pendant que Noah et Isaac bossent
- [ ] Préparer le squelette des fonctions de viz et de coût (sans données)
- [ ] Familiariser avec `sklearn.decomposition.PCA` et `sklearn.manifold.TSNE`
- [ ] Préparer la fonction de calcul de coût (testable sur des données factices)
- [ ] **Commit** : `feat: scaffold cost matrix utility`

### Phase 2 (1h30) — Quand Isaac livre les modèles

#### A. Visualisations 2D (~30 min)

##### PCA
- [ ] Implémenter `pca_2d(X)` qui retourne :
  - L'array 2D des projections
  - Le ratio de variance expliquée par les 2 premières composantes
- [ ] **Markdown** : "Les 2 premières CP capturent X % de la variance. Cela suggère que [bonne projection / projection partielle]."

##### t-SNE
- [ ] Implémenter `tsne_2d(X, perplexity=30)`
- [ ] Tester 2-3 valeurs de perplexity (`5`, `30`, `50`) → garder celle qui donne la séparation la plus claire
- [ ] Mentionner : t-SNE est non-linéaire et préserve les voisinages locaux, contrairement à PCA qui préserve les variances globales

##### Figure clé : 4 modèles côte à côte
- [ ] Faire un `plt.subplots(2, 2)` :
  - Top-left : Isolation Forest
  - Top-right : One-Class SVM
  - Bottom-left : LOF
  - Bottom-right : Elliptic Envelope
- [ ] Chaque subplot : scatter sur les coords PCA, points colorés par prédiction (-1 rouge / 1 bleu, alpha=0.4)
- [ ] Titre par subplot avec le nom du modèle ET le % d'anomalies prédites
- [ ] Sauvegarder en `outputs/figures/model_decisions_pca.png` (dpi=150)
- [ ] **Commit** : `feat: PCA and t-SNE projections + model decision overlays`

#### B. Deep Dive (~30 min) — GROS LEVIER POUR L'INSIGHT

C'est l'item 3.3 de l'énoncé : *"identify at least one observation flagged as anomalous by one model but not another. Inspect raw sensor values and explain mathematically or geometrically why this discrepancy occurs based on each model's assumptions."*

- [ ] Construire une matrice `predictions_df` avec une colonne par modèle :

```python
predictions_df = pd.DataFrame({
    "isolation_forest": preds_if,
    "ocsvm": preds_ocsvm,
    "lof": preds_lof,
    "elliptic": preds_ee,
})
predictions_df["disagreement"] = predictions_df.apply(
    lambda row: row.nunique() > 1, axis=1
)
```

- [ ] Sélectionner ~3-5 cas de désaccord intéressants (ex : LOF dit anomaly, IF dit normal)
- [ ] Pour chaque cas, afficher :
  - L'index dans le dataset
  - Les valeurs brutes des features (température, vitesse, couple, usure, type)
  - La prédiction de chaque modèle
- [ ] **Explication mathématique** (1 paragraphe par cas) :
  - **Si LOF flag mais pas IF** : "L'observation est dans une zone localement peu dense (peu de voisins proches → score LOF élevé), mais globalement dans un cluster — IF la trouve avec peu de splits car ses coordonnées tombent dans des intervalles fréquents."
  - **Si Elliptic Envelope flag mais pas LOF** : "L'observation est loin du centre de masse en distance de Mahalanobis, mais entourée de nombreux voisins similaires (cluster d'extrêmes). LOF ne la voit pas car la densité locale est élevée."
  - **Si IF flag mais pas OC-SVM** : "IF isole rapidement le point sur des seuils alignés sur les axes (ex : Tool wear très haut). OC-SVM avec kernel RBF crée une frontière courbe qui peut englober ce point si ses voisins kernelisés sont denses."
- [ ] **Commit** : `feat: deep dive on disagreement points with mathematical justification`

#### C. Module `evaluation.py` (~30 min)

- [ ] Implémenter :

```python
"""Evaluation utilities for anomaly detection models.

Anomaly convention: -1 = anomaly, 1 = inlier (sklearn standard).
Ground truth Machine failure: 1 = failure, 0 = normal.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

FP_COST = 500
FN_COST = 15_000


def to_binary(y_pred_sklearn: np.ndarray) -> np.ndarray:
    """Convert sklearn anomaly labels (-1/1) to binary (1=anomaly, 0=normal).

    Args:
        y_pred_sklearn: Array of -1 (anomaly) and 1 (inlier).

    Returns:
        Array of 0 (normal) and 1 (anomaly).
    """
    return (y_pred_sklearn == -1).astype(int)


def confusion_counts(y_true: np.ndarray, y_pred_binary: np.ndarray) -> dict[str, int]:
    """Compute TP, FP, FN, TN.

    Args:
        y_true: Ground truth binary labels (1 = failure).
        y_pred_binary: Model predictions in binary format (1 = anomaly).

    Returns:
        Dict with keys tp, fp, fn, tn.
    """
    tp = int(((y_pred_binary == 1) & (y_true == 1)).sum())
    fp = int(((y_pred_binary == 1) & (y_true == 0)).sum())
    fn = int(((y_pred_binary == 0) & (y_true == 1)).sum())
    tn = int(((y_pred_binary == 0) & (y_true == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def cost_score(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    fp_cost: float = FP_COST,
    fn_cost: float = FN_COST,
) -> dict[str, float]:
    """Compute total business cost.

    Args:
        y_true: Ground truth binary labels.
        y_pred_binary: Predicted binary labels.
        fp_cost: Cost per false positive (default €500).
        fn_cost: Cost per false negative (default €15 000).

    Returns:
        Dict with confusion counts, total cost, recall, precision.
    """
    counts = confusion_counts(y_true, y_pred_binary)
    total_cost = counts["fp"] * fp_cost + counts["fn"] * fn_cost
    recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
    precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
    return {
        **counts,
        "cost_eur": total_cost,
        "recall": recall,
        "precision": precision,
    }


def compare_models(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build a comparison table across models.

    Args:
        y_true: Ground truth binary labels.
        predictions: Dict mapping model name -> binary predictions.

    Returns:
        DataFrame with one row per model, sorted by ascending cost.
    """
    rows = [{"model": name, **cost_score(y_true, pred)} for name, pred in predictions.items()]
    return pd.DataFrame(rows).sort_values("cost_eur")
```

- [ ] **Commit** : `feat: cost matrix and model comparison utilities`

### Phase 3 (avec l'équipe — 30 min) — Part 4

#### "The Reveal" — comparer aux ground truths

- [ ] Importer `y_true` (la colonne `Machine failure` que Noah a mise de côté)
- [ ] Pour chaque modèle, convertir ses prédictions en binaire avec `to_binary()`
- [ ] Calculer `compare_models(y_true, predictions)` → tableau comparatif

- [ ] Afficher proprement :

```
| Model              | TP | FP | FN | TN   | Cost (€)  | Recall | Precision |
| ------------------ | -- | -- | -- | ---- | --------- | ------ | --------- |
| Isolation Forest   |    |    |    |      |           |        |           |
| One-Class SVM      |    |    |    |      |           |        |           |
| LOF                |    |    |    |      |           |        |           |
| Elliptic Envelope  |    |    |    |      |           |        |           |
```

- [ ] **Commit** : `feat: reveal ground truth and compute model costs`

#### Sensitivity analysis sur le coût (LE LEVIER 20/20)

- [ ] Pour le **meilleur modèle** identifié, faire varier `contamination` (ou `nu`) ∈ `[0.01, 0.02, 0.03, 0.034, 0.05, 0.07, 0.10]`
- [ ] Pour chaque valeur, ré-entraîner et calculer le coût total
- [ ] **Plot** : courbe coût total = f(contamination) avec une ligne verticale au minimum
- [ ] **Markdown** : "Le minimum de coût est atteint à contamination=X, soit Y € de pertes annuelles. À droite, on rate des pannes (FN domine). À gauche, on bombarde la maintenance d'alertes inutiles (FP domine)."
- [ ] **Commit** : `feat: sensitivity analysis on contamination`

#### Recommandation business finale

- [ ] Section markdown courte (≤ 1 page) :
  - **TL;DR exécutif** : 2 phrases. "Nous recommandons le modèle X avec contamination=Y. Pertes attendues : €Z/an, soit -W % vs. statu quo."
  - **Pourquoi ce modèle** : 1-2 phrases liant le choix aux propriétés des données (cf. l'EDA de Noah)
  - **Trade-off accepté** : "Nous tolérons N FP/an (alertes inutiles) en échange de zéro panne non détectée"
  - **Mise en œuvre** : 2-3 phrases. "Déployer en mode shadow pendant 1 mois, comparer aux interventions des techniciens, ajuster contamination si besoin."
  - **Limites** : 1-2 phrases. "Modèle non supervisé : la définition d'anomalie peut dériver dans le temps. Re-entraînement trimestriel recommandé."

- [ ] **Commit** : `docs: managerial recommendation and business strategy`

### Phase 4 (assemblage final — 30 min)
- [ ] Copier tes sections dans `notebooks/final_report.ipynb`
- [ ] Vérifier que la narrative tient debout : EDA (Noah) → Modeling (Isaac) → Visualisation (toi) → Conclusion (toi)
- [ ] Exporter toutes les figures en PNG dans `outputs/figures/`
- [ ] **Commit final** : `docs: final evaluation and business sections`

---

## Tes commits cibles (≥ 6)

```
chore: scaffold viz and eval modules
feat: scaffold cost matrix utility
feat: PCA and t-SNE projections + model decision overlays
feat: deep dive on disagreement points with mathematical justification
feat: cost matrix and model comparison utilities
feat: reveal ground truth and compute model costs
feat: sensitivity analysis on contamination
docs: managerial recommendation and business strategy
docs: final evaluation and business sections
```

---

## Anti-patterns à éviter

- **Ne pas** confondre conventions sklearn (`-1` = anomaly) et conventions de la target (`1` = failure). Toujours convertir avec `to_binary()` avant de comparer à `Machine failure`.
- **Ne pas** présenter le coût sans le tableau des confusions. Le coût brut sans contexte est moins parlant.
- **Ne pas** conclure "Modèle X est le meilleur" sans une phrase business derrière. Le grader cherche du raisonnement, pas un classement.
- **Ne pas** oublier la sensitivity analysis : c'est ce qui sépare un 16 d'un 20.
- **Ne pas** commit les outputs des notebooks (utiliser nbstripout) — ça pollue le diff Git.

---

## Coup de pouce 20/20

Ce qui distingue une analyse "correcte" d'une analyse "excellente" :
- **Faire parler les chiffres** : transformer "coût = 234 500 €" en "coût annualisé = 234 500 €, soit l'équivalent de 2.3 heures de downtime évitées par mois"
- **Le deep dive doit citer les algos** : ne pas juste dire "ce point est différent". Dire "LOF compare à 35 voisins ; ici, le voisinage est étroit donc le ratio de densité est élevé. Pour IF, ce point se sépare en 4 splits seulement, ce qui le classe comme anomalie."
- **Anticiper la question du correcteur** : "Et si le coût d'un FN était de €30 000 ?" → faire la sensitivity, montrer que la reco bouge ou pas.
- **Le verdict final doit être actionnable** : un VP Operations doit pouvoir lire ta conclusion, comprendre, et signer un budget.
