# Plan de travail — Projet ML III (Anomaly Detection)

**Équipe** : Noah, Isaac, Diego
**Objectif** : 20/20
**Dataset** : `ai4i2020.csv` (10 000 obs, 14 cols)
**Rendu** : 1 notebook Jupyter final + modules `.py`

---

## 1. Stratégie pour viser 20/20

La grille de notation pondère :

| Critère                       | Poids | Levier principal                                                                 |
|-------------------------------|-------|----------------------------------------------------------------------------------|
| Code Quality & ML Pipeline    | 30 %  | Code modulaire (`.py` réutilisables), type hints, docstrings, sklearn `Pipeline` |
| Rigueur analytique            | 30 %  | EDA poussée, chaque hyperparam justifié par un constat de l'EDA                  |
| **Insight critique**          | 40 %  | Deep dive sur le désaccord entre modèles + analyse coût rigoureuse + reco claire |

**Pour aller chercher la note max** (au-delà du minimum demandé) :
- Cross-validation de la stabilité du score (silhouette, % d'anomalies stable d'un fold à l'autre)
- Permutation importance pour expliquer *pourquoi* un point est flaggé
- Analyse de sensibilité sur le coût (et si FN = €30k ? si contamination = 5 % ?)
- Tableau comparatif propre (pas juste des `print`) : confusion matrix par modèle, coût total, F1, recall sur la classe positive
- Storytelling business : présenter la conclusion comme une recommandation stratégique au COMEX, pas un dump de chiffres

---

## 2. Structure du repo

```
ML-projet-cours/
├── ai4i2020.csv
├── PLAN.md                       ← ce fichier
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          ← Noah
│   ├── models.py                 ← Isaac
│   ├── evaluation.py             ← Diego
│   └── visualization.py          ← Diego
├── notebooks/
│   ├── 01_eda.ipynb              ← Noah (sandbox)
│   ├── 02_models.ipynb           ← Isaac (sandbox)
│   ├── 03_evaluation.ipynb       ← Diego (sandbox)
│   └── final_report.ipynb        ← assemblage en phase finale (tous)
└── outputs/
    ├── figures/                  ← png exportés
    └── results/                  ← csv des prédictions par modèle
```

**Pourquoi 3 notebooks sandbox + 1 final ?**
Les fichiers `.ipynb` génèrent des conflits Git pénibles (cell IDs, outputs). Si chacun bosse dans son propre notebook, zéro conflit. Le `final_report.ipynb` n'est assemblé qu'à la phase 3, à 6 mains.

---

## 3. Workflow Git

### Branches
- `main` — version stable, on y merge en fin de phase
- `noah/eda` — branche perso Noah
- `isaac/models` — branche perso Isaac
- `diego/eval` — branche perso Diego

### Règles
- Chacun commit **uniquement sur sa branche** pendant les phases 1 et 2
- Commits **fréquents** (toutes les ~30 min) avec messages clairs (`feat:`, `fix:`, `docs:`, `refactor:`)
- En phase 3, on merge tout sur `main` ensemble
- Avant tout commit de notebook : `nbstripout` pour virer les outputs (évite les conflits inutiles)

### Exemple de séquence de commits par personne
Pour montrer le travail réparti, viser **6-10 commits chacun minimum** :

**Noah** :
```
feat: load AI4I dataset and check schema
feat: distribution analysis on numeric features
feat: correlation matrix + heatmap
fix: handle Type column encoding (one-hot)
feat: implement preprocessing pipeline (scaler + encoder)
docs: add EDA narrative to notebook
```

**Isaac** :
```
feat: scaffold models module
feat: tune Isolation Forest with custom contamination grid
feat: implement One-Class SVM with RBF kernel
feat: tune LOF n_neighbors via stability score
feat: implement Elliptic Envelope with support_fraction
docs: justify hyperparam choices in markdown
```

**Diego** :
```
feat: PCA 2D projection for visualization
feat: t-SNE projection (perplexity tuning)
feat: overlay model decisions on PCA scatter
feat: cost matrix and confusion matrix utilities
feat: deep dive on disagreement points
feat: managerial recommendation section
```

---

## 4. Répartition détaillée

### NOAH — Data Engineer (Part 1)

**Livrables**
- `src/preprocessing.py` — module avec :
  - `load_data(path: str) -> pd.DataFrame`
  - `drop_label_columns(df) -> tuple[pd.DataFrame, pd.Series]` (sépare X et `Machine failure` + sous-types pannes — la target n'est touchée qu'en Part 4)
  - `build_preprocessor() -> ColumnTransformer` avec `StandardScaler` sur numériques + `OneHotEncoder` sur `Type`
- `notebooks/01_eda.ipynb` — exploration documentée

**Checklist EDA (≥ 30 % de la note)**
- [ ] `df.info()`, `df.describe()`, missing values
- [ ] Distributions de chaque feature numérique (histogrammes + boxplots) → identifier les skews
- [ ] Distribution de `Type` (M / L / H — quelle proportion ?)
- [ ] Matrice de corrélation + heatmap → repérer les redondances (Air temp vs Process temp ?)
- [ ] Taux réel de `Machine failure` dans le dataset → **clé** pour fixer `contamination` (regarder, puis "oublier" pour le training)
- [ ] Identifier les outliers visuels qui pourraient correspondre à des pannes
- [ ] Décision argumentée : pourquoi scaler pour LOF / Elliptic Envelope mais pas pour Isolation Forest (réponse : LOF/EE utilisent des distances euclidiennes / Mahalanobis sensibles à l'échelle, IF découpe sur des seuils axe par axe donc invariant à l'échelle)
- [ ] Décision argumentée : encoding de `Type` (one-hot ? ordinal ? — justifier)

**Narrative attendue dans le notebook**
- Markdown clair entre chaque cellule de code
- Conclure chaque sous-section par 1-2 phrases de takeaway

---

### ISAAC — ML Engineer (Part 2)

**Livrables**
- `src/models.py` — module avec :
  - `train_isolation_forest(X, contamination, **kwargs) -> IsolationForest`
  - `train_one_class_svm(X, nu, kernel, gamma) -> OneClassSVM`
  - `train_lof(X, n_neighbors, contamination) -> LocalOutlierFactor`
  - `train_elliptic_envelope(X, contamination, support_fraction) -> EllipticEnvelope`
  - `predict_anomalies(model, X) -> np.ndarray` (output -1 / 1)
- `notebooks/02_models.ipynb` — entraînement + tuning documenté

**Checklist tuning (≥ 30 % de la note — JUSTIFIER chaque choix)**
- [ ] Lire le `contamination` cible depuis l'EDA de Noah (~3.4 % dans AI4I si on regarde la vraie target — utiliser une fourchette autour : tester `[0.02, 0.034, 0.05, 0.10]`)
- [ ] **Isolation Forest** : tuner `n_estimators` (100 → 300), `max_samples`, `contamination`. Comparer 2-3 configs.
- [ ] **One-Class SVM** : tuner `nu` (équivalent `contamination` ici), `kernel` (`rbf` par défaut), `gamma` (`scale` vs `auto` vs valeur fixe). Sensible au scaling — vérifier que les données sont scalées.
- [ ] **LOF** : tuner `n_neighbors` (10, 20, 35, 50). Trade-off : trop petit = bruit local, trop grand = lisse les vraies anomalies.
- [ ] **Elliptic Envelope** : tuner `support_fraction` (None par défaut = auto, sinon 0.6-0.9). Hypothèse forte : données ~gaussiennes — VÉRIFIER avec l'EDA de Noah, et si non gaussien, le mentionner comme limite.
- [ ] Pour chaque modèle, sauvegarder les prédictions dans `outputs/results/preds_{model_name}.csv`

**Tip pour le tuning sans label** : utiliser la stabilité du % d'anomalies prédites sur des sous-échantillons (bootstrap). Un modèle stable = bon signe.

**Output attendu** : un dictionnaire `models_dict = {"isolation_forest": ..., "ocsvm": ..., "lof": ..., "elliptic": ...}` exporté pour Diego.

---

### DIEGO — Analyst & Business (Parts 3 + 4)

**Livrables**
- `src/visualization.py` — module avec :
  - `pca_2d(X) -> np.ndarray`
  - `tsne_2d(X, perplexity=30) -> np.ndarray`
  - `plot_model_decisions(X_2d, y_pred, ax, title)` (4 subplots côte à côte)
- `src/evaluation.py` — module avec :
  - `cost_score(y_true, y_pred, fp_cost=500, fn_cost=15000) -> dict` (retourne TP/FP/FN/TN + coût total)
  - `compare_models(models_dict, X, y_true) -> pd.DataFrame` (un tableau récap)
- `notebooks/03_evaluation.ipynb` — viz + analyse business

**Part 3 — Checklist viz**
- [ ] PCA 2D : variance expliquée par les 2 premières composantes (afficher le %)
- [ ] t-SNE 2D : essayer 2-3 valeurs de `perplexity` (5, 30, 50)
- [ ] **Figure clé** : 2x2 grid des 4 modèles, points colorés par prédiction (-1 rouge / 1 bleu) sur la projection PCA
- [ ] **Deep dive** (gros enjeu d'insight) :
  - Trouver une obs flaggée par 1 seul modèle (ex : LOF dit anomaly, Isolation Forest dit normal)
  - Inspecter les valeurs brutes (`X.iloc[idx]`)
  - Expliquer **mathématiquement** pourquoi : ex "LOF compare la densité locale, l'obs est isolée dans son voisinage proche → score élevé. Mais Isolation Forest la trouve facilement avec peu de splits car elle est dans une zone dense globalement."

**Part 4 — Analyse coût (LE GROS LEVIER POUR LE 20/20)**
- [ ] Reintroduire `Machine failure` (et seulement à ce moment)
- [ ] Pour chaque modèle, calculer :
  - Confusion matrix : TP, FP, FN, TN
  - Coût total = FP × 500 + FN × 15 000
- [ ] Tableau récap propre (pas juste des prints) :
  ```
  | Model           | TP | FP | FN | TN  | Cost (€) | Recall | Precision |
  | --------------- | -- | -- | -- | --- | -------- | ------ | --------- |
  | Isolation Forest| ...| ...| ...| ... | ...      | ...    | ...       |
  | ...             |    |    |    |     |          |        |           |
  ```
- [ ] Sensitivity analysis : faire varier `contamination` ∈ [0.01, 0.02, 0.03, 0.05, 0.10] sur le meilleur modèle, plotter le coût total → trouver le minimum
- [ ] **Recommandation finale** : 1 paragraphe orienté COMEX
  - "Le modèle X minimise le coût annualisé à €Y. Recommandation : déployer avec contamination=Z, en pilotant les alertes via [tableau de bord/Slack]. Trade-off accepté : N faux positifs/an pour zéro panne non détectée."

---

## 5. Phasage temporel

| Phase | Durée  | Qui                | Quoi                                                                 |
|-------|--------|--------------------|----------------------------------------------------------------------|
| 0     | 15 min | Tous               | Setup repo (git init, structure, requirements.txt), kickoff alignment |
| 1     | 45 min | Tous en parallèle  | Noah → EDA brouillon. Isaac → scaffolding `models.py`. Diego → scaffolding viz/eval. **Premier commit chacun.** |
| 2     | 1h30   | Tous en parallèle  | Noah finalise Part 1. Isaac entraîne et tune les 4 modèles. Diego prépare viz et fonctions de coût. |
| 3     | 30 min | Tous ensemble      | Noah passe `X_processed` à Isaac. Isaac passe `models_dict` à Diego. Diego produit viz + analyse. |
| 4     | 30 min | Tous ensemble      | Assemblage du `final_report.ipynb` : copier-coller des sections + transitions narratives + relecture. |

---

## 6. Points de synchro (ne pas se marcher dessus)

**Sync 1 — fin de Phase 1** (45 min)
- Noah présente le `contamination` réel observé → Isaac s'en sert pour ses grilles de tuning
- Diego confirme la signature de `predict_anomalies` qu'il attend de Isaac

**Sync 2 — fin de Phase 2** (2h15)
- Isaac livre `models_dict` + prédictions sauvegardées
- Diego peut commencer son éval avec les vraies prédictions
- Noah commence à rédiger l'intro du notebook final

**Sync 3 — assemblage** (3h)
- Tour de table : qu'est-ce qui manque, qu'est-ce qui mérite d'être renforcé ?
- Relecture croisée : chacun lit la section des deux autres et challenge

---

## 7. Anti-patterns à éviter

- Ne pas commiter le CSV en double, ni les `__pycache__/`, ni les `.ipynb_checkpoints/` → utiliser `.gitignore`
- Ne pas garder les hyperparamètres par défaut (énoncé explicite : "Do not keep default hyperparameters")
- Ne pas regarder `Machine failure` pendant les Parts 1-3 (sauf pour l'EDA de Noah qui a le droit de "noter" le taux réel pour calibrer les hyperparams, mais le mentionner explicitement)
- Ne pas conclure sans recommandation business chiffrée
- Ne pas oublier la justification mathématique dans le deep dive (Part 3.3) — c'est typiquement ce qui distingue un 16 d'un 20

---

## 8. Checklist finale avant rendu

- [ ] Notebook `final_report.ipynb` exécutable de bout en bout (`Restart & Run All`)
- [ ] Tous les modules `.py` ont docstrings + type hints
- [ ] Toutes les figures sont titrées et légendées
- [ ] Markdown narratif entre chaque section
- [ ] Conclusion business ≤ 1 page, claire, chiffrée
- [ ] `requirements.txt` à jour
- [ ] README.md avec instructions de reproduction
- [ ] Au moins 6 commits par personne, messages explicites
- [ ] Tag git `v1.0-rendu` sur le commit final
