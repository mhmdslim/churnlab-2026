# TP02 - Réponses

**Auteur :** Mohammad Slim
**Date :** 2026-10-23
**Dépôt :** `https://github.com/mhmdslim/churnlab-2026`

---

## Milestone 1 - Premier modèle

Modèle : régression logistique sur `tenure` + `MonthlyCharges` uniquement.

| Métrique | Logistic Regression | Baseline (majority) |
|---|---:|---:|
| Accuracy | 0.7828 | 0.7346 |

Matrice de confusion (LR, labels [0, 1]) :

|  | Prédit 0 | Prédit 1 |
|---|---:|---:|
| Réel 0 | 950 | 85 |
| Réel 1 | 221 | 153 |

Classe positive : `Churn = Yes` (encodé 1).

**Questions :**

- Un système qui prédit toujours "No" obtient 73.5% d'accuracy. C'est le score de la baseline.
- Oui, une accuracy élevée peut coexister avec 0 churners identifiés : la baseline en est la preuve (recall = 0).
- Seules les données de validation peuvent influencer le choix du modèle. Le test set reste réservé.

---

## Milestone 2 - Pipeline avec preprocessing

Features utilisées (6 au total) :

| Type | Colonnes |
|---|---|
| Numériques | `tenure`, `MonthlyCharges`, `TotalCharges` |
| Catégorielles | `Contract`, `InternetService`, `PaymentMethod` |

Preprocessing :

- **Numériques** : `SimpleImputer(strategy="median")` puis `StandardScaler`. Les 11 TotalCharges blancs sont convertis en NaN avant d'entrer dans le pipeline, via `pd.to_numeric` après strip. La stratégie `errors="raise"` est utilisée pour détecter toute valeur non-numérique qui ne serait pas un simple blanc.
- **Catégorielles** : `SimpleImputer(strategy="constant", fill_value="missing")` puis `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`. `handle_unknown="ignore"` encode en vecteur zéro toute catégorie absente du training. Ca évite un crash en inférence, mais ne garantit pas que la prédiction soit fiable pour cette catégorie.

Un seul appel à `pipe.fit(X_train, y_train)` entraîne tout. La prédiction accepte les colonnes originales non encodées.

**Questions :**

- `fit` calcule les statistiques (médiane, moyenne, écart-type, catégories). `transform` applique ces statistiques pour produire la matrice numérique. `predict` appelle `transform` puis passe le résultat au classifieur.
- Fitter un scaler sur les données de validation pollue les statistiques de centrage/réduction avec des informations hors-train. En production, on n'a pas accès aux futures données au moment du fit.
- Non, un Pipeline ne prévient pas le leakage si on lui passe les mauvaises données de training (par exemple, tout le dataset au lieu du train seul).
- `Pipeline` ordonne les étapes séquentiellement (imputer -> scaler -> classifier). `ColumnTransformer` route les colonnes vers des branches parallèles (num vs cat), puis concatène les sorties.

---

## Milestone 3 - Comparaison de modèles

Métrique de sélection choisie **avant** la comparaison : **average precision** sur validation. Justification : l'accuracy est trompeuse avec des classes déséquilibrées (73.5% No). L'average precision résume la courbe precision-recall et est plus informative pour la classe minoritaire.

| Modèle | Accuracy | Precision | Recall | F1 | Avg Precision | Temps (s) |
|---|---:|---:|---:|---:|---:|---:|
| baseline_majority | 0.7346 | 0.0000 | 0.0000 | 0.0000 | 0.2654 | 0.1 |
| logistic_regression | 0.8041 | 0.6439 | 0.5455 | 0.5893 | 0.6475 | 0.2 |
| decision_tree | 0.8006 | 0.5764 | 0.6631 | 0.6156 | 0.6033 | 0.0 |
| random_forest | 0.8027 | 0.6506 | 0.5267 | 0.5826 | 0.6579 | 0.5 |
| gradient_boosting | 0.7956 | 0.6393 | 0.4920 | 0.5556 | 0.6336 | 1.0 |

**Sélection : random_forest** (avg_precision = 0.6579, le plus élevé).

Le passage de 2 à 6 features améliore la plupart des métriques (accuracy LR : 0.78 -> 0.80). On ne peut pas attribuer cette amélioration au seul algorithme, puisque les features ont aussi changé.

**Questions :**

- Changer le split pour chaque classifieur rendrait la comparaison invalide : les différences de score pourraient venir du split, pas du modèle.
- Le modèle avec la meilleure accuracy n'est pas forcément le meilleur : la baseline a 73.5% d'accuracy mais n'identifie personne.
- Si deux candidats sont proches, le plus simple est préférable : moins de risque d'overfitting, plus rapide, plus facile à expliquer et à maintenir.
- Une probabilité prédite de 0.8 n'est pas automatiquement un risque calibré à 80%. La calibration est une propriété distincte de la discrimination.

---

## Milestone 4 - Artefact sauvegardé

- Artefact : `artifacts/pipeline.joblib` (RandomForestClassifier + preprocessing complet)
- Metadata : `evidence/tp02/training_metadata.json`
- Customer splits : `evidence/tp02/customer_splits.json`

Vérification reload :
- Différence max de probabilité entre original et reload : **0.0**
- Prédictions de classe : **identiques**
- Aucun appel à `fit()` dans le processus de reload

Le pipeline consomme un DataFrame avec les 6 colonnes documentées, dans les types originaux. Le parsing CSV et la validation des types restent la responsabilité du code appelant.

**Questions :**

- Sauvegarder uniquement le classifieur perd les transformations apprises (médiane d'imputation, moyenne/écart-type du scaler, catégories de l'encoder). En inférence, il faudrait les refaire à la main, avec le risque de divergence.
- L'imputer retient les médianes du training. L'encoder retient les catégories vues au training. Le scaler retient moyenne et écart-type du training.
- Deux fichiers `model.joblib` peuvent contenir des modèles différents (versions, hyperparamètres, données d'entraînement). Le nom du fichier ne garantit rien, il faut la metadata associée.
- Un sklearn Pipeline est un objet Python qui chaîne des étapes de preprocessing + modèle en mémoire. Un workflow DAG (Kubeflow Pipelines, Airflow) orchestre des conteneurs ou des tâches distribuées, avec scheduling, logging, retry. Le premier vit dans un script, le second dans une infrastructure d'exécution.

---

## Milestone 5 - Vérifications ciblées

| Check | Résultat |
|---|---|
| Pas d'overlap de customerID entre train/valid/test | OK |
| Médianes d'imputation = médianes du train uniquement | OK (tenure=29.00, MonthlyCharges=70.55, TotalCharges=1373.62) |
| predict() ne change pas les statistiques apprises | OK |
| NaN numérique + catégorie inconnue : traités sans erreur | OK (prédit classe 0) |
| Colonne manquante : erreur visible | OK (ValueError) |
| Reload : prédictions identiques sans fit() | OK |

---

## Checklist

- [x] Modèle simple + baseline avec split stratifié reproductible
- [x] Pipeline sklearn complet (preprocessing + classifieur)
- [x] Comparaison de 3 classifieurs + baseline, même split/features
- [x] Sélection justifiée sur average precision (validation)
- [x] Artefact sauvegardé, reload vérifié, metadata enregistrée
- [x] Vérifications ciblées passées
- [x] Code exécutable depuis le terminal (`uv run python scripts/train.py`)
