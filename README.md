# Churnlab 2026 - Telco Customer Churn

Projet fil rouge du cours ESILV Machine Learning Operations. L'objectif est de construire un système de prédiction de churn pour un opérateur télécom, puis de suivre son entraînement, le packager, le déployer et le monitorer. Le TP01 pose les fondations : environnement déclaré, source de données identifiée, script d'inspection exécutable, Git revu en PR.

## Question métier

Un opérateur télécom veut identifier les clients qui risquent de résilier leur abonnement. La variable cible est la colonne `Churn` du dataset (valeurs `Yes` / `No`). Le modèle doit prédire cette étiquette à partir des caractéristiques du client (ancienneté, type de contrat, moyen de paiement, etc.).

## Setup

Version Python requise : **3.11** (3.12 également supporté par le lock fourni).
Version uv de référence : **0.12.23**.

```sh
uv sync --locked
uv run python -c "import churnlab; print(churnlab.__version__)"
```

`uv sync --locked` installe l'environnement en exigeant que `uv.lock` reste cohérent avec `pyproject.toml`. Si le lock doit changer, c'est un signal à investiguer, pas à ignorer.

## Dataset

Source : [IBM Telco Customer Churn, commit pinné](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv).

Empreinte SHA-256 attendue pour les octets fournis :

```
16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91
```

Le snapshot contient 7 043 lignes et 21 colonnes. Télécharger (ou vérifier) le fichier et contrôler son hash :

```sh
uv run python scripts/download_data.py
uv run python scripts/inspect_telco.py
```

Le rapport produit atterrit dans `reports/tp01_data_overview.json`. Une copie non-sensible est conservée sous `evidence/tp01/data-overview.json` pour la soumission (le dossier `reports/` est gitignoré).

### Nature du snapshot

Il s'agit d'une **photographie clients à un instant donné**, pas d'un journal d'événements horodatés ni d'une expérience randomisée de rétention. On ne peut pas en déduire ce qui a causé le départ, ni comparer un groupe traité à un groupe témoin. Les prédictions produites doivent être lues comme des scores associés à l'état courant du client, pas comme des effets causaux d'une action commerciale.

## Structure du projet

| Dossier | Rôle |
|---|---|
| `src/churnlab/` | Code importable : chargement de données, entraînement, service |
| `scripts/` | Commandes exécutables du workflow (download, inspection, enregistrement env) |
| `tests/` | Vérifications comportementales |
| `data/raw/` | Snapshot d'entrée original + métadonnées de source |
| `data/processed/` | Splits générés (TP suivants) |
| `artifacts/` | Modèles entraînés (TP suivants) |
| `reports/` | Rapports générés (gitignoré) |
| `k8s/`, `pipelines/` | Déploiement et orchestration (TP suivants) |

## Reproductibilité : ce qui est fait, ce qui ne l'est pas

Le lock + hash du CSV garantissent que deux personnes partent des mêmes paquets et des mêmes octets de données. Ils ne garantissent **pas** l'identité OS/CPU, la qualité des labels, ou la reproductibilité d'un entraînement ML (seeds, non-déterminisme GPU, ordres de données). Ces points sont traités dans les TP suivants.
