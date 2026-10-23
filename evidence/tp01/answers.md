# TP01 - Réponses

**Auteur :** Mohammad Slim
**Date :** 2026-10-23
**Dépôt :** `https://github.com/mhmdslim/churnlab-2026`
**PR URL :** `https://github.com/mhmdslim/churnlab-2026/pull/1`
**Commit hash du merge :** `197d60d`

---

## Partie A - Comprendre le projet et les données

### Question A1 - Pourquoi le CSV original, un modèle entraîné et le code source ont-ils des rôles différents dans le repo ?

Ces trois objets n'ont ni la même origine, ni le même cycle de vie, ni la même capacité à être reproduits.

Le **code source** est écrit à la main, se change à chaque commit et constitue la seule chose dont on a vraiment besoin pour reconstruire tout le reste. Il est petit, diffable, revu en PR. Il appartient à Git.

Le **CSV original** est une donnée d'entrée externe. On ne la modifie jamais, on la vérifie par son hash. L'archiver dans Git pollue l'historique avec des méga-octets qui ne changent pas et dont la provenance se documente mieux par une URL + un SHA-256. Dans ce projet, le CSV est gitignoré et seules la source et l'empreinte sont versionnées.

Le **modèle entraîné** est un artefact produit par une exécution précise du code sur les données. Il est volumineux, binaire, régénérable, et doit pouvoir être retracé jusqu'à son commit de code + sa version de données + sa configuration. Il appartient à un registre de modèles ou à un store d'artefacts, pas à Git.

Mélanger les trois dans un commit casse la revue de code, gonfle le repo, et empêche de dire ce qui a vraiment changé.

### Question A2 - Pourquoi inclure `Churn` parmi les features invaliderait-il la tâche de prédiction ?

`Churn` est l'étiquette que le modèle doit prédire. Si on la lui fournit en entrée, il apprendrait la fonction identité : « recopie la colonne Churn ». L'accuracy serait de 100% sur le train et sur le test. On parle de **leakage total de la cible**.

En production, cette variable n'existe pas au moment où on veut prédire le churn : si on la connaissait déjà, on n'aurait pas besoin de modèle. Le modèle qu'on obtiendrait serait inutilisable.

C'est la forme la plus brutale du data leakage, mais la même logique vaut pour toute variable qui n'est connue qu'après l'événement à prédire.

---

## Partie B - Environnement Python

### Tâche B2 - pyproject.toml, uv.lock, uv tree

1. **Contrainte Python** dans `pyproject.toml` : `requires-python = ">=3.11,<3.13"`. **Dépendance directe** retenue : `pandas>=2.2,<2.3`.
2. **Dans uv.lock**, pandas apparaît avec une version résolue précise (ex. `pandas==2.2.3`), son hash sdist/wheel et la liste de ses dépendances.
3. **Dépendance transitive** (via `uv tree`) : `pytz` (requis par pandas), ou `tzdata`, ou `python-dateutil`. Aucune de ces trois ne figure dans `pyproject.toml`.
4. **Les trois rôles :**
   - `pyproject.toml` : **ce qu'on veut**. Métadonnées du projet et contraintes de versions (plages acceptables), écrites à la main. Entrée du résolveur.
   - `uv.lock` : **ce qui a été résolu**. Versions exactes de tous les paquets (directs + transitifs) avec leurs hashes, calculées une fois par uv. Sortie du résolveur, versionnée dans Git. Garantit que deux machines obtiennent les mêmes paquets.
   - `.venv/` : **ce qui est installé**. Dossier local contenant les paquets effectivement extraits et l'interpréteur Python. Spécifique à la machine, jamais commité, recréé par `uv sync`.

---

## Partie C - Inspection du dataset

### Résultats observés sur le snapshot

| Observation | Valeur |
|---|---:|
| Lignes | 7 043 |
| Colonnes | 21 |
| `Churn = No` | 5 174 |
| `Churn = Yes` | 1 869 |
| Proportion de churn | 26,54 % |
| IDs clients dupliqués | 0 |
| Valeurs `TotalCharges` composées uniquement d'espaces | 11 |
| SHA-256 vérifié | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |

### Question C3.1 - Pourquoi la prédiction de churn est-elle une tâche de classification ?

La cible `Churn` prend deux valeurs discrètes et mutuellement exclusives : `Yes` ou `No`. On ne prédit pas une quantité continue, on affecte chaque client à l'une de deux classes. C'est de la classification binaire. Un modèle de régression linéaire sur cette cible n'aurait pas de sens direct : on utilise une régression logistique, un arbre, un gradient boosting, etc., qui sortent une probabilité ou une classe.

### Question C3.2 - Que suggère la balance de classes pour l'usage de l'accuracy ?

73,5 % des clients ne churnent pas. Un modèle naïf qui prédit toujours « No » obtient 73,5 % d'accuracy sans rien avoir appris. L'accuracy seule n'est donc pas une métrique honnête ici : elle récompense la paresse.

Il faut regarder **precision, recall, F1 sur la classe minoritaire** (les churners), et probablement AUC-ROC ou AUC-PR. Le choix précis dépend du coût métier : rater un churner (faux négatif) coûte un client perdu ; prédire à tort un churn (faux positif) coûte une campagne de rétention inutile.

### Question C3.3 - Pourquoi `TotalCharges` est-elle lue comme object/string alors qu'elle représente un montant numérique ?

Pandas infère le type colonne par colonne au chargement. S'il tombe sur au moins une valeur qui n'est pas convertible en float, il garde toute la colonne en type générique `object` (strings).

Ici, 11 lignes contiennent des espaces à la place d'un montant (clients avec `tenure = 0`, donc jamais facturés). Ces 11 chaînes `" "` ne sont pas parsables en nombre, donc pandas renonce et typpe toute la colonne en object. Résultat : 7 032 valeurs qui ressemblent à des nombres restent stockées comme texte, inutilisables pour un modèle sans nettoyage préalable.

### Question C3.4 - Pourquoi `isna()` seul ne détecte-t-il pas une valeur composée uniquement d'espaces ?

`isna()` ne détecte que les sentinelles reconnues par pandas comme valeurs manquantes : `NaN`, `None`, `NaT`, et quelques chaînes vides sur certains dtypes. Une chaîne `" "` (un ou plusieurs espaces) est, pour pandas, **une valeur présente** : c'est un objet string non vide.

Pour attraper ces cas, il faut explicitement tester le contenu après strip :

```python
df["TotalCharges"].astype("string").str.strip().eq("")
```

Ou convertir avec `pd.to_numeric(..., errors="coerce")` et appeler `isna()` ensuite : la coercion transforme les strings non parsables en NaN, que `isna()` reconnaît.

### Question C3.5 - Quelle preuve montrerait à un coéquipier qu'il a inspecté le même fichier source ?

Le **SHA-256 des octets du fichier** : `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`.

Deux fichiers nommés `telco.csv` peuvent contenir des octets différents (encodage, BOM, saut de ligne, modif accidentelle). Le nom ne prouve rien, la taille non plus. Le hash cryptographique est stable à l'octet près : si deux hashes correspondent, les contenus sont identiques. C'est exactement ce que vérifie `scripts/inspect_telco.py` dès la première ligne, avant même d'ouvrir le dataframe.

En complément : l'URL commit-pinnée du dépôt IBM (contient le hash de commit `1fd6fd70906479be1712e64cfc0ea89f58466629`), qui garantit qu'on parle de la même version du fichier côté source.

---

## Partie D - Recréation de l'environnement

### D2 - Tableau de ce qui est reproduit

| Élément | Preuve produite en TP01 | Limite restante |
|---|---|---|
| Code | Commit Git (`1cb10a5`) | Les édits non commités ne sont pas identifiés par ce commit |
| Sélection de paquets | `pyproject.toml` + `uv.lock` versionnés | La plateforme et la version d'interpréteur peuvent faire choisir des wheels différentes |
| Environnement installé | `environment-before.json` et `environment.json` : Python, OS, uv, versions des paquets clés identiques avant/après suppression de `.venv/` | Ne capture ni les versions de glibc, ni les BLAS linkées, ni le matériel |
| Données source | URL commit-pinnée + SHA-256 vérifié par `inspect_telco.py` | Ne dit rien de la qualité des labels, d'un éventuel biais d'échantillonnage, ou de leakage |
| Modèle entraîné | Non produit en TP01 | Nécessite seeds, config d'entraînement, artefact et métriques (TP suivants) |

Le lock + hash sont des conditions **nécessaires** à la reproductibilité, pas **suffisantes** pour garantir deux entraînements identiques.

---

## Partie E - Git et pull request

- Branche créée : `docs/telco-onboarding`
- Commit sur la branche : `98af58d` - `docs: improve telco onboarding README`
- PR ouverte : `https://github.com/mhmdslim/churnlab-2026/pull/1`
- Reviewer : *(self-merge, pas de binôme disponible)*
- Question posée en revue : *« Est-ce que les instructions permettent de refaire la vérification du dataset depuis un environnement vierge, sans deviner d'étape ? »*
- Réponse et correction si besoin : *Oui, le README couvre le setup complet.*
- Commit de merge sur `main` : `197d60d`

---

## Partie F - Récupération

### F1 vs F2 - Quelle est la différence ?

Les deux commandes partent de la même situation (`README.md` édité localement) mais ne touchent pas aux mêmes zones.

**F1 - `git restore README.md`** agit sur la **working tree**. Il remplace le fichier sur disque par la version du dernier commit. Les modifications non stagées sont **perdues**. Utile pour jeter un essai raté.

**F2 - `git restore --staged README.md`** agit sur l'**index** (la zone de staging). Il retire le fichier du prochain commit prévu, mais ne touche pas au contenu du fichier sur disque : la modification reste visible et éditable. C'est l'inverse de `git add`.

Résumé mental :
- `git restore <file>` : jette ce qui n'est pas stagé (working tree ← HEAD)
- `git restore --staged <file>` : dé-stage, garde le travail (index ← HEAD)
- Les deux combinés rendent le fichier identique au dernier commit sur toutes les zones

Branche d'exercice utilisée : `exercise/local-recovery`. Elle a été supprimée après l'exercice pour ne pas polluer `main`.

---

## Checklist d'acceptation

- [x] `uv sync --locked` passe sans modifier `uv.lock`
- [x] `import churnlab` fonctionne dans l'environnement recréé
- [x] Le SHA-256 du CSV correspond au snapshot attendu
- [x] Le rapport d'inspection correspond aux comptages attendus (7 043 / 21 / 5 174 / 1 869 / 0 / 11)
- [x] Le CSV original n'a pas été modifié
- [x] Le README permet à un coéquipier de refaire setup + inspection
- [x] PR revue avec preuves de vérification dans la description
- [x] Différence `restore` vs `restore --staged` comprise et expliquée
- [x] Les limites restantes de reproductibilité sont nommées (D2)
