# ML on Big Data : Naive Bayes en MapReduce sur Spark

Implémentation de **Naive Bayes multinomial** (lissage de Laplace, `alpha=1`, calculs
en logarithme) en **deux versions PySpark** — une en **RDD** (`map`/`reduceByKey`) et
une en **DataFrames** (`explode` + `groupBy`/`agg`) — accompagnée d'une **étude de
scalabilité**.

Les deux versions partagent la construction du modèle et la fonction de prédiction
(`src/nb_common.py`), ce qui garantit des **prédictions rigoureusement identiques**.
Le modèle reproduit `sklearn.naive_bayes.MultinomialNB` (vérifié dans le notebook).

## Structure du projet

```
src/
  nb_common.py      # tokenisation, vocabulaire, split, calcul du modèle & prédiction (partagés)
  nb_rdd.py         # entraînement + prédiction en RDD (map / reduceByKey)
  nb_dataframe.py   # même algo en DataFrames (explode / groupBy-agg + broadcast du modèle)
  benchmark.py      # boucle de scalabilité (taille des données × local[k]) -> results.csv + PNG
data/
  download_sms_spam.py       # télécharge SMS Spam Collection -> data/sms_spam.csv (PETIT jeu)
  download_20newsgroups.py   # met en cache 20 Newsgroups via scikit-learn (GRAND jeu)
notebooks/
  demo.ipynb        # démo exécutable sur PETITES données (SMS Spam) + comparaison sklearn
tests/
  test_smoke.py     # entraîne les 2 versions sur 20 lignes et vérifie l'égalité des accuracy
results/            # sorties du benchmark (results.csv, graphes PNG)
report/             # dossier pour le rapport
requirements.txt
```

## Prérequis

- **Python ≥ 3.10** (le projet a été créé et testé avec **Python 3.12**).
  ⚠️ `pyspark==3.5.*` ne supporte pas Python 3.13/3.14 : utilisez 3.10–3.12.
- **JDK 8, 11 ou 17** requis par Spark 3.5 (⚠️ pas les JDK plus récents comme 20/21).
  Vérifiez : `java -version`.

### Installer un JDK (si absent)

macOS (Homebrew) :

```bash
brew install openjdk@17
```

Le projet sélectionne automatiquement un JDK compatible sur macOS via
`/usr/libexec/java_home -v 17` (voir `get_spark` dans `src/nb_common.py`) ; vous
n'avez donc normalement pas besoin de définir `JAVA_HOME` à la main tant qu'un
JDK 17 (ou 11 ou 8) est installé.

## Installation

```bash
# Créer l'environnement virtuel avec Python 3.12 (adapter si besoin)
python3.12 -m venv .venv
source .venv/bin/activate

# Installer les dépendances figées
pip install -r requirements.txt
```

## Récupérer les données

```bash
# Petit jeu (SMS Spam) — utilisé par le notebook et le smoke test
python data/download_sms_spam.py

# Grand jeu (20 Newsgroups) — utilisé par le benchmark de scalabilité
python data/download_20newsgroups.py
```

## Exécution

### Smoke test (rapide, sans données externes)

```bash
python tests/test_smoke.py
# ou :  pytest tests/test_smoke.py
```

Attendu : les deux versions affichent la **même accuracy** et les **mêmes prédictions**.

### Notebook de démonstration

```bash
jupyter lab notebooks/demo.ipynb
```

Le notebook tourne **de bout en bout sur les petites données (SMS Spam)** sans cluster :
chargement, vectorisation, entraînement RDD **et** DataFrame, comparaison à
`sklearn.MultinomialNB`.

### Benchmark de scalabilité

```bash
python src/benchmark.py --quick                 # rapide (4 catégories)
python src/benchmark.py                          # configuration par défaut
python src/benchmark.py --factors 1 2 4 8 --cores 1 2 4 8   # personnalisé
```

Produit `results/results.csv` et les graphes `results/scalability_datasize.png`
et `results/scalability_cores.png`.

## Notes d'implémentation (points notés)

- **Cœur MapReduce** — comptage `count_{w,c}` :
  - RDD : `flatMap` émet `((classe, mot), 1)` par occurrence, puis `reduceByKey(add)` ;
  - DataFrame : `explode(words)` (une ligne par mot) puis `groupBy(label, word).count()`.
- **Calcul du modèle** (`nb_common.build_model`) : passage des comptes entiers aux
  log-probabilités avec lissage de Laplace ; factorisé pour être identique dans les
  deux versions.
- **Broadcast** : le modèle (creux) est diffusé une seule fois par worker
  (`sc.broadcast`) puis utilisé en lecture seule lors de la prédiction (map RDD / UDF DF).
