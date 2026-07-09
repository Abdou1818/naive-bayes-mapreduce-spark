# Naive Bayes multinomial en MapReduce sur Spark

![CI](https://github.com/Abdou1818/naive-bayes-mapreduce-spark/actions/workflows/ci.yml/badge.svg)

Projet du cours **« ML on Big Data »** (Prof. D. Colazzo) — Master MIAGE SITN,
Université Paris-Dauphine.

Implémentation de **Naive Bayes multinomial** (lissage de Laplace `alpha=1`, calculs en
logarithme) en **deux versions PySpark** — **RDD** (`map`/`reduceByKey`) et **DataFrames**
(`explode` + `groupBy`/`agg`) — accompagnée d'une **étude de scalabilité**.

Les deux versions partagent la construction du modèle et la fonction de prédiction
(`src/nb_common.py`), ce qui garantit des **prédictions rigoureusement identiques** ; le
modèle reproduit `sklearn.naive_bayes.MultinomialNB` (vérifié dans le notebook et le test).

📄 **Rapport (livrable) :** [`report/rapport.pdf`](report/rapport.pdf) — source LaTeX :
[`report/rapport.tex`](report/rapport.tex).

## Structure du projet

```
src/
  nb_common.py       # tokenisation, vocabulaire, split, build_model, predict_indices, get_spark (partagés)
  nb_rdd.py          # entraînement + prédiction en RDD (flatMap / reduceByKey + broadcast)
  nb_dataframe.py    # même algo en DataFrames (explode / groupBy-agg + UDF sur modèle broadcasté)
  benchmark.py       # scalabilité (volume × local[k], --repeat N) -> results/results.csv + PNG
data/
  download_sms_spam.py       # SMS Spam Collection -> data/sms_spam.csv (PETIT jeu)
  download_20newsgroups.py   # met en cache 20 Newsgroups via scikit-learn (GRAND jeu)
notebooks/
  demo.ipynb         # démo exécutable de bout en bout sur PETITES données + comparaison sklearn
tests/
  test_smoke.py      # entraîne les 2 versions sur 20 lignes et vérifie l'égalité RDD == DataFrame
report/
  rapport.tex / rapport.pdf  # rapport (LaTeX + PDF)
  figures/                   # graphes utilisés par le rapport
  build_notebook_appendix.py # génère les annexes (code + notebook) du rapport
results/             # sorties du benchmark (results.csv, PNG) — regénérables, non versionnées
.github/workflows/   # CI : lance le smoke test à chaque push
requirements.txt
```

## Prérequis

- **Python 3.10–3.12** ⚠️ `pyspark==3.5.*` ne supporte pas Python 3.13/3.14.
- **JDK 8, 11 ou 17** ⚠️ pas les JDK plus récents (20/21…). Vérifiez : `java -version`.

Installer un JDK si absent (macOS) : `brew install openjdk@17`.
Sur macOS, `get_spark()` sélectionne automatiquement un JDK compatible via
`/usr/libexec/java_home -v 17` ; vous n'avez pas à définir `JAVA_HOME`. Sous Linux (ex. la CI),
c'est le `JAVA_HOME` de l'environnement (JDK 17) qui est utilisé.

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Récupérer les données

Le notebook et le benchmark téléchargent les données automatiquement au besoin ; on peut
aussi le faire explicitement :

```bash
python data/download_sms_spam.py         # petit jeu (notebook, smoke test)
python data/download_20newsgroups.py     # grand jeu (benchmark)
```

## Exécution

**Smoke test** (rapide, sans données externes) — vérifie RDD == DataFrame :

```bash
python tests/test_smoke.py
# variante : pip install pytest && pytest -q tests/test_smoke.py
```

**Notebook** (de bout en bout sur les petites données SMS Spam, sans cluster) :

```bash
jupyter lab notebooks/demo.ipynb
```

**Benchmark de scalabilité** (`--repeat N` moyenne les mesures ± écart-type) :

```bash
python src/benchmark.py --quick                               # rapide (4 catégories)
python src/benchmark.py --factors 1 2 4 8 --cores 1 2 4 8 --repeat 3   # complet
# -> results/results.csv + results/scalability_{datasize,cores}.png
```

## Reproduire le rapport PDF

```bash
cd report
python build_notebook_appendix.py    # (re)génère les annexes code + notebook
latexmk -pdf rapport.tex             # -> rapport.pdf
```

Prérequis LaTeX : `pdflatex` + `latexmk` (TeX Live) avec les paquets `listings`, `booktabs`,
`fancyhdr`, `setspace`, `hyperref` (installation standard).

## Notes d'implémentation (points clés)

- **Cœur MapReduce** — comptage `count_{w,c}` :
  - RDD : `flatMap` émet `((classe, mot), 1)` par occurrence, puis `reduceByKey(add)` ;
  - DataFrame : `explode(words)` (une ligne par mot) puis `groupBy(label, word).count()`.
- **Calcul du modèle** (`nb_common.build_model`) : comptes entiers → log-probabilités avec
  lissage de Laplace ; factorisé pour être identique dans les deux versions.
- **Broadcast** : le modèle (creux) est diffusé une fois par worker (`sc.broadcast`), utilisé
  en lecture seule lors de la prédiction (map RDD / UDF DataFrame).
- **Anti-fuite / bruit** (benchmark) : réplication faite **après** le split train/test
  (séparément de chaque côté) ; `--repeat N` moyenne les temps sur N exécutions.
