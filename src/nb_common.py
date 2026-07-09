"""
nb_common.py
============

Briques *communes* aux deux implémentations de Naive Bayes multinomial
(version RDD dans ``nb_rdd.py`` et version DataFrames dans ``nb_dataframe.py``).

On centralise ici tout ce qui **doit être strictement identique** entre les deux
versions pour garantir des prédictions rigoureusement égales :

1. la tokenisation (bag-of-words) ;
2. la construction du vocabulaire et de la table des étiquettes ;
3. la vectorisation d'un document en liste d'indices de mots ;
4. la *construction du modèle* à partir de comptes entiers (comptes -> log-probas) ;
5. la *fonction de prédiction* d'un document à partir du modèle.

Les points 4 et 5 sont volontairement écrits **une seule fois** : les versions RDD
et DataFrames se contentent de produire, chacune à leur manière (map/reduce),
exactement les mêmes comptes entiers, puis appellent ces fonctions communes.
Comme les comptes sont des entiers (donc reproductibles au bit près) et que la
prédiction est déterministe, les deux versions renvoient des prédictions
identiques.

La modélisation reproduit fidèlement ``sklearn.naive_bayes.MultinomialNB`` avec
``alpha=1`` et ``fit_prior=True`` :

    log P(c)      = log( N_c / N )
    log P(w | c)  = log( (count_{w,c} + alpha) / (total_c + alpha * V) )

où :
    N            = nombre de documents d'entraînement
    N_c          = nombre de documents de la classe c
    count_{w,c}  = nombre total d'occurrences du mot w dans les docs de classe c
    total_c      = sum_w count_{w,c}  (nombre total de tokens de la classe c)
    V            = taille du vocabulaire
    alpha        = paramètre de lissage de Laplace (1 par défaut)

Score d'un document d pour une classe c (tout en logarithme) :

    score(d, c) = log P(c) + sum_{w in d} count_w(d) * log P(w | c)

La classe prédite est celle qui maximise ce score.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# 1. Tokenisation (bag-of-words)
# ---------------------------------------------------------------------------
# On reproduit la tokenisation par défaut de sklearn.CountVectorizer :
#   - passage en minuscules,
#   - motif r"\b\w\w+\b" : suites d'au moins 2 caractères de mot.
# Utiliser exactement la même règle est indispensable pour pouvoir comparer nos
# résultats à ceux de sklearn.MultinomialNB dans le notebook.
_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def tokenize(text: str) -> List[str]:
    """Découpe un texte en liste de tokens (mots), comme sklearn CountVectorizer.

    >>> tokenize("Free entry, WIN a prize!!!")
    ['free', 'entry', 'win', 'prize']
    """
    if text is None:
        return []
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# 2. Vocabulaire et étiquettes
# ---------------------------------------------------------------------------
def build_vocabulary(
    tokenized_docs: Sequence[Sequence[str]], min_df: int = 1
) -> Dict[str, int]:
    """Construit le vocabulaire {mot -> indice} à partir des documents d'entraînement.

    Le vocabulaire est trié par ordre alphabétique (comme sklearn) afin que les
    indices soient parfaitement déterministes d'une exécution à l'autre.

    Paramètres
    ----------
    tokenized_docs : documents déjà tokenisés (liste de listes de mots).
    min_df         : nombre minimal de documents dans lesquels un mot doit
                     apparaître pour être conservé (1 = on garde tout).
    """
    # Compte le nombre de *documents* contenant chaque mot (document frequency).
    doc_freq: Dict[str, int] = {}
    for tokens in tokenized_docs:
        for w in set(tokens):  # set() -> un mot compte une seule fois par doc
            doc_freq[w] = doc_freq.get(w, 0) + 1

    # On ne garde que les mots suffisamment fréquents, puis on trie -> indices stables.
    kept = sorted(w for w, df in doc_freq.items() if df >= min_df)
    return {w: i for i, w in enumerate(kept)}


def build_label_index(labels: Sequence[str]) -> Dict[str, int]:
    """Construit la table {étiquette -> entier} (triée pour être déterministe)."""
    return {lab: i for i, lab in enumerate(sorted(set(labels)))}


def doc_to_indices(tokens: Sequence[str], vocab: Dict[str, int]) -> List[int]:
    """Vectorise un document en liste d'indices de mots (avec répétitions).

    Les mots hors-vocabulaire (OOV) sont ignorés : cela correspond exactement au
    comportement de sklearn, où la matrice de features a un nombre fixe de
    colonnes défini au moment du ``fit`` sur l'ensemble d'entraînement.

    Exemple : si vocab = {"free": 0, "win": 1} et tokens = ["free", "win", "free", "zzz"],
    on renvoie [0, 1, 0] (le token OOV "zzz" est retiré).
    """
    out: List[int] = []
    for w in tokens:
        idx = vocab.get(w)
        if idx is not None:
            out.append(idx)
    return out


# ---------------------------------------------------------------------------
# 3. Le modèle Naive Bayes (structure partagée)
# ---------------------------------------------------------------------------
@dataclass
class NaiveBayesModel:
    """Modèle Naive Bayes multinomial *dense-équivalent mais stocké creux*.

    Pour respecter le lissage de Laplace, sklearn stocke une log-proba pour
    CHAQUE (classe, mot) du vocabulaire, y compris les mots de compte nul dans
    une classe. Ce serait coûteux (n_classes * V valeurs). On stocke donc :

      - ``log_prior[c]``              : log P(c)
      - ``log_likelihood[c][w]``      : log P(w|c) uniquement pour les mots dont
                                        le compte est NON nul dans la classe c
      - ``log_likelihood_default[c]`` : log P(w|c) pour tout mot de compte nul
                                        dans la classe c
                                        = log( alpha / (total_c + alpha * V) )

    C'est rigoureusement équivalent au calcul dense de sklearn : lors de la
    prédiction, un mot présent dans le document mais de compte nul dans la
    classe utilise simplement la valeur ``default`` (voir ``predict_indices``).
    Cette représentation creuse est aussi ce que l'on *broadcast* aux workers.
    """

    n_classes: int
    n_docs: int
    vocab_size: int
    alpha: float
    # index entier de classe -> étiquette d'origine (str)
    idx_to_label: List[str]
    log_prior: List[float]
    # log_likelihood[c] : dict {indice_mot -> log P(w|c)} (comptes non nuls seult.)
    log_likelihood: List[Dict[int, float]]
    log_likelihood_default: List[float]


def build_model(
    *,
    n_docs: int,
    vocab_size: int,
    idx_to_label: List[str],
    class_doc_counts: Dict[int, int],
    class_token_totals: Dict[int, int],
    word_class_counts: Dict[Tuple[int, int], int],
    alpha: float = 1.0,
) -> NaiveBayesModel:
    """Construit le modèle (log-probas) à partir de **comptes entiers**.

    C'est le cœur du calcul Naive Bayes, factorisé ici pour que les versions RDD
    et DataFrames produisent EXACTEMENT le même modèle : il leur suffit de fournir
    les mêmes agrégats entiers (calculés en map/reduce), le reste est identique.

    Paramètres (tous issus d'un comptage MapReduce dans les versions Spark)
    ----------------------------------------------------------------------
    n_docs             : N   (nb total de documents d'entraînement)
    vocab_size         : V   (taille du vocabulaire)
    idx_to_label       : correspondance indice_classe -> étiquette
    class_doc_counts   : {c -> N_c}         nb de docs par classe
    class_token_totals : {c -> total_c}     nb total de tokens par classe
    word_class_counts  : {(c, w) -> count}  nb d'occurrences du mot w en classe c
    alpha              : lissage de Laplace (1.0 par défaut)
    """
    n_classes = len(idx_to_label)

    # --- Priors : log P(c) = log(N_c / N) -----------------------------------
    log_prior = [0.0] * n_classes
    for c in range(n_classes):
        n_c = class_doc_counts.get(c, 0)
        # Un prior nul (classe absente) donnerait log(0) = -inf ; en pratique
        # toutes les classes sont présentes en entraînement.
        log_prior[c] = math.log(n_c / n_docs) if n_c > 0 else float("-inf")

    # --- Log-vraisemblance par mot : log P(w|c) -----------------------------
    # Dénominateur commun à une classe : total_c + alpha * V
    denom = [class_token_totals.get(c, 0) + alpha * vocab_size for c in range(n_classes)]

    # Valeur par défaut (mot de compte nul dans la classe) : log(alpha / denom)
    log_likelihood_default = [
        math.log(alpha / denom[c]) if denom[c] > 0 else float("-inf")
        for c in range(n_classes)
    ]

    # Pour chaque (classe, mot) réellement observé : log((count + alpha) / denom)
    log_likelihood: List[Dict[int, float]] = [dict() for _ in range(n_classes)]
    for (c, w), count in word_class_counts.items():
        log_likelihood[c][w] = math.log((count + alpha) / denom[c])

    return NaiveBayesModel(
        n_classes=n_classes,
        n_docs=n_docs,
        vocab_size=vocab_size,
        alpha=alpha,
        idx_to_label=list(idx_to_label),
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        log_likelihood_default=log_likelihood_default,
    )


def predict_indices(model: NaiveBayesModel, token_indices: Sequence[int]) -> int:
    """Prédit l'indice de classe d'un document (liste d'indices de mots).

    Fonction PARTAGÉE par les deux versions -> prédictions identiques garanties.

    On calcule, pour chaque classe c :

        score(c) = log P(c) + sum_{w in doc} log P(w|c)

    (chaque occurrence d'un mot ajoute son log P(w|c), ce qui revient bien à
    multiplier par le compte du mot dans le document). On renvoie l'argmax.
    """
    best_c = 0
    best_score = float("-inf")
    for c in range(model.n_classes):
        score = model.log_prior[c]
        ll_c = model.log_likelihood[c]
        default_c = model.log_likelihood_default[c]
        for w in token_indices:
            # get(w, default) : mot de compte nul dans la classe -> valeur lissée.
            score += ll_c.get(w, default_c)
        if score > best_score:
            best_score = score
            best_c = c
    return best_c


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Exactitude (fraction de prédictions correctes)."""
    if not y_true:
        return 0.0
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


# ---------------------------------------------------------------------------
# 4. Chargement des jeux de données
# ---------------------------------------------------------------------------
def load_sms_spam(csv_path: str) -> Tuple[List[str], List[str]]:
    """Charge le jeu SMS Spam Collection depuis un CSV (colonnes label,text).

    Le fichier est produit par ``data/download_sms_spam.py``.
    Renvoie (textes, étiquettes) avec étiquettes dans {"ham", "spam"}.
    """
    import csv

    texts: List[str] = []
    labels: List[str] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            labels.append(row["label"])
            texts.append(row["text"])
    return texts, labels


def load_20newsgroups(
    categories: List[str] | None = None, subset: str = "all"
) -> Tuple[List[str], List[str]]:
    """Charge 20 Newsgroups via scikit-learn (téléchargé/caché automatiquement).

    Renvoie (textes, étiquettes) où l'étiquette est le nom de la catégorie.
    """
    from sklearn.datasets import fetch_20newsgroups

    bunch = fetch_20newsgroups(
        subset=subset,
        categories=categories,
        remove=("headers", "footers", "quotes"),  # pour un signal purement textuel
        shuffle=True,
        random_state=42,
    )
    labels = [bunch.target_names[t] for t in bunch.target]
    return list(bunch.data), labels


def replicate_dataset(
    texts: List[str], labels: List[str], factor: int
) -> Tuple[List[str], List[str]]:
    """Duplique le jeu de données ``factor`` fois (pour l'étude de scalabilité).

    Multiplier la taille des données à distribution constante permet de mesurer
    comment le temps d'exécution évolue avec le *volume*, sans changer l'accuracy.
    """
    if factor <= 1:
        return list(texts), list(labels)
    return texts * factor, labels * factor


# ---------------------------------------------------------------------------
# 5. Découpage entraînement / test
# ---------------------------------------------------------------------------
def train_test_split_texts(
    texts: List[str],
    labels: List[str],
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Découpe (textes, étiquettes) en ensembles d'entraînement et de test.

    On délègue à sklearn pour un découpage stratifié et reproductible.
    Renvoie (X_train, X_test, y_train, y_test).
    """
    from sklearn.model_selection import train_test_split

    # stratify=labels garde les mêmes proportions de classes dans train et test.
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 6. Préparation commune : textes -> (indices de classe, indices de mots)
# ---------------------------------------------------------------------------
@dataclass
class PreparedData:
    """Données prêtes pour l'entraînement Spark, partagées par les deux versions."""

    vocab: Dict[str, int]
    label_to_idx: Dict[str, int]
    idx_to_label: List[str]
    vocab_size: int
    # Chaque exemple : (indice_classe, [indices de mots avec répétitions])
    train: List[Tuple[int, List[int]]]
    test: List[Tuple[int, List[int]]]


def prepare(
    X_train: List[str],
    y_train: List[str],
    X_test: List[str],
    y_test: List[str],
    min_df: int = 1,
) -> PreparedData:
    """Tokenise, construit vocabulaire + étiquettes, et vectorise train/test.

    Cette étape déterministe est faite dans le driver (les données tiennent en
    mémoire ; c'est du prétraitement). Le vocabulaire est construit **uniquement
    sur l'entraînement** (comme un ``fit`` sklearn), puis appliqué au test.
    """
    train_tokens = [tokenize(t) for t in X_train]
    test_tokens = [tokenize(t) for t in X_test]

    vocab = build_vocabulary(train_tokens, min_df=min_df)
    label_to_idx = build_label_index(y_train)
    idx_to_label = [lab for lab, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]

    train = [
        (label_to_idx[lab], doc_to_indices(toks, vocab))
        for lab, toks in zip(y_train, train_tokens)
    ]
    # Les étiquettes de test inconnues à l'entraînement sont improbables ici
    # (split stratifié) ; on suppose qu'elles existent dans label_to_idx.
    test = [
        (label_to_idx[lab], doc_to_indices(toks, vocab))
        for lab, toks in zip(y_test, test_tokens)
    ]

    return PreparedData(
        vocab=vocab,
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
        vocab_size=len(vocab),
        train=train,
        test=test,
    )


# ---------------------------------------------------------------------------
# 7. SparkSession avec le bon JDK
# ---------------------------------------------------------------------------
def _resolve_java_home() -> str | None:
    """Trouve un JAVA_HOME compatible Spark 3.5 (JDK 8/11/17).

    Spark 3.5 ne supporte PAS les JDK plus récents (20, 21, 23...). Sur cette
    machine le ``java`` par défaut peut être trop récent : on force donc un JDK 17
    via ``/usr/libexec/java_home -v 17`` (macOS).
    """
    # Si JAVA_HOME est déjà positionné sur un JDK supporté, on le respecte.
    for version in ("17", "11", "1.8"):
        try:
            out = subprocess.run(
                ["/usr/libexec/java_home", "-v", version],
                capture_output=True,
                text=True,
                check=True,
            )
            path = out.stdout.strip()
            if path and os.path.isdir(path):
                return path
        except Exception:
            continue
    return None


def get_spark(app_name: str = "NaiveBayesMapReduce", master: str = "local[*]"):
    """Crée (ou récupère) une SparkSession configurée pour ce projet.

    - force un JDK compatible (17) si nécessaire ;
    - ``master`` permet de choisir le parallélisme (ex. "local[2]", "local[*]").
    """
    java_home = _resolve_java_home()
    if java_home:
        os.environ["JAVA_HOME"] = java_home

    # Le worker Python doit utiliser le même interpréteur que le driver (venv).
    import sys

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        # Logs moins verbeux et démarrage plus rapide en local.
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    # Les workers exécutent nos fonctions (predict_indices) et désérialisent le
    # modèle (dataclass NaiveBayesModel) : ils doivent donc pouvoir importer nos
    # modules. On expédie les fichiers source du projet à tous les exécuteurs.
    src_dir = os.path.dirname(os.path.abspath(__file__))
    for mod in ("nb_common.py", "nb_rdd.py", "nb_dataframe.py"):
        path = os.path.join(src_dir, mod)
        if os.path.isfile(path):
            spark.sparkContext.addPyFile(path)

    return spark


# ===========================================================================
# 8. Métriques d'évaluation (au-delà de l'accuracy)
# ===========================================================================
# Le papier de référence (Zheng, 2014, §2.3.3) mesure aussi la précision, le
# rappel et la F-mesure, à partir d'une table de contingence / matrice de
# confusion. On les implémente ici (multiclasse, avec moyenne « macro »).
def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int],
                     n_classes: int) -> List[List[int]]:
    """Matrice de confusion ``M[t][p]`` = nb d'exemples de vraie classe ``t``
    prédits en classe ``p``. La diagonale = prédictions correctes."""
    m = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        m[t][p] += 1
    return m


def precision_recall_f1(y_true: Sequence[int], y_pred: Sequence[int],
                        n_classes: int) -> dict:
    """Précision, rappel et F1 par classe, plus les moyennes « macro ».

    Pour une classe c (vue en « un contre tous ») :
        précision = tp / (tp + fp)   (part des prédits c qui sont vraiment c)
        rappel    = tp / (tp + fn)   (part des vrais c qu'on retrouve)
        F1        = 2·P·R / (P + R)  (moyenne harmonique)
    où tp = M[c][c], fp = somme_colonne_c hors diagonale, fn = somme_ligne_c hors diagonale.
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    per_class = {}
    precisions, recalls, f1s = [], [], []
    for c in range(n_classes):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(n_classes)) - tp   # prédits c mais pas c
        fn = sum(cm[c][k] for k in range(n_classes)) - tp   # vrais c prédits ailleurs
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
    n = max(n_classes, 1)
    return {
        "per_class": per_class,
        "macro_precision": sum(precisions) / n,
        "macro_recall": sum(recalls) / n,
        "macro_f1": sum(f1s) / n,
        "confusion_matrix": cm,
    }


# ===========================================================================
# 9. Validation croisée k-fold (§2.3.4 et §3.3.2 du papier)
# ===========================================================================
def kfold_split(items: Sequence, labels: Sequence, k: int = 5, seed: int = 42):
    """Générateur de découpes k-fold : cède ``(X_train, y_train, X_test, y_test)``.

    Le papier fait une validation croisée à 6 plis pour réduire le biais dû à un
    unique découpage. À chaque tour, 1 pli sert de test et les k-1 autres de train.
    On mélange les indices de façon reproductible (seed) avant de les répartir.
    """
    import random

    n = len(items)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    # Répartition en k plis de tailles quasi égales.
    folds = [idx[i::k] for i in range(k)]
    for f in range(k):
        test_idx = set(folds[f])
        X_tr = [items[i] for i in idx if i not in test_idx]
        y_tr = [labels[i] for i in idx if i not in test_idx]
        X_te = [items[i] for i in folds[f]]
        y_te = [labels[i] for i in folds[f]]
        yield X_tr, y_tr, X_te, y_te


# ===========================================================================
# 10. Naive Bayes CATÉGORIEL (attribut-valeur) — la variante du papier
# ===========================================================================
# Le papier applique Naive Bayes à des jeux TABULAIRES catégoriels (adult,
# mushroom, nursery...). Chaque instance a M attributs ; on estime
#     P(A_i = a | c) = (count_{a,i,c} + alpha) / (N_c + alpha * D_i)
# où D_i est le nombre de valeurs distinctes de l'attribut i (domaine), et N_c le
# nombre d'instances de la classe c. C'est la différence clé avec le multinomial :
# le dénominateur est N_c (nb d'instances) et non le total de tokens de la classe.
# Le comptage map/reduce est IDENTIQUE ; seul le calcul du modèle change.

@dataclass
class PreparedTabular:
    """Données tabulaires prêtes pour l'entraînement (partagé RDD/DataFrame)."""

    vocab: Dict[Tuple[int, str], int]   # (indice_attribut, valeur) -> indice de feature
    label_to_idx: Dict[str, int]
    idx_to_label: List[str]
    vocab_size: int
    train: List[Tuple[int, List[int]]]  # (classe, [indices de features, 1 par attribut])
    test: List[Tuple[int, List[int]]]
    feature_attr: List[int]             # indice de feature -> indice d'attribut
    attr_domain_sizes: List[int]        # attribut -> nb de valeurs distinctes (D_i)


def prepare_tabular(X_train: List[List[str]], y_train: List[str],
                    X_test: List[List[str]], y_test: List[str],
                    label_to_idx: Dict[str, int] | None = None) -> PreparedTabular:
    """Vectorise des instances tabulaires catégorielles en indices de features.

    Une « feature » est un couple (attribut, valeur) : on garde les attributs
    distincts même si deux attributs partagent une même valeur. Le vocabulaire et
    les domaines sont construits sur l'ENTRAÎNEMENT seulement (comme un fit).
    Chaque instance devient une liste de M indices (un par attribut).

    ``label_to_idx`` peut être fourni (mapping GLOBAL des étiquettes) : utile en
    validation croisée pour garder des indices de classe cohérents entre plis,
    même si une classe rare est absente du train d'un pli.
    """
    m = len(X_train[0])
    attr_values = [set() for _ in range(m)]
    for row in X_train:
        for i, v in enumerate(row):
            attr_values[i].add(v)

    # Vocabulaire trié (déterministe) : par attribut puis par valeur.
    vocab: Dict[Tuple[int, str], int] = {}
    feature_attr: List[int] = []
    for i in range(m):
        for v in sorted(attr_values[i]):
            vocab[(i, v)] = len(vocab)
            feature_attr.append(i)
    attr_domain_sizes = [len(attr_values[i]) for i in range(m)]

    if label_to_idx is None:
        label_to_idx = build_label_index(y_train)
    idx_to_label = [lab for lab, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]

    def to_indices(row: List[str]) -> List[int]:
        # Une valeur inconnue à l'entraînement (OOV) est ignorée pour cet attribut.
        out = []
        for i, v in enumerate(row):
            idx = vocab.get((i, v))
            if idx is not None:
                out.append(idx)
        return out

    train = [(label_to_idx[l], to_indices(r)) for l, r in zip(y_train, X_train)]
    test = [(label_to_idx[l], to_indices(r)) for l, r in zip(y_test, X_test)]
    return PreparedTabular(vocab=vocab, label_to_idx=label_to_idx,
                           idx_to_label=idx_to_label, vocab_size=len(vocab),
                           train=train, test=test, feature_attr=feature_attr,
                           attr_domain_sizes=attr_domain_sizes)


def build_model_categorical(
    *,
    n_docs: int,
    vocab_size: int,
    idx_to_label: List[str],
    class_doc_counts: Dict[int, int],
    class_token_totals: Dict[int, int],       # ignoré (garde la même signature)
    word_class_counts: Dict[Tuple[int, int], int],
    alpha: float = 1.0,
    feature_attr: List[int],
    attr_domain_sizes: List[int],
) -> NaiveBayesModel:
    """Construit le modèle Naive Bayes CATÉGORIEL à partir des comptes entiers.

    Même signature que ``build_model`` (pour être branché de la même façon dans
    les versions RDD/DataFrame) plus deux métadonnées : ``feature_attr`` et
    ``attr_domain_sizes``. Reproduit ``sklearn.naive_bayes.CategoricalNB``.

    log P(A_i=a | c) = log( (count + alpha) / (N_c + alpha * D_i) )
    """
    n_classes = len(idx_to_label)

    # Priors : log P(c) = log(N_c / N).
    log_prior = [
        math.log(class_doc_counts.get(c, 0) / n_docs) if class_doc_counts.get(c, 0) > 0
        else float("-inf")
        for c in range(n_classes)
    ]

    # On stocke DENSE (vocab tabulaire petit) : une log-proba par (classe, feature),
    # y compris les features de compte nul (lissées). Le dénominateur dépend de
    # l'attribut de la feature : N_c + alpha * D_{attr(feature)}.
    log_likelihood: List[Dict[int, float]] = [dict() for _ in range(n_classes)]
    for c in range(n_classes):
        n_c = class_doc_counts.get(c, 0)
        for f in range(vocab_size):
            d_i = attr_domain_sizes[feature_attr[f]]
            cnt = word_class_counts.get((c, f), 0)
            log_likelihood[c][f] = math.log((cnt + alpha) / (n_c + alpha * d_i))

    # Défaut pour une feature hors-vocabulaire au test (valeur jamais vue) : rare ;
    # on suppose un compte nul avec un domaine minimal.
    log_likelihood_default = [
        math.log(alpha / (class_doc_counts.get(c, 0) + alpha))
        if class_doc_counts.get(c, 0) + alpha > 0 else float("-inf")
        for c in range(n_classes)
    ]

    return NaiveBayesModel(
        n_classes=n_classes, n_docs=n_docs, vocab_size=vocab_size, alpha=alpha,
        idx_to_label=list(idx_to_label), log_prior=log_prior,
        log_likelihood=log_likelihood, log_likelihood_default=log_likelihood_default,
    )


# ---------------------------------------------------------------------------
# 11. Discrétisation des attributs continus (job de prétraitement du papier)
# ---------------------------------------------------------------------------
def spark_minmax(sc, rows: List[List[str]], continuous_cols: List[int]) -> Dict[int, Tuple[float, float]]:
    """Calcule (min, max) de chaque colonne continue via un job MAP/REDUCE.

    Reproduit le PREMIER job MapReduce du papier (PreprocessMapper/Reducer) :
      * MAP    : pour chaque colonne continue, émettre (col, (valeur, valeur)) ;
      * REDUCE : reduceByKey en combinant par (min, max).
    """
    def emit(row):
        for c in continuous_cols:
            v = float(row[c])
            yield (c, (v, v))

    def combine(a, b):
        return (min(a[0], b[0]), max(a[1], b[1]))

    res = sc.parallelize(rows).flatMap(emit).reduceByKey(combine).collect()
    return {c: (lo, hi) for c, (lo, hi) in res}


def discretize(rows: List[List[str]], continuous_cols: List[int],
               minmax: Dict[int, Tuple[float, float]], n_bins: int = 5) -> List[List[str]]:
    """Remplace chaque valeur continue par un identifiant de tranche (binning).

    Binning à largeur égale sur [min, max] (bornes issues du train), transformant
    l'attribut continu en attribut catégoriel (`bin0`..`bin{n-1}`) exploitable par
    le Naive Bayes catégoriel — comme le fait le papier (§3.3.3).
    """
    out = []
    for row in rows:
        new = list(row)
        for c in continuous_cols:
            lo, hi = minmax[c]
            v = float(row[c])
            if hi <= lo:
                k = 0
            else:
                k = int((v - lo) / (hi - lo) * n_bins)
                k = min(max(k, 0), n_bins - 1)   # borne dans [0, n_bins-1]
            new[c] = f"bin{k}"
        out.append(new)
    return out
