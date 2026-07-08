"""
nb_rdd.py
=========

Naive Bayes multinomial **en MapReduce sur des RDD Spark**.

C'est la version « bas niveau » : on manipule directement des RDD et on exprime
l'entraînement comme une suite d'opérations *map* / *reduceByKey*, qui est
exactement l'esprit MapReduce.

Idée clé — comptages par MapReduce :
    Un document est (indice_classe, [indices de mots]).
    * MAP    : pour chaque occurrence d'un mot w dans un doc de classe c,
               on émet la paire clé/valeur ((c, w), 1).
    * REDUCE : reduceByKey(add) additionne les 1 -> nombre d'occurrences de w
               dans la classe c, soit count_{w,c}.
    De même on compte les documents par classe et les tokens par classe.

Ces agrégats entiers sont ensuite passés à ``nb_common.build_model`` (calcul des
log-probas), puis le modèle est **broadcasté** aux workers pour la prédiction.
"""

from __future__ import annotations

from operator import add
from typing import List, Tuple

import nb_common as C


def train_rdd(sc, train_data: List[Tuple[int, List[int]]], vocab_size: int,
              idx_to_label: List[str], alpha: float = 1.0, num_partitions: int = 8
              ) -> C.NaiveBayesModel:
    """Entraîne le modèle Naive Bayes en RDD (map/reduceByKey).

    Paramètres
    ----------
    sc             : SparkContext.
    train_data     : liste de (indice_classe, [indices de mots]).
    vocab_size     : V, taille du vocabulaire.
    idx_to_label   : correspondance indice_classe -> étiquette d'origine.
    alpha          : lissage de Laplace.
    num_partitions : nombre de partitions du RDD (parallélisme des données).
    """
    # On distribue les documents sur le cluster (ici local[k]) en `num_partitions`.
    docs = sc.parallelize(train_data, numSlices=num_partitions)

    # --- (a) Nombre total de documents : N ----------------------------------
    n_docs = docs.count()

    # --- (b) Nombre de documents par classe : N_c ---------------------------
    #   MAP    : chaque doc -> (classe, 1)
    #   REDUCE : reduceByKey(add) -> somme des 1 par classe
    class_doc_counts = dict(
        docs.map(lambda cw: (cw[0], 1)).reduceByKey(add).collect()
    )

    # --- (c) Nombre total de tokens par classe : total_c --------------------
    #   MAP    : chaque doc -> (classe, nb de tokens du doc)
    #   REDUCE : somme par classe
    class_token_totals = dict(
        docs.map(lambda cw: (cw[0], len(cw[1]))).reduceByKey(add).collect()
    )

    # --- (d) Compte des mots par classe : count_{w,c}  (LE cœur MapReduce) ---
    #   MAP (flatMap) : pour un doc (c, [w1, w2, w1, ...]), on émet
    #                   ((c, w1), 1), ((c, w2), 1), ((c, w1), 1), ...
    #                   -> une paire par occurrence de mot.
    #   REDUCE        : reduceByKey(add) additionne toutes les occurrences,
    #                   donnant count_{w,c} pour chaque couple (classe, mot).
    def emit_word_class(doc: Tuple[int, List[int]]):
        c, word_indices = doc
        for w in word_indices:
            yield ((c, w), 1)

    word_class_counts = dict(
        docs.flatMap(emit_word_class).reduceByKey(add).collect()
    )

    # --- (e) Construction du modèle (comptes entiers -> log-probas) ---------
    # Calcul factorisé dans nb_common pour être identique à la version DataFrame.
    return C.build_model(
        n_docs=n_docs,
        vocab_size=vocab_size,
        idx_to_label=idx_to_label,
        class_doc_counts=class_doc_counts,
        class_token_totals=class_token_totals,
        word_class_counts=word_class_counts,
        alpha=alpha,
    )


def predict_rdd(sc, model: C.NaiveBayesModel,
                test_data: List[Tuple[int, List[int]]], num_partitions: int = 8
                ) -> List[int]:
    """Prédit les classes des documents de test, en parallèle sur les RDD.

    Le modèle est **broadcasté** : Spark l'envoie une seule fois à chaque worker
    (au lieu de le sérialiser avec chaque tâche), ce qui est essentiel quand le
    modèle est volumineux (V mots x C classes).
    """
    # BROADCAST : diffusion du modèle en lecture seule à tous les workers.
    bc_model = sc.broadcast(model)

    docs = sc.parallelize(test_data, numSlices=num_partitions)

    # MAP : chaque doc -> classe prédite, via la fonction de prédiction COMMUNE.
    # bc_model.value récupère le modèle local au worker (pas de re-sérialisation).
    preds = docs.map(
        lambda cw: C.predict_indices(bc_model.value, cw[1])
    ).collect()

    bc_model.unpersist()
    return preds


def evaluate_rdd(sc, model: C.NaiveBayesModel,
                 test_data: List[Tuple[int, List[int]]], num_partitions: int = 8
                 ) -> float:
    """Renvoie l'exactitude (accuracy) du modèle sur l'ensemble de test."""
    y_true = [c for c, _ in test_data]
    y_pred = predict_rdd(sc, model, test_data, num_partitions=num_partitions)
    return C.accuracy(y_true, y_pred)
