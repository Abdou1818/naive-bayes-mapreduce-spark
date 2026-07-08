"""
nb_dataframe.py
===============

Naive Bayes multinomial **avec l'API DataFrames de Spark**.

Même algorithme que ``nb_rdd.py``, mais exprimé avec des opérations
relationnelles de haut niveau (``explode``, ``groupBy`` / ``agg``) au lieu de
map/reduceByKey. Le moteur Catalyst optimise ces opérations.

Correspondance avec la version RDD :
    * ``explode`` d'un tableau de mots  ≈  la phase MAP (une ligne par mot) ;
    * ``groupBy(...).agg(count/sum)``   ≈  la phase REDUCE (reduceByKey).

On produit exactement les mêmes agrégats entiers que la version RDD, puis on
appelle le MÊME ``nb_common.build_model`` et la MÊME fonction de prédiction :
les deux versions renvoient donc des prédictions identiques.
Le modèle est **broadcasté** aux workers pour la prédiction via une UDF.
"""

from __future__ import annotations

from typing import List, Tuple

from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StructField,
    StructType,
)

import nb_common as C


def _to_dataframe(spark, data: List[Tuple[int, List[int]]], num_partitions: int):
    """Construit un DataFrame (label:int, words:array<int>) à partir des données.

    Chaque ligne = un document : son indice de classe et son sac de mots (liste
    d'indices de mots, avec répétitions).
    """
    schema = StructType([
        StructField("label", IntegerType(), False),
        StructField("words", ArrayType(IntegerType()), False),
    ])
    rows = [Row(label=c, words=list(w)) for c, w in data]
    return spark.createDataFrame(rows, schema=schema).repartition(num_partitions)


def train_dataframe(spark, train_data: List[Tuple[int, List[int]]], vocab_size: int,
                    idx_to_label: List[str], alpha: float = 1.0, num_partitions: int = 8
                    ) -> C.NaiveBayesModel:
    """Entraîne le modèle Naive Bayes avec l'API DataFrames (groupBy/agg)."""
    df = _to_dataframe(spark, train_data, num_partitions)
    df.cache()  # réutilisé plusieurs fois ci-dessous -> on le met en cache

    # --- (a) Nombre total de documents : N ----------------------------------
    n_docs = df.count()

    # --- (b) Nombre de documents par classe : N_c ---------------------------
    #   groupBy(label).count()  ≈  REDUCE : compte les lignes par classe.
    class_doc_counts = {
        r["label"]: r["cnt"]
        for r in df.groupBy("label").agg(F.count("*").alias("cnt")).collect()
    }

    # --- (c) Nombre total de tokens par classe : total_c --------------------
    #   size(words) = nb de tokens du doc ; on somme par classe.
    class_token_totals = {
        r["label"]: r["tot"]
        for r in df.groupBy("label")
        .agg(F.sum(F.size("words")).alias("tot"))
        .collect()
    }

    # --- (d) Compte des mots par classe : count_{w,c}  (cœur MAP/REDUCE) -----
    #   MAP    : explode(words) -> une ligne (label, word) PAR occurrence de mot
    #            (équivaut au flatMap qui émet ((c, w), 1) de la version RDD).
    #   REDUCE : groupBy(label, word).count() -> count_{w,c}.
    exploded = df.select("label", F.explode("words").alias("word"))
    word_class_counts = {
        (r["label"], r["word"]): r["cnt"]
        for r in exploded.groupBy("label", "word")
        .agg(F.count("*").alias("cnt"))
        .collect()
    }

    df.unpersist()

    # --- (e) Construction du modèle (identique à la version RDD) ------------
    return C.build_model(
        n_docs=n_docs,
        vocab_size=vocab_size,
        idx_to_label=idx_to_label,
        class_doc_counts=class_doc_counts,
        class_token_totals=class_token_totals,
        word_class_counts=word_class_counts,
        alpha=alpha,
    )


def predict_dataframe(spark, model: C.NaiveBayesModel,
                      test_data: List[Tuple[int, List[int]]], num_partitions: int = 8
                      ) -> List[int]:
    """Prédit les classes de test via une UDF s'appuyant sur le modèle broadcasté."""
    sc = spark.sparkContext

    # BROADCAST : le modèle est diffusé une fois par worker (lecture seule),
    # au lieu d'être capturé/sérialisé pour chaque tâche.
    bc_model = sc.broadcast(model)

    # UDF : applique la fonction de prédiction COMMUNE à chaque sac de mots.
    # La closure référence bc_model.value, résolu localement sur chaque worker.
    @F.udf(returnType=IntegerType())
    def predict_udf(words):
        return int(C.predict_indices(bc_model.value, words or []))

    # On attache un identifiant EXPLICITE (position dans test_data) AVANT tout
    # shuffle, afin de pouvoir réordonner les prédictions et les aligner sur
    # test_data, quel que soit le repartitionnement effectué par Spark.
    schema = StructType([
        StructField("idx", IntegerType(), False),
        StructField("words", ArrayType(IntegerType()), False),
    ])
    rows_in = [Row(idx=i, words=list(w)) for i, (_, w) in enumerate(test_data)]
    df = spark.createDataFrame(rows_in, schema=schema).repartition(num_partitions)

    df = df.withColumn("pred", predict_udf(F.col("words")))
    rows = df.select("idx", "pred").orderBy("idx").collect()

    bc_model.unpersist()
    return [r["pred"] for r in rows]


def evaluate_dataframe(spark, model: C.NaiveBayesModel,
                       test_data: List[Tuple[int, List[int]]], num_partitions: int = 8
                       ) -> float:
    """Renvoie l'exactitude (accuracy) du modèle sur l'ensemble de test."""
    y_true = [c for c, _ in test_data]
    y_pred = predict_dataframe(spark, model, test_data, num_partitions=num_partitions)
    return C.accuracy(y_true, y_pred)
