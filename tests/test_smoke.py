"""
test_smoke.py
=============

Smoke test : vérifie de bout en bout que les DEUX versions (RDD et DataFrames)
- s'entraînent sur un mini-jeu de 20 lignes,
- produisent EXACTEMENT le même modèle et donc la même accuracy.

Exécutable directement (``python tests/test_smoke.py``) ou via pytest
(``pytest tests/test_smoke.py``).
"""

from __future__ import annotations

import os
import sys

# Rendre le dossier src importable (nb_common, nb_rdd, nb_dataframe).
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)

import nb_common as C          # noqa: E402
import nb_rdd                   # noqa: E402
import nb_dataframe            # noqa: E402


# --- Mini-jeu de données : 20 messages étiquetés spam/ham -------------------
# Volontairement séparable pour que la prédiction ait du sens sur si peu de data.
MINI_TEXTS = [
    "win a free prize now",
    "free money win cash prize",
    "claim your free lottery prize",
    "congratulations you won free cash",
    "urgent free offer claim now",
    "win big money free entry",
    "free tickets claim your prize",
    "cash prize winner claim now",
    "exclusive free offer win now",
    "you won a free vacation prize",
    "hey are we still meeting today",
    "call me when you get home",
    "lunch tomorrow at noon sounds good",
    "can you send me the report please",
    "happy birthday see you tonight",
    "the meeting is moved to friday",
    "thanks for your help yesterday",
    "i will be late for dinner",
    "did you finish the homework yet",
    "see you at the office monday",
]
MINI_LABELS = (["spam"] * 10) + (["ham"] * 10)


def run() -> None:
    # Démarrage d'une SparkSession locale utilisant tous les cœurs (local[*]).
    spark = C.get_spark(app_name="SmokeTest", master="local[*]")
    spark.sparkContext.setLogLevel("ERROR")  # logs Spark discrets
    sc = spark.sparkContext

    try:
        # Découpage train/test reproductible, puis vectorisation commune.
        X_tr, X_te, y_tr, y_te = C.train_test_split_texts(
            MINI_TEXTS, MINI_LABELS, test_size=0.3, seed=0
        )
        data = C.prepare(X_tr, y_tr, X_te, y_te)

        # Entraînement des deux versions sur EXACTEMENT les mêmes données.
        model_rdd = nb_rdd.train_rdd(
            sc, data.train, data.vocab_size, data.idx_to_label
        )
        model_df = nb_dataframe.train_dataframe(
            spark, data.train, data.vocab_size, data.idx_to_label
        )

        # Évaluation.
        acc_rdd = nb_rdd.evaluate_rdd(sc, model_rdd, data.test)
        acc_df = nb_dataframe.evaluate_dataframe(spark, model_df, data.test)

        # Prédictions brutes (doivent être identiques ligne à ligne).
        pred_rdd = nb_rdd.predict_rdd(sc, model_rdd, data.test)
        pred_df = nb_dataframe.predict_dataframe(spark, model_df, data.test)

        print(f"Taille vocabulaire : {data.vocab_size}")
        print(f"Train / Test       : {len(data.train)} / {len(data.test)}")
        print(f"Accuracy RDD       : {acc_rdd:.4f}")
        print(f"Accuracy DataFrame : {acc_df:.4f}")
        print(f"Prédictions RDD    : {pred_rdd}")
        print(f"Prédictions DF     : {pred_df}")

        # --- Assertions du smoke test ---------------------------------------
        assert acc_rdd == acc_df, (
            f"Accuracies différentes : RDD={acc_rdd} vs DF={acc_df}"
        )
        assert pred_rdd == pred_df, (
            "Les prédictions ligne à ligne diffèrent entre RDD et DataFrame."
        )
        print("\nOK : RDD et DataFrame donnent des prédictions et une accuracy identiques.")
    finally:
        spark.stop()


def test_smoke():
    """Point d'entrée pytest."""
    run()


if __name__ == "__main__":
    run()
