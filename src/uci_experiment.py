"""
uci_experiment.py
=================

Variante **Naive Bayes catégoriel** (celle du papier de référence, Zheng 2014),
appliquée aux jeux tabulaires UCI (mushroom, car, nursery, adult), en Spark, avec :

  * **validation croisée k-fold** (comme le papier, §3.3.2) ;
  * **discrétisation** des attributs continus via un job MAP/REDUCE (adult, §3.3.3) ;
  * **métriques** : accuracy (moyenne ± écart-type sur les plis), plus
    précision / rappel / F1 (macro) et matrice de confusion (§2.3.3) ;
  * vérification que les versions **RDD et DataFrames** donnent des prédictions
    identiques à chaque pli.

Le comptage MapReduce est le MÊME que pour le texte (``nb_rdd`` / ``nb_dataframe``)
— seule la construction du modèle change (``build_model_categorical``), branchée
via le paramètre ``model_builder``.

Usage :
    python src/uci_experiment.py                 # tous les jeux, 6 plis
    python src/uci_experiment.py --dataset mushroom --folds 6
"""

from __future__ import annotations

import argparse
import csv
import functools
import os
import statistics
from typing import List, Tuple

import nb_common as C
import nb_rdd
import nb_dataframe

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
UCI_DIR = os.path.join(ROOT, "data", "uci")
RESULTS_DIR = os.path.join(ROOT, "results")

# Configuration de chaque jeu : séparateur, position de l'étiquette, et indices
# (dans la liste d'attributs, étiquette exclue) des attributs CONTINUS à discrétiser.
DATASETS = {
    # mushroom : étiquette en 1re colonne, 22 attributs catégoriels, rien de continu.
    "mushroom": {"label": "first", "continuous": []},
    # car : étiquette en dernière colonne, 6 attributs catégoriels.
    "car": {"label": "last", "continuous": []},
    # nursery : étiquette en dernière colonne, 8 attributs catégoriels.
    "nursery": {"label": "last", "continuous": []},
    # adult : étiquette en dernière colonne, attributs continus à discrétiser :
    #   age(0), fnlwgt(2), education-num(4), capital-gain(10), capital-loss(11),
    #   hours-per-week(12).
    "adult": {"label": "last", "continuous": [0, 2, 4, 10, 11, 12]},
}


def load_uci(name: str) -> Tuple[List[List[str]], List[str], List[int]]:
    """Charge un jeu UCI en (lignes d'attributs, étiquettes, indices continus)."""
    cfg = DATASETS[name]
    path = os.path.join(UCI_DIR, f"{name}.data")
    rows: List[List[str]] = []
    labels: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            fields = [f.strip() for f in line.split(",")]
            if cfg["label"] == "first":
                label, attrs = fields[0], fields[1:]
            else:
                label, attrs = fields[-1], fields[:-1]
            rows.append(attrs)
            labels.append(label)
    return rows, labels, cfg["continuous"]


def _run_fold(sc, spark, X_tr, y_tr, X_te, y_te, continuous, n_bins, label_to_idx):
    """Entraîne (RDD + DataFrame) sur un pli et renvoie (y_true, y_pred, acc).

    Les indices de classe sont GLOBAUX (via ``label_to_idx``) pour rester cohérents
    entre plis lors de l'agrégation des métriques.
    """
    # Discrétisation des attributs continus via le job MAP/REDUCE (min/max sur le
    # TRAIN uniquement, pour ne pas fuiter d'information du test).
    if continuous:
        minmax = C.spark_minmax(sc, X_tr, continuous)
        X_tr = C.discretize(X_tr, continuous, minmax, n_bins=n_bins)
        X_te = C.discretize(X_te, continuous, minmax, n_bins=n_bins)

    data = C.prepare_tabular(X_tr, y_tr, X_te, y_te, label_to_idx=label_to_idx)
    # Le constructeur de modèle catégoriel a besoin des métadonnées d'attributs.
    cat_builder = functools.partial(
        C.build_model_categorical,
        feature_attr=data.feature_attr, attr_domain_sizes=data.attr_domain_sizes,
    )

    m_rdd = nb_rdd.train_rdd(sc, data.train, data.vocab_size, data.idx_to_label,
                             model_builder=cat_builder)
    m_df = nb_dataframe.train_dataframe(spark, data.train, data.vocab_size,
                                        data.idx_to_label, model_builder=cat_builder)
    p_rdd = nb_rdd.predict_rdd(sc, m_rdd, data.test)
    p_df = nb_dataframe.predict_dataframe(spark, m_df, data.test)
    assert p_rdd == p_df, "RDD et DataFrame doivent prédire à l'identique"

    y_true = [c for c, _ in data.test]
    return y_true, p_rdd, C.accuracy(y_true, p_rdd)


def run_dataset(spark, name: str, folds: int, n_bins: int) -> dict:
    """Exécute la validation croisée k-fold sur un jeu et renvoie un dict de mesures."""
    sc = spark.sparkContext
    rows, labels, continuous = load_uci(name)
    label_to_idx = C.build_label_index(labels)  # pour agréger les métriques globales
    n_classes = len(label_to_idx)

    fold_accs: List[float] = []
    all_true: List[int] = []
    all_pred: List[int] = []

    for X_tr, y_tr, X_te, y_te in C.kfold_split(rows, labels, k=folds, seed=42):
        y_true, y_pred, acc = _run_fold(sc, spark, X_tr, y_tr, X_te, y_te,
                                        continuous, n_bins, label_to_idx)
        fold_accs.append(acc)
        # Indices de classe GLOBAUX -> agrégation directe des métriques sur tous les plis.
        all_true.extend(y_true)
        all_pred.extend(y_pred)

    mean_acc = statistics.mean(fold_accs)
    std_acc = statistics.pstdev(fold_accs) if len(fold_accs) > 1 else 0.0
    prf = C.precision_recall_f1(all_true, all_pred, n_classes)

    print(f"\n=== {name} ({len(rows)} instances, {n_classes} classes, "
          f"{folds}-fold CV) ===")
    print(f"  Accuracy       : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Macro P/R/F1   : {prf['macro_precision']:.4f} / "
          f"{prf['macro_recall']:.4f} / {prf['macro_f1']:.4f}")
    print(f"  Matrice de confusion (lignes=vrai, colonnes=prédit) :")
    labels_sorted = [lab for lab, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]
    for i, rowm in enumerate(prf["confusion_matrix"]):
        print(f"    {labels_sorted[i]:>12s} | {rowm}")

    return {
        "dataset": name, "n_instances": len(rows), "n_classes": n_classes,
        "folds": folds, "acc_mean": round(mean_acc, 4), "acc_std": round(std_acc, 4),
        "macro_precision": round(prf["macro_precision"], 4),
        "macro_recall": round(prf["macro_recall"], 4),
        "macro_f1": round(prf["macro_f1"], 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Naive Bayes catégoriel sur jeux UCI")
    parser.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    parser.add_argument("--folds", type=int, default=6, help="Nombre de plis (CV).")
    parser.add_argument("--bins", type=int, default=5,
                        help="Nombre de tranches pour la discrétisation des continus.")
    args = parser.parse_args()

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]

    spark = C.get_spark(app_name="uci-categorical-nb", master="local[*]")
    spark.sparkContext.setLogLevel("ERROR")
    rows_out = []
    try:
        for name in names:
            rows_out.append(run_dataset(spark, name, args.folds, args.bins))
    finally:
        spark.stop()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "uci_results.csv")
    fields = ["dataset", "n_instances", "n_classes", "folds", "acc_mean", "acc_std",
              "macro_precision", "macro_recall", "macro_f1"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nRésultats écrits dans {csv_path}")


if __name__ == "__main__":
    main()
