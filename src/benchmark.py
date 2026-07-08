"""
benchmark.py
============

Étude de **scalabilité** de Naive Bayes multinomial en Spark.

On fait varier deux facteurs et on mesure les temps d'entraînement et de
prédiction pour les deux versions (RDD et DataFrames) :

  1. la **taille des données**  : on réplique le jeu 20 Newsgroups plusieurs fois
     (facteurs de réplication), pour observer le passage à l'échelle en volume ;
  2. le **parallélisme**        : on fait varier ``local[k]`` (nombre de cœurs),
     pour observer l'accélération (speed-up) à volume constant.

Sorties :
  - ``results/results.csv``               : toutes les mesures brutes ;
  - ``results/scalability_datasize.png``  : temps vs taille des données ;
  - ``results/scalability_cores.png``     : temps vs nombre de cœurs (local[k]).

Usage :
    python src/benchmark.py                      # configuration par défaut
    python src/benchmark.py --quick              # version rapide (petits facteurs)
    python src/benchmark.py --factors 1 2 4 8 --cores 1 2 4 8
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import List

import nb_common as C
import nb_rdd
import nb_dataframe

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, "results")


def _timeit(fn):
    """Exécute ``fn`` et renvoie (résultat, temps écoulé en secondes)."""
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


def run_once(master: str, factor: int, base_texts: List[str], base_labels: List[str],
             seed: int = 42) -> List[dict]:
    """Mesure un point de benchmark pour un (master, facteur de réplication) donné.

    Renvoie une liste de dicts (une ligne par version RDD/DataFrame).
    """
    # 1) On réplique les données AVANT le split pour faire grossir le volume.
    texts, labels = C.replicate_dataset(base_texts, base_labels, factor)
    X_tr, X_te, y_tr, y_te = C.train_test_split_texts(texts, labels, seed=seed)
    data = C.prepare(X_tr, y_tr, X_te, y_te)

    # 2) SparkSession avec le parallélisme voulu (local[k]).
    spark = C.get_spark(app_name=f"bench-{master}-x{factor}", master=master)
    spark.sparkContext.setLogLevel("ERROR")
    sc = spark.sparkContext
    # Nombre de partitions ~ proportionnel au parallélisme (au moins 4).
    parts = max(4, sc.defaultParallelism)

    rows: List[dict] = []
    try:
        n_train = len(data.train)
        n_test = len(data.test)

        # --- Version RDD ----------------------------------------------------
        model_rdd, t_train_rdd = _timeit(
            lambda: nb_rdd.train_rdd(sc, data.train, data.vocab_size,
                                     data.idx_to_label, num_partitions=parts)
        )
        acc_rdd, t_pred_rdd = _timeit(
            lambda: nb_rdd.evaluate_rdd(sc, model_rdd, data.test, num_partitions=parts)
        )

        # --- Version DataFrame ---------------------------------------------
        model_df, t_train_df = _timeit(
            lambda: nb_dataframe.train_dataframe(spark, data.train, data.vocab_size,
                                                 data.idx_to_label, num_partitions=parts)
        )
        acc_df, t_pred_df = _timeit(
            lambda: nb_dataframe.evaluate_dataframe(spark, model_df, data.test,
                                                    num_partitions=parts)
        )

        common = dict(master=master, factor=factor, cores=sc.defaultParallelism,
                      n_train=n_train, n_test=n_test, vocab_size=data.vocab_size)
        rows.append({**common, "version": "rdd", "train_time_s": round(t_train_rdd, 4),
                     "predict_time_s": round(t_pred_rdd, 4), "accuracy": round(acc_rdd, 4)})
        rows.append({**common, "version": "dataframe", "train_time_s": round(t_train_df, 4),
                     "predict_time_s": round(t_pred_df, 4), "accuracy": round(acc_df, 4)})

        # Contrôle de cohérence : les deux versions doivent donner la même accuracy.
        assert abs(acc_rdd - acc_df) < 1e-9, (acc_rdd, acc_df)
        print(f"  [{master} x{factor}] train={n_train} | "
              f"RDD train={t_train_rdd:.2f}s pred={t_pred_rdd:.2f}s | "
              f"DF train={t_train_df:.2f}s pred={t_pred_df:.2f}s | acc={acc_rdd:.3f}")
    finally:
        spark.stop()

    return rows


def _plot(results: List[dict]) -> None:
    """Génère les graphiques PNG de scalabilité."""
    import matplotlib
    matplotlib.use("Agg")  # backend non interactif (pas de fenêtre)
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Graphe 1 : temps d'entraînement vs taille des données (à cores fixés) ---
    # On prend le master ayant le plus grand nombre de mesures par facteur.
    ref_master = max(
        {r["master"] for r in results},
        key=lambda m: len({r["factor"] for r in results if r["master"] == m}),
    )
    subset = [r for r in results if r["master"] == ref_master]
    if len({r["factor"] for r in subset}) >= 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        for version in ("rdd", "dataframe"):
            pts = sorted((r for r in subset if r["version"] == version),
                         key=lambda r: r["n_train"])
            ax.plot([r["n_train"] for r in pts], [r["train_time_s"] for r in pts],
                    marker="o", label=f"{version} (train)")
        ax.set_xlabel("Nombre de documents d'entraînement")
        ax.set_ylabel("Temps d'entraînement (s)")
        ax.set_title(f"Scalabilité en volume ({ref_master})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "scalability_datasize.png"), dpi=120)
        plt.close(fig)

    # --- Graphe 2 : temps vs nombre de cœurs (à facteur fixé le plus grand) ---
    factors_with_multi_cores = [
        f for f in {r["factor"] for r in results}
        if len({r["cores"] for r in results if r["factor"] == f}) >= 2
    ]
    if factors_with_multi_cores:
        ref_factor = max(factors_with_multi_cores)
        subset = [r for r in results if r["factor"] == ref_factor]
        fig, ax = plt.subplots(figsize=(7, 5))
        for version in ("rdd", "dataframe"):
            pts = sorted((r for r in subset if r["version"] == version),
                         key=lambda r: r["cores"])
            ax.plot([r["cores"] for r in pts], [r["train_time_s"] for r in pts],
                    marker="o", label=f"{version} (train)")
        ax.set_xlabel("Nombre de cœurs (local[k])")
        ax.set_ylabel("Temps d'entraînement (s)")
        ax.set_title(f"Speed-up selon le parallélisme (facteur x{ref_factor})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "scalability_cores.png"), dpi=120)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark de scalabilité Naive Bayes/Spark")
    parser.add_argument("--factors", type=int, nargs="+", default=[1, 2, 4],
                        help="Facteurs de réplication du jeu de données.")
    parser.add_argument("--cores", type=int, nargs="+", default=[1, 2, 4],
                        help="Nombres de cœurs pour local[k].")
    parser.add_argument("--categories", type=str, nargs="+", default=None,
                        help="Sous-ensemble de catégories 20 Newsgroups (défaut: toutes).")
    parser.add_argument("--quick", action="store_true",
                        help="Mode rapide : 4 catégories, facteurs [1,2], cores [1,2].")
    args = parser.parse_args()

    if args.quick:
        args.categories = ["sci.space", "rec.autos", "comp.graphics", "talk.politics.misc"]
        args.factors = [1, 2]
        args.cores = [1, 2]

    print("Chargement de 20 Newsgroups ...")
    base_texts, base_labels = C.load_20newsgroups(categories=args.categories, subset="all")
    print(f"  {len(base_texts)} documents, {len(set(base_labels))} classes.")

    all_rows: List[dict] = []

    # --- Expérience A : volume croissant à parallélisme maximal (local[*]) ---
    print("\n=== Scalabilité en VOLUME (local[*]) ===")
    for f in args.factors:
        all_rows += run_once("local[*]", f, base_texts, base_labels)

    # --- Expérience B : parallélisme croissant à volume fixe (facteur max) ---
    print("\n=== Scalabilité en PARALLÉLISME (facteur fixe) ===")
    fixed_factor = max(args.factors)
    for k in args.cores:
        all_rows += run_once(f"local[{k}]", fixed_factor, base_texts, base_labels)

    # --- Export CSV -----------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    fieldnames = ["master", "cores", "factor", "version", "n_train", "n_test",
                  "vocab_size", "train_time_s", "predict_time_s", "accuracy"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nMesures écrites dans {csv_path}")

    # --- Graphiques -----------------------------------------------------------
    _plot(all_rows)
    print(f"Graphiques écrits dans {RESULTS_DIR}/ (PNG).")


if __name__ == "__main__":
    main()
