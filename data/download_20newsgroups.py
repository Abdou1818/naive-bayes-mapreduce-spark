"""
download_20newsgroups.py
========================

Pré-télécharge (met en cache) le jeu de données **20 Newsgroups** via
scikit-learn. C'est le GRAND jeu de données utilisé pour l'étude de scalabilité
(``src/benchmark.py``).

scikit-learn met les données en cache dans ``~/scikit_learn_data`` ; ce script
force simplement ce téléchargement une fois pour toutes et affiche un résumé.

Usage :
    python data/download_20newsgroups.py
"""

from __future__ import annotations

from sklearn.datasets import fetch_20newsgroups


def main() -> None:
    print("Téléchargement / mise en cache de 20 Newsgroups via scikit-learn ...")
    # subset="all" = train + test réunis. remove=(...) enlève en-têtes/pieds/citations
    # pour ne garder qu'un signal textuel (évite un sur-apprentissage trivial).
    bunch = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )
    print(f"OK : {len(bunch.data)} documents, {len(bunch.target_names)} catégories.")
    print("Catégories :")
    for name in bunch.target_names:
        print(f"  - {name}")
    print("\nLes données sont en cache (~/scikit_learn_data) et prêtes pour le benchmark.")


if __name__ == "__main__":
    main()
