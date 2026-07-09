"""
download_uci.py
===============

Télécharge les jeux de données tabulaires catégoriels de l'UCI utilisés dans le
papier de référence (Zheng, 2014) : **mushroom, car, nursery, adult**.

Ces jeux servent à la variante **Naive Bayes catégoriel** (``src/uci_experiment.py``),
au plus proche du papier. Les fichiers bruts sont mis en cache dans ``data/uci/``.

Usage :
    python data/download_uci.py
"""

from __future__ import annotations

import os
import urllib.request

BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "uci")

# nom -> URL du fichier brut .data
DATASETS = {
    "mushroom": f"{BASE}/mushroom/agaricus-lepiota.data",
    "car": f"{BASE}/car/car.data",
    "nursery": f"{BASE}/nursery/nursery.data",
    "adult": f"{BASE}/adult/adult.data",
}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, url in DATASETS.items():
        dest = os.path.join(OUT_DIR, f"{name}.data")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"  {name:9s} déjà présent -> {dest}")
            continue
        print(f"  {name:9s} téléchargement depuis {url} ...")
        with urllib.request.urlopen(url, timeout=60) as resp:
            raw = resp.read()
        with open(dest, "wb") as fh:
            fh.write(raw)
        n = raw.count(b"\n")
        print(f"            OK ({n} lignes) -> {dest}")
    print("Terminé : jeux UCI en cache dans data/uci/.")


if __name__ == "__main__":
    main()
