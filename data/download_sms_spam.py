"""
download_sms_spam.py
====================

Télécharge le jeu de données **SMS Spam Collection** (UCI) et l'exporte en CSV
(``data/sms_spam.csv``) avec deux colonnes : ``label`` (ham/spam) et ``text``.

C'est le PETIT jeu de données, utilisé par le notebook de démonstration et le
smoke test. Il ne nécessite pas de cluster.

Usage :
    python data/download_sms_spam.py
"""

from __future__ import annotations

import csv
import io
import os
import urllib.request
import zipfile

# Miroir officiel UCI de la SMS Spam Collection (fichier zip contenant un TSV).
URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# URL de repli (autre miroir historique).
FALLBACK_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/"
    "data/sms.tsv"
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "sms_spam.csv")


def _rows_from_zip(raw: bytes):
    """Extrait les lignes (label, text) du zip UCI (fichier 'SMSSpamCollection')."""
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        # Le zip contient un fichier tabulé : "<label>\t<message>" par ligne.
        name = next(n for n in zf.namelist() if "SMSSpamCollection" in n)
        with zf.open(name) as fh:
            for line in io.TextIOWrapper(fh, encoding="utf-8"):
                line = line.rstrip("\n")
                if not line:
                    continue
                label, _, text = line.partition("\t")
                yield label.strip(), text


def _rows_from_tsv(raw: bytes):
    """Extrait les lignes (label, text) du miroir TSV de repli."""
    for line in io.StringIO(raw.decode("utf-8")):
        line = line.rstrip("\n")
        if not line:
            continue
        label, _, text = line.partition("\t")
        yield label.strip(), text


def main() -> None:
    print(f"Téléchargement depuis {URL} ...")
    try:
        with urllib.request.urlopen(URL, timeout=60) as resp:
            raw = resp.read()
        rows = list(_rows_from_zip(raw))
    except Exception as exc:  # pragma: no cover - dépend du réseau
        print(f"  échec ({exc}); tentative sur le miroir de repli...")
        with urllib.request.urlopen(FALLBACK_URL, timeout=60) as resp:
            raw = resp.read()
        rows = list(_rows_from_tsv(raw))

    # Écriture du CSV normalisé (en-tête label,text).
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", "text"])
        writer.writerows(rows)

    n_spam = sum(1 for lab, _ in rows if lab == "spam")
    print(f"OK : {len(rows)} messages écrits dans {OUT_CSV} "
          f"({n_spam} spam / {len(rows) - n_spam} ham).")


if __name__ == "__main__":
    main()
