"""
build_notebook_appendix.py
==========================

Génère le fragment LaTeX ``report/notebook_appendix.tex`` à partir du notebook
exécuté ``notebooks/demo.ipynb``. Chaque cellule est rendue explicitement avec
ses **entrées** (code) et ses **sorties** (résultats), comme l'exige l'item 5 du
barème (« working notebook ... add comments concerning input and output »).

Ré-exécuter après toute modification/ré-exécution du notebook :
    python report/build_notebook_appendix.py
"""

from __future__ import annotations

import os
import re

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NB_PATH = os.path.join(ROOT, "notebooks", "demo.ipynb")
OUT = os.path.join(HERE, "notebook_appendix.tex")
OUT_CODE = os.path.join(HERE, "code_appendix.tex")

# Fichiers source inclus dans l'annexe « tout le code » (item 5 du barème).
SOURCE_FILES = [
    "src/nb_common.py", "src/nb_rdd.py", "src/nb_dataframe.py", "src/benchmark.py",
    "src/uci_experiment.py", "tests/test_smoke.py",
    "data/download_sms_spam.py", "data/download_20newsgroups.py", "data/download_uci.py",
]

# Symboles Unicode non gérés directement en mode texte par inputenc : on les
# remplace par un équivalent ASCII/LaTeX. Les lettres accentuées latines et la
# ligature œ sont laissées telles quelles (inputenc utf8 les gère).
_TEXT_MAP = {
    "✅": "[OK]", "❌": "[X]", "⚠": "[!]",
    "—": "---", "–": "--", "→": r"$\rightarrow$",
    "≈": r"$\approx$", "±": r"$\pm$", "×": r"$\times$",
    "…": "...", "✓": "v",
    "α": r"$\alpha$", "β": r"$\beta$", "·": r"$\cdot$",
    "≤": r"$\leq$", "≥": r"$\geq$", "≠": r"$\neq$", "∑": r"$\sum$",
}
_ALLOWED_EXTRA = {"œ", "Œ"}  # œ, Œ


def escape_text(s: str) -> str:
    """Échappe les caractères spéciaux LaTeX d'un texte (hors code)."""
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def sanitize_text(s: str) -> str:
    """Remplace les symboles Unicode problématiques ; garde accents et œ."""
    out = []
    for ch in s:
        if ch in _TEXT_MAP:
            out.append(_TEXT_MAP[ch])
        elif ord(ch) < 256 or ch in _ALLOWED_EXTRA:
            out.append(ch)
        else:
            out.append("?")
    return "".join(out)


def md_inline(s: str) -> str:
    """Convertit le markdown *en ligne* (gras, code) en LaTeX, après échappement."""
    s = sanitize_text(s)
    s = escape_text(s)
    # **gras** -> \textbf{...}
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    # `code` -> \texttt{...}
    s = re.sub(r"`(.+?)`", r"\\texttt{\1}", s)
    return s


def render_markdown(src: str) -> str:
    """Rendu minimal d'une cellule markdown en LaTeX (titres, listes, citations)."""
    lines = src.splitlines()
    out: list[str] = []
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(r"\end{itemize}")
            in_list = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            close_list()
            out.append("")
            continue
        if stripped.startswith("### "):
            close_list()
            out.append(r"\medskip\noindent\textbf{" + md_inline(stripped[4:]) + r"}\par")
        elif stripped.startswith("## "):
            close_list()
            out.append(r"\medskip\noindent\textbf{\large " + md_inline(stripped[3:]) + r"}\par")
        elif stripped.startswith("# "):
            close_list()
            out.append(r"\medskip\noindent\textbf{\Large " + md_inline(stripped[2:]) + r"}\par")
        elif stripped.startswith("> "):
            close_list()
            out.append(r"\textit{" + md_inline(stripped[2:]) + r"}\par")
        elif stripped.startswith(("- ", "* ")):
            if not in_list:
                out.append(r"\begin{itemize}")
                in_list = True
            out.append(r"\item " + md_inline(stripped[2:]))
        else:
            close_list()
            out.append(md_inline(stripped) + r"\par")
    close_list()
    return "\n".join(out)


def code_block(code: str, caption: str) -> str:
    """Bloc de code (entrée) en lstlisting Python."""
    return (f"\n\\noindent\\textbf{{{caption}}}\n"
            "\\begin{lstlisting}[language=Python]\n" + code.rstrip() + "\n\\end{lstlisting}\n")


def output_block(text: str, caption: str) -> str:
    """Bloc de sortie (résultat) en lstlisting neutre."""
    # On tronque les sorties très longues pour rester lisible.
    lines = text.rstrip().splitlines()
    if len(lines) > 40:
        lines = lines[:40] + ["... (sortie tronquée)"]
    body = "\n".join(lines)
    return (f"\n\\noindent\\textit{{{caption}}}\n"
            "\\begin{lstlisting}[style=nboutput]\n" + body + "\n\\end{lstlisting}\n")


def tex_escape_filename(path: str) -> str:
    """Échappe les underscores d'un chemin pour un titre LaTeX."""
    return path.replace("_", r"\_")


def build_code_appendix() -> None:
    """Génère code_appendix.tex : tout le code source en blocs lstlisting.

    On inline le contenu (plutôt que \\lstinputlisting) pour rester compatible
    avec une installation LaTeX minimale (sans le paquet listingsutf8) : les
    caractères accentués sont gérés par la table ``literate`` du préambule.
    """
    parts = []
    for rel in SOURCE_FILES:
        code = open(os.path.join(ROOT, rel), encoding="utf-8").read().rstrip("\n")
        # Sécurité : le contenu ne doit pas contenir la fin d'environnement.
        assert "\\end{lstlisting}" not in code, rel
        parts.append(r"\subsection{\texttt{" + tex_escape_filename(rel) + "}}")
        parts.append(r"\begin{lstlisting}[style=py]")
        parts.append(code)
        parts.append(r"\end{lstlisting}")
    with open(OUT_CODE, "w", encoding="utf-8") as fh:
        fh.write("% Fichier généré par build_notebook_appendix.py — ne pas éditer à la main.\n")
        fh.write("\n".join(parts) + "\n")
    print(f"écrit : {OUT_CODE} ({len(SOURCE_FILES)} fichiers)")


def main() -> None:
    build_code_appendix()
    nb = nbformat.read(NB_PATH, as_version=4)
    parts: list[str] = []
    exec_n = 0

    for cell in nb.cells:
        if cell.cell_type == "markdown":
            parts.append(render_markdown(cell.source))
        elif cell.cell_type == "code":
            if not cell.source.strip():
                continue
            exec_n += 1
            parts.append(code_block(cell.source, f"Entrée [{exec_n}]"))
            # Sorties : stdout (stream) et résultats texte.
            texts = []
            for o in cell.get("outputs", []):
                if o.get("output_type") == "stream":
                    texts.append("".join(o.get("text", [])))
                elif "data" in o and "text/plain" in o["data"]:
                    texts.append("".join(o["data"]["text/plain"]))
            joined = "".join(texts).strip()
            if joined:
                parts.append(output_block(joined, f"Sortie [{exec_n}]"))

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("% Fichier généré par build_notebook_appendix.py — ne pas éditer à la main.\n")
        fh.write("\n".join(parts))
        fh.write("\n")
    print(f"écrit : {OUT} ({exec_n} cellules de code)")


if __name__ == "__main__":
    main()
