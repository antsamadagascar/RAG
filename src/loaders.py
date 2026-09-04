"""
Chargement des fichiers sources (PDF, TXT, MD) en texte brut.

Auteur : Ratovonandrasana Aina Ny Antsa (ETU002754)

"""

import os

from langchain_community.document_loaders import PyMuPDFLoader


def read_text_with_fallback(path: str) -> str:
    """Lit un fichier .txt/.md en devinant son encodage.

    Windows produit souvent du texte en UTF-16 ou en Windows-1252, pas en
    UTF-8. On vérifie d'abord la présence d'un BOM ou d'octets nuls (signe
    typique d'UTF-16), sinon on essaie UTF-8, Windows-1252 puis Latin-1.
    """
    raw_bytes = open(path, "rb").read()

    has_utf16_bom = raw_bytes.startswith(b"\xff\xfe") or raw_bytes.startswith(b"\xfe\xff")
    null_byte_ratio = raw_bytes.count(b"\x00") / max(len(raw_bytes), 1)
    if has_utf16_bom or null_byte_ratio > 0.2:
        try:
            return raw_bytes.decode("utf-16")
        except UnicodeDecodeError:
            pass  # finalement pas de l'UTF-16, on continue avec le reste

    for encoding in ("utf-8-sig", "windows-1252", "latin-1"):
        try:
            text = raw_bytes.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
        if len(text) == 0:
            continue
        if text.count("\ufffd") / len(text) < 0.01:  # peu de caractères illisibles
            return text

    return raw_bytes.decode("utf-8", errors="ignore")  # dernier recours


def load_pdf_text(path: str) -> str:
    """Extrait tout le texte d'un PDF en un seul bloc.

    PyMuPDF charge un Document par page. On les fusionne en un seul texte
    avant le découpage, sinon le chevauchement (overlap) ne peut jamais
    franchir une frontière de page, et on perd le contexte d'un passage
    coupé pile à un saut de page.
    """
    pages = PyMuPDFLoader(path).load()
    return "\n".join(page.page_content for page in pages)


def load_file_text(path: str) -> str:
    """Dispatch vers le bon loader selon l'extension du fichier."""
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".pdf":
        return load_pdf_text(path)
    return read_text_with_fallback(path)
