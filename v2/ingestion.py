"""
Pipeline d'ingestion : extraction du texte, découpage en chunks,
vectorisation dans Chroma.

Étape 2 du sujet. Ce module ne connaît rien à Streamlit : il prend des
fichiers uploadés en entrée et retourne un vectorstore, ce qui le rend
testable indépendamment de l'interface.
"""

import os
import tempfile
import uuid

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SEPARATORS, CHUNK_SIZE, EMBEDDING_MODEL


def read_text_with_fallback(path):
    """Lit un fichier .txt/.md en devinant son encodage.

    Tous les fichiers texte ne sont pas en UTF-8 (Windows en produit
    souvent en UTF-16 ou en Windows-1252). On regarde d'abord s'il y a un
    BOM ou beaucoup d'octets nuls (signe classique d'UTF-16), sinon on
    teste UTF-8, Windows-1252 et Latin-1 dans cet ordre.
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


def _extract_document(uploaded_file):
    """Extrait le texte brut d'un seul fichier uploadé (PDF, TXT ou MD)."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    # Les loaders LangChain veulent un chemin sur disque, donc on écrit le
    # contenu uploadé (en mémoire) dans un fichier temporaire.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        if suffix == ".pdf":
            docs = PyMuPDFLoader(tmp_path).load()
            # PyMuPDF charge un Document par page. On les fusionne en un
            # seul texte par fichier avant le découpage, sinon le
            # chevauchement (overlap) ne peut jamais franchir une frontière
            # de page, et on perd le contexte d'un passage coupé pile à un
            # saut de page.
            full_text = "\n".join(doc.page_content for doc in docs)
        else:  # .txt, .md
            full_text = read_text_with_fallback(tmp_path)
    finally:
        os.remove(tmp_path)

    return Document(page_content=full_text, metadata={"source": uploaded_file.name})


def _deduplicate_chunks(chunks):
    """Supprime les chunks strictement identiques (mise en page PDF dupliquée...)."""
    seen_texts = set()
    unique_chunks = []
    for chunk in chunks:
        if chunk.page_content not in seen_texts:
            seen_texts.add(chunk.page_content)
            unique_chunks.append(chunk)
    return unique_chunks


def process_uploaded_files(uploaded_files):
    """Extrait, découpe et vectorise les fichiers uploadés.

    Retourne (vectorstore, nombre_de_chunks).
    """
    all_docs = [_extract_document(f) for f in uploaded_files]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )
    chunks = text_splitter.split_documents(all_docs)
    chunks = _deduplicate_chunks(chunks)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Un nom de collection unique (uuid) à chaque indexation : sans ça,
    # Chroma peut réutiliser une collection par défaut au sein du même
    # processus Python et accumuler les documents d'une indexation à
    # l'autre, même sans persistance disque. On veut repartir de zéro à
    # chaque clic sur "Indexer les documents".
    vectorstore = Chroma.from_documents(
        chunks, embeddings, collection_name=f"session_{uuid.uuid4().hex}"
    )

    return vectorstore, len(chunks)
