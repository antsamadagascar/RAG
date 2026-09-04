"""
Pipeline d'ingestion : extraction -> découpage -> vectorisation.

Auteur : Ratovonandrasana Aina Ny Antsa (ETU002754)

"""

import os
import tempfile
import uuid

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SEPARATORS, CHUNK_SIZE, EMBEDDING_MODEL
from loaders import load_file_text


@st.cache_resource(show_spinner=False)
def get_embeddings_model() -> HuggingFaceEmbeddings:
    """Charge le modèle d'embedding une seule fois par session serveur.

    Sans ce cache, chaque clic sur "Indexer" rechargeait le modèle depuis
    zéro, ce qui expliquait une bonne partie de la lenteur au démarrage.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _uploaded_files_to_documents(uploaded_files) -> list[Document]:
    """Convertit les fichiers uploadés (en mémoire) en Documents LangChain.

    Les loaders LangChain veulent un chemin sur disque, donc chaque
    fichier est d'abord écrit dans un fichier temporaire.
    """
    documents = []
    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        try:
            full_text = load_file_text(tmp_path)
        finally:
            os.remove(tmp_path)

        documents.append(
            Document(page_content=full_text, metadata={"source": uploaded_file.name})
        )
    return documents


def _split_into_chunks(documents: list[Document]) -> list[Document]:
    """Découpe les documents en chunks, puis déduplique les doublons.

    Certains PDF contiennent le même passage deux fois dans leur texte
    brut (mise en page, effets de style...), ce qui donne des chunks
    identiques. On les déduplique au texte exact.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )
    chunks = splitter.split_documents(documents)

    seen_texts = set()
    unique_chunks = []
    for chunk in chunks:
        if chunk.page_content not in seen_texts:
            seen_texts.add(chunk.page_content)
            unique_chunks.append(chunk)
    return unique_chunks


def process_uploaded_files(uploaded_files) -> tuple[Chroma, int]:
    """Extrait, découpe et vectorise les fichiers uploadés.

    Retourne le vectorstore Chroma et le nombre de chunks générés.
    """
    documents = _uploaded_files_to_documents(uploaded_files)
    chunks = _split_into_chunks(documents)

    embeddings = get_embeddings_model()

    # Un nom de collection unique (uuid) à chaque indexation : sans ça,
    # Chroma peut réutiliser une collection par défaut au sein du même
    # processus Python et accumuler les documents d'une indexation à
    # l'autre, même sans persistance disque. On veut repartir de zéro à
    # chaque clic sur "Indexer les documents".
    vectorstore = Chroma.from_documents(
        chunks, embeddings, collection_name=f"session_{uuid.uuid4().hex}"
    )

    return vectorstore, len(chunks)
