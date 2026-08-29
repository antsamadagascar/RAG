import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# ÉTAPE 1 : Squelette de l'interface
# ÉTAPE 2 : Pipeline d'ingestion (extraction, chunking, vectorisation)
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "./chroma_db"


def process_uploaded_files(uploaded_files):
    """Extrait, découpe et vectorise les fichiers uploadés. Retourne le vectorstore."""
    all_docs = []

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()

        # Les loaders LangChain attendent un chemin disque : on écrit
        # temporairement le contenu uploadé (en mémoire) sur le disque.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        if suffix == ".pdf":
            loader = PyMuPDFLoader(tmp_path)
        else:  # .txt, .md
            loader = TextLoader(tmp_path, encoding="utf-8")

        docs = loader.load()

        # On remplace le chemin temporaire par le vrai nom du fichier
        # dans les métadonnées, pour un affichage propre des sources.
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        all_docs.extend(docs)
        os.remove(tmp_path)

    # Chunking : chunk_size=1000 pour garder un paragraphe avec son
    # contexte, chunk_overlap=150 (15%) pour ne pas couper une idée
    # importante pile à la frontière entre deux chunks.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=PERSIST_DIR
    )

    return vectorstore, len(chunks)

st.set_page_config(page_title="RAG Local - Clone NotebookLM", layout="wide")

st.title("📚 Assistant RAG Local")

# ---------- BARRE LATÉRALE ----------
with st.sidebar:
    st.header("Gestion des documents")

    uploaded_files = st.file_uploader(
        "Charger vos documents (PDF, TXT, MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    index_button = st.button("📥 Indexer les documents")

    if index_button:
        if not uploaded_files:
            st.warning("Veuillez d'abord charger au moins un document.")
        else:
            with st.spinner("Extraction, découpage et vectorisation en cours..."):
                vectorstore, nb_chunks = process_uploaded_files(uploaded_files)
                st.session_state.vectorstore = vectorstore
                st.session_state.nb_chunks = nb_chunks
            st.success(
                f"{len(uploaded_files)} document(s) indexé(s) "
                f"en {nb_chunks} fragments."
            )

    if "vectorstore" in st.session_state:
        st.caption(f"✅ Base indexée : {st.session_state.nb_chunks} fragments")

    st.divider()

    use_llm = st.toggle("Activer l'Assistant RAG (LLM)", value=False)

    st.divider()
    mode_label = "Assistant RAG complet" if use_llm else "Recherche sémantique pure"
    st.caption(f"Mode actuel : {mode_label}")

# ---------- ZONE PRINCIPALE (CHAT) ----------

# Initialisation de l'historique de conversation (persiste tant que l'app tourne)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique existant
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie utilisateur
if prompt := st.chat_input("Posez une question sur vos documents..."):
    # Ajout du message utilisateur à l'historique
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Placeholder de réponse : sera remplacé par la vraie logique
    # de recherche (Étape 3) et de génération LLM (Étape 4)
    with st.chat_message("assistant"):
        response = (
            "⚠️ Backend non branché pour le moment "
            "(Étape 1 : squelette d'interface uniquement)."
        )
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})