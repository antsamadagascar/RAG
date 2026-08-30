"""
TP - Système RAG Local (Clone de NotebookLM)
Master 1 - Introduction et Pratique de l'IA

Auteur : Ratovonandrasana Aina Ny Antsa (ETU002754)

Application Streamlit permettant d'indexer des documents (PDF, TXT, MD)
et d'interagir avec leur contenu selon deux modes :
  - Recherche Sémantique pure : extraits bruts, sans appel à un LLM
  - Assistant RAG complet : génération contrainte au contexte, via un
    modèle local servi par Ollama

Aucun appel à une API externe : l'ensemble du traitement (embeddings,
recherche vectorielle, génération) s'exécute en local pour garantir la
confidentialité des documents chargés.
"""

import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# ÉTAPE 1 : Squelette de l'interface
# ÉTAPE 2 : Pipeline d'ingestion (extraction, chunking, vectorisation)
# ÉTAPE 3 : Mode Recherche Sémantique pure
# ÉTAPE 4 : Mode RAG complet (retrieval + génération contrainte)
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5-coder:1.5b"  # doit correspondre au modèle chargé via `ollama pull`
RETRIEVAL_K = 4  # nombre de chunks les plus proches récupérés par question

RAG_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Tu es un assistant qui répond UNIQUEMENT à partir du contexte "
        "fourni ci-dessous. Réponds toujours en français, même si le "
        "contexte ou la question contient des mots dans une autre langue. "
        "Si la réponse ne s'y trouve pas, dis "
        "clairement que tu ne sais pas à partir des documents fournis. "
        "N'utilise AUCUNE connaissance extérieure au contexte.\n\n"
        "Contexte :\n{context}\n\n"
        "Question : {question}\n\n"
        "Réponse :"
    ),
)


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

        # Les PDF sont chargés page par page (un Document par page). On
        # fusionne tout le texte du fichier en UN SEUL Document avant le
        # chunking : sinon le chunk_overlap ne peut jamais chevaucher deux
        # pages, et un passage coupé pile à un saut de page (ex: un bloc de
        # code qui commence en haut de la page suivante) est irrémédiablement
        # séparé de son contexte.
        full_text = "\n".join(doc.page_content for doc in docs)
        all_docs.append(
            Document(page_content=full_text, metadata={"source": uploaded_file.name})
        )

        os.remove(tmp_path)

    # Chunking : chunk_size=1000 pour garder un paragraphe avec son
    # contexte, chunk_overlap=150 (15%) pour ne pas couper une idée
    # importante pile à la frontière entre deux chunks.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(all_docs)

    # Déduplique les chunks au texte strictement identique : certains PDF
    # contiennent le même passage deux fois dans leur flux de texte brut
    # (mise en page avec calques dupliqués, effets de style, etc.), ce qui
    # produit sinon des chunks parfaitement identiques dans la base.
    seen_texts = set()
    unique_chunks = []
    for chunk in chunks:
        if chunk.page_content not in seen_texts:
            seen_texts.add(chunk.page_content)
            unique_chunks.append(chunk)
    chunks = unique_chunks

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Chroma est utilisé en mémoire (sans persist_directory) : la base est
    # de toute façon reconstruite à chaque clic sur "Indexer les documents"
    # ou à chaque redémarrage de l'app (via st.session_state), donc une
    # persistance disque n'apporte rien ici. Ça évite aussi un problème de
    # verrouillage de fichiers sous Windows si on tentait de supprimer une
    # base précédente encore ouverte par le processus.
    vectorstore = Chroma.from_documents(chunks, embeddings)

    return vectorstore, len(chunks)


def format_semantic_results(results):
    """Formate les chunks retrouvés en extraits bruts + source (Étape 3)."""
    if not results:
        return "Aucun extrait pertinent trouvé dans les documents indexés."

    parts = []
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "source inconnue")
        parts.append(f"**Extrait {i}** — *source : {source}*\n\n> {doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_context(results):
    """Concatène les chunks récupérés en un bloc de contexte pour le LLM (Étape 4)."""
    return "\n\n".join(
        f"[Source : {doc.metadata.get('source', 'inconnue')}]\n{doc.page_content}"
        for doc in results
    )

st.set_page_config(page_title="Assistant Documentaire Local", layout="wide")

st.title("Assistant Documentaire Local")
st.caption("Système RAG local — Ratovonandrasana Aina Ny Antsa (ETU002754)")

# ---------- BARRE LATÉRALE ----------
with st.sidebar:
    st.header("Gestion des documents")

    uploaded_files = st.file_uploader(
        "Charger vos documents (PDF, TXT, MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    index_button = st.button("Indexer les documents")

    if index_button:
        if not uploaded_files:
            st.warning("Veuillez d'abord charger au moins un document.")
            # Si l'utilisateur a retiré tous les fichiers puis réindexé,
            # on vide aussi l'ancienne base et l'historique du chat en
            # mémoire : sinon les questions déjà affichées continueraient
            # de montrer des réponses fondées sur un contenu que
            # l'utilisateur pense pourtant avoir supprimé.
            st.session_state.pop("vectorstore", None)
            st.session_state.pop("nb_chunks", None)
            st.session_state.messages = []
        else:
            # Déduplique par nom de fichier : si le même fichier a été
            # ajouté deux fois dans l'uploader (par erreur), on ne veut
            # l'indexer qu'une seule fois pour éviter des chunks dupliqués.
            unique_files = list({f.name: f for f in uploaded_files}.values())

            with st.spinner("Extraction, découpage et vectorisation en cours..."):
                vectorstore, nb_chunks = process_uploaded_files(unique_files)
                st.session_state.vectorstore = vectorstore
                st.session_state.nb_chunks = nb_chunks
                # On efface aussi l'historique de conversation : les
                # anciens échanges faisaient référence à l'ancienne base
                # de documents. Les garder affichés après un changement
                # de base créerait une confusion entre une réponse
                # obsolète et une nouvelle réponse fondée sur les
                # documents actuellement indexés.
                st.session_state.messages = []
            st.success(
                f"{len(unique_files)} document(s) indexé(s) "
                f"en {nb_chunks} fragments. Nouvelle conversation."
            )

    if "vectorstore" in st.session_state:
        st.caption(f"Base indexée : {st.session_state.nb_chunks} fragments")

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
        if "vectorstore" not in st.session_state:
            response = "⚠️ Veuillez d'abord indexer au moins un document."
            st.markdown(response)

        elif not use_llm:
            # Mode Recherche Sémantique pure : aucun appel à un modèle
            # génératif, on retourne les chunks bruts les plus proches.
            results = st.session_state.vectorstore.similarity_search(prompt, k=RETRIEVAL_K)
            response = format_semantic_results(results)
            st.markdown(response)

        else:
            # Mode RAG complet : recherche + génération contrainte au contexte
            results = st.session_state.vectorstore.similarity_search(prompt, k=RETRIEVAL_K)
            context = build_context(results)
            final_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=prompt)

            with st.spinner(f"Génération de la réponse avec {OLLAMA_MODEL}..."):
                try:
                    llm = Ollama(model=OLLAMA_MODEL, temperature=0)
                    response = llm.invoke(final_prompt)
                except Exception as e:
                    response = (
                        "❌ Impossible de contacter Ollama. Vérifie qu'il "
                        f"tourne bien en local avec le modèle `{OLLAMA_MODEL}` "
                        f"chargé (`ollama run {OLLAMA_MODEL}`).\n\nDétail : {e}"
                    )

            st.markdown(response)

            # Transparence : on permet de vérifier les extraits utilisés
            with st.expander("📄 Voir les extraits utilisés comme contexte"):
                st.markdown(format_semantic_results(results))

    st.session_state.messages.append({"role": "assistant", "content": response})