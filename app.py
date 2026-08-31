"""
TP - Système RAG Local (Clone de NotebookLM)
Master 1 - Introduction et Pratique de l'IA

Auteur : Ratovonandrasana Aina Ny Antsa (ETU002754)

Application Streamlit qui permet de charger des documents (PDF, TXT, MD)
et de poser des questions dessus, selon deux modes :
  - Recherche Sémantique pure : renvoie les extraits bruts les plus
    proches de la question, sans passer par un LLM
  - Assistant RAG complet : récupère les extraits pertinents puis génère
    une réponse avec un modèle local (Ollama), en restant contraint au
    contexte fourni

Aucun appel à une API externe : tout tourne en local (embeddings + LLM)
pour respecter la contrainte de confidentialité du sujet.
"""

import os
import tempfile
import uuid

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
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
OLLAMA_MODEL = "qwen2.5-coder:1.5b"  # doit correspondre au modèle pull via `ollama pull`
RETRIEVAL_K = 4  # nombre de chunks récupérés par question


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


def process_uploaded_files(uploaded_files):
    """Extrait, découpe et vectorise les fichiers uploadés. Retourne le vectorstore."""
    all_docs = []

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()

        # Les loaders LangChain veulent un chemin sur disque, donc on
        # écrit le contenu uploadé (en mémoire) dans un fichier temporaire.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

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

        all_docs.append(
            Document(page_content=full_text, metadata={"source": uploaded_file.name})
        )

        os.remove(tmp_path)

    # Justification du chunking (demandée par l'énoncé) :
    # chunk_size=1000 pour garder un paragraphe entier avec son contexte,
    # chunk_overlap=150 (15%) pour ne pas couper une idée importante pile
    # à la frontière entre deux chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(all_docs)

    # Certains PDF contiennent le même passage deux fois dans leur texte
    # brut (mise en page, effets de style...), ce qui donne des chunks
    # identiques. On les déduplique au texte exact.
    seen_texts = set()
    unique_chunks = []
    for chunk in chunks:
        if chunk.page_content not in seen_texts:
            seen_texts.add(chunk.page_content)
            unique_chunks.append(chunk)
    chunks = unique_chunks

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


def format_semantic_results(results_with_scores):
    """Formate les chunks retrouvés en extraits bruts + source + score (Étape 3)."""
    if not results_with_scores:
        return "Aucun extrait pertinent trouvé dans les documents indexés."

    parts = []
    for i, (doc, score) in enumerate(results_with_scores, start=1):
        source = doc.metadata.get("source", "source inconnue")
        # Score = distance : plus petit veut dire plus proche de la question
        parts.append(
            f"**Extrait {i}** — *source : {source}* "
            f"(distance : `{score:.3f}`)\n\n> {doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def build_context(results):
    """Concatène les chunks récupérés en un bloc de contexte pour le LLM (Étape 4)."""
    return "\n\n".join(
        f"[Source : {doc.metadata.get('source', 'inconnue')}]\n{doc.page_content}"
        for doc in results
    )


def retrieve_unique(vectorstore, query: str, k: int = 4, max_distance: float = 999):
    """Récupère les k meilleurs chunks, dédupliqués.

    max_distance=999 désactive le filtrage par seuil : on a testé un vrai
    seuil de pertinence, mais les scores mesurés sur nos documents ne
    montrent pas de coupure nette entre pertinent et hors-sujet (écart
    d'à peine 0.01 dans certains cas), donc un seuil fixe ferait plus de
    mal que de bien. On garde k=4 fixe, plus simple et plus fiable.
    """
    results = vectorstore.similarity_search_with_score(query, k=k * 5)

    seen = set()
    unique = []
    for doc, score in results:
        if score > max_distance:
            continue
        text = doc.page_content.strip()
        if text and text not in seen:
            seen.add(text)
            unique.append((doc, score))
        if len(unique) >= k:
            break
    return unique


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
            # Si on réindexe avec un uploader vide, on vide aussi la base
            # et l'historique précédents : sinon le chat continuerait de
            # répondre avec un contenu que l'utilisateur croit supprimé.
            st.session_state.pop("vectorstore", None)
            st.session_state.pop("nb_chunks", None)
            st.session_state.pop("indexed_filenames", None)
            st.session_state.messages = []
        else:
            # Si le même fichier a été ajouté deux fois par erreur dans
            # l'uploader, on ne l'indexe qu'une seule fois.
            unique_files = list({f.name: f for f in uploaded_files}.values())

            with st.spinner("Extraction, découpage et vectorisation en cours..."):
                vectorstore, nb_chunks = process_uploaded_files(unique_files)
                st.session_state.vectorstore = vectorstore
                st.session_state.nb_chunks = nb_chunks
                # Liste des fichiers vraiment indexés, affichée telle
                # quelle dans la sidebar pour qu'on sache toujours ce qui
                # est réellement dans la base, sans avoir à le deviner.
                st.session_state.indexed_filenames = [f.name for f in unique_files]
                # Nouvelle base = nouvelle conversation : les anciens
                # échanges concernaient l'ancien contenu, les garder
                # affichés prêterait à confusion.
                st.session_state.messages = []
            st.success(
                f"{len(unique_files)} document(s) indexé(s) "
                f"en {nb_chunks} fragments. Nouvelle conversation."
            )

    if "vectorstore" in st.session_state:
        st.caption(f"Base indexée : {st.session_state.nb_chunks} fragments")
        filenames = st.session_state.get("indexed_filenames", [])
        if filenames:
            st.caption("Fichiers réellement indexés : " + ", ".join(filenames))

    st.divider()

    use_llm = st.toggle("Activer l'Assistant RAG (LLM)", value=False)

    st.divider()
    mode_label = "Assistant RAG complet" if use_llm else "Recherche sémantique pure"
    st.caption(f"Mode actuel : {mode_label}")


# ---------- ZONE PRINCIPALE (CHAT) ----------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Réaffiche l'historique existant à chaque rechargement de la page
for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🗂️"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Posez une question sur vos documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🗂️"):
        if "vectorstore" not in st.session_state:
            response = "Veuillez d'abord indexer au moins un document."
            st.markdown(response)

        elif not use_llm:
            # Mode Recherche Sémantique pure : pas d'appel LLM, on renvoie
            # directement les chunks les plus proches de la question.
            results_with_scores = retrieve_unique(
                st.session_state.vectorstore, prompt, k=RETRIEVAL_K
            )
            response = format_semantic_results(results_with_scores)
            st.markdown(response)

        else:
            # Mode RAG complet : recherche + génération contrainte au contexte
            results_with_scores = retrieve_unique(
                st.session_state.vectorstore, prompt, k=RETRIEVAL_K
            )
            results = [doc for doc, _ in results_with_scores]
            context = build_context(results)
            final_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=prompt)

            with st.spinner(f"Génération de la réponse avec {OLLAMA_MODEL}..."):
                try:
                    llm = Ollama(model=OLLAMA_MODEL, temperature=0)
                    response = llm.invoke(final_prompt)
                except Exception as e:
                    response = (
                        "Impossible de contacter Ollama. Vérifie qu'il "
                        f"tourne bien en local avec le modèle `{OLLAMA_MODEL}` "
                        f"chargé (`ollama run {OLLAMA_MODEL}`).\n\nDétail : {e}"
                    )

            st.markdown(response)

            # Transparence demandée par l'énoncé : on peut vérifier les
            # extraits qui ont servi de contexte à la réponse.
            with st.expander("Voir les extraits utilisés comme contexte"):
                st.markdown(format_semantic_results(results_with_scores))

    st.session_state.messages.append({"role": "assistant", "content": response})