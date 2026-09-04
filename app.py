"""
TP - Système RAG Local (Clone de NotebookLM)
Master 1 - Introduction et Pratique de l'IA

Auteur : Ratovonandrasana Aina Ny Antsa (ETU002754)

Application Streamlit qui permet de charger des documents (PDF, TXT, MD)
et de poser des questions dessus, selon deux modes :
  - Recherche Sémantique pure : renvoie les extraits bruts les plus
    proches de la question, sans passer par un LLM (mode "audit" de la
    base vectorielle, donc pas de filtrage de pertinence ici)
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

# Découpage (Étape 2) : 1000 caractères pour garder un paragraphe complet
# avec son contexte, et 150 de chevauchement (15%) pour ne pas couper une
# idée pile à la frontière entre deux chunks.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Filtrage de pertinence (utilisé uniquement en mode RAG, voir plus bas).
# On avait d'abord essayé un seuil de distance fixe pour écarter les
# chunks hors sujet, mais sur nos documents les scores ne montrent pas de
# coupure nette : un seuil fixe filtrait soit trop, soit pas assez selon
# la question. À la place, un chunk n'est gardé que s'il est nettement
# plus proche que la moyenne du lot récupéré. 0.15 veut dire "au moins 15%
# plus proche que la moyenne". À ajuster selon vos documents.
RELEVANCE_MARGIN = 0.15


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


@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    """Charge le modèle d'embedding une seule fois pour toute la session serveur.

    Avant, chaque clic sur "Indexer" recréait un HuggingFaceEmbeddings
    depuis zéro (rechargement du modèle en mémoire à chaque fois), ce qui
    expliquait une bonne partie de la lenteur ressentie. st.cache_resource
    garde l'objet en mémoire entre les réindexations tant que le serveur
    Streamlit tourne : il n'est chargé qu'une seule fois.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


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

        try:
            if suffix == ".pdf":
                docs = PyMuPDFLoader(tmp_path).load()
                # PyMuPDF charge un Document par page. On les fusionne en
                # un seul texte par fichier avant le découpage, sinon le
                # chevauchement (overlap) ne peut jamais franchir une
                # frontière de page, et on perd le contexte d'un passage
                # coupé pile à un saut de page.
                full_text = "\n".join(doc.page_content for doc in docs)
            else:  # .txt, .md
                full_text = read_text_with_fallback(tmp_path)
        finally:
            os.remove(tmp_path)

        all_docs.append(
            Document(page_content=full_text, metadata={"source": uploaded_file.name})
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
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


def retrieve_candidates(vectorstore, query: str, k: int = 4):
    """Récupère les k meilleurs chunks pour la question, dédupliqués.

    Ne filtre PAS par pertinence : montre ce qui est le plus proche, quel
    que soit son intérêt réel. C'est voulu — ça sert de base au mode
    "Recherche sémantique pure" (Étape 3), qui doit permettre d'auditer la
    base vectorielle telle qu'elle est, distance affichée, et laisser
    l'utilisateur juger lui-même de la pertinence plutôt que de la cacher.

    Retourne aussi les scores du lot élargi (k*5, avant dédoublonnage) :
    ils servent ensuite (filter_if_relevant) à estimer si les meilleurs
    résultats se détachent vraiment du reste ou pas.
    """
    pool = vectorstore.similarity_search_with_score(query, k=k * 5)
    pool_scores = [score for _, score in pool]

    seen = set()
    unique = []
    for doc, score in pool:
        text = doc.page_content.strip()
        if text and text not in seen:
            seen.add(text)
            unique.append((doc, score))
        if len(unique) >= k:
            break

    return unique, pool_scores


def compute_relevance_ceiling(pool_scores, margin: float = RELEVANCE_MARGIN):
    """Calcule le seuil de distance en dessous duquel un chunk est jugé pertinent.

    Séparé de filter_if_relevant pour pouvoir aussi afficher ce chiffre à
    l'utilisateur (calibrage de RELEVANCE_MARGIN) sans dupliquer le calcul.
    """
    if not pool_scores:
        return None
    avg_score = sum(pool_scores) / len(pool_scores)
    return avg_score * (1 - margin)


def filter_if_relevant(results_with_scores, pool_scores, margin: float = RELEVANCE_MARGIN):
    """Ne garde les résultats que si au moins un chunk se détache nettement du lot.

    Utilisé uniquement par le mode RAG complet (Étape 4) : là, un LLM va
    générer une réponse, donc il faut décider si on a vraiment de quoi
    répondre avant de l'appeler — contrairement au mode recherche pure,
    qui montre toujours le top-k brut.

    On avait essayé un seuil de distance fixe, mais sans coupure nette
    dans les scores ça ne filtrait rien d'utile (une question totalement
    hors sujet obtenait quand même une réponse du LLM, qui comblait avec
    ses connaissances générales malgré la consigne du prompt). À la place
    on compare le meilleur résultat à la moyenne du lot élargi : s'il
    n'est pas nettement plus proche (au moins `margin` de mieux), on
    considère qu'aucun chunk n'apporte plus qu'un chunk pris au hasard.
    """
    ceiling = compute_relevance_ceiling(pool_scores, margin)
    if ceiling is None or not results_with_scores:
        return []
    return [(doc, score) for doc, score in results_with_scores if score <= ceiling]


def format_semantic_results(results_with_scores):
    """Formate les chunks retrouvés en extraits bruts + source + score (Étape 3)."""
    if not results_with_scores:
        return "Aucun extrait trouvé dans les documents indexés."

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


def generate_rag_answer(context: str, question: str) -> str:
    """Construit le prompt final et interroge le modèle local Ollama."""
    final_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    try:
        llm = Ollama(model=OLLAMA_MODEL, temperature=0)
        return llm.invoke(final_prompt)
    except Exception as e:
        return (
            "Impossible de contacter Ollama. Vérifie qu'il "
            f"tourne bien en local avec le modèle `{OLLAMA_MODEL}` "
            f"chargé (`ollama run {OLLAMA_MODEL}`).\n\nDétail : {e}"
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
            new_filenames = sorted(f.name for f in unique_files)
            previous_filenames = sorted(st.session_state.get("indexed_filenames", []))
            # Si on reclique sur "Indexer" avec exactement les mêmes
            # fichiers (par exemple juste pour rafraîchir), le contenu
            # indexé n'a pas changé : pas besoin de vider la conversation.
            same_documents_as_before = new_filenames == previous_filenames

            with st.spinner("Extraction, découpage et vectorisation en cours..."):
                vectorstore, nb_chunks = process_uploaded_files(unique_files)
                st.session_state.vectorstore = vectorstore
                st.session_state.nb_chunks = nb_chunks
                # Liste des fichiers vraiment indexés, affichée telle
                # quelle dans la sidebar pour qu'on sache toujours ce qui
                # est réellement dans la base, sans avoir à le deviner.
                st.session_state.indexed_filenames = new_filenames

            if not same_documents_as_before:
                # Nouvelle base = nouvelle conversation : les anciens
                # échanges concernaient l'ancien contenu, les garder
                # affichés prêterait à confusion.
                st.session_state.messages = []

            suffix = "" if same_documents_as_before else " Nouvelle conversation."
            st.success(
                f"{len(unique_files)} document(s) indexé(s) "
                f"en {nb_chunks} fragments.{suffix}"
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

# Réaffiche l'historique existant à chaque rechargement de la page.
# Chaque message peut porter un "caption" (petit avertissement) et/ou un
# expander (extraits) en plus de son contenu principal : avant, ces
# éléments n'étaient affichés qu'au moment du direct et disparaissaient
# dès le message suivant, car ils n'étaient jamais sauvegardés. On les
# stocke maintenant dans le message lui-même pour qu'ils survivent au
# réaffichage de l'historique.
for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "📚"
    with st.chat_message(message["role"], avatar=avatar):
        if message.get("caption"):
            st.caption(message["caption"])
        st.markdown(message["content"])
        if message.get("expander_label"):
            with st.expander(message["expander_label"]):
                st.markdown(message["expander_content"])

if prompt := st.chat_input("Posez une question sur vos documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # On construit la réponse dans un dict pour pouvoir à la fois
    # l'afficher tout de suite ET la sauvegarder telle quelle dans
    # l'historique (voir le commentaire ci-dessus).
    assistant_message = {"role": "assistant", "content": ""}

    with st.chat_message("assistant", avatar="📚"):
        if "vectorstore" not in st.session_state:
            assistant_message["content"] = "Veuillez d'abord indexer au moins un document."
            st.markdown(assistant_message["content"])

        else:
            # Le même retrieval brut sert de base aux deux modes : ça
            # évite les incohérences (une question rejetée en recherche
            # mais acceptée en RAG, ou l'inverse).
            results_with_scores, pool_scores = retrieve_candidates(
                st.session_state.vectorstore, prompt, k=RETRIEVAL_K
            )

            if not use_llm:
                # Mode Recherche Sémantique pure (Étape 3) : audit de la
                # base, donc toujours les chunks les plus proches, bruts,
                # avec leur distance affichée. Pas de filtrage ici — mais
                # on prévient si rien ne se détache vraiment du lot, sans
                # rien cacher pour autant.
                if not filter_if_relevant(results_with_scores, pool_scores):
                    assistant_message["caption"] = (
                        "⚠️ Aucun extrait ne se détache nettement de la "
                        "moyenne du lot récupéré — ces résultats sont "
                        "peut-être hors sujet, mais on les affiche quand "
                        "même : c'est le principe de ce mode."
                    )
                    st.caption(assistant_message["caption"])
                assistant_message["content"] = format_semantic_results(results_with_scores)
                st.markdown(assistant_message["content"])

            else:
                # Mode RAG complet (Étape 4) : avant d'appeler le LLM, on
                # vérifie qu'au moins un chunk se détache vraiment du lot.
                relevant_results = filter_if_relevant(results_with_scores, pool_scores)

                if not relevant_results:
                    assistant_message["content"] = (
                        "Je ne trouve rien d'assez pertinent dans les "
                        "documents indexés pour répondre à cette question. "
                        "Essayez de la reformuler ou vérifiez qu'elle porte "
                        "bien sur le contenu chargé."
                    )
                    st.markdown(assistant_message["content"])

                    ceiling = compute_relevance_ceiling(pool_scores)
                    calibration_note = (
                        f"*(Calibrage : distance moyenne du lot = `{sum(pool_scores) / len(pool_scores):.3f}`, "
                        f"seuil de pertinence = `{ceiling:.3f}` avec RELEVANCE_MARGIN={RELEVANCE_MARGIN}. "
                        "Si ce rejet semble injustifié, augmentez RELEVANCE_MARGIN dans le code.)*"
                    )
                    assistant_message["expander_label"] = (
                        "Voir les extraits les plus proches trouvés (jugés pas assez pertinents)"
                    )
                    assistant_message["expander_content"] = (
                        format_semantic_results(results_with_scores) + "\n\n" + calibration_note
                    )
                    with st.expander(assistant_message["expander_label"]):
                        st.markdown(assistant_message["expander_content"])

                else:
                    results = [doc for doc, _ in relevant_results]
                    context = build_context(results)

                    with st.spinner(f"Génération de la réponse avec {OLLAMA_MODEL}..."):
                        assistant_message["content"] = generate_rag_answer(context, prompt)

                    st.markdown(assistant_message["content"])

                    # Transparence demandée par l'énoncé : on peut vérifier
                    # les extraits qui ont servi de contexte à la réponse.
                    assistant_message["expander_label"] = "Voir les extraits utilisés comme contexte"
                    assistant_message["expander_content"] = format_semantic_results(relevant_results)
                    with st.expander(assistant_message["expander_label"]):
                        st.markdown(assistant_message["expander_content"])

    st.session_state.messages.append(assistant_message)