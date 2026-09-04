"""
Assistant Documentaire Local — Interface Streamlit.

Auteur : Ratovonandrasana Aina Ny Antsa (ETU002754)

Point d'entrée de l'application. Ce fichier ne contient QUE la logique
d'interface (widgets, session_state, affichage) : la logique métier vit
dans config.py, loaders.py, ingestion.py, retrieval.py, generation.py et
formatting.py.
"""

import streamlit as st

from config import OLLAMA_MODEL, RETRIEVAL_K
from formatting import format_semantic_results
from generation import build_context, generate_rag_answer
from ingestion import process_uploaded_files
from retrieval import compute_relevance_ceiling, filter_if_relevant, retrieve_candidates

st.set_page_config(page_title="Assistant Documentaire Local", layout="wide")


# ============================================================
# BARRE LATÉRALE — gestion des documents et du mode
# ============================================================

def handle_indexing(uploaded_files) -> None:
    """Traite le clic sur "Indexer les documents"."""
    if not uploaded_files:
        st.warning("Veuillez d'abord charger au moins un document.")
        # Si on réindexe avec un uploader vide, on vide aussi la base et
        # l'historique précédents : sinon le chat continuerait de
        # répondre avec un contenu que l'utilisateur croit supprimé.
        st.session_state.pop("vectorstore", None)
        st.session_state.pop("nb_chunks", None)
        st.session_state.pop("indexed_filenames", None)
        st.session_state.messages = []
        return

    # Si le même fichier a été ajouté deux fois par erreur dans
    # l'uploader, on ne l'indexe qu'une seule fois.
    unique_files = list({f.name: f for f in uploaded_files}.values())
    new_filenames = sorted(f.name for f in unique_files)
    previous_filenames = sorted(st.session_state.get("indexed_filenames", []))
    # Si on reclique sur "Indexer" avec exactement les mêmes fichiers
    # (par exemple juste pour rafraîchir), le contenu indexé n'a pas
    # changé : pas besoin de vider la conversation.
    same_documents_as_before = new_filenames == previous_filenames

    with st.spinner("Extraction, découpage et vectorisation en cours..."):
        vectorstore, nb_chunks = process_uploaded_files(unique_files)
        st.session_state.vectorstore = vectorstore
        st.session_state.nb_chunks = nb_chunks
        # Liste des fichiers vraiment indexés, affichée telle quelle dans
        # la sidebar pour qu'on sache toujours ce qui est réellement dans
        # la base, sans avoir à le deviner.
        st.session_state.indexed_filenames = new_filenames

    if not same_documents_as_before:
        # Nouvelle base = nouvelle conversation : les anciens échanges
        # concernaient l'ancien contenu, les garder affichés prêterait à
        # confusion.
        st.session_state.messages = []

    suffix = "" if same_documents_as_before else " Nouvelle conversation."
    st.success(f"{len(unique_files)} document(s) indexé(s) en {nb_chunks} fragments.{suffix}")


def render_sidebar() -> bool:
    """Affiche la barre latérale et retourne True si le mode RAG est activé."""
    with st.sidebar:
        st.header("Gestion des documents")

        uploaded_files = st.file_uploader(
            "Charger vos documents (PDF, TXT, MD)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        if st.button("Indexer les documents"):
            handle_indexing(uploaded_files)

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

    return use_llm


# ============================================================
# ZONE DE CHAT
# ============================================================

def render_chat_history() -> None:
    """Réaffiche l'historique existant à chaque rechargement de la page."""
    for message in st.session_state.messages:
        avatar = "🧑" if message["role"] == "user" else "📚"
        with st.chat_message(message["role"], avatar=avatar):
            if message.get("caption"):
                st.caption(message["caption"])
            st.markdown(message["content"])
            if message.get("expander_label"):
                with st.expander(message["expander_label"]):
                    st.markdown(message["expander_content"])


def answer_semantic_mode(prompt: str) -> dict:
    """Construit la réponse pour le mode Recherche Sémantique pure."""
    results_with_scores, pool_scores = retrieve_candidates(
        st.session_state.vectorstore, prompt, k=RETRIEVAL_K
    )
    message = {"role": "assistant", "content": ""}

    # Pas de filtrage ici — mais on prévient si rien ne se détache
    # vraiment du lot, sans rien cacher pour autant : c'est le principe
    # de ce mode (audit brut de la base vectorielle).
    if not filter_if_relevant(results_with_scores, pool_scores):
        message["caption"] = (
            "⚠️ Aucun extrait ne se détache nettement de la moyenne du "
            "lot récupéré — ces résultats sont peut-être hors sujet, "
            "mais on les affiche quand même : c'est le principe de ce mode."
        )

    message["content"] = format_semantic_results(results_with_scores)
    return message


def answer_rag_mode(prompt: str) -> dict:
    """Construit la réponse pour le mode Assistant RAG complet."""
    results_with_scores, pool_scores = retrieve_candidates(
        st.session_state.vectorstore, prompt, k=RETRIEVAL_K
    )
    # Avant d'appeler le LLM, on vérifie qu'au moins un chunk se détache
    # vraiment du lot.
    relevant_results = filter_if_relevant(results_with_scores, pool_scores)

    message = {"role": "assistant", "content": ""}

    if not relevant_results:
        message["content"] = (
            "Je ne trouve rien d'assez pertinent dans les documents "
            "indexés pour répondre à cette question. Essayez de la "
            "reformuler ou vérifiez qu'elle porte bien sur le contenu "
            "chargé."
        )
        # Petit indicateur pour comprendre le rejet (distance moyenne du
        # lot vs seuil retenu), sans jargon de code.
        ceiling = compute_relevance_ceiling(pool_scores)
        calibration_note = (
            f"*(Distance moyenne des extraits trouvés : "
            f"`{sum(pool_scores) / len(pool_scores):.3f}`, "
            f"seuil de pertinence : `{ceiling:.3f}`)*"
        )
        message["expander_label"] = (
            "Voir les extraits les plus proches trouvés (jugés pas assez pertinents)"
        )
        message["expander_content"] = (
            format_semantic_results(results_with_scores) + "\n\n" + calibration_note
        )
        return message

    results = [doc for doc, _ in relevant_results]
    context = build_context(results)

    with st.spinner(f"Génération de la réponse avec {OLLAMA_MODEL}..."):
        message["content"] = generate_rag_answer(context, prompt)

    # Transparence demandée par l'énoncé : on peut vérifier les extraits
    # qui ont servi de contexte à la réponse.
    message["expander_label"] = "Voir les extraits utilisés comme contexte"
    message["expander_content"] = format_semantic_results(relevant_results)
    return message


def handle_new_question(prompt: str, use_llm: bool) -> None:
    """Traite une nouvelle question saisie dans le chat."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="📚"):
        if "vectorstore" not in st.session_state:
            message = {
                "role": "assistant",
                "content": "Veuillez d'abord indexer au moins un document.",
            }
        elif use_llm:
            message = answer_rag_mode(prompt)
        else:
            message = answer_semantic_mode(prompt)

        if message.get("caption"):
            st.caption(message["caption"])
        st.markdown(message["content"])
        if message.get("expander_label"):
            with st.expander(message["expander_label"]):
                st.markdown(message["expander_content"])

    st.session_state.messages.append(message)


# ============================================================
# POINT D'ENTRÉE
# ============================================================

def main() -> None:
    st.title("Assistant Documentaire Local")
    st.caption("Système RAG local — Ratovonandrasana Aina Ny Antsa (ETU002754)")

    use_llm = render_sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    render_chat_history()

    if prompt := st.chat_input("Posez une question sur vos documents..."):
        handle_new_question(prompt, use_llm)


if __name__ == "__main__":
    main()
