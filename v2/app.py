"""
TP - Système RAG Local (Clone de NotebookLM)
Master 1 - Introduction et Pratique de l'IA

Auteur : Ratovonandrasana Aina Ny Antsa (ETU002754)

"""

import streamlit as st

from config import RETRIEVAL_K
from ingestion import process_uploaded_files
from rag_chain import generate_rag_answer
from retrieval import build_context, format_semantic_results, retrieve_unique


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

            with st.spinner("Génération de la réponse..."):
                response = generate_rag_answer(context, prompt)

            st.markdown(response)

            # Transparence demandée par l'énoncé : on peut vérifier les
            # extraits qui ont servi de contexte à la réponse.
            with st.expander("Voir les extraits utilisés comme contexte"):
                st.markdown(format_semantic_results(results_with_scores))

    st.session_state.messages.append({"role": "assistant", "content": response})
