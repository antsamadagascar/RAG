import streamlit as st

# ============================================================
# ÉTAPE 1 : Squelette de l'interface (pas encore de logique métier)
# ============================================================

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
