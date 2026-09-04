"""
Configuration centralisée du système RAG.

Toutes les constantes ajustables (modèles, taille des chunks, seuils...)
sont regroupées ici pour éviter de les disperser dans le code métier.
"""

from langchain_core.prompts import PromptTemplate

# --- Modèles ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5-coder:1.5b"  # doit correspondre au modèle pull via `ollama pull`

# --- Retrieval ---
RETRIEVAL_K = 4  # nombre de chunks récupérés par question

# --- Découpage (chunking) ---
# 1000 caractères pour garder un paragraphe complet avec son contexte,
# et 150 de chevauchement (15%) pour ne pas couper une idée pile à la
# frontière entre deux chunks.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# --- Filtrage de pertinence (mode RAG uniquement) ---
# On a essayé un seuil de distance fixe au départ, mais les scores ne
# montrent pas de coupure nette entre pertinent et hors sujet sur nos
# documents. 0.15 = un chunk doit être au moins 15% plus proche que la
# moyenne du lot récupéré pour être gardé.
RELEVANCE_MARGIN = 0.15

# --- Prompt du mode RAG ---
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
