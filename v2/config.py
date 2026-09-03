"""
Configuration centralisée du système RAG.

Regrouper ces constantes ici évite d'avoir des "nombres magiques" ou des
chaînes dupliquées dans plusieurs modules (ingestion, retrieval, app...).
Pour changer de modèle d'embedding ou de LLM, un seul fichier à modifier.
"""

from langchain_core.prompts import PromptTemplate

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5-coder:1.5b"  # doit correspondre au modèle pull via `ollama pull`
RETRIEVAL_K = 4  # nombre de chunks récupérés par question

# Justification du chunking (demandée par l'énoncé) :
# chunk_size=1000 pour garder un paragraphe entier avec son contexte,
# chunk_overlap=150 (15%) pour ne pas couper une idée importante pile
# à la frontière entre deux chunks.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

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
