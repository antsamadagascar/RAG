"""
Construction du contexte et génération de la réponse via le LLM local (Ollama).
"""

from langchain_community.llms import Ollama
from langchain_core.documents import Document

from config import OLLAMA_MODEL, RAG_PROMPT_TEMPLATE


def build_context(results: list[Document]) -> str:
    """Concatène les chunks récupérés en un bloc de contexte pour le LLM."""
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
