"""
Génération contrainte au contexte via un LLM local (Ollama).

Étape 4 du sujet. Isoler l'appel au LLM ici permet, par exemple, de le
remplacer plus tard par un autre backend local sans toucher à l'UI ni au
pipeline de retrieval.
"""

from langchain_community.llms import Ollama

from config import OLLAMA_MODEL, RAG_PROMPT_TEMPLATE


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
