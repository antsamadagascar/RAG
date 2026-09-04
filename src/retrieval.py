"""
Recherche vectorielle et filtrage de pertinence.

Auteur : Ratovonandrasana Aina Ny Antsa (ETU002754)

"""

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import RELEVANCE_MARGIN, RETRIEVAL_K


def retrieve_candidates(
    vectorstore: Chroma, query: str, k: int = RETRIEVAL_K
) -> tuple[list[tuple[Document, float]], list[float]]:
    """Récupère les k meilleurs chunks pour la question, dédupliqués.

    Ne filtre pas par pertinence : sert de base au mode Recherche
    Sémantique pure, qui doit montrer les résultats bruts pour laisser
    l'utilisateur juger lui-même. Renvoie aussi les scores du lot élargi
    (k*5, avant dédoublonnage), réutilisés par filter_if_relevant.
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


def compute_relevance_ceiling(pool_scores: list[float], margin: float = RELEVANCE_MARGIN):
    """Calcule le seuil de distance en dessous duquel un chunk est jugé pertinent.

    Séparé de filter_if_relevant pour pouvoir aussi afficher ce chiffre à
    l'utilisateur (calibrage de RELEVANCE_MARGIN) sans dupliquer le calcul.
    """
    if not pool_scores:
        return None
    avg_score = sum(pool_scores) / len(pool_scores)
    return avg_score * (1 - margin)


def filter_if_relevant(
    results_with_scores: list[tuple[Document, float]],
    pool_scores: list[float],
    margin: float = RELEVANCE_MARGIN,
) -> list[tuple[Document, float]]:
    """Ne garde les résultats que si un chunk se détache nettement du lot.

    Réservé au mode RAG : contrairement au mode recherche pure, ici un
    LLM va générer une réponse, donc autant vérifier qu'on a vraiment de
    quoi répondre avant de l'appeler.
    """
    ceiling = compute_relevance_ceiling(pool_scores, margin)
    if ceiling is None or not results_with_scores:
        return []
    return [(doc, score) for doc, score in results_with_scores if score <= ceiling]
