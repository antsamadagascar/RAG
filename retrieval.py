"""
Recherche dans la base vectorielle et mise en forme des résultats.

Étape 3 (recherche sémantique pure) et brique de base de l'Étape 4
(le RAG complet réutilise retrieve_unique + build_context).
"""


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


def build_context(results):
    """Concatène les chunks récupérés en un bloc de contexte pour le LLM (Étape 4)."""
    return "\n\n".join(
        f"[Source : {doc.metadata.get('source', 'inconnue')}]\n{doc.page_content}"
        for doc in results
    )


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
