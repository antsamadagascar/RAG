"""
Mise en forme des résultats pour l'affichage Streamlit.

Auteur : Ratovonandrasana Aina Ny Antsa (ETU002754)

"""

from langchain_core.documents import Document


def format_semantic_results(results_with_scores: list[tuple[Document, float]]) -> str:
    """Formate les chunks retrouvés en extraits bruts + source + score."""
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
