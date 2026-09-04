# Assistant Documentaire Local — v2 (refactorisée)

Système RAG local (clone de NotebookLM) — Master 1, module *Introduction
et Pratique de l'IA*.

Auteur : Ratovonandrasana Aina Ny Antsa (ETU002754)

## Structure du projet

La v1 tenait dans un seul fichier `app.py`. En v2, la logique métier est
séparée de l'interface Streamlit, module par module :

| Fichier            | Rôle                                                             |
|---------------------|-------------------------------------------------------------------|
| `config.py`         | Constantes (modèles, taille des chunks, seuils) et prompt RAG    |
| `loaders.py`        | Lecture des fichiers sources (PDF, TXT/MD avec détection d'encodage) |
| `ingestion.py`      | Pipeline : découpage en chunks + vectorisation (Chroma)          |
| `retrieval.py`      | Recherche vectorielle + filtrage de pertinence                   |
| `generation.py`     | Construction du contexte + appel au LLM local (Ollama)           |
| `formatting.py`     | Mise en forme des extraits pour l'affichage                      |
| `app.py`            | Interface Streamlit uniquement (widgets, session_state, chat)    |

Chaque module est testable indépendamment de Streamlit, à l'exception de
`ingestion.py` qui utilise `st.cache_resource` pour éviter de recharger
le modèle d'embedding à chaque interaction.

## Lancer le projet

```bash
pip install -r requirements.txt
ollama pull qwen2.5-coder:1.5b   # une seule fois
ollama serve                     # si le service n'est pas déjà lancé
streamlit run app.py
```

## Fonctionnement

1. **Indexation** : les documents chargés (PDF, TXT, MD) sont extraits,
   découpés en fragments (chunks) puis vectorisés localement
   (`sentence-transformers/all-MiniLM-L6-v2`), sans appel à une API
   externe.
2. **Recherche sémantique pure** : renvoie les extraits bruts les plus
   proches de la question, avec leur score de distance — pour auditer la
   base vectorielle sans filtrage de pertinence.
3. **Assistant RAG complet** : ne garde que les extraits qui se
   détachent nettement du lot récupéré, puis génère une réponse avec un
   modèle local (Ollama), strictement contrainte au contexte fourni.
