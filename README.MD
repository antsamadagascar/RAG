# Assistant Documentaire Local

Système RAG local permettant d'interroger des documents à l'aide d'un modèle de langage local, sans dépendre d'une API externe.

Projet réalisé dans le cadre du Master 1, module *Introduction et Pratique de l'IA*.

**Auteur : Ratovonandrasana Aina Ny Antsa (ETU002754)**

## Structure du projet

```text
.
├── src/
│   ├── app.py
│   ├── config.py
│   ├── loaders.py
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── generation.py
│   └── formatting.py
│
├── data/
│   └── documents/
│       ├── rag_test_centre_formation.pdf
│       ├── rag_test_exemple.txt
│       └── rag_test_nouveau_contexte.md
│
├── test/
│   └── tests_complets_rag.md
│
├── requirements.txt
└── README.md
```

### Description des principaux fichiers

| Fichier                      | Rôle                                                                         |
| ---------------------------- | ---------------------------------------------------------------------------- |
| `src/app.py`                 | Interface utilisateur Streamlit et gestion du chat                           |
| `src/config.py`              | Configuration des modèles, paramètres et prompt RAG                          |
| `src/loaders.py`             | Chargement et lecture des documents PDF, TXT et Markdown                     |
| `src/ingestion.py`           | Découpage des documents en chunks et création des embeddings                 |
| `src/retrieval.py`           | Recherche des passages pertinents dans la base vectorielle                   |
| `src/generation.py`          | Construction du contexte et génération de la réponse avec Ollama             |
| `src/formatting.py`          | Mise en forme des extraits affichés dans l'interface                         |
| `data/documents/`            | Documents utilisés pour tester le système RAG                                |
| `test/tests_complets_rag.md` | Exemples de questions permettant de tester le RAG avec les documents fournis |

## Installation

Installer les dépendances Python :

```bash
pip install -r requirements.txt
```

Le projet utilise notamment un modèle d'embedding local :

```text
sentence-transformers/all-MiniLM-L6-v2
```

Aucune API externe n'est nécessaire pour l'indexation des documents.

## Configuration d'Ollama

Installer Ollama puis télécharger le modèle utilisé par le projet :

```bash
ollama pull qwen2.5-coder:1.5b
```

Démarrer ensuite le serveur Ollama s'il n'est pas déjà actif :

```bash
ollama serve
```

## Lancer le projet

Depuis la racine du projet, exécuter :

```bash
python -m streamlit run src/app.py
```

L'application Streamlit s'ouvre ensuite dans le navigateur.

## Documents de test

Trois documents sont fournis dans `data/documents/` :

* `rag_test_centre_formation.pdf`
* `rag_test_exemple.txt`
* `rag_test_nouveau_contexte.md`

Ils permettent de tester le chargement, l'indexation, la recherche sémantique et la génération de réponses.

Des exemples de questions sont disponibles dans :

```text
test/tests_complets_rag.md
```

Ce fichier peut être utilisé comme guide pour vérifier le comportement du système avec les différents documents.

## Fonctionnement

Le système fonctionne selon un pipeline RAG (*Retrieval-Augmented Generation*) :

1. **Chargement des documents**
   Les fichiers PDF, TXT et Markdown sont lus et leur contenu est extrait.

2. **Découpage**
   Les documents sont divisés en plusieurs fragments (*chunks*) afin de faciliter la recherche.

3. **Vectorisation**
   Chaque fragment est transformé en vecteur à l'aide du modèle `all-MiniLM-L6-v2`.

4. **Recherche sémantique**
   Lorsqu'une question est posée, les passages les plus proches sémantiquement sont recherchés dans la base vectorielle.

5. **Construction du contexte**
   Les passages pertinents sont regroupés afin de constituer le contexte transmis au modèle.

6. **Génération de la réponse**
   Le modèle local `qwen2.5-coder:1.5b`, exécuté avec Ollama, génère une réponse basée sur le contexte récupéré.

L'ensemble du traitement des documents, des embeddings et de la génération est réalisé localement.
