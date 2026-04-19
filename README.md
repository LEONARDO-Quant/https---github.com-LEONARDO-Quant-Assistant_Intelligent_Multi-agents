


# 🤖 Assistant Académique Multi-Agents (GenAI)

> **Projet réalisé dans le cadre du cours "Generative AI" à l'Université Paris 1 Panthéon-Sorbonne.**

Cet assistant intelligent est conçu pour accompagner les étudiants dans l'apprentissage de concepts complexes (Statistiques et Machine Learning). 
Il repose sur une architecture **multi-agents** orchestrée par **LangGraph**, permettant une sélection intelligente de l'expert le plus pertinent selon la requête.

-----

## 🛠️ Stack Technique

-----

## 🧠 Architecture du Système

Le cœur du projet est un graphe d'états (**LangGraph**) où un agent **Orchestrateur** analyse l'intention de l'utilisateur pour router la demande vers l'un des quatre agents spécialisés :

  * **AgentText** : Expert en analyse de documents. Il interroge le corpus RAG pour fournir des explications textuelles précises.
  * **AgentFormule** : Spécialisé dans l'extraction et le formatage de preuves mathématiques en **LaTeX**.
  * **AgentSchema** : Génère des cartes mentales dynamiques au format **Mermaid.js** pour visualiser les concepts.
  * **AgentWeb** : Utilise l'API **Tavily** pour répondre à des questions d'actualité hors du corpus local.

### 🔍 Outils 

  * **RAG (Retrieval Augmented Generation)** : Indexation de documents académiques.
      * `data/ml` : Cours de Machine Learning Avancé (DOCX).
      * `data/stats` : Cours de Statistiques (PDF).
  * **VectorStore** : FAISS (Facebook AI Similarity Search) pour une recherche vectorielle rapide.
  * **Mindmap Tool** : Rendu interactif de schémas via un `SchemaTool` personnalisé.

-----

## 📂 Structure du Projet

```text
Assistant_Intelligent_Projet/
├── app.py              # Interface utilisateur Streamlit
├── agents_langGraph.py # Logique du graphe et définition des agents
├── rag_tool.py         # Configuration RAG, embeddings et outils Tavily
├── schema_tool.py      # Utilitaire de rendu pour les schémas Mermaid
├── data/               # Corpus documentaire
│   ├── ml/             # Cours de Machine Learning
│   └── stats/          # Cours de Statistiques
├── requirements.txt    # Dépendances du projet
└── .env                # Clés API (OpenAI, Tavily)
```

-----

## 🚀 Installation et Utilisation

1.  **Cloner le dépôt :**

    ```bash
    git clone https://github.com/votre-username/nom-du-repo.git
    cd nom-du-repo
    ```

2.  **Installer les dépendances :**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Lancer l'application :**

    ```bash
    streamlit run app.py
    ```

-----

## 🎓 Contexte Académique

Ce projet illustre l'implémentation pratique de concepts de **Generative AI** :

  * Gestion de l'état (State Management) avec LangGraph.
  * Optimisation du prompt engineering pour l'orchestration.
  * Hybridation entre recherche sémantique locale (RAG) et recherche web.
  * Visualisation de données non structurées.

-----

*Développé avec VS Code, Git et une passion pour l'IA.*
