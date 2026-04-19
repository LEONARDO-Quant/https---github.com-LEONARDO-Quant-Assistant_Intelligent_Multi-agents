import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, Docx2txtLoader
import langchain_tavily
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch

# --- 1. CONFIGURATION ---
load_dotenv()

path_pdf = r"C:\Users\Utilisateur\Desktop\Panthéon-Sorbonne VII\Assistant_Intelligent_Multi-agents\data\stats"
path_docx = r"C:\Users\Utilisateur\Desktop\Panthéon-Sorbonne VII\Assistant_Intelligent_Multi-agents\data\ml"

# --- 2. CHARGEMENT ET INDEXATION ---
# Chargement PDF
pdf_loader = PyPDFDirectoryLoader(path_pdf)
# Chargement DOCX (via DirectoryLoader pour filtrer par extension)
docx_loader = DirectoryLoader(path_docx, glob="./*.docx", loader_cls=Docx2txtLoader)

docs = pdf_loader.load() + docx_loader.load()

# Split du texte
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# Création du VectorStore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- 3. LOGIQUE DE RECHERCHE (RAG) ---
def run_rag_query(query: str, mode: str):
    results = retriever.invoke(query)
    context = "\n\n".join([f"Source: {doc.metadata.get('source')}\nContenu: {doc.page_content}" for doc in results])
    
    if mode == "text":
        prompt = f"Réponds à la question suivante en utilisant uniquement le corpus documentaire et le contexte. Cite la source sous la forme (Source: 'nom_du_fichier.extension').\n\nContexte:\n{context}\n\nQuestion: {query}"
    else:
        prompt = f"Extrais les formules mathématiques en format LaTeX correspondant à la demande à partir du corpus documentaire et le contexte. Cite la source sous la forme (Source: 'nom_du_fichier.extension').\n\nContexte:\n{context}\n\nDemande: {query}"
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return llm.invoke(prompt).content

# --- 4. CRÉATION DES TOOLS ---
agent_text_tool = Tool(
    name="AgentText",
    func=lambda q: run_rag_query(q, "text"),
    description="Utile pour répondre à des questions textuelles. Renvoie sa réponse avec des citations."
)

agent_formule_tool = Tool(
    name="AgentFormule",
    func=lambda q: run_rag_query(q, "formula"),
    description="Utile pour extraire des formules au format LaTeX. Renvoie sa réponse avec des citations."
)


# --- NOUVEL OUTIL : RECHERCHE WEB (TAVILY) ---
# Nécessite TAVILY_API_KEY dans ton .env
web_search_tool = TavilySearchResults(
    name="WebSearch",
    description="Utile pour répondre à des questions sur l'actualité ou des sujets absents du corpus documentaire. Cite les sources des résultats de recherche.",
    k=3
)

# --- NOUVEL OUTIL : CARTE MENTALE (MERMAID) ---
def generate_mindmap(query: str):
    # On récupère le contexte du RAG pour construire la carte
    results = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in results])
    sources = " | ".join(list(set([doc.metadata.get('source') for doc in results])))
    
    prompt = f"""Génère uniquement code Mermaid.js pour une carte mentale (mindmap) basée sur : {query}
    Ne fait aucune phrase d'introduction.
    Utilise les informations du contexte suivant : {context}
    Met les sources sur une branche : {sources}"""
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return llm.invoke(prompt).content

agent_mindmap_tool = Tool(
    name="AgentMindmap",
    func=generate_mindmap,
    description="Génère le code Mermaid d'une carte mentale basée sur les documents. Cite les sources utilisées pour construire la carte."
)

# Mets à jour ta liste d'outils
tools = [agent_text_tool, agent_formule_tool, web_search_tool, agent_mindmap_tool]


