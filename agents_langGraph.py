import operator
from typing import Annotated, TypedDict, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from rag_tool import tools # Import de tes 4 outils
from schema_tool import SchemaTool 

# État du graphe avec historique
class AgentState(TypedDict):
    input: str
    chat_history: List[dict]
    next_node: str
    output: str

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
schema_tool = SchemaTool()

# --- NOEUD : ROUTER (L'ORCHESTRATEUR) ---
def router(state: AgentState):

    history = state.get("chat_history", [])
    query = state["input"]
    prompt = f"""Tu es un orchestrateur. Tu analyses la requête de l'utilisateur et décides quel agent est le mieux placé pour répondre : AgentText, AgentFormule, WebSearch ou AgentMindmap.
    Historique récent : {history[-5:] if history else "Aucun"}
    Analyse la requête : "{query}"
    Options :
    - 'banal' : Si c'est une salutation ou une discussion sans besoin de documents.
    - 'text' : Si c'est une question sur les stats/ML nécessitant les docs.
    - 'formule' : Si on demande une équation LaTeX.
    - 'web' : Si la question est générale/actuelle et absente des fichiers locaux.
    - 'mindmap' : Si l'utilisateur demande explicitement une carte mentale ou un schéma.
    Réponds uniquement par le mot-clé."""
    
    decision = llm.invoke(prompt).content.lower().strip()
    return {"next_node": decision}

# --- NOEUDS D'ACTION ---
def chat_node(state: AgentState):
    messages = state.get("chat_history", []) + [("user", state["input"])]
    res = llm.invoke(messages).content
    
    # On met à jour l'historique ET l'output
    return {
        "output": res,
        "chat_history": messages + [("assistant", res)]
    }

def text_node(state: AgentState):
    # On donne du contexte à la recherche RAG
    query_with_context = f"Question: {state['input']} (Contexte précédent: {state.get('chat_history', [])[-5:]})"
    res = tools[0].run(query_with_context)
    
    return {
        "output": res,
        "chat_history": state.get("chat_history", []) + [("user", state["input"]), ("assistant", res)]
    }

def formula_node(state: AgentState):
    query_context = f"{state['input']} (en lien avec: {state.get('chat_history', [])[-1:]})"
    # 1. Récupération de la formule brute via l'outil RAG
    raw_output = tools[1].run(query_context)
    
    # 2. Nettoyage des symboles parasites courants
    clean_output = raw_output.replace(r"\[", "$$").replace(r"\]", "$$")
    clean_output = clean_output.replace(r"\(", "$").replace(r"\)", "$")
    
    # 3. Prompt rapide pour extraire la définition des variables
    var_prompt = f"""À partir de ce texte contenant une formule : "{clean_output}"
    Génère une liste courte 'Définition des variables' au format Markdown (ex: - $x$ : description).
    La formule doit s'afficher correctement en LaTeX.
    Réponds uniquement avec la liste, sans introduction."""
    
    definitions = llm.invoke(var_prompt).content
    
    # 4. Combinaison de la formule et des définitions
    final_response = f"{clean_output}\n\n**Définition des variables :**\n{definitions}"
    
    return {"output": final_response}

def web_node(state: AgentState):
    query_with_context = f"Question: {state['input']} (Contexte: {state.get('chat_history', [])[-5:]})"
    res = tools[2].run(query_with_context)

    # 2. SYNTHÈSE : On demande au LLM de résumer les résultats
    summary_prompt = f"""Tu es un expert en synthèse. À partir des résultats de recherche suivants, 
    réponds à la question de l'utilisateur de manière concise (maximum 6-7 phrases).
    Cite toujoursn la source.
    Si on a une question supplémentaire va le chercher dans les résultats de recherche.
    Question : {state['input']}
    Résultats bruts : {res}"""
    
    # On utilise le LLM pour transformer le pavé de texte en réponse propre
    res = llm.invoke(summary_prompt).content

    return {
        "output": str(res),
        "chat_history": state.get("chat_history", []) + [("user", state["input"]), ("assistant", res)]
    }

def mindmap_node(state: AgentState):
    query_context = f"Sujet: {state['input']} (Basé sur les discussions précédentes: {state.get('chat_history', [])[-2:]})"
    res = tools[3].run(query_context)
    
    return {
        "output": res,
        "chat_history": state.get("chat_history", []) + [("user", state["input"]), ("assistant", "Schéma généré")]
    }


# --- CONSTRUCTION DU GRAPHE ---
workflow = StateGraph(AgentState)

workflow.add_node("router", router)
workflow.add_node("chat", chat_node)
workflow.add_node("text", text_node)
workflow.add_node("formula", formula_node)
workflow.add_node("web", web_node)
workflow.add_node("mindmap", mindmap_node)

workflow.set_entry_point("router")

workflow.add_conditional_edges("router", lambda x: x["next_node"], {
    "banal": "chat",
    "text": "text",
    "formule": "formula",
    "web": "web",
    "mindmap": "mindmap"
})

workflow.add_edge("chat", END)
workflow.add_edge("text", END)
workflow.add_edge("formula", END)
workflow.add_edge("web", END)
workflow.add_edge("mindmap", END)

# Mémoire pour l'historique
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
