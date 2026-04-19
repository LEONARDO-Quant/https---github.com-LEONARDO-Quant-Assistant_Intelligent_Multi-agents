import streamlit as st
import re
from agents_langGraph import app  # Importation de ton graphe compilé
from rag_tool import tools # Import de tes 4 outils
from schema_tool import SchemaTool

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="IA Multi-Agents Pro", layout="wide")

# --- 2. STYLE CSS (CONSERVATION DU DESIGN ORIGINAL) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    [data-testid="stSidebar"] {
        background-color: #161B22;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. INITIALISATION DE LA SESSION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.schemas = []
    # thread_id unique pour maintenir la mémoire LangGraph durant la session
    st.session_state.thread_id = "session_streamlit_unique"

# Instance de l'outil de rendu pour Mermaid
schema_tool = SchemaTool()

# --- 4. NAVIGATION ---
page = st.sidebar.selectbox("Navigation", ["💬 Chat Principal", "🧠 Laboratoire de Schémas"])

# ---------------------------------------------------------
# PAGE 1 : CHAT PRINCIPAL
# ---------------------------------------------------------
if page == "💬 Chat Principal":
    st.title("🤖 Assistant Intelligent Multi-Agents")
    st.caption("Expert en Théorie, Stats, Web et Visualisation (Propulsé par LangGraph)")

    # Affichage de l'historique des messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrée utilisateur
    if prompt := st.chat_input("Posez votre question ici..."):
        # Affichage du message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Appel à l'orchestrateur LangGraph
        with st.chat_message("assistant"):
            with st.spinner("L'IA réfléchit (Analyse des agents)..."):
                
                # Configuration de la mémoire via thread_id
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                try:
                    # Exécution du graphe
                    # On ne passe pas l'historique manuellement, le Checkpointer s'en charge
                    result = app.invoke({"input": prompt}, config=config)
                    
                    response = result.get("output", "Désolé, je n'ai pas pu générer de réponse.")
                    node_used = result.get("next_node", "agent")

                    # --- GESTION DES SCHÉMAS (MERMAID) ---
                    # Si le nœud mindmap est utilisé ou si le code contient 'mindmap'
                    if node_used == "mindmap" or "mindmap" in response.lower():
                        clean_code = response
                        if "```mermaid" in response:
                            clean_code = response.split("```mermaid")[1].split("```")[0].strip()
                        elif "mindmap" in response:
                            clean_code = "mindmap" + response.split("mindmap", 1)[1].strip()     # Rendu visuel dans le chat
                        
                        schema_tool.render(clean_code)  # Affichage dans le chat
                        if clean_code not in st.session_state.schemas:
                            st.session_state.schemas.append(clean_code)
                            st.info("💡 Schéma ajouté au Laboratoire !")
                    else:
                        st.markdown(response)  # Affichage normal pour les autres réponses

                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                except Exception as e:
                    st.error(f"Erreur lors de l'exécution du graphe : {e}")

# ---------------------------------------------------------
# PAGE 2 : LABORATOIRE DE SCHÉMAS
# ---------------------------------------------------------
elif page == "🧠 Laboratoire de Schémas":
    st.title("Visualisation Graphique")
    st.write("Historique des cartes mentales et schémas générés.")

    if not st.session_state.schemas:
        st.info("Aucun schéma disponible. Demandez une 'carte mentale' dans le chat !")
    else:
        # Affichage inversé (du plus récent au plus ancien)
        for i, sc in enumerate(reversed(st.session_state.schemas)):
            index_schema = len(st.session_state.schemas) - i
            with st.expander(f"Schéma #{index_schema}", expanded=True):
                # Rendu visuel du Mermaid
                schema_tool.render(sc)
                # Affichage du code source pour copie
                st.code(sc, language="mermaid")