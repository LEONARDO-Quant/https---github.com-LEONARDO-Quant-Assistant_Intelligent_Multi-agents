import streamlit as st
import time
import openai
from agents import TextualAgent, MathAgent, WebAgent
from agent_Master import MasterAgent
from rag_tool import RAGDocumentTool, theory_engine, stats_engine

openai.api_key = st.secrets["OPENAI_API_KEY"]



# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="IA-Master: ML & Stats Assistant",
    page_icon="🤖",
    layout="wide"
)

# Style CSS personnalisé pour une interface "stylée"
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatMessage { border-radius: 15px; border: 1px solid #ddd; margin-bottom: 10px; }
    [data-testid="column"]:nth-child(2) [data-testid="stVerticalBlock"] {
        position: sticky;
        top: 2rem;
        height: max-content;
    }
    .source-tag { 
        font-size: 0.8em; color: #007bff; background: #e7f3ff; 
        padding: 2px 8px; border-radius: 10px; font-weight: bold;
    }
    .formula-zone {
        background-color: #1e1e1e; color: #ffffff; padding: 20px;
        border-radius: 10px; border-left: 5px solid #ff4b4b;
    }
    </style>
""", unsafe_allow_html=True, 
)


# --- INITIALISATION DU SYSTÈME (SESSION STATE) ---
if "master" not in st.session_state:
    with st.spinner("🛠️ Initialisation du cerveau IA et des index FAISS..."):
        # 1. L'outil (chargé une seule fois)
        rag = RAGDocumentTool(docs_dir="data/") 
        
        # 2. Les agents spécialisés
        txt_agent = TextualAgent(rag)
        math_agent = MathAgent(rag)
        web_agent = WebAgent()
        
        # 3. Le Maître
        st.session_state.master = MasterAgent(txt_agent, math_agent, web_agent)
        st.session_state.chat_history = []

# --- LAYOUT : COLONNE GAUCHE (CHAT) / COLONNE DROITE (LABO MATHS) ---
col_chat, col_lab = st.columns([0.6, 0.4], gap="large")

with col_chat:
    st.title("🤖 Master IA Assistant")
    st.caption("Expert en Machine Learning & Statistiques - Posez vos questions !")

    # Zone d'affichage des messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Zone d'invite claire et spacieuse
    if query := st.chat_input("Posez votre question sur le ML ou les Stats..."):
        # Affichage immédiat du message utilisateur
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Réponse du Master Agent
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🔍 *Le Maître consulte les experts...*")
            
            response = st.session_state.master.process_request(query)
            
            placeholder.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})


with col_lab:
    st.subheader("🔍 Carte Mentale")
    if hasattr(st.session_state.master, 'current_schema') and st.session_state.master.current_schema:
        from schema_tool import SchemaTool
        st.session_state.schema_viewer = SchemaTool()
        st.session_state.schema_viewer.render(st.session_state.master.current_schema)
    else:
        st.info("La structure logique de ton cours s'affichera ici.")
