import streamlit as st
import openai
from agent_Master import MasterAgent 
from agents import TextualAgent, MathAgent, WebAgent, SchemaTool
from rag_tool import RAGDocumentTool, theory_engine, stats_engine 
from dotenv import load_dotenv
import os   
import numpy as np
import pandas as pd

# CHANGEMENT : On charge la clé API directement dans app.py pour éviter les problèmes de chargement dans les modules
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key or not os.getenv("TAVILY_API_KEY"):
    st.error("⚠️ Il manque une clé API dans le fichier .env !")

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="IA-Master: ML & Stats Assistant",
    page_icon="🤖",
    layout="wide"
)

# Style CSS (Ton design conservé)
st.markdown(
    """
    <style>
    .main { background-color: #f5f7f9; }
    .stChatMessage { border-radius: 15px; border: 1px solid #ddd; margin-bottom: 10px; }
    .source-tag { 
        font-size: 0.8em; color: #007bff; background: #e7f3ff; 
        padding: 2px 8px; border-radius: 10px; font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True, 
)

# --- INITIALISATION DU SYSTÈME ---
if "master" not in st.session_state:
    with st.spinner("🛠️ Initialisation du cerveau IA et des index FAISS..."):
        # On utilise directement les instances importées de rag_tool
        st.session_state.master = MasterAgent(
            theory_engine=theory_engine, 
            stats_engine=stats_engine
        )
        st.session_state.chat_history = []

# --- LAYOUT : COLONNE GAUCHE (CHAT) / COLONNE DROITE (LABO MATHS) ---
col_chat, col_lab = st.columns([0.6, 0.4], gap="large")

with col_chat:
    st.title("🤖 Master IA Assistant")
    st.caption("Expert en Machine Learning & Statistiques - Posez vos questions !")

    # Affichage de l'historique
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Zone d'entrée utilisateur
    if query := st.chat_input("Posez votre question sur le ML ou les Stats..."):
        # 1. Ajouter et afficher le message utilisateur
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # 2. Réponse du Master Agent
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🔍 *Le Maître consulte les experts...*")
            
            response = st.session_state.master.answer(query)
            placeholder.markdown(response)
            
            # --- AJOUT CRUCIAL ICI ---
            # On vérifie si le Master a généré un schéma lors de son dernier appel
            if hasattr(st.session_state.master, 'current_schema') and st.session_state.master.current_schema:
                st.write("---") # Petite ligne de séparation
                tool = SchemaTool()
                tool.render(st.session_state.master.current_schema)
            # --------------------------

            st.session_state.chat_history.append({"role": "assistant", "content": response})


with col_lab:
    st.subheader("🧪 Laboratoire de Visualisation")
    
    # On crée un conteneur avec une hauteur définie et un scroll automatique
    with st.container():
        # CSS pour forcer la colonne de droite à défiler si elle déborde
        st.markdown("""
            <style>
            [data-testid="column"]:nth-child(2) {
                max-height: 90vh;
                overflow-y: auto !important;
            }
            </style>
        """, unsafe_allow_html=True)

        master = st.session_state.get("master")
        
        if master and hasattr(master, 'schema_history') and master.schema_history:
            # On affiche les schémas
            for i, schema_code in enumerate(reversed(master.schema_history)):
                index_reel = len(master.schema_history) - i
                with st.expander(f"📊 Schéma #{index_reel}", expanded=(i==0)):
                    # On appelle le render ici
                    tool = SchemaTool()
                    tool.render(schema_code)
        else:
            st.info("Les schémas apparaîtront ici.")