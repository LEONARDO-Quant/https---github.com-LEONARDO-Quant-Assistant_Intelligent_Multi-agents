import streamlit as st
import openai
from agent_Master import MasterAgent 
from agents import TextualAgent, MathAgent, WebAgent
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
            
            # CORRECTION : On utilise la méthode .answer() définie dans ton Master
            response = st.session_state.master.answer(query)

            placeholder.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

