import streamlit as st
import streamlit.components.v1 as components

import streamlit as st
import streamlit.components.v1 as components

class SchemaTool:
    def render(self, mermaid_code: str):
        """
        Transforme un code texte Mermaid en un diagramme visuel interactif.
        """
        if not mermaid_code:
            return

        # Nettoyage au cas où l'agent ajoute des balises markdown ```mermaid
        clean_code = (
            mermaid_code.replace("```mermaid", "")
            .replace("```", "")
            .strip()
        )

        st.write("### 🧠 Visualisation du Concept")

        # Configuration HTML/JS pour Mermaid
        html_code = f"""
        <div id="graph-container" style="background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <pre class="mermaid" style="display: flex; justify-content: center;">
                {clean_code}
            </pre>
        </div>

        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ 
                startOnLoad: true, 
                theme: 'base',
                securityLevel: 'loose',
                themeVariables: {{ 
                    'primaryColor': '#FF4B4B',
                    'primaryTextColor': '#FFFFFF',
                    'primaryBorderColor': '#FF4B4B',
                    'lineColor': '#31333F',
                    'secondaryColor': '#F0F2F6',
                    'tertiaryColor': '#FFFFFF',
                    'fontFamily': 'sans-serif'
                }} 
            }});
        </script>
        """

        # Rendu du composant
        # Ajuste la hauteur (height) selon la complexité attendue de tes schémas
        components.html(html_code, height=600, scrolling=True)

# --- Exemple d'utilisation rapide ---
# tool = SchemaTool()
# tool.render("graph LR; A[Agent] -->|Génère| B(Code Mermaid); B -->|Tool| C{{Schéma Visuel}}")