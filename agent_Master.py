
from rag_tool import RAGDocumentTool, theory_engine, stats_engine
from agents import SchemaAgent, TextualAgent, MathAgent, WebAgent
import openai
import re



class MasterAgent:
    def __init__(self, theory_engine, stats_engine):
        # Initialisation des sous-agents
        self.text_agent = TextualAgent(rag_tool=theory_engine)
        self.math_agent = MathAgent(rag_tool=stats_engine)
        self.web_agent = WebAgent()
        self.schema_agent = SchemaAgent()

        self.schema_history = []  # Pour stocker les schémas générés et les afficher dans l'app
        # Initialisation de la mémoire
        self.chat_history = []
        self.current_schema = None  # Pour stocker le dernier schéma généré

        
        self.system_prompt = (
            "Tu es l'Agent Master, le cerveau d'une application multi-experts. Analyse la requête en fonction de l'historique et de la question.\n"
            "Ton rôle est d'analyser la requête de l'utilisateur et de choisir l'action appropriée :\n"
            "1. Si la question porte sur des concepts théoriques ou de définition (ex: 'C'est quoi...'), utilise la THEORIE.\n"
            "2. Si la question demande des formules, des calculs ou des stats, utilise les MATHS.\n"
            "3. Si la question porte sur l'actualité ou demande des infos externes aux cours, utilise le WEB.\n"
            "4. SCHEMA : Si l'utilisateur demande explicitement un schéma, un graphique ou une carte mentale.\n"
            "5. Si c'est une salutation ou une conversation banale, réponds directement par toi-même.\n\n"
            "Réponds UNIQUEMENT avec l'un de ces mots-clés au début : [THEORIE], [MATHS], [WEB], [SCHEMA] ou [DIRECT]."
            "Consulte toujours les 5 derniers échanges pour comprendre le contexte et les pronoms (ex: 'il', 'ça', 'encore')."
        )

    def answer(self, user_query: str):
        # 1. Préparer les messages pour le routing (avec mémoire pour comprendre les 'il', 'ça', 'encore')
        routing_messages = [{"role": "system", "content": self.system_prompt}]
        # On donne les 5 derniers échanges au router pour le contexte
        routing_messages.extend(self.chat_history[-5:]) 
        routing_messages.append({"role": "user", "content": user_query})
        
        routing_res = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=routing_messages,
            temperature=0
        )
        decision = routing_res.choices[0].message.content.upper()

        display_response = ""
        full_response_for_memory = ""

        # 2. Délégation et récupération de la réponse
        if "[SCHEMA]" in decision:
            response = self.schema_agent.answer(user_query)
            self.current_schema = response
            self.schema_history.append(response)
            display_response = "Voici le schéma ajouté au laboratoire (colonne de droite) :"
            full_response_for_memory = response # L'IA garde le code en mémoire

        elif "[THEORIE]" in decision:
            response = self.text_agent.answer(user_query)
            full_response_for_memory = response
            # Extraction si l'agent texte a aussi généré un schéma
            match = re.search(r"SCHEMA_START(.*?)SCHEMA_END", response, re.DOTALL)
            if match:
                self.current_schema = match.group(1).strip()
                self.schema_history.append(self.current_schema)
                display_response = response.replace(match.group(0), "").strip()
            else:
                display_response = response
        elif "[MATHS]" in decision:
            response = self.math_agent.answer(user_query)
            full_response_for_memory = response
            match = re.search(r"SCHEMA_START(.*?)SCHEMA_END", response, re.DOTALL)
            if match:
                self.current_schema = match.group(1).strip()
                self.schema_history.append(self.current_schema)
                display_response = response.replace(match.group(0), "").strip()
        elif "[WEB]" in decision:
            response = self.web_agent.answer(user_query)
            full_response_for_memory = response
            display_response = response     
            if match := re.search(r"SCHEMA_START(.*?)SCHEMA_END", response, re.DOTALL):
                self.current_schema = match.group(1).strip()
                self.schema_history.append(self.current_schema)
                display_response = response.replace(match.group(0), "").strip()
        else:
            # Mode DIRECT
            res = openai.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[{"role": "system", "content": "Assistant utile."}] + self.chat_history[-5:] + [{"role": "user", "content": user_query}]
            )
            display_response = res.choices[0].message.content
            full_response_for_memory = display_response

        # 3. MISE À JOUR DE LA MÉMOIRE (Contenu complet)
        self.chat_history.append({"role": "user", "content": user_query})
        self.chat_history.append({"role": "assistant", "content": full_response_for_memory})
        
        return display_response