
from pydoc import text

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
            "Priorité absolue au CORPUS (THEORIE/MATHS). N'utilise le WEB que si l'information est absente des documents ou porte sur l'actualité\n"
            "1. Si la question porte sur des concepts théoriques ou de définition (ex: 'C'est quoi...'), utilise la THEORIE.\n"
            "2. Si la question demande des formules, des calculs ou des stats, utilise les MATHS.\n"
            "3. Si la question porte sur l'actualité ou demande des infos externes aux cours, utilise le WEB.\n"
            "4. SCHEMA : Si l'utilisateur demande explicitement un schéma, un graphique ou une carte mentale.\n"
            "5. Si c'est une salutation ou une conversation banale, réponds directement par toi-même.\n\n"
            "Réponds UNIQUEMENT avec l'un de ces mots-clés au début : [THEORIE], [MATHS], [WEB], [SCHEMA] ou [DIRECT]."
            "Consulte toujours les 5 derniers échanges pour comprendre le contexte et les pronoms (ex: 'il', 'ça', 'encore')."
        )

    def answer(self, user_query: str, history=None):
        if history is not None:
            self.chat_history = history  # Met à jour la mémoire avec l'historique fourni   
        # 1. Préparer les messages pour le routing (avec mémoire pour comprendre les 'il', 'ça', 'encore')
        context = self.chat_history[-5:] if self.chat_history else []
        routing_messages = [{"role": "system", "content": self.system_prompt}]
        # On donne les 5 derniers échanges au router pour le contexte
        routing_messages.extend(context) 
        routing_messages.append({"role": "user", "content": user_query})

        decision = "[DIRECT]"  # Valeur par défaut au cas où le modèle ne répond pas comme prévu
        
        try:   
            routing_res = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=routing_messages,
                temperature=0
            )
        except Exception as e:
            decision = "[DIRECT]"

        display_response = ""
        full_response_for_memory = ""

        # 2. Délégation et récupération de la réponse
        if "[SCHEMA]" in decision:
            response = self.schema_agent.answer(user_query)
            self.current_schema = response
            display_response = "Voici le schéma généré dans l'onglet dédié"
            full_response_for_memory = response # L'IA garde le code en mémoire

        elif "[THEORIE]" in decision:
            response = self.text_agent.answer(user_query)
            display_response, full_response_for_memory = self._extract_schema_if_any(response)

        elif "[MATHS]" in decision:
            response = self.math_agent.answer(user_query)
            display_response, full_response_for_memory = self._extract_schema_if_any(response)

        elif "[WEB]" in decision:
            response = self.web_agent.answer(user_query)
            display_response, full_response_for_memory = self._extract_schema_if_any(response)

        else:
            # MODE DIRECT : Pour les salutations et conversations banales
            res = openai.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[{"role": "system", "content": "Tu es un assistant amical et intelligent."}] + context + [{"role": "user", "content": user_query}]
            )
            display_response = res.choices[0].message.content
            full_response_for_memory = display_response

        return display_response
    


    def _extract_schema_if_any(self, text: str):
        # On cherche un code Mermaid dans la réponse
        mermaid_pattern = r"(graph\s+(TD|LR|mindmap)[\s\S]+)"
        match = re.search(mermaid_pattern, text, re.IGNORECASE)
        
        if match:
            schema_code = match.group(1)
            return "Voici le schéma généré dans l'onglet dédié", schema_code
        else:
            return text, text  # Pas de schéma, on retourne le texte tel quel pour l'affichage et la mémoire

