
from chainlit import openai

from agent_formules import MathAgent
from agent_textuel import TextualAgent
from agents import MathAgent, SchemaAgent, WebAgent



class MasterAgent:
    def __init__(self, theory_engine, stats_engine):
        self.text_agent = TextualAgent(rag_tool=theory_engine)
        self.math_agent = MathAgent(rag_tool=stats_engine)
        self.web_agent = WebAgent()
        self.schema_agent = SchemaAgent()
        
        self.chat_history = [] # Mémoire persistante
        self.current_schema = None

        self.system_prompt = (
            "Tu es l'Agent Master. Analyse la requête selon l'historique.\n"
            "Fait ta recherche d'abord sur les documents (THEORIE/MATHS) avant d'aller sur le WEB et n'oublie pas les citations.\n"
            "Réponds UNIQUEMENT par : [THEORIE], [MATHS], [WEB], [SCHEMA] ou [DIRECT].\n"
            "Si la question fait référence à un élément précédent (ex: 'Explique moi ça'), "
            "base-toi sur l'historique pour choisir le bon agent."
        )

    def answer(self, user_query: str, history=None):
        self.current_schema = None # On vide le slot pour ce tour-ci
        
        # Synchronisation de la mémoire avec Streamlit
        if history is not None:
            self.chat_history = history  

        # 1. ROUTING : On donne l'historique au router pour qu'il comprenne les contextes
        context = self.chat_history[-5:] if self.chat_history else []
        routing_messages = [{"role": "system", "content": self.system_prompt}] + context + [{"role": "user", "content": user_query}]

        try:   
            routing_res = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=routing_messages,
                temperature=0
            )
            decision = routing_res.choices[0].message.content
        except:
            decision = "[DIRECT]"

        # 2. EXÉCUTION
        if "[SCHEMA]" in decision:
            self.current_schema = self.schema_agent.answer(user_query)
            display_response = "Voici le schéma généré dans l'onglet dédié."
        
        elif "[THEORIE]" in decision:
            res_rag = self.text_agent.answer(user_query)
            # Si le RAG ne trouve rien d'utile, on bascule sur le WEB
            if "Aucun résultat" in res_rag or len(res_rag) < 50:
                res_rag = self.web_agent.answer(user_query)
            display_response, self.current_schema = self._extract_schema_if_any(res_rag)

        elif "[MATHS]" in decision:
            res_math = self.math_agent.answer(user_query)
            display_response, self.current_schema = self._extract_schema_if_any(res_math)

        elif "[WEB]" in decision:
            display_response = self.web_agent.answer(user_query)

        else:
            # Mode conversationnel direct
            res = openai.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[{"role": "system", "content": "Assistant amical."}] + context + [{"role": "user", "content": user_query}]
            )
            display_response = res.choices[0].message.content

        # 3. MISE À JOUR MÉMOIRE : Crucial pour que l'agent se souvienne au prochain tour
        self.chat_history.append({"role": "user", "content": user_query})
        self.chat_history.append({"role": "assistant", "content": display_response})

        return display_response

    def _extract_schema_if_any(self, text: str):
        # On cherche d'abord nos balises personnalisées
        pattern = r"SCHEMA_START([\s\S]*?)SCHEMA_END"
        match = re.search(pattern, text)
        
        if match:
            schema_code = match.group(1).strip()
            # On nettoie le texte pour ne pas afficher le code brut dans le chat
            clean_text = re.sub(pattern, "", text).strip()
            return clean_text, schema_code
        
        # Fallback : si l'IA a mis du Mermaid sans les balises (ex: dans les blocs code)
        mermaid_pattern = r"(graph\s+(TD|LR|mindmap)[\s\S]+)"
        match_mermaid = re.search(mermaid_pattern, text, re.IGNORECASE)
        if match_mermaid:
            return text, match_mermaid.group(1).strip()
            
        return text, None