
import openai
from web_tool import WebSearchTool, tavily_tool
from rag_tool import theory_engine, stats_engine


class TextualAgent:
    def __init__(self, rag_tool):
        self.rag_tool = rag_tool
        self.system_prompt = (
            "Tu es un expert en analyse textuelle. Explique les concepts pédagogiquement. "
            "CONSIGNE : Ne donne JAMAIS de formules, concentre-toi sur les idées."
            "RÈGLE CRUCIALE : A la fin de chaque paragraphe ou explication, cite la source utilisée entre parenthèses, par exemple : (Trouvé sur: Cours_ML_Stats_Chap2.pdf"
        )

    def answer(self, user_query: str):
        # CHANGEMENT : On appelle la méthode spécifique "theory_engine" pour le moteur de recherche de théorie
        context = self.rag_tool.run(user_query) 
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"CONTEXTE :\n{context}\n\nQUESTION : {user_query}"}
        ]
        response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2)
        return response.choices[0].message.content

class MathAgent:
    def __init__(self, rag_tool):
        self.rag_tool = rag_tool
        self.system_prompt = (
            "Tu es un expert en mathématiques. Extrait les formules en format LaTeX ($$)."
            "Si aucune formule n'est trouvée, dis : 'Aucune équation détectée'."
            "RÈGLE CRUCIALE : A la fin de chaque paragraphe ou explication, cite la source utilisée entre parenthèses, par exemple : (Trouvé sur: Cours_ML_Stats_Chap2.pdf"
            "Si vraiment aucun symbole mathématique n'existe pas dans le texte, alors dis : 'Aucune équation détectée'."
        )

    def answer(self, user_query: str):
        # CHANGEMENT : On appelle la méthode spécifique "stats_engine" pour le moteur de recherche de statistiques
        context = self.rag_tool.run(user_query)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"CONTEXTE :\n{context}\n\nQUESTION : {user_query}"}
        ]
        response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2)
        return response.choices[0].message.content



class WebAgent:
    def __init__(self):
        self.tool = tavily_tool  # On utilise l'outil que tu as créé
        self.system_prompt = (
            "Tu es un expert en recherche d'informations sur le web. "
            "Ta mission est de synthétiser les résultats trouvés de manière claire. "
            "RÈGLE : Cite toujours tes sources avec les liens URL fournis par l'outil."
        )

    def answer(self, user_query: str):
        # 1. L'agent utilise l'outil pour obtenir les données brutes
        raw_results = self.tool.search(user_query)
        
        # 2. L'agent envoie ces données au LLM pour rédaction
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"RÉSULTATS DE RECHERCHE :\n{raw_results}\n\nQUESTION : {user_query}"}
        ]
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini", 
            messages=messages, 
            temperature=0.3
        )
        return response.choices[0].message.content
