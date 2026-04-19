
import openai
import re 
from web_tool import WebSearchTool, tavily_tool
from rag_tool import theory_engine, stats_engine
from schema_tool import SchemaTool



class TextualAgent:
    def __init__(self, rag_tool):
        self.rag_tool = rag_tool
        self.system_prompt = (
            "Tu es un expert en analyse textuelle. Explique les concepts pédagogiquement. "
            "CONSIGNE : Ne donne JAMAIS de formules, concentre-toi sur les idées."
            "Si l'utilisateur demande une visualisation ou si c'est complexe, "
            "tu peux générer un schéma Mermaid entre les balises SCHEMA_START et SCHEMA_END."
            "RÈGLE CRUCIALE : A la fin de chaque paragraphe ou explication, cite la source utilisée entre parenthèses, par exemple : (Cours_ML_Stats_Chap2.pdf)"
        )

    def answer(self, user_query: str):
        # CHANGEMENT : On appelle la méthode spécifique "theory_engine" pour le moteur de recherche de théorie
        context = self.rag_tool.theory_engine.run(user_query)
        
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
        context = self.rag_tool.stats_engine.run(user_query)

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
    

class SchemaAgent:
    def __init__(self):
        self.tool = SchemaTool()
        self.system_prompt = (
            "Tu es un expert en visualisation Mermaid.js. "
            "Ta mission : transformer les concepts en diagrammes. "
            "RÈGLES CRITIQUES :\n"
            "1. Réponds EXCLUSIVEMENT avec le code Mermaid brut.\n"
            "2. NE PAS mettre de balises Markdown (ex: pas de ```mermaid).\n"
            "3. NE PAS ajouter de texte d'introduction ou de conclusion.\n"
            "4. Commence directement par 'graph TD', 'graph LR' ou 'mindmap'.\n"
            "5. Si la demande est une hiérarchie d'idées, utilise 'mindmap'."
        )

    def answer(self, user_query: str):
        # On peut forcer l'IA à être encore plus spécifique dans la requête utilisateur
        prompt_final = f"Génère un schéma Mermaid pour la requête suivante : {user_query}"
        
        flux = self.tool.render(prompt_final)  # On peut aussi passer par un rendu direct pour plus de contexte visuel, à tester selon les besoins

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt_final}
        ]
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini", 
            messages=messages, 
            temperature=0 # Plus stable pour du code
        )
        
        return response.choices[0].message.content