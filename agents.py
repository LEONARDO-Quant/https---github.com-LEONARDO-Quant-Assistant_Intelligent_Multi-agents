
import openai

class TextualAgent:
    def __init__(self, rag_tool):
        self.rag_tool = rag_tool
        self.system_prompt = (
            "Tu es un expert en analyse textuelle. Explique les concepts pédagogiquement. "
            "CONSIGNE : Ne donne JAMAIS de formules, concentre-toi sur les idées."
        )

    def answer(self, user_query: str):
        # CHANGEMENT : On appelle la méthode spécifique "theory"
        context = self.rag_tool.run_theory_search(user_query) 
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"CONTEXTE :\n{context}\n\nQUESTION : {user_query}"}
        ]
        response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.1)
        return response.choices[0].message.content

class MathAgent:
    def __init__(self, rag_tool):
        self.rag_tool = rag_tool
        self.system_prompt = (
            "Tu es un expert en mathématiques. Extrait les formules en format LaTeX ($$)."
            "Si aucune formule n'est trouvée, dis : 'Aucune équation détectée'."
        )

    def answer(self, user_query: str):
        # CHANGEMENT : On appelle la méthode spécifique "math"
        context = self.rag_tool.run_math_search(user_query)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"CONTEXTE :\n{context}\n\nQUESTION : {user_query}"}
        ]
        response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.1)
        return response.choices[0].message.content

# On garde ton BiblioAgent tel quel, il servira de 3ème option pour le Maître.
