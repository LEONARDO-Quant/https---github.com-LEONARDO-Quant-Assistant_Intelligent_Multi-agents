import openai
import os
from dotenv import load_dotenv
from rag_tool import RAGDocumentTool

class MathAgent:
    def __init__(self, rag_tool: RAGDocumentTool):
        self.rag_tool = rag_tool
        self.model = "gpt-4o-mini"
        # Le System Prompt est ici configuré pour les mathématiques
        self.system_prompt = (
            "Extrait les formules mathématiques, les notations statistiques et les équations (ex: Loss functions, Gradient Descent, distributions). 
            À utiliser pour les demandes de démonstrations ou de précisions mathématiques."
            "Règles strictes :\n"
            "1. Donne la formule principale en format LaTeX (entre $$).\n"
            "2. Explique chaque variable de la formule.\n"
            "3. Si aucune formule n'est trouvée, dis : 'Aucune équation mathématique détectée dans ce passage'."
        )
        
    def answer(self, user_query: str):
        # On utilise le même outil RAG
        context = self.rag_tool.run(user_query)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"CONTEXTE DOCS :\n{context}\n\nQUESTION : {user_query}"}
        ]

        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1 # Très bas pour éviter toute erreur de calcul
        )
        return response.choices[0].message.content

# --- BLOC DE DÉMARRAGE ---
if __name__ == "__main__":
    load_dotenv()
    PATH_DATA = r"C:\Users\Utilisateur\Desktop\Panthéon-Sorbonne VII\Assistant_Intelligent_Multi-agents\data"
    
    print("🔢 Initialisation de l'Agent Formules...")
    shared_rag = RAGDocumentTool(docs_dir=PATH_DATA)
    agent_math = MathAgent(rag_tool=shared_rag)
    
    print("\n✅ Agent Formules prêt !")
    while True:
        q = input("\nDe quelle formule as-tu besoin ? : ")
        if q.lower() == 'q':
            break
        
        print("\nAnalyse en cours...")
        print("-" * 30)
        print(agent_math.answer(q))
        print("-" * 30)
