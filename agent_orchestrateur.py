
from rag_tool import RAGDocumentTool
from agents import TextualAgent, MathAgent, BiblioAgent, MasterAgent

class MasterAgent:
    def __init__(self, textual_agent, math_agent, biblio_agent):
        self.textual_agent = textual_agent
        self.math_agent = math_agent
        self.biblio_agent = biblio_agent
        self.model = "gpt-4o-mini"

    def process_request(self, user_query: str):
        """
        Le Maître analyse la requête et délègue aux spécialistes.
        """
        # 1. Étape de décision (Routing)
        decision_prompt = (
            "Analyse la requête utilisateur et détermine quels experts appeler. "
            "Réponds uniquement en JSON avec les clés 'besoin_theorie', 'besoin_math', 'besoin_biblio' (booléens)."
        )
        
        # Simulation d'un appel au LLM pour le routage (plus simple pour ton Streamlit)
        # Ici on pourrait faire un appel GPT pour décider, ou simplement déléguer systématiquement.
        
        # 2. Délégation (On peut appeler plusieurs agents si la question est mixte)
        final_report = ""
        
        if "formule" in user_query.lower() or "calcul" in user_query.lower():
            final_report += f"### 🧮 Analyse Mathématique\n{self.math_agent.answer(user_query)}\n\n"
        
        if "explique" in user_query.lower() or "c'est quoi" in user_query.lower():
            final_report += f"### 📚 Explication Théorique\n{self.textual_agent.answer(user_query)}\n\n"
            
        if "livre" in user_query.lower() or "source" in user_query.lower():
            final_report += f"### 📖 Bibliographie\n{self.biblio_agent.answer(user_query)}\n\n"

        if not final_report:
            # Par défaut, on demande au théoricien
            final_report = self.textual_agent.answer(user_query)

        return final_report



if __name__ == "__main__":
    print("🚀 Système Multi-Agents activé.")
    print("Tape 'q' pour quitter.\n")

    while True:
        # On récupère la requête de l'utilisateur ici !
        user_query = input("❓ Posez votre question (Cours/Calculs) : ")
        
        if user_query.lower() in ['q', 'quit', 'exit']:
            print("Au revoir !")
            break
            
        # On passe la requête à la fonction qui gère les experts
        interroger_tout_le_monde(user_query)
