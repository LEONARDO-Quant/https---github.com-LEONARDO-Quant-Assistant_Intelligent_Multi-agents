import os
import faiss
import json
import numpy as np
import openai
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

class RAGDocumentTool:
    def __init__(self, docs_dir: str, index_name: str = "default"):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ Clé OPENAI_API_KEY manquante.")
        
        openai.api_key = self.api_key
        
        # Chemins basés sur le nom de l'index pour éviter les collisions
        self.docs_dir = Path(docs_dir)
        self.index_path = f"{index_name}.index"
        self.meta_path = f"{index_name}.meta.json"
        
        self.embed_model = "text-embedding-3-small"
        self.chunk_size = 800
        self.chunk_overlap = 150
        
        # Initialisation du store
        self.index, self.chunks = self._prepare_store()

    def _read_pdf_text(self, path: Path) -> str:
        try:
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            return f"Erreur PDF {path.name}: {e}"

    def _load_and_split(self) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        docs_data = [] # On change le nom pour la clarté
    
        for path in self.docs_dir.rglob("*"):
            if path.suffix.lower() in [".txt", ".pdf"]:
                text = self._read_pdf_text(path) if path.suffix == ".pdf" else path.read_text()
                chunks = splitter.split_text(text)
                
                for chunk in chunks:
                    # On stocke le texte ET la source
                    docs_data.append({
                        "text": chunk,
                        "source": path.name # Le nom du fichier sert d'étiquette
                    })
        return docs_data

    
    def _embed(self, docs: list) -> list[list[float]]:
            # 1. Extraction robuste du texte
            texts = []
            for doc in docs:
                if isinstance(doc, dict):
                    # Si c'est un dictionnaire, on prend la clé "text"
                    texts.append(doc.get("text", ""))
                else:
                    # Si c'est déjà une string (cas de la requête utilisateur)
                    texts.append(str(doc))

            # 2. Nettoyage (ta logique : pas de None, pas de vide)
            cleaned_texts = [str(t) for t in texts if t is not None and str(t).strip() != ""]
            
            if not cleaned_texts:
                return []

            # 3. Appel à l'API OpenAI
            res = openai.embeddings.create(
                model=self.embed_model, 
                input=cleaned_texts
            )
            return [e.embedding for e in res.data]

    def _prepare_store(self):
        if Path(self.index_path).exists() and Path(self.meta_path).exists():
            index = faiss.read_index(self.index_path)
            chunks = json.loads(Path(self.meta_path).read_text())
            return index, chunks

        chunks = self._load_and_split()
        if not chunks: return None, []
        
        all_vectors = []
        for i in range(0, len(chunks), 128):
            all_vectors.extend(self._embed(chunks[i : i + 128]))
        
        mat = np.asarray(all_vectors, dtype=np.float32)
        index = faiss.IndexFlatL2(mat.shape[1])
        index.add(mat)
        
        faiss.write_index(index, self.index_path)
        Path(self.meta_path).write_text(json.dumps(chunks))
        return index, chunks

    # --- LES DEUX MÉTHODES DE SORTIE POUR TES AGENTS ---

    def run_theory_search(self, query: str) -> str:
        """
        Recherche des concepts théoriques, définitions et explications 
        sur le Machine Learning et Deep Learning dans les cours.
        Utile pour répondre aux questions conceptuelles (ex: "C'est quoi...").
        """
        return self._execute_search(query, k=5)

    def run_math_search(self, query: str) -> str:
        """
        Recherche des formules mathématiques, équations LaTeX et 
        notations statistiques spécifiques au Deep Learning.
        Utile pour les calculs, les fonctions de perte ou les démonstrations.
        """
        return self._execute_search(query, k=3)
    
    def run(self, query: str) -> str:
        # On choisit une recherche par défaut pour le test
        return self.run_theory_search(query)

    def _execute_search(self, query: str, k: int) -> str:
            # 1. Convertir la requête textuelle en vecteur via _embed
            # On passe [query] car _embed attend une liste
            query_vector = self._embed([query])
            
            if not query_vector:
                return "Erreur : Impossible de générer l'embedding pour la recherche."
            
            # 2. Convertir en format float32 pour FAISS
            query_np = np.array(query_vector).astype('float32')
            
            # 3. Rechercher dans l'index FAISS
            # D = distances, idxs = indices des chunks trouvés
            D, idxs = self.index.search(query_np, k)
            
            results = []
            # 4. Boucler sur les indices récupérés (idxs[0] car une seule requête)
            for i in idxs[0]:
                # Vérifier que l'indice est valide et existe dans nos chunks
                if i != -1 and i < len(self.chunks):
                    chunk_data = self.chunks[i] 
                    
                    # Formatage précis pour donner du contexte à l'IA
                    formatted = f"[SOURCE: {chunk_data['source']}]\nCONTENU: {chunk_data['text']}"
                    results.append(formatted)
            
            # 5. Retourner le contexte final ou un message d'absence
            if not results:
                return "Aucun document pertinent trouvé dans la base de connaissances."
                
            return "\n\n---\n\n".join(results)

# --- CRÉATION DES INSTANCES POUR TES AGENTS ---

# Instance 1 : Spécialisée dans les cours de ML (Concepts)
theory_engine = RAGDocumentTool(docs_dir="./data/theory", index_name="ml_theory")

# Instance 2 : Spécialisée dans les cours de Stats/Math (Formules)
math_engine = RAGDocumentTool(docs_dir="./data/stats", index_name="ml_math")

#############  #################
# --- BLOC DE TEST (NON INDENTÉ) ---
if __name__ == "__main__":
    # Remplace par ton chemin réel
    MON_DOSSIER = r"C:\Users\Utilisateur\Desktop\Panthéon-Sorbonne VII\Assistant_Intelligent_Multi-agents\data"
    
    print("--- TEST DE L'OUTIL RAG ---")
    mon_outil = RAGDocumentTool(docs_dir=MON_DOSSIER)
    
    while True:
        question = input("\n💬 Pose une question à tes documents (ou 'q' pour quitter) : ")
        if question.lower() == 'q':
            break
            
        resultat = mon_outil.run(question)
        print("\n🔍 Résultats trouvés :")
        print(resultat)
