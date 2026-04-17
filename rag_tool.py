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
    def __init__(self, docs_dir: str, index_path: str = "faiss.index"):
        # 1. Chargement de la clé API via le .env
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ Clé OPENAI_API_KEY manquante dans le fichier .env")
        
        openai.api_key = self.api_key
        
        # 2. Configuration des modèles (Correction : GPT-4o-mini inclus)
        self.embed_model = "text-embedding-3-small"
        self.chat_model = "gpt-4o-mini" 
        
        # 3. Paramètres techniques
        self.docs_dir = Path(docs_dir)
        self.index_path = index_path
        self.meta_path = index_path + ".meta.json"
        self.chunk_size = 800
        self.chunk_overlap = 150
        
        # 4. Initialisation automatique du store au démarrage
        self.index, self.chunks = self._prepare_store()

    def _read_pdf_text(self, path: Path) -> str:
        try:
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            print(f"⚠️ Erreur lecture PDF {path.name}: {e}")
            return ""

    def _load_and_split(self) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks = []
        for path in self.docs_dir.rglob("*"):
            if path.suffix.lower() == ".txt":
                text = path.read_text(encoding="utf-8", errors="ignore")
            elif path.suffix.lower() == ".pdf":
                text = self._read_pdf_text(path)
            else:
                continue
            chunks.extend(splitter.split_text(text))
        return chunks

    def _embed(self, texts: list[str]) -> list[list[float]]:
        res = openai.embeddings.create(model=self.embed_model, input=texts)
        return [d.embedding for d in res.data]

    def _prepare_store(self):
        # Vérifie si l'index existe déjà pour gagner du temps
        if Path(self.index_path).exists() and Path(self.meta_path).exists():
            print("✓ Chargement de l'index FAISS existant...")
            index = faiss.read_index(self.index_path)
            chunks = json.loads(Path(self.meta_path).read_text())
            return index, chunks

        print("⏳ Construction de l'index (cela peut prendre du temps)...")
        chunks = self._load_and_split()
        if not chunks:
            print("❌ Aucun document trouvé !")
            return None, []
        
        all_vectors = []
        # Envoi par paquets de 128 pour l'efficacité
        for i in range(0, len(chunks), 128):
            all_vectors.extend(self._embed(chunks[i : i + 128]))
        
        mat = np.asarray(all_vectors, dtype=np.float32)
        index = faiss.IndexFlatL2(mat.shape[1])
        index.add(mat)
        
        # Sauvegarde sur le disque
        faiss.write_index(index, self.index_path)
        Path(self.meta_path).write_text(json.dumps(chunks))
        print(f"✅ Index créé et sauvegardé avec {len(chunks)} segments.")
        return index, chunks

    def run(self, query: str, k: int = 4) -> str:
        """
        Recherche des explications théoriques, des définitions et des concepts de Machine Learning et Deep Learning. 
        À utiliser pour répondre aux questions de type "C'est quoi ?" ou "Expliquez le fonctionnement de...".
        """
        if not self.index:
            return "Désolé, la base documentaire est vide."

        # 1. Transformer la question en vecteur
        q_vec = np.asarray(self._embed([query])[0], dtype=np.float32).reshape(1, -1)
        
        # 2. Rechercher les morceaux les plus proches
        _, idxs = self.index.search(q_vec, k)
        retrieved_chunks = [self.chunks[i] for i in idxs[0]]
        
        # 3. Retourner le texte brut (l'agent s'occupera de répondre)
        return "\n\n--- DOCUMENT SOURCE ---\n".join(retrieved_chunks)

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
