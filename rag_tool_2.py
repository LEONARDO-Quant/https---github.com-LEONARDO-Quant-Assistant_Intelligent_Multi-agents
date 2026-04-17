import os
import faiss
import json
import numpy as np
import openai
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from docx import Document 

class RAGDocumentTool:
    def __init__(self, docs_dir: str, index_name: str = "default"):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ Clé OPENAI_API_KEY manquante.")
        
        openai.api_key = self.api_key
        
        self.docs_dir = Path(docs_dir)
        self.index_path = f"{index_name}.index"
        self.meta_path = f"{index_name}.meta.json"
        
        self.embed_model = "text-embedding-3-small"
        self.chunk_size = 800
        self.chunk_overlap = 150
        
        self.index, self.chunks = self._prepare_store()

    def _read_pdf_text(self, path: Path) -> str:
        try:
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            return f"Erreur PDF {path.name}: {e}"

    def _read_docx_text(self, path: Path) -> str:
        try:
            doc = Document(path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            return f"Erreur DOCX {path.name}: {e}"

    def _load_and_split(self) -> list[dict]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        docs_data = []
    
        for path in self.docs_dir.rglob("*"):
            if path.suffix.lower() == ".pdf":
                text = self._read_pdf_text(path)
            elif path.suffix.lower() == ".docx":
                text = self._read_docx_text(path)
            elif path.suffix.lower() == ".txt":
                text = path.read_text(encoding='utf-8', errors='ignore')
            else:
                continue

            if text.strip():
                chunks = splitter.split_text(text)
                for chunk in chunks:
                    docs_data.append({"text": chunk, "source": path.name})
        
        return docs_data

    def _embed(self, docs: list) -> list[list[float]]:
        texts = [doc.get("text", "") if isinstance(doc, dict) else str(doc) for doc in docs]
        cleaned_texts = [t for t in texts if t.strip()]
        if not cleaned_texts: return []

        res = openai.embeddings.create(model=self.embed_model, input=cleaned_texts)
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

    def run(self, query: str) -> str:
        if not self.index: return "Index vide."
        
        query_vector = self._embed([query])
        query_np = np.array(query_vector).astype('float32')
        D, idxs = self.index.search(query_np, k=5)
        
        results = []
        for i in idxs[0]:
            if i != -1 and i < len(self.chunks):
                results.append(f"[SOURCE: {self.chunks[i]['source']}]\n{self.chunks[i]['text']}")
        return "\n\n---\n\n".join(results) if results else "Aucun résultat."

# --- BLOC DE TEST AVEC CHAT ---

if __name__ == "__main__":
    # 1. Initialisation
    print("🔄 Initialisation des moteurs...")
    theory_engine = RAGDocumentTool(docs_dir="./data/theory", index_name="theory")
    stats_engine = RAGDocumentTool(docs_dir="./data/stats", index_name="stats")
    
    engines = {
        "1": {"name": "Théorie", "obj": theory_engine},
        "2": {"name": "Math", "obj": stats_engine}
    }
    
    selected = "1"
    print("✅ Prêt ! Tapez '1' ou '2' pour changer de moteur, ou 'exit' pour quitter.")

    # 2. Boucle de Chat
    while True:
        mode_name = engines[selected]["name"]
        query = input(f"\n[{mode_name}] > ").strip()

        if query.lower() in ["exit", "quit", "q"]:
            break
        if query in ["1", "2"]:
            selected = query
            print(f"🔄 Switch vers : {engines[selected]['name']}")
            continue
        if not query:
            continue

        print("🔍 Recherche en cours...")
        print(f"\n{engines[selected]['obj'].run(query)}\n")