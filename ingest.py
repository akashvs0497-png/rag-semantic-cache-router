import os, glob
import chromadb
from sentence_transformers import SentenceTransformer
from config import *


def chunk_text(text, size=200, overlap=40):
    words = text.split()
    step = size - overlap
    chunks = []
    for i in range(0, len(words), step):
        piece = words[i:i + size]
        if piece:
            chunks.append(" ".join(piece))
    return chunks


model = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PATH)

try:
    client.delete_collection("docs")
except Exception:
    pass
col = client.create_collection("docs", metadata={"hnsw:space": "cosine"})

ids, texts, metas = [], [], []
for path in glob.glob(f"{DATA_DIR}/*.txt"):
    name = os.path.basename(path)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for i, chunk in enumerate(chunk_text(text)):
        ids.append(f"{name}-{i}")
        texts.append(chunk)
        metas.append({"source": name})

embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()
col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)
print(f"Ingested {len(texts)} chunks from {len(set(m['source'] for m in metas))} files")