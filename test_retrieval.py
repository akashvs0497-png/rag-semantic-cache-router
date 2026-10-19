import chromadb
from sentence_transformers import SentenceTransformer
from config import *

embedder = SentenceTransformer(EMBED_MODEL)
docs = chromadb.PersistentClient(path=CHROMA_PATH).get_collection("docs")

query = "What is a vector database used for?"
emb = embedder.encode(query, normalize_embeddings=True).tolist()
res = docs.query(query_embeddings=[emb], n_results=3)

for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"[{meta['source']}] distance={dist:.3f}")
    print(doc[:200], "...\n")