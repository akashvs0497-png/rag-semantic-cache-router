import time, uuid
import chromadb
from config import *


class SemanticCache:
    def __init__(self, threshold=CACHE_THRESHOLD):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.col = self._make_collection()
        self.threshold = threshold

    def _make_collection(self):
        return self.client.get_or_create_collection("cache", metadata={"hnsw:space": "cosine"})

    def get(self, emb):
        if self.col.count() == 0:
            return None
        res = self.col.query(query_embeddings=[emb], n_results=1)
        similarity = 1 - res["distances"][0][0]
        if similarity >= self.threshold:
            return {
                "answer": res["metadatas"][0][0]["answer"],
                "similarity": similarity,
                "cached_query": res["documents"][0][0],
            }
        return None

    def put(self, query, emb, answer):
        self.col.add(
            ids=[str(uuid.uuid4())],
            documents=[query],
            embeddings=[emb],
            metadatas=[{"answer": answer, "ts": time.time()}],
        )

    def clear(self):
        self.client.delete_collection("cache")
        self.col = self._make_collection()