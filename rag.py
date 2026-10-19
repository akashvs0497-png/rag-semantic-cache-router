import sys, time
import chromadb
import ollama
from sentence_transformers import SentenceTransformer
from config import *
from cache import SemanticCache
from verifier import same_question
from router import combined_route, route_model
from metrics import start_request, record_call, log_request, request_cost

embedder = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PATH)
docs = client.get_collection("docs")
cache = SemanticCache()


def retrieve(emb, k=TOP_K):
    res = docs.query(query_embeddings=[emb], n_results=k)
    return res["documents"][0]


def generate(query, chunks, model):
    context = "\n\n".join(chunks)
    prompt = (
        "Answer the question using only the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )
    resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}],
                       options={"temperature": 0, "seed": 0})
    record_call("generate", model, resp)
    return resp["message"]["content"]


def _ask(query, model, use_cache):
    start = time.time()
    emb = embedder.encode(query, normalize_embeddings=True).tolist()
    note = ""
    similarity = None

    if use_cache:
        cand = cache.get(emb)
        if cand:
            similarity = round(cand["similarity"], 3)
            if similarity >= FAST_PATH:
                same, reason = True, "near-exact match, verifier skipped"
            else:
                same, reason = same_question(query, cand["cached_query"], VERIFIER_MODEL)
            if same:
                return {
                    "answer": cand["answer"], "source": "cache", "model": None,
                    "similarity": similarity, "note": reason,
                    "latency": time.time() - start,
                }
            note = "REJECTED by verifier: " + reason

    if model is None:
        choice, route_note = combined_route(query)
        model = route_model(choice)
        note = (note + " | " if note else "") + "route: " + route_note
    else:
        note = (note + " | " if note else "") + "route: model forced"

    chunks = retrieve(emb)
    answer = generate(query, chunks, model)
    if use_cache:
        cache.put(query, emb, answer)
    return {
        "answer": answer, "source": "llm", "model": model,
        "similarity": similarity, "note": note,
        "latency": time.time() - start,
    }


def ask(query, model=None, use_cache=True, log=True):
    """model=None lets the router choose. Pass a model name to force one."""
    start_request()
    r = _ask(query, model, use_cache)
    if log:
        r["tokens_in"], r["tokens_out"], r["cost"] = log_request(query, r)
    else:
        r["tokens_in"], r["tokens_out"], r["cost"] = request_cost()
    return r


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is a vector database used for?"
    r = ask(q)
    print(r["answer"])
    print(f"\n[{r['source']} | {r['model']} | {r['latency']:.2f}s | ${r['cost']:.5f}]")
