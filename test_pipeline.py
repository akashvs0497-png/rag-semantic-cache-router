from rag import ask, cache
from config import SMALL_MODEL, LARGE_MODEL

cache.clear()
ask("warm up", model=SMALL_MODEL, use_cache=False)
ask("warm up", model=LARGE_MODEL, use_cache=False)

queries = [
    "What is tokenization?",
    "What is tokenization?",
    "When was Python first released?",
    "Compare RAG with fine-tuning for adding knowledge to an LLM",
    "Why do transformers handle long-range dependencies better than RNNs?",
    "Why might a RAG system hallucinate even when relevant documents are retrieved?",
    "What are the trade-offs between a write-through and a write-back cache?",
]

for q in queries:
    r = ask(q)
    print(f"{r['source']:5} {str(r['model']):12} {r['latency']:6.2f}s | {q}")
    if r["note"]:
        print(f"      {r['note'][:110]}")
