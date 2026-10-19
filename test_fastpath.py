from rag import ask, cache

cache.clear()
ask("What is a cache in computing?", use_cache=False)  # warm up

for q in [
    "What is a vector database used for?",
    "What is a vector database used for?",
    "What are vector databases used for?",
    "What are the uses of a vector DB?",
]:
    r = ask(q)
    sim = f"sim={r['similarity']}" if r["similarity"] else ""
    print(f"{r['source']:5} {r['latency']:5.2f}s {sim:10} | {q}")
    if r["note"]:
        print(f"      {r['note'][:90]}")
