from rag import ask, cache

cache.clear()
ask("What is a cache in computing?", use_cache=False)  # warm up models

queries = [
    ("Advantages of RAG", "llm"),
    ("What are the advantages of RAG?", "cache"),
    ("Disadvantages of RAG", "llm"),
    ("How do I train a language model?", "llm"),
    ("How do I use a trained language model?", "llm"),
    ("What is a vector database used for?", "llm"),
    ("What are the uses of a vector DB?", "cache"),
]

correct = 0
for q, expected in queries:
    r = ask(q)
    ok = r["source"] == expected
    correct += ok
    sim = f"sim={r['similarity']}" if r["similarity"] else ""
    print(f"{'OK ' if ok else 'BAD'} {r['source']:5} {r['latency']:5.2f}s {sim:10} | {q}")
    if r["note"]:
        print(f"      {r['note'][:100]}")
print(f"\n{correct}/{len(queries)} as expected")
