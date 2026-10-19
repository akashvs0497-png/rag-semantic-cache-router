from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL

embedder = SentenceTransformer(EMBED_MODEL)

# (query A, query B, should they share a cached answer?)
pairs = [
    ("What is a vector database used for?", "What are the uses of a vector DB?", True),
    ("How does a transformer work?", "Explain the transformer architecture", True),
    ("What is RAG?", "What does retrieval-augmented generation mean?", True),
    ("Define word embedding", "What is a word embedding?", True),
    ("What is a cache in computing?", "Explain what caching is", True),
    ("What is Python used for?", "What are the uses of the Python language?", True),
    ("What is a cache in computing?", "How does a CPU cache work?", False),
    ("What is a vector database?", "What is a graph database?", False),
    ("What is Python?", "What is a python snake?", False),
    ("Advantages of RAG", "Disadvantages of RAG", False),
    ("How do I train a language model?", "How do I use a trained language model?", False),
    ("What is word2vec?", "What is a transformer?", False),
]

rows = []
for a, b, same in pairs:
    ea, eb = embedder.encode([a, b], normalize_embeddings=True)
    rows.append((float(ea @ eb), same, a, b))

for sim, same, a, b in sorted(rows, reverse=True):
    label = "SAME " if same else "DIFF "
    print(f"{sim:.3f}  {label} {a}  <->  {b}")

print("\nthreshold | wrongly served (DIFF hits) | missed (SAME misses)")
for t in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    wrong = sum(1 for s, same, *_ in rows if s >= t and not same)
    missed = sum(1 for s, same, *_ in rows if s < t and same)
    print(f"  {t:.2f}    |            {wrong}               |        {missed}")