from rag import embedder, retrieve, generate
from config import SMALL_MODEL, LARGE_MODEL
from metrics import start_request

CASES = [
    ("What does RAG stand for?", ["retrieval-augmented generation"]),
    ("Why do transformers handle long-range dependencies better than RNNs?", ["attention", "parallel"]),
    ("What does LLM stand for?", ["large language model"]),
]

for q, kws in CASES:
    emb = embedder.encode(q, normalize_embeddings=True).tolist()
    chunks = retrieve(emb)
    ctx = " ".join(chunks).lower()
    print("\n" + "=" * 70)
    print(q)
    print("key terms present in the retrieved chunks:", {k: (k.lower() in ctx) for k in kws})
    for model in [LARGE_MODEL, SMALL_MODEL]:
        start_request()
        ans = generate(q, chunks, model)
        found = {k: (k.lower() in ans.lower()) for k in kws}
        print(f"\n[{model}] found={found}\n{ans[:350]}")
