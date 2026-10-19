import time
from router import heuristic_route, llm_route

DATA = [
    ("What is a vector database?", "small"),
    ("Define word embedding", "small"),
    ("Who created Python?", "small"),
    ("What does RAG stand for?", "small"),
    ("What is a cache?", "small"),
    ("What is a transformer in deep learning?", "small"),
    ("What is tokenization?", "small"),
    ("What is Python used for?", "small"),
    ("When was Python first released?", "small"),
    ("What is word2vec?", "small"),
    ("Compare RAG with fine-tuning for adding knowledge to an LLM", "large"),
    ("Why do transformers handle long-range dependencies better than RNNs?", "large"),
    ("What are the trade-offs between a write-through and a write-back cache?", "large"),
    ("Explain how embeddings, vector databases, and LLMs work together in a RAG system", "large"),
    ("How does attention differ from a word embedding, and why does it matter for NLP?", "large"),
    ("What are the advantages and disadvantages of vector databases compared to traditional databases?", "large"),
    ("Design a caching strategy for an LLM app and explain its risks", "large"),
    ("How does tokenization affect how a large language model handles rare words, and what problems does that cause?", "large"),
    ("Summarize how NLP has evolved from word embeddings to large language models", "large"),
    ("Why might a RAG system hallucinate even when relevant documents are retrieved?", "large"),
]

llm_route("warm up")  # load the model first so timing is fair

for name in ["heuristic", "llm"]:
    correct = hard_missed = easy_wasted = went_large = 0
    start = time.time()
    print(f"\n===== {name} router =====")
    for q, label in DATA:
        if name == "heuristic":
            got, reason = heuristic_route(q), ""
        else:
            got, reason = llm_route(q)
        went_large += got == "large"
        if got == label:
            correct += 1
        else:
            if label == "large":
                hard_missed += 1
                kind = "HARD Q SENT TO SMALL (quality risk)"
            else:
                easy_wasted += 1
                kind = "EASY Q SENT TO LARGE (wasted cost)"
            print(f"  {kind}: {q}\n      {reason[:80]}")
    n = len(DATA)
    avg = (time.time() - start) / n
    print(f"{correct}/{n} correct | hard->small: {hard_missed} | easy->large: {easy_wasted} | "
          f"sent to large: {went_large}/{n} | avg {avg:.2f}s per route")
