# (query, key terms the answer should contain, group)
BASE = [
    ("What is a vector database?", ["embeddings", "nearest neighbor"], "simple"),
    ("Who created Python?", ["Guido van Rossum"], "simple"),
    ("When was Python first released?", ["1991"], "simple"),
    ("What is a word embedding?", ["vector", "meaning"], "simple"),
    ("What is a cache in computing?", ["data", "faster"], "simple"),
    ("What does RAG stand for?", ["retrieval-augmented generation"], "simple"),
    ("What is a transformer in deep learning?", ["attention"], "simple"),
    ("What is a large language model?", ["trained", "text"], "simple"),
    ("Why do transformers handle long-range dependencies better than RNNs?", ["attention", "parallel"], "hard"),
    ("Why might a RAG system hallucinate even when relevant documents are retrieved?", ["hallucinat"], "hard"),
    ("How do vector databases support retrieval-augmented generation?", ["embedding", "retrieval"], "hard"),
    ("What are the main differences between word embeddings and transformer models?", ["word", "attention", "context"], "hard"),
    ("How does caching improve performance, and what happens on a cache miss?", ["miss", "faster"], "hard"),
    ("Explain how large language models use the transformer architecture", ["transformer", "attention"], "hard"),
    ("What are the main design features of the Python language?", ["dynamic", "garbage"], "hard"),
]
FOLLOW = [
    ("Who created Python?", ["Guido van Rossum"], "repeat"),
    ("What is a transformer in deep learning?", ["attention"], "repeat"),
    ("Why might a RAG system hallucinate even when relevant documents are retrieved?", ["hallucinat"], "repeat"),
    ("Who is the creator of the Python language?", ["Guido van Rossum"], "paraphrase"),
    ("What is a vector DB?", ["embeddings", "nearest neighbor"], "paraphrase"),
    ("Explain what a word embedding is", ["vector", "meaning"], "paraphrase"),
    ("Why can RAG systems still hallucinate despite retrieving relevant documents?", ["hallucinat"], "paraphrase"),
    ("What is a graph database?", ["graph"], "trap"),
    ("What does LLM stand for?", ["large language model"], "trap"),
]
QUESTIONS = BASE + FOLLOW
GROUPS = ["simple", "hard", "repeat", "paraphrase", "trap"]
