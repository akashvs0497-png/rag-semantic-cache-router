from verifier import same_question
from config import SMALL_MODEL, LARGE_MODEL

DEV = [
    ("What is a vector database used for?", "What are the uses of a vector DB?", True),
    ("How does a transformer work?", "Explain the transformer architecture", True),
    ("What is RAG?", "What does retrieval-augmented generation mean?", True),
    ("Define word embedding", "What is a word embedding?", True),
    ("What is a cache in computing?", "Explain what caching is", True),
    ("What is Python used for?", "What are the uses of the Python language?", True),
    ("Advantages of RAG", "What are the advantages of RAG?", True),
    ("What is a cache in computing?", "How does a CPU cache work?", False),
    ("What is a vector database?", "What is a graph database?", False),
    ("What is Python?", "What is a python snake?", False),
    ("Advantages of RAG", "Disadvantages of RAG", False),
    ("How do I train a language model?", "How do I use a trained language model?", False),
    ("What is word2vec?", "What is a transformer?", False),
]

HELD_OUT = [
    ("How do I reverse a list in Python?", "What's the way to reverse a Python list?", True),
    ("What causes inflation?", "Why does inflation happen?", True),
    ("Best way to learn guitar", "How should I learn to play the guitar?", True),
    ("Why is the sky blue?", "What makes the sky look blue?", True),
    ("How do I open a bank account?", "How do I close a bank account?", False),
    ("What is the capital of France?", "What is the capital of Germany?", False),
    ("Pros of remote work", "Cons of remote work", False),
    ("How do I learn Spanish?", "How do I teach Spanish?", False),
]

for model in [SMALL_MODEL, LARGE_MODEL]:
    print(f"\n===== {model} =====")
    same_question("warm", "up", model)
    for name, pairs in [("DEV", DEV), ("HELD-OUT", HELD_OUT)]:
        total = right = missed = wrong = 0
        for a, b, exp in pairs:
            for x, y in [(a, b), (b, a)]:   # test both orders
                v, reason = same_question(x, y, model)
                total += 1
                right += (v == exp)
                if v != exp:
                    if exp: missed += 1
                    else: wrong += 1
                    print(f"  {'MISSED' if exp else 'WRONG HIT'} | {x} <-> {y}\n      {reason[:90]}")
        print(f"{name}: {right}/{total} correct | wrong hits: {wrong} | missed paraphrases: {missed}\n")
