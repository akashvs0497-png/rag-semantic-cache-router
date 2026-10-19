# Semantic cache + cost-aware routing for RAG (fully local)

A local RAG pipeline that avoids repeat LLM calls with a semantic cache and sends each
question to a small or large model. Runs on a laptop with Ollama, no paid APIs.

## How a request flows

1. Embed the question (`all-MiniLM-L6-v2`).
2. Look up the cache (ChromaDB, cosine similarity).
   - similarity >= 0.97: return the cached answer immediately.
   - 0.80 to 0.97: an 8B model checks whether the two questions really mean the same
     thing. The cached answer is served only on a YES.
3. On a miss, the router picks a model: heuristic rules first, then a 3B classifier if the
   rules are inconclusive. Complex questions go to `llama3.1:8b`, the rest to `llama3.2:3b`.
4. Retrieve the top 4 chunks and answer using only that context (temperature 0).
5. Log latency, tokens, route and an API-equivalent cost to SQLite; view it in a
   Streamlit dashboard.

## Why there is a verifier

Embedding similarity alone could not separate paraphrases from opposites. On a 12-pair test,
"Advantages of RAG" vs "Disadvantages of RAG" scored 0.937, higher than a true paraphrase
("uses of a vector database" vs "uses of a vector DB", 0.899). No single threshold worked,
so candidates found by similarity are verified by an LLM before being served.

## Results

24 queries over 8 Wikipedia articles: 15 unique (8 simple, 7 hard) plus 9 follow-ups
(3 exact repeats, 4 paraphrases, 2 look-alike traps). 7 of 24 requests (29%) are duplicates
by design. Temperature 0, so cost and quality are reproducible.

| Setup | Est. cost | Saved vs A | Avg latency | p95 | Key-term coverage |
|---|---|---|---|---|---|
| A: 8B only, no cache | $0.0903 | n/a | 3.61s | 8.33s | 0.83 |
| B: 3B only, no cache | $0.0313 | 65% | 2.40s | 5.00s | 0.94 |
| C: full pipeline | $0.0619 | 31% | 3.23s | 8.01s | 0.87 |

Findings:

- The cache served 7 hits, all correct, and rejected both look-alike traps
  (e.g. "What does LLM stand for?" vs a cached "What does RAG stand for?").
- The pipeline cost about 31% less than always using the 8B model, roughly two-thirds from
  the cache and one-third from routing.
- The router did not pay off on this corpus. 3B-only was cheaper (65% saved) with similar
  coverage. Routing helps only when the small model actually fails, and on Wikipedia-intro
  questions it mostly does not.
- A verified cache hit costs about the same as a 3B answer (the verifier is an 8B call).
  Only fast-path hits (exact or near-exact repeats) are free.
- Latency differences between setups are within run-to-run noise: the same setup's average
  varied from 6.1s to 3.6s between identical runs. The one latency result I rely on is that
  exact-repeat cache hits return in 0.01 to 0.2s.

## Limitations

- Small, simulated workload. The hit rate and savings depend on the assumed 29% duplicates.
- Costs are API-equivalent estimates from local token counts, using illustrative prices
  (small $1/$5, large $3/$15 per million tokens in/out), not real spend.
- Quality is scored by key-term coverage on 23 answerable questions. This is crude: a gap of
  about 2 to 3 questions is inside the noise, and it does not measure answer correctness
  in full.
- The verifier and router prompts were tuned on the same pairs they were first tested on;
  only a small held-out set (8 pairs) was kept separate.
- Embeddings alone miss abbreviation pairs (e.g. "RAG" vs "retrieval-augmented generation"
  scored 0.13), so those never reach the verifier.

## Run it

    brew install ollama
    OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=30m ollama serve   # separate terminal
    ollama pull llama3.2:3b && ollama pull llama3.1:8b
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python fetch_docs.py       # downloads 8 Wikipedia articles into data/
    python ingest.py           # builds the vector store
    python evaluate.py         # runs the 3-setup comparison (about 10 minutes)
    streamlit run dashboard.py
