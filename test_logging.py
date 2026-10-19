import os, sqlite3
from rag import ask, cache
from config import SMALL_MODEL, LARGE_MODEL
from metrics import DB_PATH

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
cache.clear()
ask("warm up", model=SMALL_MODEL, use_cache=False, log=False)
ask("warm up", model=LARGE_MODEL, use_cache=False, log=False)

for q in [
    "What is tokenization?",
    "What is tokenization?",
    "What does tokenization mean?",
    "When was Python first released?",
    "Compare RAG with fine-tuning for adding knowledge to an LLM",
    "Compare RAG with fine-tuning for adding knowledge to an LLM",
    "Why might a RAG system hallucinate even when relevant documents are retrieved?",
]:
    ask(q)

con = sqlite3.connect(DB_PATH)
print(f"\n{'src':5} {'model':12} {'sec':>6} {'tok_in':>6} {'tok_out':>7} {'cost$':>9} | query")
for src, model, lat, ti, to, cost, q in con.execute(
    "SELECT source, COALESCE(model,'-'), latency, tokens_in, tokens_out, cost_usd, query FROM requests"
):
    print(f"{src:5} {model:12} {lat:6.2f} {ti:6} {to:7} {cost:9.5f} | {q[:55]}")

n, hits, cost, lat = con.execute(
    "SELECT COUNT(*), SUM(source='cache'), SUM(cost_usd), AVG(latency) FROM requests"
).fetchone()
print(f"\n{n} requests | {hits} cache hits | est. cost ${cost:.5f} | avg latency {lat:.2f}s")
