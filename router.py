import json
import ollama
from config import SMALL_MODEL, LARGE_MODEL
from metrics import record_call

STRONG = ["compare", "difference between", "versus", " vs ", "trade-off", "tradeoff",
          "pros and cons", "advantages and", "step by step", "relationship",
          "evaluate", "analyze", "analyse", "design", "summarize how"]
WEAK = ["why", "how does", "how do", "explain", "what problems", "might"]


def heuristic_route(query):
    q = query.lower()
    words = len(q.split())
    score = 0
    if any(w in q for w in STRONG):
        score += 2
    if any(w in q for w in WEAK):
        score += 1
    if words > 18:
        score += 1
    if q.count("?") > 1 or (" and " in q and words > 12):
        score += 1
    return "large" if score >= 2 else "small"


SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "complexity": {"type": "string", "enum": ["simple", "complex"]},
    },
    "required": ["reason", "complexity"],
}

SYSTEM = """Classify how hard a question is to answer well.
simple = a single fact, definition, or short lookup.
complex = needs comparison, multiple parts, reasoning about causes or trade-offs, or combining several ideas.
Write a one-sentence reason first, then the classification as JSON.

Q: Who wrote Hamlet?
{"reason": "A single fact lookup.", "complexity": "simple"}

Q: Compare solar and wind power for a small town and recommend one.
{"reason": "Needs comparison and a reasoned recommendation.", "complexity": "complex"}

Q: What is photosynthesis?
{"reason": "A basic definition.", "complexity": "simple"}

Q: Why did the Roman Empire fall, and how did it affect later European politics?
{"reason": "Two linked parts needing causal reasoning.", "complexity": "complex"}
"""


def llm_route(query, model=SMALL_MODEL):
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Q: {query}"},
            ],
            format=SCHEMA,
            options={"temperature": 0, "num_predict": 80},
        )
        record_call("router", model, resp)
        data = json.loads(resp["message"]["content"])
        return ("large" if data["complexity"] == "complex" else "small"), data["reason"]
    except Exception as e:
        return "large", f"ERROR: {e}"  # when unsure, use the stronger model


def route_model(choice):
    return LARGE_MODEL if choice == "large" else SMALL_MODEL


def combined_route(query):
    """Heuristic first (free). If it doesn't flag the query, ask the LLM. Escalate when unsure."""
    if heuristic_route(query) == "large":
        return "large", "heuristic: clear complexity signals"
    choice, reason = llm_route(query)
    return choice, "llm: " + reason
