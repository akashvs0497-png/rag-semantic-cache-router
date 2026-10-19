import sqlite3, time
from config import SMALL_MODEL, LARGE_MODEL

DB_PATH = "./metrics.db"

# Illustrative API-equivalent prices in USD per 1M tokens. Edit to match any provider.
PRICES = {
    SMALL_MODEL: {"in": 1.00, "out": 5.00},
    LARGE_MODEL: {"in": 3.00, "out": 15.00},
}

_calls = []


def start_request():
    _calls.clear()


def record_call(kind, model, resp):
    _calls.append({
        "kind": kind,
        "model": model,
        "in": getattr(resp, "prompt_eval_count", 0) or 0,
        "out": getattr(resp, "eval_count", 0) or 0,
    })


def request_cost():
    tin = tout = 0
    cost = 0.0
    for c in _calls:
        p = PRICES[c["model"]]
        cost += c["in"] * p["in"] / 1e6 + c["out"] * p["out"] / 1e6
        tin += c["in"]
        tout += c["out"]
    return tin, tout, cost


def log_request(query, r):
    tin, tout, cost = request_cost()
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, query TEXT, source TEXT,
        model TEXT, similarity REAL, latency REAL, tokens_in INTEGER,
        tokens_out INTEGER, cost_usd REAL, note TEXT)""")
    con.execute(
        "INSERT INTO requests (ts, query, source, model, similarity, latency, "
        "tokens_in, tokens_out, cost_usd, note) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (time.time(), query, r["source"], r["model"], r["similarity"],
         r["latency"], tin, tout, cost, r["note"]),
    )
    con.commit()
    con.close()
    return tin, tout, cost
