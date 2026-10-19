from rag import ask, cache, embedder, retrieve
from config import SMALL_MODEL, LARGE_MODEL
from questions import QUESTIONS

ABSTAIN = ["i don't know", "i do not know", "doesn't explicitly", "does not explicitly",
           "doesn't mention", "does not mention", "not mentioned", "no information",
           "doesn't state", "does not state", "isn't mentioned", "not provided"]


def norm(s):
    return s.lower().replace("-", " ").replace("\u2019", "'")


def abstained(ans):
    a = norm(ans)
    return any(p in a for p in ABSTAIN)


def coverage(ans, kws):
    a = norm(ans)
    return sum(norm(k) in a for k in kws) / len(kws)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.95 * len(xs)))]


# answerable = every key term appears in the chunks retrieval returns for that question
ANSWERABLE = []
for q, kws, g in QUESTIONS:
    emb = embedder.encode(q, normalize_embeddings=True).tolist()
    ctx = norm(" ".join(retrieve(emb)))
    ANSWERABLE.append(all(norm(k) in ctx for k in kws))

print(f"Answerable from retrieved chunks: {sum(ANSWERABLE)}/{len(QUESTIONS)}")
for (q, kws, g), a in zip(QUESTIONS, ANSWERABLE):
    if not a:
        print(f"  NOT answerable from retrieval: {q}")


def run(name, **kw):
    print(f"\n===== {name} =====")
    out = []
    for (q, kws, g), ans_ok in zip(QUESTIONS, ANSWERABLE):
        r = ask(q, log=False, **kw)
        x = {"group": g, "answerable": ans_ok, "cov": coverage(r["answer"], kws),
             "abst": abstained(r["answer"]), "cost": r["cost"],
             "lat": r["latency"], "source": r["source"]}
        out.append(x)
        tag = "cache" if r["source"] == "cache" else str(r["model"]).replace("llama", "")
        flag = "ABSTAIN" if x["abst"] else ""
        print(f"  {g:10} {tag:9} {r['latency']:6.2f}s ${r['cost']:.5f} cov={x['cov']:.2f} "
              f"{'ans' if ans_ok else 'UNANS':5} {flag:7} | {q[:42]}")
    return out


ask("warm", model=SMALL_MODEL, use_cache=False, log=False)
ask("warm", model=LARGE_MODEL, use_cache=False, log=False)

import sys
order = sys.argv[1] if len(sys.argv) > 1 else "ABC"
RES = {}
for key in order:
    if key == "A":
        RES["A"] = run("A: always 8B, no cache", model=LARGE_MODEL, use_cache=False)
    if key == "B":
        RES["B"] = run("B: always 3B, no cache", model=SMALL_MODEL, use_cache=False)
    if key == "C":
        cache.clear()
        RES["C"] = run("C: full pipeline (cache + verifier + router)")
A, B, C = RES["A"], RES["B"], RES["C"]

print("\n===== SUMMARY =====")
costA = sum(x["cost"] for x in A)
print(f"{'setup':22} {'cost':>9} {'saved':>6} {'avg s':>6} {'p95 s':>6} | "
      f"{'cov(ans)':>8} {'false abst':>10} {'outside-ctx':>11}")
for name, R in [("A: 8B only", A), ("B: 3B only", B), ("C: full pipeline", C)]:
    ans = [x for x in R if x["answerable"]]
    un = [x for x in R if not x["answerable"]]
    cost = sum(x["cost"] for x in R)
    print(f"{name:22} ${cost:8.5f} {1 - cost / costA:5.0%} {mean([x['lat'] for x in R]):6.2f} "
          f"{p95([x['lat'] for x in R]):6.2f} | {mean([x['cov'] for x in ans]):8.2f} "
          f"{sum(x['abst'] for x in ans):>5}/{len(ans):<4} {sum(not x['abst'] for x in un):>6}/{len(un):<4}")

print("\ncov(ans)    = key-term coverage on questions whose answer was retrieved")
print("false abst  = said 'I don't know' although the answer was in the retrieved text")
print("outside-ctx = answered a question whose answer was NOT retrieved (ungrounded)")

hits = [x for x in C if x["source"] == "cache"]
print(f"\nPipeline cache hits: {len(hits)}/{len(C)}")
print(f"Trap questions served from cache (should be 0): {sum(1 for x in hits if x['group'] == 'trap')}")
