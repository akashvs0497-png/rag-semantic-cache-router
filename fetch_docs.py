import os, json, urllib.parse, urllib.request

TITLES = [
    "Retrieval-augmented generation",
    "Large language model",
    "Vector database",
    "Transformer (deep learning architecture)",
    "Word embedding",
    "Cache (computing)",
    "Natural language processing",
    "Python (programming language)",
]

os.makedirs("data", exist_ok=True)

for title in TITLES:
    params = urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1,
        "titles": title, "format": "json", "redirects": 1,
    })
    req = urllib.request.Request(
        "https://en.wikipedia.org/w/api.php?" + params,
        headers={"User-Agent": "rag-learning-project/0.1"},
    )
    with urllib.request.urlopen(req) as r:
        data = json.load(r)
    page = next(iter(data["query"]["pages"].values()))
    text = page.get("extract", "")
    fname = title.split(" (")[0].lower().replace(" ", "_").replace("-", "_") + ".txt"
    with open(os.path.join("data", fname), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"{fname}: {len(text.split())} words")