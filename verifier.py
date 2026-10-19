import json
import ollama
from config import SMALL_MODEL
from metrics import record_call

SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "same": {"type": "boolean"},
    },
    "required": ["reason", "same"],
}

SYSTEM = """You judge whether two user questions are duplicates: would ONE answer fully and correctly serve BOTH?
Ignore differences in wording, phrasing, word order, politeness, abbreviations, and whether the text is a full question or a short phrase. Those do NOT matter.
Answer same=false only if the questions are about different subjects, or ask for different things (for example benefits vs risks, install vs uninstall, how to do X vs how to use X).
Write a one-sentence reason first, then the verdict as JSON.

Q1: How do I boil an egg?
Q2: What's the way to cook a hard-boiled egg?
{"reason": "Both ask how to cook a hard-boiled egg; only the wording differs.", "same": true}

Q1: What are the benefits of exercise?
Q2: What are the risks of exercise?
{"reason": "One asks for benefits, the other for risks.", "same": false}

Q1: Symptoms of the flu
Q2: What are the symptoms of the flu?
{"reason": "Both ask for the symptoms of the flu; one is just a shorter phrase.", "same": true}

Q1: What is Java?
Q2: What is the island of Java?
{"reason": "One is a programming language, the other an island.", "same": false}

Q1: Which mountain is the tallest?
Q2: What is the highest mountain?
{"reason": "Tallest and highest ask for the same mountain.", "same": true}

Q1: How do I install Docker?
Q2: How do I uninstall Docker?
{"reason": "Installing and uninstalling are opposite tasks.", "same": false}
"""


def same_question(q1, q2, model=SMALL_MODEL):
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Q1: {q1}\nQ2: {q2}"},
            ],
            format=SCHEMA,
            options={"temperature": 0, "num_predict": 150},
        )
        record_call("verifier", model, resp)
        data = json.loads(resp["message"]["content"])
        return bool(data["same"]), data["reason"]
    except Exception as e:
        return False, f"ERROR: {e}"  # when unsure, treat as a cache miss
