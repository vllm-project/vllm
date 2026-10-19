"""Needle-in-haystack recall test — le discriminant qualité d'un KV 4-bit.

Prompt ~50K tokens (filler répétitif), un code unique inséré à 25 % puis 75 %
de profondeur. Réponse attendue : le code exact. Un KV quantifié 4-bit qui
dégrade le recall long-contexte se voit ici avant les benchmarks de vitesse.

Usage: python needle_test.py
"""
import json
import time
import urllib.request

API = "http://localhost:5002/v1/chat/completions"
TOKENIZE = "http://localhost:5002/tokenize"
MODEL = "qwen3.6-35b-a3b"
KEY = None
for line in open("d:/vllm/myia_vllm/.env"):
    if line.startswith("VLLM_API_KEY_MEDIUM="):
        KEY = line.split("=", 1)[1].strip()
assert KEY

PHRASE = "The quick brown fox jumps over the lazy dog near the riverbank. "
CODE = "ZX-4711-QQ"


def count_tokens(prompt: str) -> int:
    req = urllib.request.Request(
        TOKENIZE,
        data=json.dumps({"model": MODEL, "prompt": prompt}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())["count"]


for depth in (0.25, 0.75):
    n = 50_000 // 14  # ~14 tok/phrase in-sequence
    cut = int(n * depth)
    prompt = (
        PHRASE * cut
        + f"\n\nIMPORTANT NOTE: the secret code is {CODE}. Memorize it.\n\n"
        + PHRASE * (n - cut)
        + "\n\nQuestion: What is the secret code mentioned in the text above? Reply with just the code."
    )
    ct = count_tokens(prompt)
    print(f"depth={depth:.0%}: prompt={ct} tokens", flush=True)
    req = urllib.request.Request(
        API,
        data=json.dumps(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.loads(r.read().decode())
    txt = body["choices"][0]["message"]["content"] or ""
    ok = CODE in txt
    print(f"  {time.time() - t0:.1f}s → {txt.strip()[:90]!r} → {'HIT' if ok else 'MISS'}", flush=True)