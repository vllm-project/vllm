"""A/B the serving path: distinct prompts, streaming, TTFT kept apart from
decode rate.

A bench that repeats one prompt measures the prefix cache, and timing the whole
request folds seconds of prefill into the token rate. Neither is what we want
to compare here.
"""

import json
import sys
import time

import requests

import os
URL = os.environ.get("BENCH_URL", "http://localhost:8000/v1/chat/completions")
TAG = sys.argv[1] if len(sys.argv) > 1 else "?"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

PROMPTS = [
    "Write a Python function that merges two sorted linked lists. Explain the invariant.",
    "Explica como funciona el protocolo Raft para eleccion de lider, paso a paso.",
    "Implement a thread-safe LRU cache in Rust. Discuss the lock granularity.",
    "Describe como se propaga el gradiente en una capa de atencion multi-cabeza.",
    "Write a SQL query to find the second highest salary per department, and explain it.",
    "Disena un esquema de base de datos para una plataforma de reservas de hotel.",
]


def one(prompt):
    body = {
        "model": "INCCODER",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    r = requests.post(URL, json=body, stream=True, timeout=300)
    for line in r.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        payload = line[6:]
        if payload == b"[DONE]":
            break
        try:
            delta = json.loads(payload)["choices"][0]["delta"]
        except Exception:
            continue
        if delta.get("content") or delta.get("reasoning"):
            if ttft is None:
                ttft = time.perf_counter() - t0
            ntok += 1
    total = time.perf_counter() - t0
    if ttft is None or ntok < 2:
        return None
    return ttft, ntok / (total - ttft), ntok


ttfts, rates, toks = [], [], []
for r in range(ROUNDS):
    for p in PROMPTS:
        res = one(p)
        if res is None:
            print("  (peticion sin tokens, descartada)")
            continue
        ttfts.append(res[0])
        rates.append(res[1])
        toks.append(res[2])


def med(v):
    v = sorted(v)
    return v[len(v) // 2]


print(
    f"[{TAG}]  n={len(rates)}  decode mediana={med(rates):.2f} tok/s"
    f"  (min {min(rates):.2f} max {max(rates):.2f})"
    f"   TTFT mediana={med(ttfts) * 1000:.0f} ms"
    f"   tokens/resp mediana={med(toks)}"
)
