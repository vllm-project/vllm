#!/usr/bin/env python3
"""Reliability battery for the mini tier (:5003) — targets the FAILURE AXES
measured on FrogNano-4B (mission memory, vigie cycles 69-77, 2026-10), so a
challenger model is compared on exactly what fails, not on what works.

Axes (each: prompt -> verdict computed against ground truth built here):
  A0 control : unit key=value extraction            (FrogNano: strong)
  A1 counting: homogeneous count, 60 items          (FrogNano: 27 rendered / 60)
  A2 counting: heterogeneous count, 4 categories    (FrogNano: drift, 3 vs 4 cats)
  A3 derived  : arithmetic mean of 12 numbers        (FrogNano: wrong, +0.23-type)
  A4 absence  : log with 7 ERROR + 5 WARN lines     (FrogNano: fabricated absence)
  A5 condense : fixed-form digest, verbatim keys    (FrogNano: lexical corruption)
  A6 attribute: head-tag vs author-suffix           (FrogNano: reads suffix 4/4)

Prompt rules encoded (learned): never state line/item totals in the prompt
(anchor bias); content inline; thinking OFF via chat_template_kwargs (matches
the prod default and the challenger's template kwarg).

Usage:
  python mini_reliability_battery.py --base-url http://localhost:5003 \
      --model frognano-4b --api-key-env VLLM_API_KEY_MINI \
      --label frognano_baseline --out results_frognano.json
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

# --------------------------------------------------------------------------
# Ground-truth corpora (deterministic, no randomness — cross-run comparable)
# --------------------------------------------------------------------------

A1_ITEMS = [f"ITEM-{i:02d} value={100 + i}" for i in range(1, 61)]  # 60 items
A1_PROMPT = (
    "Liste de lignes d'inventaire :\n"
    + "\n".join(A1_ITEMS)
    + "\n\nCombien de lignes ITEM contient la liste ci-dessus ? "
    "Réponds avec le nombre seul, rien d'autre."
)

_A2_ROWS = [
    ("FRUIT", ["pomme", "poire", "kiwi", "mangue", "raisin", "figue"]),
    ("LEGUME", ["carotte", "poireau", "bettrave", "navet", "courge", "oseille"]),
    ("LAITIER", ["yaourt", "comte", "roquefort", "caille", "kefir", "mascarpone"]),
    ("VIANDE", ["agneau", "poulet", "dinde", "veau", "canard", "lapin"]),
]
A2_LINES = [f"{cat}: {name}" for cat, names in _A2_ROWS for name in names]
A2_PROMPT = (
    "Liste de courses, une ligne par article avec son rayon :\n"
    + "\n".join(A2_LINES)
    + "\n\nCompte les articles PAR RAYON. Réponds en JSON exactement de la forme "
    '{"FRUIT": <n>, "LEGUME": <n>, "LAITIER": <n>, "VIANDE": <n>} — nombres seuls.'
)
A2_TRUTH = {cat: len(names) for cat, names in _A2_ROWS}

A3_NUMS = [12.4, 8.7, 15.2, 3.9, 22.1, 7.5, 11.0, 19.8, 5.3, 14.6, 9.2, 6.7]
A3_TRUTH = round(sum(A3_NUMS) / len(A3_NUMS), 2)
A3_PROMPT = (
    "Relevés horaires (Go) :\n"
    + "\n".join(f"{i+1:02d}h : {v}" for i, v in enumerate(A3_NUMS))
    + "\n\nQuelle est la moyenne des valeurs listées ? Réponds avec le nombre "
    "seul, arrondi à deux décimales."
)

A4_LOG_LINES = [
    "2026-10-21 18:00:01 INFO  boot sequence start",
    "2026-10-21 18:00:04 INFO  cache mounted (42 GB)",
    "2026-10-21 18:00:09 WARN  disk slow: 12.4 MB/s",
    "2026-10-21 18:01:15 INFO  worker 1 online",
    "2026-10-21 18:01:16 ERROR timeout handshake upstream",
    "2026-10-21 18:02:40 INFO  retry scheduled",
    "2026-10-21 18:02:41 WARN  queue depth 340",
    "2026-10-21 18:03:00 INFO  worker 2 online",
    "2026-10-21 18:03:31 ERROR engine reset by peer",
    "2026-10-21 18:04:12 INFO  cache warm 61 %",
    "2026-10-21 18:04:55 ERROR crc mismatch block 0x41",
    "2026-10-21 18:05:30 WARN  ram free 3.1 GB",
    "2026-10-21 18:06:00 INFO  steady state",
    "2026-10-21 18:06:44 ERROR tls cert expired",
    "2026-10-21 18:07:02 INFO  failover engaged",
    "2026-10-21 18:07:33 WARN  latency p90 8.4 s",
    "2026-10-21 18:08:10 INFO  replica sync ok",
    "2026-10-21 18:08:51 ERROR orphan lock file",
    "2026-10-21 18:09:20 INFO  gc pass done",
    "2026-10-21 18:09:58 WARN  fan rpm 1900",
    "2026-10-21 18:10:30 INFO  snapshot saved",
    "2026-10-21 18:11:07 ERROR checksum drift zone 7",
    "2026-10-21 18:11:45 INFO  rebuild started",
    "2026-10-21 18:12:20 INFO  rebuild done",
    "2026-10-21 18:12:59 ERROR shard 3 unwritable",
    "2026-10-21 18:13:10 WARN  thermals 78 C",
    "2026-10-21 18:13:40 INFO  drain started",
    "2026-10-21 18:14:00 INFO  drained",
]
A4_TRUTH = {"ERROR": 7, "WARN": 5}
A4_PROMPT = (
    "Extrait de log machine :\n"
    + "\n".join(A4_LOG_LINES)
    + "\n\nCombien de lignes ERROR et combien de lignes WARN contient cet "
    "extrait ? Réponds en JSON exactement : {\"ERROR\": <n>, \"WARN\": <n>}. "
    "S'il n'y en a aucune, écris 0."
)

A5_STATUS = (
    "Rapport cycle : [WARN] lecteur G: tombé 15:05Z, rétabli 17:18 locale. "
    "Le moteur :5003 est resté UP, health 200 en continu, 35 degres. "
    "Miroir PG intact selon la mesure Maintenance 15:19:58Z. "
    "Garde anti-crash-loop posee 17:41 locale."
)
A5_QUOTES = ["health 200 en continu", "Miroir PG intact", "17:41 locale"]
A5_PROMPT = (
    "Texte source :\n" + A5_STATUS + "\n\nCondense en EXACTEMENT 3 lignes :\n"
    "L1: <incident en moins de 12 mots>\n"
    "L2: <etat du moteur, doit contenir la citation exacte « health 200 en "
    "continu »>\n"
    "L3: <garde posée, doit contenir les citations exactes « Miroir PG intact » "
    "et « 17:41 locale »>\n"
    "Ne reformule pas les citations, recopie-les telles quelles."
)

A6_ROWS = [
    ("DONE", "myia-po-2025|vllm"),
    ("ACK", "myia-po-2028|Maintenance"),
    ("INFO", "myia-ai-01|vllm"),
    ("DONE", "myia-po-2023|CoursIA"),
    ("ASK", "myia-po-2028|CoursIA-3"),
    ("WARN", "myia-po-2025|claudish"),
]
A6_LINES = [f"[{tag}] {author} — message de cycle {i+1}" for i, (tag, author) in enumerate(A6_ROWS)]
A6_PROMPT = (
    "Messages de dashboard :\n"
    + "\n".join(A6_LINES)
    + "\n\nPour chaque ligne, donne le TAG (le mot crocheté EN TÊTE de ligne, "
    "pas le suffixe d'auteur) et l'auteur. Réponds en JSON : liste de "
    "{\"tag\": \"...\", \"author\": \"...\"} dans l'ordre."
)

AXES = ["A0_control_kv", "A1_count_homog_60", "A2_count_heterog",
        "A3_derived_mean", "A4_absence_log", "A5_condense_verbatim",
        "A6_tag_attribution"]


# --------------------------------------------------------------------------
# Engine client
# --------------------------------------------------------------------------

def chat(base_url, api_key, model, user_msg, max_tokens=300, timeout=120):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": user_msg}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    dt = time.time() - t0
    choice = data["choices"][0]
    content = choice["message"].get("content") or ""
    return content.strip(), dt, data.get("usage", {})


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------

def verdict_a0(text):
    # Control: extract two known key=values from clean input
    p = "Capteur A : temp=41.5C ; Capteur B : free=22107MB"
    ok = "41.5C" in text and "22107MB" in text
    return "PASS" if ok else "FAIL", {"expects": ["41.5C", "22107MB"]}


def verdict_a1(text):
    m = re.search(r"\b(\d{1,4})\b", text)
    n = int(m.group(1)) if m else None
    if n == 60:
        return "PASS", {"count": n, "truth": 60}
    return "FAIL", {"count": n, "truth": 60}


def verdict_a2(text):
    try:
        # tolerate code fences
        m = re.search(r"\{.*\}", text, re.S)
        obj = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        obj = {}
    got = {k: obj.get(k) for k in A2_TRUTH}
    exact = all(got[k] == v for k, v in A2_TRUTH.items())
    return ("PASS" if exact else "FAIL"), {"got": got, "truth": A2_TRUTH}


def verdict_a3(text):
    m = re.search(r"(\d+[.,]\d+)", text)
    if not m:
        return "FAIL", {"got": None, "truth": A3_TRUTH}
    got = float(m.group(1).replace(",", "."))
    ok = abs(got - A3_TRUTH) <= 0.01
    return ("PASS" if ok else "FAIL"), {"got": got, "truth": A3_TRUTH}


def verdict_a4(text):
    try:
        m = re.search(r"\{.*\}", text, re.S)
        obj = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        obj = {}
    got = {"ERROR": obj.get("ERROR"), "WARN": obj.get("WARN")}
    # The critical failure mode: claiming absence (0) when lines exist.
    fabricated_absence = got["ERROR"] == 0 or got["WARN"] == 0
    exact = got == A4_TRUTH
    v = "PASS" if exact else ("FAIL" if fabricated_absence else "PARTIAL")
    return v, {"got": got, "truth": A4_TRUTH,
               "fabricated_absence": fabricated_absence}


def verdict_a5(text):
    present = {q: (q in text) for q in A5_QUOTES}
    v = "PASS" if all(present.values()) else "PARTIAL"
    return v, {"quotes_present": present}


def verdict_a6(text):
    try:
        m = re.search(r"\[.*\]", text, re.S)
        arr = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        arr = []
    got = [{"tag": r.get("tag"), "author": r.get("author")} for r in arr][:len(A6_ROWS)]
    tag_ok = sum(1 for g, (t, _) in zip(got, A6_ROWS) if str(g.get("tag", "")).upper() == t)
    author_ok = sum(1 for g, (_, a) in zip(got, A6_ROWS) if g.get("author") == a)
    n = len(A6_ROWS)
    v = "PASS" if tag_ok == n and author_ok == n else ("FAIL" if tag_ok <= n // 2 else "PARTIAL")
    return v, {"tag_ok": f"{tag_ok}/{n}", "author_ok": f"{author_ok}/{n}",
               "got": got, "truth": [{"tag": t, "author": a} for t, a in A6_ROWS]}


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:5003")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="VLLM_API_KEY_MINI")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import os
    key = os.environ.get(args.api_key_env, "")
    if not key:
        print(f"ERROR: env {args.api_key_env} empty", file=sys.stderr)
        sys.exit(2)

    a0_prompt = ("Extrait du registre : Capteur A : temp=41.5C ; "
                 "Capteur B : free=22107MB\n\nDonne la valeur de temp et la "
                 "valeur de free, telles quelles, sur une ligne : "
                 "temp=... free=...")

    cases = [
        ("A0_control_kv", a0_prompt, verdict_a0, 200),
        ("A1_count_homog_60", A1_PROMPT, verdict_a1, 300),
        ("A2_count_heterog", A2_PROMPT, verdict_a2, 300),
        ("A3_derived_mean", A3_PROMPT, verdict_a3, 300),
        ("A4_absence_log", A4_PROMPT, verdict_a4, 300),
        ("A5_condense_verbatim", A5_PROMPT, verdict_a5, 400),
        ("A6_tag_attribution", A6_PROMPT, verdict_a6, 500),
    ]

    results = {"label": args.label, "model": args.model,
               "base_url": args.base_url, "axes": {}}
    score = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
    for name, prompt, verdict_fn, mt in cases:
        rec = {}
        try:
            text, dt, usage = chat(args.base_url, key, args.model, prompt,
                                   max_tokens=mt)
            v, detail = verdict_fn(text)
            rec = {"verdict": v, "latency_s": round(dt, 2),
                   "completion_tokens": usage.get("completion_tokens"),
                   "raw": text[:600], "detail": detail}
        except Exception as e:  # noqa: BLE001 — a dead axis is a result
            rec = {"verdict": "FAIL", "error": repr(e)[:300]}
        score[rec["verdict"]] += 1
        results["axes"][name] = rec
        print(f"[{rec['verdict']:7s}] {name:22s} "
              f"({rec.get('latency_s', '?')}s, "
              f"{rec.get('completion_tokens', '?')} tok)")

    results["score"] = score
    print(f"\n{args.label}: {score['PASS']} PASS / {score['PARTIAL']} PARTIAL / "
          f"{score['FAIL']} FAIL")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"written: {args.out}")


if __name__ == "__main__":
    main()
