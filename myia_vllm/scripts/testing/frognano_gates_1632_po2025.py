#!/usr/bin/env python3
"""frognano_gates_1632_po2025.py — gates de concurrence N=16/32 du siège po-2025 (issue #70 P2).

Objectif : plafond maxConcurrency définitif pour le hub claudish. Borné (~5-10 min GPU) :
warm-up + N=16 + N=32 concurrentes de 128 tokens, payloads no-thinking (forme canonique
du trafic preset mini — cf. #4085 : le 4B à thinking ON brûle tout budget en reasoning).

Critères (définis avant exécution) :
  - 0 erreur HTTP, toutes finish=stop ;
  - temp GPU soutenable < 85 °C (gouverneur SYSTEM actif, cap 88 °C) ;
  - débit médian par flux >= 10 tok/s au N retenu ;
  - maxConcurrency recommandé = N/2 (marge : prompts réels preset = 1-3 K tok de prefill,
    le banc utilise des prompts courts).

Usage (host) :  python frognano_gates_1632_po2025.py [--url http://localhost:5003]
Sortie : résumé console + JSON horodaté dans myia_vllm/_logs/.
Clé lue dans myia_vllm/.env (VLLM_API_KEY_MINI) — jamais en argument, jamais affichée.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ENV_PATH = os.path.join(REPO, "myia_vllm", ".env")
LOG_DIR = os.path.join(REPO, "myia_vllm", "_logs")
MODEL = os.environ.get("BENCH_MODEL", "frognano-4b")


def load_key() -> str:
    with open(ENV_PATH, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"VLLM_API_KEY_MINI=(\S+)", line.strip())
            if m:
                return m.group(1)
    sys.exit("VLLM_API_KEY_MINI introuvable dans myia_vllm/.env")


def gpu_state() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=10).split(",")[0:4]
        return {"temp_c": int(out[0]), "util_pct": int(out[1]),
                "vram_used_mib": int(out[2]), "vram_total_mib": int(out[3])}
    except Exception as e:  # nvidia-smi absent/lent = non bloquant
        return {"error": str(e)}


def post(url: str, key: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def one_completion(base: str, key: str, max_tokens: int = 128) -> dict:
    payload = {"model": MODEL, "max_tokens": max_tokens, "temperature": 0.6,
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content":
                             "Écris une fonction Python `is_leap(year)` avec docstring, puis 3 assertions."}]}
    t0 = time.time()
    r = post(base + "/v1/chat/completions", key, payload)
    dt = time.time() - t0
    usage = r.get("usage", {})
    ct = usage.get("completion_tokens", 0)
    return {"e2e_s": round(dt, 2), "completion_tokens": ct,
            "tok_s": round(ct / dt, 1) if dt > 0 else 0,
            "finish": r.get("choices", [{}])[0].get("finish_reason")}


def phase(label: str, base: str, key: str, n: int) -> dict:
    print(f"\n=== {label} (N={n}) — avant: {gpu_state()}")
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(lambda _: one_completion(base, key), range(n)))
    dt = time.time() - t0
    tot = sum(r["completion_tokens"] for r in results)
    ok = sum(1 for r in results if r["finish"] == "stop")
    agg = round(tot / dt, 1) if dt > 0 else 0
    rates = [r["tok_s"] for r in results if r["tok_s"] > 0]
    med = round(statistics.median(rates), 1) if rates else 0
    mn = min(rates) if rates else 0
    print(f"    agrégat {agg} tok/s | médiane/flux {med} | min {mn} | {ok}/{n} stop | apres: {gpu_state()}")
    return {"n": n, "wall_s": round(dt, 2), "aggregate_tok_s": agg, "ok_stop": ok,
            "per_stream_med_tok_s": med, "per_stream_min_tok_s": mn,
            "total_tokens": tot, "per_request": results}


def post_watch(seconds: int = 30) -> list:
    """Échantillonne le GPU après la dernière phase (inertie thermique)."""
    samples = []
    for _ in range(seconds // 5):
        time.sleep(5)
        samples.append(gpu_state())
        print(f"    [watch] {samples[-1]}")
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:5003")
    args = ap.parse_args()
    key = load_key()
    os.makedirs(LOG_DIR, exist_ok=True)

    req = urllib.request.Request(args.url + "/health")
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.status == 200, "moteur non sain"
    idle = gpu_state()

    report = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "url": args.url, "gpu_idle": idle,
              "payload": "no-thinking (preset canonique)"}
    print(f"[gates] cible {args.url} — idle {idle}")
    report["warmup"] = one_completion(args.url, key)
    print(f"    warmup {report['warmup']}")
    report["n16"] = phase("Gate N=16", args.url, key, 16)
    report["n32"] = phase("Gate N=32", args.url, key, 32)
    report["post_watch"] = post_watch(30)
    report["gpu_after_watch"] = gpu_state()
    temps = [s.get("temp_c", 0) for s in report["post_watch"] if "temp_c" in s]
    report["max_temp_watch_c"] = max(temps) if temps else None

    out = os.path.join(LOG_DIR, f"frognano_gates_1632_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[gates] rapport: {out}")
    print(f"[gates] temp max watch: {report['max_temp_watch_c']}C")


if __name__ == "__main__":
    main()
