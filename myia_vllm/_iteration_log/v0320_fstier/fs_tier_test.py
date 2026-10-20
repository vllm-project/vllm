"""Éval tier fs NVMe (TieringOffloadingSpec) — harnais 3 phases.

Phases :
  seed     gold (code secret, ~17K tok) puis N fillers UNIQUES (~19K tok) pour
           évincer gold du GPU ET du CPU primaire (8 GiB en fenêtre) -> fs.
  replay   re-envoie gold EXACT, mesure latence, vérifie le code dans la réponse.
  metrics  imprime les compteurs d'offload (bytes GPU_to_CPU / CPU_to_GPU).

Usage : python fs_tier_test.py seed|replay|metrics [--n-fillers 26] [--conc 4]

Lecture attendue : replay rapide + code correct + CPU_to_GPU > 0 (chargement
depuis le tier) = chemin fs->CPU->GPU prouvé. Après un restart du moteur, le
tmpfs CPU est vide (fs survit) : le replay post-restart est LE test de survie.
"""
import argparse
import concurrent.futures
import json
import sys
import time
import urllib.request

API = "http://localhost:5002/v1/chat/completions"
METRICS = "http://localhost:5002/metrics"
MODEL = "qwen3.6-35b-a3b"
KEY = None
for line in open("d:/vllm/myia_vllm/.env"):
    if line.startswith("VLLM_API_KEY_MEDIUM="):
        KEY = line.split("=", 1)[1].strip()
assert KEY, "VLLM_API_KEY_MEDIUM absent de .env"

PHRASE = "The quick brown fox jumps over the lazy dog near the riverbank. "
CODE = "FS-TIER-90210"
RUN = str(int(time.time()))  # unicité des fillers entre deux runs


def chat(prompt: str, max_tokens: int = 8, timeout: int = 900) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        API,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    return {"latency": time.time() - t0, "body": data}


def gold_prompt() -> str:
    n = 17_400 // 14
    return (
        PHRASE * (n // 2)
        + f"\n\nIMPORTANT: the secret code is {CODE}. Memorize it.\n\n"
        + PHRASE * (n - n // 2)
        + "\n\nQuestion: What is the secret code mentioned above? Reply with just the code."
    )


def filler_prompt(i: int) -> str:
    n = 19_000 // 14
    # Préfixe unique par filler ET par run : zéro partage de prefix cache.
    return f"DOCUMENT {RUN}-{i} unique.\n" + PHRASE * n


def seed(n_fillers: int, conc: int) -> None:
    print(f"[seed] run={RUN} gold…", flush=True)
    r = chat(gold_prompt(), max_tokens=64)
    txt = (r["body"]["choices"][0]["message"]["content"] or "").strip()
    print(f"  gold {r['latency']:.1f}s -> {txt[:60]!r} code={'OK' if CODE in txt else 'ABSENT'}", flush=True)
    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
        futs = {ex.submit(chat, filler_prompt(i), 8): i for i in range(n_fillers)}
        for f in concurrent.futures.as_completed(futs):
            done += 1
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                print(f"  filler {futs[f]} ERREUR: {e}", flush=True)
            if done % 5 == 0:
                print(f"  fillers {done}/{n_fillers} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[seed] fillers done en {time.time()-t0:.0f}s — attente 25 s (flush des écritures fs)…", flush=True)
    time.sleep(25)


def fill(n_fillers: int, conc: int) -> None:
    """Fillers uniques UNIQUEMENT (aucun gold).

    Nécessaire quand le GPU KV est gros : avec KV nvfp4 = 750 747 tokens, les
    26 fillers du seed (494K) ne suffisent PAS à évincer le gold du cache GPU
    (vécu 20/10 : replay 1,26 s mais CPU_to_GPU = 0 = hit GPU, pas tier).
    """
    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
        futs = {ex.submit(chat, filler_prompt(i), 8): i for i in range(n_fillers)}
        for f in concurrent.futures.as_completed(futs):
            done += 1
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                print(f"  filler {futs[f]} ERREUR: {e}", flush=True)
            if done % 5 == 0:
                print(f"  fillers {done}/{n_fillers} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[fill] done en {time.time()-t0:.0f}s — attente 25 s (flush)…", flush=True)
    time.sleep(25)


def replay() -> None:
    print("[replay] gold exact…", flush=True)
    r = chat(gold_prompt(), max_tokens=64)
    txt = (r["body"]["choices"][0]["message"]["content"] or "").strip()
    hit = CODE in txt
    print(f"  replay {r['latency']:.2f}s -> {txt[:60]!r} code={'HIT' if hit else 'MISS'}", flush=True)
    sys.exit(0 if hit else 2)


def metrics() -> None:
    with urllib.request.urlopen(METRICS, timeout=30) as r:
        txt = r.read().decode()
    for line in txt.splitlines():
        if "kv_offload_total_bytes_total" in line or "kv_offload_total_time_total" in line:
            print(line, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["seed", "fill", "replay", "metrics"])
    ap.add_argument("--n-fillers", type=int, default=26)
    ap.add_argument("--conc", type=int, default=4)
    a = ap.parse_args()
    {"seed": lambda: seed(a.n_fillers, a.conc), "fill": lambda: fill(a.n_fillers, a.conc),
     "replay": replay, "metrics": metrics}[a.phase]()
