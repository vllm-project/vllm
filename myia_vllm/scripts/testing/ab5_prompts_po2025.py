#!/usr/bin/env python3
"""ab5_prompts_po2025.py — A/B qualité 5 prompts (protocole essai NVFP4 08/10, issue #70).

Mêmes 5 prompts que `_iteration_log/nvfp4_trial_po2025/NOTES.md` §4 : greedy
(temperature 0), `enable_thinking:false` par requête, max_tokens 1024.
Sortie : JSON horodaté dans myia_vllm/_logs/ + résumé console (tokens, t/s, réponse).

Usage : python ab5_prompts_po2025.py [--url http://localhost:5003] [--model frognano-4b]
Clé lue dans myia_vllm/.env (VLLM_API_KEY_MINI) — jamais en argument.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ENV_PATH = os.path.join(REPO, "myia_vllm", ".env")
LOG_DIR = os.path.join(REPO, "myia_vllm", "_logs")

PROMPTS = [
    ("fibonacci_iteratif", "Écris une fonction Python `is_leap(year)` avec docstring, puis 3 assertions."
     if False else
     "Écris une fonction Python `fibonacci(n)` itérative (pas de récursion) avec docstring, puis 3 assertions."),
    ("somme_dict_comprehension", "À partir de `ventes = {'a': 10, 'b': 25, 'c': 5}`, calcule la somme "
     "des valeurs > 8 avec une dict-comprehension en une ligne, et donne le résultat."),
    ("vitesse_moyenne", "Un train fait 240 km en 3 h puis 150 km en 2 h. Quelle est sa vitesse "
     "moyenne sur l'ensemble du trajet ? Détaille le calcul."),
    ("sql_deuxieme_salaire", "Écris la requête SQL qui renvoie le 2ᵉ salaire le plus élevé de la "
     "table `employes(salaire)`, en gérant proprement le cas des ex æquo (explique ta lecture)."),
    ("raccourcir_texte_fr", "Raccourcis ce texte en une phrase : « Le comité, après avoir longuement "
     "délibéré et pris en considération l'ensemble des éléments qui lui étaient soumis, a fini par "
     "trancher en faveur de l'option la plus prudente. »"),
]


def load_key() -> str:
    with open(ENV_PATH, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"VLLM_API_KEY_MINI=(\S+)", line.strip())
            if m:
                return m.group(1)
    sys.exit("VLLM_API_KEY_MINI introuvable dans myia_vllm/.env")


def run(base: str, key: str, model: str) -> dict:
    results = []
    for name, prompt in PROMPTS:
        payload = {"model": model, "max_tokens": 1024, "temperature": 0.0,
                   "chat_template_kwargs": {"enable_thinking": False},
                   "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(base + "/v1/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Authorization": f"Bearer {key}",
                                              "Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=300) as r:
            body = json.loads(r.read())
        dt = time.time() - t0
        u = body.get("usage", {})
        ct = u.get("completion_tokens", 0)
        results.append({"name": name, "e2e_s": round(dt, 2), "completion_tokens": ct,
                        "tok_s": round(ct / dt, 1) if dt else None,
                        "content": body["choices"][0]["message"].get("content", ""),
                        "reasoning_chars": len(body["choices"][0]["message"].get("reasoning") or ""),
                        "finish_reason": body["choices"][0].get("finish_reason")})
        print(f"  {name}: {ct} tok, {dt:.2f}s, finish={results[-1]['finish_reason']}")
    total_tok = sum(r["completion_tokens"] for r in results)
    total_s = round(sum(r["e2e_s"] for r in results), 2)
    return {"model": model, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": results,
            "total_tok": total_tok, "total_s": total_s}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:5003")
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", "frognano-4b"))
    ap.add_argument("--tag", default="ab")
    args = ap.parse_args()
    out = run(args.url.rstrip("/"), load_key(), args.model)
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{args.tag}_{args.model.replace('/', '_')}_{time.strftime('%Y%m%d')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[ab5] {out['total_tok']} tok en {out['total_s']}s -> {path}")


if __name__ == "__main__":
    main()
