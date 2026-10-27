#!/usr/bin/env python3
"""prefix_probe.py - cartographie du radix tree KV par classe de requetes.

Lit /logs/error_sources.jsonl DANS le conteneur moteur (middleware error_source_capture,
body_head 1500 o de chaque requete) et classe les requetes par harnais :
claude-code (injection d'identite claudish + harnais), OpenCode, sondes routeur,
RSM condensation, raw FR, sk-agent presets, Zoo.

Usage (depuis l'hote, fenetre en heures, defaut 6) :
  MSYS_NO_PATHCONV=1 docker exec -i myia_vllm-medium-swift15-27b \
      python3 - < myia_vllm/scripts/testing/prefix_probe.py          # fenetre 6 h
  ... python3 - < ... 24                                              # fenetre 24 h

Sortie : comptes par classe, shapes distinctes Claude Code (a 1500 o), LCP intra-classe
(min/med/max), contexte du fork entre les deux shapes dominantes, representatives "other".
Carte de reference et leviers par classe : myia_vllm/docs/kvcache/prefix-map.md.
Proprietaire : ai-01:vllm (mandat user 27/10 - ne plus jamais confondre les harnais).
"""
import sys
import json
import time
import collections

WINDOW_H = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0

rows = []
now = time.time()
for line in open('/logs/error_sources.jsonl', errors='ignore'):
    try:
        r = json.loads(line)
    except Exception:
        continue
    ts = r.get('ts') or 0
    if now - ts > WINDOW_H * 3600:
        continue
    rows.append((ts, r.get('status'), r.get('model') or '', r.get('body_head') or ''))
rows.sort()
print(f"total_req_window={len(rows)} window={WINDOW_H}h")


def classify(bh):
    b = bh.lower()
    if 'you are not claude' in b[:800]:
        if 'this is claude code' in b[:1500] or 'claude code' in b[:800]:
            return 'claude-code'
        if 'opencode' in b[:800]:
            return 'opencode'
        return 'claude-code-other-shape'
    if 'claude code' in b[:800]:
        return 'claude-code'
    if 'synthèse de communications' in b[:400] or 'synthese de communications' in b[:400] \
            or 'synthèse de dashboards' in b[:400]:
        return 'rsm-condensation'
    if 'sudoku' in b[:600] or 'la grille' in b[:300]:
        return 'raw-coursia-sudoku'
    if 'swift_route_ok' in b or 'count to twenty' in b or 'list the current directory' in b:
        return 'router-probe'
    if 'you are a summarizer' in b[:200] or 'you are a classifier' in b[:200]:
        return 'sk-preset-en'
    if 'assistant de calcul' in b[:300]:
        return 'raw-calcul'
    if 'assistant de tri d' in b[:300]:
        return 'raw-test-lane'
    if 'atelier d' in b[:300] or 'médiathèque' in b[:300] or 'mediatheque' in b[:300]:
        return 'raw-creative'
    if 'zoo' in b[:200].replace('zookeeper', ''):
        return 'zoo'
    return 'other'


cls = collections.Counter()
bycls = collections.defaultdict(list)
for ts, st, model, bh in rows:
    c = classify(bh)
    cls[c] += 1
    bycls[c].append((ts, model, bh))

print("=== CLASS COUNTS ===")
for c, n in cls.most_common():
    print(f"{c:26s} {n}")


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


for cc_name in ('claude-code', 'claude-code-other-shape'):
    cc = bycls.get(cc_name, [])
    if not cc:
        continue
    print(f"\n=== {cc_name.upper()} : {len(cc)} req ===")
    shapes = collections.Counter(bh[:1500] for _, _, bh in cc)
    print(f"shapes distinctes (1500 o) : {len(shapes)}")
    for s, n in shapes.most_common(8):
        print(f"  n={n} start={s[:100]!r}")
    top2 = [s for s, _ in shapes.most_common(2)]
    if len(top2) == 2:
        i = lcp(top2[0], top2[1])
        print(f"LCP top2 shapes = {i} o ; apres fork A={top2[0][i:i+80]!r} B={top2[1][i:i+80]!r}")
    sample = [bh for _, _, bh in cc[:30]]
    if len(sample) > 1:
        vals = sorted(lcp(sample[i], sample[j])
                      for i in range(len(sample)) for j in range(i + 1, len(sample)))
        print(f"LCP paires (first 30) : min={vals[0]} med={vals[len(vals)//2]} max={vals[-1]} (cap 1500)")

print("\n=== LCP WITHIN CLASS (first 12) ===")
for c, items in sorted(bycls.items()):
    sample = [bh for _, _, bh in items[:12]]
    if len(sample) < 2:
        continue
    vals = sorted(lcp(sample[i], sample[j])
                  for i in range(len(sample)) for j in range(i + 1, len(sample)))
    print(f"{c:26s} n={len(items):4d} min={vals[0]:5d} med={vals[len(vals)//2]:5d} max={vals[-1]:5d}")
