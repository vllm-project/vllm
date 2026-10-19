# v0.31.0 → v0.32.0 image bump (2026-10-19, GO user « Go pour le bump maintenant ») — EN COURS

- previous image: `vllm/vllm-openai:v0.31.0` (promoted 05/10, PR #72)
- new image: `vllm/vllm-openai:v0.32.0` (published 2026-10-18T06:50Z)
- profile changes (exactly three): `image:`, compile-cache mount `…-v0320`, volume declaration. `…-v0310` and `…-v1` volumes kept intact for rollback.
- rollback: restore the three lines to v0.31.0 + the `…-v0310` volume, then `down && up -d --env-file`.

## Why (GO user 2026-10-19, veille PR #120)

For us: WSL2 non-blocking H2D copies (#60215), hybrid+prefix-caching fixes
(#59103, #55601, #60533, #58083 quantized Mamba-2 with TP), faster startup
(shm tensor arena default #51207, daemon-held weights counted in gpu-util
#57298), security (chat-template DoS GHSA #50300, tokenizer #59491, P2P ZMQ
#51504, CORS restricted #61009), and **`max_tokens` no longer reserved from
the context window (#57035)** — the exact class of our 400 on full-window
requests (#431). Breaking changes checked: `mamba_cache_mode="all"` removed
(we run `align`), cascade attention removed (unused), transformers pin is
in-image. Full triage: `v0320_triage/NOTES.md`.

## Window

Annonce dashboards ~20:35 locale (fenêtre annoncée AVANT) · baseline bench
`bench_concurrent_scaling.py` sur v0.31 (moteur en service ~5 h, « âgé ») ·
pull image (33,1 GB, `befccf0e7e40`) · image v0.32.0 confirmée dans le
conteneur · `down` ~20:13Z · **`StartedAt 20:18:09Z`** · health 200 après
`init engine 215,06 s` (fini **20:23:36Z**) · sidecars up ~20:24Z.
**Downtime total ≈ 11 min** (boot froid sur volume `…-v0320` neuf — AUCUN stall
cette fois : warmup 71,4 s).

## Post-boot state

- **`vLLM 0.32.0`** (vérifié dans le conteneur) · **`Using V2 Model Runner`**
- args **byte-identiques** à v0.31 (fp8, offload 24 GiB, batch 8192/seqs 48,
  prefix_match_unit 16, mamba fp16, gpu-util 0.70) — diff = 3 lignes de profil only
- **KV cache : 429 319 tokens** (v0.31 : 424 610) → **+4 709 = +1,1 % à config
  identique** (premier gain mesuré de la v0.32)
- VRAM **GPU 0 19 359 / GPU 1 18 930 MiB** (bande froide référence 19 937/18 930
  — GPU 0 légèrement SOUS la référence)
- FlashInfer : `kv_cache_dtype=torch.float8_e4m3fn, arch=sm89` résolu au boot
- `shm_broadcast.py` renuméroté (812 → 871) — code changé, comportement identique

## Battery — 11/11 effectifs ([gates_v0320.py](gates_v0320.py), JSON `gates_v0320_1792441610.json`)

| Gate | Résultat |
|---|---|
| smoke | PASS after fix (\`thinking off\` — le FAIL initial était un bug de MON gate : `max_tokens=8` + thinking par défaut = budget mangé par le reasoning, piège documenté ; avec thinking off → `PONG`) |
| vision | 4/4 couleurs (`Red Green Blue Yellow`) 0,6 s |
| tool-calling | `get_weather({"city":"Lyon"})` 0,9 s |
| thinking | 154 chars reasoning, finish=stop |
| preserve_thinking | OK (`17`) 0,3 s |
| prefill-30k | 27 580 tok en 14,3 s → **1 931 tok/s** + survie (inclut le JIT du 1ᵉʳ forward) |
| prefill-95k | 89 268 tok en 37,0 s → **2 414 tok/s** + survie |
| prefill-235k | 221 830 tok en 102,0 s → **2 176 tok/s** + survie |
| concurrent N=16 ×2 | **355 → 438 tok/s** au repeat (prefix cache actif) |
| FORCING (batch (4096, 8192]) | 6× ≈4 618 tok = 24 348 tok en 2,9 s, **all-200** |

Prefill **au-dessus** de la référence v0.31 (1 595-1 608 tok/s @157K, A/B du
08/10) — à revérifier au bench (longueurs de prompt différentes ici).

## Error scan

**0 signature** (`Workspace is locked|turboquant|EngineDead|out of memory|Traceback|CUDA error`)
sur tout le segment post-boot. Watchdog v5 : sondes OK dès les premières post-boot.

## Bench A/B same-window ([bench_concurrent_scaling.py](../bench_concurrent_scaling.py), même script, mêmes prompts, temp 0)

| N | v0.31 baseline (moteur ~5 h) | v0.32 passe 1 | v0.32 passe 2 |
|---:|---:|---:|---:|
| 1 | 25,9 | 42,1 | 36,6 |
| 2 | 28,6 | 49,6 | 50,8 |
| 4 | 59,3 | 103,6 | 104,3 |
| 8 | 113,8 | 200,7 | 213,5 |
| 16 | 243,0 | 409,8 | **463,2** |
| 24 | 293,7 | 601,9 | 618,9 |
| 32 | 398,5 | 718,1 | 786,4 |
| 48 | 461,8 | 861,0 | **928,4** |

**Lecture honnête — le « +86 % » apparent n'est PAS attribuable à la v0.32.**
Deux confondants identifiés sur le baseline du jour :
1. **Le téléchargement de l'image v0.32 (33 GB) tournait pendant le bench
   baseline** (I/O disque/hôte concurrent) ;
2. moteur âgé de ~5 h (l'effet documenté joue ~±8 %, pas +80 %).

La référence fraîche historique sur ce type de harnais — **907 t/s à N=48
(08/10, v0.30, batch 8192 frais)** — suggère que le baseline du jour (461,8)
est anormalement bas. Ce qui est **solidement établi** : la v0.32 à chaud
atteint **928,4 t/s à N=48 et 463,2 à N=16, au-dessus de la meilleure référence
historique** ; le plateau passe 1 → passe 2 confirme le pattern de warm-up
documenté (la passe 1 paie la taxe JIT post-boot).

**Conséquence** : la promotion v0.32 ne s'appuie PAS sur le delta apparent
contre le baseline du jour (non concluant) mais sur : gates 11/11, config
byte-identique, KV +1,1 %, zéro erreur, et le **niveau de performance post-boot
≥ meilleure référence historique**.

## #57035 — le 400 de fenêtre-pleine est ÉLIMINÉ (test dédié, [window_full_test.py](window_full_test.py))

Cas mesuré, calibré par `/tokenize` : **prompt 260 012 tok + `max_tokens=4096`
= 264 108 > 262 144** (la classe exacte du 400 mesuré 3× le 09/10 sur v0.31,
PR #110 — « maximum context length » sans rognage).

- **v0.31 : HTTP 400.** **v0.32 : HTTP 200 en 17,6 s**, `finish=stop`.
- `max_tokens` n'est plus réservé de la fenêtre (#57035) — confirmé en direct
  sur notre serveur, pas seulement dans le changelog.
- Reste licite : un prompt **seul** > 262 144 continuera de donner 400 (vrai
  dépassement). La manœuvre de contournement de #431 (route cloud sur 400)
  devient donc moins nécessaire — à répercuter sur claudish #431.

## Verdict: **PROMOTED** (2026-10-19, GO user explicite « Go pour le bump maintenant »)

Gates 11/11 effectifs, FORCING all-200, zéro signature d'erreur, config
byte-identique, **KV 429 319 (+1,1 %)**, VRAM sous la référence froide, boot
froid 11 min sans stall, bench post-boot **≥ meilleure référence historique
(928,4 @ N=48)**, et #57035 vérifié en direct. Le delta apparent vs le baseline
du jour n'est pas retenu comme preuve (confondants documentés ci-dessus).

Watch items : cycle de surveillance suivant = premier cycle v0.32 (vérifier
stabilité, sondes, conso VRAM sous trafic) — la promotion est une clairance de
correction + performance, pas encore un soak. **Phase 2 (GO user) : éval
NVFP4-KV sur SM89 — à suivre dans `v0320_nvfp4_kv/`.**