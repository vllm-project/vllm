# NVFP4-KV sur SM89 — éval (2026-10-19, GO user « oui aux deux », item Q21b) — REJETÉ, prod revenue en fp8

- prod au moment de la fenêtre : **v0.32.0** (bump du jour, PR #122), fp8 KV 429 319 tok
- override : `KV_DTYPE_SWIFT15=nvfp4` (variable déjà câblée dans le profil, ligne 97)
- retour fp8 : `KV_DTYPE_SWIFT15=fp8` (ou absence de la variable — défaut du profil)

## Pourquoi / ce qu'on mesure

Changelog v0.32.0 #46963 : « NVFP4 KV cache now runs on SM8x and SM12x ». Vérifié
statiquement dans l'image avant la fenêtre :
- `FlashInferBackend.supports_kv_cache_dtype("nvfp4")` → `_nvfp4_kv_on_fa2()` =
  familles (80, 90, 120) → **SM89 couvert** ; sur FA2, seul `nvfp4` est accepté
  (pas `nvfp4_4over6`).
- Gates : head_size ≠ 64 (le nôtre = **256**), pas de sink, K/V symétriques — OK.
- Format page : `[K data | K scale | V data | V scale]`, uint8 ; `nvfp4_kv_cache_full_dim`
  ajoute les scales → densité attendue ≈ 1,78× fp8 (429 319 → ~760K tok potentiels).
- **Vitesse inconnue** : le writer slot-mapping FA2 « has no scale search » — c'est
  le point de risque (précédent : TQ k8v4 rejeté à −33 % N=1, PR #110).

Protocole (même que TQ) : KV capacity · bench N=1/2/4/8/16/24/32/48 · prefill ·
quality canary · frais-vs-frais (re-bench fp8 après rollback, même heure).

## Fenêtre

- bascule `down` + `up -d` composés : moteur stoppé ~20:35Z, nouveau moteur
  initialisé **20:36:20Z**, `up -d` rend la main **20:42:01Z** (attend les
  sidecars `service_healthy`) — **indispo ≈ 6-7 min**, AUCUN stall, aucune erreur.
- `kv_cache_dtype=nvfp4` confirmé dans la config engine ; FlashInfer :
  `kv_cache_dtype=torch.uint8, arch=sm89` (chemin FA2 — le writer slot-mapping).
- **KV : 750 747 tokens** (fp8 : 429 319) = **×1,749** — la densité théorique
  (1,78×) se réalise ; concurrency 262K : 2,86× (fp8 : 1,64×).
- VRAM **GPU 0 19 359 / GPU 1 18 940 MiB** — identique à la bande fp8 post-bump
  (le cache 4-bit plus grand n'a pas coûté de VRAM : mêmes pools).
- GPU 2 libre (82 MiB).

## Résultats

### Gates — 11/11 PASS (JSON : `../v0320_bump/gates_v0320_1792442729.json`)

| Gate | nvfp4 | mêmes gates en fp8 (post-bump, ~20:28Z) |
|---|---|---|
| smoke | PONG 0,44 s | PONG |
| vision | 4/4 couleurs 0,6 s | 4/4 |
| tool-calling | `{"city":"Lyon"}` 1,0 s | idem 0,9 s |
| thinking | 154 chars, stop | idem |
| preserve_thinking | `17` 0,4 s | idem |
| prefill-30k | 27 580 tok → **1 812 tok/s** | 1 931 (−6 %) |
| prefill-95k | 89 268 tok → **2 288 tok/s** | 2 414 (−5 %) |
| prefill-235k | 221 830 tok → **2 103 tok/s** + survie | 2 176 (−3 %) |
| concurrent N16 ×2 | 357 → 462 tok/s | 355 → 438 |
| FORCING (4096,8192] | all-200, 2,4 s | all-200, 2,9 s |

### Bench N=1..48 ([bench_concurrent_scaling.py](../bench_concurrent_scaling.py), 1 passe, moteur chaud post-gates)

| N | nvfp4 t/s | fp8 passe 2 (20:30Z, même soir) | Δ |
|---:|---:|---:|---:|
| 1 | 33,9 | 36,6 | −7,4 % |
| 2 | 46,5 | 50,8 | −8,5 % |
| 4 | 99,7 | 104,3 | −4,4 % |
| 8 | 202,8 | 213,5 | −5,0 % |
| 16 | 417,9 | 463,2 | −9,8 % |
| 24 | 570,3 | 618,9 | −7,9 % |
| 32 | 678,6 | 786,4 | −13,7 % |
| 48 | 767,5 | 928,4 | −17,3 % |

Confoundant noté : les deux mesures ne sont pas même-minute (20:30 vs 20:46Z) et la
référence fp8 928,4 = meilleur résultat historique de ce harnais. **Contre-bench
fp8 frais après rollback** (ci-dessous) pour trancher frais-vs-frais.

### Needle-in-haystack (recall 50 036 tok, code unique à 25 % / 75 %) — [needle_test.py](needle_test.py)

- 25 % : **HIT** (`ZX-4711-QQ`, 27,6 s)
- 75 % : **HIT** (14,8 s)
→ **la correction 4-bit tient sur ce canary** : pas de perte de recall détectable
à 50K. Le discriminant de cette éval n'est PAS la qualité, c'est la vitesse.

### Contre-bench fp8 frais (après rollback, 20:50-20:52Z, même soirée)

Rollback : down 20:47Z → moteur init 20:48:30Z → **up-issued 20:49:57Z** ;
`kv_cache_dtype=fp8` confirmé, FlashInfer `torch.float8_e4m3fn arch=sm89`,
**KV 429 319 tokens** (retour exact au chiffre du bump), health 200 en 5,5 ms.
Passe 1 post-boot **sans warm-up gates** (contrairement au nvfp4, qui avait les
gates avant lui — le fp8 est donc MESURÉ PLUS FROID que le nvfp4, ce qui rend la
comparaison conservatrice POUR nvfp4, et il gagne quand même) :

| N | nvfp4 (chaud post-gates) | **fp8 frais** | Δ nvfp4 |
|---:|---:|---:|---:|
| 1 | 33,9 | **41,1** | −17,5 % |
| 2 | 46,5 | **54,2** | −14,2 % |
| 4 | 99,7 | **104,3** | −4,4 % |
| 8 | 202,8 | **191,8** | +5,7 % |
| 16 | 417,9 | **430,0** | −2,8 % |
| 24 | 570,3 | **602,1** | −5,3 % |
| 32 | 678,6 | **735,0** | −7,7 % |
| 48 | 767,5 | **886,4** | −13,4 % |

Cohérence : fp8 frais 886,4 vs passe chaude du bump 928,4 (±5 %) — les deux
mesures fp8 du soir se valident mutuellement. Needle fp8 (contrôle) :
**HIT/HIT** (28,2 / 27,9 s) — le canary est valide, nvfp4 ne perdait pas le
recall.

## Verdict : **REJETÉ** (2026-10-19 20:52Z) — retour fp8 effectif

Même signature que TurboQuant (rejeté 09/10, PR #110) : **la capacité n'est pas
la ressource rare, la vitesse decode l'est**.

- **Ce que nvfp4 achète** : KV 750 747 tok (×1,75), concurrency 2,86× à 262K —
  mais l'occupation observée est 2-7 % et le KV fp8 (429 319 = 1,64× la fenêtre
  + tier RAM ~685K) n'a jamais saturé.
- **Ce que nvfp4 coûte** : −13 à −17 % de débit aux extrémités (N=1 : −17,5 %,
  N=48 : −13,4 % frais-vs-frais même soirée), −3 à −8 % au milieu, prefill
  −3 à −6 %. Et ce malgré une mesure nvfp4 AVANTAGÉE (chaud post-gates vs fp8
  frais sans warm-up).
- Qualité : **aucune perte détectée** (gates 11/11, needle HIT 25 %/75 %) — le
  rejet est purement un trade vitesse/capacité, pas une dégradation de sortie.
- **Re-commande** : `KV_DTYPE_SWIFT15=nvfp4 docker compose … up -d` (variable
  câblée au profil). Ré-ouvrir l'éval si (a) le workload monte l'occupation KV
  au-dessus de ~50 %, ou (b) les kernels nvfp4 mûrissent sur SM89 (watch : le
  chemin `nvfp4_trtllm` du backend FlashInfer est aujourd'hui gated Blackwell
  SM100 ; XQA decode = SM12x).

Indisponibilité totale de la fenêtre : ~20:35→20:42Z (bascule nvfp4) +
20:47→20:50Z (retour fp8) ≈ **10 min**. Aucune erreur, aucun stall, aucun
événement watchdog.

## Verdict

(à remplir)