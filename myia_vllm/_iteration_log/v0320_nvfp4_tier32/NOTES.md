# nvfp4 KV par défaut + tier offload 32 GiB — production (2026-10-20, Q22 a+b, GO user)

Fenêtre déclenchée après **3 wedges par saturation KV en une nuit** (22:22, 22:36,
00:17Z — sessions Bun ~260K tokens, KV GPU 100 %, fetch tier bloqué). Décision
user : « on commence par a) et b), puis c) si ça ne suffit pas ».

- PR config : [#128](https://github.com/jsboige/vllm/pull/128)
- Base : v0.32.0 (bump PR #122), éval nvfp4 PR #123 (rejet « capacité pas rare » —
  ré-ouvert car l'occupation a atteint 100 % la nuit même)
- Rollback 1 ligne : `KV_DTYPE_SWIFT15=fp8 KV_OFFLOAD_GB_SWIFT15=24 docker compose … up -d`

## Fenêtre

- down-issued **00:41:35Z**, down-done 00:41:47Z (12 s — tmpfs peu touché)
- up -d émis **00:42Z** (arrière-plan)
- Découvertes pré-fenêtre :
  - `docker exec` cassé sur le conteneur moteur uniquement (`setns … /proc/self/fd/6`) —
    séquelle du `docker restart` watchdog de 00:17Z ; les sidecars répondent.
    Attendu : réparé par la recréation.
  - **Piège de précédence** : le `.env` local portait `KV_OFFLOAD_GB_SWIFT15=24` qui
    écrasait le défaut du profil (rendu compose : 24 alors que le défaut disait 32).
    Ligne retirée — le profil est redevenu source de vérité unique. (env > .env > défaut)
  - Marge commit hôte pré-fenêtre : **83,7 % (62,7 GB libres)** — sous le seuil 85 %,
    serré ; le down libère le tmpfs touché (~6-10 GB).

## Vérifications post-boot (00:42→00:50Z)

| Vérif | Résultat |
|---|---|
| health | **200 en 11 ms**, container `healthy`, RC=0 |
| `kv_cache_dtype` | **nvfp4** (config engine + non-default args) |
| `kv_offloading_size` | **32.0** GiB |
| **KV GPU** | **750 747 tokens** (idem éval 20:36Z), concurrency 2,86× à 262K |
| VRAM | GPU 0 **19 859** / GPU 1 **18 940** MiB — bande identique à l'éval |
| GPU 2 | 15 208 MiB — **CoursIA travaille** (nominal, hors périmètre) |
| `docker exec` moteur | **RÉPARÉ** — la recréation a résolu le `setns /proc/self/fd/6` |
| FlashInfer | `kv_cache_dtype=torch.uint8, arch=sm89` (chemin FA2) |
| sidecars | watchdog + télémétrie démarrés après `service_healthy` (up-done 00:49:57Z) |

## ★ Découverte mécanique : le tier fs est atteignable depuis nos flags (préparé pour l'éval suivante)

Vérifié **dans le code de l'image** (pas seulement la doc « latest ») :

- `config/vllm.py:1097` `_post_init_kv_transfer_config` : si `kv_offloading_size` est
  posé, vLLM crée un `KVTransferConfig` **seulement s'il est absent**, force
  `kv_connector="OffloadingConnector"` et fait un **`.update()`** (pas un remplacement)
  de `kv_connector_extra_config` avec `cpu_bytes_to_use`.
  ⇒ **un `--kv-transfer-config` fourni par l'opérateur SURVIT à la synthèse** : on peut y
  injecter `spec_name` et `secondary_tiers` sans quitter le chemin `--kv-offloading-*`.
- `v1/kv_offload/factory.py` : `spec_name` (défaut `CPUOffloadingSpec`) ;
  `TieringOffloadingSpec` enregistré en dur dans l'image.
- `v1/kv_offload/tiering/spec.py` (docstring) : `eviction_policy` = `lru` | **`arc`**
  (+ politiques custom), `secondary_tiers` = liste de dicts `{type, …}` ;
  types dans l'image : `fs`, `obj`, `p2p`, `example`.
- `tiering/fs/manager.py` : `FileSystemTierManager(root_dir, n_read_threads=16,
  n_write_threads=16, enable_kv_events=False, locality=None)` — params passés
  directement depuis le dict du tier.
- Rappel doc : le CPU primaire est **toujours** traversé (GPU ↔ CPU ↔ fs) ;
  `locality` LOCAL/REMOTE ; gros gain attendu = **survivre aux restarts** (le tmpfs
  meurt avec le conteneur, le fs non) + capacité ~centaines de Go.

Recette candidate (fenêtre suivante, PAS appliquée) :
```yaml
--kv-offloading-size 32
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","eviction_policy":"arc","secondary_tiers":[{"type":"fs","root_dir":"<WSL-ext4>/vllm/.kv-tier-fs"}]}}'
```

Points à trancher avant la fenêtre : support **WSL ext4 natif** vs bind drvfs
(`/mnt/d` = 9p lent — le HF cache vit déjà en ext4 WSL, suivre ce modèle) ; espace
libre du volume WSL — **mesuré : `/dev/sdd` 1007 G, 161 G libres (84 % utilisés,
dont 326 G de HF cache)** ; vérifier que `arc` est bien accepté par la version de
`TieringOffloadingSpec` de l'image (**vérifié : `arc` built-in dans la docstring du
spec**) ; test replay gold/filler (pattern bench v3) avec redémarrage entre store et
replay pour prouver la survie au restart.
Option notée : `kv_offloading_backend=lmcache` (LMCacheMPConnector, serveur dédié,
tiers Redis inclus) — plus lourd, à évaluer seulement si le fs natif déçoit.

### ⚠️ Exigence n°1 de l'éval fs : PAS d'éviction, PAS de plafond

Vérifié dans `tiering/fs/manager.py` + `fs/io.py` (image v0.32.0) : le tier fs
**écrit les blocs et ne les supprime QUE sur erreur de lecture courte** — aucune
politique LRU/ARC disque, aucun `max_bytes`, aucun prune. Laissé sans gestion il
**remplit le disque jusqu'à la panne Docker Desktop** (même famille que l'incident
hôte du 24/09). Toute adoption production exige donc **un pruneur externe** :
- script de rétention (âge/mtime ou budget octets) sur `root_dir`, exécuté quand le
  moteur est arrêté (une suppression sous moteur vivant = lookup tier « hit » puis
  load ENOENT — à tester pour connaître la dégradation : miss propre ou erreur ?) ;
- ou budget disque dédié (volume dimensionné) avec surveillance §3.
Pour la fenêtre d'éval (courte) : pas de pruneur nécessaire, mais le script de
surveillance d'espace doit être posé AVANT le boot de test.

### Prochaines étapes (ordre)

1. Soak (a)+(b) cette nuit — cycle surveillance 04:47Z.
2. Éval tier fs en fenêtre dédiée (recette ci-dessus), avec pruneur + surveillance
   disque, test replay + survie au restart.
3. Disagg conditionnel (GPU 2 CoursIA + NIXL WSL2 à prouver — cf. assessment 20/10).

## Mesures

### Gates — **11/11 PASS** (JSON : `../v0320_bump/gates_v0320_1792457715.json`)

| Gate | nvfp4+32 (01:3xZ, sous trafic réel) | nvfp4 24 GiB (20:36Z, machine calme) |
|---|---|---|
| smoke | PONG 0,66 s | PONG 0,44 s |
| vision | 4/4 0,9 s | 4/4 |
| tool-calling | `{"city":"Lyon"}` 1,2 s | idem |
| thinking | 154 chars, stop | idem |
| preserve_thinking | `17` 0,4 s | idem |
| prefill-30k | 27 580 tok → **1 790 tok/s**, survie | 1 812 |
| prefill-95k | 89 268 tok → **2 151 tok/s**, survie | 2 288 |
| prefill-235k | 221 830 tok → **1 787 tok/s**, survie | 2 103 |
| concurrent N16 ×2 | **12 → 390 tok/s** | 357 → 462 |
| FORCING (4096,8192] | all-200, 2,9 s | all-200, 2,4 s |

Lecture : le premier passage N16 à 12 t/s n'est PAS une régression du config — il
chevauche le drain du prefill 235K du même batterie (~100 s de famine, exactement la
durée mesurée) **avec** le trafic Bun réel qui tournait en parallèle (les gates de
20:36Z tournaient machine calme). Le repeat à 390 t/s (cache chaud) est dans la bande
de référence. `num_preemptions_total = 0` sur tout le boot — pas de préemption malgré
le trafic. Un seul `WEDGE fail 1/2` watchdog à 00:51:56Z (post-grâce, jamais 2/2),
récupéré seul — comportement documenté.

**Smoke bench volontairement sauté** : le moteur servait du trafic réel (contrairement
à l'éval de 20:36Z) — un bench serait confondu et sans valeur. Les chiffres de débit
nvfp4 de l'éval (N=1 33,9 → N=48 767,5 ; −13/−17 % vs fp8 frais) restent la référence
du config : le tier (24→32 GiB) ne touche pas le chemin decode.

## Verdict : **FENÊTRE VALIDÉE** (20/10 ~01:40Z) — prod sur nvfp4 + 32 GiB

- Toutes les cibles atteintes : KV **750 747** (×1,75), dtype nvfp4, tier 32, exec
  réparé, VRAM inchangée, 11/11 gates sous trafic réel.
- Première fenêtre de la nuit (22:22 → 00:17Z : 3 wedges à ~90 min d'intervalle) :
  **aucun wedge depuis le nouveau boot** (soak en cours, cycle surveillance 04:47Z).
- La vraie mesure qui tranchera = le soak sous charge Minimax réelle (hit rate,
  `num_preemptions_total`, occupancy KV) — §5quater.
- PR #128 mergée avec ce record.