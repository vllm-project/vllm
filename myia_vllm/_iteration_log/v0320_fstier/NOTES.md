# Éval tier fs NVMe (TieringOffloadingSpec) — fenêtre 2026-10-20 (GO user « tu peux lancer l'éval fs? »)

Suite de `v0320_nvfp4_tier32/NOTES.md` §« Découverte mécanique ». Objectif :
prouver le **tier secondaire filesystem** (NVMe WSL ext4) — capacité ~centaines
de Go, **survit aux restarts** (le tmpfs CPU meurt avec le conteneur), replay
staged GPU ↔ CPU ↔ fs. C'est le levier « Redis bien calibré » (mandat user
19/10) qui reste avant le mur Minimax (~23/10).

## Configuration de la fenêtre (profil d'éval dédié, prod jamais touchée)

`medium-swift15-27b-fstier.yml` = copie du profil prod, 3 deltas :
- `--kv-offloading-size 8` GiB (CPU primaire volontairement petit → spill fs rapide)
- `--kv-transfer-config` : `spec_name=TieringOffloadingSpec`, `secondary_tiers=[{type:fs, root_dir:/kv-tier-fs}]`
- bind `${KV_FS_TIER_PATH}` (WSL ext4 `/home/user/vllm/.kv-tier-fs`) → `/kv-tier-fs`,
  long syntax `create_host_path: false` (fail-loud, anti-phantom-mount)

Harnais : `fs_tier_test.py` (seed gold 17K + 26 fillers uniques ~19K → 490K tok ;
replay gold exact ; metrics offload).

## Pré-window (2026-10-20 ~09:50-10:00Z)

- Marge commit hôte **86,2 % (52,9 GB libres)** — au-dessus du seuil 85 % mais
  marge absolue large (l'incident 24/09 était à 8,6 GB) ; arbitrage : GO avec
  **critères d'abandon commit > 92 % ou disque WSL < 40 GB**.
- `drop_caches` WSL root : buff/cache 76 → 44 GB, free interne 19 → 49 GB ;
  commit hôte quasi inchangé (86,1 %) → le commit vmmemWSL n'est PAS du page
  cache réclamable (tmpfs tier 34 GB + couches Docker + kernel).
- Pièges vécus : `wsl.exe` depuis Git Bash **mange les chemins POSIX** (mkdir a
  créé `/mnt/d/vllm/C:/Program Files/…` — nettoyé au rmdir chaîne vide) →
  TOUJOURS `MSYS_NO_PATHCONV=1` ; `/home/user/vllm` appartient à root →
  création du répertoire en `-u root` + chmod 777.
- ⚠ Tier fs **sans éviction ni plafond** (vérifié in-code) — fenêtre bornée ;
  purgeur externe = prérequis production.

## Mécanique vérifiée au source DANS l'image (avant le verdict)

- **Le lookup du tier fs est une simple existence de fichier** :
  `FsAsyncLookupManager.batch_lookup` → `os.path.exists(FileMapper.get_file_name(k))`
  (C extension `batch_lookup_C`/faccessat si dispo, GIL relâché). ⇒ **aucun index
  en mémoire requis** ⇒ la survie au restart est structurellement possible.
- **Noms de fichiers content-addressed** : `NONE_HASH` (graine du chain-hash) a une
  graine par défaut FIXE, donc « identical token content produces identical block
  filenames across instances ». ⚠ Nuance lue au source : la graine aléatoire
  s'applique aux algo de hash **non cryptographiques** — nous sommes en
  `--prefix-caching-hash-algo sha256` ⇒ graine fixe ⇒ **noms stables entre
  restarts** (et entre instances). C'est la condition qui rend le test de survie
  valide.
- Écritures idempotentes : `_store_block` **saute** si le fichier existe déjà ;
  écrit dans un temp puis `os.replace` (atomique).
- **`_load_block` supprime le fichier sur lecture courte** (confirme le danger
  documenté : pas d'éviction, mais des suppressions sur erreur) — et
  `kv_load_failure_policy='fail'` (vu dans le dump non-default args) ⇒ un load
  raté fait **échouer la requête**, il ne recompute pas.
- Pas d'`eviction_policy` posée au tier primaire dans cette fenêtre (défaut).
- **Layout disque observé** (utile pour le futur pruneur) :
  `<root>/<modèle>_<hash>_r<rang TP>/<3 hex>/<bloc>` — ex.
  `/kv-tier-fs/ukisai_Swift-1.5-Qwen3.8-27b-W4A16-AWQ_d0d49213ab3a/…_r0/943/…`,
  fichiers de **26 247 168 B** (≈ 25 MB, ~900 tokens/bloc, nvfp4 packé ≈ 22,6 KB/token).
  Un pruneur itère donc `*_r*/**/` par mtime/budget ; le répertoire racine reste.

## Déroulé + mesures

- **Boot éval** : 09:56:30→10:07:45Z (**11 min 15 s** — config d'offload changée ⇒ cache
  de compile partiellement invalidé), poids 23,9 s, **KV GPU identique à la prod :
  750 747 tokens**, `Creating offloading spec with name: TieringOffloadingSpec` sur les
  **2 workers TP**, namespace créé : `/kv-tier-fs/ukisai_Swift-1.5-Qwen3.8-27b-W4A16-AWQ_d0d49213ab3a`.
- **Seed** (gold 17,4K + 26 fillers uniques ~19K, 10:07:57→10:13:06Z) : gold correct en
  11,0 s ; 279 s de fillers.
- **★ Le spill CPU→fs est massif et rapide** : `du` fs = **14 GB / 540 fichiers** de
  26 247 168 B (blocs ~25 MB), `tiering_write_bytes{tier=1:fs}` = 26,3 GB en 192,7 s
  (≈ 136 MB/s effectif, écritures + promotions mêlées) ; store GPU→CPU = 14,13 GB en
  7,06 s (**2,0 GB/s**).
- **★ Lecture fs = 2,1 GB/s** : `tiering_read_bytes{tier=1:fs}` 210 MB en **0,100 s**
  (première lecture, 10:13:20Z, 8 chunk-hits).
- **Piège d'interprétation vécu (à retenir)** : les 3 replays « live » (1,26 / 0,82 /
  0,73 s) étaient servis par le **cache GPU**, PAS par le tier — `prompt_tokens_cached_by_source{device}`
  = +17 424 à chaque fois (cumul 69 696 = 4 replays), `host` = 0, `disk` = 0. Cause :
  les replays intercalés **rafraîchissent la position LRU du gold** ; 912K tokens de
  fillers n'évinc(ai)ent que les plus vieux fillers. **Le seul instrument qui tranche
  est `prompt_tokens_cached_by_source`** (device/host/disk) — `kv_offload_total_bytes{CPU_to_GPU}`
  reste à 0 tant que rien ne vient du tier, et un replay rapide ne prouve RIEN à lui seul.
- **★ TEST DE SURVIE (le test qui compte)** : `docker restart` du moteur à 10:19:51Z
  (GPU vidé, **tmpfs CPU vidé**, fs intact : 24 GB / 980 fichiers) → health 200 à
  10:24:25Z (boot 4,5 min) → replay du gold identique :

| Mesure (delta post-restart) | Valeur |
|---|---|
| latence replay | **1,41 s** (vs ~11 s prefill froid = **~8×**) |
| réponse | code `FS-TIER-90210` correct (HIT) |
| **`prompt_tokens_cached_by_source{disk}`** | **+17 088 / 17 424** (le reste = queue recalculée) |
| `prompt_tokens_cached_by_source{device}` / `{host}` | **0 / 0** |
| `tiering_chunk_hits{tier=1:fs}` | **15 / 15 requêtes** |
| `tiering_read_bytes{tier=1:fs}` | **+393,7 MB** (fs → primaire) |
| `kv_offload_load_bytes` / `total_bytes{CPU_to_GPU}` | **+393,4 MB** (primaire → GPU) |
| `store_bytes` | +104,7 MB (queue recomputée puis restockée) |
| `External prefix cache hit rate` (log) | 0,0 % → **97,9 %** (17 088/17 424 = 98,06 %) |
| `lookup_async_delay` fs | somme +0 s (résolutions < 5 ms) |

⇒ **Le tier fs survit au restart du moteur et sert des réponses correctes ~8× plus vite
qu'un prefill froid.** C'est sa valeur propre (le tmpfs, lui, meurt avec le conteneur).
Chemin complet exercé : lookup par existence de fichier → promotion fs→primaire (393,7 MB)
→ load primaire→GPU (393,4 MB).

## Verdict

**FENÊTRE VALIDÉE (2026-10-20 ~10:25Z)** — la mécanique multi-tier demandée par le user
est prouvée de bout en bout sur ce host :

1. **Spill CPU→fs** : oui, automatique dès que la primaire (8 GiB ici) déborde — 24 GB
   écrits, 980 blocs, sans erreur.
2. **Replay depuis fs** : oui, 2,1 GB/s en lecture, promotions tracées par tier.
3. **Survie au restart** : **oui — c'est la découverte clé** (le lookup est une simple
   existence de fichier sur des noms content-addressed à graine fixe ; aucun index
   persistant nécessaire).

**Ce qui n'est PAS encore tranché (avant toute adoption prod)** :
- **Pruneur obligatoire** : le tier fs n'a ni éviction ni plafond (vérifié in-code) —
  24 GB en une fenêtre de 30 min de trafic synthétique ; laissé libre il remplit le
  disque. Un pruneur externe (âge/budget) est un prérequis, et la sémantique d'une
  suppression sous moteur vivant (miss propre vs erreur ENOENT) reste à tester.
- **Coût en régime réel** : cette fenêtre mesure la mécanique, pas le gain — la question
  « le tier fs remplace-t-il le tier RAM 32 GiB, ou le complète-t-il ? » se tranche sur
  du trafic réel (le RAM tier coûte du commit hôte, le fs coûte du disque).
- **Interaction avec les stalls de fetch** (cascade 09:02→09:47Z, hors fenêtre) : à
  instruire séparément — hypothèse actuelle : pages du tmpfs poussées dans le swap
  WSL sous pression de commit hôte (swap mesuré à 9,5 GiB ; teardown = −44 GB de commit).
  Le tier fs, explicitement disque, est immunisé contre ce mode de défaillance — piste.

**Teardown** : profil d'éval `down` à 10:26Z, prod `up -d` (profil prod inchangé),
blocs fs nettoyés (dir conservé). Le profil d'éval reste dans le dépôt pour rejouer.

## Pruneur externe (2026-10-21, PR #137 + #138) — le prérequis d'adoption est écrit

`myia_vllm/scripts/monitoring/fstier_pruner.py` (+ wrapper `fstier_prune.ps1` qui
l'exécute dans WSL — l'énumération 9p depuis Windows est lente et le mangling de
chemin Git Bash s'est encore reproduit au premier essai, `MSYS_NO_PATHCONV=1`).

- **Dry-run par défaut**, `--apply` requis pour supprimer ; états écrits
  atomiquement dans `myia_vllm/logs/fstier_state.json` (gitignored, lisible des
  deux côtés) ; ne supprime que `<root>/*_r<rang>/<hex>/<bloc>` ; **jamais un bloc
  de moins de 30 min** (fraîchement écrit = peut être en cours de promotion,
  `kv_load_failure_policy='fail'`).
- **Politique de plafond corrigée le 21/10 après mesure de la chaîne physique**
  (clé registre `Lxss`) : le VHDX Ubuntu qui porte le tier est
  `C:\Users\MYIA\AppData\Local\wsl\{35b80e90-…}\ext4.vhdx` (850 GB) — **sur C:**,
  pas sur D: (le plancher « 12 % de D: » du 20/10 visait le mauvais volume et
  donnait un plafond de 0). Deux leviers explicites : **plancher absolu de 40 GiB
  sur le système invité** (il héberge le cache HF) + **réserve de 12 % sur les
  volumes hôtes gardés** (`/mnt/c`). Mesure du jour : invité 140,6 GiB libres,
  C: 826 GiB libres ⇒ **plafond 48 GiB**, la cible nominale.
- Reste à valider en fenêtre réelle : **la sémantique d'une suppression sous
  moteur vivant** (miss propre vs ENOENT sur le chemin lookup→load).