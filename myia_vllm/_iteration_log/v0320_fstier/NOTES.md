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
## Fenêtre prod 2026-10-21 19:15Z — ADOPTÉE en prod de fait + pruner ARMÉ (PR #142)

Suite de `../fstier_window_20261021/RUNBOOK.md`. Geste : down prod (RAM 32) → up
`medium-swift15-27b-fstier.yml` (RAM 8 + fs 48). Boot 4,5 min (compile-cache
valide). Indispo réelle ~5 min, puis **2ᵉ indispo 19:26→~19:39Z** : watchdog
WEDGE 2/2 pendant la ruée de re-prefill post-fenêtre (3 req, prefill 20 048 tok/s,
KV 33 % — **famine documentée, pas une panne** ; le moteur travaillait). Amélioration
candidate (non faite) : porte WEDGE sur « gros prefill en vol » (prompt throughput
élevé) avant de redémarrer.

**Chiffres du boot 19:29Z (75 min de trafic réel)** : 39,8 GB écrits fs,
6,9 GB relus, **263 chunk-hits fs / 229 K tokens servis depuis le disque**
(les blocs écrits par la génération 19:15-19:28 ont survécu au restart et
servent les replays — la feature en prod réelle), 0 préemption, commit
92,0→87,5 % à J+0 (90,8 % à J+1h30, tier RAM 8 GiB se remplissant).

**Le pruneur a failli être armé aveugle** : premier dry-run = `files_total: 0`
sur un tier de 39 GB / 1 582 blocs. Trois causes (PR #142) : regex `^_r\d+$`
(anchored, ne matche jamais `<model>_<hash>_r0`), niveau de répertoire
`<2hex>_g<gen>` manquant au layout v0.32.0 (`<ns>/<3hex>/<2hex>_g<gen>/<bloc>.bin`
— le NOTES ci-dessus décrivait `<3hex>/<bloc>`), blocs root (l'utilisateur WSL
par défaut ne peut pas supprimer — wrapper `-u root`).

**Sonde suppression sous moteur vivant : PASSÉE** (l'item ouvert ci-dessus) :
bloc >30 min supprimé à 20:4xZ → 0 erreur, gate 200, +8 chunk-hits fs servis
après (miss propre `os.path.exists`). **Tâche `fstier-prune` 30 min armée**
(`fstier_register_task.ps1`, pattern nbaudit, user-level) : apply = supprime
uniquement >cap (>30 min d'âge) ; premier passage réel 1 649 blocs / 43,3 GB →
0 suppression (sous cap), 0 erreur. Le cap auto-serre quand le tier mange le
libre invité (formule min(48, invité−40, C:−12 %)).

Reste ouvert : merge du profil fstier dans `medium-swift15-27b.yml` (propre,
au prochain cycle calme — la prod tourne sur le profil fstier tel quel),
vigie du rythme de remplissage vs cadence 30 min sous charge Minimax.

## 23/10 — le pruneur et le moteur se courent après (course PROUVÉE), et la cadence des wedges n'est pas le tier

**1. Course pruneur / lecture de bloc — corrélation 5/5, plus une hypothèse.**
Le moteur loggue `Job N block I/O failed: [Errno 2] No such file or directory`
depuis le tier fs. Relevé sur 20 000 lignes : **5 occurrences** — 00:20:05,
03:20:02, 06:51:10, 06:51:38, 11:51:12Z. La tâche `fstier-prune` tourne aux
bornes **:20 et :50** (`StartBoundary 00:20`, toutes les 30 min, `StartWhenAvailable`).
Chaque échec tombe **2 à 72 s après un début de cycle** : le tier teste
l'existence du fichier (c'est SON index), le pruneur le supprime entre le test
et la lecture, le read échoue.

Conséquence : `kv_load_failure_policy='fail'` (`config/kv_transfer.py:88`) fait
**échouer la requête** au lieu de la recalculer. ~5 requêtes cassées en 12 h —
peu, mais c'est un défaut de conception de notre dispositif, pas un aléa.

**Correctif disponible sans changement de version** : la même ligne accepte
`recompute` (`scheduler.py:183-184` → `self.recompute_kv_load_failures`), qui
re-préfille au lieu d'abandonner. Coût d'un recompute sur un préfixe de 30-60 K :
quelques secondes ; coût d'un `fail` : une erreur client + un retry.
→ **Q27 (b)** : à appliquer dans la même fenêtre que la porte watchdog.

**2. La cadence des wedges n'est PAS imputable au passage multi-tier.**
17 h calmes après la fenêtre du 21/10 19:15Z (un seul wedge, 21/10 19:28Z,
puis rien jusqu'au 22/10 12:41Z). L'explosion démarre au **mur Minimax du
22/10 22:08Z** : 14 redémarrages en 23 h (22:08, 22:47, 00:20, 00:43, 00:57,
01:11, 03:22, 03:36, 04:23, 09:00, 10:25, 10:39, 10:56, 11:29Z). Le facteur
dominant est la **charge** (469 req/2 h dont 296 >100 K), pas le tier — le tier
était déjà en place pendant les 17 h calmes. Corrélation, pas causalité ; la
contre-épreuve serait une fenêtre à charge égale avec l'ancien tier.

**3. Signal neuf : la préemption.** `num_preemptions_total` = **0** pendant
des jours, **12 en 70 min** depuis le boot de 11:30:42Z, avec
`kv_cache_usage_perc` monté à **92,5 %** (puis 86 %) et 3-5 requêtes en vol.
Mécanisme probable du wedge : KV plein → le scheduler préempte → la séquence
préemptée doit être re-préfillée → la sonde décodage de 24 tokens du watchdog
attend derrière un prefill de plusieurs dizaines de milliers de tokens →
`GEN_TIMEOUT` 90 s dépassé → 2 échecs → restart → **le KV est vidé** et la
charge reconstruit le même état. La boucle est auto-entretenue.
C'est exactement ce que la porte « prefill en vol » (Q27 (a)) casse : les
compteurs `prompt_tokens_total` avancent pendant que la sonde expire, donc le
moteur travaille et le restart est un dégât net.

**Coût réel d'un restart, mesuré** : `shutdown` 10:57:53Z → `Application
startup complete` ~10:59:5xZ = **~2 min**, pas les 15 min du boot à froid du
22/10. Le coût n'est donc pas le boot mais (a) le KV vidé, (b) les 600 s de
sommeil que le watchdog s'impose ensuite.

**Hygiène /dev/shm** : 25 G utilisés au cycle pour un plancher légitime de
8 GiB (tier RAM du profil fstier) → **14 fichiers / 17,28 GiB d'orphelins**
purgeés (bornés au `StartedAt` 11:30:42Z), 8,0 G / 13 % après, health 200 en
6,4 ms.

## 23/10 14:41Z — course 8/8 ; les 500 du jour sont des fenêtres de boot ; KV 95,4 %

- **Course pruneur/lecture : 8/8.** Trois nouveaux ENOENT (12:54:03, 13:20:37,
  14:20:12Z) tombent à +63 s, +37 s et **+11 s** d'un début de cycle pruner.
  Le cycle de 14:20:01 a supprimé **907 blocs / 22,2 GiB** (tier monté à
  70,2 GiB) : plus le cycle est gros, plus la fenêtre de course est grande.
- **Le ENOENT de 14:20:12 précède de 1 s le `fail 1/2` de 14:20:13** qui a
  mené au wedge 14:22:13 — coïncidence suggestive (1 cas sur 3 ; les deux
  autres ENOENT n'ont pas déclenché de wedge), pas une causalité établie.
- **Les 4×500 du census 2 h sont des erreurs de fenêtre de boot**, pas la
  course : 14:03:29 (+59 s après le restart 14:02:30) et 14:23:44 ×3 (+31 s
  après le restart 14:22:13, IP 90.65.170.144 = front hub). Coût
  client-visible direct des wedges.
- **KV 95,4 %** avec 5 running et **4 préemptions 17 min après le boot** :
  le moteur re-sature immédiatement ; 16 restarts en 25 h. Sans la porte
  watchdog (Q27a) ni `recompute` (Q27b), la boucle continue.
- Moteur `StartedAt 14:23:45Z`, health 200 en 3,8 ms au cycle. Commit 73,3 %,
  invité 91 % (91 GB libres), pruner 0 err, nbaudit ok (vague `Image/` en
  cours de livraison).

## 23/10 16:41Z — 17ᵉ wedge ; disque invité 93 % : l'équilibre tier/disque devient structurel

- **17ᵉ wedge 15:12:39** (restart 15:14:12), puis **87 min tenues** (1 fail 1/2
  non escaladé) avec **hit préfixe 73,8 % sur 60,9 M tokens** — l'amortissement
  sain revient dès que le moteur vit assez longtemps. Charge record : **824
  req/2 h dont 348 >100 K, pic 12 longues/min**.
- **Disque invité 93 % (72 GB libres).** Le tier a atteint **84,7 GiB** entre
  deux purges ; cap auto-serré à **27,7 GiB**, purge 16:20 = **1 574 blocs /
  38,5 GiB** (0 err). Écritures ~18 GiB/h : le pruner tient, mais l'équilibre
  est fragile — le cap 48 nominal est inatteignable tant que l'invité est à
  93 %. **Q26 (suppression des 228 Go de modèles rejetés) = le levier
  structurel.**
- **0 nouvel ENOENT depuis 14:20:12** malgré la grosse purge de 16:20 — la
  course reste prouvée 8/8 mais n'a pas rejoui cette fenêtre (le working set
  post-restart est jeune). La veille continue.
- Purge `/dev/shm` : **27 fichiers / 34,33 GiB** d'orphelins (bornés au
  StartedAt 15:14:12) → **8,1 G / 13 %**, health 200 en 5,8 ms. Commit 74,3 %.

## 23/10 18:41Z — 18ᵉ wedge ; mais la course s'est tue et le hit préfixe monte à 86,7 %

- **18ᵉ wedge 17:53:27** (restart 17:55:02), puis 45 min tenues (2 fail 1/2
  non escaladés). KV 91,8 %, **0 préemption**, **hit préfixe 86,65 %** sur
  19,3 M — meilleur ratio mesuré depuis le début du pilote : l'amortissement
  fonctionne quand le moteur vit.
- **0 ENOENT depuis 14:20:12** (4 h 20, **8 cycles de purge** consécutifs sans
  incident, dont 38,5 GiB puis 29,0 GiB supprimés) et **0×500 sur 613
  requêtes** en 2 h. La course est donc **intermittente**, pas systématique :
  hypothèse (non testée) — les entrées périmées vivent dans l'index en mémoire
  de la génération courante, qu'un restart reconstruit depuis le disque ; les
  générations courtes (comme aujourd'hui) ne laissent pas le temps de former
  ces pointeurs morts. À confronter à un moteur qui vivra longtemps.
- Disque invité **92 %** (82 GB, mieux que 93 %), tier 63,8 GiB avant purge,
  cap auto-serré 34,8. Q26 reste le levier structurel.
- Charge 558 req/2 h dont 348 >100 K, pic 10/min. Commit 74,2 %.
  Pruner 0 err, nbaudit ok, condensation repli 3 étages key=OK.

## 23/10 20:41Z — 19ᵉ wedge ; hit 91,74 % (record) ; la course rejoue : 9/9

- **19ᵉ wedge 18:59:55** (restart 19:01:29), puis **100 min tenues**.
  **Hit préfixe 91,74 %** (42,7 M/46,5 M) — nouveau record sur échantillon
  réel ; KV 90,1 %, 0 préemption. Le cap pruner est **remonté à 45,9 GiB**
  (disque invité revenu à 92 % / 84 GB) — l'auto-serrement desserre des deux
  sens.
- **NOUVEAU ENOENT à 18:50:24 = +24 s après le cycle pruner de 18:50 →
  course à 9/9.** Détail qui alimente l'hypothèse de l'index mémoire : la
  génération en cours avait démarré à 17:55 (55 min de vie au moment du
  tir), le bloc visé datait d'avant 18:20 — soit l'âge de bloc ≥ 30 min
  requis par le min-age du pruner, connu de l'index de la génération.
- Purge `/dev/shm` : 14 fichiers / **17,28 GiB** (orphelins du restart
  19:01) → 8,1 G. Commit 75,9 % ; 0×500 sur 564 req.

## 23/10 21:15-21:45Z — FENÊTRE DE MAINTENANCE (GO user Q27a+Q27b+Q26)

**Contexte** : 19 wedges en ~28 h (le watchdog tuait des moteurs sains-qui-
travaillaient, 9 cas documentés) + course pruneur/lecture 9/9 (~1 requête/2 h
perdue sur un bloc supprimé) + disque invité 92-93 % serrant le cap du tier.

**(a) Porte watchdog « prefill en vol » — Watchdog v6, ARMÉE 21:41Z.**
Sur `fail 1/2` : snapshot `vllm:prompt_tokens_total` + `generation_tokens_total`
lus sur `/metrics`. Sur `fail 2/2` : si les compteurs ont avancé → **WEDGE-BUSY**
(ligne dédiée, pas de restart, WEDGE_FAIL repositionné pour ré-évaluer à la
sonde suivante) ; si plats → restart comme avant (« counters flat » dans le
motif). **Plafond anti-dérive : 10 cycles BUSY (~5 min) puis restart**, et
réinit de BUSY_COUNT à chaque decode OK. Repli sûr : `/metrics` illisible →
compteurs 0 → plat → comportement v5. **Test à blanc 3 scénarios PASSÉ**
(avance→BUSY×3+restart au plafond ; plat→restart immédiat ; metrics KO→restart).
Recréation du seul service watchdog ; **le `depends_on` a aussi recréé le
moteur** (portait déjà le delta (b)) — l'indispo annoncée « en fin de fenêtre »
a donc eu lieu à ce moment (21:38:10Z, moteur healthy 21:40:27, decode OK
21:41:01). Première trace vivante : `21:41:32 WEDGE fail 1/2 (snapshot p=32
g=42)` — l'instrumentation fonctionne sous trafic réel.

**(b) `kv_load_failure_policy: recompute` — APPLIQUÉ, confirmé au boot.**
Une clé ajoutée au `--kv-transfer-config` du profil (champ top-level de
`KVTransferConfig`, `config/kv_transfer.py:88`). Vérifié dans les non-default
args du boot 21:38 : `kv_load_failure_policy='recompute'`, **KV 750 747
inchangé**. Effet : un bloc supprimé par le pruneur entre test d'existence et
lecture devient un re-préfill au lieu d'une requête perdue.

**Q26 — suppression 237,9 Go de modèles HF rejetés (GO user).** Liste
nominative publiée sur le dashboard AVANT le geste ; suppression via
`shutil.rmtree` in-container (le garde `guard-bash-composite` a refusé le
premier jet `cd`+`rm` — réécrit, comportement attendu). 10 repos supprimés
(tous « Models Tested and Rejected » : Ornith-1.0/1.5 ×5 quants, Qwen3.8-27B
×2 quants, Qwen3.6-27B dense, Qwen3.6-35B-A3B-FP8, dspark speculator) ;
**conservés** : Swift prod 19,6 + rollback MoE AWQ 49,5 (intact, refs=4) +
2 drafters des profils opt-in (2,3). **Disque invité 92 % → 60 %** (84 →
390 GB libres) — le cap du tier (min(48, libre−40, C:−12 %)) est désormais
plafonné par sa borne nominale 48 en toutes circonstances.

**Gates post-fenêtre** : health 200 (3,3 ms), génération réelle correcte
(« Quatre »), KV 750 747, shm purgé des orphelins du boot (8,49 GiB → 8,0 G),
watchdog v6 actif, tiers fs intact.

**Repli** : (a) = revert une ligne du service watchdog + recreate ; (b) = retirer
la clé du JSON et recreater ; Q26 = re-téléchargement HF (coût bande passante,
pas de perte irréversible : tous les repos sont publics).

## 23/10 22:41Z — PREMIER CYCLE POST-FENÊTRE : tout vert

- **62 min SANS restart** depuis la recréation 21:38:10Z (première heure
  sans wedge depuis le matin ; 99 sondes OK, le fail 1/2 de 21:41:32 n'a pas
  escaladé — pas encore de WEDGE-BUSY à arbitrer, la rafale ne s'est pas
  représentée).
- **Hit préfixe 95,85 %** (33,3 M/34,8 M) — 3ᵉ record consécutif. KV 19,6 %,
  0 préemption, charge plus légère (85 req/15 min).
- **Cap pruner à sa borne NOMINALE 48,0 GiB** pour la première fois depuis
  l'armement (tier 70,9 avant purge, 938 blocs/22,9 GiB supprimés, 0 err) ;
  disque invité **67 %** (317 GB libres).
- **0 ENOENT** dans la génération post-fenêtre (purge de 22:20 sans un seul
  miss) — `recompute` armé mais pas encore sollicité.
- Le 1×400 du census (20:41:45, corps 922 570 o, UA Bun = hub) = signature
  #431 connue (sur-fenêtre 262 144), pré-fenêtre, 1 occurrence/2 h = motif
  habituel. Commit 76,5 % ; VRAM 20 917/19 992 MiB (sous la référence).

## 23/10 23:46Z — ★ LA PORTE v6 A ARBITRÉ EN BUSY EN CONDITIONS RÉELLES (×2)

**C'est la preuve attendue depuis la fenêtre de maintenance.** Deux épisodes
`fail 2/2` ont été interceptés, compteurs en avant, moteur **non redémarré** :

```
2026-10-23 21:43:32 WEDGE-BUSY health=200 decode=000 — counters ADVANCED
             (p 32->202476 g 42->86), NOT restarting (1/10)
2026-10-23 23:27:37 WEDGE-BUSY health=200 decode=000 — counters ADVANCED
             (p 56409985->56642623 g 321175->321285), NOT restarting (1/10)
```

**Un troisième est tombé pendant la rédaction de cette note** :

```
2026-10-23 23:42:25 WEDGE health=200 decode=000 (fail 1/2; snapshot p=61295997 g=344122)
2026-10-23 23:44:26 WEDGE-BUSY ... counters ADVANCED (p 61295997->61295997 g 344122->344293)
```

Détail qui compte : sur ce troisième, **`p` est resté plat alors que `g` avançait**
— le moteur générait mais n'admettait plus de prefill (5 requêtes en attente).
La porte teste `p` **ou** `g` en avant, donc elle a correctement conclu BUSY ;
une porte qui n'aurait regardé que `p` aurait conclu à tort au wedge.

Les trois épisodes sont sortis en `RECOVERED` au cycle de sonde suivant
(`BUSY streak was 1`) : la fenêtre de famine s'est résorbée d'elle-même en
≤ 2 min. **Sous v5, ces trois-là auraient été trois démarrages** (KV effacé,
60 M de tokens de prompt à re-prefill), soit les 20ᵉ à 22ᵉ restarts de la
journée. Le plafond `BUSY_MAX=10` (~5 min) n'a jamais approché (streak max 1).

**Dénouement mesuré à 23:47:03Z** : KV **99,6 % → 66,5 %**, 5 en cours + 5 en
attente, `generation_tokens_total` 344 334 (en hausse), `num_preemptions_total`
toujours 20 — la saturation s'est vidée **par reprise**, sans restart et sans
nouvelle préemption. C'est exactement le comportement pour lequel la porte a
été armée.

**Point chaud en cours à la clôture de cette note** : le mode 5 « saturation
KV » est **actif** — KV 98,2-99,6 %, 7 flux en cours + 1 en attente,
`num_preemptions_total=20`, débit agrégé mesuré **~3 tok/s** (généré
344 122 → 344 176 en 17 s). Le moteur progresse (compteurs en avant, health
200 en 5 ms, 0 erreur, 0 ENOENT), donc **ce n'est pas un wedge** : c'est la
dégradation attendue sous 260K de contexte concurrent. On n'agit pas — un
restart effacerait le KV et reconstruirait le même état (c'est exactement ce
que la porte empêche).

**Correction d'instrument** : la hausse « disque 60 % → 69 % » est un
**artefact de comparaison** entre deux systèmes de fichiers distincts
(`overlay /` du disque docker-desktop vs `/dev/sdd /kv-tier-fs`). Mesure
propre sur 60 s : **+561 MiB/min, 0 nouveau bloc**, 297 GB libres, tier à
74 GiB (au-dessus du cap 48 GiB, la min-age 30 min protège les blocs de la
génération vivante → le pruner ne rogne que la queue froide). Aucun risque
disque. **Leçon : ne jamais enchaîner deux `df` de montages différents.**

**Reste** : census 2 h = **544 × 200, 0 × 400/500/502** ; les 23 × 401 sont
21 × `108.165.33.236` portant la clé **MEDIUM_VL morte** (`Bearer 7711C3D0…`,
retirée du `.env` le 20/10 → Q23) + 2 × 127.0.0.1 : nuisance connue, **zéro
200**, pas une brèche. nbaudit `ok:true` (5 séries, 3/série, 0 échec).
Condensation : client de repli construit à 23:05:51Z (chaîne de repli
sollicitée pendant la saturation, c'est le comportement voulu).

## 24/10 00:05Z — LE PRUNEUR ÉTAIT LA CAUSE DU « COMPTE-GOUTTE » : plafond 48 → 96 GiB (GO user)

**Symptôme rapporté** : « on sert les tokens au compte-goutte ».

**Mesure** : sonde de génération réelle = **80 tokens en 297 s (0,3 tok/s)**
avec GPU à 88 % / 256 W, donc un moteur qui travaille, pas qui dort. Sur la
même période `prompt_tokens_total` +566 640 dont **seulement 141 888 cachés
(25 %)** → le moteur recalculait 425 K tokens de contexte. Soixante secondes
plus tard, dans une fenêtre calme : **176 tok/s agrégé (~25 tok/s/flux) avec
98,8 % de cache** — soit exactement la valeur nominale documentée. Le débit
n'est donc pas bas en permanence, il s'effondre par épisodes.

**La charge est massivement longue** : histogramme `request_prompt_tokens`
sur les 543 requêtes du run → **179 dans (100 K, 200 K] et 133 > 200 K**, soit
**57 % au-dessus de 100 K tokens**.

**Le maillon manquant** : `fstier_state.json` montre la purge de
**23:50:02Z = 896 fichiers / 22,9 GiB supprimés**, en pleine sonde (la purge
de 22:20 avait fait 938 fichiers / 22,9 GiB). Ces blocs sont le **cache
préfixe** des conversations longues. Un contexte de 200 K remis en cache coûte
~1 s ; recalculé au débit de prefill du 27B dense (~1,5 K tok/s) il coûte
**~130 s**. La chaîne est donc : contexte long → blocs évincés vers le tier →
**pruneur les supprime** → tour suivant = miss → re-prefill géant → tout le
reste attend.

**Niveau de preuve** : corrélation mesurée sur deux points (25 % pendant,
98,8 % après) avec une purge datée au milieu — **pas encore une preuve
causale stricte**. Le contrôle prévu (échantillonnage du ratio toutes les
30 s à travers la purge de 00:20) a été **rendu caduc par le GO** : le plafond
relevé supprime la purge de 00:20, donc l'échantillonneur mesurera l'absence
de creux. Ce n'est pas une gêne — c'est le second bras de l'A/B — mais il faut
le dire plutôt que de présenter 00:20 comme un contrôle.

**Changement appliqué** (`fstier_prune.ps1`, une ligne + commentaire) :
`MaxGiB 48 → 96`. Réversible. Garde-fous **inchangés** :
`min(96 GiB, libre invité − 40 GiB, libre C: − 12 %)`.
Dry-run de vérification (00:02:23Z) : `max_gib 96.0`, `cap_bytes` 103 079 215 104,
`bytes_total` 59 056 128 000 (55 GiB), **`bytes_over_cap 0`**, `errors_count 0`,
libre invité 303 GiB, marge C: 169 GiB. **La purge de 00:20:02Z ne supprimera
rien.**

**Ce que ça achète et ce que ça n'achète pas.** À ~15,7 GB/h de croissance, la
rétention passe de **90 min à ~2 h 40**. C'est un facteur 1,8, pas une
solution : le besoin réel (57 % du trafic au-dessus de 100 K) dépasse tout
plafond de cache plausible. **Demande d'espace disque postée à Maintenance**
(`machine-myia-ai-01`, `[ASK]` + cross-post `workspace-Maintenance` + DM) pour
monter à 160-192 GiB si la marge existe ; y sont rappelés les ~149 Go de
volumes Docker anonymes et les 33 volumes `profiles_*` (sauf `…-v0320`, cache
de compilation de la prod) que notre inventaire RX113 du 22/10 laissait
récupérables.

**Plafond structurel non résolu** : le prefill du 27B dense est 3-4× plus lent
que le MoE remplacé le 25/09 (1,5 K vs 5,4-8,3 K tok/s). Aucun réglage de
cache ne compense ça. Si la réactivité prime sur la qualité, `medium-qwen36-stock-tq.yml`
reste le rollback armé.
