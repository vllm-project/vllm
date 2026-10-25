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

## 24/10 00:20Z — PRUNEUR INTELLIGENT : le crochet existe, il est MORT ; le bon est `lookup()`

Constat juste du user : le pruneur actuel est un bricolage (mtime + plafond,
sans rétroaction du moteur). Il ordonne la suppression par **date d'écriture**
alors que la valeur d'un bloc vient de sa **réutilisation**.

**Ce que le framework vLLM 0.32.0 semblait offrir.** `SecondaryTierManager`
déclare `touch(keys, req_context)` — « Mark chunks as recently used for
eviction policy » — et `TieringOffloadingManager.touch`
(`tiering/manager.py:591-601`) le relaie au tier primaire **et à chaque tier
secondaire**. `FileSystemTierManager` ne le surcharge pas : il hérite du no-op
et **jette** donc le signal. Une première version du module a été écrite sur ce
crochet.

**⚠️ Ce crochet est mort dans v0.32.0.** Un grep complet du paquet installé ne
trouve **aucun appelant** de `touch` du manager d'offload : les seuls `.touch(`
du code sont le *GPU block pool*, les adaptateurs LoRA, le connecteur mooncake
et `simple_kv_offload` — quatre mécanismes sans rapport. Un tier câblé sur
`touch` aurait été un **no-op silencieux qui a l'air installé**. C'est la
deuxième fois que « vérifier l'organe à la source » paie : la docstring décrit
l'intention, pas le code. Module initial supprimé, non commité.

**Le crochet vivant est `lookup()`.** `tiering/manager.py:454` appelle
`tier.lookup(key, req_context)` pour chaque tier secondaire, **par clé**, et
chronomètre l'appel ; `FileSystemTierManager.lookup` (`fs/manager.py:216`)
rend `HIT` / `MISS` / `RETRY`. Un `HIT` signifie exactement « ce bloc est dans
le tier et va être réutilisé ».

**Module livré** : `myia_vllm/kv_tier/lru_fs_tier.py`, classe `TouchOnHitFsTier`,
sous-classe de `FileSystemTierManager` qui surcharge `lookup()` et fait un
`os.utime()` sur les **HIT** uniquement. Le pruneur externe, inchangé, devient
alors un **LRU sur les lectures**. Ni journal, ni index : la récence vit dans
les métadonnées du système de fichiers, donc elle survit aux redémarrages du
process et de l'hôte. Interrupteur `MYIA_FS_TOUCH=0` pour revenir au
comportement amont sans toucher à la config.

**Pourquoi LRU est la bonne politique pour un arbre de préfixes** : les blocs
sont chaînés par hash, donc une conversation est une chaîne de fichiers.
Supprimer une racine partagée invalide tous ses descendants ; supprimer une
feuille n'invalide qu'une queue. Sous LRU-par-usage, les racines partagées sont
touchées à **chaque** tour de chaque conversation qui les utilise — elles sont
donc les blocs les plus récents du tier. La politique protège exactement ce
dont la perte coûte le plus cher, sans modéliser l'arbre.

**Vérifications faites sans redémarrer le moteur** (le module n'est pas encore
chargé en prod) :
- sous-classe correcte, `lookup` bien surchargé ;
- HIT → mtime remonté, **MISS → mtime inchangé** (témoin négatif), RETRY
  propagé ; compteurs `ok=1 miss=0 err=0` ;
- `SecondaryTierFactory.get_tier_class({'type':'TouchOnHitFsTier',
  'module_path':'lru_fs_tier'})` résout bien la classe (import hors-arbre,
  `importlib.import_module` + `getattr`, `assert issubclass`).

**Déploiement (NON appliqué — exige un redémarrage du moteur, donc GO user)** :
1. bind `./myia_vllm/kv_tier:/opt/kv_tier:ro` + `PYTHONPATH=/opt/kv_tier` ;
2. dans `kv_connector_extra_config.secondary_tiers[0]`, remplacer
   `"type":"fs"` par `"type":"TouchOnHitFsTier","module_path":"lru_fs_tier"` ;
3. redémarrer (~2 min) ; vérifier au boot que `FileSystemTierManager` est
   instancié via `TouchOnHitFsTier` et que les mtimes des blocs bougent alors
   que le moteur sert.

**Ce que ça ne règle pas** : le débit de prefill. Un LRU parfait évite de
recalculer ce qui était encore chaud, il n'accélère pas ce qui est froid.

## 24/10 00:25Z — ⚠ RÉTRACTION PARTIELLE : le pruneur n'explique PAS tous les creux

**Ce que j'ai mesuré sur 20 min d'échantillonnage (44 points, 30 s).** Trois
régimes distincts, et le plus mauvais n'a rien à voir avec le pruneur :

| régime | `d_prompt`/30 s | débit | cache | KV |
|---|---|---|---|---|
| prefill lourd | 150 K – 600 K | 48 – 205 tok/s | 59 – 100 % | 0,20 – 0,95 |
| décodage calme | 16 | 82 – 168 tok/s | n/a | 0,28 – 0,78 |
| **creux** | **0 – 16** | **0,3 – 0,9 tok/s** | n/a | 0,41 – 0,74 |

**Six fenêtres de creux en 20 min**, à 00:05:47, 00:06:17, 00:12:49, 00:13:19,
00:14:19, 00:14:49 — **aucune pendant une purge** (elles tombent à :20 et :50),
**aucune avec du prefill en cours**, **aucune avec KV saturé**, et
`num_preemptions_total` reste à 45 tout du long. **Le pruneur n'est donc pas
la cause de ces creux**, et mon attribution du « compte-goutte » au seul
pruneur (section précédente, PR #157) était **trop large**. Ce qui reste
établi : la purge du 23:50 a bien fait tomber le cache à 25 % cette fenêtre-là.

**La mesure qui explique les creux** — métriques `kv_offload` du journal moteur
(`[metrics.py:103]`, toutes les 10 s) :

```
00:18:57  lookup_async_delay ('1:fs') count=130  sum=2.76    -> 21 ms/appel
00:19:27  lookup_async_delay ('1:fs') count=104  sum=17.28   -> 166 ms/appel
00:19:47  lookup_async_delay ('1:fs') count=83   sum=199.88  -> 2,41 S/appel
00:19:47  tiering_write_time ('1:fs') = 9,51 s pour 3,94 GB écrits (414 MB/s)
```

Le **lookup** du tier fs est une simple vérification d'existence de fichier,
attendue sous la milliseconde. Elle prend **2,4 s** au moment exact où une
rafale d'écriture de 3,9 GB traverse le même disque. Le scheduler attend cette
réponse avant de progresser : le moteur ne décode plus, ne prefille plus, et
les compteurs avancent à peine — exactement la signature des six creux.

**Hypothèse testée et REJETÉE : les rafales d'écriture.** J'ai corrélé sur
**284 intervalles de 10 s** (21:41:14Z → 00:22:07Z) la latence de lookup du
tier fs et son volume d'écriture :

```
latence lookup ('1:fs') : med 19,3 ms   p90 767 ms   p99 6 111 ms   max 11 366 ms
intervalles > 500 ms : 32 / 284 = 11,3 %     > 2 000 ms : 27 / 284
  écriture > 0,5 s dans l'intervalle : n=110, latence moyenne   553 ms
  écriture <= 0,5 s                  : n=174, latence moyenne   438 ms
pires pics : 23:49:52 11,4 s (0,39 GB écrits) · 23:35:30 10,8 s (ZÉRO écrit)
```

Il n'y a **pas de séparation** entre les intervalles qui écrivent et les
autres (553 vs 438 ms), et le **pire pic de toute la fenêtre n'écrit rien**.
L'hypothèse « l'écriture sature le disque et bloque la lecture » est donc
**rejetée**. Elle était séduisante sur une fenêtre unique ; 284 intervalles la
désavouent. C'est la deuxième hypothèse que je retire dans cette note — et la
règle qui l'a attrapée est celle du dépôt : un point ne fait pas une pente.

**Ce qui reste, et qui est solide** : le lookup du tier fs, qui est une simple
vérification d'existence de fichier, a une **médiane de 19 ms** (trois ordres
de grandeur au-dessus de l'attendu), un **p90 de 767 ms** et **11,3 % des
intervalles au-dessus de 500 ms**. Le scheduler attend cette réponse avant de
progresser. C'est une cause candidate des creux, indépendante du pruneur **et**
du volume d'écriture.

**Ce qu'on ne sait pas encore** : si ces latences sont intrinsèques au chemin
`FileSystemTierManager` (pool de threads, machinerie asynchrone, `faccessat`
par clé) ou à son environnement (vhdx, page cache, contention CPU). La façon
de trancher n'est pas une analyse de plus, c'est un **A/B** : le même moteur,
mêmes charges, **sans tier fs** (RAM seule) — un redémarrage. À faire en
fenêtre, et c'est le seul geste qui répond à « ce tier aide-t-il ou coûte-t-il
plus qu'il ne rapporte ». **Ne pas relancer d'hypothèse avant cette mesure.**

## 24/10 00:41Z — ★ PREMIER ARBITRAGE BUSY MULTI-CYCLES : 6/10, zéro restart, et la latence du tier mesurée DANS la fenêtre

**La porte v6 a fait son travail dans un cas qu'elle n'avait jamais couvert.**
Six cycles BUSY consécutifs, 12 minutes, `RUNNING` jamais interrompu :

```
23:46:26 BUSY (2/10) p 61 295 997 -> 61 791 049   g 344 293 -> 344 405
23:48:26 BUSY (3/10) p 61 791 049 -> 62 002 009
23:50:26 BUSY (4/10) p 62 002 009 -> 62 423 897
23:52:26 BUSY (5/10) p 62 423 897 -> 62 761 047
23:54:26 BUSY (6/10) p 62 761 047 -> 63 023 073   g 344 782 -> 345 029
23:56:26 RECOVERED (decode ok after 1 wedge fails, BUSY streak was 6)
```

736 tokens générés en 12 min = **~1 tok/s agrégé**. Sous v5 c'était un
**redémarrage** qui aurait effacé 63 M de tokens de préfixe pour retomber dans
le même état. Le plafond `BUSY_MAX=10` n'a pas été atteint (streak max 6).

**La mesure qui manquait, prise dans la fenêtre elle-même** — latence moyenne
par appel de lookup du tier fs, par intervalle de 10 s :

| heure | latence | nb d'appels |
|---|---|---|
| 23:40:30 (avant) | 862 ms | 238 |
| **23:47:22** | **3 510 ms** | 305 |
| **23:49:52** | **11 366 ms** | 286 |
| **23:53:13** | **4 528 ms** | 120 |
| **23:54:23** | **4 867 ms** | 141 |
| 23:58:13 (après reprise) | **371 ms** | 251 |
| 23:58:43 | 603 ms | 159 |

Les latences de plusieurs secondes sont **concentrées dans la fenêtre BUSY** et
retombent sous la seconde dès la reprise. Corrélation forte, mécanisme
plausible (le scheduler attend la réponse du tier avant d'avancer).

**Réserve honnête sur la causalité.** La latence mesurée est un **délai
d'attente en file**, pas un temps CPU : 305 appels à 3,5 s en 10 s donne ~105
appels en vol (loi de Little), cohérent avec des files de 16 threads. Une
corrélation inverse reste possible — un moteur bloqué pour une autre raison
ralentit aussi son pool de lookup. Le sens de la flèche n'est pas établi, seule
la contemporanéité l'est. **C'est pourquoi l'A/B sans tier reste la prochaine
étape, et pas une conclusion.**

**Reste du cycle** : health 200 (4 ms), `RC=0`, **3 h 02 sans restart**,
260 sondes OK, OOM 0, VRAM 21 429 / 20 650 MiB, `/dev/shm` 8,1 G, disque
invité 71 % (281 GB), commit hôte **78,9 %** (marge 84,2 GB, sous le seuil 85 %),
G: 85 % et écriture OK.

**Le plafond 96 GiB se comporte exactement comme prévu** : passage de
00:20:03Z, `cap=96`, `bytes_total 66,5 GiB`, **`over=0`, `supprimé 0/0 GiB`,
0 erreur**. Première purge depuis le changement : aucune suppression. Le tier
n'a pas encore regrossi jusqu'au nouveau plafond.

**Census 2 h** : 518 requêtes, **268 au-dessus de 100 K tokens (52 %)** ;
statuts **200 × 495**, **400 × 3**, **401 × 20**.
- Les 3 × 400 sont tous du hub (Bun, `90.65.170.144`), corps **875 K à 1 007 K**
  → signature **#431 connue** (sur-fenêtre 262 144). Cadence plus élevée que
  d'habitude (1/2 h), à surveiller sans alerter.
- Les 401 sont la clé **MEDIUM_VL morte** `fp=1f7f6080cb65` (18) depuis
  `108.165.33.236` (70 requêtes, **0 × 200**) : nuisance connue.
- **IP nouvelle, sans 200** : `116.202.50.2` (3 requêtes, 0 × 200) — pas dans
  la carte. Pas d'alerte (la règle exige des 200), mais à revoir si elle
  revient chargée.

**Vigie split-brain** : `IdentityManager` émet deux fois à 00:26:47Z
« Recent heartbeat found for myia-ai-01 (0s ago) — concurrent instance
possible but not blocking ». Périmètre RSM, non bloquant, à relayer si ça se
répète — ce n'est pas un cron vLLM.

## 24/10 02:41Z — ★ PREMIÈRE UTILISATION DU PLAFOND BUSY (10/10 → restart) et le tier atteint 96 GiB

**La porte v6 a utilisé son plafond anti-dérive pour la première fois.**
Streak 8/10 → 9/10 → **10/10** de 02:05:08 à 02:09:08, puis
`RESTARTING (BUSY ceiling 10 reached ~5min — anti-drift fallback)`.
Le moteur a reçu **5 minutes pleines** de sursis pendant qu'il servait ~0,6
tok/s — puis le watchdog a arbitré le restart.

**État du moteur dans la fenêtre** (journaux `loggers.py`, 10 s) : KV
**80-97 %**, génération **0,3-1,2 tok/s**, prefill ~0 avec **un burst de
21 518 tok/s à 02:08:58** (re-prefill), **4-5 requêtes `Deferred`** et 5-8 en
attente. C'est le mode saturation/famine documenté — le moteur vit mais ne
sert plus. Le restart était l'arbitrage prévu.

**Les 5 × 500 du census (02:10:29-32) sont une CONSÉQUENCE du restart, pas sa
cause** : `EngineDeadError` remonte par `serving.py:929` pendant l'arrêt de
l'ancien process — 4 requêtes hub (Bun, corps 227-632 K) perdues en vol + la
sonde du watchdog elle-même (166 o, `curl`). C'est le coût client-visible
connu d'un restart, concentré sur sa frontière.

**Après restart** (boot 02:10:33, sain) : cache préfixe **remis à zéro** —
hit 5,5 % (848 K/15,3 M) en reconstruction, KV 88,5 %, 8 préemptions. Deux
`fail 1/2` isolés à 02:25 et 02:28, tous deux `RECOVERED`. C'est le prix
documenté : le restart qui efface le KV déclenche la vague de re-prefill.

**Le tier a atteint le nouveau plafond** : passage de 02:20:02Z avec
`total 109,1 GiB`, `over 13,1`, **536 fichiers supprimés**, 0 erreur. La
croissance mesurée est **~21 GiB/h** (66,5 → 109,1 en 2 h), plus rapide que
l'estimation initiale (15,7). À ce rythme, le cycle purge est ~4-5 h contre
~90 min à 48 GiB : le facteur d'amélioration tient (~3×), la purge n'est pas
éliminée. `/dev/shm` monté 8,1 → 17 G (26 %, sous le seuil 45 G) — le tier
RAM y vit.

**Census 2 h** : `200 × 454`, `400 × 1`, `401 × 16`, `500 × 5` (frontière du
restart). **nbaudit** `ok:true` (02:06:08Z). **Condensation** : clients
primaire et repli reconstruits à 00:55-00:56Z, nominal ensuite.
**Commit hôte 78,0 %** (marge 87,7 GB), G: 85 %, disque invité 73 %
(264 GB), VRAM 21 495 / 20 650 MiB.

**Vigie** : `IdentityManager` « concurrent instance possible for myia-ai-01 »
récidive (01:38:29Z, 02:31:20Z) — 4 occurrences en 2 h. Périmètre RSM : à
relayer au propriétaire si ça continue, ce n'est pas un cron vLLM.

**Leçons d'instrument** (deux erreurs corrigées avant publication) :
- moyenner un ratio de cache sur des fenêtres à 16 tokens de prompt donne
  53 % de « cache » là où la vraie valeur est 92 % — un dénominateur
  négligeable fabrique un chiffre faux ;
- `docker logs` a **manqué des échantillons** (un intervalle de 60 s au lieu
  de 30) : toujours calculer le débit sur le `dt` réel, jamais sur la cadence
  supposée.

## 2026-10-24 04:41Z — le tier fs bascule en lecture ; 6 × 400 inexpliqués mais instrumentés

**Moteur** : `StartedAt 02:10:33Z`, **2 h 30 sans restart**, RC=0, health 200
en 3,8 ms, OOM 0, 3 conteneurs Up. 4 `WEDGE fail 1/2` (02:25:44, 02:28:48,
03:55:24, 04:39:13) tous RECOVERED, **0 WEDGE-BUSY** : la charge est retombée
(4,6 req/min sur 6 h) et la porte n'a rien eu à arbitrer. Cache préfixe
**71,4 %** (64,08 M / 89,81 M tokens depuis le boot). Préemptions 8 (inchangé
depuis le boot — pas de saturation KV). VRAM 21 491 / 20 650 MiB, GPU 2
8 378 MiB (CoursIA). `/dev/shm` 17 G (26 %), sous le seuil.

### ★ Le tier fs est passé en lecture — la pathologie du 21/10 est inversée

Compteurs cumulés depuis le boot (un seul moteur, aucune ambiguïté de
génération), fenêtre 2 h 25 :

| Mesure | Valeur | Débit |
|---|---|---|
| `read_bytes{tier=1:fs}` | **282,4 GB** | 32,5 MB/s |
| `write_bytes{tier=1:fs}` | **190,9 GB** | 21,9 MB/s |
| `chunk_queries{tier=1:fs}` | 21 138 | — |
| `chunk_hits{tier=1:fs}` | **10 748 (50,8 %)** | — |
| `read_time` / `write_time` | 109,8 s / 628,3 s | — |

Le **ratio lecture/écriture vaut 1,48:1**, là où le 21/10 il était de 1:18
(39,8 GB écrits pour 6,9 GB relus). Le tier n'est donc plus un puits
d'écriture : il **sert**. Preuve indépendante dans le log moteur à 02:48:07Z :
`External prefix cache hit rate: 93,1 %`. C'est la réponse directe à
l'inquiétude d'usure disque du 21/10 — le disque écrit autant, mais il rend.

À noter pour ne pas sur-lire : `read_time` (0,5 s de moyenne par tranche) et
`lookup_async_delay` (17 931 s / 20 328 = **882 ms de moyenne**) mesurent des
distributions à queue lourde, pas un régime homogène — la médiane reste basse.

### Plafond 96 GiB exercé, et le terme contraignant a changé

Passage du pruneur à **04:20:02Z** : `bytes_total 133,86 GB` (124,7 GiB),
`over cap 30,78 GB`, **1 173 fichiers supprimés**, `errors_count 0`,
`elapsed_s 18,2`. Croissance nette **14,3 GiB/h** → rétention ~6-7 h.
Le terme qui mord est désormais **`max_gib` lui-même** : marge C: **173,3 Go**,
marge invité 227,5 Go, plafond 96 GiB. Le disque n'est plus le limiteur.

**Conséquence de méthode** : je n'ai PAS relevé le plafond une seconde fois.
La question ouverte n'est pas « plus de place » mais « la rétention
est-elle le facteur limitant », et aucune mesure ne le dit encore. C'est
l'A/B du tier fs (Q28, module `lru_fs_tier.py` prêt, PR #158) qui doit
trancher. Une hausse à l'aveugle aurait consommé la marge de Maintenance
sans preuve qu'elle achète quoi que ce soit.

### ★ 6 × HTTP 400 non expliqués — signature #431, cause NON établie

Tous de la même forme : `path=/v1/chat/completions`, UA `Bun/1.3.14`
(front hub), `90.65.170.144`, clé MEDIUM, corps **712 K → 1,01 Mo**,
session Claude Code (system prompt « You are NOT Claude »).

| Heure UTC | Corps (o) |
|---|---|
| 23:47:27 / 23:47:30 | 958 062 (**paire = retry**) |
| 00:06:21 | 1 007 885 |
| 01:57:04 | 712 403 |
| 02:48:06 / 02:48:16 | 942 554 (**paire = retry**) |

**Ce que la mesure dit** : sur la même fenêtre, 31 réponses **200** dépassent
900 K o et la plus grosse atteint **945 341 o**. Donc ce n'est **pas** une
frontière de taille : 945 K passe, 942 K échoue. L'hypothèse « contexte trop
long » reste la plus plausible (le nombre de tokens, pas d'octets, décide),
mais **elle n'est pas prouvée** : vLLM ne journalise pas le motif d'un 4xx à
ce niveau de verbosité, et `--tail 20 000` ne contient ni `BadRequest` ni
`maximum context`.

C'est exactement la signature que la spec demande de chercher
(`400` + `body_bytes > 100 000`). Une paire identique à 3 s d'intervalle est
en plus le motif de **retry local** que #431 interdit.

**Correctif d'instrument, pas de rustine** : `error_source_capture.py`
capture désormais le **corps de réponse des non-2xx** (`ERROR_SOURCE_ERR_BODY_BYTES`,
2 Kio, tamponné seulement si `status >= 400`). Le chemin 2xx/SSE ne paie rien ;
vérifié par test sur app ASGI factice (400 → corps capturé, 200 de 400 K o
diffusé en 10 morceaux → rien d'accumulé). **Inerte jusqu'au prochain
restart** : la prochaine occurrence sera diagnosticable d'elle-même.

### Sécurité — le mur tient, deux sources nommées

- **`108.165.33.236`** : `Go-http-client/1.1`, **49 sondes en 6 h** (une
  toutes les ~7 min) uniquement sur `/v1/models`, **49 × 401**. Clé employée =
  **MEDIUM_VL morte** (`1f7f6080cb65`, retirée du `.env` le 20/10 → Q23).
  Ce n'est pas un consommateur mais un sondeur ; aucun 200, rien à révoquer.
- **`ZOLTRAAK/0.2.0`** (scanner) : 3 requêtes à 03:53Z sur `/v1/models` avec
  des clés inventées (`ccd500f719e4`, `64a6d7ec06fd`) → **3 × 401**.
- Aucune IP inconnue avec des 200. `90.65.170.144` (hairpin site) et
  `37.187.180.135` (web1) restent les seules sources avec 200.

### Hygiène

nbaudit `ok:true` (04:06:13Z, 5 séries × 3) — et **`subpath` a basculé sur
`19451:Image/`** : la vague 2 est armée, la fanion posée le 10/10 est levé.
Disque invité **74 %** (255 GB libres), commit hôte **77,0 %** (marge 91,8 GB),
G: 85 %. Condensation nominale (bannière 3 étages `key=OK`, repli cloud actif,
aucun événement). Les 5 × 500 du census sont ceux de la frontière du restart
de 02:10, déjà documentés.

## 2026-10-24 06:41Z — RÉTRACTION : le « basculement en lecture » du tier fs n'était qu'une fenêtre ; 7ᵉ × 400

**Moteur** : `StartedAt 02:10:33Z`, **4 h 30 sans restart**, health 200 en
4,0 ms, RC=0, OOM 0, 771 sondes OK. Un seul `fail 1/2` (06:39:07, snapshot
p=152,5 M) → **RECOVERED 06:41:07**, streak BUSY 0. Préemptions **8
inchangées** (aucune saturation KV depuis le boot). VRAM 21 381 / 20 650 MiB,
`/dev/shm` 17 G (26 %), disque invité 75 % (248 GB), C: 608 GB libres,
commit hôte 76,3 % (marge 94,5 GB), G: 85 %. nbaudit `ok:true` (06:06:07Z,
vague 2 `Image/` en cours). Pruneur 06:20:02Z : 111,1 GiB total, 15,1
au-dessus, 618 fichiers supprimés, 0 erreur — le régime de purge à 96 GiB
est stable (2 cycles observés).

### ★ Rétraction — « le tier a basculé en lecture » était un effet de fenêtre

Fenêtre 04:41Z → 06:41Z (compteurs cumulés, même génération moteur) :

| Compteur | 04:41Z | 06:41Z | Δ 2 h |
|---|---|---|---|
| `read_bytes{fs}` | 282,4 GB | 322,0 GB | **+39,6 GB** (5,5 MB/s) |
| `write_bytes{fs}` | 355,6→ | 355,6 GB | **+164,7 GB** (22,9 MB/s) |
| `chunk_queries{fs}` | 21 138 | 27 151 | +6 013 |
| `chunk_hits{fs}` | 10 748 | 12 255 | +1 507 (**25,1 %**) |

Le ratio lecture/écriture de la fenêtre précédente (**1,48:1**) ne tient pas :
cette fenêtre est à **1:4,2**, et le cumul depuis boot repasse sous l'égalité
(322,0 lus / 355,6 écrits = **1:1,10**). **Interprétation corrigée** : la
fenêtre 02:41-04:41 était la **phase de re-jeu post-restart** — le cache GPU
avait été remis à zéro à 02:10 et les sessions relisaient leurs préfixes
survivants du disque. En régime établi, le tier fs redevient
**écriture-dominé** (nouveaux blocs), avec un taux de hit chunk de ~25 %.
La leçon s'applique à moi-même : *deux points ne font pas une pente* —
j'ai annoncé « la pathologie du 21/10 est inversée » sur une seule fenêtre
de 2 h ; c'était le profil transitoire, pas l'état stable.

Ce que la mesure **garantit** toujours : la valeur du tier se concentre
(a) dans la survie au restart (prouvée 20/10) et (b) dans le replay à chaud
des préfixes évictés du GPU — 25 % de hit chunk en régime établi n'est pas
négligeable, mais ce n'est plus « le disque rend plus qu'il n'écrit ».

### Amortissement préfixe, lui, est excellent et durable

`prefix_cache` cumul : 122,07 M hits / 152,64 M queries = **80,0 %**.
**Incrémentiel 2 h : 57,98 M / 62,83 M = 92,3 %** — chaque tour ne paie
presque que son delta. C'est le régime sain décrit en §5quater. Les 92 %
viennent du couple GPU+RAM ; le fs sert la queue longue (25 % de ses
probes). Cohérent avec l'objectif initial du multi-tier.

### 7ᵉ × 400 de la même famille (05:02:03Z, corps 938 639 o)

Même signature : Bun hub, `90.65.170.144`, clé MEDIUM, session Claude Code,
corps sous la barre du plus gros 200 observé (945 341 o). **7 occurrences
depuis 23:47Z** (958 062×2, 1 007 885, 712 403, 942 554×2, 938 639).
La capture du corps de réponse (PR #163) reste **inactive** — le moteur
n'a pas redémarré depuis son commit ; le champ `error_body` est absent des
1 255 lignes du log, conforme à l'attente. Le diagnostic suivra au
prochain restart.

### Census 2 h

591 req (4,9/min) : `200 × 570`, `401 × 20` (sondeur `108.165.33.236`,
clé MEDIUM_VL morte — mur tenu), `400 × 1` (ci-dessus), **0 × 500**.
Aucune IP inconnue avec 200. Condensation : aucun événement depuis 00:56Z,
bannière vivante ; deux WARN `STALE-FILE #3230` (05:46Z) = périmètre RSM,
la vue PG sert de secours comme prévu.

## 2026-10-24 08:17Z — ★ DÉPLOIEMENT `TouchOnHitFsTier` (GO user ~08:05Z) : l'éviction fs est un LRU sur l'usage

**Contexte immédiat** : le moteur était dans un épisode BUSY 7/10 (1 t/s,
plafond attendu 08:20:53) après un **2ᵉ déclenchement du plafond BUSY à
07:09:33** (restart propre 07:11:06). Déployer pendant l'épisode fait
converger les deux événements : le recreate compose est tombé dans la
branche patiente `BOOTING` du watchdog → **une seule fenêtre d'indispo** au
lieu de deux (restart watchdog + recreate).

**Pré-vol (dans le CONTENEUR réel, pas sur l'hôte)** — tout PASSÉ :
- `docker cp` du module + import : OK, `issubclass(FileSystemTierManager)` OK ;
- `file_mapper` / `get_file_name` présents dans le code installé (vérifiés
  par `inspect.getsource`) ;
- `SecondaryTierFactory.get_tier_class({'type':'TouchOnHitFsTier',
  'module_path':'lru_fs_tier'})` résout vers notre classe (le warning
  « experimental » est celui du chemin out-of-tree documenté) ;
- tests unitaires exécutés dans le conteneur : **ALL PASS** (mtime bumpé sur
  hit d'un bloc réel, miss propre sur fichier absent sans exception, `touch()`
  sur collection). Hôte sans vllm → les tests DOIVENT tourner in-container.

**Déploiement** : 3 deltas dans `medium-swift15-27b-fstier.yml`
(`type/module_path` du tier, bind `./../../../kv_tier:/opt/kv_tier:ro`,
`PYTHONPATH=/middleware:/opt/kv_tier`), `docker compose config` validé avant,
`up -d` → recreate. **Boot 08:17:27Z, compile 1,23 s** (cache v0320 chaud),
init moteur 109 s, healthy. **`Loading out-of-tree secondary tier
'TouchOnHitFsTier' from 'lru_fs_tier'` dans les DEUX process** (EngineCore +
APIServer), zéro ImportError. **KV 750 747 tokens identique**, nvfp4, noms
duaux inchangés. Génération réelle : « Quatre » (22 prompt / 3 gen).
`/dev/shm` purgé post-boot : 21 orphelins, 8,0 G/63 G (13 %).

**Gates restants au prochain cycle** (nécessitent du trafic) :
1. `chunk_hits{tier=1:fs} > 0` depuis 08:17:27 — preuve que les blocs
   d'avant le restart resservent ;
2. **preuve du touch** : les 8 plus vieux blocs échantillonnés à 08:16:13Z
   (`/tmp/oldest_blocks_before.txt` côté WSL) doivent voir des mtime
   avancer après des hits — c'est la démonstration LRU en production.

**Rollback** : `MYIA_FS_TOUCH=0` dans l'env conteneur (kill switch SANS
restart — chaque touch devient no-op) ; retour complet = 3 lignes de profil.

### Découverte adjacente — la « version industrielle » est DÉJÀ dans l'image

`pip list` dans le conteneur : **`lmcache 0.5.6`** et
**`mooncake-transfer-engine-cuda13 0.3.13.post1`** sont installés dans le
stock `vllm/vllm-openai:v0.32.0`, et `import lmcache` fonctionne (backend
cuda auto-sélectionné). Le connecteur `LMCacheConnectorV1` (LRU interne,
plus de pruneur externe) est donc le **candidat industriel de la prochaine
itération** — mais basculer de connecteur = revalider nvfp4, la survie au
restart, le comportement WSL2 : une **fenêtre d'éval dédiée** (méthode du
20/10), pas un flip en flux tendu. Mooncake (Kimi/Moonshot, FAST'25) reste
la référence architecture si on passe multi-moteurs. Le module déployé
aujourd'hui ne remplace PAS ce code industriel : il rebranche le signal de
fraîcheur (`touch()` sans appelant en v0.32.0) du tiering upstream qu'on
fait déjà tourner.

### Mesure — où les préfixes des sessions fourchent (→ issues flotte)

Census 3 h (419 requêtes Claude-Code-like, `body_head`) : **préfixe commun
à TOUTES = 143 octets (~35 tokens)**, toutes paires divergent avant 300 o.
Fourches nommées : **143 = paragraphe de mode Roo** (`Tu es Roo en mode
ask-simple/code-simple/orchestrator-simple`), **180 = variante de harnais**
(Agent SDK vs CLI), **287 = output style**. Toute différence dans le
premier paragraphe détruit tout le partage aval (outils, CLAUDE.md,
historique) — d'où l'option structurelle « base commune EN PREMIER,
mode/style à la FIN » plutôt qu'« harmoniser ».
Issues : **roo-extensions#4334** (remédiation harnais) + **claudish#640**
(jumelle, mesure au hub) + DM po-2025:claudish. Enjeu en $ : cache de prompt
Anthropic à 0,1× sur le socle partagé, prefix caching OpenAI ≥1024 tokens,
et re-prefill évité sur vllm-myia.

### Cycle 24/10 08:41Z — GATES LRU PASSÉES (2/2) + MYSTÈRE DES 400 RÉSOLU

**LRU en production — les deux gates différées du déploiement PASSÉES :**

1. **Reservice des anciens blocs** : `chunk_hits_total{tier="1:TouchOnHitFsTier"} = 406`
   en 25 min post-boot (boot 08:17:27Z), `prompt_tokens_cached_by_source{disk} =
   259 168 tokens` servis depuis le disque, `kv_offload_tiering_read_bytes_total =
   10,66 GB`. Le label du tier porte le nom de la classe — preuve que le module
   out-of-tree est bien le tier actif. Écritures 63,4 GB sur la même fenêtre
   (write-heavy conforme à la rétraction du 06:41Z).
2. **Preuve mtime par birth-time ext4** (le fichier baseline `/tmp/oldest_blocks_before.txt`
   était vide — capture ratée à 08:16Z, remplacée par la méthode plus directe) :
   sur 500 blocs à mtime récente, **111 sont nés AVANT le boot** (ex. né
   06:55:56Z, mtime 08:25:21Z) — impossible sans `utime()` : le hook touche au
   hit, le pruner oldest-mtime-first protège donc désormais les blocs en usage.
   Nouvelle baseline 8-plus-vieux posée (`/tmp/oldest_blocks_v2.txt`, CR-stripped).

**Santé post-deploy** : health 200 en 3,2 ms, 25 min up, 833 sondes OK, OOM 0,
préemptions **0** depuis boot, VRAM 21 095/20 240 (plage en charge), shm 8,1G/63G
(purgé au deploy), prefix-hit 95,7 % (17,13 M queries / 16,40 M hits), commit
hôte 79,1 % (marge 83,4 GB), G: OK, nbaudit ok — **vague 2 armée :
`subpath` basculé sur `19451:Image/`** (Audio/ n'a plus de livrables en attente).

**Le RESTARTING 08:16:53Z** du watchdog = streak BUSY 7/10 dont les compteurs
sont passés plats (p=0 g=0) PENDANT le recreate compose du deploy — la sonde
metrics voyait le moteur partir. Coïncidence des deux autorités, issue saine
(une seule indispo).

**⚠ Croissance du tier 38 GiB/h mesurée** : pruner 08:20:02Z a vu 110,3 GiB,
supprimé 276 fichiers/7,24 GiB (retour cap) ; à 08:45Z `du` = 119G. Rétention
~2,5 h au cap 96G. La croissance outpace la nuit (~21 GiB/h) — à suivre au
prochain cycle ; si soutenu, le tier churne mais la fenêtre de réutilisation
conversationnelle (~min) reste couverte d'un ordre de grandeur.

**★ MYSTÈRE DES 400 RÉSOLU — la capture `error_body` a attrapé la cause au
premier essai** (07:20:58Z, Bun, bb=926 908) :

```json
{"error":{"message":"This model's maximum context length is 262144 tokens.
However, your prompt contains at least 262145 input tokens, which exceeds
this limit even without any output tokens. (parameter=input_tokens,
value=262145)","type":"BadRequestError","param":"input_tokens","code":400}}
```

C'est la signature exacte chassée depuis la spec §2 : **window+1 = 262 145**.
Une session routée sur Swift dépasse la fenêtre **avant** son auto-compact —
l'auto-compact flotte est réglé à 263 000 > 262 144, le bandeau 262,1-263K ne
peut mécaniquement finir qu'en 400 (même famille que l'incident CoursIA-2 du
22/10 à 269 844 tok). 3× 400 sur la fenêtre 2h (07:07 bb=868K, 07:09 bb=466K —
corps de réponse vides, à qualifier — et 07:20 documentée), toutes UA Bun =
front hub, toutes `model=swift-1.5-27b`. Le remède n'est PAS côté moteur :
(i) auto-compact < plus petite fenêtre déclarée (~250K), (ii) règle #431
(router cette requête vers le cloud, jamais retry local). Rendu au user +
DM po-2025:claudish (this cycle). Les 401 restants (12) = clé VL morte
(Go-http-client, nuisance connue).

**Census 2h** : 405 req (~3,4/min = 205/h, en baisse vs 259/h) ; 273 req >100K
toutes du front hub (hairpin 90.65.170.144) — pas de re-prefill pathologique
par IP externe ; aucune nouvelle IP avec 200 (seul le hairpin + `?` direct).

### Cycle 24/10 10:41Z — moteur stable 2 h 09, ZÉRO 400, et deux mesures à ne pas surinterpréter

**Prod** : `StartedAt` inchangé 08:17:27Z (2 h 09), RC=0, health 200 en 9,1 ms,
1003 sondes OK, OOM 0, shm 8,1G/63G, 3 conteneurs Up. Deux `fail 1/2`
RECOVERED (10:20:17→10:22:17, 10:25:56→10:27:56) = famine prefill documentée,
moteur sain, zéro restart. VRAM **21 681 / 20 778** — 80 MiB au-dessus de la
plage de référence en charge (21 400-21 600), **1,3 GiB sous le seuil d'alerte
23 000** : à surveiller, pas une alerte. GPU 2 = CoursIA (8 449 MiB).

**★ Zéro 400 sur la fenêtre** (3 la précédente). `error_body` capture désormais
**20/20 des 4xx** : les 401 récurrents toutes les 6 min (Go-http-client, clé
morte connue) portent enfin leur motif `{"error":"Unauthorized"}` — la capture
non-2xx est complète, plus de 4xx muet.

**Deux mesures qui demandent un 3ᵉ point avant toute conclusion :**

1. **Hit préfixe incrémentiel 44,4 %** (85,9 M queries / 38,2 M hits sur 2 h)
   contre 92,3 % au 06:41Z et 95,7 % au 08:41Z. ⚠ Le 95,7 % du 08:41Z est
   **biaisé** (25 min post-restart, dominé par le re-jeu des mêmes préfixes) —
   même piège que la rétraction du 06:41Z. La comparaison honnête est
   92,3 % (4,6 req/min) → 44,4 % (6,0 req/min) : **la charge seule (+30 %) 
   n'explique pas −48 points**, l'hypothèse première est le **churn du tier**
   (pruner supprime 1 366 fichiers/cycle, ~35 GiB ; le tier tourne en ~1 h).
   Discriminant au prochain cycle : même charge sans restart → si 44 % tient,
   churn ; si ça remonte, artefact.
2. **Préemptions 52** (0 au 08:41Z, 8/2 h aux cycles précédents à charge
   moitié) — cohérent avec la charge doublée et des sessions 250K concurrentes
   (3 sessions pleines = 750K = tout le KV GPU). Pas d'action : les leviers
   (§5quater) sont déjà tous engagés (nvfp4, tier 8 GiB RAM + 96 GiB fs).

**Tier fs — lecture-dominé sur une fenêtre PLEINE, pas un re-jeu post-restart** :
+467 GB lus vs +205 GB écrits sur 2 h de trafic réel (**2,3:1**), contre 1:4,2
le 06:41Z. Mécanisme attendu du correctif LRU : les blocs touchés survivent aux
purges et resservent. **Ce qui manque pour le dire établi : un 3ᵉ point sur une
fenêtre sans restart** — même exigence que la rétraction du 06:41Z.

**Croissance du tier — la lecture 38 GiB/h du 08:41Z est CORRIGÉE à 17,9 GiB/h**
(pruner 10:20:03Z : 103 → 138,9 GiB en 2 h ; 1 366 fichiers supprimés, 35,8 GiB
au-dessus du plafond ; 0 erreur). Les 38 GiB/h étaient un **artefact de fenêtre
courte** (25 min) — *deux points ne font pas une pente*, appliqué une 2ᵉ fois à
ma propre mesure en 24 h. Rétention réelle ≈ 5 h au cap 96 GiB.

**Census 2 h** : 715 lignes, **6,0 req/min = 360/h** (+75 % vs 205/h) ;
695×200 / 20×401 / **0×400 / 0×5xx** ; 482 requêtes >100K **toutes** du front
hub (hairpin `90.65.170.144`, 506/506 en 200) — taille attendue d'un tour
Claude Code (le contexte entier est renvoyé), **pas** une signature de
re-prefill par elle-même ; aucune IP nouvelle avec 200.

**Reste du bilan** : condensation nominale (clients reconstruits 08:29/08:31,
0 échec, 0 repli cloud) ; nbaudit ok 10:06:02Z (vague 2 `Image/`) ; G: OK
(577 GB libres, 85 %) ; commit hôte 79,5 % (marge 81,6 GB) ; 1 WARN RSM
`[retirement #3782] lookup timed out — fail-open` (périmètre RSM, consigné).

### Addendum 10:45Z — TAXONOMIE DES 400 (12 occurrences sur 24 h) et correction d'une hypothèse

Le dépouillement horodaté (epoch + UTC + local) de **tous** les 400 du journal donne
deux familles **disjointes** :

| famille | critère | occurrences | corps réponse |
|---|---|---|---|
| **mur de contexte** | `model` parsé, body 875 K-1,01 Mo | 20:41:45, 23:47:27, 23:47:30, 00:06:21, 02:48:06, 02:48:16, 05:02:03, 07:20:58 (24/10) | JSON `maximum context length … 262144` (quand la capture est chargée) |
| **upload abandonné** | `model=None` + corps 467-868 K | 01:57:04, 07:07:23, 07:09:34 | **vide** |

**Le discriminant est `model=None`** : le middleware ne parse `model` qu'au dernier
chunk de requête (`more_body=false`). `model=None` ⇒ **le corps n'est jamais arrivé
complet** ⇒ uvicorn émet un 400 sans corps. Ce ne sont **ni des murs de contexte ni
des retries** — ce sont des abandons client en cours d'upload. Corroboré par le hub
(po-2025:claudish, DM 11:11Z) : **aucune ligne de mur côté hub à 07:06-07:10Z**, et
le « 868 K → 466 K » que j'avais lu comme un retry condensé est en réalité **le client
qui revient après compactage**.

**Correction formelle** : mon DM de 08:48Z proposait l'hypothèse « retry local
condensé ». Elle est **fausse** — retirée. Ce qui tient : #431 n'est pas en cause (la
marche haiku→Swift est unique, le 400 remonte au ContextGuard qui synthétise), et le
hub **apprend** un cap (`cap=229645`, short-circuit depuis 07:21:52Z).

**Fait vérifié côté moteur** : `GET /v1/models` déclare **`max_model_len = 262144`**
sur les deux noms servis. Le `280 000` du profil hub `vllm-myia` est donc une valeur
**figée côté hub**, pas une donnée du moteur — le remède robuste est de la **dériver
de `/v1/models`**. Risque documenté au hub : le cap appris vit en mémoire et est
**perdu à chaque recreate**, ce qui rouvre la bande 262,1-280k jusqu'au réapprentissage.

**Horloge** : le conteneur tourne en **TZ=UTC** (`local == utc`, pas +2) ; le mur de
07:20:58Z côté moteur correspond à la seconde près à la ligne du hub — discipline de
fuseau confirmée des deux côtés.

## Incident 24/10 10:45Z — arrêt de Docker Desktop : pas un crash vLLM, VM bloquée dans `sync(2)`

**Demande du user (11:25Z)** : réarmer le cron après le crash et le reboot, chercher si
le crash WSL/Docker vient de mes derniers changements (TouchOnHitFsTier 08:17Z,
`error_body`, plafond pruneur 96 GiB), et enquêter avec Maintenance.

**Cause de l'arrêt (VÉRIFIÉ, reconnue par Maintenance)** : à **10:45:00.907Z**, le
backend Docker Desktop reçoit `docker-desktop-cli POST /app/quit`, la commande
`docker desktop restart` de Maintenance (fenêtre #15157). Elle devait corriger un
`cgroup-parent` de `daemon.json` qui avait placé les 50 conteneurs, dont le moteur,
dans `coursia.slice/coursia-ci.slice` (memory.high 56 Gio, memory.max 66 Gio,
partagés avec la CI et qdrant). Le dump de goroutines de 10:45:30.96Z vient de la
minuterie de 30 s du handler `QuitDockerDesktop` (app.go:99). **Ce n'est pas un panic.**

**Arrêt du moteur : propre.** Il reçoit `[shutdown] API server: shutdown triggered`
à 10:45:03Z, alors qu'il servait normalement juste avant (KV 43-88 %, prefill 600 à
1 900 t/s). Il est sorti à 10:45:07.36Z, exit 0, soit **3,6 s** après le SIGTERM.
Les **39 conteneurs** sont tous sortis entre 10:45:04.0Z et 10:45:07.6Z, exit 0 ;
`claudish-sidecar` à 10:45:04.2Z malgré son StopTimeout de 120 s.

**Ce qui a bloqué la VM (VÉRIFIÉ, `vm/init.log`)** :

| heure (Z) | événement VM |
|---|---|
| 10:45:03.724 | `control: gracefully shutting down` → SIGTERM à dockerd |
| 10:45:15.50 | `dockerd stopped gracefully after 11.77s` |
| 10:45:15.80 | poweroff `all processes have shutdown` |
| 10:45:25.53 | `init` ERROR `timeout waiting for init services to stop` — goroutine 1 bloquée dans **`unix.Sync()`** (sync(2)) ← `poweroff.shutdown` (shutdown.go:56) |
| 10:47:03.667 | backend : `stopped waiting for VM to be ready to power off: context deadline exceeded` (120 s après 10:45:03.7) |
| 10:47:40 | relance ratée (`bootstrapping main distribution … timed out`) → reboot Windows du user à 11:00:55Z |

**Aucun conteneur n'a bloqué l'extinction.** C'est le `sync()` lancé avant
l'extinction qui n'est pas revenu, pendant plus de 108 s. Dans WSL2, `sync()`
s'applique à tout le noyau de la VM : il vide aussi l'ext4 de la distro Ubuntu,
où vit le palier fs, le VHDX qdrant et les montages 9p/drvfs vers Windows. **Le
palier fs reste donc un suspect légitime, mais les mesures ne le désignent pas.**
Données sales de la VM Ubuntu à 10:36Z : **27,9 Mo** (reclaim-wsl-memory.log de
Maintenance). `file_dirty` du cgroup moteur après reboot : 45 Ko. Aucun événement
disque côté Windows entre 10:20 et 10:58Z. Au reboot, le kmsg de 11:04:57Z montre
`EXT4-fs (sdg)/(sdf): recovery complete`, signe de journaux mal fermés.

**Comparaison avec l'extinction propre de 11:43Z** (redémarrage Maintenance, conteneurs
arrêtés d'abord à partir de 11:34:41Z) : dockerd sorti en 2,16 s, puis **`disks synced`
en 0,36 s**, VM prête aussitôt. **Cette comparaison ne tranche pas** entre deux causes :
- (a) des écritures encore en vol à 10:45, où le `sync()` démarrait 8 s après la
  sortie du dernier conteneur ;
- (b) un système de fichiers coincé, remis en état par le reboot Windows.
Pour trancher : `time sync` dans Ubuntu juste avant un futur stop.
**Leçon opératoire transmise à Maintenance** : sur ai-01, arrêter les conteneurs
écrivains **avant** `docker desktop stop`.

**Correction transmise à Maintenance** : les 96 GiB sont le plafond du palier
**disque**, pas de la RAM. Le palier RAM est fixé à 8 GiB. cgroup moteur mesuré à
11:3xZ : 16,3 Go (anon 6,5 + shmem 9,4 + cache ~0,1), pic 24,6 Go au boot, PSI 0,00,
events high/max/oom à 0 (vue conteneur ; la tranche parente n'est visible que de
l'hôte).

**Retour après le redémarrage de Maintenance** : StartedAt 11:45:10.56Z, **health 200
à 11:47:52Z** (boot 2 min 40, cache de compilation chaud). Génération réelle
« Quatre » en 0,48 s, KV 750 747, TouchOnHitFsTier rechargé, sidecars Up, `/dev/shm`
à 8,0 G avec 0 orphelin (purge inutile), palier fs intact (4 572 blocs, 113 G), VRAM
20 684 / 19 986 MiB. La sortie de `coursia-ci.slice` est à confirmer par Maintenance.

**Auto-correction** : j'avais présenté `MYIA_FS_TOUCH=0` comme un kill switch « sans
restart ». C'est **faux** : `_ENABLED` est lu une fois, à l'import du module, donc il
faut **recréer le conteneur**. Docstring de `lru_fs_tier.py` corrigée.

**Cron** : `64a58306` a survécu au reboot ; le doublon `9505c4a1` créé après le reboot
a été supprimé, il reste un seul cron de surveillance.

## Cycle 24/10 12:41Z — redémarrage par le plafond BUSY après 20 min de saturation, renvois client à ~400 s

**Ce qui s'est passé (watchdog, VÉRIFIÉ).** Trois épisodes depuis la génération moteur
de 11:45:10Z :
- **11:54:22→11:58:23Z** : premier `fail 1/2`, puis `WEDGE-BUSY 1/10` (p 46 226 → 254 381),
  puis `RECOVERED`. C'est le **premier BUSY→RECOVERED en vrai** de la v6 : rattrapage
  post-redémarrage sans relance du moteur.
- **12:06:59→12:09:00Z** : `fail 1/2` → `RECOVERED`.
- **12:10:55→12:30:59Z** : `fail 1/2` puis **10 cycles BUSY d'affilée**, compteurs en
  avance à chaque fois (p 8 371 844 → 10 248 447, soit +1,88 M tokens de prompt cache
  compris ; g 56 590 → 57 744, soit +1 154 tokens générés en 20 min, ~1 t/s). Au 10ᵉ :
  `RESTARTING (BUSY ceiling 10 reached ~5min — anti-drift fallback)`. Arrêt de l'API
  12:32:30Z, StartedAt 12:32:33.88Z, route prête 12:34:32Z, soit **~3 min 40 sans moteur**.

**Correction de durée — le plafond fait ~20 min, pas ~5.** En état WEDGE, un cycle dure
`GEN_TIMEOUT` 90 s + `INTERVAL` 30 s = 120 s ; 10 cycles = **~20 min** (mesuré :
12:12:55 → 12:30:59). Le « ~5 min » du message et de l'arbitrage Q27 vient d'un calcul sur
`INTERVAL` seul. Le plafond est donc 4× plus tolérant qu'annoncé ; le texte du message
reste à corriger au prochain recreate du watchdog.

**État moteur pendant la fenêtre (log par minute, 12:07→12:32Z)** : Running 5-7, Waiting
1-6, Deferred 2-6, `KV fetch: 1-2 waiting to start` presque en continu, KV en dents de
scie 67-95 %, génération ~1 t/s. Environ une grosse requête atteint son premier token
toutes les ~5 min (rafales `pp` 17-21 K t/s = ~200 K tokens comptés d'un coup). Hit
local 76 → 64 %, externe 43 → 24 %. Les préemptions de l'ancienne génération sont
perdues avec elle (compteur remis à 0 au boot).

**Lecture : surcharge par longueur de contexte, pas un blocage.** Les corps de 640 à
817 Ko pèsent ~180-230 K tokens (ratio mesuré le 24/10 : 3,53 o/token). Six requêtes de
cette taille en cours demandent ~1,2 M tokens de KV, contre **750 747** de capacité GPU.
Le moteur ne peut donc en tenir que 3 à 4 à la fois et charge/décharge le reste depuis
les paliers. Le débit de service (~1 grosse requête / 2-5 min) est inférieur au débit
d'arrivée : 49 corps > 100 Ko terminés entre 11:58 et 12:31Z, tous du hub (UA Bun,
x_forwarded_for = site).

**Renvois à l'identique à ~400 s (VÉRIFIÉ par empreinte du contenu).** Cinq corps
reviennent avec la même taille, le même début et la même fin :
- 816 600 o : 12:15:51, 12:22:42, 12:29:24 ;
- 747 393 o : 12:18:10, 12:25:03, 12:31:57 ;
- 641 539 o : 12:20:06, 12:26:50 ;
- 804 875 o : 12:20:53, 12:25:25 ;
- 725 781 o : 12:23:22, 12:30:09.
Le `ts` d'`error_sources.jsonl` est écrit à la **fin** de l'échange HTTP (après
`await self.app`, `error_source_capture.py:131-146`). Un intervalle entre deux fins
mesure donc la durée de la tentative suivante : **4 intervalles sur 5 font 402-414 s**.
Quelqu'un abandonne vers 400 s et renvoie la même requête, qui repart en queue. Ce n'est
pas l'`API_TIMEOUT_MS` du canon Claude Code (600 000 ms) ; l'émetteur du renvoi n'est pas
établi (hub ou client) → question posée à claudish.

**Abandon côté moteur.** Le middleware transmet `receive` tel quel (y compris
`http.disconnect`), il ne masque donc pas la déconnexion. Reste ouvert : uvicorn 0.54 /
starlette 1.8 détectent-ils la déconnexion **pendant le prefill**, avant le premier
octet ? L'instrument est `vllm:request_success_total{finished_reason="abort"}` : à 0
depuis 12:32 (aucune tentative n'a encore atteint 400 s). S'il reste à 0 alors que des
renvois réapparaissent, les tentatives abandonnées sont menées à terme pour rien.

**Ce qu'a fait le palier fs après le redémarrage de 12:32Z (VÉRIFIÉ, 10 min).**
- Tokens de prompt servis depuis le disque : 1 229 632 (56 %). Depuis le GPU :
  729 552 (33 %). Calculés : 230 699 (11 %).
- Palier fs : 28,2 Go lus contre 12,3 Go écrits.
- Chunks : 1 075 hits fs sur 2 296 requêtes.
Le palier tient son rôle : il survit au redémarrage et sert la majorité du rattrapage.

**Après le redémarrage, 12:32→12:48Z** :
- 28 requêtes terminées. Moyennes : file 47,6 s, prefill 14,8 s, bout en bout 124 s.
- 3 préemptions. KV entre 72 et 100 %. Running 3-4, Waiting 2-4.
- Sondes `OK` mais lentes (espacées de 64 à 80 s).
- Condensation : disjoncteur ouvert à 12:18-12:23Z, repli cloud réussi. C'est la
  dégradation prévue.

**Coût et bénéfice du plafond dans ce régime.** Il a sauté trois fois aujourd'hui
(02:09, 07:09, 12:30Z), à chaque fois avec des compteurs en avance. Coût mesuré :
- ~3 min 40 sans moteur ;
- caches GPU et RAM vidés : le hit local passe de 64 % à 25-30 % ;
- requêtes en vol coupées.
Bénéfice : aucun mesuré, car redémarrer ne réduit pas la charge qui arrive. Le choix
reste à l'utilisateur : registre Q29.

**Census 2 h (fenêtre de 95 min)** : 267 requêtes, ~2,8/min (contre ~6/min au cycle
précédent).
- Statuts : 252 × 200, 15 × 401 (clé VL morte, sondeur connu `108.165.33.236`). Aucun
  400 ni 5xx.
- Corps : 135 entre 100 Ko et 1 Mo. C'est la longueur des contextes qui sature, pas le
  nombre de requêtes.
- IP externes avec des 200 : site et web1 seulement, aucune nouvelle.

**Hôte** : boot Windows 11:01:38Z, commit 65,6 % (marge 136 Go), C: 603,8 Go libres,
`/dev/shm` 16 G (26 %), VRAM 21 059 / 20 370 MiB.

## 24/10 14:08Z — watchdog v6.1 : plus de redémarrage quand les compteurs avancent (Q29 = b)

**Arbitrage utilisateur, 24/10 ~14:05Z** : « OK pour b ». Un moteur dont les compteurs de
tokens avancent n'est plus redémarré, quelle que soit la durée de la surcharge. On
redémarre seulement quand les compteurs sont plats (vrai blocage) ou quand le moteur ne
répond plus.

**Ce qui a changé dans `medium-swift15-27b-fstier.yml`** (service watchdog uniquement) :
- Le bloc du plafond `BUSY_MAX` ne redémarre plus. À chaque multiple de `BUSY_MAX`
  cycles BUSY, il écrit une ligne `BUSY-LONG ALERT` (durée estimée = cycles × 2 min).
- Bandeau `v6.1`. Il indique la vraie durée d'un cycle en état WEDGE : ~2 min
  (GEN_TIMEOUT 90 s + INTERVAL 30 s). Le plafond de 10 cycles valait donc ~20 min, et non
  ~5 min comme on l'avait écrit dans l'arbitrage Q27.
- Inchangés : le redémarrage sur compteurs plats et les chemins ENGINE-HUNG / ENGINE-DOWN.

**Validation avant déploiement** :
- Script extrait du yml avec `$$`→`$`, puis `sh -n` : OK.
- Banc simulé (`.tmp/wd_sim.sh`, faux `curl`/`docker`/`sleep`) :
  - compteurs qui avancent : aucun appel docker, `BUSY-LONG` aux cycles 10 et 20 ;
  - compteurs plats : `docker restart` appelé.
- `docker compose config -q` : OK. Le dry-run ne recréait que le service watchdog.

**Déploiement** : `up -d --no-deps watchdog-swift15-27b` à 14:08:23Z. Le `StartedAt` du
moteur n'a pas bougé (12:32:33Z avant et après) et RC vaut 0. Première sonde à 14:08:27 :
`WARMUP decode=000`. La grâce de 120 s a été consommée sans être comptée comme un
blocage. C'est le comportement attendu sur un moteur saturé.

**État à 14:10Z** : 1 h 38 sans redémarrage, mais toujours en surcharge.
- Running 4, Waiting 7, KV 99,5 %.
- 58 préemptions depuis 12:32Z.
- Entre 12:32 et 14:08Z, 3 `fail 1/2`, tous suivis de `RECOVERED`.
- Le palier disque sert ~80 % des tokens de prompt depuis le redémarrage.

Le remède à la surcharge reste l'admission côté hub : il entre plus de contextes longs
que les 750 747 tokens de KV n'en tiennent.

**Deux mesures ouvertes, relayées à claudish** :
1. Un 400 à 13:29:42Z : 914 723 octets, `input_tokens=262145`, soit la fenêtre + 1. Il
   arrive après le recreate du hub (Gate 5b, 12:37Z), ce qui en fait un critère possible
   de #651 (plafond de contexte appris perdu au recreate).
2. Compteur d'abandons toujours à 0, alors que 4 corps ont été renvoyés à l'identique
   après 12:32Z. Deux hypothèses restent ouvertes : les originaux ont abouti, ou la
   déconnexion n'est pas détectée pendant le prefill.

### 24/10 14:13Z — le compteur d'abandons ne voit pas les déconnexions client

**Question** : quand un client se déconnecte, vLLM annule-t-il la requête, y compris si
elle attend encore une place dans le KV ? claudish voyait des « fantômes ». Les clients
renvoient le même corps toutes les ~287 s, alors que les entrées de fin côté moteur sont
espacées de 402 à 414 s, et `request_success_total{finished_reason="abort"}` reste à 0.

**Sonde** (`.tmp/abort_probe.sh`) : deux requêtes à prompt court (stream puis non-stream)
envoyées sur un moteur saturé (Waiting 6-7), avec un client qui abandonne au bout de 6 s.
- Le middleware enregistre la fin de chaque échange à 14:13:13Z et 14:13:19Z,
  c'est-à-dire **à la seconde même de la déconnexion**. Le handler HTTP rend la main
  immédiatement, même pour une requête encore en file.
- Le compteur `abort` reste à 0 après ces deux abandons.

**Source v0.32.0** :
- `async_llm.py:779-781` et `:1045-1047` : sur `CancelledError`, appel de
  `self.abort(request_id, internal=True)`.
- `output_processor.py:552` (`abort_requests`) : retire l'état de la requête et émet une
  sortie `FinishReason.ABORT`, sans passer par l'enregistrement des stats de fin. Ce
  chemin n'alimente donc pas `request_success_total`.

**Conclusion** :
- Le compteur `abort` est **aveugle aux déconnexions client**. Le « abort=0 » du cycle de
  12:41Z ne mesurait rien.
- vLLM annule bien à la déconnexion, y compris une requête qui attend une place dans le
  KV.
- Si des requêtes restent ~2 min dans le moteur après le départ du client, c'est que la
  connexion **hub → vLLM** est restée ouverte. La question passe donc côté hub : une
  déconnexion client y annule-t-elle la requête amont ?

## Cycle 24/10 14:41Z — crash moteur 14:32:09Z sur un bug amont du tiering, IP fleet inconnue

**Crash** : `AssertionError: Chunk … not found in cache` (`cpu/manager.py:221`
`_prepare_load`), appelé par `scheduler.py:1289` → `update_state_after_alloc` →
`tiering/manager.py:588 prepare_load` → tier primaire CPU. EngineCore mort
(`EngineDeadError`), 16 requêtes en 500, Docker relance le conteneur (RC 0→1). Ce n'est
pas le watchdog : il était au streak BUSY 8 avec des compteurs qui avançaient, et v6.1 ne
redémarre plus dans ce cas. Ce n'est pas autoheal non plus : RC a monté, et le label
autoheal=False est en place.

**Mécanisme, côté amont** (PR vllm#60729, défaut n°2 « éviction après lookup ») :
- `lookup()` déclare un bloc présent dans le tier RAM, mais ne l'épingle pas.
- Dans la même passe du scheduler, une promotion depuis le tier fs (déclenchée par le
  `lookup()` d'une autre requête) évince ce bloc pour faire de la place.
- `prepare_load()` affirme alors que le bloc est présent, et l'assertion tue le moteur.

Les conditions sont réunies ici : tier RAM primaire de 8 GiB toujours plein, beaucoup de
hits fs (~50 % des tokens de prompt passent par `external_kv_transfer`), saturation.
- Ce cas est rapporté par #60729 (deux fois en prod, tiering CPU + fs, v0.30.0) et par
  #54914 (v0.28/v0.29). Ces rapports utilisaient tous DSpark ou MTP3.
- La PR #62615 corrige le même ordre de lookup/épinglage/promotion, mais seulement pour le
  chemin P2P.
- Aucun correctif n'existe pour le chemin scheduler.
- `kv_load_failure_policy=recompute` ne couvre pas ce cas, car l'assertion part avant
  toute tâche de chargement.

**TouchOnHitFsTier n'est pas en cause** : son `lookup()` rend le résultat du parent sans
le modifier, et la trace passe par le tier primaire CPU. Les promotions, elles, sont le
comportement normal du tiering amont avec n'importe quel tier secondaire.

**Notre point de données, posté en amont** :
[vllm#60729 c.6385564912](https://github.com/vllm-project/vllm/pull/60729#issuecomment-6385564912).
C'est le premier rapport sans décodage spéculatif. Il écarte le défaut n°1 (couverture
MTP) et pointe vers le n°2.

**Coût de l'incident** :
- ~2,5 min sans moteur : 14:32:09Z → première génération réussie à 14:34:40Z (compile
  cache chaud, tier fs intact).
- Caches GPU et RAM vidés. Dans les 10 premières minutes, 90 % des tokens de prompt sont
  quand même servis depuis le cache : 41 % GPU, 49 % tiers externes.
- Surcharge revenue dès 14:38Z. Le watchdog est en BUSY et ne redémarre pas, conformément
  à Q29=b.
- Premier crash de cette signature en ~3 jours de profil fs (déployé le 21/10 19:15Z).
  `grep 'not found in cache'` sur le journal disponible ne trouve que ce crash.

**Leviers** (décision utilisateur, registre Q30) :
- (a) accepter et surveiller ;
- (b) tier RAM primaire 8 → 24 GiB : moins d'évictions dans le primaire, donc une
  probabilité plus faible, mais l'effet n'est pas mesuré ; coût ~16 Go de commit ;
- (c) retirer le tier fs, ce qui supprime le déclencheur mais aussi le palier qui sert la
  moitié du rattrapage.

Un patch local du gestionnaire CPU est écarté : ce serait un arbre de patchs, contraire à
la préférence stock/industriel.

**IP inconnue avec des 200 : `212.133.41.47`** (13:01, 13:07, 13:21Z).
- Trois requêtes `OpenAI/JS 4.104.0` avec la clé MEDIUM (empreinte census `be312daf7c1c`),
  via l'URL publique.
- Contenu : prompts de condensation RSM de nos propres dashboards (mentions ai-01 et
  po-2024).
- RDAP RIPE : `212.133.41.0/24` = DXC TECHNOLOGY, FR (enregistré via Level3).
Lecture : une instance RSM de la flotte qui sort par un réseau d'entreprise. Ce n'est pas
une intrusion, mais l'IP n'est réclamée par personne. Elle va au registre Q31 et une
demande de réclamation est postée sur global.

**Census 2 h** : 425 requêtes (~3,6/min).
- Statuts : 390 × 200, 18 × 401, 16 × 500 (tous à 14:32:09, le crash), 1 × 400 (13:29:42Z,
  fenêtre + 1, déjà relayé).
- Corps : 256 entre 100 Ko et 1 Mo.
- Sondes ZOLTRAAK (`116.202.50.2`) et sondeur `108.165.33.236` : 401 uniquement.

**Hôte et santé** :
- Commit 67,4 % (marge 129 Go). VRAM 20 934 / 19 990 MiB.
- `/dev/shm` purgé de 8,1 Go d'orphelins de la génération morte (17G → 8,0G).
- Condensation en repli cloud pendant la saturation (disjoncteur), comme prévu.
- nbaudit ok (vague 2 Image/). G: 85 %, écriture OK.

## Cycle 24/10 16:41Z — 2ᵉ crash du même bug amont, redémarrage à tort par le watchdog (corrigé, v6.2), 56 % de renvois identiques

**Deux redémarrages depuis le cycle de 14:41Z.**

1. **15:18:04Z : redémarrage déclenché à tort par le watchdog v6.1.** Il était en BUSY
   depuis 14 cycles (le dernier, 15:16:04, montrait `p 6167402->6528046`). Au cycle
   suivant, la lecture de `/metrics` a dépassé son `--max-time 5`. `read_counters` a alors
   imprimé `0 0` : awk termine sur des variables vides, que `%d` rend à 0. Ce `0 0`
   ressemblait à des compteurs plats, donc à un vrai blocage, d'où le `docker restart`
   d'un moteur qui travaillait. Les 11 requêtes en vol ont reçu un 500 (10 du hub + la
   sonde), et les caches GPU et RAM ont été vidés.
   C'est un trou dans la mise en œuvre de Q29=b : le repli « métriques illisibles →
   redémarrage » datait de la conception v6, avant b. À 16:47Z, `/metrics` répond en
   8-10 ms : la lenteur était passagère, liée à la charge.
2. **15:35:33Z : 2ᵉ crash `AssertionError: Chunk … not found in cache`**, avec une trace
   identique à 14:32:09Z (`scheduler.py:1289` → `offloading/scheduler.py:1256` →
   `tiering/manager.py:588` → `cpu/manager.py:221`). Juste avant :
   `Running 4, Waiting 7, Deferred 5, KV fetch: 1 waiting to start`.
   Docker a relancé le moteur (RC 0→1 ; le `docker restart` de 15:18 avait remis RC à 0).
   Container StartedAt 15:35:37Z, serveur prêt à 15:37:47Z, soit ~2 min 15 s sans service
   et 6 × 500 (5 du hub + la sonde).
   Le crash survient 15 min après le redémarrage de 15:18. Un cache vidé déclenche une
   ruée de promotions fs→RAM, exactement les conditions de la course. Le lien de cause
   est SUPPOSÉ, pas mesuré.

**Watchdog v6.2** (`medium-swift15-27b-fstier.yml`) :
- `read_counters` passe à `--max-time 15` et imprime `-1 -1` quand les deux compteurs ne
  peuvent pas être lus.
- Au 2ᵉ échec de décodage, une lecture illisible (maintenant, ou au snapshot du 1ᵉʳ
  échec) donne `WEDGE-UNKNOWN` : pas de redémarrage, le snapshot est repris dès qu'une
  lecture redevient valide.
- Redémarrage seulement après 3 cycles illisibles de suite (~6 min, décodage toujours en
  échec, `METRICS_KO_MAX=3`), ou sur compteurs plats comme avant.
- Le message de redémarrage dit désormais laquelle des deux causes l'a déclenché.

Validation :
- Script extrait du yml (`$$`→`$`), `sh -n` OK.
- Banc `.tmp/wd_sim2.sh` (faux `curl`/`docker`/`sleep`), 5 modes :

  | Mode | Comportement observé |
  |---|---|
  | compteurs qui avancent | BUSY, aucun appel docker |
  | compteurs plats | `docker restart` |
  | `/metrics` toujours KO | UNKNOWN 1/3, 2/3, restart au 3ᵉ |
  | KO une seule fois au 2/2 | UNKNOWN puis BUSY |
  | snapshot KO au 1/2 | UNKNOWN puis BUSY |

- `docker compose config -q` OK ; le dry-run ne recrée que le watchdog.

Déploiement : `up -d --no-deps watchdog-swift15-27b` à 16:44:48Z. StartedAt moteur
inchangé (15:35:37.876Z avant et après), RC=1 inchangé. Première sonde à 16:44:54 :
`OK decode=200`.

**Amont.** #62615 (« Pin batch lookup HITs before secondary-tier promotions ») écrit
« Only in P2P discovery mode ». Lu dans le code v0.32.0 du conteneur :
- `offloading/scheduler.py` (L687 parcours préfixe, L724 parcours fenêtre glissante)
  appelle `manager.lookup(key)` clé par clé ;
- un HIT secondaire passe par `_initiate_promotion()` → `primary_tier.prepare_write([key])`,
  qui peut évincer un bloc primaire non épinglé déjà compté HIT par le même parcours ;
- `prepare_load()` asserte ensuite sur ce bloc.

C'est la forme exacte de leur cas d'échec, sur le chemin scheduler. Point de données
posté : [vllm#62615 c.6386878540](https://github.com/vllm-project/vllm/pull/62615#issuecomment-6386878540).
Il demande d'appliquer le même ordre (`probe_batch` → épinglage → `promote_batch`) au
parcours du scheduler. `kv_load_failure_policy=recompute` ne couvre pas ce cas : l'assert
part du scheduler, avant tout chargement.

**Q30.** Le seuil « 2ᵉ crash en 7 jours » est franchi en 63 min : la recommandation passe
à (b), tier RAM 8 → 24 GiB, effet non mesuré. Un contexte de 200 K tokens fait ~4,6 GiB à
~23 Ko/token (mesure fs du 20/10, nvfp4). 8 GiB en tiennent moins de deux, 24 GiB environ
cinq. Décision utilisateur, rien d'appliqué.

**La saturation s'auto-entretient par les renvois.** Mesure depuis le boot 15:37:47Z, à
~16:50Z (72 min) :
- Hub (`Bun/1.3.14`) : **72 soumissions pour 32 corps distincts, soit 40 renvois
  identiques (56 %)**. Signature : taille + 200 derniers octets de tête et de queue.
- Copies par corps : 20 corps en 1 exemplaire, puis 4×2, 2×3, 1×4, 1×5, 2×6, 1×7, et
  **un corps soumis 10 fois**. Les 32 corps distincts font tous plus de 100 Ko.
- Écart entre copies (fin d'échange à fin d'échange) : médiane 198 s, p90 603 s, max
  681 s ; 10 copies à moins de 60 s de la précédente.
- Moteur : `request_success_total` donne 24 `stop` + 18 `length` (les sondes de 24 tokens
  finissent en `length`), tous clients confondus. Le census, lui, compte 72 + 11 + 43
  réponses 200.
- **Un 200 du census ne veut donc pas dire « servi »** : un flux SSE envoie son en-tête
  200 avant le premier token, et un abandon par déconnexion n'entre pas dans les
  compteurs de fin (PR #172).
- Mécanisme côté hub (réponse claudish 14:50Z, `proxy-server.ts:1778`, épinglé W16) :
  - une déconnexion client avant le premier octet est indétectable ;
  - le seul plafond en amont est le watchdog du premier événement, à 300 s ;
  - le client renvoie, et la requête renvoyée repart en fin de file.
- Débit moteur pendant ce temps : génération ~1 t/s la plupart des intervalles de 10 s,
  avec 4-7 requêtes en cours et 3-7 en attente. Les étapes sont dominées par des
  morceaux de prefill de 8 192 tokens, à ~1,5 K t/s sur le dense 27B. Pics de 40-113 t/s
  quand aucun prefill n'est en vol.
- Relayé à claudish pour l'issue d'admission en tokens.

**Census 2 h** : 240 requêtes (~2/min, contre 3,6/min au cycle précédent).
- Statuts : 219 × 200, 17 × 500 (11 à 15:18:32-34Z, redémarrage watchdog ; 6 à
  15:35:33-34Z, crash), 4 × 401 (`108.165.33.236`, clé VL morte), **0 × 400**.
- Clés : `be312daf7c1c` (MEDIUM) ×236, `1f7f6080cb65` (VL morte) ×4.
- IP en 7 h :
  - aucune IP nouvelle avec des 200 ;
  - `66.249.79.32` (plage Googlebot) : 1 requête, 0 × 200 ;
  - `212.133.41.47` : toujours 3 requêtes au total, aucune depuis 13:21Z (Q31).

**Q31.** 4 sièges se sont exclus par mesure de leur sortie WAN : po-2024, po-2025, po-2028
(`90.65.170.144`) et web1 (`37.187.180.135`, VPS OVH). Aucune réclamation.
po-2025 signale aussi, côté hub, des requêtes `swift-1.5-27b` avec l'UA
`open-code-review/v1.12.13` depuis `4.150.181.241` (plage Azure FR, peut-être un runner
CI). Invisible côté moteur (sortie hairpin du hub) ; c'est au census par clé de claudish
de le trancher.

**Cache.** Depuis le boot : hit préfixe GPU 17,7 %, externe 41,4 % (log moteur 16:48Z).
Préemptions : 35 depuis 15:37Z. KV 72-99 %.

**Hôte et santé** :
- Commit 68,7 % (marge 124 Go). VRAM 21 779 / 20 976 MiB, 1,2 GiB sous l'alerte.
- `/dev/shm` purgé : 8 orphelins des deux générations mortes, dont 2 × 8 GiB de mmap du
  tier (24G → 8,1G).
- Pruneur 16:20Z : 97,6 GiB pour un cap de 96, 67 fichiers supprimés, 0 erreur.
- Condensation en repli cloud de 15:21 à 15:31Z, pendant les redémarrages (prévu).
- nbaudit ok (vague 2 Image/). G: 85 %, écriture OK. Prouveur : aucune passe.

## 24/10 18:37Z — tier RAM 8 → 24 GiB (Q30=b, GO user) après 5 crashs du bug amont ; CORRECTION du cycle 16:41Z

**Correction : le redémarrage de 15:18Z était un crash, pas une erreur du watchdog.** Le log moteur
porte `AssertionError: Chunk … not found in cache` à **15:18:32Z** (EngineCore pid 108), avec la même
trace que les autres. Le watchdog horodate sa ligne **en début de cycle** (`TS=` avant une sonde de
décodage pouvant durer 90 s) : son « 15:18:04 RESTARTING » a été écrit après le retour de la sonde,
donc **après** le crash, sur un moteur déjà mort. Les compteurs illisibles venaient du moteur mort, et
les 11 × 500 de 15:18:32-34Z sont les victimes du crash. La v6.2 (PR #175) reste une protection
valable (une lecture ratée ne doit pas suffire à redémarrer), mais elle ne corrige pas un incident
réel de ce jour. **Leçon : une heure de watchdog date le début du cycle ; dater un redémarrage sur
le log moteur, jamais sur la ligne du watchdog.**

**Bilan du 24/10 : 5 crashs, même assert.** 14:32:09, 15:18:32, 15:35:33, 17:10:07 et 17:50:10Z.
- RC 1 → 3 sur les deux derniers (relances Docker). Le watchdog n'a rien redémarré : il était en
  BUSY-LONG à 17:34 et 18:26.
- 500 par crash : 16, 11, 6, 2, 11.
- **La même clé de bloc** (`b'\xc7\xa3\x04\t>\xe6…\x03'`) a fait planter le moteur à 15:35:33 **et**
  17:50:10 : le déclencheur se répète sur un même préfixe. Lien avec les renvois identiques du hub
  SUPPOSÉ, non mesuré.
- Côté utilisateurs : une utilisatrice externe (clé scopée, sans repli cloud) signale des 502 via
  `models.myia.io` cet après-midi. Les heures coïncident avec les crashs ; non vérifié côté hub.

**État amont (lu 18:40Z, REST) :** dernière release `v0.32.0` (18/10), celle que nous servons.
Aucun correctif publié ni fusionné sur main :
- #54914 (issue, même assert, DeepSeek-V4 sur H100, ouverte depuis septembre) ;
- #60729 (PR, diagnostic seulement : meilleur message d'erreur) ;
- #62615 (PR, la bonne forme : épingler les HIT avant les promotions ; couvre le mode P2P
  seulement) ;
- #50014 / #49952 (PR, promotions qui s'évincent entre elles ; voisin, pas le même mécanisme).
Toutes ouvertes.

**Changement appliqué** (`medium-swift15-27b-fstier.yml`) : défaut `KV_OFFLOAD_GB_FSTIER` 8 → 24.
- `.env` ne porte pas la variable ; `docker compose config -q` OK ; le dry-run ne recrée que le
  moteur.
- Annonce sur global + workspace-vllm avant le geste.
- Commit hôte avant : 68,0 % (marge 126 Go). `/dev/shm` avant : 17G/63G.
- Recreate `up -d --no-deps vllm-medium-swift15-27b` à 18:37:18Z, StartedAt 18:37:28Z.
- **Critère d'échec :** un nouveau crash du même assert après ce boot → recommander (c), retirer le
  tier fs, seule option qui supprime le mécanisme (plus de promotion fs → RAM).

## 24/10 18:49Z — image overlay `vllm-openai-v0320-tierpin:v1` : notre correctif du bug amont (GO user « toute la séquence »)

**Pourquoi une image dédiée.** Aucun correctif amont n'est publié ni fusionné (voir ci-dessus), et le
tier RAM plus grand ne fait que réduire la probabilité. Le user a demandé un correctif et une image
« le temps que ça lande dans main », puis donné son GO pour toute la séquence (correctif, tests,
image, redémarrage).

**Mécanisme lu dans le code v0.32.0** (`vllm/v1/kv_offload/tiering/manager.py`,
`cpu/manager.py`) :
- `TieringOffloadingManager.lookup()` rend `HIT` pour un bloc du tier RAM **sans l'épingler** :
  le scheduler ne l'épingle que plus tard, dans `update_state_after_alloc` → `prepare_load`.
- Un `lookup()` suivant, **dans le même pas de scheduler**, qui trouve un bloc seulement sur le tier
  fs lance une promotion : `_initiate_promotion` → `primary_tier.prepare_write([key])`.
- `CPUOffloadingManager.prepare_store` ne protège de l'éviction que `set(keys)` (la clé promue) :
  l'éviction LRU peut prendre le bloc déjà compté HIT, puisqu'il est encore à `ref_cnt == 0`.
- `prepare_load` trouve alors un bloc absent : `AssertionError: Chunk … not found in cache`
  (`cpu/manager.py:221`), et le moteur meurt.
- Chemins atteignables depuis le scheduler (pas seulement en P2P, que couvre la PR #62615) :
  la boucle de bord de queue partielle dans `_lookup` rend `complete_hit` même après des
  promotions ; la marche en fenêtre glissante continue après un MISS qui remet
  `pending_in_window` à zéro.

**Correctif (2 fichiers, `configs/docker/overlay_v0320_tierpin/PATCH.diff`).**
- Le gestionnaire de tiers retient les HIT RAM rendus pendant le pas courant (`_step_hit_keys`).
- Il les passe en `extra_protected` au `prepare_write` de la promotion. `CPUOffloadingManager`
  accepte ce paramètre et l'unit à la clé promue via une vue d'appartenance sans copie (les
  politiques d'éviction ne font que `key in protected`).
- L'ensemble est vidé à la fin de `on_schedule_end()` et dans `reset_cache()`.
- Si aucun bloc libre ne reste, `prepare_write` rend `None` : la promotion échoue, le bloc est
  compté MISS et sera recalculé. Plus de crash.

**Tests** (dans un conteneur jetable de l'image de base, `overlay_v0320_tierpin/tests/`) :
- sur le stock, `test_promotion_does_not_evict_a_hit_reported_this_step` lève **l'assert de prod
  mot pour mot** : le test reproduit le crash ;
- avec le correctif, les 3 tests de régression et les **113 tests amont v0.32.0**
  (`tiering/test_tiering_offloading.py`, `cpu/test_manager.py`) passent : **116/116** ;
- mêmes résultats sur l'image construite.

**Image.** `FROM vllm/vllm-openai@sha256:befccf0e…` (le digest de la release v0.32.0), deux
`COPY`, un `compileall` et une assertion de build qui vérifie la présence du correctif.
Image `vllm-openai-v0320-tierpin:v1`, `sha256:391801291d8c…`.

**Déploiement.**
- Profil `medium-swift15-27b-fstier.yml` : `image: ${VLLM_IMAGE_SWIFT15:-vllm-openai-v0320-tierpin:v1}`
  (`.env` ne porte pas la variable).
- Recreate `up -d --no-deps vllm-medium-swift15-27b` à 18:49:28Z, StartedAt 18:49:34Z, health 200
  à **18:52:50Z** (3 min 16, cache de compilation chaud).
- Gates : génération réelle « Quatre » ; `_step_hit_keys` présent ×5 dans le source chargé par le
  moteur ; KV **750 747** inchangé ; tier RAM 24 GiB, tier fs `TouchOnHitFsTier` chargé dans les
  deux process ; VRAM 19 902 / 18 938 MiB ; RC 0.
- `/dev/shm` : 48G (77 %, au-dessus du seuil 45G) → purge bornée au StartedAt, **1 fichier** (le
  mmap de 24 GiB de la génération 18:37Z) → **24G, 39 %**.

**Rollback** : `VLLM_IMAGE_SWIFT15=vllm/vllm-openai:v0.32.0 docker compose -f
myia_vllm/configs/docker/profiles/medium-swift15-27b-fstier.yml --env-file myia_vllm/.env up -d
--no-deps vllm-medium-swift15-27b`.

**Critère d'échec.** Un nouveau `Chunk … not found in cache` après 18:49:34Z signifie que le
correctif ne couvre pas tous les chemins : recommander alors (c), retirer le tier fs.

**Retrait de l'image** dès qu'une release amont porte un correctif équivalent (suivre #62615,
#60729, #54914). Le correctif est proposé en amont (commentaire avec le diff).

## 24/10 20:01Z — Q32 : `--no-scheduler-reserve-full-isl` retiré (GO user « Go », ~19:58Z)

**Constat.** Après le correctif tierpin, plus d'assert, mais le moteur étouffait sous charge :
**66 préemptions en ~66 min** (génération 18:49Z), KV GPU ~95 %, fenêtres de décodage sous
1 tok/s, timeouts de condensation. 5-6 conversations de ~200K tokens étaient admises en même
temps pour 750 747 tokens de KV, puis se préemptaient mutuellement (re-prefill en boucle).

**Cause.** Le profil portait `--no-scheduler-reserve-full-isl`. Le défaut vLLM (`True`) n'admet
une requête en attente que si **son prompt entier** tient dans le KV
(`allocate_slots(full_sequence_must_fit=…)`). Le scheduler est FCFS avec blocage en tête de file :
la première requête qui ne tient pas arrête l'admission du pas. Désactivé, il suffit que la
**1re tranche** (8192) tienne, et la régulation se fait par préemption. Le flag vient de l'import
1:1 de la config canonique Genesis le 16/05 (commit `e425dc66ca`, prévue pour 2 séquences
simultanées). Avec le KV de 1,2M et 2-7 % d'occupation de l'époque, le garde-fou ne jouait jamais.
Il n'a jamais été réévalué sur Swift.

**Ce qu'on attend du défaut.** Les requêtes en surnombre **attendent en file**, contexte au chaud
dans les tiers RAM/fs, au lieu d'entrer et de se préempter. Le débit par requête baisse, mais on
évite l'effondrement (arbitrage user : « tout le monde attende un peu et que la vitesse soit
diminuée disons de moitié … plutôt que divisée par cent »). Le levier suivant reste en réserve,
**non appliqué** (un changement à la fois) : `--watermark`, une fraction de blocs gardée libre à
l'admission (défaut 0.0).

**Déploiement.**
- Profil : ligne retirée + commentaire d'en-tête. `docker compose config -q` OK, 0 occurrence dans
  le rendu.
- Snapshot avant : `.tmp/pre_q32_metrics.txt` (génération 18:49Z, ~66 min). Préemptions 66,
  succès stop 128 / length 49, 0 abort, 0 error.
- Recreate `up -d --no-deps vllm-medium-swift15-27b` à 20:01:47Z, StartedAt 20:01:52Z, health 200
  à **20:04:54Z** (3 min, cache de compilation chaud).
- Gates : `scheduler_reserve_full_isl` absent des « non-default args » ; génération réelle
  « Quatre » ; KV **750 747** ; 0 `not found in cache` ; VRAM 20 488 / 19 562 MiB ; RC 0.
- `/dev/shm` 48G (77 %) → purge bornée au StartedAt, 7 fichiers, 24,63 GiB → **24G, 39 %**.

**Mesure à faire** (cycles suivants, à charge comparable) : préemptions ≈ 0, aucune fenêtre sous
1 tok/s, pas de hausse des 500, temps d'attente en file (`vllm:request_queue_time_seconds`).

**Rollback** : remettre la ligne `--no-scheduler-reserve-full-isl \` après `--async-scheduling \`,
puis `docker compose -f myia_vllm/configs/docker/profiles/medium-swift15-27b-fstier.yml --env-file
myia_vllm/.env up -d --no-deps vllm-medium-swift15-27b`.

## Cycle 24/10 20:41Z — premières mesures après Q32, trous de décodage résiduels expliqués

**État.** health 200 (3 ms), RC 0, StartedAt 20:01:52Z, OOM 0, **0 `not found in cache`** depuis
le boot overlay de 18:49Z (deux générations). VRAM 21 473 / 20 552 / 45 MiB, `/dev/shm` 24G (39 %),
3 conteneurs Up. Watchdog depuis 20:01Z : 3 × `fail 1/2` récupérés (20:10, 20:18, 20:26Z),
1 BUSY, **0 restart**. Marge commit hôte **70,5 %** (278,5/395,3 Go). G: écriture OK (85 % plein).
nbaudit `ok:true`, 0 échec consécutif (vague `19451:Image/`).

**Q32, premiers chiffres (43 min, recharge post-restart incluse, trop tôt pour conclure).**

| | avant Q32 (génération 18:49Z, ~66 min) | après Q32 (20:01Z → 20:44Z, 43 min) |
|---|---|---|
| préemptions | 66 | **2** |
| requêtes terminées | 177 (2,7/min) | 149 (3,5/min) |
| génération | 164 820 tok (~42 tok/s moy.) | 126 046 tok (~49 tok/s moy.) |
| erreurs / abandons | 0 / 0 | 0 / 0 |
| tokens de prompt servis par cache | — | **91 %** (GPU 2,55 M · RAM 12,73 M · disque 1,33 M sur 18,24 M) |
| attente en file / TTFT / e2e (moyennes) | — | 21,3 s / 36,0 s / 74,4 s |

Le trafic n'est pas le même d'une génération à l'autre : ces deux colonnes indiquent une tendance
et ne valent pas un A/B. Deux points ne font pas une pente.

**Trous de décodage résiduels : mécanisme mesuré.** Sur les 189 fenêtres actives de 10 s depuis
le boot, **38** ont produit moins de 1 tok/s au total avec 5-6 requêtes en cours. Le log affiche
alors P=0, le KV monte, puis une rafale de P suit. Cause : le prefill d'un long prompt **non
caché** (~200 K tokens). Une étape qui porte une tranche pleine de 8 192 tokens dure ~5 s sur le
27B dense (~1 600 tok/s). Chaque décodage en cours ne reçoit qu'un token par étape, soit ~0,2 tok/s
par flux. Il ne s'agit pas d'une sur-admission (Q32 l'a réglée). C'est l'entrelacement
prefill/décodage. Note de lecture : « Avg prompt throughput » ne crédite un prompt qu'à la **fin**
de son prefill. Il lit donc 0 pendant le prefill par tranches, puis une rafale.

**Levier identifié, non appliqué (registre Q35).** v0.32 `SchedulerConfig` (source vérifiée,
`config/scheduler.py:79-90`, `v1/core/sched/scheduler.py:636-660`, exposé en CLI par
`engine/arg_utils.py:600-605`) :
- `--long-prefill-token-threshold N` (défaut 0 = pas de plafond) : plafonne les tokens d'un long
  prefill par étape **quand plus d'une requête est éligible**. Une requête seule garde le budget
  entier.
- `--long-prefill-token-threshold-adaptive` : relève le plafond à
  `max_num_batched_tokens // nb_requêtes`. À 5 requêtes, ~1 600, ce qui retire l'essentiel de
  l'effet. Déconseillé ici.
- `max_num_active_seqs` (limite RUNNING sans réduire les graphes) et `max_num_queued_reqs` (503
  au-delà) existent aussi. Non pertinents pour ce symptôme.
- `throttle_prefills` / `prefill_capacity_bound` : équilibrage DP uniquement, sans objet à DP=1.

Ordre de grandeur **calculé, non mesuré**, à partir du prefill mesuré (~1 600 tok/s, ~25 ms de
fixe par étape) : avec N=512, l'étape passe à ~0,35 s, le décodage concurrent à ~2-3 tok/s par
flux, et le long prompt prend +5 à 10 %. Mesures à faire si le GO tombe : fenêtres < 1 tok/s,
décodage par flux pendant un long prefill, débit de prefill sur prompt long. Retour arrière si le
prefill perd plus de 15 %.

**Sécurité (7 h, `x_forwarded_for`).** 200 : site `90.65.170.144` (642), sans XFF (326, flotte
interne), web1 `37.187.180.135` (2), `92.184.141.186` (2, 18:15:09Z, OpenAI/JS 4.104.0, clé
MEDIUM, corps 20/54 K, même empreinte que `92.184.113.119` du 20/10 = pool mobile flotte,
condensation RSM, **non-alerte**). 401 : `108.165.33.236` Go-http-client ×30 (scanner, refusé).
500 : 46, toutes dans la fenêtre des crashs du bug amont (14:30-17:50Z), aucune depuis l'overlay.
Aucun 400 à corps > 100 K.

**Condensation.** Bannière 20:41Z `primary=qwen3.6-35b-a3b @ :5002 … key=OK`. Repli cloud réussi
à 15:22-15:31Z, pendant les crashs : dégradation gracieuse nominale. Les WARN du log sont des
`BaselineManager NonNominativeBaselineService`, périmètre RSM, pas vllm.

## 24/10 21:10-21:35Z — mandat user « config au mieux » : rétention du tier disque + KV +10,8 %

**Mandat** (user 21:10Z, en réponse à Q33/Q34/Q35) : « je te laisse faire tes changements de config au
mieux, tu mesureras le changement de débit global, l'objectif est de servir le mieux et le plus vite ».
Q35 (seuil long-prefill) écarté par le user : « 2 à 3 tok/s pour moi c'est pareil que 0.2 ».

### Diagnostic (mesuré avant tout geste)

- **Q32 a amélioré le débit global** : sur la même image, 67 min avant contre 71 min après, génération
  41 → **59 tok/s** (+45 %), requêtes 2,6 → **4,2/min**, préemptions 66 → 3. Télémétrie 1/min : minutes
  avec file 93 % → 52 %, minutes figées (gen < 5 avec des requêtes en cours) 20 % → 9 %. La gêne
  ressentie vient de l'**attente silencieuse** : file p90 ≤ 50 s, p99 ≤ 240 s ; côté hub (claudish,
  5 h), 203/327 requêtes Swift ont un TTFT > 30 s et 104 > 120 s, aucune > 300 s.
- **La file est une saturation KV, pas un blocage de tête** : sur les 40 minutes avec file, le KV est à
  80-100 % pendant 28. Il y avait bien un blocage de tête en FCFS (`scheduler.py` : `if new_blocks is None:
  break`, identique sur upstream main du 24/10), mais backfiller ne gagnerait presque rien. La capacité
  est d'environ 141 blocs sur 527 par conversation de 200K, soit ~3,7 conversations en même temps.
- **Origine des tokens de prompt** (20:04→21:16Z) : cache GPU 26 %, tier RAM 61 %, tier disque 5 %,
  calcul 7,5 % (~670 tok/s en moyenne). Les trous viennent de **7 prefills de 100-200K tokens calculés**
  en 71 min, ~95 s de moteur partagé chacun.
- **Le tier disque ne retenait que 1,6 h** : 97 G à son plafond de 96 GiB, fichier le plus ancien
  1,58 h, ~60 GiB/h d'écriture. Le pruneur supprimait ~25 GiB toutes les 30 min (21:20Z : 951 fichiers).
  Toute conversation reprise après plus de 1,6 h (lanes en cron 2-6 h) repayait donc tout son prefill.
- **Défaut de formule du pruneur** : `cap = min(max, free − réserve)` comparait la taille totale du tier
  à l'espace libre seul, sans compter les octets déjà tenus par le tier. Le tier plafonnait ainsi à la
  moitié de la place autorisée.

### Gestes

1. **Pruneur, sans coupure** : la formule devient `cap = min(max, tier + free − réserve)` pour l'invité
   et pour C: (le vhdx ne rétrécit pas, mais le tier réutilise les blocs ext4 qu'il libère). `MaxGiB`
   passe de 96 à **250**. La réserve de 12 % de C: (arbitrage Maintenance du 20/10) est inchangée.
   Dry-run : cap 250,0 GiB (C: 251, invité 331), tier 97,8 GiB, 0 suppression. Effet au passage de
   21:50Z ; rétention attendue ~4 h au débit d'écriture actuel (calcul, pas mesure).
2. **Restart 21:26:47Z** (annoncé sur le global à 21:26Z), health 200 à 21:30:09Z (3 min 22 s,
   cache de compilation chaud) :
   - `gpu-memory-utilization` 0,70 → **0,73** (`${GPU_UTIL_SWIFT15:-0.73}`, rollback = 0.70). KV
     7,41 GiB/GPU → **831 723 tokens** (+10,8 %, concurrence 3,17× à 262K contre 2,86×).
   - Q34 (a), partie hors empreinte de compilation : retrait de `NCCL_P2P_DISABLE`,
     `CUDA_DEVICE_MAX_CONNECTIONS`, `--disable-custom-all-reduce` (mesuré :
     `torch.cuda.can_device_access_peer(0,1)` = False, le P2P n'est jamais utilisé) et `shm_size`
     (ignoré avec `ipc: host`).
   - **Gardés bien qu'inertes** : `--gdn-prefill-backend triton`, `VLLM_USE_DEEP_GEMM=0`,
     `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`. Ils entrent dans l'empreinte du cache de compilation
     (`additional_config` / `envs.compile_factors`, alors que `disable_custom_all_reduce` et
     `gpu_memory_utilization` sont des facteurs ignorés). Les retirer coûtait une compilation à froid
     (~11 min de coupure le 19/10) pour un gain nul ; ils partiront à la prochaine montée d'image.
3. **Portes** : génération réelle 200 (« 391 », 0,3 s), watchdog `OK … engine is warm` à 21:30:22Z,
   0 `not found in cache` / OOM / Traceback, RC 0. VRAM juste après le boot GPU 0 21 478 / GPU 1
   20 724 MiB. `/dev/shm` 48G → 24G (purge bornée au StartedAt, 7 fichiers).

### Mesure « après » à faire (cycle suivant, même fenêtre horaire si possible)

Base « avant » : `.tmp/m_now.txt` (20:04:54→21:16:25Z) et `.tmp/pre_cfg_metrics.txt` (→ 21:26:15Z),
plus la télémétrie 1/min. Indicateurs, par levier :
- débit global : `generation_tokens_total`/s, requêtes servies/min, minutes figées, minutes avec file ;
- latence : file / TTFT p50-p90-p99, et table `[ttft]` du hub sur 22:00→23:30Z (demandée à claudish) ;
- tier disque : part `disk` de `prompt_tokens_cached_by_source`, nombre de requêtes à > 50K tokens
  calculés, âge du plus vieux fichier du tier ;
- KV : nombre de requêtes en cours quand il y a une file, préemptions, VRAM GPU 0 en charge
  (alerte > 23 000 MiB ; rollback `GPU_UTIL_SWIFT15=0.70` en cas de dépassement).

## 24/10 22:41Z — première mesure « après » : brouillée par le froid post-restart, verdict reporté

Fenêtres de même durée (~71 min) : AVANT = 20:04→21:16Z (post-Q32, gpu-util 0,70, tier disque
96 GiB) ; APRÈS = 21:30→22:41Z (0,73, 250 GiB). Compteurs `/metrics` des deux générations de
moteur (`.tmp/cmp_after.py`), télémétrie 1/min (`.tmp/telem_win.py`, `.tmp/telem_sat.py`).

| | AVANT | APRÈS | Écart |
|---|---|---|---|
| génération (tok/s) | 59,2 | 49,5 | −16 % |
| requêtes servies (/min) | 4,16 | 3,03 | −27 % |
| tokens de prompt calculés (tok/s) | 659 | 758 | +15 % |
| part du prompt calculée / RAM / GPU / disque | 7,5 / 60,9 / 26,5 / 5,2 % | 13,1 / 54,8 / 28,5 / 4,4 % | |
| tokens calculés par requête (moyenne) | ~9 500 | ~15 000 | +58 % |
| requêtes à 20-50K tokens calculés | 16 | 51 | |
| requêtes à > 100K tokens calculés | 7 | 6 | |
| file > 120 s | 3,0 % | 0,5 % | mieux |
| TTFT > 120 s | 8,0 % | 21,4 % | pire |
| préemptions | 3 | 3 | = |
| décodage par flux, minutes sans file | 19,8 tok/s | 24,4 (2ᵉ moitié) | pas de ralentissement moteur |

Lecture :
- **L'attente en file a baissé ; c'est le prefill qui s'est allongé.** TTFT moins file = temps de
  prefill, et les requêtes recalculent en moyenne 58 % de tokens de plus.
- **Le surcroît de prefill décroît dans l'heure qui suit le restart** (log moteur, prefill calculé
  par tranche de 10 min) : 21:3x 846 · 21:4x 737 · 21:5x 841 · 22:0x 921 · 22:1x 1 054 · 22:2x
  734 · 22:3x 491 · 22:4x 313 tok/s ; la génération remonte en même temps (22:3x 79, 22:4x 70
  tok/s, au-dessus de la moyenne AVANT). Télémétrie : minutes avec file 76 % sur 21:30→22:05,
  50 % sur 22:05→22:41 (AVANT 52 %), longueur moyenne de file 1,91 puis 0,86 (AVANT 1,19), minutes
  figées 13 % puis 6 % (AVANT 9 %).
- **Cause supposée, cohérente avec les mesures mais non prouvée** : le restart vide le cache GPU
  **et** le tier RAM (24 GiB, recréé à vide au boot). Seuls les blocs déjà descendus sur disque
  survivent, si bien que les conversations actives repaient leur fin de contexte (les 20-50K
  derniers tokens). **Un restart coûte donc environ une heure de prefill supplémentaire, pas
  seulement les 3-4 min de boot** : à peser dans toute décision de restart.
- Le décodage par flux n'est pas plus lent (24,4 contre 19,8 tok/s/flux hors file) : rien
  n'indique de pagination GPU 0 à 0,73. VRAM GPU 0 en charge 22 323 MiB (seuil 23 000).
- Le gain attendu de la rétention disque (conversations reprises après 1,6 h) n'est pas encore
  observable : plus vieux bloc du tier 2,9 h à 22:41Z (contre 1,6 h avant), tier 129 GiB, 0
  suppression au passage de 22:20Z (cap 250 GiB).

**Verdict reporté** au cycle suivant (fenêtre 22:41→00:41Z, sans effet de restart), avec la table
`[ttft]` du hub (claudish, 22:00→23:30Z, plus la tranche post-Q32 seule 20:05→21:25Z). Comme la
demande baisse la nuit, comparer d'abord les indicateurs normalisés (file, minutes figées, tokens
calculés par requête, décodage par flux) plutôt que les volumes.

Autres relevés du cycle : condensation RSM en repli cloud à 20:38Z et 21:48Z (timeout primaire
90 s pendant les pointes, repli nominal) ; `108.165.33.236` toujours 401-only (clé VL morte, suivi
au census) ; `92.184.141.43` = même clé et même client (`OpenAI/JS 4.104.0`) que po-2027 en
`92.184.141.186` → rotation d'IP mobile, pas d'alerte.

## 25/10 00:47Z — mesure « après » propre (22:41→00:47Z) : pas de gain de débit mesurable ce soir ; verdict définitif reporté à une fenêtre de jour à volume comparable

Fenêtre propre (sans effet de restart, 126 min) vs base AVANT (20:05→21:16Z, 72 min — qui
elle-même portait son propre coût de restart Q32, donc la comparaison est plus proche qu'il n'y
paraît). `.tmp/cmp_delta.py` (delta de snapshots de la même génération de moteur).

| | AVANT (0.70, fs 96) | APRÈS propre (0.73, fs 250) | J-1 même heure (23/10, ancienne config) |
|---|---|---|---|
| génération tok/s | 59,2 | 46,3 | 55,7 (télémétrie) |
| requêtes /min | 4,2 | 3,6 | — |
| file > 120 s | 3,0 % | **1,3 %** | — |
| TTFT > 120 s | 8,0 % | 8,8 % | — |
| part du prompt calculée | 7,5 % | 10,5 % | — |
| tokens calculés p90 / >100 K | 20 K / 2,3 % | 50 K / 3,8 % | — |
| préemptions (fenêtre) | 3 | 6 (9 depuis boot) | 66/h ce jour-là |
| flux concurrents (running) | 3,6 | 4,1 | 3,4 |
| KV moyen / quand file | 75 % / 81 % | 69 % / 85 % | 54 % / 80 % |
| décodage/flux hors file | 19,8 | 12,3 (total) / 17,3 (fin de nuit) | 24,0 |

Lecture :
- **Ce qu'on peut créditer au changement** : la queue d'admission s'améliore (queue > 120 s
  divisée par deux, p99 identique), le moteur tient plus de conversations simultanées (4,1 vs
  3,4-3,6 running ; KV occupé plus haut), et la rétention disque passe de 1,6 h à **5,0 h**
  (plus vieux bloc 21:46Z à 00:47Z ; tier 241,6/250 GiB, 0 suppression à 00:20Z, cap atteint au
  passage de 00:50Z → premières rotations attendues).
- **Ce qu'on ne peut pas créditer ce soir** : le débit global baisse (−22 %) mais la demande
  baisse aussi (−19 % de tokens de prompt) et le mélange nocturne est plus lourd en gros prefills
  frais (>100 K tokens calculés : 3,8 % des requêtes ; p90 des tokens calculés 50 K). C'est la
  nuit du maxing Minimax (cap tandem doublé), pas la même population que la veille. Le décodage
  par flux hors file (17,3 en fin de nuit vs 19,8 avant, 24,0 la veille) est dans le bruit d'un
  mélange à 5 flux, pas un ralentissement mesuré — rien n'indique de pagination GPU à 0,73
  (GPU 0 stable 22 309 MiB, 0 OOM).
- **Verdict définitif** : fenêtre de jour à volume comparable (demain, fenêtre à définir avec le
  hub). Engagement tenu côté instruments : table hub 22:45→00:30Z non encore livrée à 00:49Z
  (relancée), moteur côté rendu ici.
- **0 assert `not found in cache` depuis l'overlay (18:52Z)** → condition de la PR amont
  vllm-project tenue à ce stade ; ouverture prévue au cycle du matin (le 25/10 « au matin »,
  ~06:41Z), sauf contre-ordre.

## 25/10 03:40Z — confirmation croisée par le hub : le changement de config ne dégrade pas, mais n'améliore pas le TTFT vu par les clients

Table `[ttft]` du hub (po-2025:claudish, message du 24/10 23:36Z), même instrument, `swift-1.5-27b` :

| Tranche | n | >30 s | >120 s | >300 s | moyenne | max |
|---|---|---|---|---|---|---|
| post-Q32 20:05→21:25Z | 231 | 33,8 % | 6,5 % | 0 | 36,6 s | 275,8 s |
| restart + dette de prefill 22:00→22:37Z | 51 | 80,4 % | 29,4 % | 0 | 83,5 s | 220,5 s |
| **stabilisée 22:45→00:30Z** | **137** | **40,9 %** | **6,6 %** | **0** | **39,2 s** | **262,2 s** |

Les deux instruments concordent :
1. **La dette de prefill du restart est bien un artefact d'une heure** (hub : 29,4 % → 6,6 % de
   requêtes > 120 s ; moteur : prefill calculé 1 054 → 313 tok/s sur la même période).
2. **À l'état stabilisé, le TTFT vu par les clients est identique à la base post-Q32** (>120 s :
   6,5 → 6,6 % ; moyenne 36,6 → 39,2 s ; >30 s 33,8 → 40,9 %). Donc **ni gain ni perte mesurés**
   pour gpu-util 0,73 + tier disque 250 GiB sur cet indicateur, ce que confirme le débit moteur
   (−22 % avec −19 % de demande, nuit de maxing).
3. `>300 s` reste **0 sur trois tranches cumulées (n=419)** : la coupure du hub n'a jamais été
   approchée.
4. **Ce qui reste structurel** : ~40 % des requêtes client dépassent 30 s avant le premier token à
   l'état stabilisé. C'est la file native + la saturation KV sur des conversations longues +
   le prefill lent du 27B dense, pas un effet de la config de ce soir. Le levier restant est du
   niveau **politique d'admission / priorité** (décision user), pas du niveau réglage moteur.

**État des leviers de capacité (ce qu'on garde)** : KV 831 723 (+10,8 %) ; tier disque à 250 GiB,
**première rotation réelle à 03:20Z (887 fichiers supprimés, 0 erreur)**, plus vieux bloc 6,3 h
(les blocs re-servis voient leur mtime rafraîchi, donc la borne basse n'est pas l'âge de rétention) ;
9 préemptions depuis le boot, **stables depuis 00:47Z** (l'ancienne config en faisait 66/h) ;
hit rate du cache de préfixe 41,9 % cumulé (139,3 M requêtes / 58,4 M hits).

**Fenêtre de jour** : proposée à claudish le lundi 26/10 09:00→11:00Z (jour ouvré, volume
comparable), à confirmer quand les volumes du week-end seront connus.

## 25/10 05:41Z — engagement tenu : PR amont ouverte (vllm-project#63551), 0 assert après 11 h

- Condition remplie : **aucun assert `not found in cache` depuis le déploiement de l'overlay**
  (18:52Z 24/10 → 05:41Z 25/10 = 11 h), contre 5 crashs/jour sur le stock.
- Branche `kv-offload-protect-step-hits` = upstream/main (e4a5b55373) + le correctif (applique
  propre, il date d'hier et upstream n'a pas touché ces fichiers depuis) + le test de régression
  déplacé sous `tests/v1/kv_offload/tiering/` (layout amont) + `ruff format` (1 fichier
  reformaté). Commit `b55d41fbf3`, DCO signé.
- **PR ouverte : [vllm-project/vllm#63551](https://github.com/vllm-project/vllm/pull/63551)** —
  titre `[Bugfix][KV Offload] Protect the step's reported HITs from promotion eviction`. Cross-ref
  postée sur #62615 (c.6394054527). Fixes #60729 (case 2), related #54914.
- Prod au même cycle : health 200, RC=0, VRAM GPU 0 22 329 MiB, préemptions 14 (+5 dans la nuit,
  à suivre), tier fs 268,4 GiB avec rotations normales (751 fichiers à 05:20Z, 0 err).

## 25/10 07:41Z — litige routage clos (concession claudish) ; PAYG chiffré ; conditions (a)-(d) répondues ; deriveContextLimit corrigé

**Litige (escalade user 05:55Z + 06:52Z).** ai-01:claudish a rétracté son « jamais servi de
lanes haiku » (msg-20261025T073124, 08:31Z) après appariement req→resp de toutes les réponses
swift du jour : 815/876 (93 %) répondent à des requêtes haiku (po-2028 479, po-2023 186,
po-2024 124, ai-01 22) ; pilote né le 20/10 (0 resp/j les 17-19/10 → 1 234 le 20/10), jamais
interrompu. Rectificatif publié sur global par claudish. Deux narratives internes rectifiées
et consignées chez eux.

**Ce qui brûle en PAYG = la taille, pas le filet.** Frontière nette mesurée par claudish :
swift sert jusqu'à 937 Ko de corps, ds-flash commence à 970 Ko. À ~4 o/jeton : prompt +
max_tokens > 262 144 → 400 moteur → saut #431 → terminator PAYG. Cause d'explosion des
volumes : MiniMax M3.1 **et** le plan Qwen (ds-v4.1) morts tous deux le 23/10 (0 resp/j
chacun) — ds-flash est le seul terminator restant pour TOUTES les cascades.
**Chiffre PAYG nuit 25/10 00→08Z** (2 498 fichiers ds-flash) : 2,47 M tok out + 5,08 M in +
433,5 M cache-read ≈ **~2,5 €** (ancre liste ~1 €/M out, ancre console à faire — claudish ne
publie pas d'€ ferme sans elle). Remis au user ce cycle.

**Ma nuit sur PAYG, cause racine rectifiée.** Mes cycles 00:47Z/05:41Z ne sont pas tombés sur
ds par ordre de cascade : leurs corps (>970 Ko) dépassaient la fenêtre moteur → 400 → PAYG.
L'option B (swift en 4ᵉ marche de la cascade sonnet, qui aurait couvert mes cycles) a été
soumise puis **retirée** par ai-01 (arbitrage user) → registre Q36.

**Correction technique déposée (contre « la fenêtre réelle 3× plus large »)** : le KV 831 723
est la capacité **agrégée** du cache, pas la fenêtre par requête. Limite par requête =
`max_model_len` = 262 144 (prompt + max_tokens), contexte natif du modèle — pas élargissable
(YaRN 1M : qualité non testée + une requête 500K = 60 % du KV). `deriveContextLimit` (#660
claudish) dérivé du moteur obtient exactement 262 144 : il aligne le hub sur la vraie
déclaration mais **ne fait pas rentrer les gros corps**. Leviers réels : cap `max_tokens` hub
sur le step swift (8-16K → prompt admissible ~246-254K, rattrape la frontière 937/970 Ko) ou
fenêtre de condensation ai-01 330K→~220K. Choix = arbitrage user → **registre Q36** (a/b/c,
reco (a)).

**Conditions (a)-(d) du filet haiku continu — répondues à po-2025:claudish avec mesures**
(msg-20261025T074625) : (a) admission = prompt ENTIER réservé (Q32), prompt moyen mesuré
107 369 tok/req (13 % du KV), grosse session ~240K = 29 % ; (b) prefix cache ON, 91,8 % des
tokens servis du cache depuis boot (49,6 % GPU + 39,4 % RAM + 2,8 % fs) ; (c) budget #654
600K Σ + BIG_LIMIT 3 = **OK premier armement** (pire cas 3×240K = 720K > budget → bride avant
le KV, ~230K de tête pour croissance + consommateurs directs ; garde-fou = préemptions,
16/34,5 h actuellement, resserrer si retour vers la signature 66/h) ; (d) trafic continu
confirmé (mandat user 25/10). Armement reste chez claudish (annonce global d'abord).

**Sécurité cycle** : IP `92.184.141.244` = 2 req 200 /7 h (famille mobile Free, non réclamée —
probable rotation mobile type po-2027 `92.184.141.186`/`92.184.103.95`) → signalée au user,
surveillance. Le bucket `?` (649 req/7 h, 200) = appels du hub via la gateway Docker sans
XFF : interne, nominal.

**État moteur** : health 200, StartedAt 21:26:46Z inchangé, 0 assert (18 h), préemptions 16,
VRAM GPU 0 **22 725 MiB** (ligne d'alerte 23 000, à surveiller — rollback GPU_UTIL 0.70 si
franchie), GPU 1 21 398, /dev/shm 25 G sous le seuil 45 G, 2 223 req/34,5 h depuis boot.

## 25/10 08:41Z — restart 09:02:45Z : crash NCCL TP1 (nouveau mode), watchdog correct, /dev/shm purgé

**Événement.** Le moteur a redémarré à 09:02:45Z sur décision du watchdog — **vrai wedge** :
`WEDGE: API up but decode stuck, counters flat` (p=270 451 159, g=2 401 688 identiques à 09:00:45
et 09:02:45). ExitCode 0 (docker restart), nouveau boot 09:04:20Z, sain depuis (sondes OK
continues, 0 assert). Intervention 2 min 20 s après le figage des compteurs — comportement
nominal de v6.1/v6.2.

**Cause racine (log moteur).** À 09:00:28, `Worker_TP1` meurt sur
`RuntimeError: NCCL error: unhandled cuda error` dans `all_gather` — pile complète :
`sample_tokens` → `compute_logits` (qwen3_5.py:430) → `_gather_logits` →
`tensor_model_parallel_all_gather` → pynccl. « unhandled cuda error » au collectif = erreur
CUDA **asynchrone** qui remonte au premier appel synchronisant — le fautif réel est en amont
dans le stream (kernel de quant/attn non identifié). Moteur mort-vivant ensuite (health 200,
génération figée, `shm_broadcast … 60 seconds` ×3) jusqu'au restart.

**Contexte de charge.** Matinée très chargée : ~48 M tokens de prompt traités entre 06:48 et
09:00 (~5 900 tok/s), préfixe hit 72,9 % / externe 80,1 %, KV 40,7 %, 4 req en vol au crash.
Épisodes `WEDGE fail 1/2` répétés 06:48→08:17 (chaque fois RECOVERED) = pattern prefill
documenté, pas des précurseurs du crash. **Première occurrence** de ce mode : 0 NCCL error
dans les 11,5 h de runtime précédentes (depuis le boot 21:26Z 24/10). Coïncidence d'horaire
avec la cascade de stalls du 20/10 09:02Z notée — deux points, pas une pente.

**Suites.** (1) `/dev/shm` était repassé à 49G/63G (orphelins de la génération tuée) → purge
bornée au StartedAt : 7 fichiers, retour **24G = 39 %** (plancher = tier RAM). (2) Claude signé
pour exclure 09:00-09:10Z de ses tables (comme le wedge hub 00-01Z). (3) Impact standard du
restart : ~1 h de re-prefill à froid + les 4 requêtes en vol perdues. (4) Aucun geste moteur
sans GO ; si le mode récidive → chercher l'erreur CUDA amont avec `NCCL_DEBUG=INFO` (une
seule occurrence ne justifie pas un restart-config).

**Watch.** Post-restart : VRAM 22 183/21 136 à froid (0.73), préemptions remises à 0,
nbaudit ok (vague 2 Image/ active), commit hôte 78,1 % (marge 86,5 GB), G: OK, condensation
nominale (re-credentials 05:44 = post-wedge hub).

## 25/10 10:41Z — post-restart sain sous charge diurne forte ; census +212.133.41.33 ; D: nettoyé (Maintenance)

Routine. Moteur (boot 09:04:20Z) sain sous **charge diurne très forte** : 78,4 M tokens de
prompt en 2,2 h (~9 900 tok/s), hit préfixe **86,3 %** (l'amortissement post-restart refait
son socle — mieux que le 72,9 % cumulé d'avant-crash), préemptions 1, running 3, 0 assert,
un seul `WEDGE fail 1/2` (11:16Z, récupéré). VRAM sous charge à 0.73 : GPU 0 **22 417** /
GPU 1 21 408 — marge 583 MiB avant la ligne 23 000, à suivre. GPU 2 = training CoursIA
(15 Go, nominal). Commit hôte 83,8 % (marge 63,9 GB) — en montée matinale documentée, seuil 85 %.
`/dev/shm` stable 24G. nbaudit ok (vague 2 Image/, 11:06Z).

**Census** : `212.133.41.33` (3 req, 2×200) = même profil exact que `212.133.41.47` Q31
(UA `OpenAI/JS 4.104.0`, corps 15-54 Ko = condensation RSM) → IP d'égress corporate du même
acteur, ajoutée à la carte famille 212.133.41.x. Un 400 isolé au milieu (corps 15 848,
model=None) sans suite.

**Hors cycle** : règle user D: « transitoire seulement » appliquée par Maintenance —
`D:\vllm_image_backups` (25,45 Go, 3 tars) supprimé, copies sha256-vérifiées consolidées
sous `F:\Backups-Cloud\vllm_image_backups\` (empreintes mesurées des deux côtés avant geste ;
journal campagne Maintenance #165). Aucun tar n'était porteur prod.

## 25/10 13:41Z — routine diurne saine ; signature #431 mesurée pour la première fois ; clé dominante identifiée ; `/tokenize` qualifié pour #674

**Prod** : health 200 en 7 ms, `RC=0`, `StartedAt=09:04:20Z` (restart NCCL du matin, PR #190),
3 conteneurs Up, 0 OOM, `/dev/shm` **24 G/63 G (39 %)**, GPU 0 **22 648** / GPU 1 **21 408** /
GPU 2 15 011 MiB (CoursIA, hors périmètre). Watchdog : 1 restart (09:02:45Z, déjà documenté),
puis **4 `fail 1/2` sans suite** (11:16:18, 12:40:54, 13:17:28, 13:24:21Z), 1 514 sondes OK.

**Moteur sur 4 h 34** (compteurs créés 09:07:25Z) : prompt **136,1 M tok** dont
`local_compute` **4,7 M (3,5 %)**, device 109,1 M, external_kv_transfer 22,3 M, host 19,2 M,
disk 3,1 M → **hit préfixe 80,2 %** ; génération 1,266 M tok (77 tok/s moyens) ;
**1 300 requêtes** (stop 821 + length 479, 0 error/abort) ; **préemptions 2**.
Latences : TTFT p50 2,5 / p90 20 / p99 80 s ; **file p50 0,3 / p90 0,8 / p99 20 s** ;
inter-token p50 25 ms / p90 50 ms ; décodage p50 10 / p90 120 / p99 480 s.

**★ Première occurrence mesurée de la signature #431** — 2 × `400` corps **901 417 B** à
**08:14:58Z et 08:15:08Z**, IP `90.65.170.144` (hairpin site), UA `Bun/1.3.14` (hub),
`model=swift-1.5-27b`, `error_body` : « maximum context length is 262144 … prompt contains at
least 262145 input tokens ». **Deux tentatives à 10 s d'intervalle = le retry local que la règle
#431 interdit**, puis repli. Les deux sont **antérieures à l'arbitrage user de 12:25Z** : elles
datent l'état d'avant le correctif que claudish ouvre. Aucune récidive depuis 12:25Z.

**★ Clé dominante identifiée** — le middleware stocke `authorization[:24]+"..."`, soit
`"Bearer "` + 17 caractères : sous ce schéma `sha256[:12]` = **`be312daf7c1c` EST
`VLLM_API_KEY_MEDIUM`** (vérifié sur `myia_vllm/.env`, aucune valeur imprimée). Cela ferme la
question ouverte de po-2025:claudish (« absente des 14 slots du hub ») : le hub porte une
**référence** `${…}`, pas la valeur — c'est le piège d'instrument qu'ils avaient eux-mêmes
documenté le 08/10. Part de trafic : **1 639 / 1 703 requêtes en 6 h = 96 %**.

**Sécurité** — `ZOLTRAAK/0.2.0 (security-research; chesstournamentlift@gmail.com)` depuis
`116.202.50.2` (Hetzner) : **3 × `GET /v1/models` → 401** (08:44:42-08:45:04Z) ; `129.80.213.237`
(Mozilla) 1 × 401 ; `108.165.33.236` (Go-http-client) **59 × 401 depuis 07:48Z** avec
`fp=1f7f6080cb65`, clé absente de notre `.env` **et** du hub. **Aucun accès accordé nulle part.**
`82.66.89.184` (poste Paris user, Q13) réapparu 12:37→13:29Z, 6 req 200 ; `92.184.141.244`
(2 req) — les deux en condensation RSM (UA `OpenAI/JS`, même clé).

**★ `/tokenize` qualifié pour #674** (question bloquante de claudish) : `POST /tokenize` →
**200 en 0,059 s** (1 011 tok), **0,037 s** (10 011 tok), **0,861 s** (**292 231 tok**, corps
1,3 Mo) **alors que le moteur servait du trafic** ; rend `count`, `tokens` et
**`max_model_len=262 144`** — donc le champ de vérité de `deriveContextLimit` est lisible côté hub
sans moteur. Il **compte au-delà de la fenêtre sans erreur** (292 231 > 262 144 accepté) : la
décision « skip context » peut se prendre sur le compte exact. Pas de réservation KV (API-server).

**Signal condensation po-2025 (4/4 repli cloud sur le statut)** — **moteur disculpé par les
chiffres** : file p90 **0,8 s**, TTFT p90 **20 s**. Le discriminant est le **budget de génération** :
statut 15,4 Ko ≈ **4,1 K tokens de sortie** à 25-50 ms/token = **100-200 s** > timeout client 90 s ;
résumé 2-4 Ko ≈ 0,5-1 K tokens = 12-50 s → passe. Correctif **côté client RSM** (roo-extensions) ;
aucun levier moteur, le moteur tronque à la fenêtre (correction claudish 09:42Z).

**Middleware** : journalise désormais `x-claudish-client` / `x-claudish-machine` (format figé par
po-2025:claudish) — **inerte jusqu'au prochain restart**, aucun geste moteur pour l'activer.

**Hôte** : commit **84,4 %** (marge 61,5 Go, seuil 85 %, aucun travail lourd) ; G: écriture OK
(84 %, 628 Go libres) ; nbaudit `ok:true` 0 échec (vague 2 `Image/`). **Q36 = (c)**, option (a)
« cap max_tokens » **réfutée par la mesure** (claudish : prompt > fenêtre → 400, sinon la
génération est rabotée) ; les deux règles Q36 sont prises par claudish (implémentation après
#679/#675, recreate unique demain après la fenêtre 09:00-11:00Z).

## 25/10 15:41Z — routine saine ; amortissement porté par les tiers (7,8 % calculés) ; 0 récidive #431 ; breaker condensation re-trippé

**Prod** : health 200 en 6,4 ms, `RC=0`, `StartedAt` inchangé (09:04:20Z), 0 OOM, `/dev/shm` 39 %
stable, 3 conteneurs Up. VRAM GPU 0 **22 613** / GPU 1 **21 408** MiB (sous l'alerte 23 000) ;
GPU 2 24 057 MiB = training CoursIA à plein (nominal, hors périmètre). Watchdog : que des
`fail 1/2` (4 nouveaux depuis 13:52Z), compteurs avancants, **0 restart**, 1 675 sondes OK.

**Delta 2 h** (même génération, 13:41→15:41Z) : prompt **+45,84 M tok** dont
`local_compute` **+3,56 M = 7,8 % seulement** — le reste servi par les caches :
device 36,5 %, **host tier 54,0 %** (24,8 M rejoués), external KV transfer 56,1 % (25,7 M),
disk 2,1 %. **L'amortissement tient l'après-midi** : les 246 corps >100 Ko des sessions
Claude-Code (539-663 Ko, tailles toutes différentes = sessions distinctes, pas un re-prefill
en boucle) sont absorbés par les tiers. Génération **+634,6 K tok (≈88 tok/s)**, +490 requêtes
(4,1/min), **préemptions +8 (2→10)** — pression KV de pointe modérée, aucun wedge 2/2.

**#431** : **0 récidive** de la signature `400 + corps > 100 Ko` depuis 08:15:08Z.

**Condensation** : le circuit breaker s'est **re-ouvert à 14:56:51Z** (primaire essayé →
timeout 90 s → repli cloud ×3) — 5ᵉ occurrence du jour. Diagnostic + addendum déjà chez
roo-extensions (timeout statut 4,1 K tok + breaker sans half-open), pas de réponse à 15:41Z.

**Sécurité** : 200 uniquement depuis des IP connues (hairpin 321, interne 181, poste Paris
`82.66.89.184` 2, web2 `51.75.200.22` 2). `koara.live` persiste : **20 × 401 sur 2 h** (~10/h,
clé morte, toujours aucun accès).

**Hôte** : commit **89,3 %** (353,2/395,3 Go, marge 42,1 Go) — **à 0,7 point de la barre CRIT
90 %**, montée diurne vmmemWSL documentée ; aucun travail lourd lancé ce cycle (règle §3).
G: écriture OK ; nbaudit `ok:true` (vague 2 Image/, 0 échec).

## 25/10 16:50Z — capacité : ce que font VRAIMENT les tiers à l'éviction + état #4334

**Question user : « le tier fs ne prend pas la main pour ces évictions ? »** — **SI**, et voici les
chiffres depuis le boot (09:07→16:45Z, même génération) :

| flux | volume | temps | débit |
|---|---|---|---|
| GPU→CPU (store) | **352 GiB** | 162 s | **2,2 GB/s** |
| CPU→GPU (load) | **1 099 GiB** | 299 s | **3,7 GB/s** |
| RAM tier → fs (write) | **352 GiB** | — | — |
| fs → RAM (read) | **29,9 GiB** | — | — |

Replays : **51 089 blocs** rechargés depuis le tier RAM, **1 222 blocs depuis le fs**
(1 222 × 26 247 168 B = 32,07 GB = exactement `tiering_read_bytes`, cohérence vérifiée).
Taux de hit par *lookup* : RAM **15,5 %** (51 089/330 431), fs **6,1 %** (1 222/20 109) — le
complément = des préfixes jamais vus (miss attendu), pas une perte.

**Correction de ma formulation** (« chaque éviction = un re-prefill ») : **faux**, l'éviction est
**capturée**. Ce qui coûte un re-prefill c'est un préfixe **plus vieux que la rétention** du tier fs.
**Pruneur à 16:50:04Z** : tier à **299 GB pour un cap de 250 GiB** → 1 166 fichiers / **30,6 GB
supprimés** en 2,03 s, **0 erreur**, 11 393 fichiers. À ~44,6 GiB/h écrits (352 GiB/7,9 h) la
rétention réelle est **≈ 5,5-6 h** — au-delà, le préfixe est vraiment perdu.
**Asymétrie à noter** : on écrit **12×** plus sur le fs qu'on n'en relit (352 vs 29,9 GiB) → le tier
RAM est le vrai cheval de trait, le fs est la longue traîne (~85 MB/s d'écriture continue, sans erreur).

**Question user : « du neuf sur les préfixes qui forkent à l'octet 143 ? »** — **oui, et le chiffre
bouge** : le worker po-2024 a localisé la cause dans `zoo-code/src/core/prompts/system.ts` — le
`roleDefinition` (phrase d'identité du mode) est l'**élément n°1** du system prompt, donc les 10 modes
de la flotte ne partageaient que **18 octets** (pas 143 : 143 était le point de divergence observé
côté Claude Code, 18 est la mesure sur le prompt de mode). **PR #4354 MERGÉE aujourd'hui 09:30Z** :
`roleDefinition` unique de 568 o pour la flotte, identité du mode déplacée *verbatim* dans
`customInstructions` → **LCP 18 B → 568 B**, la divergence passant de la tête à la queue (tout ce qui
est entre — outils, capacités, règles, instructions globales — entre dans le préfixe partagé).
**Deuxième point de fork trouvé indépendamment** : `sections/skills.ts` filtre les skills **par mode**,
en amont des règles → re-forke tôt pour des lanes aux jeux de skills différents.
**⚠️ PAS DÉPLOYÉ** : `--deploy-global` réécrit le `custom_modes.yaml` de **chaque** lane → décision
owner/coordinateur ; tant qu'il n'est pas passé, **#4334 n'a aucun effet en production**.
`claudish#640` (mesure côté hub) : **0 commentaire, aucune activité depuis sa création il y a 32 h**.
