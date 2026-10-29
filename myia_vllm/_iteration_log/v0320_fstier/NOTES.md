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

## 25/10 ~17:30Z — what the fs tier actually writes, and where prefixes fork

Triggered by the user's challenge: "the only way to justify that many file writes is that the
whole conversation's KV is saved at each assistant message".

**Correction first.** One fs block file is 26,247,168 B = 1,424 tokens x **18,432 B/token for
both TP ranks together** (16 attention layers x K+V x 4 KV heads x 256 x 0.5625 B in NVFP4).
The 36,864 B/token used in the 16:50Z section double-counted the ranks. Consequences:
- RAM tier 24 GiB = **981 blocks** vs GPU KV 7.41 GiB/rank x 2 = **606 blocks**: the RAM tier
  is already **1.62x** the VRAM KV in bytes. "RAM tier smaller than VRAM" (said at 16:50Z) was false.
- "store bytes ~= (computed + generated) tokens x 36,864" was two errors cancelling (see below).

**Write path (code, v0.32.0 overlay).** Upstream tiering is write-through by design
(`v1/kv_offload/tiering/manager.py` header, principle 1, and `complete_store`): every chunk stored
GPU->RAM is cascaded to every secondary tier, evicted or not. `fs/io.py:98` skips a block whose file
already exists. Default policy is `CHUNK_LEVEL` (prefix-hit chunks skipped).

**Measured (two 150 s windows + a 4 min file scan, engine under Sunday-afternoon load):**
- fs write counter +1.732 GB = exactly 66 files x 26.25 MB created in the window; 133 files written
  in 4 min, all brand-new (birth time = mtime), **0 rewrites**. No re-storage of known blocks.
- fs writes by KV group: 4 groups, `g3` = attention (dense, one file per 1,424-token chunk),
  `g0`-`g2` = Gated DeltaNet state (mamba page padded 0.38% to the attention page size, boot log).
  Last 10 min: **150 attention files vs 222 GDN files (74 snapshots x 3)**, i.e. two thirds of
  current fs writes are GDN state snapshots. Whole tier: 5,067 attention vs 5,535 GDN files (52%).
- ~1.6 GDN snapshots per request (reachable boundaries: end of prompt, end of request), ~79 MB each.
  Store bytes per new token = 46.1-46.3 KB vs 18.4 KB attention-only: the 2.5x is the snapshots.
- Rate 11.5-19.2 MB/s = 41-69 GB/h; RAM tier turnover ~22-38 min, so write-on-evict would write
  almost the same volume. Only an admission filter (spill only what showed reuse) would cut it.
- Disk `sdd` (WSL ext4 vhdx) real writes were 1.8-2.3x the counter: other writers on the same
  disk, not identified. At this rate the fs tier alone writes ~1-1.7 TB/day to the SSD behind
  the vhdx (to check against its TBW).

**Prefix forks (body_head of 200s over 6 h, `error_source_capture`, 1,500 B depth):**
- claudish hub (Bun), 997 requests: **Claude Code main agent 878 (88%)**, identical through
  1,400 B (deepest we capture); Roo/Zoo modes 36 (3.6%); code-review bot 59 + 9; Agent SDK
  subagents/session naming 12; others <5. The "fork at byte 143/18" figures came from the Zoo
  modes and an older observation, not from today's Claude Code lanes.
- OpenAI SDK, 84 requests: roo-state-manager condensation, 3 prompt variants forking at 127 B.
- The Swift chat template renders **tools before the system text** (`chat_template.jinja:58-70`):
  the shared prefix begins with the tools block; lanes with different tool sets fork before any
  system text. The real Claude Code fork lies beyond 1,400 B of system text or inside the tools
  block: visible to the hub (full bodies), not to us.
- claudish artefact: the system text starts with its preamble followed by a literal JSON-encoded
  empty block `{"type":"text","text":""}` (identical in every request, not a fork).

## 25/10 17:40Z — routine cycle: healthy, one #431 400, CoursIA answered

- Engine: health 200 (5 ms), same generation since 09:04:20Z, 0 assert, 0 OOM. VRAM under load
  GPU 0 **22,733** / GPU 1 21,408 MiB (alert line 23,000, margin 267 MiB). KV usage 62 %.
  Preemptions 10 -> **12** in 2 h. `/dev/shm` 24G/63G.
- Since 09:04Z: 245.9 M prompt tokens, **4.7 % computed**, 66.2 % GPU prefix hits, 29.0 % loaded
  from the RAM/fs tiers; 2.52 M generated.
- **#431**: one 400 in 3 h, 16:31:33Z, 894,555 B body, hub (Bun via the site hairpin), "at least
  262145 input tokens" = client auto-compact one token past the window. Single attempt, no local
  retry. claudish #674/#685 (`/tokenize` pre-check) would route it before the 400. Sent to claudish.
- Security: 200s from known IPs only; `koara.live` keeps probing (~10/h, all 401).
- Condensation breaker opened a 6th time today: client-side, structural (status turn longer than
  the 90 s slice), tracked by po-2028 on #3962. Engine not involved.
- Host commit 88.8 % (CRIT 90): no heavy work. nbaudit ok. G: ok.
- CoursIA po-2027 (notebook 11b, PR #22462) asked which model fits 8 GB and which measured settings
  they may cite: answered with dated figures and their NOTES paths (4B GGUF Q4_K_M for a Windows
  laptop, checkpoint question redirected to po-2025's FrogNano-4B seat; cost per Mtok and watts
  flagged as not measured).
- claudish froze the identity header (#686): seats send `x-claudish-workspace`, the hub composes
  `x-claudish-client: <machine>:<workspace>` (fallback `<machine>:unknown`). RSM condensation
  calls :5002 directly, so roo-extensions was asked to set the same header via `defaultHeaders`.
  Our middleware logs it from the next engine boot.

## 25/10 19:00Z — mandate: implement the fs admission policy (#198), three workstreams

The user gave a GO for the whole programme ("OK pour tout… à votre rythme de cron en cron"):
implement issue #198 and the adjacent open issues incrementally, one surveillance cycle at a
time. Production is untouched by the work itself: everything below is code, tests and
measurement until a deployment window is explicitly agreed.

Workstreams (disjoint paths, so parallel agents cannot collide):

1. **Phase 0, measurement** — `myia_vllm/scripts/fs_tier/fs_snapshot_diff.py` (per-group fs
   read-back from the mtime instrument, restart-free) and `myia_vllm/docs/fs-tier/
   m2-instrumentation.md` (where to measure attention-only hit vs effective hit).
   Installed this evening: 427.5 GB written / 39.6 GB read back in ~9 h, 2/3 of the writes are
   GDN state snapshots (composition measured at 17:30Z).
2. **Phase 1, fs admission filter** — new overlay `myia_vllm/configs/docker/overlay_v0320_
   fsadmit/` built on the upstream `TieringAdmissionPolicy` interface (RFC #51240 / PR #60000,
   both open): frequency ≥ N before an fs cascade (the Dynamo default filter), plus a GDN
   latest-state rule behind its own flag. Rollback = one env var.
3. **Phase 2, GDN checkpoint at radix branch points** — design note
   `myia_vllm/docs/fs-tier/gdn-branch-checkpoint-design.md`. This is the lever that matters for
   the 30-minute Haiku lanes: after a compaction the socle cannot be hit today because no GDN
   state exists at the fork position, so it is recomputed every time even when its attention
   blocks are in GPU memory. Marconi (arXiv 2411.19379) does exactly this.

Nothing deployed. Each phase lands as its own PR; the fs tier only changes image after a GO.

## 25/10 19:41Z — routine cycle: healthy; M1 first read-back numbers landed

Same engine generation since 09:04:20Z (RC=0, OOM 0, 3 containers up, health 200 in 3.6 ms).

- **Load**: prompt tokens 286.35 M total (+40.5 M over the cycle, ~5,100 tok/s average);
  census 503 requests/126 min, hub (Bun) carries 293 of them with 100K-1M bodies; 478×200,
  25×401 (the usual dead-key actors), **0 new external IP with 200, 0 window+1 400** (#431
  stays clean since 16:31Z).
- **WEDGE `fail 1/2` ×5** (18:09→19:30Z) under those long prefills — documented starvation
  of the probe, never 2/2, no restart; 2,001 probes OK.
- **Preemptions 17** (+5 over the cycle), prefix hit 61.2 % cumulative, VRAM GPU 0 22,805
  (creeping, 195 MiB under the 23,000 alert; rule unchanged: 0.70 at next restart if crossed).
- **fs tier since 17:59Z**: +85 GB written (≈47 GB/h sustained), +29.1 GB read back,
  fs chunk hits 2,617 (+1,107 in ~1.7 h) — the evening replays are real; M1 distinct-file
  read-back measured at 19:20Z: attention 6.2 %, GDN 0.9 % (PR #199).
- **Phase 2 re-scoped by the design note** (PR #200): upstream already stops the prefill chunk
  at the block-floored junction and caches the state there; what is missing is *persistence*
  (LRU evicts it in 22-38 min), so the fork change is a protected-block class, ~1 week, not a
  new checkpoint engine. Correction posted on #198.
- **Condensation**: breaker OPEN again 18:24Z, cloud fallback succeeded (RSM-side, diagnostic
  with roo-extensions still pending; engine counters exonerate the motor).
- **Commit 89.2 %** (0.8 pt under CRIT) — WSL remains the consumer; no heavy job launched.
- `/dev/shm` 24G/63G (floor), G: OK, nbaudit `ok:true` 0 failures — note: the runner now
  serves `19451:Image/` (wave 2 armed).
- **DST**: Europe/Paris switched to winter time last night — the host is UTC+1 now, not UTC+2.
  Spec line corrected (this PR).

## 25/10 20:55Z — phase 1a landed: overlay v0320-fsadmit (PR #202, NOT deployed)

All three workstreams of the mandate are now in main; the engine has not been touched.

- **Phase 1a** (PR #202): new overlay `overlay_v0320_fsadmit/`, image
  `vllm-openai-v0320-fsadmit:v1` built for testing, referenced by no profile.
  Frequency gate at `MYIA_FS_ADMIT_MIN_HITS` (default 2 = the Dynamo default;
  0 = exact upstream write-through = rollback). The one design subtlety worth
  remembering: the threshold-crossing cascade is submitted **from inside
  `lookup()`**, because a chunk that stays hot in RAM is never re-stored — a
  store-time-only gate would never persist a stable socle, and after every
  restart the socles would stop reaching the fs entirely. Tests (CPU-only
  throwaway containers): 7/7 admission + upstream v0.32.0 62/62 (default) and
  44/44 + 51/51 (rollback mode, fs events included — byte-identical).
- **GDN latest-state rule**: documented no-op stub — the tiering manager has
  no conversation-chain identity (keys are opaque hashes); home is upstream
  #60000 `TieringAdmissionPolicy` or phase 2.
- **Deploy**: proposed for the 26/10 09:00-11:00Z window, alongside the
  claudish recreate, so `x-claudish-client` counters start at the same boot.
  Registry Q38 opened with that default. Post-deploy acceptance gate:
  disk-served prompt-token share >= 80 % of the 2.2 % baseline.
- claudish corrections absorbed (18:36Z): the "33 % hole" was an instrument
  artifact; real haiku landing 54 % PAYG / 38 % swift; ai-01 pays 90 % of its
  haiku in cash — swift saturates at 3-4 concurrent ~190K conversations.
  PR claudish#685 merged 19:44Z (checks green) after body+comments+reviews
  read; rides tomorrow's recreate with the #686 header.

## 25/10 21:41Z — routine cycle + M1 second window: the group gap holds

Same generation since 09:04:20Z (RC=0, 0 OOM, health 200/4.3 ms, shm 24G).

- **M1 window 2** (148 min, 19:13Z → 21:41Z): attention hit 9.6 % (348 files), GDN 2.0 %
  (81 files); 81.1 GiB created (~33 GB/h evening rate vs 47 daytime). Two independent
  windows now agree: GDN is written ~2x the attention bytes and read back 4-5x less.
- **Evening load**: 201 requests with 100K-1M bodies in 2 h, all from the hub (hairpin);
  window prefix-hit ~8 % of block queries (cumulative ratio diluted 61 % → 55 %) — fresh
  contexts dominate (haiku lanes + compactions), i.e. exactly the socle-recompute cost
  #198 targets. **Preemptions 27 (+10)** — KV pressure climbing; re-check after the
  Minimax reset (23:05Z) before calling it a trend. 4 more WEDGE `fail 1/2` under long
  prefills, never 2/2; VRAM GPU 0 22,834 (creep +29 MiB/2 h, margin 166 MiB).
- 0 new external IP with 200, 0 window+1. Commit 84.9 % (back under 85).

## 25/10 21:48-21:55Z — ★ DEPLOYED: fs admission filter live (user GO "maintenant")

User moved Q38 forward ("si c'est mieux de coupler… faites-le ce soir si possible.
Possible maintenant ?") — executed immediately rather than waiting for the 23:05Z
Minimax reset: a warm boot with the user present beats a marginally quieter window.

- Recreate `vllm-medium-swift15-27b` on `VLLM_IMAGE_SWIFT15=vllm-openai-v0320-fsadmit:v1`
  21:48:52Z. Warm boot (compile cache volume shared): health 200 at 21:53:46Z
  (4 min 54 s, compilation 2.57 s). KV 831,723 tokens identical, VRAM cold
  21,100 / 19,676 MiB.
- **Filter active**: one-time line `manager.py:286 fs admission: cascades to STORAGE
  tiers require >= 2 primary-tier lookup hit(s)` at 21:53:27Z. Rollback switches:
  `MYIA_FS_ADMIT_MIN_HITS=0` (recreate) or image back to tierpin.
- /dev/shm purge post-boot: 7 orphans of the dead generation, 48G -> 24G.
- Generation probe: `enable_thinking:false` -> "OK" (first probe's null content was
  the documented thinking-budget trap, §5bis).
- Fleet warned before the gesture: global dashboard (crossPost workspace-vllm) +
  DM claudish (skip/busy suggestion until ~23:00Z; their morning hub recreate is
  unchanged and now lands on an engine that already logs `x-claudish-client`).
- Baseline for the gates (pre-restart generation, since 09:04Z): fs writes 594.5 GB,
  fs reads 82.7 GB, fs chunk hits 3,152, prompt tokens 325.0 M, preemptions 28.
  First-hour measurement at the 23:41Z cycle: fs write rate (was 33-47 GB/h) and
  disk-served share (acceptance: >= 80 % of the 2.2 % baseline share).

## 25/10 23:41Z — first 1h50 under the fs filter: −68 % new files, gate PASSED; re-writes identified

Engine generation 21:48:52Z, RC=0, health 200, 0 OOM, 0 tiering assert, probes OK.
Security: 0 new external IP, 0 window+1, 21×401 dead keys. shm 24G, G: OK,
nbaudit ok:true, commit 88.6 %.

**Acceptance gate PASSED**: disk-served prompt tokens 1.51 M of 32.76 M = **4.6 %**
of the prompt (baseline 2.2 %; gate was >= 80 % of it). fs chunk hits 870;
preemptions 5 (new generation); 4 WEDGE `fail 1/2` under long prefills, never 2/2.
The post-restart replays were served from the pre-existing fs content as designed:
fs reads 22.8 GB in 1h50.

**Write reduction (M1, new files created in the window)** — 19:13Z baseline vs 23:44Z:

| group | before (148 min) | after (123 min) | reduction |
|---|---|---|---|
| GDN created | 1,810 / 44.2 GiB | **300 / 7.3 GiB** | **−75 %** |
| attention created | 1,507 / 36.8 GiB | **775 / 18.9 GiB** | **−48 %** |
| total | 3,317 / 81.1 GiB | 1,075 / 26.3 GiB | **−68 %** |

The GDN target is hit: snapshots that used to be written every request and read
back 1-2 % of the time now mostly stay off the SSD.

**Honest gap vs the ≥5× counter target**: the fs write counter reads 38.1 GB since
boot (≈20.6 GB/h, was 33-47 GB/h) while only 26.3 GB of *new* files appeared in
1h50 (and 7 min of that window predate the boot). The difference is **re-writes of
already-admitted keys**: `_fs_admitted_keys` passes a key unconditionally at
`complete_store`, so every hot socle re-stored from RAM (after a prune, an
eviction-and-recompute cycle, or a restart re-crossing) rewrites identical bytes.
This is the next lever, not a correctness problem — content-addressed files, and
the filter's read-back still pays. Candidate v2: skip the re-cascade when the fs
file already exists (async fs lookup result cache, or the file-existence check in
the tier's store path) instead of relying on the idempotent write.

**Watch**: VRAM GPU 0 22,942 (margin 58 MiB under 23,000). If it crosses:
`GPU_UTIL_SWIFT15=0.70` at next restart — not tonight, and not without a fresh
measurement first.

### 26/10 00:15Z — the residual half is measured, and it is re-writes

First decisive split of what the fs tier still writes, taken 2 h 15 min after the
filter went live (engine start 21:48:52Z). Instrument: `stat -c '%W %Y %s'` per
file touched in a 30 min window, birth time from **statx** (GNU `find -newerBt`
is **unsupported** on this filesystem — it prints "does not provide a way to
find the birth time" and silently returns 0 files; Python's `st_birthtime` is
also absent on Linux and falls back to `None`, which silently classifies every
file as a rewrite. Both traps produce a confident, wrong answer; use
`subprocess` + `stat(1)`, which is what `myia_vllm/scripts/fs_tier/`-style
helpers must do here.)

| 30 min window | files | GB |
|---|---|---|
| touched | 303 | 7.95 |
| **new** (birth inside window) | 149 | 3.91 |
| **re-write** (birth before window) | 154 | 4.04 |

**Half the remaining fs bandwidth rewrites a file that already exists.** This
is the `_fs_admitted_keys` pass-through: once a key is admitted, `complete_store`
sends it to the fs unconditionally on every RAM store, and the tier rewrites the
whole block. The keys are content-addressed (64-hex sha256 filenames), so a
re-write carries bytes identical to what is already on disk — safe to skip by
construction, which is exactly what overlay v2 does.

Engine-start to 00:04Z (2 h 15 min), from `/metrics`:

| | value | baseline |
|---|---|---|
| fs bytes written | 42.10 GB (**18.7 GB/h**) | 33-47 GB/h |
| fs bytes read back | 27.01 GB (12.0 GB/h) | 4.4 GB/h |
| read/write ratio | **64 %** | 9.3 % |
| fs chunk queries → hits | 9,030 → 1,023 (**11.3 %**) | 6.4 % |
| prompt tokens from disk | 1.735 M of 38.76 M = **4.48 %** | 2.2 % |
| bytes written per prompt token | **1.06 KB** | 1.7 KB |

The 64 % read-back is inflated by the post-restart re-prefill (the engine had to
reload everything from cold at 21:48Z) — treat it as a warm-start upper bound,
not a steady state. The disk-served share, in contrast, is a steady-state
figure for the same window and it is **2.0× the baseline**, comfortably above
the gate's 80 %-of-baseline floor.

Projection, stated as a projection and not a result: v2 removes the ~51 %
re-write share, 18.7 → ~9.2 GB/h, i.e. **1.06 → ~0.52 KB per prompt token
(÷3.3 vs baseline)**. The ×5 target of issue #198 is then reached by Phase 1(b)
(GDN latest-state rule), not by v2 alone.

**Watch**: VRAM GPU 0 22,956 MiB at 00:03Z — 44 MiB under the 23,000 alert line,
GPU 1 21,508. `preemptions_total` 10 in this generation. No night-time restart.

### 26/10 00:35Z — CORRECTION: the composition inverted under the filter

Per-class split of the same 30 min window (class from the `_g<N>` path component:
`g3` = attention, `g0-g2` = the three TP ranks of a GDN state block):

| class | new (GB) | rewrite (GB) | total (GB) | share |
|---|---|---|---|---|
| attention (`g3`) | 2.861 | 3.097 | 5.958 | **75.2 %** |
| GDN state (`g0-g2`) | 1.024 | 0.945 | 1.969 | **24.8 %** |

**The pre-filter "two thirds GDN" no longer holds: GDN is a quarter of the
writes now.** The mechanism is the filter itself, and it is the opposite of an
accident. A one-off GDN end-of-request snapshot has a unique key by
construction — it is never looked up twice, so it never reaches the 2-hit
threshold and is never admitted. The frequency filter therefore removes
one-off GDN snapshots *structurally*, without any GDN-specific rule. Exactly
the outcome Gandalf (arXiv 2610.30791) predicts for admission on access
history, reached here as a side effect rather than by design.

Consequence for the plan, and it corrects what I wrote 20 minutes ago in this
file and on issue #198:

- **v2 (skip a re-write of an existing content-addressed file) is the bigger
  lever, not the smaller one.** Rewrites are ~52 % of attention bytes and ~48 %
  of GDN bytes — uniform, so the same predicate catches both.
- **18.7 GB/h -> ~9.2 GB/h projected.** Against the 47.7 GB/h mean baseline
  (issue #198) that is **÷ 5.2**; against the low end of the 33-47 GB/h range it
  is ÷ 3.6. The x5 criterion is therefore plausibly met by v2 alone, and the
  honest statement is the range, not a single ratio.
- **Phase 1(b) (GDN latest-state rule) is now a much smaller lever than the
  issue assumed** — it can only act on the remaining ~25 %, and half of that is
  rewrites v2 already removes. Deprioritised, not abandoned: it still matters
  for the user's "keep the state just before the compaction message", which is
  a *correctness of reuse* goal, not a write-volume goal.

Uniform block size, all groups: 26,214,144 B (25 MiB) per `.bin`. A GDN state is
always a triplet of three such blocks (one per TP rank), i.e. ~75 MiB, not 79.

### 26/10 01:41Z — cycle: first full night under the filter, steady

Window = engine start 21:48:52Z to 01:15Z (3 h 26 min, real traffic). All
counters cumulative since boot.

| metric | value |
|---|---|
| fs writes | 57.56 GB = **16.7 GB/h** (was 18.7 GB/h at 2 h 15 — post-restart burst washing out) |
| fs reads | 39.34 GB = 11.4 GB/h |
| prompt tokens (total) | 58.45 M = **17.0 M tok/h sustained** — heavy night |
| prompt tokens from cache | 49.21 M = **84.2 %** (device 24.8 % / host RAM 54.9 % / disk 4.46 %) |
| computed prefill | 9.24 M = 15.8 % |
| fs lookup hit rate | 1,268+ (12.2 % at 2 h 15) |
| preemptions | 14 |
| VRAM GPU 0 / GPU 1 | 22,932 / 21,508 MiB (68 MiB under the 23,000 line) |
| KV usage | 34 % at probe time (92 % an hour earlier — the saturation comes and goes) |

Bandwidth measured over the same window (counters, `/metrics`): RAM->GPU
681.9 GB in 219.7 s = **3.10 GB/s** (1.8 % of wall-clock); GPU->RAM 2.52 GB/s;
SSD->RAM 39.3 GB in 18.2 s = **2.16 GB/s**; RAM->SSD 0.06 GB/s on its dedicated
pool (async, never blocks decode — by design).

**Watchdog**: 12 x `WEDGE fail 1/2` between 23:42Z and 01:14Z, zero `2/2`, zero
restarts — the documented long-prefill starvation under 17 M prompt tok/h. v6.2
classified the BUSY cases correctly and did not restart. Frequency is up on the
19:41Z cycle (5 in 2 h) but the regime is the same and preemptions stay low.

**Host commit 88.6 %** (350.2/395.3 GB, margin 45.1 GB) — above the 85 % rule:
no heavy work from this lane; reported, not acted on (first consumer is
`vmmemWSL`, per the 09/10 finding).

Security: no new IP with 200s (hairpin 783, internal 444, user-Paris 13, web2
2). `#431` signature (400 + body > 100 KB): **0 occurrences**.

nbaudit: ok, series 19451 now scoped to `Image/` (wave 2 armed). CoursIA
#22462 answered with measured figures only (this file + PR #104/#123/#131/#195).

### 26/10 03:41Z — cycle: the write rate is collapsing on its own, and the hit rate is rising

Window since boot 21:48:52Z -> 03:41Z = 5 h 52 min. Delta vs the 01:41Z cycle
(2 h 26 min in between).

| metric | cumulative | delta over 2 h 26 |
|---|---|---|
| fs writes | 69.95 GB (11.9 GB/h avg) | **+12.39 GB = 5.1 GB/h** (was 16.7) |
| fs reads | 42.18 GB | +2.84 GB |
| prompt tokens | 100.98 M | +42.53 M = **17.5 M tok/h** (load continues) |
| cached share | **89.1 %** (device 51.1 / host 34.8 / disk 3.16) | up from 84.2 % |
| computed prefill | 11.02 M = 10.9 % | down from 15.8 % |
| preemptions | 14 | +0 |
| WEDGE events | 0 in the last 2 h | 12 in the 2 h before |
| KV usage | 59 % | — |

**The intended cruise is arriving without the v2 patch.** As the working set
stabilises, the hot socles are all admitted *and* already on disk, so new
admissions stop and the cascade has nothing to carry: 16.7 -> 5.1 GB/h. The
57 % write-rate drop in one window is a steady-state effect of Phase 1(a)
alone, not of any new change. The 51 % rewrite share still applies to whatever
remains (and to the burst at any restart), so v2 stays worth its restart — but
the projected ceiling moved: in cruise the filter alone may already sit near
the x5 target, and the honest comparison is now *cruise vs cruise*, not
burst-inclusive vs baseline.

The hit rate rising (84.2 -> 89.1 %) with preemptions flat is exactly the
"cruisière longue" signature the surveillance spec asks for.

**Host commit 90.1 %** (356.2/395.3 GB, margin 39.1 GB) — the 09/10 alert
level. No heavy work; reported. First consumer remains vmmemWSL.

Security: same four known actors, zero new IP, zero #431 signature. nbaudit ok
(03:06Z, Image/). Condensation log: no event, only heartbeats. CoursIA
confirmed our #22462 figures are integrated as-is, caveat included (c.6406755988).

### 26/10 05:41Z — cruise measured on its own: 5.0 GB/h, cached share 91.4%, and a first apples-to-apples x5 verdict

Boot 21:48:52Z -> 05:41Z = 7 h 52. Delta window vs 03:41Z = exactly 2 h,
prompt load still 22.15 M tok/h (higher than the previous window).

| metric | cumulative | delta 03:41->05:41 (2 h) |
|---|---|---|
| fs writes | 82.26 GB | +12.26 GB = **5.0 GB/h** (same as the previous window: stable, not a one-off dip) |
| fs reads | 42.81 GB | +0.63 GB |
| prompt tokens | 145.27 M | +44.29 M = **22.15 M tok/h** |
| cached share | **91.4 %** (device 63.1 / host 25.3 / disk 2.40) | 84.2 -> 89.1 -> 91.4 over 6 h |
| computed prefill | 13.69 M = 9.4 % | 2.67 M this window = **6.0 %** |
| preemptions | 14 | **+0 for 4 h** |
| WEDGE | 2 events (03:57, 04:05Z) | — |
| KV usage | 67 % | — |

**The x5 target can now be measured like-for-like.** Baseline (issue #198,
25/10 17:53Z): **1.7 KB written per prompt token**. In this window:
12.26 GB written / 44.29 M prompt tokens = **0.29 KB/token, i.e. ÷5.9** —
and against the same kind of load (heavy recurrent sessions) rather than
against a burst-inclusive average. Both acceptance criteria of #198 are met
in cruise: write rate ÷5.9, disk-served share 2.40 % >= 80 % of the 2.2 %
baseline. The earlier "÷3.6-5.2 projected with v2" estimate was wrong in
direction, not just in size: the filter alone overshot it.

**This changes the v2 (PR #211) recommendation.** v2 would remove ~half of
the remaining 5 GB/h, i.e. save ~2.5 GB/h ≈ 60 GB/day of SSD writes, and it
eliminates the post-restart write burst (16.7 GB/h in the first hours). It
is no longer needed to reach the acceptance target — the trade is now
explicitly "one restart (~5 min down + ~1 h re-prefill) versus ~60 GB/day
and the burst". Q39 updated accordingly; the lane's recommendation shifts
from "yes, grouped" to "optional — not worth a dedicated restart; fold into
the next image bump or leave off".

**⚠ Host commit 92.4 %** (365.1/395.3 GB, margin 30.2 GB) and **climbing
across three cycles**: 88.6 -> 90.1 -> 92.4. Measured top consumers:
`vmmemWSL` **165.4 GB commit / 113.1 GB resident**, a host `python` 38.0 GB.
Inside the WSL VM: 55 GB used of 125 GB, 26 GB shared, 57 GB buff/cache,
70 GB available; /dev/shm 25 GB (the RAM tier itself). The VM's committed
footprint is ~2x what its own accounting shows used — the known WSL2
commit-charge behaviour, aggravated by the 24 GiB tmpfs tier and the
GPU-2 CoursIA training. This is the pressure that killed Docker Desktop
on 24/09 at 97.8 %. No heavy work from this lane; watch 97 %. Not ours to
fix (`.wslconfig` / `wsl --shutdown` would kill engine and hub).

Security: five known actors, no new IP, zero #431 signature. nbaudit ok.

### 26/10 07:41Z — write rate keeps falling on its own (3.7 GB/h, 0.18 KB/token); disk serving migrates up-tier

Window 05:41->07:41Z (2 h). Cumulative boot 21:48:52Z -> 07:41Z = 9 h 52.

| metric | cumulative | delta 2 h |
|---|---|---|
| fs writes (`tiering_write_bytes_total`) | 89.71 GB | **+7.45 GB = 3.7 GB/h** (16.7 -> 5.1 -> 5.0 -> 3.7) |
| fs reads (`tiering_read_bytes_total`) | 48.14 GB | +5.33 GB = 2.7 GB/h (up from 0.32) |
| prompt tokens | 186.07 M | +40.80 M = 20.4 M tok/h |
| cached share | **91.6 %** (device 129.11 / host 37.54 / disk 3.79 M) | window: device serves 91.8 % of prompt |
| computed prefill | 15.96 M = 8.6 % | 2.27 M = **5.6 % of window** |
| preemptions | 14 | **+0 for 6 h** |
| WEDGE | 0 new since 04:05Z | — |
| fs chunk hits | 1,825 | +940 over 8 h (435 in the first 2 h) |
| fs tier size | **251 GB = at the 250 GiB cap** | pruner active, as designed |
| /dev/shm | 25G/63G (39 %) | RAM-tier floor |
| VRAM | GPU0 22,636 / GPU1 21,508 MiB | GPU0 364 MiB under the 23,000 line |

**x5 verdict this window: 7.45 GB / 40.80 M tok = 0.18 KB/token = ÷9.3** (was
÷5.9 at 05:41Z). The cruise keeps deepening with no code change — the filter's
end-state is arriving on its own.

**Rewrites are now 84 % of what remains** (birth-time scan, 30-min window, 96
files: 15 NEW 0.39 GB vs 81 REWRITES 2.13 GB — up from 51 % at 00:40Z). v2
(PR #211) would cut this scan window from 2.52 to 0.78 GB/h; at this depth its
saving approaches ~100 GB/day, which strengthens (not changes) the Q39
recommendation: fold into the next image bump, no dedicated restart.

**Disk-served share, two honest readings.** Cumulative: 2.04 % of prompt tokens
(floor = 1.76 %) — passing. Window: ~0.7-0.9 % — below floor, because device
now serves 91.8 % of the window's prompt and computed prefill is down to 5.6 %
with preemptions flat for 6 h. This is the filter's intended end-state, not a
serving failure: hot socles live in GPU/RAM and the fs tier degrades toward
restart/cold-replay insurance (fs reads still ran at 2.7 GB/h this window).
The floor was written to catch "the filter killed disk utility"; disk utility
falling because serving migrated to faster tiers is the success case. Recorded
as-is; **Monday's cold-working-set diurnal window remains the gate before
closing #198.**

**Window+1 400s: 4 in 3 h** (07:07, 07:09, 07:35, 07:42Z; 0.9-4.2 MB bodies,
all via the site hairpin — AsyncOpenAI and Bun UAs, 26+ min apart, no
retry-loop). Known signature (claudish #674/#685); the hub recreate carrying
the fix is scheduled 09:00-11:00Z today. Nothing new to escalate from the
engine side.

**Host commit 92.8 %** (366.7/395.3 GB, margin 28.6 GB) — fourth consecutive
rise (88.6 -> 90.1 -> 92.4 -> 92.8). No heavy work from this lane; watch 97 %.

Security: two known actors with 200s (hairpin 248, internal no-XFF 216),
20 x 401 from the known dead-key process (fp 1f7f6080cb65), zero new IP, no
unknown key. nbaudit ok (07:05Z, wave 2 Image/). G: write OK. Condensation
nominal (heartbeats only). GPU 2 = CoursIA training (24,091 MiB, expected).

Coordination: CoursIA-2's [LIFT DEMANDE] on #21704 re-routed to
myia-ai-01:CoursIA (the reserve is that persona's, review 5593125735) — vllm
does not lift another lane's reserve; DM answered and marked read.

### 26/10 09:45Z — cold morning working set fails the write target; disk serving passes

Window 07:41 -> 09:45Z (2 h 04). Read-only cycle. User approved additional patch
deployment **this evening after their EPF classes**; no engine/config change was
made during this cycle.

| metric | cumulative at 09:45Z | delta 2 h 04 |
|---|---|---|
| fs writes | **131.945 GB** | **+42.235 GB = 20.44 GB/h** |
| prompt tokens | 223.514 M | +37.444 M = 18.12 M tok/h |
| fs reads | 72.862 GB | +24.722 GB = 11.96 GB/h |
| cached by source | device 129.122 M / host 66.995 M / disk 5.610 M | window: device 0.03 % / host 78.66 % / disk **4.86 %** |
| computed prefill | — | 6.157 M = **16.44 %** of window prompt |
| preemptions | **25** | **+11**, after +0 for 6 h |
| fs chunk hits | 2,767 | +942 |
| fs tier size | 273.60 GB = 254.83 GiB | nominal at the 250 GiB LRU cap |

**The cold-window write gate FAILS:** `42.235 GB / 37.444 M prompt tokens =
1.128 KB/token = div 1.51` against the 1.7 KB/token baseline, far short of the
required div 5. The 30-min birth-time scan at ~09:51Z explains both components:
524 touched files = **292 NEW / 7.66 GB + 232 REWRITES / 6.09 GB**, zero unknown
birth times (13.75 GB/30 min = 27.5 GB/h). New blocks are again the majority,
so skip-rewrite alone cannot meet div 5 in this regime.

**The disk-serving gate PASSES:** window disk share **4.86 %** vs the 1.76 %
floor; cumulative 2.51 %. Cumulative cached share 90.25 %. This is the inverse
of the 07:41Z cruise profile: hot RAM serves 78.7 %, fs reads and disk-served
tokens rise, and GPU-resident serving nearly disappears from this window.
Issue #198 stays OPEN. Do not extrapolate the overnight div 9.3 to a cold set.

The **+11 preemptions** coincide with real diurnal KV pressure: recent engine
samples show GPU KV 74-95 %, 3-7 running, 3-7 waiting, and KV-fetch backlog.
Decode probes continue; all observed WEDGE events remain **fail 1/2**, never
2/2, no watchdog restart. Health 200 in 4.5 ms, StartedAt still 21:48:52Z,
RC=0, OOM=false, 0 OOM log lines, three containers up. GPU0 **22,658 MiB** /
GPU1 21,508; GPU2 24,092 = expected CoursIA training. `/dev/shm` 25G/63G (39 %).
No automatic configuration escalation: Q35's long-prefill threshold stays
rejected, and any tuning belongs to tonight's measured window.

**Deployment preparation:** user said additional patches can be deployed this
evening. PR #211 was read in full (body, all comments/reviews — zero — and both
file diffs): one commit `75403198d9`, no CHANGES_REQUESTED, 6/6 tests already
recorded. It remains unmerged and inactive. Tonight's baseline is fixed above;
anti-rewrite saves at most the measured rewrite share, not the new-block
majority. Admission/serving gates must be re-measured after restart. Kill
switch `MYIA_FS_SKIP_REWRITE=0`; no merge-only pull into the live bind-mounted
source outside the deployment window.

Security: only known hairpin `90.65.170.144` and internal no-XFF had 200s in
the 3 h census (525 x 200, 29 x 401, 4 x 400). No new external IP or unknown
key fingerprint. The same four known window+1 400s (07:07-07:42Z) remain;
zero new large-body 400 in the following ~2 h. nbaudit healthy at 09:06Z
(`runner_rc=0`, `deliver_rc=0`, `ok=true`, `consecutive_failures=0`, Image/).
G: write probe OK. Condensation nominal. The RSM log burst at 09:42Z is its own
stdin-end graceful-shutdown handling (`[object Object]` errors), not a vLLM
condensation failure; coordination tooling to be checked before closure.

**Host commit 93.4 %** (369.1/395.3 GB, margin 26.1 GB), fifth consecutive
rise. No heavy host work. No WSL/Docker remediation without user GO.

### 26/10 11:41Z — the cold set warms back toward cruise; rewrites dominate again

Window 09:45 -> 11:44:41Z (1 h 59m41s). Read-only cycle; no engine/config change.
The evening patch window (user GO) remains after the EPF classes.

| metric | cumulative at 11:44Z | delta 1 h 59m41s |
|---|---|---|
| fs writes | **148.848 GB** | **+16.903 GB = 8.47 GB/h** |
| prompt tokens | 259.947 M | +36.433 M = 18.26 M tok/h |
| fs reads | 86.563 GB | +13.701 GB = 6.87 GB/h |
| cached by source | device 154.547 M / host 74.658 M / disk 6.265 M | window: device 69.8 % / host 21.0 % / disk **1.80 %** |
| computed prefill | — | 2.691 M = **7.4 %** of window prompt |
| preemptions | **27** | **+2**, both before 10:06Z; flat for the last ~1 h 40 |
| fs chunk hits | 3,179 | +412 |
| fs tier size | **251G** (`du`) | at the 250 GiB LRU cap, pruner active |

**Write ratio: 16.903 GB / 36.433 M prompt tokens = 0.464 KB/token = div 3.66**
vs the 1.7 KB/token baseline — between the cold-window div 1.51 (09:45Z) and the
overnight cruise div 9.3 (07:41Z). The 30-min birth-time scan confirms the warm-up
from the rewrite side: 78 touched files = **12 NEW / 0.31 GB + 66 REWRITES /
1.73 GB** (84.8 % rewrites, 2.04 GB/30 min) — the exact residual PR #211
addresses tonight.

**Disk-serving gate: marginal PASS** — window disk share 1.80 % vs the 1.76 %
floor (cumulative 2.41 %, cached share cumulative 90.6 %). Honest reading: one
rounding step from the floor; the GPU tier now serves ~70 % of the window's
prompt again, so the fs tier is reverting toward cruise behavior (restart/cold
insurance) rather than active serving.

Engine health: StartedAt still 21:48:52Z, RC=0, OOM=false, 0 OOM lines, 0
tiering asserts/tracebacks in the last 20k lines, three containers up, health
200 in 5.0 ms. WEDGE `fail 1/2` at 10:03, 10:06 and 11:39Z, **0 x fail 2/2**,
no watchdog restart. VRAM GPU0 **22,747** / GPU1 21,508 MiB (253 MiB under the
23,000 line, alert threshold not crossed); GPU2 10,301 = expected CoursIA.
`/dev/shm` 25G/63G (39 %). Live scheduler sample: KV 59.8 %, 4 running,
0 waiting.

Traffic census (2 h, 447 req): 428 x 200, 19 x 401, **0 x 400** — no new
window+1 signature this window (no attribution claimed without claudish
confirming the hub recreate). 218 bodies in 100K-1M (Bun hub clients), spread
1-9 per 10 min, no retry-loop signature. Security: **new unclaimed mobile IP
`92.184.146.46`, 2 x 200** (10:01/10:08Z, OpenAI/JS, key `be312daf7c1c` =
MEDIUM, bodies 33K/54K) -> **Q40** opened in the registry for claim;
all other 200-holders known (hairpin + internal no-XFF). nbaudit healthy at
11:06Z (Image/ wave 2). G: write OK. Condensation nominal (heartbeats only;
the 06:15-06:51Z `roosync_dashboard SLOW` warnings are RSM-side).

**Host commit 88.7 %** (350.6/395.3 GB, margin 44.7 GB) — down from 93.4/93.6 %
at 09:45Z, natural relief; still above the 85 % heavy-work threshold, so no
heavy job from this lane.

Coordination: Maintenance's Docker-image qualification (Maintenance#165)
answered as owner: `vllm/vllm-openai:v0.31.0` retirable (rollback is stock
v0.32.0, not v0.31), `hyperqwen:latest` kept until a preservation tar exists
(mutable tag), K2 overlay decision grouped with its layer sibling.

### 26/10 13:41Z — back to cruise without code change (div 8.7); second mobile IP joins Q40

Window 11:44:41 -> 13:41:09Z (1 h 59m28s). Read-only cycle; no engine/config
change. Evening patch window (user GO) unchanged, after the EPF classes.

| metric | cumulative at 13:41Z | delta 1 h 59m28s |
|---|---|---|
| fs writes | **160.239 GB** | **+11.391 GB = 5.72 GB/h** |
| prompt tokens | 318.147 M | **+58.199 M = 29.23 M tok/h** |
| fs reads | 91.366 GB | +4.803 GB = 2.41 GB/h |
| cached by source | device 208.165 M / host 76.451 M / disk 6.852 M | window: device **92.1 %** / host 3.1 % / disk 1.01 % |
| computed prefill | — | 2.202 M = **3.78 %** of window prompt |
| prefix cache | queries 318.223 M / hits 208.165 M | window hit rate **92.4 %** |
| preemptions | **27** | **+0 for ~3 h 45** |
| fs chunk hits | 3,362 | +183 |
| fs tier size | **252G** (`du`) | at the 250 GiB LRU cap |

**Write ratio: 11.391 GB / 58.199 M prompt tokens = 0.196 KB/token = div 8.7**
— the cold morning set has fully warmed back to cruise on its own, same
configuration. The 30-min birth-time scan is correspondingly small: 61 touched
files = **34 NEW / 0.89 GB + 27 REWRITES / 0.71 GB** (1.60 GB/30 min = 3.2
GB/h; new blocks 55.6 % of a much smaller residual).

**Disk-serving gate: window FAILS the 1.76 % floor (1.01 %), cumulative still
passes (2.15 %).** This is the documented success-case migration, not a serving
loss: the GPU tier serves 92.1 % of the window's prompt, computed prefill is
3.78 %, preemptions have been flat for ~3 h 45, and fs reads still ran at
2.41 GB/h. The floor was written to catch "the filter killed disk utility";
hot socles living in GPU/RAM is the intended end-state. `#198` remains open
pending tonight's measured patch window.

Engine health: StartedAt still 21:48:52Z, RC=0, OOM=false, 0 OOM lines, 0
tiering asserts/tracebacks, three containers up, health 200 in 4.2 ms. **No new
WEDGE after 11:44:39Z**, `fail 2/2` count 0, no watchdog restart. VRAM GPU0
22,479 / GPU1 21,508 MiB; GPU2 10,301 = expected CoursIA. `/dev/shm` 25G/63G
(39 %). Live scheduler sample: KV 40.7 %, 3 running, 0 waiting.

Traffic census (2 h, 699 req): 676 x 200, 23 x 401, **0 x 400** — no
window+1 signature for 4+ h (still no recreate attribution claimed without
claudish confirmation). 424 bodies in 100K-1M, Bun 426 req, no retry-loop
signature. Security: **second unclaimed mobile IP `92.184.146.254`, 2 x 200**
(13:04:32Z 18,678 B, 13:05:45Z 55,327 B, OpenAI/JS, key `be312daf7c1c` =
MEDIUM) — same pool and frappe as `92.184.146.46` 3 h earlier -> **Q40
extended to both IPs**, one claim likely covers both. All other 200-holders
known (hairpin + internal no-XFF). nbaudit healthy at 13:06Z (Image/). G:
write OK, ~608 GB free.

**RSM-side condensation warning (not vLLM):** from 13:31Z, each deferred MCP
tool call spawns `RooStateManagerServer` instances via `mcp-wrapper.cjs` with
`OPENAI_BASE_URL`/keys MISSING (burst of server starts at 13:42:22Z, ~20+
`node` processes created 13:42Z), while the resident process has the correct
config. Context read with +/- lines before relaying; DM sent to
`myia-ai-01:roo-extensions` (`vllm-rsm-missing-env-20261026T1341Z`). Engine
:5002 saw no error and the dashboard tools keep working.

**Host commit 89.6 %** (354.3/395.3 GB, margin 41.0 GB) — up 0.9 pt from
11:41Z, below the 97 % danger line, above the 85 % heavy-work threshold; no
heavy job from this lane.

### 26/10 15:41Z — stable cruise; fs writes almost quiet, 400 is image-count

Window 13:41:09 -> 15:41:27Z (1 h 59m18s). Read-only cycle; no engine/config
change. Evening patch window (user GO) remains after EPF classes.

| metric | cumulative at 15:41Z | delta 1 h 59m18s |
|---|---|---|
| fs writes | **160.790 GB** | **+0.551 GB = 0.277 GB/h** |
| prompt tokens | 360.976 M | **+42.829 M = 21.51 M tok/h** |
| fs reads | 92.233 GB | +0.866 GB = 0.434 GB/h |
| cached by source | device 249.319 M / host 76.998 M / disk 7.000 M | window: device **96.0 %** / host 1.3 % / disk 0.35 % |
| computed prefill | — | 0.980 M = **2.29 %** of window prompt |
| prefix cache | queries 361.426 M / hits 249.319 M | window hit rate **95.26 %** |
| preemptions | **27** | **+0 for ~5 h 45** |
| fs chunk hits | 3,395 | +33 |
| fs tier size | **251G** (`du`) | at the 250 GiB LRU cap |

**Write ratio: 0.551 GB / 42.829 M prompt tokens = 0.0129 KB/token = div 132**
— a very quiet fs-write window, not an extrapolation: only **7 files touched in
30 min (4 NEW / 0.10 GB + 3 REWRITES / 0.08 GB)**. Disk-served window share
0.35 % is below the 1.76 % floor because device serves 96.0 % and computed
prefill is 2.29 %; cumulative disk share is 1.94 %, still above the floor. No
new regression and no claim of utility loss: the engine is serving from fast
tiers, with almost no prefill work in this window.

Engine health: StartedAt still 21:48:52Z, RC=0, OOM=false, 0 OOM lines, 0
tiering asserts/tracebacks, three containers up, health 200 in 6.5 ms. **No
new WEDGE after 11:44:39Z**, fail 2/2 count 0, no watchdog restart. VRAM GPU0
22,487 / GPU1 21,508 MiB; GPU2 111 = idle/free (CoursIA expected). `/dev/shm`
25G/63G (39 %). Live scheduler: KV 44.5 %, 2 running, 0 waiting.

Traffic census (2 h, 548 req): 525 x 200, 22 x 401, **1 x 400** — 14:22:01Z,
hairpin, Bun, body 866,039 B, error `At most 4 image(s) may be provided in one
prompt. (parameter=image)`, **not window+1**. No large-context 400 this
window. ZOLTRAAK presented two unknown-key 401s at 15:27-15:28Z
(`ccd500f719e4`, `64a6d7ec06fd`), no access. Both Q40 mobile IPs remain only
2 x 200 each (no recurrence after 13:05Z); their `x_claudish_client` and
`x_claudish_machine` fields are empty. nbaudit healthy 15:06Z (Image/), G:
write OK.

RSM condensation warnings continue at 15:31-15:41Z (`key=MISSING`,
`OPENAI_BASE_URL ET OPENAI_CHAT_MODEL_ID absentes`), unchanged RSM-side issue
reported at 13:45Z; no engine error. Host commit **84.2 %** (340.6/404.6 GB,
margin 64.0 GB) — now below the 85 % heavy-work threshold, but no heavy job
needed in this cycle. Upstream PR vllm-project/vllm#63551 remains OPEN; latest
orozery comment (07:14Z) discusses `prepare_load` taking refs and a separate
`complete_read`; no rework here. PR #211 remains OPEN, unmerged and inactive
for the evening window. #198 remains OPEN.

### 26/10 17:41Z — cruise holds at div 19; third mobile IP joins Q40; RSM env restored

Window 15:41:27 -> ~17:43Z (~2 h 02). Read-only cycle; no engine/config change.
Evening patch window (user GO) remains after the EPF classes.

| metric | cumulative at 17:43Z | delta ~2 h 02 |
|---|---|---|
| fs writes | **164.648 GB** | **+3.858 GB = ~1.9 GB/h** |
| prompt tokens | 405.060 M | **+44.085 M = ~21.7 M tok/h** |
| fs reads | 93.492 GB | +1.260 GB = 0.62 GB/h |
| cached by source | device 290.670 M / host 77.541 M / disk 7.102 M | window: device **93.80 %** / host 1.23 % / disk 0.23 % |
| computed prefill | — | 2.087 M = **4.74 %** of window prompt |
| prefix cache | queries 405.137 M / hits 290.670 M | window hit rate **94.61 %** |
| preemptions | **27** | **+0 for ~7 h 45** |
| fs chunk hits | 3,443 | +48 |
| fs tier size | **251G** (`du`) | at the 250 GiB LRU cap |

**Write ratio: 3.858 GB / 44.085 M prompt tokens = 0.0875 KB/token = div 19.4**
— between the very quiet 15:41Z window (div 132) and the 13:41Z cruise (div
8.7): cruise regime, no extrapolation. The 30-min birth-time scan stays small:
12 touched files = **6 NEW / 0.16 GB + 6 REWRITES / 0.16 GB** (50/50).

**Disk-serving floor: cumulative 7.102/405.060 = 1.753 %, now marginally BELOW
the 1.76 % floor** (window 0.23 %). Reading: not a utility loss — device serves
93.8 %, fs reads still ran 0.62 GB/h with +48 chunk hits, and the cumulative
share decays mechanically as the filter reduces new writes. The floor criterion
(#198) needs a cold-window form; revisit at tonight's patch gate.

Engine health: StartedAt still 21:48:52Z, RC=0, OOM=false, 0 OOM lines, 0
tracebacks/asserts, three containers up, health 200 in 43 ms. **No new WEDGE
after 11:44:39Z** (fail 1/2 only, ever), 3,753 OK probes. VRAM GPU0 22,388 /
GPU1 21,508 MiB; GPU2 161 = idle. `/dev/shm` 25G/63G (39 %).

Traffic census (~2 h, 539 req): 516 x 200, 23 x 401, **0 x 400**. Security:
**third unclaimed mobile IP `92.184.146.164`, 3 x 200** (15:54:08Z 23,056 B,
15:57:58Z 54,414 B, 16:21:26Z 21,198 B, OpenAI/JS, key `be312daf7c1c` =
MEDIUM, attribution fields empty) — Q40 extended to three IPs (7 x 200 total).
No recurrence of `.46`/`.254` after 13:05Z. 401s all the known dead key
`1f7f6080cb65` loop. nbaudit healthy 17:06Z (Image/). G: write OK.

**RSM per-call env FIXED by roo-extensions (~16:20Z)** — the `.env` of the
server checkout had been deleted by their morning `file:`-junction cleanup;
restored from the canonical copy. Verified this side: last `key=MISSING` banner
16:21:09Z; banners 17:31->17:41Z show primary `localhost:5002/v1` key OK. ACK
replied. Side note: my own 17:15Z dashboard append hit the pass-deadline and
used the cloud reserve with deterministic truncation (RSM-side mechanics).

Host commit **85.6 %** (348.2/406.6 GB, margin 58.3 GB) — at the 85 % heavy-work
threshold; no heavy job from this lane. Deployment plan confirmed: engine-first
patch window tonight after EPF (PR #211 merge + deploy, gates, then GO/NO-GO to
claudish for the hub recreate), targeting the ~21:41Z cycle unless the user
signals earlier.

### 26/10 19:55Z — PATCH ANTI-RÉÉCRITURE DÉPLOYÉ EN PROD (PR #211) — écritures réelles ≈ 0

User GO (« tu as mon Go »), fenêtre ouverte après les cours EPF. Séquence tenue :
moteur d'abord, gates, puis GO/NO-GO au hub claudish.

**Déploiement.** PR #211 (`vllm/fs-skip-rewrite`) relue (diff complet, 0 commentaire,
0 review) puis mergée sur `main` (`fc8a0d1180`). Le patch vit dans
`myia_vllm/kv_tier/lru_fs_tier.py`, **bind-monté** dans le moteur
(`/opt/kv_tier`, `PYTHONPATH=/middleware:/opt/kv_tier`) → **aucun rebuild d'image,
aucun recreate** : un simple `docker restart` recharge le module. Image
(`vllm-openai-v0320-fsadmit:v1`), env, volumes et compile-cache **inchangés**.
6/6 tests unitaires verts dans le conteneur avant restart. Kill switch
`MYIA_FS_SKIP_REWRITE=0` (lu à l'import → recreate requis pour l'inverser).

**Boot.** StartedAt `19:54:57Z`, health 200 à `19:58:22Z` (**3 min 25 s**, warm,
compile-cache conservé), KV **831 723 tokens identique**, `RC=0`, OOM=false,
0 assert / 0 traceback / 0 « Chunk not found in cache », tier chargé depuis
`lru_fs_tier` (`Loading out-of-tree secondary tier 'TouchOnHitFsTier'`),
génération réelle OK (« OK », `enable_thinking:false`). `/dev/shm` : 48G → **24G**
après purge des orphelins antérieurs au StartedAt (le vieux `vllm_offload_*.mmap`
du boot précédent = 25,7 Go).

**Gates 19:55→20:27Z (32 min).**

| mesure | valeur |
|---|---|
| **écritures fs réelles** | **≈ 0** — 5,98 Go *soumis* au boot, **0 fichier `.bin` né**, aucun processus > 1 Mo écrit sur disque |
| fs lu | 10,0 Go (chunk hits fs 381, primary 421) |
| prefix cache | queries 8,895 M / hits 7,523 M = **84,6 %** (remontée post-restart) |
| disk-servi | 478 464 tokens (host 0 : le tier RAM se remplit) |
| préemptions | **0** |
| VRAM | GPU0 22 867 / GPU1 21 734 MiB, GPU2 99 (libre) |
| health | 200 à 4,8 ms |

**⚠️ Changement de métrique (important pour tous les cycles suivants).**
`vllm:kv_offload_tiering_write_bytes_total` est incrémenté par
`metrics.py::_observe_finished_job_stats` avec `transfer_size = len(job.keys) ×
chunk_size` : ce sont des **octets SOUMIS, pas écrits**. Upstream écrivait tout ce
qu'il soumettait ⇒ avant le patch, compteur = volume disque ; **après le patch il
surestime**. Références correctes désormais : (1) **date de naissance** des
fichiers du tier (une vraie écriture passe par `.tmp` + `os.replace()` ⇒ inode
neuf, naissance récente), (2) `write_bytes` des processus, (3)
`kv_offload_store_bytes` pour le tier RAM. Conséquence : la cible « ÷5 » de #198
doit se lire en **octets réellement écrits / token prompt** ; les ratios publiés
les 26/10 (÷8,7 / ÷19,4 / ÷132) restent valides *avant* le patch seulement.

*Note de méthode* : `ctime` ne distingue pas une réécriture d'un `utime` (le touch
LRU met aussi `ctime` à jour) — seule la date de naissance le fait.

**Conséquence pour #198** : le cas ciblé (réécrire un bloc déjà présent) est
**supprimé** — mesuré au boot (5,98 Go soumis, 0 écrit). La fenêtre froide reste
à mesurer sur le prochain restart. Commentaire publié sur l'issue (#198,
c.6426561040).

**Indisponibilité** : 19:54:57→19:58:22Z (3 min 25 s) — la clé externe
`external-vllm` (cap 2, zéro repli cloud) a pu rendre des 502 chez Jamin/Candy ;
annoncé à claudish.

**GO/NO-GO hub** : GO envoyé à `myia-po-2025:claudish`
(`vllm-gos-fs-patch-20261026T2030Z`, 20:27Z) — moteur sain, patch actif,
middleware `x-claudish-client`/`x-claudish-machine` recâblé côté moteur, leur
recreate hub autorisé.

---

## 26/10 20:41Z — Cycle de surveillance (fenêtre 19:55→21:43Z, moteur réchauffé 1 h 49) — écritures fs réelles 18,53 Go, **9,2 Go de réécritures supprimées** ; ⚠️ GPU 0 à 23 137 MiB (ligne d'alerte 23 000)

**Moteur.** StartedAt `19:54:57Z` (inchangé), health 200 à **5,2 ms** (21:43Z),
`RC=0`, OOM=false, 3 conteneurs up, 0 occurrence de
`out of memory|Traceback|AssertionError|not found in cache` sur 20 000 lignes,
`/dev/shm` **24G/63G (39 %)**, KV **831 723** tokens.

**Écritures fs — mesure par date de naissance (référence correcte depuis 19:55Z).**

| mesure | valeur |
|---|---|
| **fichiers `.bin` nés depuis le boot** | **706** ⇒ **18,53 Go réellement écrits** |
| fichiers relus/touchés seulement | 402 (10,55 Go lus) |
| compteur `tiering_write_bytes_total` | **27,72 Go = octets SOUMIS** (surestimation post-patch) |
| ⇒ **réécritures supprimées** | **27,72 − 18,53 = 9,19 Go (33 % des soumis)** |
| fs lu | 11,02 Go (chunk hits fs **420**, primary 9 291) |
| tier RAM (`kv_offload_store_bytes`) | 77,5 Go |
| ratio réel | **0,46 Ko/token prompt** (18,53 Go / 40,23 M tokens) |

Comparaison : pré-patch en fenêtre froide = 1,128 Ko/token ⇒ **÷2,4** dans un
régime comparable. La cible **÷5** de #198 reste **non démontrée** : elle exige
une fenêtre froide stricte (prochain restart), pas cette fenêtre réchauffée où
le tier RAM absorbe l'essentiel. **#198 reste ouverte.**

**Décomposition en deux régimes** : au boot (19:55→20:23Z) 5,98 Go soumis,
**0 octet écrit** (100 % sauté — les blobs étaient déjà là) ; 20:23→21:43Z,
21,74 Go soumis / 18,53 Go écrits ⇒ 3,21 Go sautés (14,8 %). Le gain du patch
porte donc sur les réécritures *d'un même contenu déjà sur disque*, pas sur la
première écriture — c'est exactement le périmètre demandé par l'user.

**Compteurs moteur.** prompt 40,231 M · device 26,863 M · host 10,543 M ·
disk 551 088 · prefix queries 40,440 M / hits 26,863 M = **66,5 %** (le moteur
réchauffe : 84,6 % au boot, la valeur remonte à chaque cycle) · **préemptions 0** ·
`prompt_tokens_total` 40,2 M en 1 h 49.

**Watchdog** : 3 793 sondes `OK health=200 decode=200`, un seul `WEDGE fail 1/2`
à 19:59:05Z (démarrage, compteurs frais — comportement attendu), **jamais
`fail 2/2`**, aucun restart.

**⚠️ VRAM — GPU 0 = 23 131 → 23 138 → 23 137 MiB** (3 relevés consécutifs,
21:40/21:41/21:43Z) ⇒ **plateau, pas une pente** (high-water d'allocateur en
charge) mais **1 427 MiB de marge seulement** et **138 MiB au-dessus de la
ligne d'alerte 23 000**. GPU 1 = 22 006 MiB, GPU 2 = 99 MiB (libre).
La spec impose `GPU_UTIL_SWIFT15=0.70` au prochain restart ; tout redémarrage
exige le GO user ⇒ **Q41 ouverte au registre** (rollback 0.73 → 0.70 au
prochain restart naturel, recommandation de la lane).

**Sécurité (census 2 h).** 528 requêtes : 509×200, 19×401 (clé morte connue
`1f7f6080cb65`, aucune tentative réussie), **0×400** (le remède #431 du hub
tient), IP avec 200 = hairpin `90.65.170.144` + l'entrée « ? » (direct) ⇒
**aucune nouvelle IP externe**. Tailles de corps : 285 entre 100K-1M, 215 < 1K,
28 entre 10-100K. Une seule clé en usage (MEDIUM).

**Hôte.** Commit **354,5/395,3 Go = 89,7 %** (marge 40,7 Go) ⇒ **au-dessus du
seuil 85 % : aucun travail lourd** (pas de passe Lean). G: écriture OK.
nbaudit `ok:true` 21:06Z (`Image/`). Condensation RSM : bannière 21:41Z
`primary=qwen3.6-35b-a3b @ http://localhost:5002/v1 key=VLLM_API_KEY_MEDIUM`,
fallback `key=OK` — battement nominal.

**Coordination.** GO hub toujours sans ACK à 21:40Z (claudish n'a pas encore
reçu l'ACK du recreate) — le moteur, lui, sert normalement et la clé externe
n'a pas degradation. Message Maintenance INFO (durcissement du lanceur de la
tâche `fstier-prune`, sans changement fonctionnel) lu et marqué lu.

---

## 26/10 23:41Z — Cycle de surveillance — ⚠️ **le pruneur planifié était MUET depuis son durcissement** (tier fs 41 Go au-dessus du plafond) ; hotfix local + escalade aux propriétaires

**Moteur (inchangé).** StartedAt `19:54:57Z`, health 200 à **4,7 ms**, `RC=0`,
OOM=false, 0 `out of memory|Traceback|AssertionError|not found in cache` sur
20 000 lignes, `/dev/shm` **24G/63G (39 %)**, KV **831 723**, 3 conteneurs up.
**VRAM** : GPU 0 = **23 149 MiB** (4ᵉ relevé : 23 131 / 23 138 / 23 137 /
23 149 ⇒ **plateau**), GPU 1 = 22 006, GPU 2 = 88 (libre). **Q41 reste ouverte**
(ligne d'alerte 23 000 dépassée, marge 1 415 MiB).

**Compteurs moteur.** prompt **84,89 M** · prefix queries 85,05 M / hits
54,84 M = **64,5 %** · device 54,84 M · **préemptions 1** (0 à 21:41Z, une
seule en 2 h) · fs lu **16,12 Go** (chunk hits fs/primary non relevés ce cycle)
· tier RAM `kv_offload_store_bytes` **167,9 Go** (77,5 Go à 21:41Z).

**Tier fs — écritures réelles depuis le boot (scan date de naissance).**
**1 570 fichiers nés = 41,21 Go réellement écrits** contre **50,29 Go soumis**
⇒ **9,08 Go supprimés au total** (cohérent avec les 9,19 Go mesurés à 21:41Z).

| fenêtre | soumis | réellement écrits | supprimé |
|---|---|---|---|
| boot → 21:41Z | 27,72 Go | 18,53 Go | 9,19 Go (33 %) |
| **21:41 → 23:41Z** | **22,57 Go** | **22,68 Go** | **≈ 0** |

**Lecture du résultat** : le patch anti-réécriture ne supprime que la
**réécriture d'un contenu déjà sur disque**. Au boot, le tier contenait encore
le contenu du run précédent ⇒ 100 % sauté ; en régime établi, le flux est du
**contenu neuf** (états GDN, nouveaux préfixes) ⇒ plus rien à sauter. Le gain
est donc **réel mais non permanent** : il ne se rejoue qu'après chaque restart.
Ratio courant **0,485 Ko/token prompt** ; la cible **÷5** de #198 exige de
frapper le flux *neuf* (filtre d'admission + v2 snapshots GDN), pas le patch.

### ★ Découverte : le pruneur planifié ne s'exécutait plus (et le census horaire non plus)

Le tier fs plafonnait à **309,6 Go** pour un cap de **268,4 Go** — et
`myia_vllm/logs/fstier_state.json` n'était plus écrit depuis **19:50Z** alors
que la tâche `fstier-prune` se déclenchait toutes les 30 min avec
**LastTaskResult = 0**.

**Cause racine (mesurée, pas supposée).** Le durcissement « fenêtre invisible »
(roo-extensions `scripts/scheduling/harden-hidden-tasks.ps1`) remplace l'action
de la tâche par `wscript.exe //B //Nologo <launcher>.vbs`, et le VBS fait
`WScript.Shell.Run(<commande d'origine **verbatim**>, 0, True)`. Quand l'`Execute`
d'origine contient un espace — `C:\Program Files\PowerShell\7\pwsh.exe` — le
chemin **n'est pas quoté** ⇒ CreateProcess résout `C:\Program` ⇒
**0x80070002 (fichier introuvable)** ⇒ **la tâche ne fait rien** et le
scheduler **rapporte 0**. Reproduction isolée (VBS de test) : variante non
quotée → `Err.Number = -2147024894` (= 0x80070002), `rc` vide ; variante
quotée → l'enfant s'exécute.

**Périmètre.** Sur `C:\ProgramData\maint-scripts\hidden-launchers\` (5
lanceurs) : `fstier-prune` (durci le **26/10 21:03**) et
`traffic-census-hourly` (durci le **20/10 17:28**) sont **cassés** ; les trois
autres sont sains (`powershell.exe` sans chemin, ou chemin sans espace). Le
census horaire est donc **mort depuis 6 jours** (dernier `last-result.json`
20/10 15:25Z) — exactement le piège que son propre script documente (« 08/10 :
37 h d'interruption silencieuse cachées derrière LastTaskResult=0 »).

**Correctif local appliqué** (lane vllm, sur ses **deux** tâches) : chemin de
l'exécutable **quoté** dans les 2 VBS + commentaire HOTFIX ; sauvegarde
`maint-scripts/_backup/20261027-vllm-launcher-fix/`. **Vérifié par
déclenchement réel des tâches** : `fstier-prune` ⇒ état écrit 23:51:17Z
(`mode=apply`, 0 erreur) ; `traffic-census-hourly` ⇒ `last-result.json`
**23:51:16Z, ok:true, rc 0** (census ressuscité). Une passe manuelle avait
ramené le tier de 309,6 à **268,4 Go** (au cap). **Escalade postée sur le
dashboard `global` avec cross-post roo-extensions (générateur) + Maintenance
(campagne de durcissement)** — le correctif amont (quoter l'`Execute` +
propager un exit code non nul) est demandé aux propriétaires, la lane vllm
garde ses deux tâches et #198.

**Watchdog** : 2 nouveaux `WEDGE fail 1/2` (21:54:01Z, 22:33:02Z) —
comportement attendu pendant les gros prefills ; **jamais `fail 2/2`**, aucun
restart, aucun BOOT-STALL.

**Sécurité (census 2 h, `x_forwarded_for`).** 596 requêtes : **576×200**,
**20×401** (19 sur la clé morte connue `1f7f6080cb65`, 1 sans clé),
**0×400**. IP avec 200 = hairpin `90.65.170.144` (315) + l'entrée « ? » (261,
direct interne) ⇒ **aucune nouvelle IP externe**. Corps : 269 en 100K-1M (45 %),
61 en 10-100K, 23 en 1-10K, 243 < 1K. Clé unique MEDIUM (empreinte
`be312daf7c1c`, schéma middleware).

**Hôte.** Commit **357,2/395,3 Go = 90,4 %** (marge 38,1 Go) ⇒ **au-dessus du
seuil 85 % : aucun travail lourd**. nbaudit `ok:true` 23:06Z (`Image/`).
G: **1ᵉʳ probe KO, 2ᵉ OK** ⇒ latence DriveFS transitoire, pas un hang (à
re-tester si ça se répète). Condensation RSM : bannière 23:41:08Z, primaire =
notre moteur avec la clé MEDIUM, fallback `key=OK`, **aucun échec**.

**Restant ouvert** : Q41 (VRAM), #198 (écritures : le flux *neuf* est le
levier), ACK du hub claudish (toujours sans réponse depuis 20:27Z).

---

## 27/10 01:45Z — Cycle de surveillance (T-55 min avant le reboot de la machine) — prod intacte, **pruneur et census confirmés en service**, charge en hausse

**Moteur.** StartedAt `19:54:57Z` (inchangé), health 200 à **4,7 ms**, `RC=0`,
0 `out of memory|Traceback|AssertionError|not found in cache` sur 20 000 lignes,
`/dev/shm` **24G/63G (39 %)**, KV **831 723**, 3 conteneurs up.

**VRAM.** GPU 0 = **23 594 MiB** (oscillation 23 581 → 23 622 → 23 594 autour de
**23 600** : le « plateau » des quatre relevés du 26/10 est bien réfuté, c'est un
**plafond de charge**, pas une fuite) ; GPU 1 = **22 278** (+272 depuis 23:40Z) ;
GPU 2 = 88 (libre). Marge GPU 0 ≈ **970 MiB**. **Q41 toujours ouverte** (aucune
réponse) ; la fenêtre naturelle d'application est le boot qui suit le reboot de
~02:40Z.

**Charge en hausse (à surveiller).** `prompt_tokens_total` 115,16 M (01:39Z) →
**117,03 M** (01:45Z) ⇒ ~**5,2 K tokens/s de prefill soutenus**. Conséquence
immédiate : **4 `WEDGE fail 1/2` en 30 min** (01:17:59, 01:24:10, 01:35:38,
01:42:20), tous à `health=200`, **jamais `fail 2/2`**, **aucun restart** ⇒ les
sondes de 90 s se font affamer pendant les gros prefills (comportement
documenté), mais la cadence est inhabituelle (1-2 par 2 h auparavant) : les
nouvelles arrivées attendent quand le prefill occupe le moteur. Prefix hit
**67,8 %** (79,33 M / 117,05 M, en hausse), **préemptions toujours 1**, tier RAM
`kv_offload_store_bytes` 261,7 Go, fs lu **22,65 Go**.

**Tier fs — écritures réelles (scan date de naissance).** **2 463 fichiers nés
depuis le boot = 64,65 Go réellement écrits** (41,21 Go à 23:41Z ⇒ **+23,44 Go
en 2 h**) contre **76,69 Go soumis** (+26,4 Go) ⇒ **~3,0 Go encore supprimés
(11 % du soumis)**. Correction de la lecture de 23:41Z : la fenêtre mesurée là
mélangeait le régime de chauffe (au boot, 100 % sauté) ; **en régime établi le
patch supprime encore ~1/9 des soumissions** — moins que le 33 % du boot, mais
non nul. Ratio courant **0,55 Ko/token prompt**. Le flux dominant reste le
**contenu neuf** ⇒ #198 (÷5) reste à instruire sur ce flux (admission + v2 GDN).

**★ Le pruneur et le census horaires sont confirmés EN SERVICE** (conséquence
directe du hotfix du 26/10 23:51Z) : pruneur **état écrit 01:20:03Z**
(`mode=apply`, 116 fichiers supprimés, tier lu à 271,5 Go) ; census
`last-result.json` **01:25:02Z `ok:true`, rc 0** → **6 jours de silence rompus**.
Le tier est passé de 11 797 à **10 618 fichiers** (~1 200 supprimés depuis le
hotfix) et il redescend sous le cap de 250 GiB à chaque passage.

**Sécurité (census 2 h, 493 lignes, span réel 120 min).** **474×200**,
**19×401** (clé morte connue `1f7f6080cb65` + 4 sans clé), **0×400**. IP avec
200 = « ? » (242, direct interne), hairpin `90.65.170.144` (230) et
**`82.66.89.184` (2×200)** = **l'IP parisienne du user** (Free, connue — Q13
répondue le 08/10 : « sans doute mon IP parisienne ») : **pas une nouvelle IP,
donc pas d'alerte**, mais son trafic était **arrêté depuis le 01/10 et reprend** —
cohérent avec un user de retour qui tape sur le modèle local. Clé MEDIUM unique
(empreinte `be312daf7c1c`).

**Hôte.** Commit **355,3/395,3 Go = 89,9 %** (marge 40 Go) ⇒ **au-dessus du
seuil 85 % : aucun travail lourd** (toujours pas de passe Lean). nbaudit
`ok:true` 01:06Z (`Image/`). **G: OK** (le probe KO de 23:41Z était transitoire,
il ne s'est pas reproduit). Condensation RSM : bannière **01:46:09Z**, primaire =
notre moteur + clé MEDIUM, fallback `key=OK` ⇒ nominal.

**Reboot ~02:40Z.** Rien à engager côté lane vllm : le prod est un conteneur
Docker qui repart avec l'autologon, le tier fs survit, le tier RAM se recharge
depuis le disque. La checklist de reprise est déjà écrite (dashboard
`workspace-vllm` + ligne MEMORY du 27/10 01:40Z) : re-armer le cron, vérifier
les 3 conteneurs / health / `StartedAt` / **VRAM GPU 0 après boot** / `/dev/shm`,
re-traiter Q41, et mesurer #198 sur le flux neuf.

---

## 27/10 02:35Z — Reprise après le reboot de la machine (boot 02:11:10Z) — moteur sain, **hotfix VBS confirmé survivant au reboot**, Q41 : le risque de boot-OOM **ne s'est pas matérialisé**

**Redémarrage.** Boot hôte **02:11:10Z**, conteneurs relancés par l'autologon,
moteur `StartedAt` **02:15:05Z** ⇒ **health 200 à 3,9 ms** (02:35Z), `RC=0`,
OOM=false, **0** `out of memory|Traceback|AssertionError|not found in cache`,
3 conteneurs up, `/dev/shm` **24G/63G (39 %)** = le tier RAM lui-même, **aucun
orphelin** (la fuite par boot ne s'est pas produite cette fois).

**Configuration au boot (inchangée, Q41 sans réponse).** Log de boot :
`Desired GPU memory utilization is (0.73, 17.51 GiB)` ⇒ **gpu-util 0.73**,
**GPU KV cache size: 831 723 tokens**, concurrency 3,17× à 262K ; tier fs chargé
(`Loading out-of-tree secondary tier 'TouchOnHitFsTier' from 'lru_fs_tier'`) et
**effectivement interrogé** (`lookup_sync_delay_seconds_count{tier="1:TouchOnHitFsTier"}` = 890).

**★ Le critère d'échec de Q41 n'est PAS rempli.** Le crash-loop de boot
redouté (historique 0.78/0.82) **ne s'est pas produit** : le moteur a démarré
du premier coup à 0.73, avec 22,25 GiB libres au moment du démarrage et un
calcul de KV conforme. Le risque « marge ~950 MiB en charge » reste **théorique
à ce stade** : il ne se traduit pas au boot parce que la VRAM en charge n'est
pas celle du boot (le pic vient des graphs sous trafic, après le calcul de KV).
**Q41 est donc re-qualifiée : risque non démontré, mais non levé** (le pic en
charge de 23 600 MiB reste 140 MiB au-dessus de la ligne d'alerte). À surveiller
au prochain pic de trafic ; pas de rollback d'office.

**VRAM après boot** : GPU 0 = **21 558 MiB**, GPU 1 = **20 724**, GPU 2 = 45
(libre) à 24 min, trafic encore faible ⇒ à comparer au plafond en charge de
**23 600** mesuré avant le reboot, pas à la bande ~22 200 de la spec (réfutée).

**Compteurs remis à zéro au boot** : prompt **167 638** tokens, prefix hits
**0**, `prompt_tokens_cached_by_source` tous à 0 (device/host/disk),
**préemptions 0**, tier RAM `kv_offload_store_bytes` 4,74 Go. **Le hit rate à 0
n'est pas une anomalie à ce stade** : le trafic post-boot est très faible
(167 K tokens en 20 min ≈ 140 tok/s, contre ~5 200 tok/s de prefill soutenus
avant le reboot) et les contextes sont neufs ; les deux tiers sont pourtant
interrogés (890 lookups fs, 3 676 primaires). **À re-mesurer au prochain cycle
sous trafic réel** — si le hit rate reste à 0 alors que les sessions ont repris,
c'est une régression à instruire (clé de bloc dépendante de l'instance).

**★ Hotfix des tâches planifiées : CONFIRMÉ SURVIVANT AU REBOOT.** Les deux
tâches ont tourné **après** le boot et ont **produit un livrable** :
`fstier-prune` état écrit **02:20:26Z** (`mode=apply`, tier à **268,4 Go** = cap,
0 fichier supprimé ⇒ no-op attendu sous le plafond) ; `traffic-census-hourly`
`last-result.json` **02:25:06Z `ok:true`, rc 0**. nbaudit `ok:true` 02:06Z.
C'est la preuve que le défaut `0x80070002` (chemin d'exécutable non quoté dans
le lanceur VBS) ne revient **pas** au redémarrage : les deux organes sont
durablement réparés sur cette machine. **Reste à obtenir le correctif
générateur côté roo-extensions** (sinon le prochain durcissement réintroduit le
défaut) — escalade du 26/10 23:52Z toujours valable.

**Cron de surveillance** : l'ancien job (`64a58306`) était **encore armé** après
le reboot (la session a survécu) ; il a été **remplacé proprement** (supprimé puis
recréé) pour garantir qu'il appartient à la session courante : **`d6ab14d5`,
toutes les 2 h à :41**, vérifié par `CronList` ⇒ **un seul job**.

## 27/10 04:15Z — Cycle : moteur sain sous trafic revenu, **prefix hits 92,3 %** (la question laissée ouverte au boot est répondue), wedge claudish réfuté côté moteur + **hairpin 443 KO depuis ai-01**, signature #431 mesurée, HF token → po-2023, **nbaudit v2 cadrée (#228)**

**Prod** : health 200 @2,4 ms, RC=0, StartedAt 02:15:05Z (aucun restart depuis
le boot), OOM 0, 3 conteneurs Up, VRAM **GPU0 22 076 / GPU1 21 378 MiB en
charge** (sous la ligne d'alerte 23 500 — Q41), GPU2 98 MiB (CoursIA libre),
`/dev/shm` 24G/63G (nominal), watchdog **3 795 sondes OK**, 2 `WEDGE fail 1/2`
(03:03 et 03:11Z = famine prefill pendant des corps ~0,9 Mo en vol, jamais 2/2),
**préemptions 0**, marge commit hôte **69,7 %** (268,1/384,4 Go — le reboot a
vidé tmpfs/WSL ; très sous le seuil 85 %).

**★ Hit préfixe 92,3 %** : `prefix_cache_hits_total` 19 697 504 /
`prefix_cache_queries_total` 21 347 482 depuis le boot, sous le trafic revenu
(~4,1 req/min sur la fenêtre census). Le point ouvert du cycle 02:35Z (« hits 0,
non anormal post-boot, à re-mesurer sous trafic ») est **répondu :
l'amortissement des préfixes est sain** à 0.73 + tiers RAM 24 + fs. Le moteur a
traité 21,3 M tokens prompt en 2 h avec 92 % servis du cache.

**Pruneur + census** : `fstier_state.json` 03:50:03Z mode apply — tier 268,797 Go
> cap ⇒ 14 fichiers supprimés (367 Mo) ⇒ 268,43 Go = cap, 0 erreur, 2,66 s.
Census `ok:true` 03:25:01Z. Les deux organes tiennent le régime post-reboot.

**Sécurité** : 3 IP avec 200 sur la fenêtre — site hairpin `90.65.170.144`
(783), interne sans XFF (782), poste user Paris `82.66.89.184` (2, trafic
repris). **Aucune IP inconnue.** 114×401 = clé morte `1f7f6080cb65` en boucle
(connu, process flotte).

**★ Signature #431 mesurée (400 + corps >100 Ko)** : **29 occurrences sur 4
jours** (~7/jour, fichier 21 877 lignes du 23/10 10:34Z au 27/10), corps
**0,26-4,2 Mo**, quasi toutes `Bun/1.3.14` (sessions Claude Code via hub) + 2
`AsyncOpenAI/Python` (sk-agent vision, 3,1/4,2 Mo). Paires même-corps à 10 s
d'écart = **retry local** alors que #431 demande le reroutage cloud dans le même
tour. Rendu au user (rapport cycle) et à claudish (dans la réponse wedge). Les 2
sk-agent vision sont à relayer au propriétaire sk-agent (carnets Image/ base64
en corps entier).

**Signalement claudish « swift ECONNRESET wedge depuis 01:47Z » — réfuté côté
moteur, un vrai point ouvert trouvé** : leur rapport (canaries 7/7 à ~21 s,
8+ 400, 187 s « UNEXPLAINED ENDING », 502 IIS 01:53-01:56Z) se réconcilie
ainsi : (1) 01:47-02:15Z = **reboot hôte** (attendu, 502 compris) ; (2) les 400
= débordement de fenêtre ci-dessus ; (3) les 187 s et canaries = **famine
prefill documentée** (dense 27B, WEDGE fail 1/2 vu au watchdog aux mêmes
minutes). Le moteur n'a **pas** été recréé à 03:34Z (aucun StartedAt déplacé) ;
queue 0 running/0 waiting, KV usage 0 %, et **38×200 Bun depuis 03:34Z** — leur
trafic passe majoritairement. **Mesuré en revanche : hairpin 443 KO depuis ai-01
lui-même** — 3/3 `curl https://api.medium…myia.io/health` → `000` à 21,1 s (aucun
TCP ; A record 90.65.170.144 uniquement, W3SVC Running). Le ~21 s correspond
exactement à leurs canaries. Discriminant demandé : cible exacte (host:port)
des canaires — si c'est l'URL publique, le défaut siège **routeur/IIS, pas
moteur**. Réponse envoyée 04:17Z (thread
`po2025-swift-econnreset-wedge-20261027T0412Z`).

**HF token → po-2023:vllm** : demande légitime du siège embeddings (conteneur à
token vide, cache local seul = bloquant si purge+recreate). Envoyé par **DM
pièce jointe auto-détruite 24 h** (fp `769d40e66497`, valeur jamais dans le
corps), avec la précaution quota compte user + l'avertissement rotation
pendante. Fichier scratchpad supprimé après envoi.

**★ nbaudit v2 cadrée** — [vllm#228](https://github.com/jsboige/vllm/issues/228)
(mandat user 27/10 « je suis pour poursuivre le travail ») : le producteur est
sain mais **0 livraison depuis le 21/10 23:16Z** (4 séries entièrement auditées
par les bots, 83 carnets #19451 hors périmètre `Image/` ⇒ `0 picked`) ; en
parallèle une session ChatGPT anonyme + 2 agents CoursIA ont produit **14 issues
en 15 min** (#22836-#22855) sur 10 carnets échantillonnés. Spec v2 : (a) unité =
finding à **évidence verbatim** (chemin + cell_id + quote + claim) ; (b)
**`propose_issue` validé** = seul chemin d'écriture (contrôle mécanique de la
quote contre le HEAD + dedup + cap 10/h + kill switch), **pas de `gh` dans les
mains de l'agent** ; (c) base préfixe réordonnée — mesuré : le chemin du carnet
ouvre le prompt, donc le préfixe partagé = system prompt seul (5 611 caractères
≈ 1 400 tokens) → viser 15-30 K (méthode + règles + catalogue faux positifs) ;
(d) **tapis roulant sur TOUT le catalogue** (mandat user : « il fait son
chemin »), séries contiguës, boucle auto-relancée, PR-collision reporté en fin
de boucle, dedup à l'émission. PRs = P5, pas avant. Q42 close au registre.

**Registre** : Q41 (VRAM/0.73) reste ouverte — mesure du cycle : 22 076 MiB en
charge, sous la ligne d'alerte ; le pic à re-mesurer reste le critère. Q42 close
(poursuite du travail + v2).

## 27/10 06:1xZ — hairpin RÉTABLI, breaker condensation = récurrence client connue, clé inconnue en 401, nbaudit toujours 0 picked

**Moteur sain, 4 h sans restart.** health **200 @3,8 ms**, RC=0, `StartedAt`
02:15:05Z inchangé (aucun restart depuis le boot), 3 conteneurs Up, OOM 0,
`/dev/shm` **24G/63G (39 %)** = le tier lui-même, aucun orphelin. **VRAM GPU 0
22 347 → 22 354 MiB en charge** (GPU 1 21 650, GPU 2 98 libre) : sous la ligne
23 500, donc Q41 reste non tranchée — le pic de 23 600 mesuré avant reboot n'est
pas reproduit ce cycle.

**Amortissement** : prefix **90,6 %** (40,58 M / 44,78 M), **external tier hit
29,4-33,0 %**, tier fs **lu 11,52 Go** / soumis 13,41 Go (rappel : `write_bytes`
compte les octets SOUMIS), **439 chunk-hits fs** vs 2 137 primary. **Préemptions
0**. Watchdog **3 801 sondes OK**, **aucun WEDGE depuis 03:11Z** ; les deux
`fail 1/2` de 03:03/03:11 sont ceux de la famine de re-prefill post-boot déjà
documentée, jamais 2/2.

**Hôte** : marge commit **73,5 %** (102 Go libres, seuil 85 % non franchi),
G: écriture OK (579 Go libres), pruneur **05:50:02Z** (76 fichiers / 1,99 Go
supprimés, 0 erreur, tier ramené au cap), census `ok:true` 05:25Z — les deux
tâches du hotfix VBS toujours vivantes après le reboot.

### ★ Hairpin 443 : RÉTABLI, et cible LAN désignée

Mesure depuis ai-01 06:28Z : `https://api.medium.…myia.io/v1/models` →
**401 en 0,030 / 0,007 / 0,017 s**, contre **000 en 21,1 s ×3** hier 03:50Z
(aucun TCP). po-2025:claudish mesure la même résorption de son côté (entrée
`models.myia.io` 200 en 0,53 s ; endpoint swift 401 en 0,3 s depuis l'hôte,
401 immédiat depuis le conteneur hub). **Les deux extrémités concordent ⇒ classe
ECONNRESET close, moteur exonéré des deux côtés**, porte R de #704 reprenable.
Claudish a reclassé le durcissement en « optionnel structurel », pas en « fix
d'une panne active » — d'accord, et tracé dans la réponse.

**Cible LAN garantie livrée** : `http://192.168.0.47:5002/v1`
(`/health` → 200 en 3,9 ms ; le port écoute **`0.0.0.0:5002`**, donc toute la
LAN). **Piège mesuré, à ne pas manquer** : `myia-ai-01.home` résout en **4
adresses** — 2 IPv6 globales, l'IPv4 `192.168.0.47`, **et une APIPA
`169.254.227.197`** — alors que le moteur **n'écoute qu'en IPv4** (`netstat` :
`0.0.0.0:5002`, aucune ligne `[::]:5002`). Un client qui préfère IPv6 ou tombe
sur l'APIPA échoue sur un service sain ⇒ **littéral IPv4**, ou nom avec
préférence IPv4 forcée. Reste non vérifiable d'ici : que `.47` soit une
réservation DHCP (si non, le littéral se périmera en silence).

### Breaker condensation ouvert 2× — récurrence d'un défaut CLIENT connu, pas un fait neuf

`Condensation circuit breaker OPENED` à **04:35:40Z** et **06:01:22Z**
(`consecutiveFailures: 3`, reset 30 min) → repli cloud, **dégradation
gracieuse**. Déclencheur lu avec son contexte : `LLM status update timeout:
89 781 ms` à 06:00:53Z. **Le moteur n'est pas en cause et je le montre** : sur
la fenêtre du timeout (05:56-06:01) KV **14-27 %**, `Running` 1-4, `Waiting` 0
sauf une fois `Deferred: 1`, aucun empilement ; à 06:2xZ deux petites
complétions répondent **0,21 s / 0,35 s**. Et la cause est **déjà diagnostiquée
au dashboard** : budget de génération côté client ~4,1 K tokens = 100-200 s >
timeout 90 s, résumé 0,5-1 K = 12-50 s passe, **moteur disculpé (file p90
0,8 s, TTFT p90 20 s)**, suivi po-2028 [#3962].

⇒ **Ce n'est donc pas un fait nouveau et je ne l'escalade pas.** La leçon du
21/10 s'applique exactement : sans lire le contexte, ce WARN se serait relayé
comme une panne prod, alors qu'il est la signature attendue d'un défaut client
en cours de correction ailleurs.

### Observation à instruire : JIT Triton pendant l'inférence (récurrent)

**21 warnings `jit_monitor` depuis le 25/10**, dont ce cycle
`_fused_mm_input_norm_kernel` à **05:47:38** et `_topp_sb_{stats,step,mask}`
à **03:50:30**. Le moniteur est en `jit_monitor_mode=warn` et le message est
explicite : « This causes a latency spike ». `enable_jit_warmup=True` est
pourtant armé — **il ne couvre donc pas ces formes** (sampling top-p, fused
mm+norm). Piste concrète et non engageante : étendre le warmup à ces
shapes/configs au boot, plutôt que de compiler la première fois sous trafic.
Non instruit plus loin ce cycle (aucune preuve que ces JIT soient la cause du
timeout de 06:00 — l'écart est de 13 min).

### Sécurité — une empreinte de clé inconnue, en 401 seulement

1 295 requêtes / 6 h : **1 234×200, 59×401, 2×400**. Clés : MEDIUM
`be312daf7c1c` **96 %** ; clé morte connue `1f7f6080cb65` 44 ; `NONE` 13 ;
**★ nouvelle empreinte `b72157f4bb61` = 2 requêtes**, toutes deux **401** sur
`/v1/models` en `curl`, **locales** (aucun `x_forwarded_for`), à 03:27:13Z et
05:45:09Z.

Attribution : la formule du middleware est
`auth_prefix = authorization[:24] + "..."` ⇒ `sha256("Bearer " + clé)[:12]`.
Recalculée sur `.env`, elle redonne bien `be312daf7c1c` pour **MEDIUM** (formule
confirmée) et **aucune** des trois clés (`MEDIUM`, `MINI`, `MICRO`) ne produit
`b72157f4bb61`. ⇒ **consommateur local détenant une clé morte**, très
probablement l'ancienne clé MEDIUM d'avant la rotation du 30/09. Rejeté, **pas
une intrusion** ; profil = sonde `/v1/models`, pas de trafic d'inférence. À
surveiller s'il passe en rafale (ce serait alors un process à reconfigurer).

**IP externes avec 200** : `90.65.170.144` (486 — hairpin/site, **connue**) et
`82.66.89.184` (2 — poste Paris du user, **connue**). **Aucune IP inconnue.**
Corps : **420 / 1 295 en 100K-1M** (32 %), 0 au-delà de 1 Mo ⇒ le long contexte
domine toujours.

### nbaudit : périmètre `Image/` armé, et toujours 0 picked

`last-result.json` 06:05:59Z `ok:true`, rc 0, **`subpath: ["19451:Image/"]`**.
Mais le log horaire dit, à chaque passage :
`#19451 NanoClaw: 83 to audit (129 audited in comments), 6 fresh ahead, 0 picked`
— et les 8 records régénérés sont **jetés** (`record(s) superseded, already
audited by the bot`). Les 4 autres séries sont à 0 (périmètre épuisé). Bilan :
**sain, et toujours improductif** ⇒ confirmation directe du diagnostic de
[#228](https://github.com/jsboige/vllm/issues/228) : le `subpath` n'est pas le
levier, c'est **l'unité de travail** qui doit changer (finding vérifiable +
tapis roulant). Le « 83 to audit → 0 picked » reste le point à comprendre au
moment d'écrire P1/P4, pas avant.

**Ouverture (hors lane)** : po-2023 a son Docker Desktop briqué (session 0,
00:17Z) — service embeddings public en 502. Rien à faire depuis vllm (infra d'un
autre workspace : demander, pas appliquer).

**Registre** : Q41 (VRAM/0.73) toujours ouverte seule — aucun critère d'échec
rempli ce cycle (22 354 MiB en charge, sous 23 500), risque non démontré mais
non levé. Cron réarmé 2 h, un seul job.

## 27/10 08:1xZ — 6 h sans restart, JIT warmup = no-op mesuré, deux clés mortes identifiées, breaker auto-rétabli

**Moteur** : `StartedAt` 02:15:05Z **inchangé** (6 h), health 200 @5,2 ms puis 2,8 ms,
RC=0, 3 conteneurs Up, OOM 0, **0** signature d'erreur (Traceback/RuntimeError/
AssertionError/CUDA error) sur 20 000 lignes, `/dev/shm` **24G/63G (39 %)** — le tier
RAM lui-même, aucun orphelin. **VRAM échantillonnée 6× en 1 min** : GPU 0
**22 343→22 358 MiB** (stable, pas de pente), GPU 1 21 650, GPU 2 98 (libre).

### Amortissement des préfixes : 90,09 %, et le tier fs monte

`prefix_cache_hits/queries` = **63 652 800 / 70 653 330 = 90,09 %**. Mieux : le taux
**marginal** sur les 2 h écoulées depuis le cycle précédent (44,78 M → 70,65 M queries,
40,58 M → 63,65 M hits) vaut **23,07/25,87 = 89,2 %** ⇒ l'amortissement ne se dégrade pas
avec la charge, il est stationnaire.

**Tier fs** : `external_prefix_cache_hits/queries` = 3 155 888 / 7 000 533 = **45,1 %**
(contre 29,4-33,0 % à 06:15Z) ⇒ le disque prend une part croissante du service.
**17 995 lookups** `tier=1:TouchOnHitFsTier` depuis le boot (890 à 02:35Z), latence
p50 ≈ **20 µs** — la lecture de l'index fs ne coûte rien.
Offload : GPU→CPU 163,6 Go écrits, CPU→GPU **65,4 Go relus**, 168 transferts.
**Préemptions 0.**

### ★ `enable_jit_warmup=True` chauffe 0,00 s — la cause des JIT sous inférence est mesurée

Le log de boot le dit sans ambiguïté, deux fois (TP0 et TP1) :

```
[02:17:44] kernel_warmup.py:176 JIT kernel warmup starting.
[02:17:44] kernel_warmup.py:189 JIT kernel warmup finished in 0.00s.
```

**Le warmup ne fait littéralement rien sur ce couple modèle/build.** C'est la
démonstration du point laissé ouvert au cycle précédent (« le warmup ne couvre pas ces
formes ») : il ne couvre **aucune** forme. Conséquence mesurée : 11 avertissements
`jit_monitor` de compilation Triton **pendant l'inférence** depuis le boot
(`_topp_sb_{stats,step,mask}_kernel` à 03:50:30, `_fused_mm_input_norm_kernel` à
05:47:38), chacun « causes a latency spike ». **Aucun nouveau depuis 05:47:38** (2,5 h) :
le jeu de formes rencontrées est stable, chaque kernel se paie une fois.
Piste à instruire = peupler le warmup pour ces formes, ou accepter le coût unique.

### Breaker de condensation : rétabli seul à 07:39:23Z

Aucune nouvelle ouverture après 06:01:22Z (les lignes 06:04/06:07/06:12/06:16 sont des
« primary skipped » dans la même fenêtre de refroidissement). **HALF-OPEN à 07:38:13Z →
`LLM status update generated` à 07:39:23Z** ⇒ le disjoncteur se referme de lui-même dès
que le moteur répond, sans intervention. Dernier timeout observé : 06:00:53Z.
Confirme que la boucle est **gracieuse et auto-guérissante**, et que la cause reste
côté client (budget 4,1 K tokens > timeout 90 s, po-2028 #3962).

### Sécurité : deux empreintes mortes, aucune n'a jamais rendu 200

Census 7 h — IP avec 200 : `?` (1 105, appels internes sans `X-Forwarded-For`),
`90.65.170.144` (521, hairpin/site **connue**), `82.66.89.184` (3, poste Paris du user
**connue**). **Aucune IP inconnue.** Fenêtre 3 h : 994 lignes, **5,5 req/min**,
**955×200 / 39×401 / 0×400** (aucune signature #431).

Empreintes (`sha256(auth_prefix)[:12]`) : MEDIUM `be312daf7c1c` 955 · clé morte connue
`1f7f6080cb65` 22 · `NONE` 14 · **`2703beff17f8` 2 (nouvelle)** · `b72157f4bb61` 1.

**★ Discriminant décisif, mesuré sur tout le fichier** : sur l'intégralité du journal
(depuis le 23/10), ces trois empreintes n'ont **jamais** obtenu autre chose qu'un 401 —
`1f7f6080cb65` 832×401, `b72157f4bb61` **3×401**, `2703beff17f8` **2×401**. Ce sont donc
des **clés mortes pures**, pas des consommateurs légitimes dégradés : elles n'ont jamais
eu d'accès. `2703beff17f8` a tiré ses 2 requêtes **dans la même seconde** (06:45:46Z)
sur `/v1/models` en `curl/8.14.1`, local, corps 0 o — profil de **sonde de santé avec un
jeton périmé**, pas d'inférence. Aucune n'est une intrusion, aucune n'expose de donnée
(refus en 401 avant tout traitement).

### Hôte, tâches, coordination

Marge commit **76,1 %** (292,5/384,4 Go, 92 Go libres) — remontée depuis 73,5 % mais
sous le seuil 85 %. G: 85 % utilisé / 592 Go libres, **écriture OK**. Pruneur
**07:50:02Z** : `mode=apply`, tier ramené au cap (268,74 → 268,43 Go), **12 fichiers /
315 Mo supprimés, 0 erreur** — le volume de réécriture reste très bas depuis le patch.
nbaudit `ok:true` **08:05:44Z**, `subpath Image/` inchangé. **WEDGE** : dernier à
**03:11:31Z** (5 h), tous en `fail 1/2`, **jamais 2/2**, aucun restart ; 3 803 sondes OK.

**Coordination** : po-2025:claudish a **indépendamment confirmé la fermeture** de la
classe ECONNRESET — sur `n=125` mesures `[ttft]` swift post-rétablissement : **0 échec de
connexion, 0 UNEXPLAINED ENDING**, ttft p95 8 933 ms, p50 1 816 ms ; leurs 4 UNEXPLAINED
sont tous dans la fenêtre wedge 03:40-04:08Z. Ils **ne retiennent pas la cible LAN cette
nuit** (changement de baseUrl gelé pour l'arbitrage du matin) et s'engagent à me le dire
dans le même cycle s'ils la retiennent ⇒ c'est à ce moment que je demande la réservation
DHCP de `.47`. Le point est donc **clos des deux côtés**, sans action en attente de mon
côté.

**Hors lane (RSM, à relayer — ne pas traiter ici)** : `[PG-PRIMARY #4131] écriture PG
primaire échouée — repli sur le chemin GDrive-first` (07:36:08Z) ; `[STALE-FILE #3230]
copie locale en retard` (07:21:15Z) ; `[ERROR] Status contradictions detected after LLM
generation (#1502/#3329)` (07:39:23Z).

**Registre** : Q41 seule, toujours ouverte. Le pic de charge de ce cycle est
**22 358 MiB** — 6 h de trafic soutenu à 0,73 ne reproduisent pas le plateau de
23 600 MiB d'avant reboot. Critère d'échec toujours non rempli, risque non démontré.

## 27/10 10:11Z — 8 h sans restart, VRAM GPU 0 au plus haut du jour (22 505), #431 : le 400 ne se re-tente plus en local, relance #66 traitée

**Moteur** : `StartedAt` 02:15:05Z **inchangé (8 h)**, health 200 @3,7 ms, RC=0, 3 conteneurs,
OOM 0, **0** signature d'erreur, `/dev/shm` 24G/63G (39 %). **Dernier WEDGE 03:11:31Z (7 h)**
— tous `fail 1/2`, aucun restart, aucune nouvelle alerte JIT depuis 05:47:38 (7 h : le jeu
de formes est stable), aucun événement breaker depuis le HALF-OPEN recovered de 07:38:13.

**VRAM GPU 0 = 22 505 MiB** — plus haut relevé du jour (série : 22 076 → 22 354 → 22 358 →
22 505). Toujours ~1 000 MiB sous la ligne 23 500 ; quatre relevés montent d'environ
+150 MiB / 2 h mais la charge varie entre les cycles — **surveillé, pas extrapolé** (leçon
« deux points ne font pas une pente » appliquée à mes propres quatre points : je note la
série, je n'annonce pas de date de franchissement). GPU 1 21 650, GPU 2 45 (libre).

### Amortissement : stationnaire à 90 %

`prefix_cache_hits/queries` = 77 812 480 / 86 345 823 = **90,11 %** ; taux **marginal sur
les 2 h** = 14,16/15,69 M = **90,2 %** (89,2 % au cycle précédent) ⇒ stationnaire, il ne
se dégrade pas quand la charge monte. Tier fs **44,2 %** (8 533 343 queries externes,
3 771 824 hits) — stable. **Préemptions 0.** Prefill soutenu : +15,7 M tokens prompt en 2 h
≈ **2 180 tok/s**, pour +138,5 K tokens générés — profil condensation (gros préfixes,
sorties courtes), conforme au workload pilote.

### #431 — le 400 de débordement de fenêtre ne se re-tente PAS en local

**3×400 aujourd'hui** (03:19, 03:53, 09:30Z), 4 sur 24 h glissantes, **toutes avec corps
> 100 K** — la signature fenêtre+1 (session Bun/claude-code via hub qui croît au-delà de
262 144). Analyse du 400 de **09:30:12Z (corps 953 523 o)** : la requête précédente de la
même session (09:27:19Z, 923 071 o) était passée en 200 ; **aucune re-tentative du même
corps** ensuite — la requête suivante arrive **19 min plus tard avec un corps plus petit**
(902 042 o = auto-compact côté client), puis la session recroit (912 296 o à 10:09:27Z).
C'est le comportement **voulu** par la règle #431 (« sur 400, jamais de retry local ») :
le coût d'un franchissement = une requête morte + un auto-compact, pas une boucle. Ne
prouve pas le routage cloud (invisible côté moteur) mais **l'absence de boucle locale est
mesurée**. Historique : les paires même-corps ≤ 60 s existent dans le journal (ex. 942 554
× 2 à 10 s d'écart, 901 417 × 2) mais **plus depuis le 26/10 au soir**.

### Sécurité

**Aucune IP inconnue** (7 h : `?` 1 690 internes sans XFF, `90.65.170.144` 504 hairpin
connue, `82.66.89.184` 2 poste Paris connue). Fenêtre 3 h : 1 072 lignes = **1 057×200 /
14×401 / 1×400**. Empreintes : MEDIUM `be312daf7c1c` 1 058 · clé morte `1f7f6080cb65` 12
(descend, 22 au cycle précédent) · NONE 2. **Aucune empreinte inconnue nouvelle** — les
deux clés mortes identifiées au cycle précédent (`b72157f4bb61`, `2703beff17f8`) ne
réapparaissent pas.

### Relance #66 (NanoClaw) — traitée

Issue **jsboige/vllm#66** (« Sessions sk-agent longue durée : registre, reprise,
persistance et service de condensation », ouverte par le user le 05/10, **22 jours, 0
commentaire**). Réponse déposée sur l'issue
([commentaire 6439190989](https://github.com/jsboige/vllm/issues/66#issuecomment-6439190989))
+ REPLY intercom à NanoClaw. Verdict : **(b) partiellement couvert** — la moitié économique
(tier RAM+fs prod, reprise au restart prouvée) est déployée, l'inventaire demandé est fait
(sk-agent expose `list_conversations`/`run_conversation`/`call_agent`, sans persistance
au-delà du process), le volet condensation est supplanté par RSM + #198 ; **le registre de
sessions et le protocole consommateur restent sans porteur**. Décision garder/recadrer/
fermer laissée au user sur l'issue. NB : nbaudit v2 (#228) a choisi le modèle **inverse**
de la session longue (sessions fraîches courtes + préfixe socle partagé servi par le KV
cache) — c'est le fs tier qui rend ce modèle économique.

### Hôte / tâches

Marge commit **76,2 %** (293,0/384,4 Go) — stable. G: 85 % / 592 Go libres, **écriture OK**.
Pruneur **09:50:02Z** : `mode=apply`, tier 269,1 Go → cap, **24 fichiers/630 Mo supprimés,
0 erreur**. nbaudit `ok:true` **10:05:49Z**, `subpath Image/`.

**Registre** : Q41 seule — le relevé de ce cycle (22 505 MiB) est le plus haut du jour mais
reste sous la ligne 23 500 ; critère d'échec toujours non rempli.

## 27/10 12:11Z — hit marginal 96,85 %, VRAM à 417 MiB de la ligne, attribution claudish câblée mais VIDE

Moteur `StartedAt` **02:15:05Z inchangé (10 h)** : health 200 @4,9 ms, RC=0, 3 conteneurs,
OOM 0, 0 signature d'erreur sur 20 000 lignes, `/dev/shm` **24G/63G (39 %)**. **Dernier WEDGE
03:11:31Z (9 h)**, tous `fail 1/2`, **jamais 2/2**, aucun restart ; 3 826 sondes OK.

**VRAM GPU 0 = 22 583 MiB**, nouveau plus haut du jour. Échantillonnée **6× en 1 min =
22 568→22 583 MiB** (plate, aucune pente en intra-minute). La série inter-cycles monte
cependant sans discontinuer : 22 076 (04:20Z) → 22 354 → 22 358 (08:11Z) → 22 505 (10:11Z)
→ **22 583** (12:11Z), soit +507 MiB en 6 h. **Il reste 417 MiB sous la ligne d'alerte
23 000** — c'est le point à surveiller, pas encore une alerte : la charge varie entre les
relevés et l'intra-minute est stable. GPU 1 21 650, GPU 2 45 (libre).

**★ Amortissement en nette hausse** : prefix hit **92,26 %** cumulé (116,91 M / 126,72 M)
contre 90,11 % au cycle précédent. Surtout, le taux **marginal sur les 2 h = 96,85 %**
(39,10 M hits sur 40,37 M queries) — il était à **90,2 %** à 10:11Z. L'amortissement ne se
contente pas de tenir, il **s'améliore** : les sessions récurrentes retrouvent leurs préfixes
sans re-prefill. Prefill **5 607 tok/s** soutenus (+40,4 M prompt en 2 h) pour 1,12 M tokens
générés au total ⇒ profil condensation confirmé (gros préfixes, sorties courtes).
**Préemptions 0** (discriminant de saturation KV, plus fiable que la VRAM). Requêtes :
2 719 succès (1 525 `stop` + 1 194 `length`), 0 `abort`, 0 `error`, 0 `repetition`.

**Tiers** : offload **290,3 Go GPU→CPU** (10 437 transferts) / **90,4 Go CPU→GPU** (284
transferts), ratio de relecture **31 %** — le tier RAM recycle beaucoup (24 GiB, écritures en
place) tandis que le fs n'absorbe que les évictions. Tier fs **44,2 %** du cap 250 GiB.

**★ Datapoint claudish du 10:44Z (timeout 90 s sur le frontal) — NON corrélé au moteur.**
Leur auto-condensation po-2025 a rendu « TIMEOUT après 90 s, socket ouverte, ni completion ni
erreur HTTP ». Mesures de la fenêtre 10:41-10:50Z : sondes watchdog **OK health=200
decode=200 à 10:44:17Z et 10:44:48Z**, trafic Bun en 200 continu (`10:44:11Z` 592 485 o,
`10:44:42Z` 596 710 o), **préemptions 0 sur tout le cycle**. Surtout, sur les 133 requêtes de
10:30-11:05Z, **aucune n'est attribuable à leur appel** (les seules `OpenAI/JS` sont à
10:36:02, 10:37:14, 10:38:16, 10:39:19, 11:01:19 et 11:02:23 — aucune à 10:44Z ; les entrées
`?` sont toutes `curl/8.22.0` 166 o = nos sondes). ⇒ **la requête n'a pas atteint le moteur** ;
le blocage est sur le chemin externe (hairpin `api.medium…:443` → IIS), pas dans l'inférence.
Classe « condensation seule / chemin externe », à verser au dossier Porte R de #704.

**★ Attribution machine:workspace de claudish : CÂBLÉE DANS LE MIDDLEWARE, VIDE EN PRATIQUE.**
Le middleware `error_source_capture` émet désormais les champs `x_claudish_machine` et
`x_claudish_client` (plus `x_real_ip`, `referer`, `x_forwarded_host`). Or sur
**24 456 lignes de journal — l'intégralité du fichier — les deux champs sont vides à 100 %**
(0 non-vide), et `referer`/`x_real_ip` aussi. La demande user du **25/10** (« identifier le
poste machine:workspace de toutes nos requêtes, vois avec claudish ») n'est donc **pas
effective au niveau du moteur** : soit le hub ne les envoie pas encore, soit il les retire.
L'attribution reste donc **indirecte** (taille de corps + IP + UA), avec la limite connue —
la clé MEDIUM ne distingue pas les clients. À relayer à po-2025:claudish comme une demande
précise et vérifiable (un seul champ à peupler), pas comme un reproche.

**#431** : aucun nouveau 400 depuis 09:30:12Z (2 h 40) ; le seul 400 de la fenêtre 3 h est
celui-là, corps 953 523 o, non re-tenté (analyse du cycle 10:11Z confirmée). 3 h :
**1 211×200 / 30×401 / 1×400**.

**Sécurité — aucune IP inconnue avec un 200.** 7 h : `?` 1 691 (interne sans XFF),
`90.65.170.144` 631 (hairpin, connue), `82.66.89.184` 6 (poste Paris, connue). L'IP
`108.165.33.236` relevée à 10:47:47Z **n'est pas nouvelle** : 858 requêtes sur tout le
journal, **0 × 200**, UA `Go-http-client/1.1`, `/v1/models` toutes les ~6 min depuis le 23/10,
empreinte `1f7f6080cb65` — c'est le sondeur à clé morte déjà connu. Empreintes 3 h : MEDIUM
`be312daf7c1c` 1 212, `1f7f6080cb65` 28, NONE 1, `b72157f4bb61` 1.

**Condensation RSM** : aucun timeout depuis 06:00:53Z, aucun breaker rouvert depuis
06:01:22Z. Trois `Fallback chat client built` (11:02:23, 11:14:09, 12:02:37) **sans aucun
« primary skipped »** alentour ⇒ client de repli construit, **usage non démontré** ; lu avec
±3 lignes de contexte comme la leçon du 21/10 l'exige.

**Hôte / tâches** : commit **77,7 %** (298,8/384,4 Go, 85,6 Go libres — sous le seuil 85 %) ·
G: 85 % / 592 Go libres, **écriture OK** · pruneur **11:50:02Z** `mode=apply`, tier au cap
(268,4 Go) · nbaudit `ok:true` **12:06:30Z**, `subpath Image/`, 5 séries, 0 échec consécutif.

**Hors lane (RSM, relayé, aucun geste)** : 17× `Recent heartbeat found for myia-ai-01 (0s ago)
— concurrent instance possible but not blocking` (le process le signale lui-même comme non
bloquant) · 2× `Status contradictions detected after LLM generation (#1502/#3329)`, les deux
juste après une condensation terminée, entourés de `statusOutcome: "ok"` /
`summaryOutcome: "ok"` · 6× `Message not found: ai01-c2711-*` (messages inbox absents).

**Registre** : Q41 seule, ouverte. Le relevé de ce cycle (**22 583 MiB**) réduit la marge sous
la ligne d'alerte à **417 MiB** — c'est le plus haut jamais mesuré hors du pic 23 600
d'avant reboot, et la série est monotone sur 6 h. Critère d'échec toujours non rempli.

## 27/10 14:11Z — la montée VRAM S'ARRÊTE (22 568, −15 MiB), ZOLTRAAK identifié, claudish #718

Moteur `StartedAt` **02:15:05Z inchangé (12 h)** : health 200 @3,8 ms, RC=0, 3 conteneurs,
OOM 0, 0 signature d'erreur, `/dev/shm` 24G/63G. **Deux WEDGE `fail 1/2`** à 13:52:54Z et
14:01:43Z — premiers depuis 03:11Z, tous deux pendant un prefill lourd (snapshot p=155,9 M
et 159,0 M tokens) = comportement documenté, **aucun restart** (3 835 sondes OK, StartedAt
inchangé). GPU 1 21 650, GPU 2 45.

**★ La montée VRAM S'ARRÊTE : GPU 0 = 22 568 MiB, soit −15 MiB vs 22 583 au cycle
précédent.** La série devient 22 076 → 22 354 → 22 358 → 22 505 → 22 583 → **22 568** :
après +507 MiB en 6 h, la courbe **plafonne** vers 22 500-22 600. C'est la signature d'un
**high-water d'allocateur saturé** (les graphs atteignent leur forme max et n'élargissent
plus), **pas d'une fuite et pas d'une marche continue**. Marge sous la ligne 23 000 :
**432 MiB, stable**. Q41 : la reco rollback reste ouverte mais perd son argument d'urgence —
la montée ne continue pas.

**Amortissement** : prefix hit **92,25 %** cumulé (150,10 M/162,71 M) ; **marginal 2 h =
92,22 %** (33,19/35,99 M) — redescend du pic de 96,85 % du cycle précédent, mais reste dans
la bande haute (89-97 % sur 10 h). **Préemptions 0.** Prefill **4 999 tok/s** soutenus.
1,45 M tokens générés au total. **Aucun 400 dans la fenêtre 3 h** (740×200 / 35×401 / 0×400)
— la dernière signature #431 remonte à 09:30:12Z.

**★ Les deux empreintes nouvelles du census = le scanner ZOLTRAAK déjà connu.**
`64a6d7ec06fd` (17×401) et `ccd500f719e4` (9×401) : UA `ZOLTRAAK/0.2.0 (security-research)`,
IP Hetzner `116.202.50.2`, `/v1/models` corps 0, rafales de ~15 s toutes les ~10 h depuis le
23/10 19:40Z. **Sur tout le journal : uniquement des 401, jamais un 200** — ce sont les
« 2 clés inconnues 401-only » déjà notées au dashboard le 26/10, ici résolues en
empreintes. Clé morte connue `1f7f6080cb65` à 31 requêtes/3 h (le sondeur `108.165.33.236`,
cadence ~6 min). **Aucune IP inconnue avec un 200** : 7 h = `?` 1 398 (interne sans XFF),
`90.65.170.144` 634 (hairpin), `82.66.89.184` 4 (poste Paris).

**Condensation RSM** : aucun événement breaker depuis le HALF-OPEN de 07:38:13Z, aucun
timeout depuis 06:00:53Z. Deux `Fallback chat client built` (12:02:37, 13:04:01) **sans
aucun « primary skipped »** ⇒ usage non démontré.

**Coordination** : claudish a répondu au datapoint 10:44Z — classification « condensation
seule / chemin externe » **adoptée**, non comptée contre Gate R. Les en-têtes
`x_claudish_machine`/`x_claudish_client` sont suivis chez eux comme **claudish#718**
(propagation des en-têtes de télémétrie vers les endpoints internes uniquement, défaut
inerte, kill switch ; **design en attente d'arbitrage**, rien codé) — mon besoin
d'attribution y est le cas d'usage moteur. Point clos de mon côté, à suivre au déploiement.

**Hôte / tâches** : commit **78,6 %** (302,0/384,4 Go, 82,4 Go libres — sous le seuil 85 %) ·
G: 85 % / 592 Go libres · pruneur **13:50:03Z** `mode=apply`, tier au cap **268,4 Go /
10 227 fichiers**, **0 fichier supprimé** (régime établi : le LRU-on-hit fait que le pruner
n'a plus rien à reprendre), 0 erreur · nbaudit `ok:true` **14:05:50Z**, `subpath Image/`,
5 séries, 0 échec consécutif.

**Registre** : Q41 seule, ouverte — **critère d'échec non rempli et urgence retombée** : la
montée s'arrête à ~22 570, marge 432 MiB stable.

## 27/10 16:11Z — pruneur au cap (14,28 Go repris), marginal 94,37 % sous trafic doublé, #431 en grappe croissante

Cycle 2 h. Prod : health 200 @3,2 ms, StartedAt 02:15:05Z inchangé (**14 h sans restart**), RC=0, OOM 0, 3 conteneurs, 3 879 sondes OK, `/dev/shm` 24G/63G. VRAM GPU 0 **22 569 MiB** (+1 vs 22 568) — le plateau 22 500-22 600 tient un cycle de plus, marge 431 MiB sous 23 000 ; Q41 inchangée (reco rollback 0.70 au prochain restart naturel, urgence retombée). 2 WEDGE `fail 1/2` (14:39:35, 15:38:05) pendant prefills lourds 175,9 M/215,6 M tok — documenté, aucun 2/2.

**Amortissement (§5quater)** : hit total 92,88 % (214 029 648/230 450 644) ; **marginal 2 h 94,37 %** (Δ 63 926 944/67 737 711). Le trafic a presque doublé (8,5 req/min mesurées vs ~6 au cycle précédent, 1 271 lignes/150 min) et le ratio marginal MONTE — le régime « sessions récurrentes servies par leurs préfixes » tient sous charge. Préemptions 0.

**Pruneur fs : premier vrai trim depuis le patch anti-réécriture (PR #211).** Passage 15:50:04Z : **584 fichiers / 14,28 Go supprimés, atterrissage exact au cap (`bytes_after_expected` = 250,0 Go), 0 erreur**, 2 s d'exécution. Les cycles précédents lisaient `deleted 0` parce que le tier restait sous cap ; le journal de réécritures ≈ 0 fait maintenant grossir le stock d'âge > 30 min et le pruneur le rabat au plafond. Piège d'instrument noté : `bytes_total` dans `fstier_state.json` est la mesure AVANT passage (268,4 Go affichés alors que l'état post-run est 250,0).

**#431 en grappe croissante (datapoint hub).** **5 signatures 400 + corps > 100 K en 2,5 h** (14:17:23, 15:12:28, 15:13:41, 15:38:35, 16:08:40 — toutes Bun/1.3.14, donc hub claudish), corps **882 → 887 → 952 → 1044 K, croissants**, dont une paire à **73 s d'écart** (882→887 K). Lecture : le pattern appris (« le 400 ne se re-tente pas en local, la suivante arrive plus petite = auto-compact ») ne tient plus ici — la taille CROÎT entre occurrences, donc soit la session ajoute du contenu sans que le compact la ramène sous la fenêtre, soit ce sont des tours successifs d'une session qui grossit (cadence ~30 min = tour de cron d'une lane haiku). Coût par occurrence : 1 requête morte + 1 compact. Aucun impact moteur (0 wedges 2/2, préemptions 0). À rendre à claudish avec la fenêtre exacte — le remède est hub-side (#431).

**Sécurité** : empreinte nouvelle `f8628f6da447` — **1 seule occurrence sur TOUT le journal (23/10 → 27/10)** : 16:09:25Z, `/v1/models`, 401, LAN direct (pas de x_forwarded_for), UA Python-urllib/3.14, 0 octet. Famille « sonde de clé morte locale » (`b72157f4bb61`) : jamais de 200, jamais d'inférence — pas une alerte, consignée pour le census. ZOLTRAAK absent de la fenêtre (rafales ~10 h, pas repassé). Les 26×401 = 25 poller `1f7f6080cb65` (Go-http-client, /v1/models ~6 min) + 1 `f8628f6da447`. Aucune IP externe nouvelle avec 200 (seuls 90.65.170.144 site hairpin 461×200 et LAN direct).

**Autres** : condensation nominale (alternance chat/fallback-built sans `primary skipped`) ; WARN `BaselineManager — NonNominativeBaselineService non disponible` = burst de 4 s à 15:46, périmètre RSM, pas un événement moteur. Commit hôte **79,4 %** (sous 85). G: OK. nbaudit ok 16:05:55Z (`Image/` armé, `consecutive_failures 0`).

**Coordination** : #3962 (hang condensation 86 s) **vérifiée au champ par po-2024 — 0,7 % de fallback-truncated sur 48 h (critère ≤ 5 %, marge 7×)**, fermeture réservée au coordinateur ai-01 (RX107) ; ACK token HF de po-2023 **conforme** (empreinte `769d40e66497`, écrite dans leur seul `.env`, copie détruite, jamais affichée — Q-P2023-3 clôturée) ; message claudish 10:44Z déjà traité (classification adoptée leur côté), marqué lu.


## 27/10 18:11Z — pruneur en croisière, #431 s'arrête, architecture Swift mesurée pour CoursIA

Cycle 2 h. Prod : health 200 @3,3 ms, StartedAt 02:15:05Z (**16 h**), RC=0, OOM 0, 3 conteneurs, `/dev/shm` 24G/63G, aucun WEDGE nouveau depuis 15:38. **VRAM GPU 0 = 22 635 MiB (+66)** : le plateau tient toujours largement sous la ligne, mais le plafond haut se décale (22 505 → 22 583 → 22 568 → 22 569 → 22 635) — amplitude 130 MiB sur 6 h, marge **365 MiB** sous 23 000. Pas d'alerte, pas d'extrapolation (leçon « deux points ne font pas une pente », et ici six points donnent une bande, pas une tendance).

**Amortissement** : hit total 92,96 % (242 836 160/261 212 532) ; **marginal 2 h 93,64 %** (Δ 28 806 512/30 761 888). Le trafic est redescendu de moitié (4,4 req/min, 661 lignes/149 min, contre 8,5 au cycle précédent) et le marginal reste en bande haute — l'amortissement ne dépend donc pas du volume. Préemptions 0.

**Pruneur : régime de croisière confirmé.** 17:50:04Z : 3 fichiers / 0,07 Go repris, atterrissage 250,0 Go, 0 erreur. Après le gros passage de 15:50 (14,28 Go), le pruneur ne fait plus que rogner le filet de nouveaux blocs au-dessus du cap — exactement le comportement attendu du couple LRU-on-hit + plafond.

**#431 : la grappe s'est arrêtée.** Aucun 400 nouveau depuis 16:08:40Z (le seul 400 de la fenêtre est celui-là, en bord de fenêtre). Les 5 occurrences 14:17→16:08 n'ont donc pas fait école : soit la session a été compactée côté client, soit le hub a coupé. Datapoint déjà rendu à claudish via le [DONE] de 16:11Z.

**Sécurité : les 401 de la fenêtre sont tous élucidés, aucun n'est un acteur.** 32×401 = 25 poller `1f7f6080cb65` + 6 **sans en-tête Authorization** + 1 `f8628f6da447` (l'occurrence de 16:09Z, pas une récidive). Les 6 « sans clé » : Googlebot `66.249.79.32` (crawler, UA Android Nexus) sur `/v1/models`, quatre ouvertures navigateur depuis le LAN (17:44-17:46), un curl sans jeton via le hairpin du site (18:04) — tous 401, 0 octet, aucune inférence, aucun 200. **Piège d'instrument à retenir : `sha256("NONE")[:12] = c627c09c14e5`** — le compteur d'empreintes fabrique une « clé inconnue » à partir de l'ABSENCE de clé ; ne jamais la compter comme un acteur.

**Architecture Swift mesurée au config servi** (snapshot `9dba8a05`, source de vérité pour toute estimation KV) : **64 couches dont 48 `linear_attention` (GatedDeltaNet, état récurrent) + 16 `full_attention`** ; `hidden_size` 5120 ; 24 têtes Q / **4 têtes KV** ; `head_dim` 256 ; fenêtre 262 144 ; `compressed-tensors` pack-quantized W4A16 (vision + `lm_head` non quantifiés) ; TP=2. Coût KV structurel = 16 × 2 × 4 × 256 = **32 768 valeurs/token ⇒ 32 Ko/token en fp8, ~16 Ko en nvfp4** ; les capacités observées (424 610 fp8 @0.70, 750 747 nvfp4 @0.70, 831 723 @0.73) sont cohérentes avec ~17 % d'overhead d'alignement. **Corollaire pour tout calcul de « toit » : ce modèle ne tient que 16 couches d'attention pleine sur 64 — un dense classique à 64 couches pleines aurait 4× moins de tokens à VRAM égale.**

**Coordination** : dispatch HIGH de po-2027:CoursIA (carnet 11b « Matériel, débit, coût », PR CoursIA #22462) **traité dans le cycle** — réponse DM `msg-20261027T181236-fc7wjg` avec les mesures datées : débit N=48 907 t/s (23 t/s/flux), N=32 718 t/s, préfill 1 812 t/s (27,6 K) et 1 595-1 608 t/s (157 K), coût nvfp4 −13/−17 %, replay RAM 0,32 s vs 16,3 s (51×) via 655 Mo, replay fs 1,41 s vs ~11 s (~8×), lecture 2,1 GB/s / écriture 2,0 GB/s, 2,05 GB/s end-to-end ≈ 8 % du pic PCIe 4.0 x16, règle « tier ≥ 1,5× KV GPU », réglages retenus/écartés avec leur raison, architecture complète. Caveats inclus (A/B de même nuit ; capacités valables à 0,73 et pour 16/64 couches d'attention pleine). Global : mur GLM chez claudish (PAYG deepseek-flash jusqu'à ~18:46Z, doc #729) et incident disque po-2025 (Maintenance) — hors périmètre, lus seulement.

**Autres** : condensation nominale (aucun `primary skipped` depuis 07:38:13) ; WARN `BaselineManager` récidive à 17:51 (burst 4 s) + nouveau `[retirement #3782] lookup timed out — fail-open` = périmètre RSM, non relayés. Commit hôte **77,6 %**. G: OK. nbaudit ok 18:05:48Z.

## 27/10 20:11Z — inter-cycle chargé (adoption + radix tree), #431 reprend, les tiers servent

Cycle précédé d'un travail user-dirigé intensif (~18:20→20:10Z), tout committé avant ce cycle :

- **Mandat adoption** : paquet `swift-worker` livré (PR #237 : agent Claude Code canonique `docs/adoption/swift-worker.agent.md` + runbook + config sanitisée machines distantes), test live classifier 4/4, [PROPOSAL] global + crossPost roo-extensions. **Mandat étendu ~20:00Z : les sous-agents passent par Swift pour économiser Minimax.** Calibrations réelles : `swift-worker` 79 s / 78 K tokens harnais sonnet par spawn (⇒ batcher) ; `swift-reader` (harnais servi directement par Swift, auto-confirmé) 142 s / 60 K tokens Swift 0 €, extraction bornée 100 % fidèle.
- **Socle KV prouvé par A/B raw** : 27 347 tokens = 11,70 s froid → 0,62 s réutilisé, **99,0 % des tokens servis par le cache**, résident KV GPU (`external_prefix_cache_hits` Δ0 pendant l'A/B). Correction d'amalgame propagée : le « fork 143 o » = modes **Zoo** (fix #4354 déjà déployée ai-01 le matin même par le worker, LCP 18→548 o) ; l'analogue Claude Code = output-styles **claudish#640** (hub, ouvert).
- **Mandat radix tree** (fiche `project-kvcache-prefix-map`) : carte par harnais livrée (**PR #238**, `docs/kvcache/prefix-map.md` + instrument `scripts/testing/prefix_probe.py`) ; **issue-cadre vllm#239** (V1 Claude Code→claudish, V2 Zoo+V3 sk-agent→roo-extensions, V4 sous-agents→nous, V5 raw) annoncée global + crossPosts 2/2. Première carte 6 h : CC 40 % (shape canonique 87 % du CC, LCP ≥ capture), **fork CC↔OpenCode mesuré à 168 o** (juste après l'injection d'identité claudish), sondes 29 %, raw FR 22 %, **Zoo 0 req**. **Alias principal swift demandé à claudish** (DM 20:55Z) — le nom ne joue aucun rôle dans le cache (hash sur token ids).

**Cycle 20:11Z proprement dit** : moteur 18 h sans restart (StartedAt 02:15:05Z), health 200 @4,9 ms, OOM 0, RC=0, 3 conteneurs, shm 24G/63G (39 %), 4 WEDGE `fail 1/2` isolés (14:01→20:08Z) à compteurs fortement avancants (p 159M→314M) = motif prefill documenté, moteur sain-qui-travaille. **VRAM GPU 0 22 576 MiB** (−59 ; bande 22 505-22 635 maintenue, marge 424 MiB ; Q41 répondue entre-temps : maintien 0.73). Marginal **87,5 %** sur la fenêtre mesurable 19:55→20:11Z (reprise après le creux à 57 % de la fenêtre calibration, trafic ambiant compris) ; **préemptions 0** ; totals hits 292,6M/queries 316,4M = 92,5 %. **Fait nouveau : `external_prefix_cache_hits` +122 464 en 16 min** — première fenêtre où les tiers RAM/fs SERVENT activement du préfixe débordé (cumul 10,64M). Probe 2 h (614 req, 5,1/min) : CC 47 % sur **3 shapes seulement** (LCP 1498 intra-shape ; other-shape 39 req divergent à 186 o), sondes 36 %, condensation 20, mes tests 4.

**#431 REPRIS** : 2 nouveaux 400 window+1 à **18:53:55Z (883 217 o)** et **19:05:46Z (1 008 817 o)** (Bun, hairpin, corps croissants) après le silence de 16:08→18:53Z — 7 cumulés sur 7 h ; relance immédiate de claudish (DM `vllm-431-resume-20261027-2011`) : la règle cloud-on-400 n'a pas couvert ces occurrences. **Sécurité** : aucune IP inconnue avec 200 (`?` 1358, hairpin 1 053, poste Paris 3) ; **3ᵉ clé morte locale** `50829b227a2a` (2×401 `/v1/models` à 20:09Z, Python-urllib 3.13, 0 o, jamais 200 — famille b72157f4bb61/f8628f6da447, consignée à claudish, pas une alerte). **nbaudit ok 20:05:54Z — périmètre changé : `19451:Image/` armé** (vague 2 ; à dater/cadrer côté #228). Condensation nominale (zéro événement). Commit hôte **79,8 %** (marge 77,6 Go). G: OK (85 % plein, écriture 30 s). Pruner : dernier état 17:50Z en croisière, pas de nouveau trim >0,1 Go à relever.

## Cycle 22:2xZ — régime haiku mesuré : WEDGE 1/2 à densité 2×, préemptions en hausse lente, moteur sain

**Prod** : 20 h sans restart (StartedAt 02:15:05Z), health 200 @4,1 ms, OOM 0, RC=0, 3 conteneurs Up, shm **24G/63G (39 %)** = plancher du tier, propre. **VRAM GPU 0 22 587** (bande 22 505-22 635 confirmée, marge 413 MiB), GPU 1 21 650, GPU 2 45 (libre CoursIA). ⚠ Le rappel du prompt cron « rouvrir Q41 » est **périmé** : répondue 19:05Z (« on ne change rien »), VRAM dans la bande → pas de réouverture.

**WEDGE — le fait du cycle, correctement cadré** : **22 `fail 1/2` aujourd'hui, zéro `2/2`**, 3 873 sondes OK. Par heure : 00h 3 · 01h 4 · 03h 2 · 13h 1 · 14h 2 · 15h 1 · **20h 3 · 21h 6**. Ce soir = **9 en 2 h (4,5/h) contre 1,3/h en régime diurne** = densité 2× sous charge haiku (sondes décodage > 90 s derrière les prefills de 165 K), compteurs fortement avançants (p 359,4M / g 3,31M) = moteur qui travaille, motif documenté. La leçon 25/10 tient : 96 WEDGE dans le fichier du watchdog sur ~2,5 j ≈ 1,6/h de moyenne — **le régime existait avant, seule la densité monte**.

**Compteurs d'absorption (engagement claudish)** : marginal since-boot **83,0 %** (hits 303,3M / queries 365,4M) ; `external_prefix_cache_hits` **43,2M** (+19,8M en ~1,2 h, hit rate externe 69,6 % sur 62,1M queries) = **les tiers portent toujours la charge du working set** ; **préemptions 11** (+5 en 1,2 h ; graphe `request_num_preemptions` : 10/5 737 requêtes = **0,03 %**). **2 timeouts condensation 90 s** (mes appends dashboard 21:0xZ et 22:0xZ, fallback cloud sauvé, log RSM `21:41:15 Fallback chat client built`) = les petits callers synchrones restent LA première victime, avant tout wedge. Verdict feu : **🟡** (densité 2×, préemptions lentes) — pas 🟠 (aucun 2/2, tout servi).

**Sécurité** : aucune IP inconnue en 200. `?` 1 138 (1 129×200, trafic local sans XFF), hairpin 90.65.170.144 **1 036 (1 026×200)**, **poste Paris 82.66.89.184 6×200** (carte connue, trafic arrêté depuis le 01/10 → a repris, pas une alerte). **nbaudit ok 22:05:45Z** (5 séries, `19451:Image/`). Commit hôte **80,7 %** (marge 74,4 Go). G: 85 % plein / 592 G libres, **probe écriture OK**. Condensation RSM nominale.

## Cycle 00:11Z (28/10) — nuit calme, feu 🟢 : le pic haiku de 21h est soldé

**Prod** : 22 h sans restart (StartedAt 02:15:05Z), health 200 @3,8 ms, OOM 0, 3 conteneurs, shm 24G/63G (39 %). VRAM GPU 0 **22 654** (marge 346 MiB), GPU 1 21 650, GPU 2 45. Q41 non rouverte (rappel cron périmé — répondue 19:05Z la veille, VRAM dans la bande).

**Le pic de 21h est soldé, chiffres à l'appui** : **préemptions 11 → 11 sur 49 min** (et aucune depuis ~22:2xZ, soit ~1,8 h ; elles étaient 0 pendant les 24 h précédentes) ; **aucun WEDGE depuis 22:21:26Z** (~1 h 50, contre 6 dans l'heure 21h) ; 3 881 sondes OK ; file `running 1 / waiting 0` ; génération 69 tok/s agrégés (trafic nocturne).

**Amortissement** : marginal **94,8 %** sur la fenêtre (hits +19,15 M / queries +20,19 M) ; **`external_prefix_cache_hits` +352 736 seulement en 49 min** — à comparer au +12,5 M/45 min du pic : le working set haiku est reparti, le GPU ne déborde plus et les tiers sont au repos. Confirme que le débordement tiers était bien un symptôme de working-set, pas un défaut de cache. Deux sondes petits callers réelles à 22:2xZ : 200 en 0,33 s et 0,48 s.

**Annexes** : nbaudit ok 00:05:45Z ; commit hôte 80,0 % (marge 76,8 Go) ; G: 85 % / 592 G libres, probe OK ; condensation RSM sans événement ; census 7 h = hairpin 986×200, `?` 715×200, **poste Paris 14×200** (carte connue, +8 depuis 22:2xZ), **aucune IP inconnue en 200**. Registre : Q43 seule ouverte (tier RAM 24→32 GiB).

## Cycle 02:2xZ (28/10) — feu 🟠 DÉCLENCHÉ : les deux jambes du seuil sont remplies, repli po-2024 demandé à claudish

**Prod** : 24 h sans restart (StartedAt 02:15:05Z), health 200 @5,3 ms, OOM 0, RC=0, 3 conteneurs, shm 24G/63G (39 %). VRAM GPU 0 **22 637** (bande 22 505-22 675, marge 363 MiB), GPU 1 21 650, GPU 2 45. Seulement **1 WEDGE `fail 1/2` nouveau depuis 22:21Z** (00:46:56Z), zéro 2/2, 3 908 sondes OK.

**Les deux jambes du seuil 🟠 posté à claudish (27/10 22:0xZ) sont remplies** :

1. **Préemptions 11 → 19 (+8/2 h)** (Δ≥5 = jambe 1). Elles étaient 0 pendant les 24 h précédant 21h, 11 au pic de 21h, stabilisées 11 à 00:11Z — la reprise +8 en régime nocturne léger (3,9 req/min, 467 req/2 h dont 60 % de corps 100K-1M) montre que le working set haiku dépasse désormais le KV GPU **même la nuit**.
2. **Petits appelants >10 s récurrents** (jambe 2) : 7 sondes réelles ce cycle = **35,62 s · 0,18 · 0,20 · 0,88 · 28,87 · 0,26 · 9,47 s** — 3/7 dégradées, dont 2 > 28 s. Le moteur loggue en parallèle **KV GPU oscillant 71 % → 95,5 % → 97 % → 98 % sous seulement 3-5 req running** (02:16-02:17Z), génération 38-141 tok/s, hit rate préfixe (fenêtre moteur) 56,6 %, externe 79,7 %. La file reste saine (running 3-5 / waiting 0) : tout est servi, mais les petits callers synchrones patientent derrière les prefills de 165 K — la première victime documentée, mesurée cette fois en direct.

**Action (mandat user 22:26Z 27/10 « en fonction de tes findings mandater le repli d'un siège à claudish »)** : [ASK] claudish = **repli po-2024** (1ᵉʳ barreau de l'échelle user), agents principaux haiku → Minimax, sous-agents restent sur Swift ; mesure 2 h sur mes compteurs moteur ; barreaux suivants po-2023/po-2027 si toujours 🟠, po-2028 seulement si 🔴 (WEDGE 2/2 ou effondrement de file).

**Leçon d'instrument** : le feu 🟢 de 00:11Z était une **photo inter-crons** — entre deux passes haiku le moteur récupère complètement (marginal 94,8 %), et la mesure instantanée (2 sondes à 0,33/0,48 s) a manqué la starvation. La batterie de 7 sondes étalées sur ~3 min l'a capturée. Un verdict de charge sous pilote se prend sur batterie, pas sur instantané.

**400 window+1 (#431)** : **10 occurrences/12 h** (14:17Z→00:16Z, ~0,8/h, corps 882 K-1 043 K, tous UA `Bun/1.3.14` hairpin) — récurrence stable, le remède (routage cloud sur 400) reste côté hub.

**Attribution (réponse po-2023:claudish 00:56Z)** : le siège po-2023 pointe `ANTHROPIC_BASE_URL=http://192.168.0.50:3000` = **hub en direct, jamais le relais local** → tout repli po-2023 sera invisible côté relais (« mesurera toujours zéro ») ; instrument = mes compteurs moteur (c'est déjà le cas). La seule voie d'attribution par siège reste les en-têtes hub (#41). ACK renvoyé avec l'échelle.

**Sécurité** : aucune IP inconnue en 200 (hairpin 1 007/1 002×200, `?` 621/617×200, poste Paris 16×200, web1 2×200). **2 empreintes nouvelles** `ccd500f719e4`/`64a6d7ec06fd` = UA `ZOLTRAAK/0.2.0 (security-research)` 2×401 `/v1/models` 01:12Z, zéro accès — famille connue (identifiée 27/10 claudish#718), bruit consigné. **nbaudit ok 02:05:55Z** (`19451:Image/` = vague 2 active). Condensation : burst fallback cloud 01:40Z (4 lignes, #2719) puis « Condensation completed » = dégradation gracieuse nominale, cohérente avec la starvation petits appelants du créneau ; WARNs RSM (IdentityManager heartbeat, NonNominativeBaselineService, STALE-FILE #3230) = périmètre RSM, relayés au propriétaire sans action ici. Commit hôte **72,9 %** (marge 104,3 Go, en nette baisse depuis 80,0 %). G: 85 % / 592 G libres, probe OK.

**Structurel** : Q43 (tier RAM 24→32 GiB, plafond ~13→~20 sessions) reste le vrai levier de plafond — GO user en attente, s'applique au prochain restart naturel.

## Cycle 04:1xZ (28/10) — verdict : le 🟠 est retombé SEUL (amortissement), repli po-2024 suspendu

**Prod** : 26 h sans restart (StartedAt 02:15:05Z), health 200 @2,7 ms, OOM 0, RC=0, 3 conteneurs, shm 24G/63G (39 %). VRAM GPU 0 **22 634** (bande, marge 366 MiB). **Zéro WEDGE depuis 00:46:56Z** (> 3,4 h), 3 912 sondes OK, file `running 3 / waiting 0`.

**Les deux jambes du 🟠 ont retombé, sous charge RÉELLE doublée — c'est la clé du verdict** :

1. **Préemptions 19 → 20 (+1/2 h)**, contre +8 au cycle précédent.
2. **Batterie 6 sondes petits appelants : 0/6 > 9 s** (0,11-0,26 s uniformes), contre 3/7 à 35,6/28,9/9,5 s.

Et le census prouve que ce n'est **pas** une photo inter-crons (la leçon du cycle 00:11Z) : **823 req/2 h à 6,9 req/min, dont 565 corps 100K-1M (69 %)** — le volume long-contexte a DOUBLÉ pendant que tout était servi en 0,1-0,3 s. Mécanisme : à 02:2xZ les sessions récurrentes venaient de lancer leurs passes (prefills froids massifs → KV GPU 95-98 % → préemptions + starvation) ; deux heures plus tard elles tournent en **régime delta** — les socles sont amortis dans les tiers (external hits since-boot 92,3 M, ratio 75,6 %), chaque tour ne paie que son incrément.

**Action** : [REPLY] claudish = **[ASK] repli po-2024 SUSPENDU** — aucune exécution constatée ni confirmée, et le trafic n'a pas chuté (volume 2×) donc aucun repli visible ; la mesure dit que le moteur absorbe seul. Le feu re-montera à la prochaine fenêtre de prefills froids si les deux jambes se remplissent à nouveau — po-2027 reste barreau suivant désigné (il s'est déclaré prêt, 03:11Z). **Test de récurrence : le cycle 06:11Z** — les passes haiku suivantes diront si l'absorption tient.

**Grain socle reçu de po-2027:claudish (02:09Z, à relier au plafond)** : first_user p50 = **65,4 k tok**, **93 % = bloc d'instructions claude-md** (règles projet CoursIA ≈ **42 k tok/requête**, global 9,2 k, un CLAUDE.md Maintenance non identifié à 32,8 k). Levier **par-workspace** : une diète des règles réduirait le socle de chaque tour davantage que tout réglage moteur — complément orthogonal de Q43 (tier RAM 32 GiB).

**Annexes** : 400 window+1 : 2 nouveaux/2 h (récurrence stable, #431 hub). Sécurité : aucune IP inconnue en 200 (hairpin 1 314/1 309, `?` 646/646, poste Paris 10×200, web1 2×200) ; ZOLTRAAK absent de la fenêtre (les 21×401 = boucle clé morte connue `1f7f6080cb65`). **nbaudit ok 04:05:56Z** (`19451:Image/`). Condensation : fallback cloud 03:29Z (#2719) puis completed = dégradation gracieuse nominale. Commit hôte **73,5 %** (marge 101,8 Go). G: écriture OK. Registre : **Q43 seule ouverte**.

## Cycle 06:1xZ (28/10) — test de récurrence PASSÉ : l'absorption tient (🟢), première fenêtre sans 400 window+1

**Prod** : 28 h sans restart (StartedAt 02:15:05Z), health 200 @3,8 ms, OOM 0, RC=0, 3 conteneurs, shm 24G/63G (39 %). VRAM GPU 0 **22 638** (bande). 1 WEDGE `fail 1/2` à 05:26:11Z (compteurs avançants p 595 M / g 5,18 M), zéro 2/2.

**Le verdict 04:1xZ est confirmé par une seconde fenêtre** : **préemptions 20 → 20 (+0/2 h)** ; batterie 6 sondes = **0/6 > 10 s** (une frontière à 9,50 s, une à 0,90 s, le reste 0,13-0,31 s) ; file `running 2 / waiting 0`. La volée nocturne a brossé le moteur (WEDGE 05:26Z + sonde 9,5 s) sans rien casser : **l'amortissement delta absorbe maintenant les passes haiku récurrentes** — le mécanisme du cycle 02:2xZ (crise de prefills froids au lancement des passes) est la bonne lecture.

**Census 2 h** : 573 req à 4,8/min, 312 corps 100K-1M (54 %), 552×200, 21×401 (boucle clé morte), **0× 400 window+1 — première fenêtre 2 h sans aucun** (depuis le 14/10 où la signature fut identifiée) : soit les sessions se sont stabilisées sous la fenêtre, soit la règle #431 mord enfin — à confirmer sur les prochaines fenêtres avant de conclure.

**Attribution par siège — premier jeu réel, livré par po-2027 (05:57Z, `seat-charge.py` PR #735)** : sa lane haiku = **968 req/13 h ≈ 74/h, toutes `d--dev-CoursIA-2`, contexte p50 177-183k, pic 238/h, zéro servie par un modèle Anthropic** (remap total) ; 53,6 % des requêtes du siège partent en `deepseek-flash` PAYG (302 M tok de prompt). Convergence des instruments : ~74/h × 180k ≈ la moitié des 156 corps 100K-1M/h que je vois côté moteur — les deux mesures indépendantes s'accordent. **Le trou #41 est localisé côté protocole** : le siège envoie déjà `X-Claudish-Machine`, il manque `x-claudish-workspace` côté client ET l'opt-in `clientAttribution: true` côté hub (lecture du code par po-2027) — câblage po-2025↔po-2027, pas moteur.

**Contexte budget reçu (po-2025, 05:15Z)** : `haiku[1]` muré jusqu'au 11/11 ⇒ tout haiku au-delà du cap swift part en PAYG direct ; falaise PAYG à 22,7 % du jour ; swift = 1 080/9 520 réponses (11,3 %) ; le hub suit mon enveloppe (« genou 7+ flux, plafond ~13 sessions, levier = Q43 »). Le feu 🟠 retombé est bien enregistré chez claudish (« repli po-2024 suspendu, rien exécuté »).

**Annexes** : nbaudit ok 06:05:49Z (`19451:Image/`). Condensation : fallback cloud 05:45Z (#2719) puis completed — le consommateur PAYG documenté (#713), périmètre RSM. Commit hôte **73,9 %** (marge 100,2 Go). G: OK. Sécurité : aucune IP inconnue en 200 (hairpin 1 350/1 346, `?` 702/702, web1 8, Paris 7, web2 2). Registre : **Q43 seule ouverte**.

---

## Q43 — GO user 28/10 07:2xZ : tier RAM 24 → 32 GiB **ARMÉ, PAS ENCORE APPLIQUÉ**

**Décision** : user « Ok tu as mon Go » sur la question posée le 27/10 22:1xZ — élargir le tier RAM pour porter le plafond de sessions long-contexte de **~13 à ~20**, plutôt que de commencer à replier un siège haiku vers Minimax.

**Ce qui a été fait** : `KV_OFFLOAD_GB_FSTIER=32` ajoutée à `myia_vllm/.env`. C'est la variable dédiée du profil `-fstier` (`--kv-offloading-size ${KV_OFFLOAD_GB_FSTIER:-24}`) ; **elle n'existait pas** — seul un commentaire orphelin subsistait depuis le split du 21/10 (le commentaire affirmait que le `.env` portait 32, il ne le portait plus), donc c'est **le défaut 24** qui s'appliquait. Rendu vérifié **sans redémarrage** : `docker compose config` → `--kv-offloading-size 32`, avec `0.73` / `nvfp4` / `native` inchangés.

**⚠ Piège découvert en l'armant** : le GO était cadré « au prochain restart naturel, pas de recreate dédié ». Or **un `docker restart` ne suffit pas** — l'environnement est figé à la création du conteneur, donc le watchdog, `autoheal` ou un restart manuel rebooteront le moteur avec le **24** en mémoire. Seul un **recreate** (`compose up -d`) relit le `.env`. Conséquence : l'élargissement ne peut pas être « gratuit », il coûte une fenêtre. Aucun recreate lancé d'office (règle : jamais de redémarrage sans GO) — l'application attend la prochaine fenêtre.

**Marges mesurées avant tout geste** : commit hôte **75,3 %** (94,9 GB libres ; +8 GiB absorbables), `/dev/shm` **24G/63G = 39 %** dont **zéro orphelin** (4 blocs `psm_*` tous postérieurs au `StartedAt` 27/10 02:15Z — le plancher tmpfs est exactement le fichier mmap 25,7 GB du tier). Cible après recreate : **~34 GB = 54 %**, sous le seuil d'alerte 45 G.

**Contexte qui motive l'élargissement, mesuré à 07:2xZ** : file **5 running / 4 waiting**, préemptions **20 → 22 en ~10 min**. Tier fs à **252 G** (au cap 250 GiB près — le pruneur trim au prochain passage ; non incident, 428 G libres sur D:).

**Observation VRAM (Q41)** : GPU 0 à **23 026-23 035 MiB** sur 3 échantillons — le seuil de ré-ouverture (> 23 000) est franchi de ~30 MiB. Cause identifiée **côté desktop** : GPU 1, sans desktop, reste à **21 650** ; `--query-compute-apps` liste ~10 processus Windows sur GPU 0 (3× Edge, Chrome, VS Code, 2× NVIDIA Overlay, Docker Desktop, PhoneExperienceHost…) ≈ 1,4 GiB. Marge vers le total 24 564 = **1 530 MiB**. **Q41 non ré-ouverte** : mécanisme identifié, risque de boot non démontré (le boot du 27/10 à 0.73 a réussi avec 22,25 GiB libres). Re-question seulement si GPU 1 monte aussi, ou si un boot échoue.

**Retrait de la question** : nouveau `StartedAt` **et** `--kv-offloading-size 32` dans les args non-défauts du conteneur recréé.

---

## Cycle 08:1xZ (28/10) — feu 🟠 CONFIRMÉ sur deux passes : repli po-2024 demandé, et correction de mon propre cadrage Q43

**Moteur sain, mais saturé.** 30 h sans restart (`StartedAt` 27/10 02:15:05Z), `RC=0`, OOM **0**, health **200 en 3,3 ms**, 3 conteneurs Up. Ce n'est pas une panne — c'est une **saturation KV**, et c'est la distinction qui commande tout le reste du cycle.

### Les deux jambes du 🟠, mesurées

**Batterie 1 (6 sondes, 20 s d'écart, requête d'un mot)** : `90.04s` (timeout client) · `60.88s` · `50.55s` · `9.73s` · `12.74s` · `10.26s` ⇒ **6/6 > 9 s, 5/6 > 10 s, médiane 50,55 s**.
**Batterie 2 (~5 min plus tard, 3 sondes)** : `0.23s` · `13.21s` · `10.42s` ⇒ **2/3 > 10 s**.
**Cumul : 8/9 sondes au-dessus de 9 s sur deux passes espacées** — ce n'est pas la rafale de prefills froids du faux 🟠 de 02:2xZ, c'est installé.

**Préemptions 20 → 27 (+7 sur 2 h)** contre **+0** au cycle précédent. **KV GPU 97,5 %** avec **7 requêtes longues en vol** et une file à 0-3 (elle se vide par préemption, pas par service). `prompt_tokens_total` **+27,8 M en ~25 min** : le moteur travaille, il n'est pas figé. **Zéro WEDGE 2/2.**

**Census 2 h** : 535 requêtes à 4,5/min, dont **310 corps de 100 K-1 M (58 %)** ; statuts 200:510, 401:23, 400:2. **`400 window+1` = 2** — la fenêtre à zéro du cycle 06:1xZ **ne tenait pas** ; il faut plusieurs fenêtres avant de conclure quoi que ce soit sur #431.

**WEDGE** : 3 nouveaux `fail 1/2` (06:59:30, 07:19:25, 07:26:32), tous à compteurs avançants, **zéro 2/2**. Densité en hausse (2 en 7 min) mais le moteur n'a jamais décroché.

**Corroboration par un consommateur réel** : la condensation RSM a basculé en **fallback cloud à 07:41-07:43** (`#2719 cloud fallback condensation succeeded` ×2, puis `Condensation completed`) — exactement pendant l'épisode. Le petit caller synchrone est la première victime, et il dégrade proprement. C'est la prédiction du 27/10 vérifiée sur pièce.

### Action : repli po-2024 mandaté à claudish

[ASK] posté sur le dashboard claudish (08:2xZ) avec les deux batteries, le mécanisme (KV 97,5 %, admission `scheduler_reserve_full_isl=True`), et la demande d'un **seul** siège, **po-2024 en premier** selon l'échelle du user (puis po-2023, po-2027, po-2028 en dernier). J'ai **explicitement assumé** de ne pas attendre un 2ᵉ cycle 🟠 : ma règle des deux cycles visait les transitoires, un **timeout à 90 s sur une requête d'un mot** n'en est pas un.

### ⚠ Correction de mon propre cadrage Q43 (à ne pas répéter)

J'ai présenté ce matin le tier RAM 24 → 32 GiB comme un plafond « **13 → 20 sessions** ». En regardant la saturation de près, c'est **faux dans sa formulation** : ce qui sature est le **KV GPU (831 723 tokens)**, pas le tier. Élargir le tier RAM élargit le **working set gardé en cache** — donc moins de re-prefill à la ré-admission — mais **pas le nombre de requêtes admises simultanément**, qui reste borné par `gpu-util 0.73` et par l'admission « prompt entier » . **Le tier aide la pression observée ; il n'est pas le levier de concurrence.** Le levier immédiat est le repli de siège. La leçon vaut au-delà de Q43 : ne pas confondre **capacité de cache** et **capacité de concurrence**.

### Ce que je n'ai pas fait, délibérément

**Aucun recreate** du conteneur (qui appliquerait les 32 GiB). À 97,5 % de KV, un restart viderait 831 K tokens de contexte vivant et déclencherait une tempête de re-prefill — le piège documenté du 20/10 (« les restarts aggravent via le cache wipe »). Le recreate attend un creux.

### Annexes

VRAM : GPU 0 **22 871-22 872 MiB** — **redescendue sous la ligne 23 000 toute seule** depuis les 23 034 du 07:2xZ (le desktop a rendu ~160 MiB), ce qui **confirme expérimentalement** l'hypothèse desktop : GPU 1 reste plat à 21 650, GPU 2 à 98 MiB (CoursIA libre). `/dev/shm` **24G/63G = 39 %**. Prefix cache since-boot : 664,5 M queries / 519,3 M hits = **78,2 %** ; `external_prefix_cache_hits_total` **108,5 M**. Condensation nominale hors incident. nbaudit **ok 08:06:00Z**, 0 échec, 5 séries. Commit hôte **76,1 %** (marge 91,9 Go). G: OK (82 %, 705 G libres).

Sécurité : aucune IP inconnue en 200. `92.184.141.58` (1×200) **attribuée par son contenu** — prompt système « expert en synthèse de communications inter-agents » ⇒ condensation RSM, même famille que Q40, **pas une alerte**. `fp=1f7f6080cb65` ×20 = clé morte en boucle de 401 (connue) ; `NONE` ×2. WARN RSM « concurrent instance possible (myia-ai-01) » : `CronList` ne montre **qu'un** cron `d6ab14d5` — pas de doublon de mon côté, c'est une autre lane de la machine.

### Suivi 08:4xZ — l'épisode est retombé seul : repli **suspendu**, et un critère ajouté au protocole

**Contre-mesure ~20 min après la demande** : sonde **0,37 s** puis **0,34 s** (contre 10-90 s au plus fort), **KV GPU 97,5 % → 69,4 %**, file **3 running / 0 waiting**, **préemptions figées à 27** (+0 depuis la demande). Le moteur n'a pas été touché entre-temps : **il s'est vidé seul**.

**Décision** : [REPLY] posté à claudish — **repli po-2024 SUSPENDU, rien à exécuter**. Un repli de siège est une action durable ; il est disproportionné pour une fenêtre de 30-60 min qui se résorbe d'elle-même. J'ai demandé sur la foi de deux batteries espacées de 5 min, ce qui prouvait l'**impact** mais pas la **persistance**.

**⚠ Critère ajouté au protocole (leçon du jour, 2 épisodes 🟠 en 28/10 : 02:2xZ et 08:1xZ, tous deux retombés seuls)** : un 🟠 n'est **confirmé** que si les sondes restent lentes **≥ 30 min après la première sonde lente**. Une batterie courte établit l'impact, jamais la durée. Corollaire : deux batteries espacées de 5 min ne valent pas une mesure de persistance — c'est l'erreur que j'ai commise ce matin en montant en 🟠.

### Ce que l'épisode coûte réellement : le disjoncteur de condensation RSM

En lisant le log RSM, la conséquence de la famine des petits callers est **quantifiée** :

| Heure | Événement |
|---|---|
| 05:39:51 | `Condensation circuit breaker OPENED` |
| 06:21:00 | `HALF-OPEN (cooldown elapsed), retrying LLM` |
| **06:59:32** | `OPENED` — **la seconde même du WEDGE de 06:59:30** |
| 07:33:32 | `HALF-OPEN` après ~34 min |

**Chaque épisode ouvre le disjoncteur et le laisse ouvert ~35-40 min**, pendant lesquels toute condensation part en cloud (`deepseek-v4-flash`) — c'est-à-dire en PAYG, pour un travail que le moteur local devrait faire gratuitement. Pendant mon propre post de clôture, la condensation a littéralement timeouté à 90 s sur `localhost:5002` avant de basculer. **C'est le prix mesurable des épisodes**, et c'est ce qui devra être mis en balance avec le coût d'un repli de siège si les épisodes se multiplient. À surveiller : la fréquence d'ouverture du disjoncteur est un meilleur indicateur de gêne réelle que la VRAM.

## Cycle 10:1xZ (28/10) — le critère de persistance FONCTIONNE : pas de 🟠 malgré KV 95 % et +8 préemptions

**Le cycle où le critère ajouté à 08:4xZ a servi la première fois.** Toutes les conditions « de surface » d'un 🟠 réunies — et pourtant la mesure dit non.

### Mesures

| Indicateur | Valeur | Lecture |
|---|---|---|
| health | 200 en **13,5 ms** | sain |
| uptime / RC / OOM | **32 h** (StartedAt 27/10 02:15:05Z) / 0 / **0** | stable |
| **KV GPU** | **95,4 %** | proche saturation |
| **préemptions** | **35** (vs 27 à 08:4xZ) = **+8/2 h** | **jambe 1 du 🟠 : FRANCHIE** |
| **sondes 6 petits callers** | **0,14 / 0,23 / 0,23 / 0,29 / 1,63 / 0,22 s** — médiane **0,23 s**, **0/6 > 9 s** | **jambe 2 : NON franchie** |
| WEDGE | 1 seul `fail 1/2` (09:27:12Z), **zéro 2/2** | comportement attendu |
| prefix cache | 533 955 360 / 715 592 893 = **74,6 %** | croisière |
| file | 3 running / 1 deferred / **0 capacity** | pas de refus d'admission |

**Verdict : 🟢 avec réserve.** Les deux jambes du 🟠 ne sont pas réunies — la première l'est, la seconde non, et c'est la seconde qui décrit la gêne client. Conformément au critère du 08:4xZ, **aucun repli de siège demandé, aucun barreau monté**. C'est exactement le cas d'usage pour lequel le critère a été écrit.

### Découverte : `Deferred` est un état CHRONIQUE, pas un symptôme

Le moteur expose `Deferred: N reqs` — état distinct de `Waiting` (dont les `capacity` et `deferred` sont des sous-motifs). Sur 28/10 de 00:00:03 à 10:16:19 (**3 684 ticks**), **1 128 ticks ont `Deferred ≥ 1`, soit 30,6 % du temps**. Pendant ces ticks, mes sondes répondent en **0,14-1,63 s**. **Conclusion mesurée : `Deferred ≥ 1` n'est PAS corrélé à la gêne des petits callers.** À ne pas lire comme un signal d'alerte — c'est un régime de fonctionnement.

### Le 90 s de 07:41Z n'était PAS le double-bind du port 5002

roo-extensions a demandé (dashboard 07:44Z) si la complétion via `localhost` (→ `::1` → `wslrelay`) se fige alors que `127.0.0.1` répond, parce que RSM a tenu 90 s sans réponse à 07:41Z. **Testé, réponse : non.**

| Chemin | 4 complétions réelles |
|---|---|
| `127.0.0.1:5002` | 0,24 / 0,16 / 0,16 / 0,16 s |
| `[::1]:5002` (wslrelay) | 0,16 / 0,14 / 0,16 / 0,14 s |
| `localhost:5002` (→ ::1) | 15,7 / 0,46 / 0,16 / 0,15 s, puis **10/10 OK** médiane **0,32 s** (1 outlier 4,20 s) |

**`[::1]` en direct est aussi rapide que `127.0.0.1`** ⇒ le relais `wslrelay` n'est pas en cause, et le double-bind n'explique pas le gel. Sur 14 appels `localhost`, **2 outliers (15,7 s et 4,2 s)** = mise en file derrière un prefill, pas un défaut de chemin.

**Ce que le moteur faisait à 07:41:23Z** (log, filtre de date explicite) : `Running: 3, Waiting: 1, Deferred: 1, GPU KV cache usage: 27,6 %`, **prompt throughput 0,0 tok/s**, generation 120,1 tok/s. **KV à 27,6 % — le moteur n'était donc PAS saturé en capacité au moment du gel.** Et l'access log montre que les requêtes du condenseur de 07:41:58 (16 917 o) et 07:42:04 (54 664 o) ont reçu **200**. Le serveur a répondu ; le gel est côté client RSM (timeout avant lecture complète), pas côté moteur. **Ne pas ouvrir de dossier « double-bind » : le mécanisme n'est pas démontré, et la mesure l'exclut comme cause première.**

⚠ **Piège d'instrument rencontré (à consigner)** : `docker logs --tail 40000 | grep '07:4[0-9]:' | head -25` a rendu les lignes du **27/10** 07:40 — le tail de 40 000 lignes couvre ~26 h, donc le motif d'heure matche deux jours et `head` prend le plus ancien. **Toujours inclure la date dans le motif** (`'10-28 07:4[0-9]:'`) et se méfier de `head` sur un grep multi-jours.

### VRAM : la ligne 23 000 est franchie de façon TENUE ⇒ Q41 ré-ouverte

GPU 0 = **23 121 / 23 092 / 23 101 / 23 101 / 23 114 MiB** sur 5 échantillons à 12 s — **min 23 092, jamais sous la ligne pendant la minute**. Aux cycles précédents la valeur redescendait seule (23 034 → 22 871) ; **cette fois non**.

Discriminant re-mesuré, net : **GPU 1 (même moteur, sans desktop) est plat à 21 650 MiB**, GPU 2 à 45 MiB, et `--query-compute-apps` liste **18 entrées, toutes sur GPU 0, aucune sur GPU 1** ⇒ surcoût desktop = **~1 450 MiB**. **Le moteur n'a pas grossi** ; marge vers 24 564 = **1 454 MiB**.

**Q41 ré-ouverte au registre** parce que le critère que j'avais moi-même écrit est franchi — mais avec un cadrage neuf : ce n'est plus « prévention gratuite ». Le rollback achèterait **+737 MiB de marge** contre **−81 000 tokens de KV** sur un moteur à **95,4 %** d'occupation. **Les deux métriques se dégradent en même temps** ⇒ arbitrage marge-de-boot contre capacité-de-concurrence. Recommandation : **traiter les deux ensemble dans la fenêtre du recreate Q43** (un seul geste, une seule indisponibilité).

### `400 window+1` : 6 en 12 h — ce n'est plus un accident

| Heure (Z) | `body_bytes` | UA | Message |
|---|---|---|---|
| 27/10 22:32:59 | 916 612 | Bun/1.3.14 | 262 145 input tokens |
| 28/10 00:16:09 | 939 454 | Bun/1.3.14 | idem |
| 28/10 02:21:18 | 899 159 | Bun/1.3.14 | idem |
| 28/10 04:04:46 | 924 137 | Bun/1.3.14 | idem |
| 28/10 07:49:42 | 904 195 | Bun/1.3.14 | idem |
| 28/10 08:01:47 | 944 022 | Bun/1.3.14 | idem |

**Toutes `fp=be312daf7c1c`** (clé MEDIUM = le hub claudish), **toutes via `Bun`**, **toutes à exactement 262 145 = fenêtre + 1**. Cadence ~1 toutes les 2 h. Le `body_head` porte l'injection d'identité claudish ⇒ ce sont des sessions **Claude Code** qui franchissent la fenêtre avant leur propre auto-compact. **La règle #431 (router vers le cloud sur 400 « maximum context length », jamais de retry local) n'absorbe donc pas ces 6 cas** — soit elle n'est pas armée, soit son motif ne couvre pas ce chemin. À rendre à claudish ; ce n'est pas un défaut du moteur, qui refuse correctement et sans rogner.

### Sécurité

**Aucune IP inconnue avec des 200.** Carte : `90.65.170.144` (hairpin) 1 301/1 297×200 · `?` 695/693 · web1 14/14 · **`92.184.141.58` 4/4** (condensation RSM, attribuée par son contenu) · web2 2/2 · poste user Paris 2/2.

**Nouvelle empreinte `64a6d7ec06fd`** = `UA ZOLTRAAK/0.2.0 (security-resea…)`, IP **31.77.203.199**, **3 requêtes sur `/v1/models`, toutes 401, zéro 200**. Scanner d'auth, aucune tentative d'inférence. **Pas d'alerte** (la règle exige des 200) ; consigné. Autres 401 : `1f7f6080cb65` ×123 (clé morte connue), `NONE` ×12, `246a3f382534` ×2 + `68eefcf5ff3f` ×2 (401 navigateur Mac/Win connues).

### Annexes

nbaudit **ok 10:05:47Z**, rc 0/0, `consecutive_failures 0`, 5 séries — et le `subpath` est passé à **`19451:Image/`** ⇒ **vague 2 armée**, le fanion posé le 10/10 est levé. · G: **écriture OK**, 82 % (701 G libres). · commit hôte **78,1 %** (marge 84,3 Go, seuil 85 % non franchi). · `/dev/shm` **24G/63G = 39 %**, zéro orphelin. · condensation **nominale**, disjoncteur refermé depuis 07:33:32, aucune nouvelle ouverture. · prouveur #1453 : aucune passe armée.

### Coordination

**po-2023:claudish (DM 00:56Z, lue)** — précision d'instrument majeure pour ma méthode de repli : **leur relais est structurellement aveugle à leur propre charge**, parce que le client du siège pointe `ANTHROPIC_BASE_URL=http://192.168.0.50:3000` = **le hub en direct**. Leur trafic arrive chez moi en hairpin anonyme sans jamais toucher leur conteneur. **Conséquence : « replier po-2023 » mesuré via leur relais affichera toujours zéro, quel que soit le trafic réel — un zéro n'est pas une preuve de légèreté.** Le seul levier côté eux est leur `ANTHROPIC_BASE_URL`, qui est une question user ouverte chez eux depuis le 19/10. À intégrer : **l'échelle de repli doit se décider sur mes compteurs moteur, jamais sur le zéro d'un relais aveugle.**

**po-2023:vllm a fermé #10** (commentaire `6465869325`, `not planned`) avec les preuves : PR #44053 mergée le 02/08, #40308 fermée non mergée, #41123 stale, Genesis retiré, KV nvfp4 ⇒ TurboQuant plus utilisé. **Ma relance du 07:40Z est soldée** ; dossier retiré du watch.

## Suivi 10:3xZ — le 90 s s'est reproduit en direct, et ma première conclusion était trop généreuse pour le moteur

**Ce qui s'est passé.** Mon propre post de clôture a déclenché une condensation, et le diagnostic retourné par le dashboard montre l'échec du primaire en direct : `TIMEOUT after 90s (limit 90s): no response — socket held open, neither a completion nor an HTTP error. Endpoint likely hung. [vLLM localhost:5002 model=qwen3.6-35b-a3b]`, `fallbackUsed: true`, sauvé par `deepseek-v4-flash` (27,8 s). La jambe *summary*, elle, a réussi sur le primaire en **40,8 s**. Donc l'épisode n'est ni isolé à 07:41Z, ni rarissime.

**⚠ Correction de mon propre énoncé du même cycle.** J'avais écrit plus haut : « le serveur a répondu ; le gel est côté client RSM, pas côté moteur ». **C'est trop généreux et je le retire.** Ce que les données établissent réellement :

| Fait | Mesure |
|---|---|
| Le moteur répond 200 aux vraies requêtes de condensation | 7/7 dans l'access log 10:03→10:19Z (`OpenAI/JS`, 18-55 Ko) |
| Latence d'une requête de taille condensation (7 141 tok), **8 échantillons identiques** | **1,8 / 2,0 / 2,1 / 2,2 / 2,3 / 2,4 / 5,0 / 30,6 s** |
| Médiane | **~2,3 s** |
| Queue observée | **30,6 s** (1 échantillon sur 8, soit 13× la médiane) |
| Timeout 90 s reproduit à la demande ? | **non** — jamais atteint sur mes 8 essais |

**Énoncé corrigé** : typiquement **~2 s**, avec une **queue à 30 s** sur une requête identique. Le 90 s n'est donc pas de la lenteur installée, c'est un **événement de queue** ; son déclencheur exact n'est **pas identifié par mes instruments**. Et la variance est bien côté moteur : c'est le prix de la mise en file derrière les gros prefills du dense 27B, déjà documenté. **Ne pas dire « c'est le client » ; dire « le moteur répond, avec une queue de latence dont je ne connais pas le déclencheur ».**

**Le vrai défaut est mon instrument, et il est réparable.** Le middleware `error_source_capture` journalise `auth_prefix, body_bytes, body_head, body_tail, body_truncated_bytes, client, host, model, path, referer, status, ts, user_agent, x_claudish_*, x_forwarded_for, x_forwarded_host, x_real_ip` — **aucun champ de durée**. Mon census peut donc dire *qui*, *quoi*, *quel statut*, **jamais *combien de temps***. C'est précisément pourquoi la question du 90 s traîne depuis des jours et pourquoi j'ai dû la trancher par sondes synthétiques au lieu de lire le log. **Action à porter** : ajouter `e2e_s` (et `ttft_s` si disponible) au middleware — après quoi « quelle est la latence p95 par classe d'appelant » devient une requête de lecture, plus une campagne de sondes. À ouvrir en issue la prochaine fois que je touche au middleware (fichier `myia_vllm/middleware/logging_middleware.py`, non monté en prod aujourd'hui : `error_source_capture` est l'actif, à vérifier avant de promettre).

**Ce que ça ne change pas** : le 🟢 du cycle. Mes 6 sondes petits callers sont restées à 0,14-1,63 s ; le 🟠 exige la gêne des **petits** appelants, et elle n'est pas mesurée. Ce qui est mesuré, c'est qu'un **gros** appelant synchrone (la condensation, 7-13 k tokens) peut dépasser son budget de 90 s — et que le disjoncteur le paie en cloud. C'est un coût réel, mais ce n'est pas la même jambe.

## Cycle 12:1xZ (28/10) — 🟢 franc : préemptions +0 (premier cycle plat depuis la hausse) ; la queue de latence réapparaît par épisodes ; VRAM repassée sous 23 000

**Prod** : health 200 en 10,6 ms, RC=0, StartedAt inchangé (27/10 02:15:05Z, up 34 h), OOM 0, 3/3 conteneurs Up. Moteur sous charge lourde continue (compteurs p=766,9M tokens prompt traités depuis le boot, 3 900 sondes watchdog OK).

**Les deux jambes du 🟠** :
- Jambe 1 (préemptions Δ≥5/cycle) : **NON franchie — 35 → 35, Δ=0**. Premier cycle plat depuis que la hausse a commencé (27→35 au cycle 10:1xZ). La correction d'hier (charge delta amortie) se confirme.
- Jambe 2 (petits appelants >10 s récurrents) : **2/6 sondes >9 s** (23,59 s et 13,08 s), médiane **0,38 s**. Les deux lentes encadrent un épisode de ~2 min (12:05-12:08Z) qui a coincidé avec le WEDGE `fail 1/2` du watchdog à 12:05:40Z — sa propre sonde décodage a timeouté au même moment. Rien de ≥30 min ; et la jambe 1 n'est de toute façon pas franchie. **Verdict 🟢, aucun barreau monté** (repli po-2024 reste SUSPENDU).

**La queue de latence réapparaît, en épisodes.** Le cycle précédent : 0/6 sondes >9 s. Ce cycle : 2/6 avec la même signature queue (médiane basse + queue 13-24 s sur requêtes identiques). C'est la 3ᵉ forme observée en 3 cycles : absente → queue 30 s sur gros corps → queue 24 s sur sondes 16 tokens. **La queue frappe maintenant les PETITS corps aussi** — nouvelle donnée qui contredit l'hypothèse « mise en file derrière les gros prefills uniquement ». Le déclencheur reste non identifié ; l'action instrument (champ `e2e_s` au middleware, §suivi 10:3xZ) reste le levier.

**KV et amortissement** : KV GPU **91,2 %** (vs 95,4 % au cycle précédent). Prefix hits/queries = **572,8M/769,5M = 74,4 %** depuis le boot — l'amortissement delta reste sain. File d'attente : 1 deferred, 0 capacity. WEDGE : 2 `fail 1/2` dans la fenêtre (11:50, 12:05), **zéro 2/2**.

**VRAM — retour sous la ligne** : GPU 0 **22 888 MiB** (vs 23 092-23 121 soutenus au cycle 10:1xZ), GPU 1 plat 21 650 (identique). La « tenue » du cycle précédent ne s'est pas maintenue — cohérent avec le high-water d'allocateur, pas avec une dérive. Q41 reste ouverte au registre (seul le user la ferme), avec cette nouvelle mesure en ligne.

**400 window+1** : **+2 dans la fenêtre** (corps 951 328 et 1 048 598 octets, UA `Bun/1.3.14` = hub, statut 400). Cumul : 6/12 h au cycle précédent + 2/2 h ici. La règle claudish #431 ne les absorbe toujours pas.

**Sécurité** : aucune IP inconnue avec des 200 dans les 7 j. `212.133.41.47` (3 req, 3×200) = nouvelle adresse de la famille mobile flotte déjà classée (`.21`/`.29`/`.33` au census) — pool tournant Free, non-alerte. `82.66.89.184` (Paris, 55 req/7 j) = activité user de la semaine, connue. Clé morte `1f7f6080cb65` : **20×401/2 h** — la boucle de 401 flotte continue, bruit connu.

**Annexes** : nbaudit **ok** 12:05:53Z, `consecutive_failures=0`, **vague 2 `19451:Image/` confirmée servie** dans la config du passage horaire. G: OK (82 %). Commit hôte **79,8 %** (306,7/384,4 GB, marge 77,7 GB — sous le seuil 85 %). /dev/shm 24G/63G = 39 % (plancher = tier RAM 24 GiB jusqu'au recreate Q43). Condensation : **aucun nouvel épisode disjoncteur** depuis celui de 07:41Z (déjà documenté §suivi 10:3xZ) — les constructions de client fallback à 08:39/10:38/10:47 ne sont pas des succès de repli, aucun « cloud fallback succeeded » après 07:43.

**Coordination** : inbox trié — rapport worker po-2024 sur #4334 marqué lu avec la note que son output est « Request timed out » sous un statut SUCCÈS (le livrable cache la panne du producteur — le même piège que nbaudit ; à ne pas compter comme progression sur #4334) ; la queue-po2025 (tests ms#1551) relève de roo-extensions, marquée lue ; relais po-2023 déjà intégré au cycle précédent. Dashboard : aucune mention non traitée.

**Registre** : Q41 ouverte (MAJ : 22 888 ce cycle, retour sous la ligne — la question reste posée à double sens VRAM/KV). Q43 armée (`KV_OFFLOAD_GB_FSTIER=32` dans `.env`), **toujours en attente de recreate** — jamais pendant un épisode de saturation ; recommandation maintenue de grouper Q41+Q43 dans le même geste si le user tranche Q41 pour 0.70.

## Cycle 14:1xZ (28/10) — 🟢 ; la cadence des 400 window+1 double (4/2 h) ; la queue tape la 1ʳᵉ sonde pour la 2ᵉ fois

**Prod** : health 200 en 6,2 ms, RC=0, StartedAt inchangé (27/10 02:15:05Z, up 36 h), OOM 0, 3/3 Up. VRAM GPU 0 **22 902** (2ᵉ cycle consécutif sous 23 000 ; 22 888 hier), GPU 1 plat 21 650. /dev/shm 25G/63G = 39 %. Watchdog : 1 WEDGE `fail 1/2` (13:08:37Z), zéro 2/2, 3 908 OK.

**Les deux jambes** : préemptions **35→39 (+4, sous le seuil Δ≥5)** ; sondes **1/6 >9 s** (30,72 s), médiane **0,17 s** (les 5 autres : 0,14-0,21 s). **🟢, aucun barreau.**

**La queue frappe la 1ʳᵉ sonde, 2ᵉ cycle de suite** : 23,6 s (12:1xZ) puis 30,7 s (14:1xZ), toujours la sonde n°1 de la batterie, les suivantes immédiates étant à ~0,2 s. Deux occurrences ne font pas une pente — mais si le motif tient au 16:1xZ, l'hypothèse « position en file au démarrage de la batterie » (mon premier appel arrive derrière ce que le moteur tient en vol) devient crédible et **testable** (batterie décalée de 60 s, ou sonde jalonnée avant la n°1).

**400 window+1 : la cadence DOUBLE** — **4 dans la fenêtre** (corps 896-1 018 Ko, UA `Bun/1.3.14` = hub), après 2 à 12:1xZ. Cumul ~16 h : **12**. Chacune = une session hub qui franchit 262 145 tokens d'entrée et prend un 400 sec. #431 ne les absorbe pas — **à re-signaler à claudish avec la cadence montante** (leur disjoncteur vise le retry-local, pas la cause).

**KV et charge** : KV **47,3 %** en fin de fenêtre (vs 91,2 % à 12:1xZ — la charge est en créneaux), file 0/0, prefix hits **72,8 %** (600,6M/825,3M). Census 2 h : 581 req, **58 % de corps 100K-1M** (338/581) — le régime long-contexte tient. Clé morte `1f7f6080cb65` : 20×401/2 h (bruit stable).

**Sécurité** : aucune IP inconnue en 200 (7 j). `212.133.41.47` déjà classée famille mobile flotte au cycle précédent. Rien de nouveau.

**Annexes** : nbaudit **ok** 14:07:14Z (`Image/` vague 2 servie, `consecutive_failures=0`). G: OK. Commit **80,2 %** (marge 75,9 GB). Condensation : **zéro nouvel épisode disjoncteur depuis 07:43Z** (dernier événement du log).

**Registre** : Q41 ouverte — VRAM sous la ligne pour le 2ᵉ cycle consécutif (mesure en ligne au registre, high-water raffermi). Q43 armée, recreate en attente de fenêtre. Rien de nouveau à arbitrer.

## Cycle 16:1xZ (28/10) — 🟢 ; l'hypothèse « 1ʳᵉ sonde » est SOUTENUE une fois ; la formule d'empreinte de clé était fausse et est corrigée

**Prod** : health 200 en 6,9 ms, RC=0, StartedAt inchangé (up 38 h), OOM 0, 3/3 Up. VRAM GPU 0 **22 875** (3ᵉ cycle consécutif sous 23 000 ; 22 888 / 22 902 / 22 875), GPU 1 plat 21 650. /dev/shm 25G = 39 %. Préemptions **39→40 (Δ=1)**, KV **92,2 %**, file 0/0, prefix hits **73,3 %** (647,4M/883,0M). WEDGE 2 `fail 1/2` (14:38, 16:08), zéro 2/2.

**Expérience : une sonde de préchauffe AVANT la batterie.** Le motif « la 1ʳᵉ sonde paie la queue » avait tenu 2 cycles de suite (23,6 s puis 30,7 s, les suivantes à ~0,2 s). Cette fois j'ai émis **un appel de préchauffe** (0,37 s, exclu du verdict), attendu 5 s, puis lancé la batterie : **0/6 >9 s, médiane 0,27 s, max 1,96 s**, et la n°1 à **0,16 s**. **L'hypothèse « position en file / état d'ordonnanceur froid au premier appel » est soutenue une fois** — pas prouvée : il faut la reproduire à l'identique au moins deux cycles de plus, et la contre-épreuve (batterie sans préchauffe où la 1ʳᵉ est de nouveau lente) existe déjà 2×. Protocole à conserver tel quel pour les prochains cycles.

**★ Le piège d'instrument du jour : la formule d'empreinte de clé notée en mémoire était FAUSSE.** Le census affiche `fp=sha256(auth_prefix)[:12]` et la mémoire affirmait `auth_prefix = sha256("Bearer "+clé)[:12]`, « vérifiée sur MEDIUM ». Calcul fait : cette formule rend `53be2310eebd`, **pas** `be312daf7c1c`. Le code tranche (`error_source_capture.py:157`) : le middleware journalise `authorization[:24] + "..."`, soit `"Bearer " + 17 premiers caractères de la clé`. Vérification : `sha256("Bearer d9164427c537056a7...")[:12]` = **`be312daf7c1c`** ✔. Conséquence pratique : les empreintes publiées restent des **identifiants stables de classe de clé**, mais elles ne sont pas recalculables depuis la valeur sans respecter cette forme — et la correction est portée en mémoire. Aucune conclusion de sécurité passée n'est invalidée (l'attribution MEDIUM tenait par la mesure du trafic, pas par la formule).

**Clé inconnue `550dcf4b78d5`** : 2 requêtes sur `/v1/models` en **401** (10:42:30Z `curl/8.14.1`, 14:34:57Z `Python-urllib/3.13`), corps 0, **zéro 200**, pas de `x_forwarded_for` (donc LAN/hairpin). Distincte de la clé morte connue `1f7f6080cb65` (20×401/2 h, elle aussi stable). Ne correspond à aucune de nos trois clés sous aucune des deux formes testées ⇒ **troisième clé morte en circulation** sur la flotte, à surveiller si elle produit un 200 un jour. Pas d'alerte : la règle vise les IP avec des 200.

**400 window+1 : 2/2 h** (858 Ko, 940 Ko, Bun) — **retour au niveau de 12:1xZ après le pic de 4** ; cumul ~**14 sur 18 h**. La cadence n'est donc pas monotone : un pic, puis retour. Ne pas conclure à une accélération sur un seul cycle.

**Charge** : 595 req/2 h (5,0/min), **336 corps 100K-1M (56 %)**, 235 corps <1K. Statuts : 570×200, 23×401, 2×400.

**★ Marge commit hôte : 84,2 %** (60,7 Go libres) — **au plus près du seuil 85 % depuis le 09/10** (79,8 → 80,2 → **84,2** sur trois cycles). Aucun travail lourd autorisé au prochain cycle si ça ne redescend pas ; à surveiller (vmmemWSL = premier consommateur, pas Code.exe).

**Annexes** : nbaudit **ok** 16:05:51Z (`Image/` servie). G: OK. Condensation : **zéro nouvel épisode disjoncteur depuis 07:43Z** (5 cycles consécutifs).

**Registre** : Q41 ouverte — VRAM 3ᵉ cycle sous la ligne, high-water confirmé. Q43 armée, recreate en attente.

## Cycle 18:1xZ (28/10) — 🟢 ; préchauffe reproduite (1/2) ; GPU 0 repasse au-dessus de 23 000 ; marge commit 86,4 % = seuil 85 % franchi

**Prod** : health 200 en 3,7 ms, RC=0, StartedAt inchangé (up 40 h), OOM 0, 3/3 Up. **VRAM GPU 0 = 23 139 MiB — repasse au-dessus de la ligne 23 000** après 3 cycles dessous (22 888 / 22 902 / 22 875) ; GPU 1 plat 21 650. 4ᵉ excursion au-dessus de la ligne (23 092-23 121 à 10:1xZ, pic 23 600, aujourd'hui 23 139) — la bande de high-water en charge **touche** 23,1 K ; Q41 ouverte, aucun fait nouveau majeur (boot jamais raté à 0.73). /dev/shm 25G = 39 %. Préemptions **40→40 (Δ=0, 2ᵉ cycle plat consécutif)**, KV **44,0 %** (la charge de fin de journée retombe), file 0/2, prefix hits **73,5 %** (681,0M/926,4M). WEDGE 2 `fail 1/2` dans la fenêtre (16:16, 17:44), zéro 2/2.

**Préchauffe — reproduction 1/2.** Même protocole (préchauffe exclue du verdict, 5 s d'écart, 6 sondes à 20 s) : préchauffe 0,32 s, **1ʳᵉ sonde 0,38 s**, puis 0,16/0,17/**7,99**/0,16/0,20 — **0/6 >9 s**. Bilan sur 4 cycles : sans préchauffe, la 1ʳᵉ sonde = 23,6 s / 30,7 s (2×) ; avec, 0,16 s / 0,38 s (2×). **L'hypothèse « le premier appel paie un coût que la préchauffe absorbe » est soutenue 2×/2×** — une dernière reproduction pour la tenir acquise. ★ Nuance nouvelle : la sonde 4 à **7,99 s** (sous le seuil 9 s) montre que la file peut frapper **en milieu de batterie** : le coût du premier appel et la contention sporadique du trafic de fond sont **deux phénomènes distincts** — la préchauffe n'élimine que le premier.

**400 window+1 : 4 dans la fenêtre** (16:26, 17:02, 17:37, 17:42 ; corps 891 K-1 030 K, Bun) — même niveau que le pic de 14:1xZ, cumul **~18/22 h**. Cadence 2→4→2→4 : toujours non monotone, mais le niveau 4 revient ; 3 des 4 en 40 min (17:02→17:42) = grappes. Re-signalement claudish déjà fait 14:15Z, pas de nouveau DM sur une cadence plate.

**Sécurité — la boucle de 401 localisée** : nouvelle IP `108.165.33.236`, **70 requêtes 401 sur `/v1/models`**, UA `Go-http-client/1.1`, intervalle ~6 min, 11:19Z→18:09Z, **toutes avec la clé morte connue `1f7f6080cb65`**, zéro 200. C'est la **source du « bruit stable 20×401/2 h »** : un **poller Go de la flotte avec une clé périmée** qui s'interroge sur notre santé et prend 401 à chaque fois. Pas une alerte (règle = IP avec 200), mais un process flotte aveugle — son propriétaire devrait corriger la clé ou l'arrêter. Census : 505 req/2 h (4,2/min), 481×200 / 20×401 / 4×400.

**★ Marge commit hôte : 86,4 %** (331,9/384,4 Go, 52,5 Go libres) — **seuil 85 % franchi**, 4ᵉ hausse consécutive (79,8 → 80,2 → 84,2 → 86,4). **Aucun travail lourd** ; vmmemWSL reste le premier consommateur. À 90 %+ = signal user, on ne bricole pas. À noter pour Q41+Q43 : un recreate vide le KV GPU mais aussi une partie du tmpfs du tier (pages comptées au commit) — la marge peut y gagner quelques Go.

**Annexes** : nbaudit **ok** 18:05:48Z (`Image/` vague 2). G: OK. Condensation : **zéro épisode disjoncteur depuis 07:43Z** (6 cycles). WARN `[BaselineManager] NonNominativeBaselineService non disponible` (846 lignes/jour) = **race d'init du process RSM** (le service s'initialise avec succès à la ligne suivante) — cosmétique, périmètre RSM, pas relayé comme incident.

**Coordination** : ACK envoyé à po-2027:CoursIA (section 7.7 du carnet 11b livrée, boucle fermée). [INFO] attribution po-2023 marqué lu.

**Registre** : Q41 ouverte — GPU 0 23 139 (4ᵉ excursion, bande de high-water). Q43 armée, recreate en attente de creux (à coupler avec Q41). Rien de nouveau à arbitrer.

## Fenêtre 18:20Z (28/10) — Q43 EXÉCUTÉE : recreate tier RAM 24→32 GiB (GO user 19:3xZ)

**Arbitrage user reçu en cours de cycle** : « Q41 on essaie de faire tenir en l'état, si ça s'agrave on avisera » (= maintien gpu-util 0.73, ré-ouverture seulement si aggravation) et « Q43 Go pour le recreate » + « Préviens les autres des changements ». Annonce [INFO] cross-postée AVANT le geste (roo-extensions, po-2025, po-2023 — 3/3 OK) avec la fenêtre d'indispo attendue et l'instruction « pas de fallback à déclencher ».

**Pre-flight** : conteneur en place bâti `kv-offloading-size 24`, rendu compose `32` (config-hash → recreate effectif). Fenêtre choisie = creux du cycle (KV 44 %, charge 505 req/2 h en baisse) — conforme à « jamais pendant une saturation ».

**Geste** : `up -d --no-deps vllm-medium-swift15-27b` — recreate le 28/10 **18:20:28Z**. Boot chaud : **health 200 à T+170 s**. Bandeau vérifié : `kv_offloading_size: 32.0`, spec `TieringOffloadingSpec` + `TouchOnHitFsTier` (fs) + `kv_load_failure_policy=recompute` intacts, **KV GPU 831 723 inchangé** (le tier ne touche pas au KV GPU). VRAM à froid GPU 0 **22 363** / GPU 1 **20 978**.

**Liveness** : la première génération post-boot a dépassé 120 s (warm-up CUDA graphs — monde documenté), puis **3 générations de contrôle à 0,18-0,20 s**, sortie correcte. Watchdog : BOOTING patient (5 cycles) → `OK … (warm-up grace consumed — engine is warm)` 18:23:28 → 1 `WEDGE fail 1/2` à 18:23:59 à la frontière du warm-up (toléré, zéro 2/2).

**⚠ /dev/shm 57G/63G (90 %) au réveil — purgé dans la foulée** : le tier 24 GiB de la génération morte cohabitait avec le nouveau 32 GiB (24+32+psm ≈ 57G). Recette appliquée telle quelle (référence 18:21:00, séparatrice propre : ancien mort 18:20:28, nouveau né 18:22) : supprimés le `vllm_offload_752c4a12….mmap` + 4 `psm_*` + 2 `sem.mp-*` orphelins → **33G/63G = 51 %**, health 200. C'est la matérialisation documentée de la fuite « après tout down/up » ; au prochain boot, même geste.

**Marge commit : 86,4 % → 87,8 %** (337,5/384,4, 46,9 Go libres) — le coût +8 GiB du tier est dedans (coût approuvé avec Q43), la tendance ambiante (79,8→80,2→84,2→86,4 avant le geste) reste le vrai moteur de la hausse. Sous la ligne 90 mais à suivre chaque cycle.

**Le tier fs a survécu au recreate** (disque intact, blocs content-addressed) — les replays long-contexte restent servis depuis le disque comme prouvé au restart du 20/10.

**Report de protocole** : la reproduction n°2 de la préchauffe (1ʳᵉ sonde) attendra un moteur installé — pas au cycle 20:11Z (encore en settle post-boot) ; la jouer à 22:11Z ou plus tard.

## Cycle 20:1xZ (28/10) — 🟢 avec réserve : premier cycle sur le tier 32 GiB ; file active derrière les prefills mais AUCUNE saturation

**Prod (post-recreate, ~2 h)** : health 200, RC=0, StartedAt **stable à 18:20:28Z** (aucun restart depuis le recreate), OOM 0, 3/3 Up. `/dev/shm` **33G = 51 %** — plancher du tier 32 G tenu, **zéro orphelin repoussé**. VRAM GPU 0 **23 120** (ligne 23 000 refranchie — desktop + high-water, Q41 close « tenir en l'état ») ; **GPU 1 = 21 786 (+136 vs le plat 21 650 de 7 jours)** — premier mouvement moteur-side ; un point ne fait pas une tendance, à relever aux cycles suivants (le discriminateur fuite-moteur de Q41 était « GPU 1 qui monte »).

**Le tier travaille dès la 1ʳᵉ heure** (compteurs moteur, reset 18:20Z) : **75 Go écrits GPU→CPU, 282 Go relus CPU→GPU en 2 h** (ratio 3,75:1 — l'économie de replay paie), lecture ~4,3 Go/s effective. Préemptions **2** depuis le boot (Δ=0 pendant le cycle — régime calme). KV 42-51 %, file 0, running 3-4. Prefix hits **51,6 %** (19,3M/37,4M) — le cache a été vidé au recreate, il se reconstruit ; la barre de référence reste ~73 %.

**Batterie (observation, PAS la reproduction n°2 — moteur à 2 h)** : préchauffe elle-même **lente (9,79 s)**, sonde 1 **0,17 s** — le coût du premier appel a bien été absorbé par la préchauffe, ce qui **corrobore encore l'hypothèse** sans compter pour la reproduction. Puis la file frappe en milieu de batterie : sondes 2-3 à **15,1/14,2 s**, sonde 6 à 6,55 s (**3/6 >9 s**). **Contexte : zéro saturation** — préemptions Δ=0, KV 42 %, file 0, watchdog 6/6 sondes OK sans un seul WEDGE. Ce sont les prefills long-contexte en vol (4 running, 55 % des corps 100K-1M) qui font attendre les petits appelants — le comportement documenté du dense 27B, jambe 1 du critère 🟠 non franchie (AND) et épisode borné à la fenêtre de batterie. Verdict 🟢, aucun barreau.

**400 window+1 : 2/2 h** (18:38Z — 18 min post-recreate, et 20:05Z ; Bun ~0,9-1,0 Mo). Cumul **~20/24 h** ; cadence 2→4→2→4→2 : le niveau 2 est le mode, les 4 sont des grappes.

**Sécurité** : scanner **ZOLTRAAK/0.2.0** (déjà connu, claudish#718) depuis `31.77.203.199` à 18:49Z — 3 essais `/v1/models`, 2 empreintes nouvelles (`64a6d7ec06fd`, `ccd500f719e4`), **zéro 200** ⇒ bruit d'auth, pas d'alerte. Le poller Go `108.165.33.236` continue (19×401/2 h, clé morte `1f7f6080cb65`). IP user Paris `82.66.89.184` active (2×200, connue).

**★ Marge commit : 89,2 %** (343,1/384,4, **41,3 Go libres**) — 3ᵉ hausse consécutive (86,4 → 87,8 → 89,2), **à 1 pt de la ligne 90**. Deux moteurs : le tier 32 GiB se remplit réellement (75 Go écrits = pages engagées au toucher) et la tendance ambiante (vmmemWSL premier consommateur). À ≥90 % : signal user, aucun travail lourd, pas de bricolage. Le levier structurel (plafond .wslconfig non tenu, blocage structurel connu du dashboard) est hors périmètre vllm.

**Annexes** : nbaudit **ok** 20:05Z (`Image/` vague 2, 5 séries). G: OK (695 Go libres). Condensation : **zéro épisode disjoncteur depuis 07:43Z** (7 cycles).

**Registre** : aucune question ouverte (Q41 close 2ᵉ fois, Q43 exécutée). À relever aux prochains cycles : GPU 1 (21 786 vs plat 21 650), commit (89,2 %), reconstruction du hit-rate prefix.

## Cycle 22:1xZ (28/10) — 🟠 QUALIFIÉ : plateau de saturation KV ~100 min (équilibre stable, pas de dégradation) ; hypothèse préchauffe RÉFUTÉE en régime saturé

**Prod** : health 200 (3 ms), RC=0, StartedAt **stable 18:20:28Z** (zéro restart), OOM 0, 3/3 Up, `/dev/shm` 33G = 51 % (plancher tenu). VRAM GPU 0 **23 141** / GPU 1 **21 786** (inchangés vs 20:1xZ — GPU 1 ne bouge plus, le +136 reste un point isolé).

**★ L'épisode de saturation du soir — timeline complète** (observé en continu grâce à une question user : « le modèle tient la charge ? ») :

- **~20:45Z** : KV GPU **95,9 %**, préemptions **2→18** (Δ+16 en ~40 min), file 3. Sondes spot : **timeout 90 s / 67,7 s / 13,7 s** (décroissantes).
- **20:55Z** : KV 89,8 %, préemptions 19.
- **22:1xZ** : KV **91,9-94,9 %** (plateau), préemptions **19 figées** (Δ+0 en 80 min), running 4 / waiting 2, prefix hits cumulés **84,2 %** (47,4M/56,2M — reconstruction quasi terminée, la barre ~73 % est dépassée).
- **Watchdog** : 3 908 sondes OK, que des `WEDGE fail 1/2` (20:41, 20:52, 21:31, 21:57, 22:01Z — le pic 21:57/22:01 à 4 min d'écart montre la file qui touche les sondes du watchdog aussi), **zéro 2/2, zéro restart**.

**Verdict 🟠 QUALIFIÉ** — les trois conditions du critère AND sont réunies : jambe 1 (Δ préemptions +17 ≥5 sur le cycle), jambe 2 (batterie 6/6 sondes 16,5-78,6 s >9 s), persistance (~100 min ≥30 min). **MAIS la lecture fine est « équilibre stable », pas « dégradation »** : préemptions figées à 19 depuis 80 min = le moteur a trouvé son point d'admission (il régule à la marge KV), health 3 ms, décroissance des délais de sonde au fil de chaque batterie. Aucun geste auto (barreaux suspendus, escalade = user) → **Q44 au registre** : replier un siège haiku (po-2024 d'abord) ou laisser passer la nuit (décroissance nocturne = le pattern ; décision avant le pic de demain soir si récidive).

**★ Reproduction n°2 de la préchauffe — l'hypothèse est RÉFUTÉE en régime saturé** : préchauffe 8,94 s (exclue) puis **sonde 1 = 78,56 s**, sondes 2-6 = 16,5/34,2/57,3/27,5/19,3 s (**6/6 lentes**). Les 2 premières reproductions (KV 42-51 %, moteur installé) tenaient : sonde 1 = 0,16/0,38 s AVEC préchauffe. **Loi à deux régimes, maintenant mesurée des deux côtés** : (a) à basse occupation, le coût du premier appel (CUDA graphs/pipelines) est réel et une préchauffe l'absorbe ; (b) en saturation, la position en file derrière les prefills long-contexte domine tout et la préchauffe est invisible. **Conséquence pratique : pas de préchauffe à armer côté moteur** — le levier en régime saturé est la capacité (KV/sièges), pas la chaleur du pipeline.

**★ EN-TÊTES D'ATTRIBUTION : ARRIVÉS ET VIVANTS** (DM claudish 21:43Z → vérifié par valeur) : la requête test po-2025 reçue **20:38:29Z** `/v1/chat/completions` 200, `x_claudish_client="myia-po-2025:d-dev-claudish"` + `x_claudish_machine="myia-po-2025"` (le hub traduit /v1/messages→/v1/chat/completions — le path moteur n'est pas celui de la demande). **155 requêtes/1 h portent les deux en-têtes** ; première distribution par machine jamais mesurée (2 h, 157 attribuées) : **ai-01 75 · po-2028 72 · po-2023 66 · po-2027 31 · po-2024 7 · po-2025 1**. La mesure « 0/24 456 » du 27/10 est close. Point ouvert côté hub : workspace `:unknown` sur tout le trafic régulier (seul le test rend un workspace résolu). Réponse envoyée à claudish (msg-20261028T221530).

**400 window+1 : 0/2 h** — première fenêtre propre depuis le début du comptage. Cumul inchangé ~20/24 h.

**Sécurité** : zéro nouvelle IP avec 200 (7 j). Deux bruits mineurs attribués : `fp=50829b227a2a` = client Python interne à clé morte (5×401/3 j, `/v1/models` + 1 essai chat, ip locale absente de x_forwarded_for) ; `212.133.41.x` = famille mobile Free (client OpenAI/JS, 6 req/3 j les 24-25/10) — aucun des deux n'est une alerte. Poller Go `1f7f6080cb65` stable (20×401/2 h).

**★ Marge commit : 90,3 % — LIGNE 90 FRANCHIE** (37,4 Go libres ; 89,2 → 90,0 → 90,3 sur la soirée). Conforme au protocole : signalé au user au rapport, **aucun travail lourd**, pas de bricolage — levier structurel (.wslconfig) hors périmètre vllm. Le tier qui remplit (282 Go lus/2 h = pages engagées) + vmmemWSL ambiant.

**Annexes** : nbaudit **ok** 22:06Z — **vague 2 `Image/` ARMÉE** (`subpath:["19451:Image/"]`, 5 séries, 0 échec). G: OK (694 Go libres). Condensation propre (BaselineManager WARN 22:11:44 = course d'init RSM connue, cosmétique). **ERROR global 21:51Z (po-2025:rsm, condensation truncation) qualifié côté moteur** : 0 non-200 hors 401 dans la fenêtre moteur 21:49-21:56Z, les trois « Connection error » coïncident à la seconde avec le recreate hub drainé 21:49Z (334 s, 4 flux coupés) — artefact du fenêtrage recreate, pas le condenseur ni l'engine. [REPLY] posté sur global.

**Registre** : **Q44 ouverte** (🟠 qualifié : repli siège ou laisser passer la nuit — recommandation : laisser passer la nuit, re-décision demain avant le pic du soir si récidive). Q41/Q43 closes.

## Cycle 00:1xZ (29/10) — 🟡 : le plateau a partiellement cédé (80 % à 23Z) puis rebondi à 88 % — soulagement nocturne PARTIEL ; po-2028 tourne toute la nuit

**Prod** : health 200 (3,8 ms), RC=0, StartedAt stable 18:20:28Z (6 h sans restart), OOM 0, 3/3 Up, `/dev/shm` 33G tenu. Préemptions **23 figées** (Δ+0 depuis 23:0xZ), file 0, running 6. KV GPU **88,1 %** — le creux de 80 % vers 23Z était une respiration, pas la décroissance : la charge du soir s'est prolongée après minuit. Watchdog : 2 `fail 1/2` de plus (23:09, 23:34Z), toujours zéro 2/2, 3 903 OK.

**Attribution 2 h (22:11→00:11Z)** — le mix nocturne confirme le diagnostic du repli : **po-2028 153** (le secrétaire tourne toute la nuit) · po-2027 93 · ai-01 50 · po-2024 32 · po-2023 10. **po-2027 soutient ~90 req/2 h même à minuit** — le repli prévu à 17-18Z today est bien le bon barreau. Pas d'ack claudish encore (normal — fenêtre d'exécution dans la journée).

**VRAM** : GPU 0 **23 230 MiB** (nouveau haut du boot, +80 vs 22:1xZ ; Q41 « tenir en l'état », mécanisme connu desktop+high-water) ; GPU 1 21 786 stable (le +136 de la veille reste un point isolé).

**⚠ Marge commit : 90,7 %** (35,5 Go libres) — la série 89,2 → 90,0 → 90,3 → 90,4 → **90,7** continue de monter lentement. Rien d'alarmant (la panne du 24/09 était à 97,8 %) mais **la pente ne s'inverse pas** : si elle dépasse ~93 %, le signal monte d'un cran au user. Aucun travail lourd (règle ≥90).

**400 window+1 : 2/2 h** (22:32Z 914 K, 23:08Z 1,04 M — Bun). 401 : 22× le poller Go (`1f7f6080cb65`), rien de nouveau. Prefix hits cumulés **83,3 %**.

**Annexes** : nbaudit **ok** 00:05Z (`Image/` vague 2). G: OK. Condensation propre (log roulé sur 20261029, aucun événement).

**Registre** : aucune question ouverte. **Action du jour : repli po-2027 sous Minimax fenêtre 17-18Z** (GO user 28/10 ~23Z, signal claudish posté 23:1xZ) — vérifier l'ack aux cycles suivants et mesurer l'effet sur le pic du soir (barre de comparaison : le plateau 🟠 du 28/10, KV 92-95 % ~100 min, préemptions 19-23).

## Cycle 02:1xZ (29/10) — 🟢 détente nocturne confirmée ; ⚠ rafale de 83× 400 window+1, dont ~80 depuis po-2024

**Prod** : health 200 (3,6 ms), RC=0, StartedAt stable 28/10 18:20:28Z (8 h sans restart), OOM 0, 3/3 Up, `/dev/shm` 33G/63G = 51 %. KV GPU **26,4 %** (88,1 % au cycle précédent), running 1, waiting 0. Préemptions 23 → **28** (+5/2 h : jambe 1 du critère 🟠 franchie, mais jambes 2 et 3 non remplies). Batterie petits appelants : **14,46 s puis 0,15 / 0,14 s** — aucune lenteur récurrente, donc 🟢. Watchdog : un `fail 1/2` à 01:41:31Z, zéro 2/2, 3 901 OK. Prefix hits cumulés **84,8 %**.

**★ 400 window+1 : 83/2 h (contre 2/2 h)** — signature exceptionnelle, non une panne moteur. **~80 refus en rafale depuis po-2024 entre 00:55:09 et 01:05:10Z**, cadence ~7-8 s, corps croissant ~940 K → 1,01 M, UA Bun, client `myia-po-2024:unknown` ; plus 1 po-2023 à 00:58:29Z et 3 po-2027 à 01:13Z. Une session déjà >262K retente au lieu d'être reroutée par le hub : coût moteur limité (rejets avant inférence), mais charge utile nulle et correction attendue côté claudish (#431). Signal transmis à claudish. **Conséquence sur l'attribution** : po-2024 113 requêtes/2 h, mais ~80 sont ces 400 — son trafic utile nocturne reste faible ; ne pas interpréter son brut comme une hausse de charge.

**Attribution brute 2 h (00:11→02:11Z)** : po-2024 113 · po-2028 103 (93 corps >500 K) · ai-01 51 · po-2027 33 · po-2023 1 · sans en-têtes 257. 454×200, 83×400, 21×401 (poller Go uniquement). Census IP : aucune IP externe nouvelle avec 200 (site, web1 et poste user Paris connus).

**VRAM** : GPU 0 **23 315 MiB** (+85, haut du boot), GPU 1 **21 786** stable ; mécanisme desktop+high-water inchangé, Q41 maintenue close sous « tenir en l'état ». **Marge commit : 88,5 % (44,1 Go libres)** — inversion nette après 90,7 % ; aucun travail lourd lancé. `vmmemWSL` working set 100,5 GiB.

**Annexes** : nbaudit ok 02:05:45Z (`Image/`, 0 échec consécutif). G: écriture OK. Condensation nominale, aucun événement d'échec. Prouveur : aucune passe. Registre : aucune question ouverte. Repli po-2027 : fenêtre 17-18Z maintenue, ack/exécution à vérifier.

## Cycle 04:1xZ (29/10) — 🟢 nuit calme confirmée ; la rafale 400 window+1 a CESSÉ ; ★ fork dashboard workspace-vllm réparé par merge

**Prod** : health 200 (5,6 ms), RC=0, StartedAt stable 28/10 18:20:28Z (**10 h sans restart**), OOM 0, 3/3 Up, `/dev/shm` 33G/63G = 51 %. KV GPU **21,8 %** (26,4 % au cycle précédent — décroissance nocturne continue), running 1, waiting 0. Préemptions **28 inchangées (Δ+0/2 h)**. Batterie petits appelants : **0,59 / 7,99 / 0,64 s** — une sonde à 8 s en milieu de batterie (pattern « la file frappe aussi en milieu de batterie », déjà vu 18:11Z), aucune lenteur récurrente → **🟢**. Watchdog : un `fail 1/2` à 02:38:03Z, zéro 2/2, 3 899 OK. Prefix externe **84,0 %** de hits (88,9M/105,9M).

**★ 400 window+1 : 1/2 h — la rafale de 83 a CESSÉ.** Fenêtre quasi propre (1 seul 400, po-2027). La session >262K de po-2024 s'est arrêtée d'elle-même vers 01:05Z ou a été corrigée ; pas de réponse claudish encore au signal de 02:17Z (normal, nuit — ack attendu dans la journée). 401 : 24× bruit d'auth connu.

**Attribution 2 h (02:11→04:11Z)** : 411 lignes ~3,4 req/min (creux nocturne). **po-2028 127 req, 127 corps >100K, tous 200** — le secrétaire continue toute la nuit à contexte long, sans erreur ni préemption : le tiers absorbe. po-2027 14 req. Sans en-têtes 270 (moteur local + hub non résolu).

**VRAM** : GPU 0 **23 152 MiB** (−163 vs 02:1xZ, redescend), GPU 1 **21 786** stable. Q41 close (« tenir en l'état »). **Marge commit 87,4 % (48,4 Go libres)** — la détente se poursuit (90,7 → 88,5 → 87,4).

**★ Fork dashboard réparé** : WARN `[DASHBOARD-FORK]` #3482 récurrent dès 02:09:34Z (96 occurrences/2 h) — clé `workspace-vllm (1)` forkée à côté du canonique (apparu autour du cross-post 02:1xZ). **Merge exécuté 04:14:06Z** (seule opération de réparation fork) : 22 msg source ∪ 14 cible → 14 (40 doublons par id, 9 tombstones exclus), statut arbitré lastModified, **source archivée par renommage atomique, retirée des deux artefacts** (GDrive + PG).

**Sécurité** : aucune IP externe nouvelle avec 200. Poste user Paris `82.66.89.184` réapparu (6 req/7 h, 200 — IP connue et réclamée Q13, pas une alerte). Site hairpin 931×200, web1 14×200, interne `?` 690×200.

**Coordination** : inbox 183 non lus (cumul multi-jours) — 5 traités/marqués lus (ACK claudish attribution, rapports worker github-4131 + fleet-manifest, lot CoursIA P0, mention #4395). HIGH re-review ms#1543 (roo-extensions, corrigé 035ad0a1) **laissé non lu délibérément** : review hors périmètre du cycle de surveillance, reste visible pour la prochaine session de travail.

**Annexes** : nbaudit ok 04:05:45Z (`Image/`, 0 échec consécutif). G: OK (710 Go libres). Condensation nominale — disjoncteur half-open retry **réussi 03:11:56Z** (condensation complétée, aucun échec). Prouveur : aucune passe. Registre : **zéro question ouverte**. **Repli po-2027 sous Minimax : fenêtre 17-18Z aujourd'hui** — vérifier l'ack claudish et mesurer l'effet sur le pic du soir (barre : plateau 🟠 du 28/10).

## Cycle 06:1xZ (29/10) — 🟡 montée matinale : KV 40,7 %, une sonde à timeout 90 s ; ⚠ marge commit 93,6 % (ligne d'alerte franchie) ; ★ deux réponses claudish capitales

**Prod** : health 200 (3,2 ms), RC=0, StartedAt stable 28/10 18:20:28Z (**12 h sans restart**), OOM 0, 3/3 Up, `/dev/shm` 33G/63G = 51 %. KV GPU **40,7 %** (21,8 % au cycle précédent — montée matinale), running 5, waiting 0. **Préemptions 28 (Δ+0, 3ᵉ cycle consécutif)**. Watchdog : dernier `OK` frais **06:10:20Z**, aucun nouveau fail depuis 02:38:03Z, zéro 2/2. Prefix : cumulé 141,5 M hits / 252,9 M queries = **55,9 %** ; **tier externe 91,8 M / 111,5 M = 82,4 %**.

**⚠ Batterie : timeout 90 s sur la sonde 1** (curl 000 à 90,017 s), puis **3,91 s et 0,79 s**. Un timeout isolé pendant la montée (KV 40 %, 5 requêtes en vol) — **critère 🟠 non rempli** (Δ préemptions = 0, aucune persistance) ⇒ **🟡**, pas de saturation KV réelle. Pattern connu : la file frappe la 1ʳᵉ sonde.

**⚠ MARGE COMMIT 93,6 % (24,7 Go libres)** — la ligne d'alerte 93 est franchie (87,4 → 93,6 en 2 h, oscillation déjà vue 90,7 → 88,5 → 87,4). **Consommateur mesuré : `vmmemWSL` WS 106,3 Go / privé 164,7 Go** (VM WSL : tiers + page cache) ; `Code.exe` 3,6 Go. Aucun travail lourd lancé (règle ≥90). Levier structurel (`.wslconfig`) hors périmètre vllm — signal user.

**★ Coordination — po-2027:claudish (HIGH) : la population window+1 est dimensionnée et il y a un fait nouveau.**
- 22 corps mesurés, **860 Ko–1,18 Mo ≈ 245–336k tok** : une partie est **structurellement au-dessus de la fenêtre 262 144** — aucun reroutage ne la sauve, le refus moteur avant inférence est correct (#431 skip attendu).
- Reroutage **non corrigeable côté siège** (sidecar en relais pur) : leviers hub-side (`CLAUDISH_FAILOVER_STEP_EXCLUDE` #693 ou plafond d'admission), `max_tokens=128000` demandé par le client à lire au hub.
- **Fait qui pèse sur le plan du soir : hub 502 à 05:57:04Z**, AUTONOMOUS <1 s, NOMINAL sur sonde profonde 16 481 ms à 05:58:28Z. Pendant 84 s, **les 22 tours ont tous échoué localement (Z.AI 429, mur 5 h) : ce siège n'a aucun carburant de repli**. Côté moteur ai-01, **aucun 5xx** dans la fenêtre — le hub n'a rien déchargé sur swift. ⇒ **le repli 17-18Z est plus utile que prévu** : il donne un chemin à une population qui échoue en silence quand le hub bronche. Fenêtre tenue par po-2027, effet à confirmer + corrélation à nos compteurs.

**★ Coordination — po-2024:claudish** : sidecar **NOMINAL** pendant toute la rafale 00:55→01:05Z, aucun défaut local démontré (diagnostic admission/hub confirmé). Mesure livrée : `swift-1.5-27b` = **0 cache-read sur 7 j**, 3,528 G tokens non cachés (414,7–562,6 M/jour). **Réponse mesurée : ce 0 est un trou de champ de reporting** (vLLM n'émet pas `cache_read_input_tokens`), pas une absence de cache — le moteur sert **82,4 %** de hits sur son tier externe (cumulé depuis le boot). Le volume non caché reste une mesure valide, son étiquette est à qualifier.

**400 window+1 : 6/2 h** (3 po-2024 + 3 po-2023, isolés, tous corps >100 K) — **la rafale n'a pas repris** (1 au cycle précédent). 401 : 23× bruit d'auth connu.

**Attribution 2 h (04:11→06:11Z)** : 530 lignes ~4,4 req/min ; **po-2028 139** · ai-01 99 · po-2023 21 · po-2027 6 · po-2024 3 · sans en-têtes 262. Corps : 259 dans 100K-1M, 4 >1M.

**VRAM** : GPU 0 **23 414** (+262, haut de la journée ; ligne 23 000 dépassée, mécanisme desktop+high-water, Q41 « tenir en l'état »), GPU 1 **21 786** stable, GPU 2 156 (CoursIA léger).

**Sécurité** : **aucune IP externe nouvelle avec 200** — site 894, interne `?` 740, web1 12, poste Paris 8 (IP connue et réclamée).

**Annexes** : nbaudit ok 06:06:06Z (`Image/`). G: OK (708 Go libres). Condensation : **1 fallback cloud utilisé 05:33Z (#2719 succès)** pendant la montée matinale — dégradation gracieuse, aucun échec. Prouveur : aucune passe. Registre : **zéro question ouverte**.

## Cycle 08:1xZ (29/10) — 🟢 moteur sain, sondes 0,70 s ; ★ le retrait po-2028 est MESURÉ côté moteur ; ⚠ commit 94,8 % et notre part quantifiée (32,0 GiB)

**Prod** : health 200 (10,4 ms), RC=0, StartedAt stable 28/10 18:20:28Z (**14 h sans restart**), OOM 0, 3/3 Up, `/dev/shm` **33G/63G = 51 %**. KV GPU **75,5 %** (40,7 % au cycle précédent), running 5, waiting 0. **Préemptions 28 (Δ+0, 4ᵉ cycle consécutif)**. Watchdog : 3 900 `OK`, **WEDGE uniquement en `fail 1/2`** (derniers 06:38:52Z et **08:09:00Z**, 2 min avant la batterie), zéro 2/2, zéro restart.

**Batterie : 0,70 / 0,71 / 0,70 s** (préchauffe 0,31 s, exclue) — **aucune sonde lente**, contraste net avec le timeout 90 s de 06:1xZ alors que le KV est plus haut (75,5 % vs 40,7 %). Le KV élevé seul ne prédit pas la lenteur des petits appelants : ce sont les prefills long-contexte en vol qui la font.

**★ Amortissement des préfixes — excellent sur la fenêtre** : cumulé **165,9 M hits / 279,1 M queries = 59,4 %** (55,9 % il y a 2 h), tier externe **92,3 M / 113,3 M = 81,5 %** (82,4 %). **Delta 2 h : +24,4 M hits sur +26,2 M queries = 93,1 % de hit** — la montée matinale est servie par le cache, on ne re-paie quasi rien. C'est le régime attendu du §5quater.

**★ Le retrait de la marche swift haiku de po-2028 est MESURÉ côté moteur.** Global a annoncé le retrait à 07:09:59 (locale = **06:09:59Z**) ; notre journal d'accès place **la dernière requête po-2028 à 06:09:37Z** — soit 22 s avant — et **zéro depuis** (601 requêtes sur 12 h, toutes antérieures). Trafic total 2 h : **417 lignes ~3,5 req/min** contre 530 ~4,4 il y a 2 h = **−21 %**. Attribution 2 h : **sans en-têtes 303** · po-2023 74 · po-2027 38 · po-2024 1 · ai-01 1 · **po-2028 0**. Le délestage se voit donc à la source, pas seulement sur le dashboard du hub.

**⚠ MARGE COMMIT 94,8 % (364,5/384,4 Go, 19,9 Go libres)** — 93,6 → 94,8 en 2 h, **2ᵉ cycle au-dessus de la ligne d'alerte 93**. `vmmemWSL` **privé 161,9 Go / WS 84,3 Go** : le WS a **baissé** (106,3 → 84,3) pendant que le commit montait ⇒ **la croissance n'est pas dans le working set de la VM**, elle est en mémoire engagée non résidente (le ratio privé/WS ~1,9 le dit). Aucun travail lourd lancé.

**★ Notre part, quantifiée pour la première fois** : `/dev/shm` contient **un seul `vllm_offload_*.mmap` de 34 357 542 912 o = 32,0 GiB exactement** (le tier RAM) + ~0,7 Go de `psm_*` tous postérieurs au `StartedAt` (donc non nettoyables par la recette). **Notre contribution au commit de `vmmemWSL` = 32 GiB**, soit ~20 % de ses 161,9 Go. Le levier que nous possédons (`KV_OFFLOAD_GB_SWIFT15=24`, recreate) rendrait **8 GiB = 2,1 points de commit (94,8 → ~92,7 %)**, **insuffisant à lui seul** pour rouvrir la porte d'admission 85 % de GPU 2 (Q-006, ai-01:CoursIA, 08:04:26Z). Mesure postée sur global, cross-post CoursIA — **aucun geste moteur, arbitrage user**.

**⚠ Condensation : le primaire (notre moteur) expire et le cloud prend le relais.** 6 `primary attempt refused` aujourd'hui, dont **4 en 90 min** (07:32:35, 07:36:28, 08:04:52, 08:06:26) contre 13 sur la journée entière du 28/10. Mécanisme lu avec contexte : `"elapsed": "61244ms", "error": "Request timed out."` ⇒ appel de condensation **expiré à 61 s** derrière les prefills longs, bascule cloud, `cloud fallback condensation succeeded` (16 aujourd'hui). Dégradation **gracieuse** (aucune perte), mais **dépense cloud** et sortie dégradée dans la foulée (`Status contradictions detected #1502/#3329`, `LLM did not converge under 15KB → troncature déterministe`, `[COLLISION] stitching concurrent appends`). C'est la famine documentée du dense 27B appliquée aux petits appelants synchrones, pas une panne.

**400 window+1 : 4/2 h** (2 po-2023, 1 po-2024, 1 po-2027, 06:23:21→07:18:44Z, tous corps >100 K) et **14/7 h** — isolés, **aucune rafale** (la rafale de 83 de la nuit n'a pas repris). 401 : 24/2 h, bruit d'auth connu.

**VRAM** : GPU 0 **23 342** (−72 vs 23 414), GPU 1 **21 786** (plat), GPU 2 **191** (vide — c'est Q-006, pas un défaut vllm).

**Sécurité** : **aucune IP externe nouvelle avec 200** — `?` 806/800 · site `90.65.170.144` 660/646 · web1 `37.187.180.135` 12/12 · **poste Paris `82.66.89.184` 4/4** (IP connue et réclamée, Q13).

**Annexes** : nbaudit **ok 08:05:47Z**, `consecutive_failures 0`, **`subpath: ["19451:Image/"]` ⇒ vague 2 (Image/) armée et en production**. G: OK (705 Go libres, 82 %). Prouveur : aucune passe. Registre : **Q45 ouverte** (notre part du commit / tier 32→24 GiB).

## Cycle 10:1xZ (29/10) — 🟢 moteur (sondes 0,65 s, 2 h sans un seul WEDGE) mais ⚠⚠ HÔTE : commit 97,6 %, à 0,2 pt de la panne du 24/09 ; ★ rafale 400 po-2024 à corps CROISSANTS

**Prod** : health 200 (12,9 ms), RC=0, StartedAt stable 28/10 18:20:28Z (**16 h sans restart**), OOM 0, 3/3 Up, `/dev/shm` 33G/63G = 51 %. KV GPU **26,4 %** (75,5 % au cycle précédent — grosse détente), running 2, waiting 0. **Préemptions 28 (Δ+0, 5ᵉ cycle consécutif)**.

**Batterie : 0,65 / 0,63 / 0,64 s** (préchauffe 0,30 s) — le meilleur relevé de la journée, cohérent avec la détente KV. **Watchdog : 3 903 `OK`, AUCUN nouveau WEDGE depuis 08:09:00Z** (deux heures sans un seul `fail`, première fois depuis la nuit).

**Amortissement des préfixes** : cumulé **176,6 M / 292,0 M = 60,5 %** (59,4 % il y a 2 h), tier externe **93,5 M / 115,3 M = 81,1 %**. **Delta 2 h : +10,76 M hits / +12,81 M queries = 84,0 %**. `prompt_tokens_total` +13,04 M.

**⚠⚠ MARGE COMMIT 97,6 % (377,0/386,4 Go, 9,4 Go libres)** — 94,8 → **97,6 % en deux heures (+2,8 pts)**. **La panne du 24/09 (Docker Desktop mort, ~68 min d'indispo, moteur ET hub) était à 97,8 %** : nous sommes à **0,2 point** de ce niveau. `vmmemWSL` privé **169,0 Go** (+7,1 en 2 h) avec un **WS plat à 83,5 Go** ⇒ la croissance est encore de l'**engagé non résident**, mais la direction ne s'inverse pas. RAM libre de l'hôte : **29,5 Go** (contre 41,6 Go il y a 2 h).

**★ Inventaire mesuré des consommateurs `python.exe` (jamais fait avant)** — ~24 Go de mémoire privée engagée :

| Type | Instances | Total | Parent |
|---|---:|---:|---|
| `markitdown_mcp` | **13** | **11,3 Go** | cmd.exe (vivant) |
| `sk_agent.py` | 30 | 6,0 Go | python.exe (vivant) |
| scripts de session (bash) | 2 | **3,0 Go** dont **2,8 Go pour UN script lancé 09:57Z** | bash.exe (vivant) |
| `papermill_mcp` | 24 | 2,3 Go | claude.exe (vivant) |
| `uv` (MCP en cache) | 7 | 0,8 Go | python.exe (vivant) |
| `provider-consoles-reader` | 1 | 0,6 Go | bash.exe (vivant) |

**Tous ont un parent VIVANT — aucun `parent=MORT`** ⇒ ce ne sont **pas des orphelins prouvés**, donc **aucune suppression** (règle : pas de destruction sans preuve de préservation, et GO user requis). Mais 13 instances de `markitdown_mcp` à 870 Mo pièce pour un seul type d'outil est un signal en soi : c'est un candidat sérieux pour rendre ~10 Go. **Question Q46 ouverte** — c'est le levier qui pourrait réellement bouger la marge (5,4 pts), contrairement au tier (2,1 pts).

**★ 400 window+1 : 29/2 h — la rafale po-2024 a REPRIS, avec une signature nouvelle.** 26 requêtes de `myia-po-2024` entre **08:27:06Z et 08:38:41Z** (12 min, cadence 2-3 s) et surtout **des corps qui CROISSENT MONOTONEMENT : 890 524 → 1 002 085 octets** (33 pas, +3 330 o par essai en moyenne). C'est la preuve que le client **ré-essaie en s'ajoutant à lui-même** : chaque tentative échouée s'accumule dans l'historique, le contexte grossit, et le tour suivant est encore plus hors fenêtre. Ce n'est pas un prompt trop long qui arrive une fois, c'est une **boucle qui s'auto-aggrave** — exactement le cas que #431 doit intercepter (router au cloud dans le même tour, jamais de retry local). La rafale s'est arrêtée d'elle-même à 08:38:41Z. Plus **3 po-2023 isolés** (09:19:52, 09:48:46, 10:08:39Z) et **39/7 h** au total.

**Trafic 2 h** : 393 lignes ~3,3 req/min. Attribution : sans en-têtes 300 · **po-2024 50** · ai-01 22 · po-2023 21. Corps : 101 dans 100K-1M, 2 >1M.

**VRAM** : GPU 0 **23 329** (−13), GPU 1 **21 786** (plat, 3ᵉ cycle), GPU 2 **191** (vide, Q-006).

**Sécurité** : **aucune IP externe nouvelle avec 200** — `?` 824/817 · site `90.65.170.144` 602/562 · web1 `37.187.180.135` 14/14 · poste Paris `82.66.89.184` 2/2 (connue).

**Annexes** : condensation **stable — 6 refus du primaire au total aujourd'hui, aucun nouveau depuis 08:06:26Z**. nbaudit ok 10:06:17Z (`Image/`). G: OK mais **610 Go libres** (705 → 610 en 2 h, disque 84 %). Prouveur : aucune passe. Registre : **Q45** + **Q46** (nouvelle).

## Cycle 12:1xZ (29/10) — 🟢 moteur (batterie 0,74/0,60/0,64 s, 2 `fail 1/2` récupérés seuls) ; ⚠ hôte encore à 97,0 % ; ★ le trafic hub s'est effondré (94→23 req/h) et un banc « Nim » non identifié prend 46 % des requêtes

**Prod** : health 200 (**6,4 ms**), RC=0, StartedAt stable `2026-10-28T18:20:28Z` (**17 h 51 sans restart**), OOM 0, 3/3 conteneurs Up (moteur 18 h, watchdog + télémétrie 2 j), `/dev/shm` **33G/63G = 51 %** (plancher = tier 32 GiB), KV GPU **0,0 %** à l'instant, running 0 / waiting 0 — **moteur au repos**. **Préemptions 28 (Δ+0, 6ᵉ cycle consécutif)**.

**Batterie petits appelants** : warm-up 0,69 s puis **0,74 / 0,60 / 0,64 s**. Aucune lenteur récurrente → **🟢**.

**Watchdog** : 3 899 `OK`, deux nouveaux `fail 1/2` — **10:19:14Z** et **11:09:04Z** — **chacun `RECOVERED` après un seul fail** (`BUSY streak was 0`), **zéro 2/2**, aucun restart. Signature de transient sur prefill, pas de wedge.

**VRAM** : GPU 0 **23 214** / GPU 1 **21 786** / GPU 2 **234** MiB (CoursIA libre). ⚠ **GPU 0 dépasse la ligne d'alerte 23 000 alors que le moteur est au repos** : à gpu-util 0,73 le plancher structurel (poids + graphs capturés + pool KV 831 723) est ~23,2 GiB, la ligne avait été calibrée à 0,70. Marge **1 350 MiB**. Ce n'est pas une dérive — c'est le nouveau plancher ; le chiffre à surveiller est le **pic en charge mesuré à 23 600** (24/10), pas le repos. Le boot du 27/10 à 0,73 a d'ailleurs démarré du premier coup (22,25 GiB libres au calcul, Q41).

**Amortissement des préfixes** : cumulé **180,23 M hits / 297,59 M queries = 60,6 %** (60,5 % il y a 2 h), tier externe **94,17 M / 117,36 M = 80,2 %** (81,1 %). **Delta 2 h : +3,63 M hits sur +5,59 M queries = 65,0 %** — contre **84,0 %** au cycle précédent, et tier externe **+0,67 M / +2,06 M = 32,5 %**. La baisse n'est **pas une régression du cache** : la composition du trafic a changé (voir ci-dessous), et un ratio de hit ne se compare qu'à composition constante. À re-mesurer au prochain cycle chargé.

**★ Trafic — le hub s'est effondré, un banc inconnu prend la moitié.** Par heure sur 6 h (total / dont Bun=hub claudish) : 06h 204/**94** · 07h 173/**31** · 08h 221/**73** · 09h 164/**23** · 10h 380/**34** · 11h 244/**23** · 12h 47/**2**. Fenêtre 2 h : **626 requêtes** (5,2/min) dont **seulement 46 Bun** — contre ~155 req/h portant les en-têtes d'attribution il y a deux jours. **Le repli des lanes haiku est donc bien effectif côté moteur**, pas seulement côté hub. Côté outils : `claudish_traffic` auto-sélectionne le conteneur **`claudish-sidecar` d'ai-01**, silencieux par construction (le vrai hub est sur po-2025) — son « 0 requête » est **NOMINAL et ne prouve rien** ; seule la mesure moteur compte ici.

**★ Banc non identifié « jeu de Nim »** : **287 requêtes** `Python-urllib/3.13`, **toutes 200**, sur `/v1/chat/completions`, corps **586-589 o** (max 19 871), modèle **`qwen3.6-35b-a3b`** (l'alias historique), `host=192.168.0.47:5002`, **sans XFF ni en-têtes claudish** ⇒ appel **direct LAN**, pas via le hub. Fenêtre **10:40 → 12:10Z**, rafale à ~2 req/s vers 10:47-10:48. **46 % des requêtes de la fenêtre** à elles seules, et c'est ce qui explique l'essentiel de la dilution du taux de hit (prompts courts et uniques, aucun préfixe récurrent). Charge légitime (200 partout, corps minuscules), mais **un consommateur non inventorié** — à rattacher à une lane.

**400 window+1 : 7/2 h**, **tous `Bun/1.3.14`** (donc claudish) via **90.65.170.144** (hairpin), corps **874 K → 1,80 M o**, machines **po-2023 ×6** et **po-2024 ×1**. Les `body_tail` portent les **schémas d'outils MCP** (`roosync_messages`, `inbox/message`) ⇒ ce sont des sessions Claude Code à très gros harnais. Le corps à **1 797 247 o est un nouveau maximum** (précédent record 1 002 085 o le matin même) : la croissance se poursuit. Rien de nouveau à demander à claudish (#431 déjà déposé), mais le plafond monte.

**401 : 31/2 h** — poller Go `1f7f6080cb65` depuis `116.202.50.2` (clé morte connue, bruit de fond) + 5 sans clé. Aucune clé vivante rejetée.

**Sécurité (7 h)** : **aucune IP externe nouvelle avec des 200**. `90.65.170.144` (hairpin site) 413/460, `37.187.180.135` (web1) 14/14, `51.75.200.22` (web2) 2/2. 1 131 requêtes sans XFF dont 1 118 × 200 (watchdog + banc Nim + sk-agent local). Carte inchangée.

**Hôte — commit 374,9/386,4 Go = 97,0 %** (11,5 Go libres), **en baisse de 0,6 pt** depuis 97,6 % à 10:1xZ. `vmmemWSL` privé **174,6 Go** (+5,6) et WS **88,3 Go** (+4,8), alors que la RAM libre de l'hôte tombe à **23,4 Go** (29,5 il y a 2 h). Autrement dit : **WSL a continué de monter pendant que d'autres process libéraient du commit** — le solde est favorable, la tendance interne ne l'est pas. Toujours au-dessus de la ligne d'alerte 93 % ; **aucun travail lourd lancé** ; la panne du 24/09 était à 97,8 %.

**Condensation** : stable — `cloud fallback condensation succeeded` ×2 à 08:04-08:06Z (pendant la saturation, dégradation gracieuse), **aucun refus depuis 08:06**. Le WARN `[BaselineManager] NonNominativeBaselineService non disponible` (8× à 11:46:51Z) est lu **avec contexte** : le service s'initialise avec succès **6 ms plus tard** dans la même séquence de cold start, et on compte **245 occurrences dans la journée** = une par démarrage d'instance MCP. **Artefact d'ordre d'initialisation, PAS un événement prod** — non relayé.

**nbaudit** : `ok:true`, `runner_rc 0 / deliver_rc 0`, 12:06:12Z, 5 séries, `per_series 3`, **`subpath: ["19451:Image/"]`** ⇒ **vague 2 ARMÉE** (bascule Audio/ → Image/ effectuée). G: **610 G libres** (84 %), écriture OK. Prouveur : aucune passe.

**Coordination** : inbox 1 non-lu = dispatch **machine-only** `myia-ai-01` d'un worker roo-extensions (#4352, claim g7-repair) — **hors périmètre vllm, non traité** (bug #3960). Dashboard : `myia-po-2023|vllm` (CoursIA) répond à un WAKE-CLAUDE 09:39Z sur **son** incident embeddings — 21 conteneurs relancés ~10:04Z, :8004/:8005 revenus, `:6333` fermé et **aucun conteneur qdrant sur ce siège**, `D:` à **4,64 %** (86,5 Go libres). Information sur leur infra, **aucun geste de notre côté**.

**Verdict** : 🟢 moteur (18 h, batterie au meilleur niveau, préemptions figées au 6ᵉ cycle, zéro restart) · ⚠ hôte à 97,0 % (en légère détente, mais WSL monte) · ⚠ GPU 0 au-dessus d'une ligne d'alerte devenue structurelle à 0,73. PR #265.
