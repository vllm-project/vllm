# Épisode 4 — notes de matière (période 21/09 → 21/10) — NON PUBLIÉ

GO user 21/10 ~23:35Z (« MAJ de l'espace consacré à votre workspace comme dans le passé,
pas mal de choses à raconter »). Épisode 3 = PR #18501 (29/09, chapitres 12-18 → conclusion 19).
Épisode 4 = chapitres 19+ (renumérotés à l'assemblage), conclusion réécrite.
Proposition postée sur workspace-CoursIA 21/10 23:4xZ. Concertations lancées :
po-2025:vllm (FrogNano) + po-2023:embeddings (incident+migration) — échéance cycle suivant.

## Chronologie (source = MEMORY.md Active Work Log + topic files, tout est VÉRIFIÉ sauf mention)

- **22-23/09** v0.30.0 promu (+23-25 % N=16 chaud) ; **spec-dec landscape CLOS 0/4**
  (DFlash crash boot #51581, DSpark 7,3 % acceptance + mur UVA V2-runner, MTP bloqué
  loader bug #58807 — noté upstream, maintainer dessus ; ngram −67-88 %) ; DFlash2 overlay
  #51684 validé en fenêtre (16/17 gates) mais ratios 0,22-0,66× → MoE conservé.
- **25/09** Swift-1.5-Qwen3.8-27B promu (GO user, pivot qualité) — le 27B dense remplace le MoE.
- **28/09→05/10** épisodes doc + adoption (nb-audit pilotes, prover 1453, SkAgent 600 s).
- **05/10** v0.31.0 + **offload KV RAM tier 24 GiB** (pin=1 falsifié ×2 : monolithique ET
  chunked #51081 — walking-dead engines) ; replay 51× mesuré ; règle tier ≥ 1,5× GPU KV.
- **07-08/10** **incident embeddings** (panne HW po-2026) — section sœur attendue ;
  nbaudit mort 37 h (VBS exit-code Avalé) ; batch 8192 + seqs 48 adoptés (08/10 soir,
  +35 % N=32, user green light).
- **09/10** TQ k8v4 rejeté sur Swift (−33 % N=1) ; auto-compact 266k ≠ RSM (résolu) ;
  marge commit 90 % → discipline seuils 85/95.
- **14/10** **migration embeddings po-2026 → po-2023** (siège dédié, runbook, pont
  NVIDIA→CDI) — section sœur attendue.
- **19/10** **5ᵉ mode de wedge : saturation KV** (KV 100 % + fetch pending = discriminateur) ;
  bump v0.32.0 ; nvfp4 évalué (rejeté −13-17 %) ; **fuite VRAM GPU 0 ~21 Go** driver-level,
  évaporée seule (Q20 WHEA/nvlddmkm ouverte) ; user : onboardé comme failover Minimax.
- **20/10** **nvfp4 ADOPTÉ pour la capacité** (3 wedges saturation en une nuit ; KV ×1,75 =
  750 747) ; tier 24→32 ; /dev/shm purge 21,8 GiB (fuite par génération) ; **6ᵉ mode :
  41×502 = boucle restart auto-entretenue** (restart vide le cache → re-prefill 260K →
  re-saturation) ; **éval tier fs : SURVIE AU RESTART PROUVÉE** (replay disque 1,41 s vs
  11 s, 15/15 chunk-hits) ; 7ᵉ mode : stores 77 MB/s + swap WSL (tmpfs→swap hypothèse).
- **21/10** **fenêtre multi-tier GPU→RAM→FS** (RAM 8 + fs 48, commit 92→87,5 %) ; 2ᵉ
  indispo par WEDGE-famine post-fenêtre (leçon watchdog) ; **pruner : 3 bugs interceptés
  AVANT armement** (regex ancré = voyait 0 bloc, niveau _g<gen> manquant, fichiers root) ;
  sonde suppression-vivant PASSÉE ; tâche 30 min armée (1ᵉʳ passage Result=0) ; pilote
  haiku→swift actif (maxConcurrency 8, ~228 req/h mesurés) ; mandats veille (spec-dec
  conditionnel + modèles : flint 40B-A2.5B SWE-Pro 89,7 sans vLLM/AWQ, Sage-A1.5
  qwen3_5_moe+VLM servable).

## Sections sœurs (à insérer après retour des sièges)

- FrogNano (po-2025:vllm) : naissance du siège, milestone 21/10 (premier consommateur
  régulier po-2028, 4 lanes, ~30× économie lecture, sorties relues obligatoires).
- Embeddings (po-2023) : incident 07/10, migration 14/10, topologie actuelle.

## Leçons candidates (la conclusion aura 12+ → réécrire)

1. Un pruner « OK » peut être aveugle : 0 bloc vu sur 40 GB — tester l'organe sur le VRAI
   layout avant l'armement (écho : vérifier l'organe au source, prover #1453).
2. Chaque génération morte laisse des orphelins (/dev/shm) — mais un down PROPRE rend tout :
   le contraste 20/21-10.
3. Le watchdog peut tuer un moteur sain-qui-travaille (famine ≠ panne) — la porte
   « prefill en vol » reste à poser.
4. La capacité KV achetée (nvfp4 −17 %) vaut moins que la CAPACITÉ DE SURVIE (fs tier) :
   le bon angulaire est le coût d'un contexte REVU, pas la taille du cache.
5. La montée en charge se pilote par le hub (busy-skip), pas par le moteur.

## Mécanique restante

Draft chapitres → worktree `docs/vllm-story-4` → extension journal + conclusion →
PR atomique (provenance chiffres dans le body, grain MED/docs, prev = merge #143
`26c76ee6c7`) → pings (workspace-CoursIA + Epic #4427 + tracker à ouvrir) → review ai-01.

## Section sœur FrogNano — REÇUE 22:31Z (option b, texte de po-2025:vllm, insérer quasi tel quel)

**FrogNano-4B (tier mini, po-2025:vllm — issue jsboige/vllm#70)**
- 07-08/10 : naissance du siège po-2025:vllm — hébergement FrogNano-4B-2609 sur la 3080 Ti 16 Go,
  seul moteur du parc sur ce GPU, servi derrière claudish (`frognano-4b` + alias `mini`).
- 08/10 : double quant NVFP4 poids + KV fp8 → KV vivante 421 582 tok (×4,26 cumulé), N=16 ×1,91
  en débit ; puis correctif d'UTILITÉ : thinking OFF par défaut côté serveur — sinon ×34 latence
  et ×100 tokens pour la même réponse, `content` vide sous un plafond client court.
- 19/10 : moteur v0.32.0 (GO user), boot 211 s, KV −5,9 % immatériel à 2-7 % d'occupation.
- 21/10 : premier consommateur régulier (po-2028, 4 lanes : digest flotte tick 12 h + pré-tri de
  commentaires PR, ~30× d'économie sur la part LECTURE) ; puis premier organe consommateur dans
  le dépôt : PR CoursIA #21885 (`pr_comment_pretriage.py` — 2 894 tok in, 5,6 s, mesurés sur #21352).
- Leçon du siège : le vrai obstacle à l'adoption n'a jamais été le débit — c'est le coût mental
  côté consommateur. Règle : le mini lit et pré-trie, ne verdict jamais ; sorties relues.
- Fenêtre de mesure 21/10 : Spark-X2.6-4B-INT4 testé et refusé (non-servable v0.32.0, arch
  absente du registre) — place de 2ᵉ choix ouverte à un futur bump.

## Stopgap condensation — voie propre reçue (22:24Z, à armer avec roo-extensions)

Hub claudish DIRECT (`models.myia.io/v1`, modèle `frognano-4b`, auth = `CLAUDISH_PROXY_KEY`
déjà en place) — JAMAIS le sidecar (n'injecte pas la clé cluster, errata po-2025 du 21/10 00:20Z).
Pas de clé brute à distribuer. Cadriage : fenêtre 32 K, thinking OFF (défaut serveur désormais),
consigne forme fixée + « cite verbatim ». Latence mesurée 1,9-3,3 s digests courts.
Statut : en attente de roo-extensions (fix client + accord config RSM ai-01 = infra partagée).
