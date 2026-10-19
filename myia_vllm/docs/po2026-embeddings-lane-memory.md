# Mémoire opérationnelle — lane embeddings po-2026 (portée par le dépôt)

Condensé des mémoires de session validées 08-10/2026. Chaque point = doctrine éprouvée
en production. Détail technique : `embeddings-po2026-runbook.md` (même dossier).

## Rôles & doctrines

- **RÔLE VIGIE** (incident 250 Go, 08/2026) : ce service est le point de convergence
  du trafic d'indexation de TOUTE la flotte. Une cadence anormale n'est PAS
  automatiquement une bonne nouvelle — un pic sans cause identifiée = ASK immédiat
  (« qui consomme ? »), JAMAIS « pic nominal, flotte très active ». Une mesure locale
  ne prouve jamais qui consomme (l'IP client n'est pas loggée).
- **Santé Qdrant AVANT analyse de trafic** : `healthz` sur les DEUX endpoints (LAN
  `192.168.0.47:6333` + proxy `qdrant.myia.io`). Proxy down + LAN OK = problème
  po-2023, pas Qdrant ; les deux down = Qdrant down (ROUGE, wake signal).
  Timeout ≠ connection reset : RST = service mort, timeout = réseau/VM.
- **Dashboards** : vigie → workspace `Embeddings` ; collectif vllm → workspace `vllm` ;
  cross-workspace → `global` + cross-post désignant LE point de rendez-vous unique ;
  DM = décision/urgence uniquement. `[WAKE-CLAUDE]` = global uniquement.
- **Crons session-only** : meurent à chaque reboot/fermeture VS Code. Le skill vérifie
  via CronList (idempotent, jamais de doublon). Réarmement sur instruction user
  (doctrine ghost-crons : le harness peut ressusciter un cron session-only au reboot —
  un cron dupliqué = leak à auditer). Chaîne historique po-2026 : `831c80b9` →
  `53fb1b20` → `890d21a3` → `7f3d5392` (session Embeddings, retirée à la passation).
- **UAC** : l'agent déclenche le popup (`Start-Process powershell -Verb RunAs`), le
  user clique Oui — JAMAIS demander à l'user de taper la commande. Dry-run AVANT tout
  geste UAC-touching, sortie du dry-run postée sur le dashboard.

## Signature & causes racines (toutes vérifiées)

- **Docker daemon down** : HTTP 000 + GPU 0 MiB. Cold start : kill Docker Desktop +
  `wsl --shutdown` + relance, ~160 s, conteneurs auto-restart.
- **Wedge sidecar « SLOW-not-dead »** : `/health` 200 ne prouve RIEN (angle mort n°9).
  Signature : vLLM `:8005` direct = 200 MAIS `:8004` refuse + logs docker gelés à une
  seq fixe. Cause racine (06/10) : écriture log synchrone gRPC-FUSE gelait l'event
  loop. **Fix = LOG_POOL (2 threads)** — un stall volume ne pendent plus que les logs.
  Fallback : `docker restart embedding-proxy-logger`.
- **Gels host SPOF n°4/5 (04/10)** : `vmmemWSL` 22,4 Go/35,3 Go commit → pagefile-thrash.
  `autoMemoryReclaim=gradual` insuffisant seul. **Fix (05/10, GO user)** : `.wslconfig`
  `memory=16GB` (swap 32 Go inchangé → plafond commit VM 48 Go = levier résiduel,
  touche la remediation Mathlib → arbitrer seul). Premier test : host jamais gelé
  depuis ; la pression s'est manifestée côté partage de fichiers (wedges).
- **Restarts VM docker-desktop (06/10, RCA parrain — logs backend vérifiés)** : les
  2 « restarts daemon » (09:01:47Z OOM-kill interne ; 11:03:15Z mort vpnkit-bridge →
  VM recréée) sont des **restarts de la VM WSL** — **acte 2 du cap vmmem** : l'hôte
  ne gèle plus, la pression se déverse DANS la VM (OOM + vpnkit ; wedges sidecar =
  même famille). Zéro perte, auto-start propre, daemon prêt 4,4 s. **Si restarts VM
  >2/jour sous charge → lever = mémoire INTERNE VM** (co-locataires quantconnect/qc-mcp
  ×4, vLLM host-side, sidecar), à instruire avec Maintenance.
- **Le jsonl du sidecar se VIDE à chaque restart** (`on_startup` unlink) — copier avant
  tout restart si le brut doit être conservé.
- **Toujours `--no-deps`** en `docker compose up -d` sur le compose prod (sinon il
  tente de créer qdrant-local → pull proxy → abort).

## Conventions mesures

- **Double baromètre obligatoire** : `roo_tasks_semantic_index` ET total partitions
  `ws-*` (une vague ws-* de +1,3 M a été invisible sur roo_tasks seul, 01/10).
- Warm embeddings : 2 requêtes (cold ~600 ms après restart), cible <250 ms, >300 ms
  sur 2 cycles = JAUNE. Vérif post-incident = POST réel via l'edge public, dims 2560.
- `SYNC_STALE` = artefact heartbeat, pas une panne (la progression Qdrant est la
  preuve). Convention flotte : po-2023/po-2024, jamais po-203/po-204.
- Compteur sidecar `/​_proxy_stats` reset à chaque restart — dater les resets avant
  d'interpréter un delta.

## Historique utile (condensé)

- 08/2026 : incident 250 Go (pics 3 273 pts/h étiquetés « nominaux » à tort —
  ~50 % doublons d'une boucle d'indexation) → doctrine vigie.
- 09/2026 : rotation clé API (exposition git) — préfixes 8 chars max ; un one-shot
  critique ne vit jamais dans un cron session-only. Upgrade vLLM v0.23.0 (30/09) :
  warm ~x10, VRAM 12,2→6,7 Go ; image v0.21 purgée 06/10 (re-pull skopeo si besoin).
- 10/2026 : SPOF n°5 (freeze 13,5 h) → cap vmmem ; wedges → LOG_POOL ; siège vllm
  posé 06/10 (PR #78), passation vigie de la session historique Embeddings.
- 06/10 PM : **PASSATION VALIDÉE** — 1er cycle du nouveau siège `D:\Dev\vllm`
  (rapport 13:35 locale sur workspace-Embeddings, JAUNE borné ; erratum titre :
  « 15:35 » → 13:35). La vague ~150 k pts/h = indexation du clone du siège
  (`ws-fc1a37694049b210`, 65e partition, one-shot) — **le démarrage d'un siège
  du fork génère sa propre vague d'indexation** (valable pour po-2025). LOG_POOL
  test dur passé (484 req/30 min sous charge, zéro gel). 2e restart daemon
  11:03:28Z — RCA : restarts VM WSL, voir « Signature & causes racines ».
  Cron vigie : nouveau siège `f10b0717` (**5 h à :17**, mandat user — coordination
  siblings incluse) ; historique `7f3d5392` retiré par embeddings-99 (clôture
  postée sur workspace-Embeddings).
- 07/10 : **boucle Zoo chiffrée + série restarts VM = 6 le 06/10** (dernier
  23:41:27Z, StartedAt identiques ×5 conteneurs). Démonstration par les nombres :
  restart 23:41Z → vague ré-indexation 6 195 req / 407 k inputs en 2 h, puis 2e
  vidage ~07Z (démarrage matinal machines) → net ws −286 k. Le couple
  restarts↔vidages s'auto-alimente ; levier structurel = mémoire VM (option (b),
  portée par Maintenance vers l'arbitrage user). Scan (d) livré : profil SANS
  réservation hôte — leviers `mem_limit` (à étalonner sous vague, risque
  OOM-kill moteur), batched/seqs, swap VM. **Mandat adoption user 07/10**
  (lead po-2025) : compteurs chaque cycle + recrutement consommateurs externes
  (baseline postée : externe hors flotte = 0). 7 erreurs sidecar transitoires
  (0,1 % trafic) — candidat : logger le type d'exception.

## Passation po-2023 (2026-10-20)

po-2026 est HS matériel depuis le 09/10 ; la flotte a migré l'embedder vers
**po-2023** (déploiement Maintenance/ai-01, cutover IIS 14/10 00:50Z, GPU pin
bi-GPU PR #118). **Nouveau siège de vigie : lane `vllm` @ myia-po-2023** (repo
`D:\Dev\vllm`), inaugurée le 20/10 (mandat user), cron **5 h à :17** (continuité
du mandat). Doc siège : `seats/po2023-embeddings.md`. Ce document et le runbook
restent la référence technique (pièges v0.23, rôle vigie, recovery) — seuls les
chemins changent : `.env` prod = `D:\Production\Embeddings\compose\.env`,
rapports → dashboard workspace `vllm`. Gaps de migration consignés au registre
du nouveau siège : gouverneur thermique absent, `QDRANT_API_KEY` absente,
HF token vide dans le conteneur courant (bénin, cache local).
