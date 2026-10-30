# Runbook embeddings po-2026 (Qwen3-4B AWQ) — leçons, pièges, recovery

Siège : po-2026, service d'embeddings de la flotte (Roo Code indexation + OWUI RAG ×7).
**Siège migré sur po-2023 le 14/10/2026** (panne matérielle de po-2026) ; le reste de ce
document vaut pour le siège courant — le profil et les chemins sont inchangés.
Profil : `myia_vllm/configs/docker/profiles/embeddings-qwen3-4b-awq-po2026.yml`.
Issue de rattachement : jsboige/vllm#71. Tout chiffre ci-dessous est **mesuré** au siège.

## Pièges vLLM v0.23.0 — runner pooling (appris en crash-loop, 30/09)

1. **`cudagraph_mode: FULL` interdit avec pooling** : vLLM override silencieusement en
   PIECEWISE puis assert au boot. Ne pas le poser.
2. **PIECEWISE exige la compilation** : `mode: NONE` (pas de compile) + PIECEWISE =
   crash au boot. Les défauts (compile AOT + PIECEWISE) sont le seul combo valide.
3. Le champ s'appelle **`mode`** (pas `level`) pour les options compilation.
4. `gpu-memory-utilization 0.75` + profiling des graphs ≡ **0.72 effectif** (les graphs
   consomment ~0,4 GiB hors pool). Passer à 0.7798 pour un KV équivalent à 0.75 nominal.
5. **`--enforce-eager` retiré (30/09)** : les OOM torch.compile de 2026-04 (16 Go VRAM,
   pré-RAM-64Go) ne se reproduisent plus. Gain mesuré : warm 142-250 ms → 15-17 ms (~x10),
   VRAM 12,2 → 6,7 Go. Ne pas le remettre sans re-tester ces OOM.

## Chaîne & non-régression

- Public : `https://embeddings.myia.io` (IIS ARR po-2023) → sidecar `:8004` → vLLM `:8005`.
- **Contrat stable** : `qwen3-4b-awq-embedding`, dims 2560, URL publique — tout changement
  = réindexation flotte (6,2M+ pts). Vérif post-restart : POST réel via l'edge public doit
  rendre `dims: 2560`.
- Cohérence v0.21↔v0.23 vérifiée à l'upgrade : cos = 0.999998 sur 4 textes de référence.

## Recovery éprouvé

| Symptôme | Geste | Preuve/leçon |
|---|---|---|
| Docker daemon down (HTTP 000 + GPU 0 MiB) | Kill Docker Desktop + `wsl --shutdown` + relance, attendre ~160 s | [[docker-daemon-crash-detection]] siège |
| Sidecar lent/wedge (« SLOW-not-dead », `/health` 200 mais inférence lente) | `docker restart embedding-proxy-logger` | Angle mort n°9 : un health 200 ne prouve rien, toujours tester un POST réel |
| Pull Docker Hub 401/403 | Workaround **skopeo en conteneur détaché** via socket Docker (`MSYS_NO_PATHCONV=1`), image injectée au store local | Keyring Docker Desktop HS (non touché) ; procédure éprouvée 30/09 (image 29,8 Go) |
| Surchauffe | Gouverneur `gpu-thermal-governor.ps1` (schtask /5 min) : cap 1600 MHz à 88 °C×2, restore à 78 °C×3 | Validé en charge réelle (2 caps productifs 28-29/09 + flush 01/10 sans cap) |
| **Installation du pilote NVIDIA → moteur mort, redémarrage impossible** | **Reboot Windows** (reco) ou `wsl --shutdown` + relance Docker | **Pont CDI stale** : `no driver store paths found` — voir section dédiée (incident 07-08/10, 11 h+) |

## Installation du pilote NVIDIA → pont CDI stale (incident 07-08/10/2026, 11 h+)

**Ne JAMAIS toucher au système pendant une installation de pilote** (ni restart de conteneur,
ni kill — le GPU disparaît du device tree en cours d'install, toute action est futile et peut
perturber l'installeur). Le gouverneur gère RAM/thermique indépendamment.

**Déroulé mesuré** : NVIDIA App 18:50:36 → GPU retiré du device tree (`nvlddmkm` STOPPED exit 31,
`nvidia-smi` absent) → EngineCore vLLM crash `EngineDeadError` à 18:56 → l'APIServer parent
s'arrête PROPREMENT ensuite (`Application shutdown complete` — c'est une **conséquence**, pas
un arrêt volontaire) → conteneur `exited`.

**Piège d'interprétation** : `RestartPolicy=unless-stopped` + `RestartCount=0` + logs d'arrêt
propres font croire à un **arrêt explicite** (lu ainsi par au moins deux lanes pendant l'incident).
Faux : chaque tentative de relance échoue sur le pont CDI **avant** toute incrémentation du
compteur. `RestartCount=0` ne prouve pas un arrêt volontaire.

**Blocage** : le pilote revient, le GPU est sain, mais le redémarrage du conteneur échoue de
façon **déterministe** :

```text
failed to create automatic CDI modifier: failed to generate CDI spec for mode "auto":
failed to create discoverer for WSL driver: no driver store paths found
```

Le CDI Docker/WSL référence l'ancien pilote ; il ne se reconstruit qu'au **restart de la VM WSL**.

**Reprise** : reboot Windows (le conteneur `unless-stopped` remonte seul), OU GO user explicite
pour `wsl --shutdown` + relance Docker (~160 s — **coupe les co-locataires** du parc :
claudish-proxy, myia-mcp-proxy, qdrant_local, hermes… d'où le GO et non le geste unilatéral).

**Validation obligatoire** (angle mort n°9) : warm <0,5 s ×2 **ET** POST réel public avec
**dims 2560** — jamais un `/health` seul (le sidecar rend 500 en 7 ms pendant tout l'incident,
ce qui est fidèle mais ne dit rien de la reprise).

**Effet flotte** : `roosync_search` (semantic) et `codebase_search` tombent en text-fallback
(`embedding_api_error` / `embedding_unreachable`) — la recherche sémantique de TOUTE la flotte
dépend de ce moteur, pas seulement l'indexation Zoo.

## Gouverneur thermique (mutualisable laptop GPU)

- Undervolt persistant au boot : schtask `GPU-Undervolt-Cap` → `nvidia-smi -lgc 210,1800`
  (RTX 3080 Ti Laptop : 86-87 °C → 68 °C max, 117 W vs 150 W, latences inchangées).
- Gouverneur : see `myia_vllm/scripts/gpu-thermal-governor.ps1`. Heartbeat horaire dans
  `logs/gpu-governor.log` (silence = tâche morte). Rollback : `schtasks /Delete /TN GPU-Thermal-Governor /F`.
- Garde RAM anti-freeze incluse : <1,5 Go libres + instance RSM >4 Go → kill (doctrine 09-07 :
  ne jamais proposer la fermeture des fenêtres VS Code).

## Garde de vivacité (livrée 29/10/2026, PR #275)

Un garde-fou **autonome** qui prouve la chaîne par une **vraie requête** et relance le
seul conteneur impliqué. Ni lui ni le vigie 5 h ne dépendent d'une session ouverte.

| | |
|---|---|
| Script | `myia_vllm/scripts/monitoring/embedding_liveness_guard.py` |
| **Copie d'exécution** | `D:\Production\Embeddings\bin\embedding_liveness_guard.py` — **hors dépôt** |
| Enregistrement | `myia_vllm/scripts/monitoring/register_liveness_guard.ps1` (tâche utilisateur `/IT`, 5 min, sans élévation, dry-run `-WhatIf` obligatoire) |
| Journal | `D:\Production\Embeddings\logs\liveness-guard.jsonl` (une ligne par passe) |

**Preuve exigée** : POST réel `:8004` avec la clé → HTTP 200 + `model == qwen3-4b-awq-embedding`
**+ 2560 dims**. `:8005` en localisation seule. Whisper (`whisper-webui`) **au niveau HTTP
seulement** — il est épinglé GPU 1, qui est le GPU d'une autre lane : jamais de transcription.

**Rails** : jamais de restart d'un conteneur de moins de 5 min (leçon watchdog v5) ; skip si
Docker Desktop est arrêté délibérément ; cap 2 restarts/conteneur/h ; **401/403 = alerte, pas
de restart** (un restart ne répare pas une clé fausse).

### ⚠️ Une tâche planifiée ne s'exécute JAMAIS depuis le worktree git

**Incident du 29/10/2026 : la garde est morte 4 h 17 sans un seul signe.** Elle pointait vers
`myia_vllm/scripts/monitoring/embedding_liveness_guard.py` **dans le dépôt**, et un simple
`git checkout main` a retiré le fichier du disque. Chaque passe échouait en
`LastTaskResult=2` (`ERROR_FILE_NOT_FOUND`) **pendant que `Get-ScheduledTask` affichait
`Ready`** — la forme la plus mauvaise pour un moniteur : la tâche a l'air armée et ne rend
que du silence.

**Règle** : un dépôt est une source, pas un emplacement d'exécution. L'enregistrement
installe une **copie vérifiée par SHA256** hors dépôt et pointe la tâche dessus.

**Vérifier la vivacité de la garde** (et non pas son existence) :

```powershell
Get-ScheduledTaskInfo -TaskName 'Embeddings Liveness Guard' |
  Select-Object LastRunTime, LastTaskResult, NextRunTime
Get-Content 'D:\Production\Embeddings\logs\liveness-guard.jsonl' -Tail 1
```

`LastTaskResult = 0` **et** une ligne datée il y a moins de 10 min = garde vivante. Une tâche
`Ready` avec `LastTaskResult = 2` est **morte** — c'est exactement le piège ci-dessus.

## Mémoire host (SPOF n°4/5, résolu 05/10)

Mécanisme des gels : `vmmemWSL` ballonnait (22,4 Go RAM / 35,3 Go commit) → pagefile-thrash
→ host gelé ~13,5 h (04/10). `autoMemoryReclaim=gradual` actif mais insuffisant seul.
**Levier appliqué 05/10 (GO user)** : `.wslconfig` `memory` 24→16 Go (swap 32 Go inchangé →
plafond commit VM 48 Go = levier résiduel, touche la remediation Mathlib → arbitrage seul).
Backup pré-cap au siège. Post-cap mesuré : vmmemWSL 4,3 Go.

## Sidecar capture (attribution de trafic)

Toujours en container (jamais `python` host : firewall → ARR 503 collant ~19 min).
Commande complète dans le profil compose. Logs `embedding_requests.jsonl` : hash sha256
par input → détecte les boucles de réindexation (incident 250 Go : 49,8 % de doublons).
`MAX_LOG=100000` (la flotte tire ~235 req/min).

## Disque (2026-10-06, jsboige/Maintenance#70)

- Image v0.21.0 purgée (48,4 Go) après migration du sidecar sur v0.23.0 (le sidecar
  tournait SUR l'image v0.21 — dépendance à vérifier avant toute purge d'image).
- L'espace hôte C: n'est restitué qu'à la compaction du VHDX docker_data
  (arrêt Docker + `Optimize-VHD` élevé) — à fenêtrer si besoin.
