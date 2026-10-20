---
description: Cycle surveillance vllm 2h (PROVISOIRE, fenêtre pilote haiku→swift ~23/10) puis retour 6h — la spec versionnée du cron
---

# Cycle surveillance vLLM (2 h PROVISOIRE → 6 h) — spec versionnée

**Le cron ne porte qu'une ligne : `/vllm-surveillance`** (+ fallback chemin ci-dessous). Tout changement de spec passe par un commit/PR sur ce fichier — plus de spec volumineuse inline dans le prompt cron. Convention flotte 07-08/10/2026 (« la spec vit dans le dépôt »), fermée à 3/3 sièges.

Fallback si le skill ne charge pas : lire ce fichier (`d:/vllm/.claude/commands/vllm-surveillance.md`) et l'exécuter tel quel. **Fin de cycle : re-armer le cron** (`CronCreate`, **cadence 2 h à :17 PROVISOIRE**, session-only) et vérifier par `CronList`.

**⏱ CADENCE PROVISOIRE 2 h (user 20/10, ~23→26/10) — pendant le pilote haiku→swift** : charge nouvelle (po-2027+po-2028, ~582 req/h) sur une config qui n'a tenu que le régime nocturne. À chaque cycle : watch §5quater (prefix-hit, preemptions, re-prefill) + **marge commit hôte (§3, seuil 85 %)**. **Retour à 6 h** une fois la fenêtre Minimax passée et la charge stabilisée (reset quota 25/10) : remettre la ligne cadence à « 6 h à :17 » et cette note en ligne de changelog.

## Prod concernée (état 2026-10-09)

**Swift-1.5-Qwen3.8-27B W4A16-AWQ + FP8 KV sur stock `vllm/vllm-openai:v0.31.0` + offload KV RAM 24 GiB (TOUJOURS non pinné)** — localhost:5002, GPUs 0/1 TP=2, profil `medium-swift15-27b.yml`, gpu-util 0.70, **batch 8192 + `max-num-seqs 48` (adoptés 08/10 soir, PR #104 — frais-vs-frais : N=32 +35 %, N=48 907 t/s, prefill +21 % ; avant : 4096/32)**, 262K, **KV 424 610**, noms duaux `swift-1.5-27b` + `qwen3.6-35b-a3b` (alias). Rollback armé : `medium-qwen36-stock-tq.yml`. Conteneurs : `myia_vllm-medium-swift15-27b` · `myia_vllm-watchdog-swift15-27b` (v5, **GEN_TIMEOUT 90 s**) · `myia_vllm-wedge-telemetry-swift15`. **GPU 2 = CoursIA** (training quand occupé, 78-300 MiB quand libre — un GPU 2 chargé n'est PAS une anomalie vllm).

**NE JAMAIS modifier/redémarrer sans GO user.** Jamais `pin=1` avec l'offload (`cudaHostRegister` WSL2 falsifié ×2). Si un boot-OOM récidive malgré 0.70 : fait nouveau majeur — diagnostiquer GPU 0 au moment du boot, ne pas rebaisser mécaniquement.

## ⚠️ Discipline fuseaux (piège vérifié)

Hôte = UTC+2 ; conteneurs, `StartedAt`, logs RSM = UTC ; `docker logs --since/--until` prend l'heure LOCALE. Ancrer sur l'epoch, suffixer les heures citées (`12:32Z`).

**Le piège qui a fabriqué un faux signalement (08/10).** `Get-Date -Format u` rend l'heure **LOCALE** avec un suffixe **`Z`** : on obtient un horodatage qui *ressemble* à de l'UTC et ne l'est pas — décalage de +2 h, du mauvais côté. Un WARN « `:5003` injoignable depuis ~13:2xZ » était en réalité **11:2xZ**, et tombait dans une fenêtre de recréation du siège voisin : signalement faux, coût réel pour le voisin qui a dû le réfuter par la mesure. Même erreur commise sur ce siège le même jour (`Get-Date -Format "…HH:mm:ss 'Z'"`). **Toujours** `(Get-Date).ToUniversalTime()` (PowerShell) ou `datetime.now(timezone.utc)` (Python), et vérifier la conversion **avant** de poster une heure : sur cette machine, **l'heure locale est l'heure Z + 2**. Un signalement horodaté faux est un signalement faux.

## ⚠️ `docker logs --since` INFIBLE depuis le 30/08 → `--tail` uniquement

Le chemin lecture-complète (`--since`, `--tail` géant) voit le segment MORT du journal ; le petit `--tail N` (≈2 500-4 000 lignes watchdog, 20 000 moteur) atteint le segment vivant. Symptôme : `--since 12h | grep -c` = 0 alors que `--tail 2` est frais → c'est CE bug, pas un service arrêté. Borner chaque segment lu et raisonner en comptes absolus.

## (1) Check prod read-only

```bash
curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" --max-time 8 http://localhost:5002/health
docker inspect myia_vllm-medium-swift15-27b --format 'RC={{.RestartCount}} status={{.State.Status}} health={{.State.Health.Status}} StartedAt={{.State.StartedAt}}'
docker logs --tail 20000 myia_vllm-medium-swift15-27b 2>&1 | grep -ac 'out of memory'
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
docker ps --filter name=myia_vllm --format '{{.Names}}\t{{.Status}}'
# /dev/shm hôte (ipc:host — NOUS seuls l'utilisons) : fuite à chaque boot, cf. encadré ci-dessous
docker exec myia_vllm-medium-swift15-27b df -h /dev/shm | tail -1
# wedge : le signal faisant autorité = la sonde décodage du watchdog (pas la télémétrie seule)
docker logs --tail 4000 myia_vllm-watchdog-swift15-27b 2>&1 | grep -E "WEDGE health|CRASH-LOOP|RESTARTING|BOOT-STALL" | grep -vE '^\s'
docker logs --tail 4000 myia_vllm-watchdog-swift15-27b 2>&1 | grep -c 'OK health=200 decode=200'
```

Attendu : health 200 (<10 ms), RC stable, OOM 0, 3 conteneurs Up, sondes OK. **VRAM référence @0.70/N=48** : à froid après boot GPU 0 19 937 / GPU 1 18 930 MiB, **en charge 21 400-21 600 / 20 400-20 700** (la config 48 flux élargit les graphs sous trafic, ~1,5 GiB de plus qu'à froid). **GPU 0 > 23 000 MiB → alerter** (marge < 1,5 GiB = boot-OOM au prochain restart + pagination WDDM qui imite un wedge) — **ne pas remonter gpu-util**.

**Dérive VRAM — mesurée puis INFIRMÉE comme tendance (09/10)** : 21 408 (08/10 22:5xZ) → **21 594** (09/10 04:47Z, +186) → **21 435** (09/10 10:47Z, **−159**). Le cycle de 04:47Z avait lu une pente de +30 MiB/h et annoncé une alerte ; **la pente n'a pas tenu** — c'est un **high-water d'allocateur** (charge soutenue qui garde les graphs à leur forme max), **pas une fuite**. Leçon : **deux points ne font pas une pente** ; ne pas annoncer une extrapolation (« J+2 ») sur 6 h de données. La charge qui produit ce high-water est mesurée en §5ter. `StartedAt` stable vs cycle précédent ; un déplacement avec RC=0 = restart externe → identifier la cause (penser à `autoheal` : `docker ps | grep -i heal`, redémarreur concurrent INVISIBLE, RestartCount flat).

Comportement documenté, PAS une panne : toute requête ≥ ~100K tokens **affame les nouvelles arrivées** (dense 27B) — le watchdog tolère 90 s. Depuis le batch 8192 la fenêtre recule (prefill +21 %), mais un WEDGE `fail 1/2` (jamais 2/2) pendant une sonde prefill ou un gros prompt est le comportement attendu, **y compris déclenché par nos propres sondes de banc** (vécu 08/10 21:38 et 21:46, deux `fail 1/2` pendant les sondes 157K, zéro restart). Un WEDGE pendant un gros prefill = attendre le post-mortem avant d'agir ; le moteur est sain.

### ⚠ `/dev/shm` hôte — fuite à chaque boot (mesurée 20/10)

Le moteur est le **seul conteneur `ipc=host`** de la machine (51 conteneurs inspectés le 20/10) : **tout ce qui vit dans ce shm est à nous**. Plancher légitime = **le tier lui-même** (32 GiB en nvfp4) + les blocs `psm_*` transitoires de la génération courante (~0,5-1 GiB). **Tout le reste est orphelin** : chaque génération morte laisse ses `psm_*` (25/161/241 MB) et ses `sem.mp-*`, et **la fenêtre d'éval fs du 20/10 a laissé 2 × 8 GiB de `vllm_offload_*.mmap`** (les tiers du profil d'éval, mtime antérieurs au boot prod). État mesuré avant nettoyage : **49G/63G = 77 %**, dont **21,8 GiB d'orphelins sur 58 fichiers** ; après : **32G = 51 %, 31G libres**, moteur health 200 en 5,9 ms.

Un `/dev/shm` plein = ENOSPC pour les blocs suivants → **crash moteur** ; sous charge (fenêtre Minimax) la marge est le facteur limitant. **Seuil d'alerte : > 45G utilisés (≈ 72 %).**

**Recette** — ne supprime que les fichiers **antérieurs au démarrage du conteneur** (donc jamais un fichier mappé ni la génération vivante) :

```bash
docker inspect myia_vllm-medium-swift15-27b --format '{{.State.StartedAt}}'   # borne (UTC)
MSYS_NO_PATHCONV=1 docker exec myia_vllm-medium-swift15-27b sh -c \
  'find /dev/shm -maxdepth 1 -type f ! -newermt "2026-10-20 10:25:05" -print -delete'
```

Effet de bord mesuré : les pages tmpfs comptant dans le commit hôte, les 17 GiB rendus ont fait passer la marge commit de **86,4 % → 83,4 %** (sous le seuil §3). **Nettoyer après toute fenêtre d'éval et après tout `down/up`** — c'est là que la fuite se matérialise.

## (2) Condensation (logs roosync)

**Emplacement (trouvé le 09/10, il manquait depuis deux cycles)** : `%TEMP%\roo-state-manager-logs\roosync-<YYYYMMDD>.log` (le nom suit le DÉMARRAGE du process, pas minuit — lire jour + veille ; des rotations `…-96.log`…`…-99.log` coexistent, prendre le nom exact du jour).

```bash
L="$LOCALAPPDATA/Temp/roo-state-manager-logs"
grep -aiE 'condensat' "$L/roosync-$(date -u +%Y%m%d).log" | grep -avE 'LLM config:'
grep -aE '\[(ERROR|WARN)\]' "$L/roosync-$(date -u +%Y%m%d).log" | grep -avE 'LLM config' | tail -20
```

Burst ≥ 12 évts/min = jest (paths `dashboard-test-*`/`__test-data__`) ; vrai échec prod = minute isolée à 1-2 évts. `cloud fallback condensation succeeded` isolé pendant une indispo vLLM = nominal (dégradation gracieuse).

**CORRECTION (09/10) — le banner « Condensation LLM config » EST dans le log**, à `[INFO]` toutes les 5 min (`primary=qwen3.6-35b-a3b @ http://localhost:5002/v1 | cloud-fallback=… key=OK`). La spec affirmait le contraire (« jamais dans le log, stderr ») : **faux**, et cela faisait passer un battement de santé pour une absence de signal. Le banner **est le battement qui prouve que la condensation est configurée sur notre moteur** — le lire, ne pas l'exclure. Ce qu'on **filtre** (`grep -v 'LLM config:'`), ce sont les 200+ battements pour ne garder que les **événements** réels (aucun le 09/10 : condensation nominale).

**Ce que le log contient aussi (et qui n'est PAS vllm)** : `[MessageManager] Error reading message file during parallel cache build: G:\…msg-…json` (messages du 02/10 illisibles) + `[MessageManager] Inbox cache rebuild hit its budget after 6200/6281 files (#3205)` + `[#3292] Explicit-id population approaching starvation threshold: 100/100`. **Périmètre RSM, pas vllm** — les relayer au propriétaire, ne pas les traiter ici.

**⚠️ Seuil de condensation 266k — RÉSOLU (10/10) : c'est l'auto-compact de Claude Code, PAS un paramètre RSM.** Réponse de roo-extensions depuis le code (web1, 09/10) : **RSM n'a aucun seuil en tokens** — ses condensations sont bornées **octets** (51 200 max) ≈ ~20k tokens. Le « 95 % de 280k = 266k » cité par le user est le seuil d'**auto-compact de Claude Code lui-même**. Le vrai risque résiduel : une session Claude Code routée vers Swift croît vers 262-266k et dépasse **notre** fenêtre (262 144) **avant** de déclencher son propre auto-compact — v0.31 répond alors 400 **sans jamais rogner** (mesuré 09/10, 3 sondes frontière, PR #110). **La parade n'est pas côté moteur** : règle déposée sur claudish #431 ([c.6091076678](https://github.com/jsboige/claudish/issues/431)) — sur 400 « maximum context length », router **cette requête** vers le cloud dans le même tour, **jamais de retry local** (l'entrée ne se rogne pas, elle se contourne). **Signature à continuer à chercher dans le census §5ter : `status 400` + `body_bytes > 100 000`** — première occurrence mesurée = à rendre au user ET à claudish immédiatement (elle prouverait que #431 n'est pas encore armé ou ne couvre pas ce cas). L'escalade RSM d'origine est close sans action.

## (2bis) Producteur de pré-audit (nbaudit) — une ligne, à lire chaque cycle

```bash
cat "$LOCALAPPDATA/nbaudit/last-result.json"
```

`{"ts":…,"runner_rc":…,"deliver_rc":…,"ok":…,"consecutive_failures":…}` — état du **dernier** passage horaire. `ok:false` ou `consecutive_failures > 0` = **le producteur de dossiers preaudit est mort** (les consommateurs Hermes/NanoClaw retombent alors en audit *full-read*, plus coûteux pour eux). Un `consecutive_failures` qui monte = panne installée, pas un incident d'une heure.

**Couverture : le runner ne sert que les séries configurées** (`17107`, `17239`, `17357`, `17692` 02-ML-Cours, **+ `19451` scopé à `Audio/` = vague 1 COMPLÈTE depuis le 08/10 soir, PR #101**). Un carnet **hors périmètre** n'aura jamais de dossier preaudit, panne ou pas — vérifié 08/10 : `2.8b-Theorie-PAC-Lean.ipynb` (02-ML-Cours), réclamé par Hermes, était **absent des 3 checklists** d'origine, donc son « aucun dossier preaudit » n'était **pas** l'outage. Avant d'expliquer une absence par la panne, vérifier que le carnet est **dans le périmètre** (`gh api repos/jsboige/CoursIA/issues/<n>` — REST, pas GraphQL). Le scoping par série vit dans `nb_audit_hourly.ps1` (`--subpath 19451:Audio/`) ; **le runner saute de lui-même tout carnet `.ipynb` touché par une des 50 PRs ouvertes les plus récentes** (dérivé par passage — la liste statique PR #19868 de l'ordre avait pourri en 5 h, superseded par #19854) ; échec du scan = fail-open. **Vagues 2-4 autorisées** (feu vert user 08/10 soir) : basculer le préfixe (Image/, Video/, puis prose/queue selon l'ordre convergé), une vague à la fois. **Fanion vague 2 (posé 10/10)** : armer `Image/` (25 carnets, 0 cochés) quand Audio/ n'a plus de **livrables en attente** — état 10/10 05:20Z : **34/38 cochés**, 4 restants ; ne pas basculer tant qu'un carnet Audio non-coché et non-PR-skippé peut encore recevoir un dossier (sinon on l'abandonne sans pré-audit). Gel 09/10 soir levé : il avait pour motif l'armement failover, **suspendu depuis par arbitrage user**.

**⚠️ Lectures GitHub en REST, jamais en GraphQL.** `gh issue view` / `gh pr view` / `gh api graphql` dépensent le budget **GraphQL**, qui est compté **par UTILISATEUR** : toutes les lanes de toutes les machines partagent **un seul seau de 5 000/h** et le vident (35 min d'arrêt de ce producteur le 08/10). ⚠️ **`gh api rate_limit` ne voit pas cette panne** — il annonçait `.resources.graphql` = 4967/5000 pendant que *tout* appel GraphQL, même `{viewer{login}}`, était refusé : ne pas s'y fier pour diagnostiquer. **REST est par token et reste libre** → `gh api repos/<owner>/<repo>/issues/<n>` (+ `/comments --paginate --jq '.[]'`). Corrigé par PR #98 ; **tout outil de la flotte lisant GitHub en GraphQL est exposé au même seau**, ce qui peut expliquer des « rate limit » ailleurs.

**Pourquoi ce fichier existe et pourquoi il faut le lire ici.** Le passage a été mort **37 h** (07/10 00:05Z → 08/10 13:05Z, 32 échecs) sans que rien ne le dise : la tâche planifiée se lisait `Ready` / `LastTaskResult 0` / 0 passage manqué. **Le code de sortie ne peut pas remonter** — la tâche passe par `wscript.exe` sur `nb_audit_hourly.launcher.vbs`, dont le `Run(…, 0, False)` n'attend pas et ne rend rien (mesuré 08/10 : un VBS de cette forme lançant un enfant qui sort 7 rend **0**). Seule une trace **écrite** survit. Corollaire général : **un livrable qui sort 0 après l'échec de son producteur cache la panne** — vérifier la santé du producteur, jamais le statut du livrable.

## (3) Santé G:/ DriveFS (fail-closed RSM si pathologique)

```bash
df -h /g | tail -1
timeout 30 sh -c "echo probe > '/g/Mon Drive/Synchronisation/RooSync/.vllm-probe.tmp'" && rm -f '/g/Mon Drive/Synchronisation/RooSync/.vllm-probe.tmp' && echo OK || echo KO
```

Écriture KO/timeout > 30 s = DriveFS hang → le noter au rapport, ne pas boucler sur les posts dashboard (ils échoueront), et le signaler à roo-extensions (le dossier #4131 suit la bascule PG).

**Marge commit hôte — seuil 85 %** pour tout travail lourd (prouveur, `lake build`) :

```bash
powershell -NoProfile -Command "\$m=Get-CimInstance Win32_PerfRawData_PerfOS_Memory; \$c=\$m.CommittedBytes/1GB; \$l=\$m.CommitLimit/1GB; 'Committed {0:N1}/{1:N1} GB = {2:N1}% (marge libre {3:N1} GB)' -f \$c,\$l,(100*\$c/\$l),(\$l-\$c)"
```

**⚠️ Seuil FRANCHI le 09/10 (90,1 % : 352,9/391,8 Go)** — il était à 78,1 % six heures plus tôt. **Premier consommateur : `vmmemWSL` (~93 GiB)**, pas `Code.exe` (le levier « fermer des fenêtres VS Code » ne pèse plus le premier poids — même constat que po-2025, fait converger par deux machines). Cause probable : la VM WSL2 (Docker Desktop + `tmpfs` du tier KV 24 GiB + le training CoursIA de GPU 2) qui monte par bouffées. La panne du 24/09 était à **97,8 %** ⇒ à 90,1 % **on ne lance AUCUN travail lourd** et on **signale**, on ne bricole pas (`.wslconfig`/`wsl --shutdown` = hors périmètre vllm, et `wsl --shutdown` tuerait moteur **et** hub).

## (4) Prouveur Lean (#1453) — état : AUCUNE passe armée

Passe 23 close le 08/10 : **le blocage est un trou de bibliothèque, pas de harnais** — `oneStepWitnesses` retourne `[]` (`ReidemeisterCombinatorial.lean:166-172`) → `verifyMoves ≡ decide (d₁ = d₂)` réflexif seul, `verifyMoves_sound` vacuous. **Ne pas relancer de passe sur `unknotting_11n102_upper`** tant que l'énumération n'est pas implémentée (verrou = réponse coordinateur / tâche bibliothèque). Leçon consignée : vérifier l'organe AU SOURCE avant d'en faire un chemin de clôture — une docstring décrit l'intention, pas le code.

Si une NOUVELLE passe est armée un jour (lanceur `myia_vllm/scripts/adoption/prover_1453/`, pattern `run_passNN_*.sh`) : lancement **seulement** si marge < 85 % **ET** à la fenêtre **:30Z** (jamais à cheval sur le runner nbaudit :05Z — chevauchement = ~50 tok/s/flux et le cap client 240 s tue le tour de réflexion) ; `cp` backup `.pre-passNN` avant ; **écrire la marge lue dans le post de lancement** ; compter TRUE_PLACEHOLDER avant/après ; verdict honnête sur #1453 + DM coordinateur (si la passe échoue, le dire tel quel).

## (5) SÉCURITÉ standing — delta IP externes avec 200

Source : `/logs/error_sources.jsonl` DANS le conteneur (toutes requêtes). **Aggréger sur `x_forwarded_for`, JAMAIS sur `client`** (le champ client ne montre que la passerelle Docker 172.19.0.1).

```bash
MSYS_NO_PATHCONV=1 docker exec myia_vllm-medium-swift15-27b python3 -c "
import json,time,collections
now=time.time(); W=7*3600; tot=collections.Counter(); ok=collections.Counter()
for line in open('/logs/error_sources.jsonl',errors='ignore'):
    try: r=json.loads(line)
    except: continue
    if now-(r.get('ts') or 0)>W: continue
    ip=(r.get('x_forwarded_for') or '?').split(',')[0].strip()
    tot[ip]+=1
    if (r.get('status') or 0)==200: ok[ip]+=1
[print('%16s tot=%-5d 200=%d'%(i,t,ok[i])) for i,t in tot.most_common(25) if ok[i]>0 and not i.startswith('172.')]
"
```

**Carte connue (mise à jour 08/10)** : **Jamin** = `88.183.141.187` + `176.172.94.128` (RGAA, confirmé user 08/10 — PAS Candy) + IPv6 déclarée `2001:861:8ac2:e3f0:8602:b5b8:31ab:6424` (jamais observée au 08/10 ; toute apparition = Jamin, ne pas alerter) · **Candy : PAS encore observée** (annoncée plus mesurée ; attribution future par clé scopée, plus par IP) · site (hairpin) `90.65.170.144` · poste user Paris `82.66.89.184` (Q13 répondue 08/10 : « sans doute mon IP parisienne », SUPPOSÉ, Free, trafic arrêté depuis le 01/10) · po-2027 `92.150.81.115` + **`92.184.103.95` (IP sortante mobile, auto-signalée par po-2027 le 10/10 — 2ᵉ IP connue de la machine, ne pas alerter)** · web1/web2 `37.187.180.135`/`51.75.200.22` · familles mobiles Free `92.184.x`/`88.183.x`. **Nouvelle IP non réclamée avec des 200 = alerte user immédiate dans le cycle + entrée registre.** Jamais d'attribution par géoloc registre seule (leçon : l'IP « allemande » était le site lui-même). Le census complet vit en mémoire : `project_vllm_external_ip_census_20261007.md`.

### (5bis) Consommateurs externes — clé scopée `external-vllm` (Jamin / Candy)

**Empreinte `45BDC569E1E3`.** Noms autorisés : `frognano-4b`, `mini`, `swift-1.5-27b`, `qwen3.6-35b-a3b` — les deux derniers sont **le même moteur** (vérifié 08/10 : les deux rendent `200` en OpenAI **et** en Anthropic, `served_by=qwen3.6-35b-a3b`). Tout cloud (`glm-5.3`…) = **403**, sans clé = **401**. Cap **2 requêtes concurrentes** par clé.

**ACTIVE et EFFECTIVE — preuve produite le 08/10 14:02Z.** Redeploy drainé du hub (`OUTCOME success (reload config: allowlist += swift-1.5-27b)`, conteneur parti 14:02:02Z), puis **matrice vérifiée avec la clé résolue** par po-2025:claudish sur `/v1/messages/count_tokens` (route **sans inférence**) : `swift-1.5-27b` / `mini` / `frognano-4b` → **200** ; `glm-5.3` (hors allowlist) → **403 `[InboundKey] … is not permitted`**. C'est le **refus distinct et nommé** qui prouve que `inboundKeys` est chargé et mord — pas un 401 générique. Empreinte inchangée `45BDC569E1E3`. Historique : le status « INERTE » datait d'avant le redeploy ; le 401 de po-2025 n'était pas une preuve (retiré par eux) — **seule la matrice avec contre-épreuve tranche**. **Piège d'instrument** (vécu par claudish) : `config.json` porte `${EXTERNAL_VLLM_KEY}` — une *référence*, pas la valeur ; tester le littéral rend un 401 « invalid proxy authentication » qui imite une clé morte. **La remise aux externes est débloquée** (geste user).

À relever chaque cycle (mandat user 08/10 : « tu relèveras régulièrement les traces ») :

- **Delta d'IP avec 200** (script §5) — Jamin a ≥2 IPv4 mesurées + 1 IPv6 déclarée ; **Candy n'a jamais été observée**, sa première attribution se fera **par clé**, pas par IP. Toute IP hors carte avec des 200 reste une **alerte user immédiate**.
- **Volumétrie** : le census **par clé** (`inbound_key`) vit côté hub → à demander à po-2025:claudish (campagne 30 j en cours, non bloquante). Côté moteur, on ne voit que l'IP.
- **Piège de lecture** : `max_tokens < 256` ⇒ **réponse vide** (`content` null, budget entièrement consommé par le thinking) alors que le moteur est sain — ne pas le lire comme une panne de lane (leçon partagée 08/10, claudish s'y est fait prendre aussi). Idem sur `/v1/messages` : le premier bloc est `type=thinking`.
- **Levier si dérapage** : la clé se **révoque seule**, sans toucher `VLLM_API_KEY_MEDIUM` — c'est le contrôle prévu (le user soupçonne Jamin de chercher à « maxer »).

### (5ter) Census de trafic moteur — qui charge `:5002` (ajouté 09/10)

**Pourquoi.** Le 09/10, ce census a révélé **~60 req/min soutenues (≈3 600/h)** sur la prod — dont la composition n'était documentée nulle part (la baseline d'adoption était ~550 appels/**jour**). C'est aussi lui qui **explique le high-water VRAM** du §1. Le middleware `error_source_capture` journalise `user_agent`, `auth_prefix`, `body_bytes`, `model`, `path`, `status` — **tout est déjà là**, il suffit de l'agréger.

**⚠️ Le log TOURNE vite** (4 395 lignes ≈ **1,2 h** au débit mesuré) : une fenêtre demandée il y a 6 h **n'existe plus**. Toute question sur une fenêtre passée doit être posée **avant** rotation, ou couverte par une agrégation persistante.

```bash
MSYS_NO_PATHCONV=1 docker exec myia_vllm-medium-swift15-27b python3 -c "
import json,time,collections,hashlib
now=time.time()
rows=[json.loads(l) for l in open('/logs/error_sources.jsonl',errors='ignore') if l.strip()]
rec=[r for r in rows if now-(r.get('ts') or 0)<=12*3600]
span=max(1,(rec[-1]['ts']-rec[0]['ts'])/60) if rec else 1
print('lignes %d  fenetre %.1f min  ~%.1f req/min'%(len(rec),span,len(rec)/span))
print('--- UA ---')
for u,c in collections.Counter((r.get('user_agent') or '?')[:34] for r in rec).most_common(8): print('%6d %5.1f/min %s'%(c,c/span,u))
print('--- cle (empreinte SEULE, jamais de prefixe publie) ---')
for a,c in collections.Counter(r.get('auth_prefix') or 'NONE' for r in rec).most_common(4):
    print('%6d %s'%(c,'NONE' if a=='NONE' else 'fp='+hashlib.sha256(a.encode()).hexdigest()[:12]))
print('--- taille de corps ---')
b=collections.Counter()
for r in rec:
    n=r.get('body_bytes') or 0
    b['<1K' if n<1000 else '1-10K' if n<10000 else '10-100K' if n<100000 else '100K-1M' if n<1000000 else '>1M']+=1
for k in ['<1K','1-10K','10-100K','100K-1M','>1M']: print('%6d %s'%(b[k],k))
print('--- statut ---'); print(collections.Counter(r.get('status') for r in rec).most_common(4))
"
```

**Attendu / à interpréter** :
- **Sondes watchdog** `ua=curl` ~2/min, `"Count to twenty."` : normal.
- **Condensation** `ua=Python-*` « expert en synthèse … 15 Ko » : normal, horaire.
- **Clé unique `fp=<MEDIUM>`** en tête : c'est la clé partagée des consommateurs internes (hub claudish, sk-agent, Roo, sondes). **Vérifier l'identité de la clé AVANT toute conclusion de sécurité** — comparaison côté hôte (`.env`), en n'imprimant que des `sha256[:12]` :
  ```bash
  cd /d/vllm && python -c "
  import hashlib
  vals={}
  for l in open('myia_vllm/.env',encoding='utf-8',errors='replace'):
      if '=' in l and not l.strip().startswith('#'):
          k,v=l.strip().split('=',1); vals[k.strip()]=v.strip().strip('\"').strip(chr(39))
  tok='<les 16 premiers car. du auth_prefix, SANS le Bearer>'
  for k,v in vals.items():
      if 'KEY' in k.upper() and v: print('%-28s fp=%s match=%s'%(k,hashlib.sha256(v.encode()).hexdigest()[:12],v.startswith(tok)))
  "
  ```
- **Clé hors des connues (`MEDIUM`, `MEDIUM_VL`, `MINI`, `MICRO`, `external-vllm`) = anomalie** → §5 (alerte) + registre.
- **⚠️ MAIS une clé connue ne dit RIEN sur le client (arbitrage po-2025, 09/10).** Le hub claudish appelle `:5002` **sous MEDIUM pour *tous* ses clients** : la clé est un **seuil de sécurité**, pas un **discriminateur d'identité**. Pour attribuer une charge, le discriminant est la **TAILLE DES CORPS** — un client à contexte long (~57 K) se distingue par `body_bytes > 100 Ko`, jamais par la clé. Ne pas conclure « c'est interne, donc c'est bénin » : un client interne qui boucle (cf. latch Zoo #4025, ~20 h) pèse autant qu'un externe.
- **`body_bytes` > 100 Ko = requêtes long-contexte** : ce sont elles qui pilotent le high-water VRAM. Un client qui boucle là-dessus (cf. latch Zoo #4025) se voit ici **avant** de se voir en VRAM.

### (5quater) Charge Minimax/failover — amortissement des préfixes (ajouté 19/10, mandat user « surveiller l'utilisation comme le feu »)

**Contexte.** Le moteur est onboardé comme marche de failover **Minimax** (mur quota ~23/10 09:00Z, reset 25/10 23:05Z — claudish #543) ; cap tandem Cand/Jamin doublé ; **maxing PERMANENT plusieurs jours attendu** = sessions récurrentes dont les préfixes sont déjà vus. Régime sain : chaque tour ne paie que son **delta** (gros prefill payé une fois, replay tier 51× moins cher). Pathologie : **re-prefill massifs récurrents** quand le working set dépasse GPU KV 429K + tier ~685K ≈ 1,11M tok → LRU jette du contexte vivant. Position lane (19/10, #543 c.6277127297) : cap progressif 2→3→5, routage-cloud avant le mur.

**À relever chaque cycle tant que la charge Minimax est active :**

1. **Hit rate prefix** (compteurs `/metrics`, reset à chaque boot — noter l'âge moteur) :
   ```bash
   curl -s http://localhost:5002/metrics | grep -E '^vllm:(prefix_cache|gpu_prefix_cache|cpu_prefix_cache).*(queries|hits)_total'
   ```
   En croisière longue (sessions récurrentes) le ratio hits/queries doit **monter**. Ratio plat ~0 avec des corps >100K en jeu = **amortissement cassé** → alerte user + registre.
2. **Signature re-prefill (census §5ter)** : même IP/empreinte avec `body_bytes > 100K` **répétés à cadence de tour** (au lieu de delta 1-10K) = le client re-paie tout son contexte à chaque tour.
3. **Préemptions** : `vllm:num_preemptions_total` (reset au boot) qui **monte en continu** = saturation KV réelle (plus discriminant que la VRAM). `kv_offload_cpu_cache_usage_perc` reste une gauge in-flight, PAS une jauge de remplissage.
4. **Wedge watch renforcé** : les prefills géants affament les sondes (GEN_TIMEOUT 90 s) ; `WEDGE fail 1/2` pendant un gros prefill = attendu ; `2/2` → vérifier d'abord que des sessions longues sont en vol avant de conclure à une panne.
5. **Escalade = décision user, jamais auto** : si préemptions récurrentes + hit rate cassé + re-prefill massifs → (a) **marge commit d'abord** (§3 — le tier 24 GiB peut monter 32-48 GiB SI marge < 85 %), (b) **ré-éval NVFP4** (re-commande 1 var `KV_DTYPE_SWIFT15=nvfp4`, record PR #123 : KV 750K mais −13/−17 % débit). **TQ k8v4 dominé par nvfp4 sur chaque métrique mesurée — ne jamais re-tester.**

## (6) Triage coordination

Inbox `roosync_messages` : lire l'adressé `myia-ai-01:vllm`, répondre/ack, `mark_read` ciblé (**`bulk_mark_read` exige un filtre `from`/`subject` — sans filtre il marque TOUT l'inbox**). Adressage machine-only = bug #3960 : ignorer les dispatchs d'autres workspaces. Dashboards : lire global + workspace-vllm ; rendre les lignes demandées (split-brain, matrices, etc.) ; relayer à l'user les réponses aux issues ouvertes par la lane.

## Clôture (OBLIGATOIRE, cet ordre)

1. **Commit + PR AVANT le rapport** — jamais annoncer du non-committé (`gh auth switch -u jsboige`, PR `--repo jsboige/vllm`, merge `--merge` ; si gh GraphQL limited → fallback REST).
2. `append` **[DONE]** sur workspace-vllm (prod, condensation, G:, prover, sécurité, registre restitué).
3. Ligne Active Work Log dans `MEMORY.md` (mémoire `~/.claude/projects/d--vllm/memory/`).
4. **Re-arm le cron** (6 h à :17, session-only, prompt = `/vllm-surveillance` + fallback) + `CronList` vérifié.
5. Zéro question bloquante au user : toute question va au registre `user-question-registry.md` (5 champs, dont l'attendu et le mécanisme de retrait), restitué en bloc au rapport.

Règles standing : famille spec-dec CLOSE (0/4, ne pas re-tester) · jamais de valeur de clé en clair (empreintes sha256[:12] uniquement) · infra d'un autre workspace = demander, pas appliquer.
