# Lane po-2025:vllm — instructions du siège (FrogNano-4B, tier mini)

## Identité

- **Ce siège** (clone `D:\dev\vllm`, branche de travail `po2025/frognano-seat`)
  = lane **FrogNano-4B / tier mini de la flotte**, sibling #1 sur le fork
  `jsboige/vllm` (cf. `../../../FLEET-WORKSPACE.md` — le lire d'abord, il porte
  les conventions flotte).
- **Mission** : héberger `microsoft/FrogNano-4B-2609` sur po-2025 « à haute
  vitesse, disponibilité et parallélisme » avec garde thermale obligatoire,
  puis le faire utiliser à la hauteur de sa capacité (issue
  [jsboige/vllm#70](https://github.com/jsboige/vllm/issues/70) = suivi unique).
- Inauguré le 2026-10-06 (mandat user). Phases : P0 inventaire (faite) · P1
  workspace+profil+garde · P2 déploiement+gates · P3 bench capacités · P4
  exposition flotte (claudish + domaine Mini + sk-agent dédié) · P5 adoption.
- **Modèle servi depuis le 2026-10-08** : `capyctl/FrogNano-4B-2609-NVFP4`
  (modelopt 0.47.0 MIXED_PRECISION — MLP en NVFP4, projections attention/GDN en
  FP8, reste BF16), servi en **W4A16 via les noyaux Marlin** parce que notre
  SM 8.6 n'a **pas** de tensor cores FP4 (le producteur a mesuré la même chose
  sur SM 8.9). **+ FP8 KV depuis l'après-midi du 08/10** (`--kv-cache-dtype
  fp8`, backend FLASHINFER — validé sur SM 8.6 par essai). Cumulé mesuré sur la
  journée, mêmes script et machine que la référence BF16 : **KV 91 853 →
  391 748 tok (×4,26, 11,96× pleine longueur)**, N=16 561,6 → **1 073,4 t/s
  (×1,91)**, N=32 938,0 → **1 642,4 t/s (×1,75)**, 68 °C max, tool-calling OK,
  0 erreur, capture CUDA graph OK. **Tier RAM offload : évalué et REFUSÉ**
  (libre hôte 7,2 Go mesuré < garde 4 Go + besoin ~9 Go ; un tier abordable
  serait < KV GPU = puits write-only). **Leçons** : (1) la décision du 06/10
  « NVFP4 = Blackwell, donc inexploitable » reposait sur une hypothèse trop
  large — la garde réelle de vLLM est `has_device_capability(75)` ; (2) le
  fp8 KV « à tester » sur SM 8.6 passe proprement. Rollbacks armés :
  **1 pas** `mini-frognano-4b-po2025-kvauto-rollback.yml` (NVFP4, KV auto) ·
  **2 pas** `…-bf16-rollback.yml`. Détail et limites :
  `_iteration_log/nvfp4_trial_po2025/NOTES.md` + `_iteration_log/fp8kv_trial_po2025/NOTES.md`.
  **+ thinking désactivé par défaut** le 08/10 après-midi (KV re-mesurée
  **421 582** tok, 12,87×) — voir la section Surveillance ; dossier
  `_iteration_log/thinking_default_fix_po2025/NOTES.md`.

## Surveillance

- **Cron vigie siège : 4 h à :33** (session-only — CronList d'abord, réarmer si
  absent après un reboot ; le prompt du cron porte la check-list complète).
- Rapports → **dashboard `workspace-vllm`** (RDV des 3 sièges vllm). DM
  toujours adressés `machine:workspace` (bug RooSync #3960). Escalade
  cross-workspace → dashboard `global` + crossPost.
- Télémétrie thermique : task user-level `vllm-po2025-thermal-logger`
  (log `myia_vllm/_logs/gpu-thermal.log` ; battements à :03, **horaires en
  pratique** alors que la doc d'origine dit /5 min — à réconcilier).
  **Gouverneur SYSTEM VIVANT depuis le 06/10 15:53 locale** (VÉRIFIÉ le 08/10
  13:04 via **ses propres artefacts** : 69 heartbeats continus dans
  `_logs/gpu-governor.log`, `cap_applied=OK (210,1800)`,
  `governor-install-result.txt` Running/Ready, `hot=0` — il a observé la porte
  NVFP4 à 68 °C en restant `mode=normal`). **Se prononcer sur le gouverneur
  UNIQUEMENT via ses artefacts** (`gpu-governor.log`, `gpu-governor-state.json`,
  `governor-install-result.txt`), **JAMAIS via `Get-ScheduledTask` non élevé** :
  l'énumération est aveugle aux tâches SYSTEM — le 08/10 ce piège a fait
  propager « gouverneur absent » (faux) du matin au soir. Discriminant si
  besoin : `schtasks /query /tn GPU-Thermal-Governor-po2025` → **« Accès
  refusé » = la tâche existe** (protégée) ; « fichier introuvable » = absente.
  Le logger user-level est **gardé volontairement** : il logue
  `engine=up`/`max24h`/util%, que le gouverneur ne logue pas. Scripts :
  `scripts/gpu-thermal-{logger,governor}-po2025.ps1`.
- **Thinking DÉSACTIVÉ PAR DÉFAUT depuis le 08/10 après-midi**
  (`--default-chat-template-kwargs '{"enable_thinking": false}'`) — correction
  d'**utilité**, pas de performance : le tier mini sert du travail mécanique, le
  raisonnement y est un coût pur. **Mesuré AVANT correctif**, même question et
  même moteur : thinking ON → **2,78 s / 200 tok** ; OFF → **0,08 s / 2 tok**
  (×34 en latence, ×100 en tokens pour la même réponse) ; et **`content` VIDE**
  sous un plafond client courant (`finish_reason=length`,
  `reasoning_tokens = completion_tokens`). C'est le pire cas de promotion : le
  tier « bon marché » se présentait 30× plus lent, 100× plus cher et **muet**.
  Un consommateur qui veut délibérer l'active **par requête**
  (`chat_template_kwargs: {"enable_thinking": true}` — champ de **PREMIER**
  niveau du corps, jamais dans `extra_body`). **Vérifié après correctif** :
  sans kwarg `Paris` en 2 tokens / 0,43 s ; `max_tokens=16` ne vide plus la
  réponse ; `enable_thinking:true` fait bien revenir la réflexion (200 tok) ;
  tool-calling `qwen3_coder` intact (`get_weather {"city":"Paris"}`) ; KV
  **421 582** (12,87×). Dossier :
  `_iteration_log/thinking_default_fix_po2025/NOTES.md`.
- **Doctrine qui RESTE (un défaut serveur ne dispense pas de lire juste).** Une
  sonde (vigie, watchdog) lit `reasoning` **autant que** `content` : un `content`
  vide avec `reasoning_tokens = completion_tokens` et `finish_reason=length`
  reste un signal à interpréter, pas une panne — et il redeviendra courant dès
  qu'un appelant active le thinking.

## Non-négociables

1. **Garde thermale AVANT toute charge GPU soutenue** (incidents passés :
  83 °C en banc, 85 °C à vide en avril). Chaque fenêtre de charge est
  **annoncée sur workspace-vllm avant le geste**, effets de bord compris.
2. **Le hub claudish (même machine) est prioritaire** CPU/RAM — moteur plafonné
  (OMP_NUM_THREADS=4, mem limit, gpu-util ≤0.80), RAM libre hôte ≥ ~4 Go.
3. **Clés dans `myia_vllm/.env` (gitignored) — JAMAIS dans un commit, dashboard
  ou DM** ; distribution par DM privé autodestructif uniquement.
4. Jamais de restart Docker automatisé sans cadre user ; jamais de suppression
  sans preuve de preservation.
5. Un chiffre cité est mesuré, avec sa source ; qualifier VERIFIIE / RAPPORTE /
  SUPPOSE. Commits conventionnels ; PR vers `jsboige/vllm` (`--merge`, jamais
  squash) ; jamais pousser vers `upstream`.
6. Convention flotte : écrire **po-2023/po-2024** (jamais po-203/po-204).

## Documents & chemins du siège

- Profil canonique : `configs/docker/profiles/mini-frognano-4b-po2025.yml`
  (= le modèle **NVFP4** en production ; contrat + mesures + rollback en
  en-tête — le lire avant tout geste). Rollback armé :
  `…-bf16-rollback.yml` (même `container_name`, même volume de cache).
- Essai qui a mené à la promotion : `_iteration_log/nvfp4_trial_po2025/`
  (NOTES.md + rapports de porte + réponses A/B versionnés).
- Faits modèle : config décodée + paysage quants dans la mémoire du siège
  (`~/.claude/projects/d--dev-vllm/memory/`) — ⚠ la ligne « NVFP4 (Blackwell) »
  du paysage des 17 quants est **corrigée** : le NVFP4 se sert en W4A16 Marlin
  dès SM 7.5 (cf. la promotion du 08/10 ci-dessus).
- Logs runtime : `myia_vllm/_logs/` (gitignored — **gitignoré veut dire non
  versionné** : toute preuve qu'on veut garder se recopie dans
  `_iteration_log/<essai>/`).
- Modèle : cache HF Windows `C:\Users\jsboi\.cache\huggingface` (pattern twin
  po-2026) ; image `vllm/vllm-openai:v0.32.0` (bump 19/10, PR #125 ; était
  v0.31.0 — trouvé par le sweep entropie fan-out du 25/10, issue #205).
- Registre questions user : mémoire siège `user-question-registry.md`
  (licence Apache/mit, UAC gouverneur, alias servi).
