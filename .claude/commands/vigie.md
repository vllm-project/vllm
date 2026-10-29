---
description: Cycle de surveillance court du siège FrogNano po-2025 (cron 5h) — santé moteur, thermique, gouverneur, compteurs adoption, [DONE] workspace-vllm.
---

# VIGIE vllm po-2025 (siège FrogNano-4B, issue jsboige/vllm#70)

Cycle court ~3-5 min. Porté par cron session-only `17 */5 * * *` (5 h à :17 — cadence relevée de 4 h à 5 h par le user le 29/10 après le reboot de po-2025) dont le prompt est une seule ligne invoquant cette commande.

## 1. Lectures (paralléliser autant que possible)

- **Dashboard** : `roosync_dashboard(read, type workspace, workspace vllm, section all)`.
- **Inbox DM** : traiter uniquement les DM adressés à `po-2025:vllm`. Les DM machine-level sont visibles par tous les workspaces (bug #3960) → hors-lane = ignorés, laissés non lus.
- **Machine** (PowerShell) : health moteur `:5003` (200 attendu), `docker stats --no-stream` (empreinte RAM/VRAM), GPU temp + util (`nvidia-smi`), vmmem host.
- **Gouverneur thermique** : se prononcer via SES PROPRES artefacts (`_logs/gpu-governor.log` heartbeat, `gpu-governor-state.json`, `governor-install-result.txt`) — jamais via l'énumération schtasks non élevée, qui est aveugle aux tâches SYSTEM. Les heartbeats de `gpu-thermal.log` sont ceux du logger (monitor-only), pas du gouverneur.

## 2. Seuils

- WARN empreinte moteur > 12/16 Gio.
- Critique : temp GPU ≥ 85 °C. WARN : temp ≥ 75 °C ET util < 10 % (signature pompe défaillante). vmmem > 20 Go → WARN.
- Gouverneur muet/hb stale → WARN sur le dashboard, JAMAIS réinstallation (UAC unique de la lane déjà consommé le 06/10).

## 3. Adoption (mandat flotte — voir mémoire project-fleet-adoption-mandate)

- Compteurs moteur (`request_success_total`, finish reasons, erreurs) : delta depuis le cycle précédent.
- Qualifier chaque requête : **interne flotte** (sondes, gates, autres sièges, ai-01 T2/T3) vs **EXTERNE** (hors sièges vllm). Identifier l'origine via `claudish_traffic` le cas échéant.
- Nouveau consommateur externe détecté → ACK sur workspace-vllm (qui consomme, volume, régularité).
- **Un appel de délégation RÉEL par cycle** (mandat user 08/10 : « faire tourner le modèle de délégation en schedule »). Un `mcp__sk-agent__call_agent` sur un preset `mini-*` (`mini-summarizer`, `mini-repo-scan`, `mini-coder-fix`, `mini-web-research` → FrogNano) avec une micro-mission **utile au cycle** — p.ex. résumer les derniers messages du dashboard, ou pré-trier un fichier sous `_logs/`. Consigner le résultat dans la ligne adoption ; un échec se rend en **WARN** (c'est un signal de chaîne cassée, pas un détail). Raison : ça fait bouger `request_success_total` par un **usage réel** au lieu d'une sonde, et ça prouve la chaîne sk-agent × sous-agents à chaque cycle. Coût : quelques centaines de tokens, aucune fenêtre de charge à annoncer.
- **… ET SA VÉRIFICATION, par un étage distinct du producteur** (mandat user 08/10 : « il faut des agents derrière qui vérifient que ce qui est produit est valable »). Une production du mini ne compte **que** vérifiée — le producteur n'est jamais son propre vérificateur. Deux modes, par ordre de force :
  1. **Exécution** — le plus fort, à préférer : choisir une micro-mission dont le résultat est **comptable** (« liste les items `[ASK]` et les questions ouvertes des 20 derniers messages du dashboard »), puis **recompter soi-même dans la source** (lecture du dashboard / `grep`). Le verdict **cite** ce qui a été exécuté, et le **nombre attendu vs obtenu**.
  2. **Étage distinct** — si la production n'est pas comptable (un résumé, un diagnostic) : la faire juger par un **autre** agent (`analyst`, tier médium), jamais par le même modèle. Voir le type `.claude/agents/mini-verifier.md`.
  **Interdit : accepter un « c'est exact » ou un « c'est fait » sans trace d'exécution.** Précédent mesuré (pilote d'audit de notebooks ai-01, 22/09) : appels d'outil sur **1 notebook sur 9**, verdict « exact » affirmé **sans que rien n'ait été exécuté**. Le verdict va dans la ligne adoption **avec sa source** ; un verdict sans source se rend en **WARN**.

## 4. Rendu

- **[DONE]** sur workspace-vllm : santé + thermique + ligne adoption (req interne/externe depuis dernier cycle) + points de watch.
- Qualificateurs systématiques : VERIFIÉ / RAPPORTE / SUPPOSE — jamais propager un fait non vérifié.
- Timestamps : le dashboard est en UTC, mes labels en locale (Paris, +2) — le préciser en cas d'ambiguïté.

## 5. Garde-fous

- JAMAIS de restart Docker sans cadre user.
- Si du travail de code a été fait dans le cycle : commit + push AVANT le rapport [DONE].
- Fin de cycle : `CronList` → si le cron vigie n'existe plus, re-créer `17 */5 * * *` (recurring, session-only) avec pour prompt une ligne invoquant `/vigie`.
