# Lane po-2023:vllm — instructions du siège embeddings

> Siège inauguré le 2026-10-20 (mandat user) après la panne matérielle de po-2026
> (HS 09/10). Prédécesseur : [`po2026-embeddings.md`](po2026-embeddings.md) — ses
> doctrines vigie et son runbook restent la référence technique.

## Identité

- **Ce siège** (`myia_vllm/` du clone `D:\Dev\vllm`) = poste de travail de la lane
  **embeddings de la flotte** sur le fork `jsboige/vllm`. Siblings vllm :
  `myia-ai-01` (prod Swift-1.5-27B), `myia-po-2025` (FrogNano + hub).
- **Runtime de service** : depuis le cutover IIS du 2026-10-14 00:50Z, la chaîne
  entière tourne sur po-2023 — IIS ARR local → sidecar `:8004`
  (`embedding-proxy-logger`) → vLLM `:8005` (`qwen3_4b_awq_embedding_server`).
- **Machine bi-GPU** : GPU 0 = RTX 3080 Ti interne 16 Go (embedder, pinné via
  `device_ids: ['0']` + `CUDA_DEVICE_ORDER=PCI_BUS_ID`, PR #118) ; GPU 1 = RTX 3090
  externe 24 Go, **occupé par une autre lane — ne jamais y toucher**. Au déploiement :
  vérifier que nvidia-smi index 0 = le 3080 Ti libre.

## Surveillance (mission première)

- **Cron 5 h à :17** (mandat user 2026-10-20, continuité du mandat po-2026 d'octobre),
  session-only : CronList d'abord, ré-armement idempotent, jamais de doublon.
- Spec : skill `/surveillance-embeddings` (écrit pour po-2026 — traduire les chemins,
  cf. « Chemins du siège » ci-dessous ; les pièges et le rôle vigie restent intégralement valables).
- **Rapports → dashboard workspace `vllm`** (tag `DONE`). Le workspace `Embeddings`
  portait le fil po-2026 ; bascule à l'inauguration po-2023.
- Clé de test : `D:\Production\Embeddings\compose\.env` (`VLLM_API_KEY`) — jamais
  affichée ni postée. Contrat stable vérifié chaque cycle : POST réel public →
  **dims 2560**.

## Non-négociables (héritées po-2026)

1. `https://embeddings.myia.io` + modèle `qwen3-4b-awq-embedding` + dims 2560 stables —
   tout changement = réindexation flotte (6,2M+ points).
2. Clés uniquement dans les `.env` gitignored — jamais dans un commit, dashboard ou DM.
3. Jamais de suppression sans preuve de préservation ; jamais de restart Docker
   automatisé sans cadre user (co-locataires sur po-2023 : claudish-proxy, comfyui,
   stacks TTS/whisper, searxng…).
4. Un chiffre cité est mesuré, avec sa source ; qualifier VERIFIÉ / RAPPORTE / SUPPOSÉ.
5. Commits conventionnels ; PR vers `jsboige/vllm` ; ne jamais pousser vers `upstream`.
6. Convention flotte : écrire **po-2023/po-2024** (jamais po-203/po-204).

## Chemins du siège (po-2023)

- `.env` prod : `D:\Production\Embeddings\compose\.env` (`VLLM_API_KEY`,
  `HUGGING_FACE_HUB_TOKEN`). **Pas de `QDRANT_API_KEY`** (gap de migration — baromètres
  points non mesurables, registre Q-P2023-2).
- Compose actif : profil repo `configs/docker/profiles/embeddings-qwen3-4b-awq-po2026.yml`
  (nom historique conservé), lancé avec `--env-file` vers le `.env` prod ci-dessus.
- Sidecar : script `D:\Production\Embeddings\scripts\embedding-proxy-logger.py`,
  logs `D:\Production\Embeddings\logs\capture-20261013\` (jsonl vidé à chaque restart
  du sidecar — copier avant restart si préservation nécessaire).
- Gouverneur thermique : **ABSENT de ce siège** (ni schtask ni undervolt migrés ;
  runbook po-2026 = procédure prête). Vigie : alerte >85 °C (registre Q-P2023-1).
- Qdrant prod : `https://qdrant.myia.io` + LAN `http://192.168.0.47:6333` (distant .47).

## Gaps de migration (consignés au registre du siège)

1. Gouverneur thermique non migré (Q-P2023-1) — surveillance manuelle à chaque cycle.
2. `QDRANT_API_KEY` absente (Q-P2023-2) — santé healthz vérifiable, points non.
3. HF token vide dans le conteneur courant (Q-P2023-3, bénin — cache local) — à la
   prochaine re-création : `--env-file D:\Production\Embeddings\compose\.env`.
