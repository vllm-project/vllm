# Spark-X2.6-4B-INT4 — fenêtre de test po-2025 (2026-10-21 soir, GO user)

**Verdict : NON SERVABLE sur stock vLLM v0.32.0 — les deux chemins refusés à la
validation `ModelConfig`, avant tout chargement de remote-code. FrogNano
restauré (retour chaud), fenêtre close.**

## Contexte et déclencheur

- GO user (21/10 ~23:40 locale) : « Si le traffic est bas c'est peut-être le
  moment de tester https://huggingface.co/XHToken/Spark-X2.6-4B-INT4 » — sur le
  créneau creux mesuré (2 POST/h vs ~340 sur 18:25→20:23Z).
- Spark-X2.6-4B était **candidat de veille** depuis l'évaluation du 21/10
  (user : « Ok pour tes recos »). Déclencheur de re-test déclaré à l'époque :
  « arch `spark2_5` dans le core vLLM sans remote-code **OU** boot test
  v0.32.0 avec l'INT4 officiel » — cette fenêtre exécutait la seconde branche.

## Préparation (vérifiée)

- Modèle : `XHToken/Spark-X2.6-4B-INT4` (Apache 2.0, ~14 h au dépôt),
  téléchargé 3,4 Go réels (`model.safetensors` 3 411 522 856 B — pas de stub),
  snapshot `4cd05c9e3bd23b08331fcc70fea6dbd5139ef29f`. Config :
  `Spark2_5ForCausalLM`, `model_type spark2_5`, `max_position_embeddings`
  1 048 576, chat template jinja **et** embarqué dans tokenizer_config.
  Remote-code embarqué (`configuration_spark.py`, `modeling_spark.py` 19 Ko).
- Profil de test : `mini-spark26-4b-test-po2025.yml` — slot :5003 en
  **déplacement** (pas de co-host : FrogNano ~14,2/16 Go à gpu-util 0.78),
  gpu-util 0.78, max-model-len 32768, prefix caching, `enable_thinking:false`
  par défaut (même kwarg que FrogNano), compile-cache dédié, `restart: no` +
  `autoheal=False` (pas de crash-loop nocturne), noms servis `spark-2.6-4b`
  seuls (honnêtes, pas d'alias silencieux pendant la fenêtre).
- **Batterie comparée prête avant le swap** : `mini_reliability_battery.py`
  (7 axes à ground-truth, température 0, thinking off, contenu inline, jamais
  le total dans l'énoncé). **Référence FrogNano mesurée sur le moteur vivant
  avant le geste : 6 PASS / 0 PARTIAL / 1 FAIL** (`battery_frognano_baseline.json`)
  — FAIL sur A3 moyenne dérivée (faiblesse connue) ; A1 comptage@60 PASS
  ce soir (3 tok) là où l'historique (autre mise en forme, instance plus
  dure) donnait 27/60 — la batterie de ce soir est l'A/B interne
  même-difficulté, pas une réplication des instances historiques.
- Fenêtre annoncée sur workspace-vllm AVANT le geste (messageId
  `fenetre-po2025-spark26-test-20261021T2240Z`), 404 assumés sur
  `frognano-4b`/`mini` pendant la fenêtre.

## Tentative 1 — chemin natif (`--trust-remote-code` seul)

Conteneur mort en ~50 s, EXIT=1, pas d'OOM. Capture intégrale :
`boot_full.log`. Erreur décisive (vLLM 0.32.0, `arg_utils.py:1955` →
`ModelConfig`) :

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
  Value error, Model architectures ['Spark2_5ForCausalLM'] are not supported
  for now. Supported architectures: dict_keys([… ~400 entrées, pas de
  Spark* …])
```

Le refus précède **tout** chargement de remote-code : c'est une gate de
registre, pas un échec d'exécution du code embarqué.

## Tentative 2 — backend générique (`--model-impl transformers`)

Une seule relance (coût ~2 min d'indispo, dans la fenêtre annoncée). Même
gate, second message :

```
  Value error, The Transformers implementation of 'Spark2_5ForCausalLM' is
  not compatible with vLLM.
```

Le backend transformers de v0.32.0 a sa **propre liste de compatibilité** —
Spark2_5 n'y figure pas non plus. Deux chemins, deux refus : verdict clos
pour cette version d'image.

## Restauration

`down` spark (22:2xZ) → `up` FrogNano (profil canonique, cache + compile
volume chauds) → santé revenue (voir la ligne FROGNANO_HEALTH du harnais).
Compteurs :5003 reset par le double swap — les prochains cumuls repartent de
zéro (ne pas rapporter de delta à cheval sur la fenêtre).

## Ce que la fenêtre produit malgré l'échec

1. **Le déclencheur de veille est RÉSOLU par la mesure** : « boot test
   v0.32.0 » = REFUS sur les deux chemins. Le candidat ne redevient
   testable que si (a) `Spark2_5ForCausalLM` entre au registre vLLM (core ou
   liste transformers) dans une version > v0.32.0, ou (b) un plugin officiel
   NVIDIA (la carte mentionne un Spark-plugin, chemin Ascend documenté, pas
   CUDA) — à revoir au bump moteur suivant.
   La carte sert sur `vllm/vllm-openai:latest` : l'arch est donc attendue
   dans une release post-v0.32.0.
2. **Le harnais `mini_reliability_battery.py` reste l'organe d'A/B** de la
   piste « modèle un poil plus gros » (mandat user 21/10) — candidat tête
   Qwen3.5-9B (NVFP4 ~6 Go, dossier au 21/10). Toute fenêtre future :
   baseline sur le moteur vivant PUIS swap, exactement ce protocole.
3. **Règle d'interprétation carte/recette** : une recette vLLM sur une carte
   modèle ne garantit pas la compatibilité avec NOTRE image épinglée —
   « latest » ≠ v0.32.0. Toujours grep le registre de l'image avant de
   promettre une fenêtre (ici la gate était lisible avant le down :
   `docker run --rm --entrypoint bash vllm/vllm-openai:v0.32.0 -c "grep -i
   spark …/registry.py"` — fait après coup, absence confirmée ; la méthode
   « lire la garde dans l'image » se généralise, cf. NVFP4 capability 75).

## Mandats consignés ce soir (user, avant coucher)

1. **Adoption éventuelle d'un remplaçant = alias doubles** : le nouveau nom
   EN PREMIER + `frognano-4b` et `mini` en alias pour la continuité de
   service (à câbler dans le profil d'adoption, pas dans un profil de test).
2. **Promotion re-mandatée sans limite de 6 cycles** : relancer
   l'utilisation sur les dashboards « jusqu'à un rythme de consommation de
   la capacité significatif » — la barre de sortie précédente (6 cycles) est
   relevée ; la boucle horaire continue avec la barre nouvelle (part
   significative de la capacité : à chiffrer p.ex. en POST/h soutenu vs
   N=32 1 642 t/s et KV 396 639 tok).

## Artefacts

- `boot_full.log` — capture intégrale tentative 1 (logs saisis avant down :
  les logs meurent avec le conteneur)
- `battery_frognano_baseline.json` — référence FrogNano 7 axes, 22:15Z
- `mini_reliability_battery.py` (`myia_vllm/scripts/testing/`) — harnais A/B
- `mini-spark26-4b-test-po2025.yml` (`myia_vllm/configs/docker/profiles/`) —
  profil de test conservé (re-test au bump moteur : deux lignes à changer,
  image + éventuellement retrait de `--model-impl transformers`)
