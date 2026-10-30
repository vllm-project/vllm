# Essai comparatif Nanbeige4.2-3B (BF16 + AWQ LostGentoo) vs FrogNano-4B — siège po-2025

**Date** : 2026-10-30 soir · **Siège** : po-2025 (RTX 3080 Ti Laptop 16 Go, SM 8.6, 64 Go)
**Mandat** : user 30/10 — « Go pour un test comparatif » + « vérifier si les perfs [de l'AWQ LostGentoo] sont conservées par rapport au modèle principal » + SenseNova-Loop-9B à instruire (verdict : §6)
**Issue** : jsboige/vllm#70 (suivi du siège) · registre Q9

---

## 1. Annonce et déroulé

Fenêtre annoncée sur workspace-vllm AVANT le geste (patron de l'essai NVFP4 du 08/10) :
référence FrogNano capturée sur le moteur vivant → down → serve BF16 (référence du
modèle principal) → serve AWQ (candidat quant) → portes + A/B identiques → restauration.
Effet de bord annoncé : `:5003` (`frognano-4b` et l'alias `mini`) indisponible pendant la
fenêtre — l'alias volontairement NON servi sur l'essai.

## 2. Référence FrogNano (same-evening, moteur v0.32.0 NVFP4+FP8KV, 30/10 18:42 locale)

Capturée juste avant le down (`frognano_gates_1632_po2025.py` + `ab5_prompts_po2025.py`,
JSON dans `myia_vllm/_logs/`, recopiés en §Artefacts) :

| Mesure | FrogNano-4B (30/10) | (rappel 08/10, v0.31.0) |
|---|---|---|
| Warm-up 1 flux | **92,7 t/s** | 60,7 |
| N=16 agrégat | **1 312,1 t/s** (médiane 83,0) | 915,2 |
| N=32 agrégat | **1 827,7 t/s** (médiane 57,9) | 1 416,5 |
| Erreurs HTTP | 0 | 0 |
| Temp max | 54 °C | 69 °C |
| VRAM | 12,7/16,4 Gio | 11,8 |
| A/B 5 prompts | 1 899 tok / 16,38 s, 5/5 `finish=stop` | — |

Le moteur v0.32.0 est nettement au-dessus des chiffres v0.31.0 du 08/10 (même script,
même machine) — **toute comparaison se fait contre la colonne 30/10**.

## 3. Premier boot BF16 : CRASH-LOOP sur stock v0.32.0 — le support #56071 est inopérant

Téléchargements vérifiés au byte (`du -sb` dans le conteneur, pas de stub 9p) :
BF16 `Nanbeige/Nanbeige4.2-3B` snapshot `b82e54bd` = **8 362 159 704 o** ;
AWQ `LostGentoo/Nanbeige4.2-3B-AWQ-W4A16-ASYM` snapshot `4e39f802` = **3 699 277 053 o**.

Profil `eval-nanbeige42-bf16-po2025.yml` sur image stock → **crash-loop, 6 restarts,
`/health` jamais 200**. Cause racine (docker logs) :

```
AssertionError: layer name model.layers.0.self_attn.loop_attns.0.attn
should only contain one integer
  vllm/model_executor/models/utils.py:1029 (extract_layer_index)
```

**Lecture** : la PR upstream #56071 (merged 14/09, présente dans v0.32.0) ne fait qu'ajouter
`"NanbeigeForCausalLM": ("transformers", "TransformersForCausalLM")` au registre. Le
backend transformers parse ensuite les noms de poids pour dériver l'index de couche — et
les noms **loopés** du Looped Transformer portent DEUX entiers (`layers.0` +
`loop_attns.0`) → assert. La fonction `extract_layer_index` sait gérer 2 entiers
(paramètre `num_attn_module`), mais l'appelant du backend transformers ne l'utilise pas.
**La fiche HF « fork requis » avait raison** ; notre vérification pré-fenêtre (ligne de
registre présente) était nécessaire mais pas suffisante — mesuré.

Incident de conduite (sans effet) : 5 s seulement entre le down FrogNano et l'up eval
(règle siège ≥ 15 s) — pas de figerage cette fois, mais la règle reste.

## 4. Overlay `vllm-openai-v0320-nanbeige:v1` — support natif greffé sur stock v0.32.0

Le fork officiel `Nanbeige/vllm` (branche `nanbeige42`, commit `62f6de733d` « Add support
for Nanbeige4.2 », dernier push fork 14/09) porte une implémentation **NATIVE** :
`vllm/model_executor/models/nanbeige.py` (573 lignes) + config + parsers
reasoning/tool « nanbeige » (adaptateurs Qwen3-engine). **9 fichiers, +796 lignes** au
total — exactement le patron de l'overlay tierpin d'ai-01 (stock v0.32.0 épinglé par
digest + fichiers runtime greffés).

Greffe (`.tmp/nanbeige_overlay/` → `configs/docker/overlay_v0320_nanbeige/`) :
- `models/nanbeige.py` + `transformers_utils/configs/nanbeige.py` : **copiés tels quels**
  — imports 100 % compatibles v0.32.0, **vérifié en conteneur avant build** (montage des
  fichiers sur l'image stock + import test) ;
- `models/registry.py` : la ligne #56071 `("transformers", "TransformersForCausalLM")`
  **remplacée** par `("nanbeige", "NanbeigeForCausalLM")` ;
- 5 fichiers registre/`__init__` : hunks Nanbeige uniquement (+1 à +8 lignes chacun) —
  le fork date du 27/08, ses versions entières ne sont PAS prenables (dérive upstream
  8 semaines : mimo a changé de parser, registry a bougé, etc.).

Validation dans l'image (RUN du Dockerfile) : registre natif ✓, import module ✓,
`lazy_parsers` reasoning ✓ et tool ✓, convertor `num_hidden_layers × num_loops` ✓.

Les deux profils d'éval portent désormais `image: vllm-openai-v0320-nanbeige:v1` +
`--reasoning-parser nanbeige --tool-call-parser nanbeige --enable-auto-tool-choice`
(parsers natifs du fork, adaptateurs Qwen3-engine).

## 5. Deuxième boot (overlay + fp8 KV) : le modèle CHARGE — coût KV du Loop mesuré

Premier boot overlay (KV auto/bf16) : le modèle natif **charge complètement** (poids,
torch.compile, LBNHC layout, init moteur) puis bute sur la taille du cache :

```
ValueError: To serve at least one request with the model's max seq len (32768),
5.5 GiB KV cache is needed, which is larger than the available KV cache memory
(3.67 GiB). Estimated maximum model length is 21888.
```

**Le coût KV du Looped Transformer, mesuré** : 22 couches × `num_loops=2` = **44 couches
effectives** → 44 × 8 KV heads × 128 × 2 (K+V) × 2 o (bf16) = **176 Ko/tok**, soit
**~11× FrogNano servi** (16 Ko/tok en fp8) et 2× l'estimation papier (90 Ko/tok — qui
comptait 22 couches, pas 44). En bf16-KV le siège ne sert que ~21 888 tok de contexte.

Décision : `--kv-cache-dtype fp8` sur les DEUX configs d'éval (88 Ko/tok, 32 K tient) —
même posture que la prod FrogNano (KV quantifié), isole la variable poids BF16↔AWQ.
Second effet mesuré attendu : capacité KV BF16 ≈ 3,67 Gio ≈ 42 K tok (1,3 conversation
pleine longueur) — un coût structurel du modèle, pas de la fenêtre.

## 6. Batterie

### 6.1 Nanbeige4.2-3B BF16 (overlay natif, fp8 KV, 30/10 ~19:30 locale)

Boot sain **2 min 17 s** (18:27:50→18:30:07Z, 0 restart, graphes CUDA FULL capturés).
Poids **7,87 GiB** / 62,9 s. **KV GPU 38 752 tok** — 1,18× la pleine longueur 32 K
(contre 11,96× FrogNano). Reasoning parser `nanbeige` actif au banner.

| Mesure | FrogNano-4B (réf 30/10) | **Nanbeige BF16** | ratio |
|---|---|---|---|
| N=16 agrégat | 1 312,1 t/s (méd. 83,0) | **427,0 t/s** (méd. 26,8) | **0,33×** |
| N=32 agrégat | 1 827,7 t/s (méd. 57,9) | **741,7 t/s** (méd. 23,2) | **0,41×** |
| A/B 5 prompts | 1 899 tok / 16,4 s | **2 804 tok / 97,1 s** | **~0,17× en débit** |
| Erreurs HTTP | 0 | **0** | = |
| Temp max | 54 °C | **65 °C** | +11 °C |
| VRAM | 12,7 Gio | **12,8 Gio** | = |
| KV tokens | 391 748 | **38 752** | 0,10× |

A/B : 5/5 réponses, mais **beaucoup plus verbeux** (SQL coupe au plafond
`finish=length` à 1 024 tok ; FrogNano terminait à 1 003) et ~4× plus lent par prompt.
Tool-calling natif **OK** : `finish=tool_calls`, `{"city":"Paris"}`, `content` propre,
0 fuite de raisonnement (parser `nanbeige` = adaptateur Qwen3-engine).

Lecture d'étape : sur le **débit**, le Looped Transformer 44-couches-effectives est
structurellement ~3× plus lent que FrogNano (GDN hybride) sur cette carte — ce n'est
pas la faute du quant, la référence BF16 le montre déjà.

### 6.2 Nanbeige4.2-3B AWQ W4A16-ASYM (LostGentoo) — crash v1, fix overlay v2

Premier boot AWQ (overlay `:v1`, KV fp8) : **crash-loop, ~20 restarts**, `/health`
jamais 200. Cause racine (docker logs) :

```
AttributeError: 'CompressedTensorsConfig' object has no attribute 'get_cache_scale'
  nanbeige.py:452 (load_weights)
```

**Lecture** : le bloc fork-era `self.quant_config.get_cache_scale(name)` (chargement
des échelles KV du checkpoint) appelait une API disparue en v0.32.0 — remplacée par
`get_cache_scale_mapper()` dans `AutoWeightsLoader` (`base_config.py:217`,
`models/utils.py:467` ; stock `llama.py` n'a plus AUCUN bloc manuel équivalent).
La jambe BF16 n'avait jamais exercé ce chemin : sans checkpoint quantifié,
`quant_config` est `None` et le bloc était sauté — c'est le quant qui l'a révélé.
Le risque « loader pack-quantized » déclaré en tête de profil ne s'est PAS réalisé :
c'est notre greffe qui était incomplète, pas le loader.

**Fix (overlay `:v2`)** : bloc supprimé de `nanbeige.py` (greffe adaptée à l'API
v0.32.0, comme pour registry/parsers) — inoffensif pour ce checkpoint qui porte
`"kv_cache_scheme": null` (aucune échelle KV à charger) ; les noms d'échelle
résiduels tombent dans `maybe_remap_kv_scale_name` plus bas.

### 6.3 Batterie AWQ (overlay `:v2`, fp8 KV, 30/10 ~19:55 locale)

Boot sain **2 min 20 s** (0 restart). Poids **3,5 GiB** / 41,5 s (vs 7,87 BF16 = 2,25×
plus léger). **MarlinLinearKernel actif** pour CompressedTensorsWNA16 — le vrai chemin
W4A16 optimisé. **KV GPU 88 912 tok = 2,71× la fenêtre 32 K** (vs 38 752 = 1,18× en
BF16 : les 4,3 GiB libérés par le quant vont au KV, ×2,29).

| Mesure | FrogNano-4B (réf) | Nanbeige BF16 | **Nanbeige AWQ** | AWQ/réf | AWQ/BF16 |
|---|---|---|---|---|---|
| Warm-up 1 flux | 92,7 t/s | — | **75,1 t/s** | 0,81× | — |
| N=16 agrégat | 1 312,1 (méd. 83,0) | 427,0 (méd. 26,8) | **1 165,2 (méd. 73,3)** | **0,89×** | **2,73×** |
| N=32 agrégat | 1 827,7 (méd. 57,9) | 741,7 (méd. 23,2) | **1 665,9 (méd. 52,5)** | **0,91×** | **2,25×** |
| A/B 5 prompts | 1 899 tok / 16,4 s | 2 804 tok / 97,1 s | **3 022 tok / 35,1 s** | 0,74×/flux | 2,77× |
| Erreurs HTTP | 0 | 0 | **0** | = | = |
| Temp max | 54 °C | 65 °C | **57 °C** | +3 °C | −8 °C |
| VRAM | 12,7 Gio | 12,8 Gio | **12,8 Gio** | = | = |
| KV tokens | 391 748 (11,96×) | 38 752 (1,18×) | **88 912 (2,71×)** | 0,23× | 2,29× |
| Tool-call natif | OK | OK | **OK** `{"city":"Paris"}` | = | = |

A/B : 4/5 `finish=stop` (SQL coupe à 1 024 comme BF16 — verbeux, 3 022 tok vs 1 899
FrogNano). Qualité à l'inspection (pas un bench formel) : réponses françaises
cohérentes, code avec docstring, même structure que le BF16 — **rien d'observé de
dégradé par le quant**.

## 8. Verdict (réponses aux trois questions du user)

**« Les perfs sont-elles conservées par rapport au modèle principal ? » — OUI, et
largement améliorées.** Le W4A16-ASYM Marlin est **~2,7× plus rapide que le BF16**
sur cette carte (N=16 : 1 165 vs 427 t/s ; A/B : 86 vs 29 t/s/flux). Le BF16
non-quantifié était le mauvais rendement (bande passante), pas une référence de
sommet : le quant le corrige au lieu de le coûter. Rien de dégradé observé côté
qualité (inspection + tool-call exact + 0 fuite de raisonnement).

**vs FrogNano (la question du siège)** : **quasi-parité de débit** (0,89× à N=16,
0,91× à N=32, 0,81× single-stream) — mais **0,23× la capacité KV** (88 912 vs
391 748 tok : le Looped Transformer paie 88 Ko/tok en fp8, 5,5× FrogNano) et le
contexte 262 K natif du modèle est **inservable** à cette échelle sur 16 Go. Coût
d'hébergement : l'**overlay maison obligatoire** (le support upstream #56071 crash
sur les noms loopés ; notre overlay = fork officiel Aug greffé sur v0.32.0, 2
adaptations d'API : registry/parsers + le bloc `get_cache_scale` supprimé).

**Décision** : FrogNano reste en prod (la capacité conversationnelle et le zéro-overlay
dominent pour un moteur de délégation multi-consommateurs). Nanbeige4.2-3B-AWQ devient
le **candidat mini documenté** : bancs qualité devant FrogNano (SWE-Verified 63,6 vs
61,5, TB2.0 44,1 vs 31,1 — harnais externes, non mesurés chez nous), débit quasi égal,
licence Apache-2.0 propre. La vraie cible reste **Nanbeige4.3-3B** (distillé du 33B,
annoncé) : le harnais complet — overlay, profils, portes, A/B — est **réutilisable tel
quel** pour sa publication, seule la ligne du modèle change.

## 9. Restauration

Down eval AWQ → **attente 20 s** (règle siège ≥ 15 s, respectée) → up
`mini-frognano-4b-po2025.yml` → health 200 + **preuve par génération réelle**
(`OK-FROGNANO`). Détails dans le rapport de fenêtre sur workspace-vllm.

## 7. SenseNova-Loop-9B-Preview (GGUF) — verdict demandé par le user

« il est supporté je crois » → **non, pas sur notre pile vLLM** (mesuré 30/10) :
- arch `sensenova_loop` : **0 occurrence** dans notre arbre vLLM v0.32.0
  (backend natif ET registre transformers) — pas servable en safetensors non plus ;
- le dépôt GGUF (`notfakerv/…`) exige le **runtime llama.cpp du fork Clover** (PR #31306)
  — vLLM ne sert du GGUF que pour des archs nativement implémentées ;
- Q8_0 = 14,03 Gio sur une carte de 16 Go : à la limite même si l'arch existait ;
- texte seul, validé par l'auteur à 8K de contexte.
La base `sensenova/SenseNova-Loop-9B-Preview` (safetensors, déployable vLLM/SGLang selon
sa carte — nos 0 occurrences portent sur l'implémentation **dans notre image**) reste un
objet de veille : l'arch two-stream mHC/looped est la même famille que Nanbeige. À
revoir si vLLM ajoute `sensenova_loop` en natif.

## Artefacts

- Profils : `configs/docker/profiles/eval-nanbeige42-{bf16,awq}-po2025.yml` (image `:v2`).
- Overlay : `configs/docker/overlay_v0320_nanbeige/` (Dockerfile + 8 fichiers) —
  `v1` = greffe initiale (BF16), `v2` = + suppression du bloc `get_cache_scale` (§6.2).
- JSONs versionnés : réf FrogNano `frognano_gates_1632_20261030_184219.json` +
  `ab_frognano_ref_frognano-4b_20261030.json` ; BF16
  `frognano_gates_1632_20261030_193111.json` + `ab_nanbeige_bf16_nanbeige42-3b_20261030.json` ;
  AWQ `frognano_gates_1632_20261030_195746.json` + `ab_nanbeige_awq_nanbeige42-3b_20261030.json`
  (les originaux gitignorés vivent dans `myia_vllm/_logs/`).
- Harnais réutilisable pour Nanbeige4.3-3B : `frognano_gates_1632_po2025.py`
  (`BENCH_MODEL` env) + `ab5_prompts_po2025.py`.
