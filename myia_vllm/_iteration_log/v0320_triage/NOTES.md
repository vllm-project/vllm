# Veille MAJ — v0.32.0 (tour du 19/10, demande user « notamment le KVCache »)

**Position** : prod sur **v0.31.0** (publiée 05/10, promue 05/10). **v0.32.0 publiée
le 18/10 06:50Z.** Ce document trie la release notes pour NOTRE stack
(Swift-1.5-Qwen3.8-27B W4A16 pack-quantized, hybride GDN, FP8 KV 424 610, offload
RAM 24 GiB, TP=2, SM89 Ada, WSL2, batch 8192/seqs 48, préfix caching).

---

## 1. KVCache — le tour demandé

| Sujet | Contenu v0.32.0 | Pertinence nous |
|---|---|---|
| **NVFP4 KV cache sur SM8x** (#46963) | « NVFP4 KV cache now runs on SM8x and SM12x » + SM90 (#60620), XQA decode SM12x (#60452) | **LA piste nouvelle.** SM8x = SM80-89 → **Ada incluse**. 4-bit → densité ~2× fp8 (notre fp8 = 424 610 tok → ~800K+ potentiel si équivalent TQ). Aucune promesse de vitesse : TQ k8v4 avait perdu 33 % en N=1 (rejeté 09/10) ; NVFP4 a ses propres kernels, **à mesurer en fenêtre dédiée**. |
| `--kv-cache-dtype-skip-layers` honoré (#60023) | KV par précision par couche (Qwen) | Pas d'usage identifié aujourd'hui ; à garder en tête (mix fp8/haute précision par couche). |
| **M-RoPE cache 4× plus petit** (#59853) | Qwen3-VL/3.5/3.8 M-RoPE + YaRN matching Transformers | **Nous concerne** (vision Swift) — moins de VRAM cache multimodal. Gain léger, jamais bloquant chez nous. |
| `int4_per_token_head` KV (#56198) | Heads non-puissance-de-2 | Non-SM89 a priori, hors périmètre. |
| GLM-5.x FP8 KV défaut SM100 (#60140/#60281), `fp8_ds_mla` SM90 (#59246), `ultraquant_4bit` ROCm (#57057) | — | Hors SM89. |
| KV connectors (NIXL #54483/#58860, Mooncake #57952, KVCR #59899) | Copies coalescées, block sizes mixtes, recovery | **Pas notre usage** (pas de prefill/decode disaggregation). |
| Faster KV block free queue (#60269) | — | Micro-perf scheduler, bénin. |

**Verdict KVCache** : rien qui change notre config actuelle (fp8 + offload reste le
bon choix mesuré) ; **une seule piste à évaluer : NVFP4 KV sur Ada** — en fenêtre
dédiée, frais-vs-frais, avec le protocole TQ (N=1/16/32 + KV + prefill).

## 2. Ce qui nous touche hors KV (arguments de bump)

- **WSL2 : non-blocking H2D copies (#60215)** — fix plateforme directe (MRV2).
- **Hybride + prefix caching** : stale block hashes sur sessions tronquées (#59103),
  prefix-cache bounds (#55601/#60533), **quantized Mamba-2 avec TP (#58083)** — notre
  stack exactement (hybride quantifié TP=2 + prefix caching).
- **`max_tokens` ne réserve plus la fenêtre (#57035)** — « long prompts are no longer
  rejected » : **la classe exacte de notre 400 de #431** (v0.31 répond 400 sans
  rogner, mesuré 09/10). Comportement à re-mesurer au bump — pourrait rendre le
  400 de fenêtre-pleine obsolète côté moteur.
- **Startup/mémoire** : shm tensor arena par défaut (#51207), **daemon-held weights
  comptés dans gpu-util (#57298)** (calcul VRAM plus juste), cached memory reclaimed
  au chargement (#57516), `vllm download-kernels` (#58765) + `--preload-autotune`
  (#60085) → attaque du boot ~8 min.
- **Sécurité** : GHSA chat template DoS (#50300 — **on sert des templates**),
  tokenizer `max_token_id` (#59491), P2P ZMQ (#51504) ; **CORS plus « * » par défaut
  (#61009, breaking)**.
- **Speculative decoding** : correctness fixes ngram/Mamba/MTP/prefix caching
  (#56531, #59759, #60797) — watch MTP (cf. §3).

## 3. Breaking changes à traiter si bump

- `mamba_cache_mode="all"` **retiré** (#58997) — nous sommes sur `align` (défaut),
  OK.
- `transformers>=5.16.1,<5.20.0`, `tokenizers>=0.23.2` (#57387/#59796).
- CORS restreint (#61009) — vérifier les consommateurs cross-origin.
- Cascade attention retiré (#61126) ; Python 3.10 drop (#60402) ; torch.compile SP et
  AsyncTP retirés (#60517) — non utilisés chez nous.

## 4. Watch items — statuts vérifiés le 19/10 (REST)

| Item | Statut 19/10 | Suite |
|---|---|---|
| **MTP Swift #58807** (notre issue) | **open** ; fix pris par he-yufeng le 26/09 (détection via index safetensors) ; pas de PR mergée ; dernier update = notre datapoint 29/09 | Re-tester au merge — **MTP toujours bloqué** |
| DFlash2 quantized qkv #51684 | **open**, gelé depuis 25/09 | Re-test au merge |
| TQ × MTP #53406 / #53410 | **open**, gelés depuis 22/08 | Re-test au merge |

## 5. Verdict

**v0.32.0 est une candidate de bump crédible** — profil de gains identique à celui
qui avait justifié v0.31 (fixes WSL2 + hybride + startup + sécurité), plus le
changement `max_tokens` qui touche directement notre cas #431. Procédure standard :
fenêtre dédiée, 13 gates, A/B **frais-vs-frais même-heure** (leçon de confondance du
08/10), rollback 3 lignes. **Décision user.**

Après bump (si GO) : éval **NVFP4-KV sur SM89** comme successeur possible de fp8 —
protocole TQ (N=1/16/32/48 + KV + prefill + qualité).