# Qwen3.5/3.6 GGUF + MTP Support — Status

## Architektur der Lösung (Stand: 2026-07-09)

GGUF-Support lebt seit vLLM-Commit `6635279d8` im Out-of-Tree-Plugin
[vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin)
(Kernels, Loader, Quant-Layer, Config-Parser, Auto-Detection). Der frühere
Ansatz dieses Branches (In-Tree-Kopien des alten GGUF-Codes) ist verworfen
— ihm fehlten die CUDA-Kernels und die komplette Loader-Verdrahtung.

Die Arbeit teilt sich jetzt auf zwei Repos auf:

### 1. Plugin-Fork (`/spinning/vllm-gguf-plugin`, editable installiert)

- **`weights_adapter/qwen35.py`** (neu): Qwen3.5/3.6-Adapter
  - model_type `qwen3_5(_text)` / `qwen3_5_moe(_text)` → GGUF-Arch
    `qwen35` / `qwen35moe`
  - Mapping-Lücken von gguf-py gefixt (`linear_attn.dt_bias`,
    suffixlose Params wie `A_log`)
  - **Rück-Transformationen der llama.cpp-Konvertierung** (die Kern-Bugs):
    - Gemma-Style-RMSNorm-Gammas: GGUF speichert `w+1` → `−1` beim Laden
      (alle Norms außer `linear_attn.norm`)
    - `ssm_a` = `−exp(A_log)` → `A_log = log(−ssm_a)`
    - GDN-V-Head-Retiling (grouped→tiled) bei `num_v_heads != num_k_heads`
      wird invertiert: `in_proj_qkv`(V-Zeilen), `in_proj_z`, `in_proj_b/a`,
      `A_log`, `dt_bias`, `conv1d`(V-Kanäle), `out_proj`(Spalten,
      blockaligniert auf Quant-Typ — Q8_0 ok, Q6_K out_proj würde
      hard-erroren)
  - `conv1d` (channels, kernel) → (channels, 1, kernel)
  - `token_embd` wird on-the-fly dequantisiert (vLLMs qwen3_5-Embedding
    hat keinen quant_config)
  - MTP-Draft-Mapping: `blk.<n_layers>.nextn.*` → `mtp.*`,
    `blk.<n_layers>.<attn/ffn>` → `mtp.layers.0.*`
- **`quantization/params.py`**:
  - Tuple-Shard-Support im Weight-Type-Loader (Qwen3.5-GDN lädt
    `in_proj_qkv` als fused Tensor in Shards `(0,1,2)`)
  - Speicherleck-Fix: Container des ausgemusterten Lazy-Parameters werden
    beim Materialisieren geleert (~6.4 GiB/Rank bei 27B TP=2)

### 2. vLLM-Fork (Branch `qwen35-gguf-mtp`, Basis `b5a2adec4`)

Nur Core-Fixes, kein GGUF-spezifischer Code:

- `registry.py`: `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM` registriert
  (Klassen existierten, waren aber nicht registriert)
- `qwen3_5.py`: `Qwen3_5ForCausalLMBase` bekommt `IsHybrid` (+ die drei
  Mamba-State-Klassenmethoden) und `SupportsMRoPE` (Text-only-Positionen)
- `speculative.py`: MTP-Konvertierung auch für `qwen3_5_text`/
  `qwen3_5_moe_text`; Draft-ModelConfig erbt `model_weights` vom Target

## Nutzung

```bash
pip install -e /spinning/vllm-gguf-plugin   # bzw. vllm-gguf-plugin>=0.0.4

python -m vllm.entrypoints.openai.api_server \
  --model /pfad/zu/model-Q6_K.gguf \
  --tensor-parallel-size 2 \
  --mamba-cache-mode align \
  --dtype bfloat16 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Voraussetzung für lokale GGUF-Dateien: `config.json` (Text-Config,
`model_type: qwen3_5_text`, ohne quantization_config) + Tokenizer-Dateien
neben der GGUF-Datei, oder `--tokenizer <original-hf-repo>`.
Remote geht direkt: `--model unsloth/Qwen3.5-2B-GGUF:Q8_0 --tokenizer Qwen/Qwen3.5-2B`.

## Testergebnisse (2026-07-09, 2× RTX 3080 20GB, TP=2)

| Test | Ergebnis |
|---|---|
| Qwen3-0.6B GGUF (Sanity, Standard-Attention) | ✅ kohärent |
| Qwen3.5-2B Q8_0, TP=1 und TP=2 | ✅ kohärent |
| Qwen3.6-27B heretic-v2 Q6_K, TP=2 | ✅ kohärent, Load 11.5 GiB/Rank |
| MTP-Spec-Decode (27B, num_speculative_tokens=3) | ✅ Acceptance 62–80 %, mean accepted len 2.85–3.39 |
| Benchmark 27B+MTP (max-num-batched-tokens 1600) | Decode 41–58 tok/s, Prefill 1120 tok/s (7k-Token-TTFT 6.3 s) |

## Bekannte Grenzen / offene Punkte

- ~~Prefill-Durchsatz~~ gefixt: MMQ-Kernel werden nur noch bis
  `VLLM_GGUF_MMQ_MAX_TOKENS` (Default 16) genutzt, größere Batches gehen
  über Dequant+cuBLAS → Prefill 156 → 1120 tok/s (7.2×) bei
  unverändertem Decode.
- `out_proj` muss blockaligniert quantisiert sein (Q8_0/Q4_0/...; Q6_K mit
  head_v_dim=128 nicht invertierbar → klarer Fehler mit Hinweis).
- mmproj/Vision (multimodal) nicht unterstützt — Text-only.
- MTP + CUDA-Graph-Profiling braucht Headroom: mit
  `--max-num-batched-tokens 1600` und `--gpu-memory-utilization 0.86`
  stabil auf 20-GB-Karten; Default-Werte OOMen knapp.
- Plugin-Logger (`vllm_gguf_plugin.*`) sind in vLLMs Logging-Config
  unsichtbar (außerhalb des `vllm.*`-Namespace).
- MoE-Variante (`qwen35moe`) gemappt, aber ungetestet.
