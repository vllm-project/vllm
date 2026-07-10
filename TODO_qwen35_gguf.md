# Qwen3.5/3.6 GGUF + MTP Support — Status

## Architecture of the solution (as of 2026-07-09)

Since vLLM commit `6635279d8`, GGUF support lives in the out-of-tree plugin
[vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin)
(kernels, loader, quant layer, config parser, auto-detection). The earlier
approach of this branch (in-tree copies of the old GGUF code) has been
abandoned — it lacked the CUDA kernels and the entire loader wiring.

The work is now split across two repos:

### 1. Plugin fork (`/spinning/vllm-gguf-plugin`, installed editable)

- **`weights_adapter/qwen35.py`** (new): Qwen3.5/3.6 adapter
  - model_type `qwen3_5(_text)` / `qwen3_5_moe(_text)` → GGUF arch
    `qwen35` / `qwen35moe`
  - fixes mapping gaps in gguf-py (`linear_attn.dt_bias`,
    suffix-less params such as `A_log`)
  - **Inversion of the llama.cpp conversion transforms** (the core bugs):
    - Gemma-style RMSNorm gammas: GGUF stores `w+1` → subtract `1` on load
      (all norms except `linear_attn.norm`)
    - `ssm_a` = `−exp(A_log)` → `A_log = log(−ssm_a)`
    - GDN v-head retiling (grouped→tiled) when `num_v_heads != num_k_heads`
      is inverted: `in_proj_qkv` (v rows), `in_proj_z`, `in_proj_b/a`,
      `A_log`, `dt_bias`, `conv1d` (v channels), `out_proj` (columns,
      block-aligned to the quant type — Q8_0 fine; a Q6_K out_proj would
      hard-error)
  - `conv1d` (channels, kernel) → (channels, 1, kernel)
  - `token_embd` is dequantized on the fly (vLLM's qwen3_5 embedding has
    no quant_config)
  - MTP draft mapping: `blk.<n_layers>.nextn.*` → `mtp.*`,
    `blk.<n_layers>.<attn/ffn>` → `mtp.layers.0.*`
- **`quantization/params.py`**:
  - tuple-shard support in the weight-type loader (Qwen3.5 GDN loads
    `in_proj_qkv` as a fused tensor with shards `(0,1,2)`)
  - memory-leak fix: containers of retired lazy parameters are cleared on
    materialization (~6.4 GiB/rank at 27B TP=2)

### 2. vLLM fork (branch `qwen35-gguf-mtp`, base `b5a2adec4`)

Core fixes only, no GGUF-specific code:

- `registry.py`: registers `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM`
  (the classes existed but were not registered)
- `qwen3_5.py`: `Qwen3_5ForCausalLMBase` gains `IsHybrid` (plus the three
  mamba-state classmethods) and `SupportsMRoPE` (text-only positions)
- `speculative.py`: MTP conversion also for `qwen3_5_text` /
  `qwen3_5_moe_text`; the draft ModelConfig inherits `model_weights` from
  the target
- `qwen3_5_mtp.py`: multimodal processor registered on `Qwen3_5MTP`
  (the class declared `SupportsMultiModal`, but without the registration
  MTP + vision crashed during dummy profiling)

## Usage

```bash
pip install -e /spinning/vllm-gguf-plugin   # or vllm-gguf-plugin>=0.0.4

python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model-Q6_K.gguf \
  --tensor-parallel-size 2 \
  --mamba-cache-mode align \
  --dtype bfloat16 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Prerequisite for local GGUF files: `config.json` (text config,
`model_type: qwen3_5_text`, without quantization_config) + tokenizer files
next to the GGUF file, or `--tokenizer <original-hf-repo>`.
Remote works directly: `--model unsloth/Qwen3.5-2B-GGUF:Q8_0 --tokenizer Qwen/Qwen3.5-2B`.

## Test results (2026-07-09, 2× RTX 3080 20GB, TP=2)

| Test | Result |
|---|---|
| Qwen3-0.6B GGUF (sanity, standard attention) | ✅ coherent |
| Qwen3.5-2B Q8_0, TP=1 and TP=2 | ✅ coherent |
| Qwen3.6-27B heretic-v2 Q6_K, TP=2 | ✅ coherent, load 11.5 GiB/rank |
| MTP spec decode (27B, num_speculative_tokens=3) | ✅ acceptance 62–80 %, mean accepted len 2.85–3.39 |
| Benchmark 27B+MTP (max-num-batched-tokens 1600) | decode 41–58 tok/s, prefill 1120 tok/s (7k-token TTFT 6.3 s) |

## Known limits / open items

- ~~Prefill throughput~~ fixed: MMQ kernels are only used up to
  `VLLM_GGUF_MMQ_MAX_TOKENS` (default 16) tokens; larger batches go through
  dequant+cuBLAS → prefill 156 → 1120 tok/s (7.2×) with unchanged decode.
- `out_proj` must be quantized block-aligned (Q8_0/Q4_0/...; a Q6_K
  out_proj with head_v_dim=128 is not invertible → clear error with hint).
- ~~mmproj/vision~~ supported: if a `*mmproj*.gguf` sits next to the model
  and the config.json contains a `vision_config`,
  `Qwen3_5ForConditionalGeneration` is served and the vision encoder is
  loaded from the mmproj file (tested: Q4_K_M + mmproj-BF16, image
  description correct, also combined with MTP). Vision requires the M-RoPE
  fields in config.json (use the full original config) plus
  `preprocessor_config.json` next to the model.
- MTP + CUDA-graph profiling needs headroom: stable on 20-GB cards with
  `--max-num-batched-tokens 1600` and `--gpu-memory-utilization 0.86`;
  default values OOM narrowly.
- Plugin loggers (`vllm_gguf_plugin.*`) are invisible to vLLM's logging
  config (outside the `vllm.*` namespace).
- MoE variant (`qwen35moe`) is mapped but untested.
