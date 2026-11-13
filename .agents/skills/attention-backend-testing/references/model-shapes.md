# Attention shapes from current flagship models

Read from each model's Hugging Face `config.json` on 2026-10-24. Re-read the
configs when a new model generation lands; do not hand-edit numbers without a
source. `H` = query heads, `KV` = KV heads, `D` = head dim, all at TP=1.
Per-rank shapes at TP=t are `H/t` and `max(KV/t, 1)` (KV heads are replicated
once `t > KV`).

## Which backend family each model exercises

| Model (HF id) | Arch | Attention kind | H / KV / D | SWA | Other | Dense MHA backends (FA, ROCM_AITER_FA, TRITON_ATTN...) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3 (`Qwen/Qwen3-235B-A22B`) | Qwen3MoeForCausalLM | full GQA, 94 layers | 64 / 4 / 128 | none | qk-norm | yes, every layer |
| Qwen3.8 (`Qwen/Qwen3.8-27B`) | Qwen3_5ForConditionalGeneration | hybrid: 48 GDN + 16 full (every 4th) | 24 / 4 / 256 | none | gated attn, partial rotary 0.25 | yes, full layers |
| Qwen3.8 (`Qwen/Qwen3.8-2.4T-A95B`) | Qwen3_5MoeForCausalLM | hybrid: 69 GDN + 23 full | 64 / 4 / 256 | none | as above | yes, full layers |
| Qwen3.8-Flash-Next (`Qwen/Qwen3.8-Flash-Next`) | Qwen4ExpForConditionalGeneration | hybrid: 36 linear + 12 full (QSA) | 24 / 2 / 256 | none | QSA sparse | no: uses its own `QWEN4_EXP_QSA_TRITON` backend; shape still useful |
| DeepSeek V3 (`deepseek-ai/DeepSeek-V3`) | DeepseekV3ForCausalLM | MLA | 128 heads, kv_lora 512, rope 64, nope 128, v 128 | none | | no: MLA backends |
| DeepSeek V4 (`deepseek-ai/DeepSeek-V4-Pro`, `-V4-Flash`) | DeepseekV4ForCausalLM | compressed sparse MQA | 128 (Pro) / 64 (Flash) / 1 / 512 | 128 | indexer 64x128, compress ratios 4/128 | no: DSv4 backends |
| DeepSeek V4.1 (`deepseek-ai/DeepSeek-V4.1-Flash`) | DeepseekV41ForCausalLM | compressed sparse MQA | 64 / 1 / 512 | 128 | indexer 32x128, compress ratios 1/2 | no |
| MiniMax M3 (`MiniMaxAI/MiniMax-M3`) | MiniMaxM3SparseForCausalLM | block-sparse (MSA / AITER sparse PA) | 64 / 4 / 128 | none | qk-norm, partial rotary 0.5 | no for the LM (own sparse backend); ViT encoder can use ROCM_AITER_FA |
| MiniMax M3.1 | not published as a full model; only `MiniMaxAI/MiniMax-M3.1-Flash` exists | same as M3 | 64 / 4 / 128 | none | mxfp8 weights | as M3 |
| GLM-5.2 (`zai-org/GLM-5.2`) | GlmMoeDsaForCausalLM | DSA sparse MLA | 64 heads, kv_lora 512, nope 192, rope 64, v 256 | none | indexer 32x128 | no: sparse MLA |
| GLM-5.3 (`zai-org/GLM-5.3`) | GlmMoeDsaForCausalLM | DSA sparse MLA | as GLM-5.2 | none | fp8 | no |
| GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`) | Glm5NextForConditionalGeneration | hybrid linear + DSA | 64 heads, nope 256, v 256, no rope | none | | no |
| Gemma 4 (`google/gemma-4-31B-it`) | Gemma4ForConditionalGeneration | 50 sliding + 10 global | sliding 32 / 16 / 256; global 32 / 4 / 512 | 1024 | final logit softcap 30, KV sharing in E2B/E4B | sliding layers yes; global D=512 exceeds ROCM_AITER_FA head sizes |
| Gemma 4 MoE (`google/gemma-4-26B-A4B-it`) | Gemma4ForConditionalGeneration | 25 sliding + 5 global | sliding 16 / 8 / 256; global 16 / 2 / 512 | 1024 | as above | as above |
| Muse Spark Open (`meta-models/Muse-Spark-Open-816B-A42B`) | MuseSparkForConditionalGeneration | 46 sliding + 16 full | 128 / 16 / 64 | 2048 | final logit softcap 20; fused qk-norm+rope+KV insert bypasses the KV write | yes, every layer (head 64) |

The Muse Spark repo is gated; the numbers were checked against its own
`config.json` with an HF token (the public mlx-community conversion matches).

## Derived test dimensions for dense MHA backends

- Head sizes: 64 (Muse Spark), 128 (Qwen3, MiniMax ViT/LM shape), 256 (Qwen3.8,
  Gemma 4 sliding), 512 (Gemma 4 global: negative/selector test where the
  backend does not support it).
- GQA ratios (H/KV at TP=1): 8 (Muse Spark), 16 (Qwen3, Qwen3.8-2.4T), 6
  (Qwen3.8-27B), 12 (Qwen3.8-Flash-Next), 2 (Gemma 4 sliding), 8 (Gemma 4
  global).
- TP-sharded shapes worth one case each: Qwen3 at TP8 = 8 / 1 / 128 (KV
  replicated), Muse Spark at TP8 = 16 / 2 / 64, Qwen3.8-27B at TP4 = 6 / 1 / 256.
- Sliding windows: 1024 (Gemma 4), 2048 (Muse Spark), 128 (DeepSeek V4 family,
  not a dense-MHA consumer but a useful small-window edge).
- Mixed SWA + full layers in one model: Gemma 4, Muse Spark. Mixed head dims
  across layers: Gemma 4 (256 / 512).
- Hybrid with linear attention: Qwen3.8, GLM-5.3-Flash, so the dense backend
  shares the KV cache pool and block size with GDN/mamba layers.
