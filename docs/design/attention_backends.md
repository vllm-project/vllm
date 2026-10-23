# Attention Backend Feature Support

The priority and feature tables on this page are auto-generated from the
attention backend registry by
`docs/mkdocs/gen_files/generate_attention_backends.py`, based on the checks in
`AttentionBackend.validate_configuration()`.

## Setting the Attention Backend

### Command Line

There are two ways to specify the backend from the command line:

**Option 1: Using `--attention-backend` (simple)**

```bash
vllm serve <model> --attention-backend FLASH_ATTN
```

**Option 2: Using `--attention-config.backend` / `-ac.backend` (structured config)**

```bash
# Dot notation
vllm serve <model> --attention-config.backend FLASH_ATTN
vllm serve <model> -ac.backend FLASH_ATTN

# JSON format
vllm serve <model> --attention-config '{"backend": "FLASH_ATTN"}'
vllm serve <model> -ac '{"backend": "FLASH_ATTN"}'
```

> **Note:** `--attention-backend` and `--attention-config.backend` are mutually
> exclusive. Use one or the other, not both.

### Python API

Use `AttentionConfig` with the `LLM` class:

```python
from vllm import LLM
from vllm.config import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

# Method 1: Using AttentionConfig with enum
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    attention_config=AttentionConfig(backend=AttentionBackendEnum.FLASH_ATTN),
)

# Method 2: Using attention_backend parameter with string
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    attention_backend="FLASH_ATTN",
)
```

## Backend Selection Behavior

### Triton/FlashAttention Composite

On Hopper, `TRITON_FLASH_ATTN` is preferred for compatible multimodal-prefix
configurations. It uses Triton when the current queries need bidirectional
image attention and FlashAttention for causal text prefills and decode, with a
shared KV cache. The causal child must resolve to FA4. This happens
automatically for FA4-only shapes such as head size 512 and models whose
version policy promotes all layers to FA4, including Gemma 4. A standalone
head-size-256 configuration whose version policy selects FA3 falls back to
Triton for the whole layer.

No attention override is needed for supported multimodal models:

```bash
vllm serve google/gemma-4-31B-it
```

### Triton/FlashInfer Composite

On Blackwell, `TRITON_FLASHINFER` is preferred for compatible
multimodal-prefix configurations, including Gemma 4 with BF16 or FP8 KV cache. It uses
Triton when a batch's current queries require bidirectional image attention,
and FlashInfer for causal text prefills and decode. Historical image tokens
alone do not select Triton.

This instantiates `create_composite_attention_backend` from
`vllm/v1/attention/backends/composite.py` with Triton,
FlashInfer, and `MMPrefixAttentionRouting`. The reusable factory owns child
implementations, metadata dispatch, compatible cache requirements, and workspace
sharing. The routing policy selects the child and defines graph-capture safety.
Other combinations can reuse the same machinery.

The backend supports head dimensions 256/512, FP16/BF16 and FP8 KV cache,
and 64-token kernel pages with a head-major cache layout. TRTLLM handles causal
attention at both head dimensions; its hdim512 kernels do not support 128-token
pages.
For Gemma 4, full CUDA graphs cover single-token batches; multi-token batches
use the non-full-graph execution path. Models whose image masks extend beyond
the sliding window cannot use full attention graphs with this composite.
Context parallelism, R-SWA, attention sinks,
and adaptive verification are not supported by this composite.

It can also be selected explicitly:

```bash
vllm serve google/gemma-4-31B-it \
    --attention-backend TRITON_FLASHINFER
```

### Manual Selection

When you explicitly set a backend via `--attention-backend` or `AttentionConfig`:

1. The backend is **validated** against your configuration (model dtype, head
   size, compute capability, etc.)
2. If the backend **doesn't support** your configuration, an error is raised
   with the specific reason
3. If valid, the backend is used

Example error when selecting an incompatible backend:

```text
ValueError: Selected backend FLASHMLA is not valid for this configuration.
Reason: ['compute capability not supported']
```

### Automatic Selection

When no backend is specified (the default):

1. vLLM iterates through backends in **priority order** (see tables below)
2. Each backend is validated against your configuration
3. The **first compatible backend** is selected
4. If no backend is compatible, an error is raised listing all backends and
   their incompatibility reasons

## Backend Priority (CUDA)

When no backend is explicitly selected, vLLM chooses the first
compatible backend from these priority-ordered lists.

Priority is **1 = highest** (tried first).

### Standard Attention (MHA, MQA, GQA)

--8<-- "gen:priority-standard"

### MLA Attention (DeepSeek-style)

--8<-- "gen:priority-mla"

> **\*** For sparse MLA, FP8 KV cache always prefers `FLASHINFER_MLA_SPARSE`. With BF16 KV cache, `FLASH_ATTN_MLA_SPARSE_FA4` (then `FLASHINFER_MLA_SPARSE`) is preferred for low query-head counts (<= 16) on SM 10.x, while `FLASHMLA_SPARSE` is preferred otherwise.
>
> **Note:** ROCm and CPU platforms have their own selection logic. See the platform-specific documentation for details.

## Legend

| Column | Description |
| ------ | ----------- |
| **Dtypes** | Supported model data types (fp16, bf16, fp32) |
| **KV Dtypes** | Supported KV cache data types (`auto`, `fp8`, `fp8_e4m3`, etc.) |
| **Block Sizes** | Supported KV cache block sizes (%N means multiples of N) |
| **Head Sizes** | Supported attention head sizes |
| **Sink** | Attention sink support (for StreamingLLM) |
| **Non-Causal** | Non-causal (bidirectional) attention support for decoder models |
| **Sparse** | Sparse attention support (MLA only) |
| **MM Prefix** | Multimodal prefix full attention support |
| **DCP** | Decode Context Parallelism support (`--decode-context-parallel-size`) |
| **Attention Types** | Supported attention patterns (Decoder, Encoder, Enc-Dec) |
| **Compute Cap.** | Required CUDA compute capability (N/A for non-CUDA backends) |

**Symbols:** ✅ = Supported, ❌ = Not supported

## Standard Attention (MHA, MQA, GQA) Backends

--8<-- "gen:table-standard"

> **†** FlashInfer Native is the regular FlashInfer path. XQA is the SM90 decode path exposed through FlashInfer's TRTLLM decode API. trtllm-gen is used on SM100 and supports sinks. Disable XQA/trtllm-gen via `--attention-config.use_trtllm_attention=0`.
>
> **\*** Specify the FlashAttention version via `--attention-config.flash_attn_version=2`, `3`, or `4`. Default is FA4 on SM100+ (Blackwell), FA3 on SM90 (Hopper), FA2 otherwise.
>
> On Blackwell, when the FlashAttention backend is selected, `head_size=256` is served
> by a dedicated FA4 kernel that requires a KV cache block size of 128 (advertised
> automatically, so it is picked as long as `--block-size` is not pinned) and does not
> support logit soft capping, attention sinks, mm_prefix/R-SWA masking, DCP, or windowed
> encoder attention. Those configurations transparently fall back to FA2. A pinned
> `--block-size` that is not a multiple of 128 instead makes FlashAttention ineligible
> for such models, which is an error if the backend was requested explicitly.

### b12x

The optional [b12x](https://pypi.org/project/b12x/) backend supports causal
decoder attention on NVIDIA SM120 and SM121 GPUs. Install and select it with:

```bash
uv pip install "vllm[b12x]"
vllm serve <model> --attention-backend b12x
```

## MiniMax M3 Sparse Attention Backends

Block-sparse GQA backend used by MiniMax M3 sparse ("lightning indexer")
layers. It is wired in directly by the model and is not part of the
automatic priority lists above. A lightning indexer scores KV blocks, the
top-k blocks (plus fixed init/local blocks) are selected, and attention
attends only to those blocks; index keys live in a separate side cache.

--8<-- "gen:table-minimax"

### MiniMax M3 specialized decode on MI355X

`--monokernel` enables a fused sparse-layer pipeline on TP4 MI355X deployments of
`amd/MiniMax-M3-MXFP4`. It requires `--moe-backend aiter`, FP8 KV and index
caches, block size 128, and Model Runner V2 (`VLLM_USE_V2_MODEL_RUNNER=1`).
Decode steps with at most 16
physical token rows use the fused launch, including EAGLE3 verification and
FULL graph replay. Other steps run the standard layers.

Both attention projections must use online per-channel FP8 quantization.
For example, add these options to a TP4 deployment:

```bash
--monokernel --moe-backend aiter --kv-cache-dtype fp8 --block-size 128 \
--attention-config '{"indexer_kv_dtype":"fp8"}' \
--quantization-config '{"targets":{"*self_attn*qkv_proj":"fp8_per_channel","*self_attn*o_proj":"fp8_per_channel"}}'
```

The default `layer` implementation fuses normalization, FP8 projections, the
indexer, sparse attention, and the MoE. Adapted from
[ATOM](https://github.com/ROCm/ATOM/tree/e543f331c78028dfa6ed63987c4fae919734c898/atom/models/minimax_m3/mono),
it uses group-of-32 MXFP8 activations for the MXFP4 experts, BF16 router
weights and logits, and a BF16 rotary table, matching ATOM's precision choices.
The kernels and their runtime live in vLLM; ATOM is not a runtime dependency.
The existing FlyDSL and AITER packing utilities are still required. Prefill,
the first three dense layers, and decode steps above 16 physical rows use the
standard implementation. Native weights and caches remain available for those
steps.

To retain the previous FFN-only implementation, set
`--kernel-config '{"monokernel_mode":"ffn"}'`. This uses BF16 expert
activations and an FP32 router, with native attention and projection launches.
Existing deployments without FP8 attention projections must select this mode
or configure FP8 projections before enabling the new default.

With `--attention-backend TRITON_ATTN`, TP4, FP8 KV caches, and at most eight
sequences, `layer` mode uses AITER unified attention for dense prefills of at
least 64 query tokens, with BF16 queries and probabilities as in ATOM.
Quantized queries are dequantized before this attention call. Short queries
retain the Triton decode kernel with 256 softmax segments.

For long text prompts, `VLLM_USE_FASTOKENS=1` selects the optional `fastokens`
BPE backend for Hugging Face tokenizers. Install `fastokens>=0.2.0` before
enabling it. This setting also applies to tokenizers used with layer mode.

For an EAGLE3 draft, also set `"quantization":"fp8_per_channel"` inside
`--speculative-config` to quantize its linears. The target's
`--quantization-config` does not configure the draft model.

The evaluated fastokens and EAGLE3 configuration uses `--max-num-seqs 2`.
Higher sequence limits changed generation behavior in model evaluations,
including reasoning that reached the output-token limit.

This layer-mode recipe was evaluated with the 102-case MiniMax verifier and
AgentX at client concurrency one and four on TP4 MI355X. It uses three EAGLE3
draft tokens and real rejection sampling:

```bash
uv pip install fastokens==0.2.0
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_FASTOKENS=1
export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_USE_BREAKABLE_CUDAGRAPH=0 OMP_NUM_THREADS=1

vllm serve amd/MiniMax-M3-MXFP4 \
  --tensor-parallel-size 4 --distributed-executor-backend mp \
  --language-model-only --trust-remote-code --monokernel --moe-backend aiter \
  --kv-cache-dtype fp8 --block-size 128 --attention-backend TRITON_ATTN \
  --attention-config '{"indexer_kv_dtype":"fp8"}' \
  --quantization-config '{"targets":{"*self_attn*qkv_proj":"fp8_per_channel","*self_attn*o_proj":"fp8_per_channel","*mlp.gate_up_proj":"fp8_per_channel","*mlp.down_proj":"fp8_per_channel"}}' \
  --kernel-config '{"enable_jit_warmup":false}' \
  --speculative-config '{"method":"eagle3","model":"Inferact/MiniMax-M3-EAGLE3-GQA","num_speculative_tokens":3,"quantization":"fp8_per_channel"}' \
  --max-model-len 1048576 --max-num-seqs 2 --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.95 --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,22,24,26,28,30,32,34,36,40,48,56,64]}' \
  --default-chat-template-kwargs '{"thinking_mode":"enabled"}' \
  --enable-auto-tool-choice --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3
```

For the AgentX performance comparison only, the speculative configuration also
used `"rejection_sample_method":"synthetic"` and
`"synthetic_acceptance_rates":[1.0,0.78,0.0]`, matching the reference benchmark.
The model evaluation used real rejection sampling as shown above.

## MLA (Multi-head Latent Attention) Backends

MLA uses separate backends for prefill and decode phases.

### Prefill Backends

To explicitly select a prefill backend, use
`-ac.mla_prefill_backend=<BACKEND>` (e.g., `FLASH_ATTN`, `FLASHINFER`).
Otherwise, the prefill backend is selected automatically at runtime based on
hardware and configuration.

--8<-- "gen:table-mla-prefill"

> **‡** Automatic selection tries FlashAttention first. On Blackwell
> (SM100), the fallback order is TRT-LLM Ragged, FlashInfer, then
> TokenSpeed MLA; for (qk_nope_head_dim=192, qk_rope_head_dim=64,
> v_head_dim=256) TRT-LLM Ragged is tried before FlashAttention. On other
> GPUs, only FlashAttention is considered.

### Decode Backends

MLA decode backends are selected using the standard
`-ac.backend=<BACKEND>` argument (e.g., `FLASHMLA`, `TRITON_MLA`).

--8<-- "gen:table-mla-decode"

### DeepSeek V4 Decode Backends

DeepSeek V4 sparse MLA uses its own decode backends, selected via
`--attention-backend=<BACKEND>` (e.g., `FLASHMLA_SPARSE_DSV4`,
`FLASHINFER_MLA_SPARSE_DSV4`). They share the V4 sparse-index
pipeline (compressor + SWA + indexer, 256-token blocks, head 512);
default on NVIDIA is `FLASHINFER_MLA_SPARSE_DSV4` on SM12x and
`FLASHMLA_SPARSE_DSV4` on other supported CUDA architectures.

--8<-- "gen:table-mla-v4-decode"
