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

> **\*** For sparse MLA, FP8 KV cache always prefers `FLASHINFER_MLA_SPARSE`. With BF16 KV cache, `FLASHINFER_MLA_SPARSE` is preferred for low query-head counts (<= 16), while `FLASHMLA_SPARSE` is preferred otherwise.
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

## MiniMax M3 Sparse Attention Backends

Block-sparse GQA backend used by MiniMax M3 sparse ("lightning indexer")
layers. It is wired in directly by the model and is not part of the
automatic priority lists above. A lightning indexer scores KV blocks, the
top-k blocks (plus fixed init/local blocks) are selected, and attention
attends only to those blocks; index keys live in a separate side cache.

--8<-- "gen:table-minimax"

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
> TokenSpeed MLA. On other GPUs, only FlashAttention is considered.

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
