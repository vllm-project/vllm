# LiteTopK and LiteDSA

LiteTopK and LiteDSA are opt-in SM100a prefill optimizations for models that
use DeepSeek Sparse Attention (DSA). LiteTopK avoids materializing the full
query-by-sequence logits tensor while selecting sparse indices. LiteDSA packs
the real attention heads from adjacent queries into a 128-row sparse-attention
tile and reuses their deduplicated union of selected keys.

## Requirements

- NVIDIA B200 (`sm_100a`);
- CUDA 12.8 or newer;
- a CUDA compiler and C++ build toolchain available for the first-use
  LiteTopK JIT build;
- a model with a supported DSA indexer layout;
- FP8 query and KV tensors for the grouped LiteDSA attention path;
- single-request chunked prefill with a supported query length.

Unsupported devices, shapes, cache formats, decoding, and multi-request
chunks use the stock vLLM implementation. The fused indexer is approximate:
qualification measures top-k recall against the stock selector. LiteDSA uses
the exact per-query membership mask over a grouped union, but floating-point
outputs can differ slightly because the execution order changes.

## Enable the optimized paths

Enable only LiteTopK:

```bash
VLLM_LITETOPK=1 VLLM_DSA_MODE=litetopk vllm serve MODEL
```

Enable LiteTopK together with grouped FP8 LiteDSA attention:

```bash
VLLM_LITETOPK=1 VLLM_DSA_MODE=litedsa vllm serve MODEL
```

The LiteTopK CUDA sources are included in the wheel and compiled into the
user cache on first use. A source digest is included in the extension name so
different source revisions cannot silently share a binary.

DeepSeek-V4's BF16 packed C128 attention specialization is experimental and
is not part of the wheel build. It additionally requires an explicitly built
TVM-FFI module selected with `VLLM_DSV4_PACKED_SO`; otherwise it falls back to
the stock attention path.

## Validation

Set `VLLM_LITETOPK_CHECK=1` to compare fused index selections with the stock
selector during qualification. This mode adds substantial work and is not a
performance configuration. Candidate overflow and selector status are
fail-stop conditions; no truncated candidate list is returned to attention.

When reporting performance, keep the model, tensor-parallel and
expert-parallel topology, prefill chunk, cache dtype, scheduler, and input
tokens identical across the raw, LiteTopK, LiteDSA, and combined arms.
