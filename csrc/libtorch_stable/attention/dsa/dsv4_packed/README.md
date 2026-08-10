# DSv4 head-packed sparse attention (C128A prefill)

At TP8 DeepSeek-V4 owns 8 of 64 heads per rank, but FlashMLA's sparse prefill
kernel only accepts h_q=64, so the stock path pads q with 56 zero heads and
discards 7/8 of every tensor-core tile. This kernel packs `128 / h` query
tokens x `h` real heads into one 128-row tile instead. Exactness comes from
two per-query ranges rather than a bitmask: query g attends union ranks
`[0, pref_g) U [lo_g, hi_g)`, which is enough because a C128A index row is
structurally a shared compressed prefix followed by a contiguous SWA window.

## Layout

- `litedsa_attention_sm100_dsv4.cuh` — bf16 kernel (fork of the GLM fp8
  masked kernel)
- `litedsa_dsv4_atoms.cuh` — `SM100_MMA_F16BF16_2x1SM_SS_NOELECT` + traits
- `litedsa_dsv4.cu` / `litedsa_dsv4_binding.cu` — host entry, TVM-FFI export
  `dsv4_masked_mla_bf16`
- `dsa_dsv4.so` — local sm100a build artifact (not tracked). Build it with
  `build_dsv4.sh`; the default output is
  `/tmp/litedsa_dsv4_build/dsa_dsv4.so`. Select the resulting artifact with
  `VLLM_DSV4_PACKED_SO`. Set `PYTHON_BIN`, `CUDA_HOME`, and
  `LITEDSA_DSV4_BUILD_DIR` when their defaults do not match your environment.
The kernel has a compile-time `static_assert` for the 232448-byte SM100
dynamic-shared-memory cap. Its current `SharedMemoryPlan` is 215024 bytes.

## Driver

`vllm/model_executor/layers/dsv4_packed_attn.py`, called from the C128A
prefill branch of `vllm/models/deepseek_v4/nvidia/flashmla.py`. It derives the
union list and per-query ranges from positions alone, so the
`[num_tokens, topk+window]` combined index matrix is never materialized, and
caches them per chunk (they depend only on positions, not on the layer).

## Env

| var | default | meaning |
|---|---|---|
| `VLLM_DSV4_PACKED_ATTN` | 0 | enable the packed path |
| `VLLM_DSV4_PACKED_CHECK` | 0 | also run the stock kernel and log the per-layer relative error (validation; halves speed) |
| `VLLM_DSV4_PACKED_SO` | — | override the kernel .so path |

The path falls back to the stock kernel for any shape it cannot serve exactly
(multi-request chunk, token count not divisible by the packing factor,
`128 % h != 0`, `128 / h > 16`, d != 512), and if the .so fails to load.

## Historical packing sweep

The table below predates the current four-arm `raw / litetopk / litedsa /
combo` E2E harness. It isolates an earlier packed-attention iteration and is
kept only as a TP-scaling reference; do not use it as the current end-to-end
result.

| TP | heads/rank | pack | baseline | packed | ratio |
|---|---|---|---|---|---|
| 8 | 8 | 16 tok | 54.980 s | **46.744 s** | **1.176x** |
| 4 | 16 | 8 tok | 59.653 s | 52.982 s | 1.126x |
| 2 | 32 | 4 tok | 76.009 s | 69.952 s | 1.087x |

The ratio falls with TP because the padding tax is `64 / heads-per-rank`.
Kernel-level against captured production data: 6.565 ms -> 0.839 ms (7.83x),
lse max abs 3e-6, output relative error 0.0025.

Known remaining cost: two host-side copies (q repack, output scatter) worth
~12% of the packed path at TP8 and more at lower TP.
