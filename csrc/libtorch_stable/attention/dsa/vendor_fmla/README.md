# Vendored FlashMLA headers (LiteDSA grouped sparse attention)

A minimal, self-contained subset of DeepSeek's FlashMLA
(<https://github.com/deepseek-ai/FlashMLA>, MIT, upstream commit `9241ae3`)
needed by the grouped sparse prefill attention kernel
(`csrc/libtorch_stable/attention/dsa/litedsa.cu`). These are header-only
SM100 primitives (TMA descriptors, tcgen05 UMMA wrappers, barrier and
warp-specialization helpers) plus the sparse-attention parameter struct.
They depend only on CUTLASS/CuTe and the C++ stdlib.

Upstream license: see `LICENSE.deepseek-flashmla`
(MIT, Copyright (c) 2025 DeepSeek).

## Contents

Unmodified from upstream:

- `utils.h`, `defines.h`, `sm100/helpers.h`
- `kerutils/` (15 headers: device intrinsics, TMA, SM80/90/100 helpers)

Modified from upstream:

- `params.h` — adds the per-query membership mask fields (`membership`,
  `membership_qm`, `h_per_q`, `topk_length_per_q`, `q_group_div`,
  `out_scale`) consumed by the masked kernel below.

New in this tree (derived from upstream's `sm100/prefill/sparse/fwd/head128`):

- `sm100/prefill/sparse/fwd/head128_fp8/` — an fp8 (e4m3) variant of the
  head128 sparse prefill forward with per-query membership masking. Q/K/V
  are read as e4m3 with per-tensor scales, both GEMMs run on fp8 tensor
  cores, and a query-major membership bitmask restores exact per-query
  attend-sets inside the kernel. Includes `mma_fp8_noelect.h`, a 2-SM fp8
  MMA atom written for this variant.

Because `head128_fp8/` is a derivative rather than a copy, keeping it in
sync with upstream FlashMLA is manual; contributing the variant back
upstream would remove that burden.
