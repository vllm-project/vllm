# Vendored FlashMLA kernel (LiteDSA grouped sparse attention)

`sm100/prefill/sparse/fwd/head128_fp8/phase1.cuh` is the self-contained
SM100 FP8 grouped sparse-attention kernel compiled by
`csrc/libtorch_stable/attention/dsa/litedsa.cu`.  It amalgamates the required
FlashMLA parameter definitions, TMA/tcgen05 helpers, barriers, MMA atoms, and
the masked attention implementation into one header; no other local
FlashMLA header is part of the compile graph.

The implementation is derived from DeepSeek FlashMLA
(<https://github.com/deepseek-ai/FlashMLA>, upstream commit `9241ae3`) and
adds query-major membership masking for exact grouped attention.  See
`LICENSE.deepseek-flashmla` for the upstream MIT license.
