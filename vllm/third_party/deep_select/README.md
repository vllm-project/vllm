# DeepSelect bf16 cluster top-k

The kernel the DeepSeek-V4.1 specialized decode runs for its dense indexers'
token top-k (see
`vllm/models/deepseek_v41/nvidia/specialized_decode/attention/cluster_topk.py`).

The specialized decode vendors this kernel instead of calling the
ahead-of-time DeepSelect build (`cmake/external_projects/deepselect.cmake`,
`torch.ops.deep_select`): that build's cluster kernel is DeepSelect's own,
without the rewrite below, its gather signals completion barriers in the
wrong CTA (deepseek-ai/DeepSelect#16), and it gives a row holding a NaN the
`0x3F3F3F3F` marker, which the sparse-index table build does not mask.

The files are SGLang's vendored copy of
[deepseek-ai/DeepSelect](https://github.com/deepseek-ai/DeepSelect) at
[sgl-project/sglang](https://github.com/sgl-project/sglang) commit
`656331295619a4d0267b40b7da0b5cd5b549fcbc`, path
`python/sglang/kernels/jit/csrc/deep_select/vendor/`:

- DeepSelect at commit `382d62a` (with the SM90 cluster variant of upstream
  PR #14), MIT licensed (`LICENSE`, Copyright (c) 2025 DeepSeek), including the
  kerutils subset under `3rdparty/`.
- SGLang's changes, Apache-2.0, marked `[SGLang]`: the rewritten
  `cuda_kernels/v3_cluster/topk_select.cuh` (contiguous split, all-to-all
  `st.async` histogram exchange, per-warp output cursor; sglang #44051) and
  the page-table epilogue (sglang #41364).
- vLLM's changes, marked `[vLLM]`: programmatic dependent launch in
  `cuda_kernels/v3_cluster/topk_select.cuh` (the grid waits on its
  predecessor before its first global access, triggers its dependents once
  its window has landed, and launches with the PDL attribute).

Only the cluster kernel is kept: SGLang's `cuda_kernels/v3/` and
`cuda_kernels/v3_fp32/` (one CTA a row) and its `entry.cuh` are left out.
