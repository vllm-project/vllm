# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMD DeepSeek-V4 fused compressor adapter (gfx950 / CDNA4).

The fused compressor kernels (CSA / HCA / indexer) are built into the _rocm_C
extension from ``csrc/rocm/dsv4_*_compress.cu`` and exposed as
``torch.ops._rocm_C.dsv4_{csa,hca,indexer}_compress`` (gfx950 only).

  - ``hip_compress_dispatch.py`` : the adapter ``compress_norm_rope_store_hip``
    + ``hip_compressor_supported`` used by ``compressor.py`` (opt-in via
    ``VLLM_ROCM_DSV4_HIP_COMPRESSOR``).

Correctness tests and the DVFS-robust micro-benchmark live with the rest of the
kernel suite, not in this package:
  - ``tests/kernels/attention/test_dsv4_compress.py``           (byte-exact)
  - ``tests/kernels/attention/test_dsv4_compress_arch_guard.py`` (gating)
  - ``tests/kernels/attention/dsv4_compress_utils.py``          (scaffolding)
  - ``benchmarks/kernels/benchmark_dsv4_compress.py``           (perf)
"""
