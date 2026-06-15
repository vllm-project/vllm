# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL kernels for the Kimi-K2.5 NVFP4 specialized model.

Each kernel lives in its own module:

- :mod:`.qkv_rmsnorm_k_pe_fused` -- fused Q/KV-LoRA RMSNorm + key RoPE.
- :mod:`.decode_rope_concat_quant_fp8_and_cache_mla` -- fused decode-query
  RoPE + concat + FP8 quant, paired with the paged FP8 KV-cache write.

Shared CuTe DSL compile helpers live in :mod:`.cutedsl_utils`.

The kernel modules are imported explicitly (rather than re-exported here) so
that importing this package does not require the optional ``cutlass``
dependency or a Blackwell GPU.
"""
