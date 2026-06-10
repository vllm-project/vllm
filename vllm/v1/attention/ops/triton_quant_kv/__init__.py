# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-mode KV cache quantization kernels.

The core attention kernel
(:mod:`vllm.v1.attention.ops.triton_unified_attention`) handles modes
``NONE``, ``FP8_PER_TENSOR``, ``INT8_PER_TOKEN_HEAD`` and
``FP8_PER_TOKEN_HEAD`` directly via constexpr branches.  This package
holds only the pieces that need a bespoke kernel:

  * :mod:`.int8_fp8_per_token_head` — the per-(token, head) absmax
    quantize-on-write kernel shared by INT8 / FP8 (the read side is the
    core kernel).
  * :mod:`.int4_per_token_head` — INT4, whose attention read loop is structurally
    different (split-dot + sub-byte unpack), so it owns both the write
    (reshape) and read (attention) entry points.
  * :mod:`._hadamard` — the random Hadamard transform used by INT4.

Dispatch is an explicit two-way branch (INT4 vs. everything else) in
:mod:`vllm.v1.attention.ops.triton_reshape_and_cache_flash` (write) and
:mod:`vllm.v1.attention.ops.triton_unified_attention` (read).
"""
