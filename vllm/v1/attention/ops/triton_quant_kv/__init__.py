# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache quantization kernels that the core kernel can't share.

The core kernel handles NONE, FP8 per-tensor and INT8/FP8 per-token-head
via constexpr branches. This package holds only the bespoke pieces:
``int8_fp8_per_token_head`` (per-(token,head) absmax write kernel; read side
is the core kernel) and ``int4_per_token_head`` (packed write + split-dot
read + the RHT transform). INT4 vs. everything-else is dispatched explicitly
in ``triton_reshape_and_cache_flash`` (write) and ``triton_unified_attention``
(read).
"""
