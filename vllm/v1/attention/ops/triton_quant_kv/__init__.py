# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache quantization kernels that the core kernel can't share.

The core kernel handles NONE, FP8 per-tensor and INT8/FP8 per-token-head
via constexpr branches (write side in ``triton_reshape_and_cache_flash``,
read side in ``triton_unified_attention``). This package holds only the
INT4 pieces, which need sub-byte packing and a Hadamard rotation the core
kernel can't express: ``int4_per_token_head`` (packed write + split-dot
read + the RHT transform). INT4 vs. everything-else is dispatched explicitly
in both core modules.
"""
