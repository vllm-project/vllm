# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared per-token-head KV packing helpers (Triton + FA int4).

Keeps FlashAttentionBackend.customize_spec and TritonAttentionBackend.customize_spec
on the same C=136 formula for D=128 INT4 (avoids the float16→C=264 trap).
"""

from __future__ import annotations

from dataclasses import replace

import torch

from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode


def per_token_head_state_content_bytes(spec: AttentionSpec) -> int | None:
    """Return packed content bytes for per-token-head modes, else None."""
    mode = spec.kv_quant_mode
    if not mode.is_per_token_head:
        return None
    hs_k, hs_v = spec.head_size, spec.head_size_v
    # INT4 storage cells are always uint8 (2×nibble / byte), independent of
    # the logical activation dtype on the AttentionSpec.
    if mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        hs_k, hs_v = hs_k // 2, hs_v // 2
        data_dtype_size = get_dtype_size(torch.uint8)
    else:
        data_dtype_size = get_dtype_size(spec.dtype)
    scale_bytes = get_dtype_size(torch.float32)
    return (hs_k + hs_v) * data_dtype_size + 2 * scale_bytes


def customize_per_token_head_spec(spec: AttentionSpec) -> AttentionSpec:
    """Publish ``state_content_bytes`` for per-token-head packing if unset."""
    if spec.state_content_bytes is not None:
        return spec
    content = per_token_head_state_content_bytes(spec)
    if content is None:
        return spec
    return replace(spec, state_content_bytes=content)


def int4_packed_layout(head_size: int) -> tuple[int, int, int]:
    """Return ``(hs_packed, scale_pad_u8, content_bytes)`` for INT4 D=head_size.

    Layout per (token, kv_head):
    ``[K(hs)|K_scale(pad)|V(hs)|V_scale(pad)]`` with ``pad = 4`` (fp32 as u8).
    """
    hs = head_size // 2
    pad = get_dtype_size(torch.float32)  # / sizeof(uint8)
    content = 2 * (hs + pad)
    return hs, pad, content
