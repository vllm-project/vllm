# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""
Ascend attention wrapper that computes key L2 norms for Knorm.

Wraps ``AscendAttentionBackendImpl.forward`` to compute per-token key
L2 norms before delegating to the original Ascend attention implementation.

The norm computation is pure PyTorch (``key.float().norm(dim=-1).mean(dim=1)``)
and works on NPU without any Ascend-specific API.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from vllm.knorm.config import KnormConfig

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Global buffer for pending per-token norms (on NPU device).
# ---------------------------------------------------------------------------
_pending_norms_gpu: list[torch.Tensor] = []
_original_ascend_forward: Callable | None = None


def get_pending_norms() -> list[torch.Tensor]:
    """Return pending norm tensors (still on NPU)."""
    return list(_pending_norms_gpu)


def clear_pending_norms() -> None:
    """Clear the pending norm buffer."""
    _pending_norms_gpu.clear()


def store_layer_norms(norms: torch.Tensor) -> None:
    """Store per-token key norms from one attention layer.

    Args:
        norms: Float tensor of shape ``[num_tokens]``, on NPU device.
    """
    _pending_norms_gpu.append(norms.detach())


# ---------------------------------------------------------------------------
# Norm computation — device-agnostic, works on NPU
# ---------------------------------------------------------------------------


def compute_key_norms(key: torch.Tensor, reduce_op: str = "mean") -> torch.Tensor:
    """Compute per-token importance score from key tensor.

    Args:
        key: shape ``[num_tokens, num_kv_heads, head_size]``
        reduce_op: 'mean', 'max', or 'sum' across heads.

    Returns:
        shape ``[num_tokens]`` float32. Lower norm = more important.
    """
    head_norms = key.float().norm(dim=-1)  # [num_tokens, num_kv_heads]
    if reduce_op == "max":
        return head_norms.max(dim=1).values
    elif reduce_op == "sum":
        return head_norms.sum(dim=1)
    else:  # mean (default)
        return head_norms.mean(dim=1)


# ---------------------------------------------------------------------------
# Wrapper installation
# ---------------------------------------------------------------------------


def install_ascend_wrapper() -> None:
    """Wrap attention impl forward to collect key norms (idempotent).

    Auto-detects the platform:
    - Ascend NPU: wraps ``CUSTOM`` slot (filled by vllm-ascend).
    - CUDA GPU:  wraps ``FLASH_ATTN`` slot (built-in FlashAttention).
    """
    global _original_ascend_forward

    if _original_ascend_forward is not None:
        return  # Already installed.

    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    # Auto-detect platform: Ascend CUSTOM → FLASH_ATTN fallback
    impl_cls: type
    try:
        backend_cls = AttentionBackendEnum.CUSTOM.get_class()
        impl_cls = backend_cls.get_impl_cls()
    except ValueError:
        backend_cls = AttentionBackendEnum.FLASH_ATTN.get_class()
        impl_cls = backend_cls.get_impl_cls()

    _original_ascend_forward = impl_cls.forward
    config = KnormConfig()

    @functools.wraps(_original_ascend_forward)
    def _patched_forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if key is not None and config.is_active:
            key_norms = compute_key_norms(key, config.norm_reduce_op)
            store_layer_norms(key_norms)
        return _original_ascend_forward(
            self,
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

    impl_cls.forward = _patched_forward
