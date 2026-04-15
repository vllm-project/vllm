# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend protocol for KV cache quantization modes.

A ``QuantKVBackend`` owns the cache write path (``reshape_and_cache``) and
the attention read path (``unified_attention``) for one ``KVQuantMode``.
The core attention/reshape kernels stay quantization-agnostic; each mode
that wants its own data layout, packing, or pre-rotation lives in a
self-contained module under ``quant_kv/`` and registers an instance of
this class on import.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from vllm.v1.kv_cache_interface import KVQuantMode

if TYPE_CHECKING:
    pass


class QuantKVBackend(ABC):
    """Cache write + attention read for one KV quantization mode.

    Subclasses implement ``reshape_and_cache`` and ``unified_attention``
    for a single :class:`KVQuantMode`, and call
    :func:`vllm.v1.attention.ops.triton_quant_kv.register` at module import.
    The dispatcher in :mod:`triton_unified_attention` and
    :mod:`triton_reshape_and_cache_flash` looks up the backend lazily on
    first use, so unused modes pay zero import or compile cost.
    """

    # ----- Static metadata --------------------------------------------------
    #: Mode this backend implements.  Must be set by subclasses.
    mode: KVQuantMode
    #: Number of cache *bytes* used per logical KV element (1 unless packed).
    packing_factor: int = 1
    #: Whether this mode allocates its own per-(token, head) scale buffers.
    needs_scale_caches: bool = False

    # ----- Cache shape introspection ----------------------------------------
    def packed_head_size(self, head_size: int) -> int:
        """Storage head size after packing: ``head_size // packing_factor``."""
        assert head_size % self.packing_factor == 0, (
            f"head_size={head_size} is not divisible by packing factor "
            f"{self.packing_factor} required by {self.mode.name}"
        )
        return head_size // self.packing_factor

    def allocate_scale_caches(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Allocate aux per-(token, head) scale buffers.

        Default: this mode does not need scale caches.  Override for
        per-token-head modes that store one float per (token, head).
        """
        return (None, None)

    # ----- Cache write path -------------------------------------------------
    @abstractmethod
    def reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        k_scale_cache: torch.Tensor | None = None,
        v_scale_cache: torch.Tensor | None = None,
    ) -> None:
        """Write *key*/*value* into the paged cache for this mode.

        Per-token-head modes also write into ``k_scale_cache`` /
        ``v_scale_cache``.
        """

    # ----- Attention read path ----------------------------------------------
    # Only modes that need a bespoke attention loop (INT4 / INT2 with
    # split-dot + sub-byte unpack) override this.  INT8 / FP8 per-token-head
    # use the core kernel via a constexpr branch and never call this method.
    def unified_attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        out: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        seqused_k: torch.Tensor,
        max_seqlen_k: int,
        softmax_scale: float,
        window_size: tuple[int, int],
        block_table: torch.Tensor,
        softcap: float,
        sinks: torch.Tensor | None,
        alibi_slopes: torch.Tensor | None,
        use_alibi_sqrt: bool,
        qq_bias: torch.Tensor | None,
        output_scale: torch.Tensor | None,
        mm_prefix_range: torch.Tensor | None,
        k_scale_cache: torch.Tensor | None = None,
        v_scale_cache: torch.Tensor | None = None,
        # Optional 3D-decode pre-allocated buffers (same as the core kernel)
        seq_threshold_3D: int | None = None,
        num_par_softmax_segments: int | None = None,
        softmax_segm_output: torch.Tensor | None = None,
        softmax_segm_max: torch.Tensor | None = None,
        softmax_segm_expsum: torch.Tensor | None = None,
    ) -> None:
        """Run paged attention with this mode's KV layout, writing into *out*."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement a bespoke attention "
            f"kernel.  This mode should be handled by the core kernel."
        )
