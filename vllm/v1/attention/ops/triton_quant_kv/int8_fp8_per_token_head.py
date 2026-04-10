# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 and FP8 per-token-head KV cache quantization backends.

Both modes share the exact same code path: load KV + cast to Q's dtype
+ multiply S/P by per-(token, head) scales.  The only differences are:

  * the storage dtype (``torch.int8`` vs the platform's fp8 type), which
    Triton infers from the cache pointer, and
  * ``(QUANT_MAX, QUANT_MIN)`` for the reshape kernel.

So both backends live in the same file and share every helper.  The
shared kernels live in :mod:`._per_token_head_kernel`.

Symmetric quantization::

    scale = absmax / QUANT_MAX
    q = clamp(round(x / scale), QUANT_MIN, QUANT_MAX)
    x_hat = q * scale
"""

from __future__ import annotations

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.v1.attention.ops.triton_quant_kv import register
from vllm.v1.attention.ops.triton_quant_kv._per_token_head_kernel import (
    run_attention,
    run_reshape_and_cache,
)
from vllm.v1.attention.ops.triton_quant_kv.base import QuantKVBackend
from vllm.v1.kv_cache_interface import KVQuantMode

# ---------------------------------------------------------------------------
# Per-mode quantization range
# ---------------------------------------------------------------------------
# Symmetric INT8 dynamic range
_INT8_QUANT_MAX = 127.0
_INT8_QUANT_MIN = -128.0

# FP8 range from the platform's fp8 dtype (e4m3 on NVIDIA, e4m3fnuz on AMD).
_FP8_QUANT_MIN, _FP8_QUANT_MAX = get_fp8_min_max()


class _PerTokenHeadBackend(QuantKVBackend):
    """Common implementation for INT8 / FP8 per-token-head modes.

    Subclasses set ``mode``, ``_quant_max`` and ``_quant_min``.  Both the
    attention and reshape kernels are shared via
    :mod:`._per_token_head_kernel`.
    """

    packing_factor = 1  # one element per cache byte
    needs_scale_caches = True

    # Set by subclasses
    _quant_max: float
    _quant_min: float

    def allocate_scale_caches(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (num_blocks, block_size, num_kv_heads)
        return (
            torch.zeros(shape, dtype=torch.float32, device=device),
            torch.zeros(shape, dtype=torch.float32, device=device),
        )

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
        assert k_scale_cache is not None and v_scale_cache is not None, (
            f"{self.mode.name} requires k_scale_cache / v_scale_cache"
        )
        run_reshape_and_cache(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            slot_mapping=slot_mapping,
            quant_max=self._quant_max,
            quant_min=self._quant_min,
        )

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
        seq_threshold_3D: int | None = None,
        num_par_softmax_segments: int | None = None,
        softmax_segm_output: torch.Tensor | None = None,
        softmax_segm_max: torch.Tensor | None = None,
        softmax_segm_expsum: torch.Tensor | None = None,
    ) -> None:
        assert k_scale_cache is not None and v_scale_cache is not None, (
            f"{self.mode.name} requires k_scale_cache / v_scale_cache"
        )
        # Imported here (not at module level) so this backend module
        # doesn't pull envs/heuristics from the core unless actually used.
        import vllm.envs as envs
        from vllm.v1.attention.ops.triton_unified_attention import _get_tile_size

        run_attention(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            softmax_scale=softmax_scale,
            window_size=window_size,
            block_table=block_table,
            softcap=softcap,
            sinks=sinks,
            alibi_slopes=alibi_slopes,
            use_alibi_sqrt=use_alibi_sqrt,
            qq_bias=qq_bias,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            is_batch_invariant=envs.VLLM_BATCH_INVARIANT,
            get_tile_size=_get_tile_size,
        )


# ---------------------------------------------------------------------------
# Concrete backends
# ---------------------------------------------------------------------------
class Int8PerTokenHeadBackend(_PerTokenHeadBackend):
    """KV cache backend for ``KVQuantMode.INT8_PER_TOKEN_HEAD``."""

    mode = KVQuantMode.INT8_PER_TOKEN_HEAD
    _quant_max = _INT8_QUANT_MAX
    _quant_min = _INT8_QUANT_MIN


class Fp8PerTokenHeadBackend(_PerTokenHeadBackend):
    """KV cache backend for ``KVQuantMode.FP8_PER_TOKEN_HEAD``."""

    mode = KVQuantMode.FP8_PER_TOKEN_HEAD
    _quant_max = _FP8_QUANT_MAX
    _quant_min = _FP8_QUANT_MIN


register(Int8PerTokenHeadBackend())
register(Fp8PerTokenHeadBackend())
