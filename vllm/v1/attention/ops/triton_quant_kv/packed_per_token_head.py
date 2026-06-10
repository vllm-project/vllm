# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sub-byte per-token-head KV cache quantization (INT4).

INT4 is the one mode whose attention read path is structurally different
from the core kernel (split-dot + sub-byte unpack), so it gets its own
reshape (write) and attention (read) entry points here.  Everything else
(``NONE``, ``FP8_PER_TENSOR``, ``INT8`` / ``FP8`` per-token-head) is
handled by the core kernel via constexpr branches.

INT4 uses a per-(token, head) dynamic scale + a single RHT pre-rotation
on the inputs and inverse RHT on the output:

+----------+------------+---------------------+----------------------+
| Mode     | Packing    | Pre-rotation        | Scale encodes        |
+==========+============+=====================+======================+
| INT4     | 2 / byte   | Single RHT          | ``scale`` + 4-bit zp |
|          |            | (random Hadamard)   | (stego in mantissa)  |
+----------+------------+---------------------+----------------------+

The actual Triton kernels live in the two sibling private modules
(:mod:`._packed_attention` and :mod:`._packed_reshape`); this module just
wraps them with the RHT pre/post-rotation.
"""

from __future__ import annotations

import torch

from vllm.v1.attention.ops.triton_quant_kv._hadamard import single_rht
from vllm.v1.attention.ops.triton_quant_kv._packed_attention import _launch_packed_attn
from vllm.v1.attention.ops.triton_quant_kv._packed_reshape import (
    _reshape_cache_int4_kernel,
    _run_reshape_kernel,
)

# 2 × int4 packed per storage byte.
_INT4_PACKING_FACTOR = 2


def reshape_and_cache_int4(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
) -> None:
    """Pre-rotate (RHT), pack to INT4 and write into the paged cache."""
    key = single_rht(key.float()).to(key.dtype)
    value = single_rht(value.float()).to(value.dtype)
    _run_reshape_kernel(
        _reshape_cache_int4_kernel,
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
        slot_mapping=slot_mapping,
        packing_factor=_INT4_PACKING_FACTOR,
    )


def unified_attention_int4(
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
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    seq_threshold_3D: int | None = None,
    num_par_softmax_segments: int | None = None,
    softmax_segm_output: torch.Tensor | None = None,
    softmax_segm_max: torch.Tensor | None = None,
    softmax_segm_expsum: torch.Tensor | None = None,
) -> None:
    """Paged attention over the INT4 packed cache, writing into *out*.

    The forward RHT has norm ``sqrt(head_size)``, so ``softmax_scale`` is
    divided by ``head_size`` and the inverse RHT divides the output by
    ``head_size`` as well.
    """
    q_orig_dtype = q.dtype
    q = single_rht(q.float()).to(q_orig_dtype)
    head_size = q.shape[2]
    softmax_scale = softmax_scale / head_size

    _launch_packed_attn(
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
        packing_factor=_INT4_PACKING_FACTOR,
    )

    out_f = single_rht(out.float(), inverse=True) / head_size
    out.copy_(out_f.to(q_orig_dtype))
