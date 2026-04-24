# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sub-byte per-token-head KV cache quantization factories (INT4 + INT2).

Both modes share the same skeleton — per-(token, head) dynamic scale +
Hadamard pre-rotation on the inputs and inverse Hadamard on the output
— but differ in their quantization math and packing factor:

+----------+------------+---------------------+----------------------+
| Mode     | Packing    | Pre-rotation        | Scale encodes        |
+==========+============+=====================+======================+
| INT4     | 2 / byte   | Single RHT          | ``scale`` + 4-bit zp |
|          |            | (random Hadamard)   | (stego in mantissa)  |
+----------+------------+---------------------+----------------------+
| INT2     | 4 / byte   | Full Hadamard       | ``norm / d^1.5``     |
|          |            | (no random sign)    | (centroid lookup)    |
+----------+------------+---------------------+----------------------+

The attention read kernel and reshape write kernels live in the two
sibling private modules (:mod:`._packed_attention` and
:mod:`._packed_reshape`).  This module only wires them into a
:class:`QuantKVFactory` pair and registers them.
"""

from __future__ import annotations

import torch

from vllm.v1.attention.ops.triton_quant_kv import register
from vllm.v1.attention.ops.triton_quant_kv._hadamard import (
    fast_hadamard_transform,
    single_rht,
)
from vllm.v1.attention.ops.triton_quant_kv._packed_attention import _launch_packed_attn
from vllm.v1.attention.ops.triton_quant_kv._packed_reshape import (
    _reshape_cache_int2_kernel,
    _reshape_cache_int4_kernel,
    _run_reshape_kernel,
)
from vllm.v1.attention.ops.triton_quant_kv.base import QuantKVFactory
from vllm.v1.kv_cache_interface import KVQuantMode


class _PackedFactory(QuantKVFactory):
    """Shared factory for sub-byte packed per-token-head modes.

    Subclasses declare the mode-specific pieces as class attributes /
    classmethods; the ``reshape_and_cache`` / ``unified_attention``
    bodies are identical and live here.

    Mode-specific hooks (must be set/overridden by subclasses)
    ---------------------------------------------------------
    ``_reshape_kernel``
        The ``@triton.jit`` reshape kernel for this mode.
    ``_rotate_kv(x)``
        Pre-rotation applied to K / V before packing (RHT for INT4,
        full Hadamard for INT2).
    ``_rotate_q(q)``
        Pre-rotation applied to Q before attention.  Typically the same
        rotation as ``_rotate_kv`` so the dot product is preserved.
    ``_unrotate_out(out, head_size)``
        Inverse rotation on the kernel output, written back in-place.
    ``_transform_softmax_scale(scale, head_size)``
        Optional rescaling of ``softmax_scale`` before the kernel (INT4
        divides by ``head_size`` to absorb the RHT scale; INT2 is a
        no-op).
    """

    needs_scale_caches = True

    # Filled in by subclasses.
    _reshape_kernel: object

    @staticmethod
    def _rotate_kv(x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _rotate_q(q: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _unrotate_out(out: torch.Tensor, head_size: int) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _transform_softmax_scale(scale: float, head_size: int) -> float:
        return scale

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
        key = self._rotate_kv(key.float()).to(key.dtype)
        value = self._rotate_kv(value.float()).to(value.dtype)
        _run_reshape_kernel(
            self._reshape_kernel,
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            slot_mapping=slot_mapping,
            packing_factor=self.packing_factor,
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
        assert k_scale_cache is not None and v_scale_cache is not None

        q_orig_dtype = q.dtype
        q = self._rotate_q(q.float()).to(q_orig_dtype)
        head_size = q.shape[2]
        softmax_scale = self._transform_softmax_scale(softmax_scale, head_size)

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
            packing_factor=self.packing_factor,
        )

        out_f = self._unrotate_out(out, head_size)
        out.copy_(out_f.to(q_orig_dtype))


class Int4PerTokenHeadFactory(_PackedFactory):
    """KV cache factory for ``KVQuantMode.INT4_PER_TOKEN_HEAD``."""

    mode = KVQuantMode.INT4_PER_TOKEN_HEAD
    packing_factor = 2  # 2 × int4 per byte
    _reshape_kernel = _reshape_cache_int4_kernel

    # RHT pre-rotation gaussianizes data → better quantization.  The
    # forward RHT has norm ``sqrt(head_size)``, so ``softmax_scale`` is
    # divided by ``head_size`` and the inverse RHT divides the output
    # by ``head_size`` as well.
    @staticmethod
    def _rotate_kv(x: torch.Tensor) -> torch.Tensor:
        return single_rht(x)

    @staticmethod
    def _rotate_q(q: torch.Tensor) -> torch.Tensor:
        return single_rht(q)

    @staticmethod
    def _unrotate_out(out: torch.Tensor, head_size: int) -> torch.Tensor:
        return single_rht(out.float(), inverse=True) / head_size

    @staticmethod
    def _transform_softmax_scale(scale: float, head_size: int) -> float:
        return scale / head_size


class Int2PerTokenHeadFactory(_PackedFactory):
    """KV cache factory for ``KVQuantMode.INT2_PER_TOKEN_HEAD``."""

    mode = KVQuantMode.INT2_PER_TOKEN_HEAD
    packing_factor = 4  # 4 × int2 per byte
    _reshape_kernel = _reshape_cache_int2_kernel

    # Full Hadamard (no random sign).  Its own inverse — so the output
    # rotation is identical.  No softmax_scale adjustment: the ``d^1.5``
    # factor is absorbed into the stored scale at write time.
    @staticmethod
    def _rotate_kv(x: torch.Tensor) -> torch.Tensor:
        return fast_hadamard_transform(x)

    @staticmethod
    def _rotate_q(q: torch.Tensor) -> torch.Tensor:
        return fast_hadamard_transform(q)

    @staticmethod
    def _unrotate_out(out: torch.Tensor, head_size: int) -> torch.Tensor:
        return fast_hadamard_transform(out.float())


register(Int4PerTokenHeadFactory())
register(Int2PerTokenHeadFactory())
