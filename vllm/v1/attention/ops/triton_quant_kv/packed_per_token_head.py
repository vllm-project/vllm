# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sub-byte per-token-head KV cache quantization factory (INT4).

Per-(token, head) dynamic scale + a single RHT (random Hadamard)
pre-rotation on the inputs and its inverse on the output; two values are
packed per byte and the scale carries a 4-bit zero-point steganographed in
its mantissa.

The attention read kernel and reshape write kernel live in the two
sibling private modules (:mod:`._packed_attention` and
:mod:`._packed_reshape`).  This module only wires them into a
:class:`QuantKVFactory` pair and registers them.
"""

from __future__ import annotations

import torch

from vllm.v1.attention.ops.triton_quant_kv import register
from vllm.v1.attention.ops.triton_quant_kv._hadamard import (
    _get_rht_signs,
)
from vllm.v1.attention.ops.triton_quant_kv._packed_attention import _launch_packed_attn
from vllm.v1.attention.ops.triton_quant_kv._packed_reshape import (
    _reshape_cache_int4_kernel,
    _run_reshape_kernel,
)
from vllm.v1.attention.ops.triton_quant_kv.base import QuantKVFactory
from vllm.v1.kv_cache_interface import KVQuantMode


class _PackedFactory(QuantKVFactory):
    """Base factory for the sub-byte packed per-token-head mode.

    The subclass declares the mode-specific pieces as class attributes /
    classmethods; the ``reshape_and_cache`` / ``unified_attention``
    bodies live here.

    Mode-specific hooks (must be set/overridden by the subclass)
    -----------------------------------------------------------
    ``_reshape_kernel``
        The ``@triton.jit`` reshape kernel for this mode.
    ``_rotate_kv(x)``
        Pre-rotation applied to K / V before packing (RHT).
    ``_rotate_q(q)``
        Pre-rotation applied to Q before attention.  Typically the same
        rotation as ``_rotate_kv`` so the dot product is preserved.
    ``_unrotate_out(out, head_size)``
        Inverse rotation on the kernel output, written back in-place.
    ``_transform_softmax_scale(scale, head_size)``
        Optional rescaling of ``softmax_scale`` before the kernel (INT4
        divides by ``head_size`` to absorb the RHT scale).
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
        key = self._rotate_kv(key)
        value = self._rotate_kv(value)
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
        q = self._rotate_q(q)
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

    # Cached HD1^T matrix for forward RHT and HD1 for inverse, both as
    # contiguous [D, D] tensors. Using ``x @ hd1t`` directly (no reshape)
    # leverages PyTorch's batched matmul broadcast: [N, H, D] @ [D, D].
    # This is ~10 µs per call vs ~41 µs with reshape+matmul+reshape.
    _hd1t_cache: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
    _hd1_cache: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
    _rdna3_reshape_ready: bool | None = None

    @classmethod
    def _get_hd1t(
        cls, d: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """HD1^T for forward RHT: x_rot = x @ HD1^T."""
        key = (d, str(device), dtype)
        if key not in cls._hd1t_cache:
            from vllm.v1.attention.ops.triton_quant_kv._hadamard import (
                _get_hadamard_matrix,
                _get_rht_signs,
            )

            H = _get_hadamard_matrix(d, dtype, device)
            D1 = _get_rht_signs(d, 0, device, dtype)
            hd1 = H * D1[None, :]
            cls._hd1t_cache[key] = hd1.T.contiguous()
            cls._hd1_cache[key] = hd1.contiguous()
        return cls._hd1t_cache[key]

    @classmethod
    def _get_hd1(cls, d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """HD1 for inverse RHT: x_unrot = x @ HD1."""
        key = (d, str(device), dtype)
        if key not in cls._hd1_cache:
            cls._get_hd1t(d, device, dtype)  # populates both caches
        return cls._hd1_cache[key]

    @classmethod
    def _check_rdna3_reshape(cls) -> bool:
        if cls._rdna3_reshape_ready is None:
            from vllm.platforms import current_platform

            cls._rdna3_reshape_ready = (
                current_platform.is_rocm()
                and hasattr(torch.ops, "_C")
                and hasattr(torch.ops._C, "reshape_cache_int4_rdna3")
            )
        return cls._rdna3_reshape_ready

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
        """Fused RHT + INT4 quantize via HIP kernel on RDNA3."""
        assert k_scale_cache is not None and v_scale_cache is not None
        if (
            self._check_rdna3_reshape()
            and key.dtype == torch.float16
            and key.shape[2] == 128
        ):
            rht_signs = _get_rht_signs(key.shape[2], 0, key.device, torch.float32)
            torch.ops._C.reshape_cache_int4_rdna3(
                key,
                value,
                key_cache,
                value_cache,
                k_scale_cache,
                v_scale_cache,
                rht_signs,
                slot_mapping,
            )
            return
        # Fallback: matmul RHT + Triton quantize
        super().reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
        )

    def _rotate_kv(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self._get_hd1t(x.shape[-1], x.device, x.dtype)

    def _rotate_q(self, q: torch.Tensor) -> torch.Tensor:
        return q @ self._get_hd1t(q.shape[-1], q.device, q.dtype)

    def _unrotate_out(self, out: torch.Tensor, head_size: int) -> torch.Tensor:
        return (out @ self._get_hd1(head_size, out.device, out.dtype)) / head_size

    @staticmethod
    def _transform_softmax_scale(scale: float, head_size: int) -> float:
        return scale / head_size


register(Int4PerTokenHeadFactory())
