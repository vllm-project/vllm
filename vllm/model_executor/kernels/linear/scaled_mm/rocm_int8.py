# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""W8A8 INT8 skinny GEMM for ROCm.

Uses the wvSplitK_w8a8 kernel for small batch sizes (N<=5) where
activations fit in LDS. Falls back to Triton int8xint8 scaled_mm for
larger batches (prefill) so the int8 weights stay resident without an
extra dequantized copy.

The dispatch is wrapped in a torch.library custom op so torch.compile
treats it as opaque, avoiding issues with data-dependent branches
on symbolic batch sizes.
"""

from contextlib import nullcontext

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.utils.platform_utils import num_compute_units

from .ScaledMMLinearKernel import Int8ScaledMMLinearLayerConfig
from .triton import TritonInt8ScaledMMLinearKernel

# INT8 activations in LDS: 1 byte each, minus 128 bytes reserved for
# auxiliary shared memory (dynamic quant scales, reduction scratch).
# gfx9: 64KB-128, gfx95x: 160KB-128 (kernel checks at runtime)
LDS_CAPACITY_BYTES = 64 * 1024 - 128


def _w8a8_apply_impl(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    i_s: torch.Tensor | None,
    bias: torch.Tensor | None,
    cu_count: int,
) -> torch.Tensor:
    """Dispatch between wvSplitK_w8a8 and Triton int8 scaled_mm.

    Registered as a custom op so torch.compile treats it as opaque,
    avoiding issues with the data-dependent m<=5 branch.
    """
    import vllm._custom_ops as ops

    out_dtype = x.dtype
    m = x.shape[0]
    k = w_q.shape[0]  # weights are [K, N]
    n = w_q.shape[1]

    use_wvsplitk = (
        m <= 5
        and k % 16 == 0
        and n % 16 == 0
        and k * m <= LDS_CAPACITY_BYTES
        and (i_s is None or i_s.numel() == 1)
    )

    if use_wvsplitk:
        w_t = w_q.t()  # [K, N] -> [N, K]

        if w_s.numel() == 1:
            w_scale_chan = w_s.to(out_dtype).expand(n).contiguous()
        else:
            w_scale_chan = w_s.to(out_dtype).contiguous()

        a_scale = i_s.to(torch.float32).reshape(1) if i_s is not None else None
        act = x.contiguous()

        return ops.wvSplitK_w8a8(
            w_t,
            act,
            w_scale_chan,
            a_scale,
            cu_count,
            bias,
        )

    # Fallback: explicit quantize + Triton int8 scaled_mm.
    x_q, x_s, _ = ops.scaled_int8_quant(x.contiguous(), i_s, None, symmetric=True)

    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm,
    )

    return triton_scaled_mm(
        x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=out_dtype, bias=bias
    )


def _w8a8_apply_fake(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    i_s: torch.Tensor | None,
    bias: torch.Tensor | None,
    cu_count: int,
) -> torch.Tensor:
    m = x.size(0)
    n = w_q.size(1)
    return torch.empty((m, n), dtype=x.dtype, device=x.device)


def _register_w8a8_op():
    lib = torch.library.Library("_rocm_skinny_w8a8", "DEF")
    lib.define(
        "w8a8_apply(Tensor x, Tensor w_q, Tensor w_s, "
        "Tensor? i_s, Tensor? bias, int cu_count) -> Tensor"
    )
    lib.impl("w8a8_apply", _w8a8_apply_impl, "CUDA")
    lib.impl("w8a8_apply", _w8a8_apply_fake, "Meta")
    return lib


_W8A8_LIB = _register_w8a8_op()


class ROCmInt8SkinnyGemmLinearKernel(TritonInt8ScaledMMLinearKernel):
    """W8A8 per-channel int8 skinny GEMM for ROCm.

    Uses the wvSplitK_w8a8 kernel for small batch sizes where both
    int8 activations and weights fit the LDS constraint. Falls back
    to Triton int8 scaled_mm for larger batches (prefill).

    Dispatch is wrapped in a torch.library custom op so torch.compile
    sees a single opaque call, avoiding symbolic shape issues with
    the m<=5 branch condition.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "requires ROCm."

        if not envs.VLLM_ROCM_USE_SKINNY_GEMM:
            return False, "requires VLLM_ROCM_USE_SKINNY_GEMM to be enabled."

        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if not c.input_symmetric:
            return False, "supports symmetric quantization only."
        return True, None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, _ = self._get_layer_params(layer)

        # i_zp is checked at can_implement time (symmetric only), but guard
        # here so the fallback path still works if somehow reached.
        if i_zp is not None:
            return super().apply_weights(layer, x, bias)

        m = x.shape[0]
        k = w_q.shape[0]
        n = w_q.shape[1]

        ctx = (
            nullcontext()
            if torch.compiler.is_compiling()
            else torch.profiler.record_function(f"w8a8_apply {m}x{n}x{k}")
        )
        with ctx:
            return torch.ops._rocm_skinny_w8a8.w8a8_apply(
                x,
                w_q,
                w_s,
                i_s,
                bias,
                num_compute_units(),
            )
