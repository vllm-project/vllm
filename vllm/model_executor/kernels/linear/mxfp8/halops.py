# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 W8A8 linear kernel backed by HALOps' ``dense_gemm``."""

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    swizzle_mxfp8_scale,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

logger = init_logger(__name__)

# dense_gemm reads K in whole 128-element blocks and its SfAtom scale layout is
# 128-row aligned on both operands.
HALOPS_ALIGN = 128
HALOPS_MODE = "w8a8-mxfp8"


def _import_halops():
    try:
        from halops import dense_gemm, group_wise_quant
    except ImportError as exc:
        raise ImportError(
            "HALOps GEMM backend requested, but the `halops` package is not "
            "installed. Install it with "
            '`uv pip install "vllm[halops]" -f '
            "https://www.paddlepaddle.org.cn/packages/stable/cu132/halops/`; "
            "the find-links URL must match the CUDA build of torch."
        ) from exc
    return dense_gemm, group_wise_quant


def _halops_mxfp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    n, k = weight.shape
    out_dtype = x.dtype
    x2d = x.reshape(-1, k)
    if x2d.dtype not in (torch.bfloat16, torch.float16):
        x2d = x2d.to(torch.bfloat16)
    m = x2d.shape[0]
    if m == 0:
        return x.new_empty((*x.shape[:-1], n))
    x2d = x2d.contiguous()

    dense_gemm, group_wise_quant = _import_halops()
    q_input, sfa = group_wise_quant(x2d, mode=HALOPS_MODE, use_swizzle=True)

    output = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    dense_gemm(
        input=q_input,
        weight=weight,
        sfa=sfa,
        sfb=weight_scale,
        output=output,
        bias=None if bias is None else bias.to(torch.bfloat16),
        alpha=None,
        sm_count=-1,
        mode=HALOPS_MODE,
    )
    return output.to(out_dtype).view(*x.shape[:-1], n)


def _halops_mxfp8_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=x.dtype, device=x.device)


direct_register_custom_op(
    op_name="halops_mxfp8_linear",
    op_func=_halops_mxfp8_linear,
    fake_impl=_halops_mxfp8_linear_fake,
)


class HALOpsMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM via HALOps ``dense_gemm`` (SM100)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "HALOps dense_gemm requires a CUDA device."
        if not current_platform.is_device_capability_family(100):
            return False, "HALOps dense_gemm requires Blackwell (SM100)."
        try:
            _import_halops()
        except ImportError as exc:
            return False, str(exc)
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        if c.bmm_batch_size is not None:
            return False, "HALOps dense_gemm does not implement grouped BMM."
        n, k = c.weight_shape
        if k % HALOPS_ALIGN != 0:
            return (
                False,
                f"HALOps requires K divisible by {HALOPS_ALIGN}, got {k}.",
            )
        if n % HALOPS_ALIGN != 0:
            return (
                False,
                f"HALOps requires N divisible by {HALOPS_ALIGN}, got {n}.",
            )
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K]
        n, k = weight.shape
        scale = layer.weight_scale.data
        assert weight.dtype == torch.float8_e4m3fn and weight.ndim == 2, (
            f"HALOps MXFP8 expects an fp8 2D weight, got "
            f"{weight.dtype} {tuple(weight.shape)}."
        )
        expected = (n, k // MXFP8_BLOCK_SIZE)
        assert tuple(scale.shape) == expected, (
            f"HALOps MXFP8 expects per-32 weight scales {expected}, got "
            f"{tuple(scale.shape)}."
        )
        # dense_gemm consumes the uint8 SfAtom layout that group_wise_quant
        # emits for weights; vLLM's swizzle is bit-identical to it.
        n_padded = (n + HALOPS_ALIGN - 1) // HALOPS_ALIGN * HALOPS_ALIGN
        sfb = swizzle_mxfp8_scale(scale, M=n, K=k).view(n_padded, k // MXFP8_BLOCK_SIZE)
        replace_parameter(
            layer, "weight", Parameter(weight.contiguous(), requires_grad=False)
        )
        replace_parameter(layer, "weight_scale", Parameter(sfb, requires_grad=False))
        layer.weight_block_size = [1, MXFP8_BLOCK_SIZE]

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.ops.vllm.halops_mxfp8_linear(
            x, layer.weight, layer.weight_scale, bias
        )
