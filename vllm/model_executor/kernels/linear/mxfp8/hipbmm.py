# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 linear via AITER hipBLASLt on gfx950."""

import torch

import vllm.envs as envs
from vllm._aiter_ops import _ensure_hipb_mm_extension_initialized, rocm_aiter_ops
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig
from .rocm_native import _as_block32_scale


def _quantize_mxfp8_gfx950(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter import dynamic_per_group_scaled_quant

    rows, columns = input.shape
    assert input.dtype in {
        torch.float16,
        torch.bfloat16,
    }, f"input dtype must be float16 or bfloat16, got {input.dtype}"
    assert columns % MXFP8_BLOCK_SIZE == 0
    padded_rows = (rows + 31) // 32 * 32
    scale_columns = (columns // MXFP8_BLOCK_SIZE + 7) // 8 * 8
    output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    scale = torch.empty(
        (padded_rows, scale_columns), dtype=MXFP8_SCALE_DTYPE, device=input.device
    )
    dynamic_per_group_scaled_quant(
        output, input, scale, group_size=MXFP8_BLOCK_SIZE, shuffle_scale=True
    )
    return output, scale


class RocmHipbmmMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 linear via AITER hipBLASLt for every input row count.

    Quantize FP16/BF16 activations with AITER, which writes shuffled scales
    directly. Keep native-compatible weight scales and a separate padded,
    swizzled copy for hipBLASLt.
    """

    supports_pre_processed_weights = True

    def __init__(self, c: Mxfp8LinearLayerConfig) -> None:
        super().__init__(c)
        # Create the handle/workspace before compiled forward or graph capture.
        # Cached weights skip process_weights_after_loading.
        _ensure_hipb_mm_extension_initialized()

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm() or not current_platform.supports_mx():
            return False, "requires ROCm with native MX support"
        if not envs.VLLM_ROCM_USE_AITER_LINEAR_HIPBMM:
            return False, "VLLM_ROCM_USE_AITER_LINEAR_HIPBMM is disabled"
        if envs.VLLM_BATCH_INVARIANT:
            return False, "hipBLASLt heuristic selection is not batch invariant"
        if not rocm_aiter_ops.is_linear_enabled():
            return False, "AITER linear is disabled or not supported"
        return True, None

    @classmethod
    def can_implement(cls, config: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        if config.bmm_batch_size is not None:
            return False, "batched MXFP8 GEMM is not supported"
        if config.weight_shape[1] % MXFP8_BLOCK_SIZE != 0:
            return False, (
                f"AITER MXFP8 quantization requires K divisible by {MXFP8_BLOCK_SIZE}"
            )
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        N, K = weight.shape
        assert K % MXFP8_BLOCK_SIZE == 0, (
            f"weight columns must be divisible by {MXFP8_BLOCK_SIZE}, got {K}"
        )
        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale = layer.weight_scale.data
        if weight_scale.shape != (N, scale_k):
            raise ValueError(
                f"Expected weight_scale shape {(N, scale_k)}, got {weight_scale.shape}."
            )
        if weight_scale.dtype != MXFP8_SCALE_DTYPE:
            raise ValueError(
                f"Expected {MXFP8_SCALE_DTYPE} weight_scale, got {weight_scale.dtype}."
            )
        weight_scale = weight_scale.contiguous()
        # Keep two layouts because some consumers bypass apply_weights:
        # DeepSeek's wo_a dequantization and monokernel read weight_scale
        # directly in logical order, compacted to 32x32 blocks when possible.
        # hipBLASLt instead needs padded, shuffled per-row 1x32 scales and
        # cannot currently read compact 32x32 scales. Build weight_scale_hipbmm
        # from the expanded scales before compacting the original parameter.
        replace_parameter(
            layer, "weight_scale_hipbmm", rocm_aiter_ops.shuffle_scale(weight_scale)
        )
        # Reuse native's repeated-row compaction for direct scale consumers.
        block_scale = _as_block32_scale(weight_scale)
        replace_parameter(
            layer, "weight_scale", weight_scale if block_scale is None else block_scale
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from aiter import HipblasLtMatmulScaleMode

        out_dtype, in_shape = x.dtype, x.shape
        x2d, x_scale = _quantize_mxfp8_gfx950(x.reshape(-1, in_shape[-1]))
        mode = HipblasLtMatmulScaleMode.BLK32_UE8M0_32_8_EXT
        out = torch.ops.aiter.hipb_mm(
            x2d,
            layer.weight.mT,
            solution_index=-1,
            bias=bias,
            out_dtype=out_dtype,
            scaleA=x_scale,
            scaleB=layer.weight_scale_hipbmm,
            scaleA_mode=mode,
            scaleB_mode=mode,
        ).reshape(*in_shape[:-1], layer.weight.shape[0])
        return out
