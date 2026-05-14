# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform

_flashinfer_ar_fp8_op = None
_flashinfer_ar_fp8_pattern = None


def _try_get_flashinfer_ar_fp8():
    """Return (op, pattern_code) for the FlashInfer fused AR+RMS+FP8 op, or
    (None, None) if FlashInfer isn't loaded or the op isn't registered."""
    global _flashinfer_ar_fp8_op, _flashinfer_ar_fp8_pattern
    if _flashinfer_ar_fp8_op is False:
        return None, None
    if _flashinfer_ar_fp8_op is not None:
        return _flashinfer_ar_fp8_op, _flashinfer_ar_fp8_pattern
    try:
        from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
            ar_fusion_patterns,
            flashinfer_trtllm_fused_allreduce_norm,
        )

        _flashinfer_ar_fp8_op = flashinfer_trtllm_fused_allreduce_norm
        _flashinfer_ar_fp8_pattern = ar_fusion_patterns.kARResidualRMSNormFP8Quant
    except (ImportError, AttributeError):
        _flashinfer_ar_fp8_op = False
        return None, None
    return _flashinfer_ar_fp8_op, _flashinfer_ar_fp8_pattern


@dataclass
class QuantizedActivation:
    data: torch.Tensor
    scale: torch.Tensor
    orig_dtype: torch.dtype
    orig_shape: torch.Size
    quant_key: QuantKey


def rms_norm_input_quant(
    norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None,
    linear: torch.nn.Module | None = None,
    *,
    prev_linear: torch.nn.Module | None = None,
) -> tuple[torch.Tensor | QuantizedActivation, torch.Tensor]:
    needs_ar = (
        prev_linear is not None
        and getattr(prev_linear, "tp_size", 1) > 1
        and not getattr(prev_linear, "reduce_results", True)
    )

    quant_key = getattr(linear, "input_quant_key", None)
    if quant_key == kFp8StaticTensorSym:
        return _rms_norm_fp8_static_per_tensor(norm, x, residual, linear, needs_ar)

    if needs_ar:
        x = tensor_model_parallel_all_reduce(x)
    if quant_key is None:
        if residual is None:
            return norm(x), x
        out, residual = norm(x, residual)
        return out, residual
    raise NotImplementedError(f"rms_norm + {quant_key} not wired")


def _rms_norm_fp8_static_per_tensor(
    norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None,
    linear: torch.nn.Module,
    needs_ar: bool,
) -> tuple[QuantizedActivation, torch.Tensor]:
    out_q = torch.empty(x.shape, dtype=current_platform.fp8_dtype(), device=x.device)

    if needs_ar and residual is not None:
        ar_op, pattern_code = _try_get_flashinfer_ar_fp8()
        max_token_num = getattr(linear, "fused_ar_max_token_num", 0)
        if ar_op is not None and x.ndim == 2 and max_token_num > 0:
            ar_op(
                allreduce_in=x,
                residual=residual,
                norm_out=None,
                quant_out=out_q,
                scale_out=None,
                rms_gamma=norm.weight.data,
                rms_eps=norm.variance_epsilon,
                pattern_code=pattern_code,
                scale_factor=linear.input_scale,
                world_size=get_tensor_model_parallel_world_size(),
                launch_with_pdl=True,
                fp32_acc=True,
                max_token_num=max_token_num,
            )
            return _to_qa(out_q, x, linear), residual

        x = tensor_model_parallel_all_reduce(x)

    if residual is None:
        torch.ops._C.rms_norm_static_fp8_quant(
            out_q,
            x,
            norm.weight.data,
            linear.input_scale,
            norm.variance_epsilon,
        )
        residual = x
    else:
        torch.ops._C.fused_add_rms_norm_static_fp8_quant(
            out_q,
            x,
            residual,
            norm.weight.data,
            linear.input_scale,
            norm.variance_epsilon,
        )
    return _to_qa(out_q, x, linear), residual


def _to_qa(
    out_q: torch.Tensor, x: torch.Tensor, linear: torch.nn.Module
) -> QuantizedActivation:
    return QuantizedActivation(
        data=out_q,
        scale=linear.input_scale,
        orig_dtype=x.dtype,
        orig_shape=x.shape,
        quant_key=kFp8StaticTensorSym,
    )
