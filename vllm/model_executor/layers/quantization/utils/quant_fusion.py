# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform

if current_platform.is_cuda():
    from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
        ar_fusion_patterns,
        flashinfer_trtllm_fused_allreduce_norm,
    )


@dataclass
class QuantizedActivation:
    data: torch.Tensor
    scale: torch.Tensor
    orig_dtype: torch.dtype
    orig_shape: torch.Size
    quant_key: QuantKey


def _fp8_static_per_tensor(
    norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None,
    linear: torch.nn.Module,
    needs_ar: bool,
    max_token_num: int,
) -> tuple[QuantizedActivation, torch.Tensor]:
    out_q = torch.empty(x.shape, dtype=current_platform.fp8_dtype(), device=x.device)

    if needs_ar and residual is not None and x.ndim == 2:
        flashinfer_trtllm_fused_allreduce_norm(
            allreduce_in=x,
            residual=residual,
            norm_out=None,
            quant_out=out_q,
            scale_out=None,
            rms_gamma=norm.weight.data,
            rms_eps=norm.variance_epsilon,
            pattern_code=ar_fusion_patterns.kARResidualRMSNormFP8Quant,
            scale_factor=linear.input_scale,
            world_size=get_tensor_model_parallel_world_size(),
            launch_with_pdl=True,
            fp32_acc=True,
            max_token_num=max_token_num,
        )
        return _to_fp8_qa(out_q, x, linear), residual

    if needs_ar:
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
    return _to_fp8_qa(out_q, x, linear), residual


def _nvfp4(
    norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None,
    linear: torch.nn.Module,
    needs_ar: bool,
    max_token_num: int,
) -> tuple[QuantizedActivation, torch.Tensor]:
    from vllm._custom_ops import create_fp4_output_tensors, scaled_fp4_quant

    if needs_ar and residual is not None and x.ndim == 2:
        m, n = x.shape
        out_q, scale_out = create_fp4_output_tensors(
            m, n, x.device, is_sf_swizzled_layout=True
        )
        flashinfer_trtllm_fused_allreduce_norm(
            allreduce_in=x,
            residual=residual,
            norm_out=None,
            quant_out=out_q,
            scale_out=scale_out,
            rms_gamma=norm.weight.data,
            rms_eps=norm.variance_epsilon,
            pattern_code=ar_fusion_patterns.kARResidualRMSNormFP4Quant,
            scale_factor=linear.input_global_scale_inv,
            world_size=get_tensor_model_parallel_world_size(),
            launch_with_pdl=True,
            fp32_acc=True,
            max_token_num=max_token_num,
        )
        return _to_nvfp4_qa(out_q, scale_out, x), residual

    if needs_ar:
        x = tensor_model_parallel_all_reduce(x)

    if residual is None:
        out = norm(x)
        residual = x
    else:
        out, residual = norm(x, residual)

    out_q, scale_out = scaled_fp4_quant(
        out, linear.input_global_scale_inv, is_sf_swizzled_layout=True
    )
    return _to_nvfp4_qa(out_q, scale_out, x), residual


def _to_fp8_qa(
    out_q: torch.Tensor, x: torch.Tensor, linear: torch.nn.Module
) -> QuantizedActivation:
    return QuantizedActivation(
        data=out_q,
        scale=linear.input_scale,
        orig_dtype=x.dtype,
        orig_shape=x.shape,
        quant_key=kFp8StaticTensorSym,
    )


def _to_nvfp4_qa(
    out_q: torch.Tensor, scale_out: torch.Tensor, x: torch.Tensor
) -> QuantizedActivation:
    return QuantizedActivation(
        data=out_q,
        scale=scale_out.view(torch.float8_e4m3fn),
        orig_dtype=x.dtype,
        orig_shape=x.shape,
        quant_key=kNvfp4Dynamic,
    )


# Dispatch table: linear activation quant key → impl.
QUANT_IMPLS: dict[QuantKey, Callable] = {
    kFp8StaticTensorSym: _fp8_static_per_tensor,
    kNvfp4Dynamic: _nvfp4,
}


class FusedAllReduceRMSQuant(torch.nn.Module):
    """Fused (AllReduce +) RMSNorm (+ input-quant for the consumer linear).

    Decoder layers instantiate one of these per (norm, consumer-linear) pair.
    Dispatch is resolved once at construction: we read the consumer's
    ``input_quant_key`` to pick the impl from ``QUANT_IMPLS``, and decide
    whether the upstream all-reduce is our responsibility by inspecting the
    producing ``prev_linear`` (RowParallelLinear with ``reduce_results=False``
    leaves the tensor as per-rank partials).

    The impl returns a ``QuantizedActivation`` carrying the kernel-ready FP8
    or NVFP4 buffers; the consumer linear unpacks that on the fast path.
    """

    def __init__(
        self,
        norm: torch.nn.Module,
        linear: torch.nn.Module,
        prev_linear: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.linear = linear
        self.needs_ar = (
            prev_linear is not None
            and getattr(prev_linear, "tp_size", 1) > 1
            and not getattr(prev_linear, "reduce_results", True)
        )
        quant_key = getattr(linear, "input_quant_key", None)
        self.impl = QUANT_IMPLS.get(quant_key) if quant_key is not None else None
        self.max_token_num = (
            get_current_vllm_config().scheduler_config.max_num_batched_tokens
            if self.impl is not None and self.needs_ar
            else 0
        )

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor | QuantizedActivation, torch.Tensor]:
        if self.impl is not None:
            return self.impl(
                self.norm, x, residual, self.linear, self.needs_ar, self.max_token_num
            )

        if self.needs_ar:
            x = tensor_model_parallel_all_reduce(x)
        if residual is None:
            return self.norm(x), x
        out, residual = self.norm(x, residual)
        return out, residual
