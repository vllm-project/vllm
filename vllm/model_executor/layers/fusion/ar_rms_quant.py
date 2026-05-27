# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from collections.abc import Callable

import torch

from vllm.config import PassConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform

# Defined in allreduce_rms_fusion only when flashinfer_comm is importable.
# On non-CUDA or CUDA without FlashInfer, leave them None and let the call
# sites fall back to a tensor_model_parallel_all_reduce path.
ar_fusion_patterns = None
flashinfer_trtllm_fused_allreduce_norm = None
if current_platform.is_cuda():
    with contextlib.suppress(ImportError):
        from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
            ar_fusion_patterns,
            flashinfer_trtllm_fused_allreduce_norm,
        )

# Workspace size table -> per-world-size MB cap for flashinfer fused AR.
# Resolved at import time (the lookup pulls in lazy imports that dynamo can't
# trace) so the per-call sizing math stays a pure int op dynamo can handle.
_FI_AR_MAX_SIZE_MB: dict[int, float] = (
    PassConfig.default_fi_allreduce_fusion_max_size_mb()
    if flashinfer_trtllm_fused_allreduce_norm is not None
    else {}
)


def _max_token_num_for(x: torch.Tensor, world_size: int) -> int:
    """Workspace-derived upper bound on tokens for the flashinfer fused AR
    kernel; mirrors the math in ``FlashInferAllReduce._ensure_workspace``.
    Returns 0 when flashinfer fused AR isn't supported for this world size.
    """
    max_size_mb = _FI_AR_MAX_SIZE_MB.get(world_size)
    if not max_size_mb:
        return 0
    return int(max_size_mb * 1024 * 1024) // (x.shape[-1] * x.element_size())


def _allreduce_rms_norm_fp8_static_tensor(
    norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None,
    linear: torch.nn.Module,
    needs_ar: bool,
    max_token_num: int,
) -> tuple[QuantizedActivation, torch.Tensor]:
    out_q = torch.empty(x.shape, dtype=current_platform.fp8_dtype(), device=x.device)

    if (
        flashinfer_trtllm_fused_allreduce_norm is not None
        and needs_ar
        and residual is not None
        and x.ndim == 2
        and max_token_num > 0
    ):
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


def _allreduce_rms_norm_nvfp4(
    norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None,
    linear: torch.nn.Module,
    needs_ar: bool,
    max_token_num: int,
) -> tuple[QuantizedActivation, torch.Tensor]:
    from vllm._custom_ops import create_fp4_output_tensors, scaled_fp4_quant

    if (
        flashinfer_trtllm_fused_allreduce_norm is not None
        and needs_ar
        and residual is not None
        and x.ndim == 2
        and max_token_num > 0
    ):
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


def _allreduce_rms_norm(
    norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None,
    consumer: torch.nn.Module | None,
    needs_ar: bool,
    max_token_num: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        flashinfer_trtllm_fused_allreduce_norm is not None
        and needs_ar
        and x.ndim == 2
        and max_token_num > 0
    ):
        if residual is not None:
            # AR + add + RMSNorm: x is mutated to the norm result.
            flashinfer_trtllm_fused_allreduce_norm(
                allreduce_in=x,
                residual=residual,
                norm_out=None,
                quant_out=None,
                scale_out=None,
                rms_gamma=norm.weight.data,
                rms_eps=norm.variance_epsilon,
                pattern_code=ar_fusion_patterns.kARResidualRMSNorm,
                scale_factor=None,
                world_size=get_tensor_model_parallel_world_size(),
                launch_with_pdl=True,
                fp32_acc=True,
                max_token_num=max_token_num,
            )
            return x, residual

        # AR + RMSNorm (no add). Provide a zero residual buffer + dedicated
        # norm_out; the AR'd input becomes the downstream residual.
        zero_residual = torch.zeros_like(x)
        norm_out = torch.empty_like(x)
        flashinfer_trtllm_fused_allreduce_norm(
            allreduce_in=x,
            residual=zero_residual,
            norm_out=norm_out,
            quant_out=None,
            scale_out=None,
            rms_gamma=norm.weight.data,
            rms_eps=norm.variance_epsilon,
            pattern_code=ar_fusion_patterns.kARResidualRMSNorm,
            scale_factor=None,
            world_size=get_tensor_model_parallel_world_size(),
            launch_with_pdl=True,
            fp32_acc=True,
            max_token_num=max_token_num,
        )
        return norm_out, x

    if needs_ar:
        x = tensor_model_parallel_all_reduce(x)
    if current_platform.is_cuda():
        # CUDA: pin to the C++ fused kernel so deployment doesn't honor
        # IrOpPriorityConfig and silently fall to inductor codegen.
        if residual is None:
            out = torch.empty_like(x)
            torch.ops._C.rms_norm(out, x, norm.weight.data, norm.variance_epsilon)
            return out, x
        torch.ops._C.fused_add_rms_norm(
            x, residual, norm.weight.data, norm.variance_epsilon
        )
        return x, residual
    # Other platforms (ROCm/CPU/XPU/TPU): defer to RMSNorm.forward, which
    # routes to the platform's registered impl (aiter, native, etc.).
    if residual is None:
        return norm(x), x
    out, residual = norm(x, residual)
    return out, residual


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


# Dispatch table: linear activation quant key to implementation.
QUANT_IMPLS: dict[QuantKey, Callable] = {
    kFp8StaticTensorSym: _allreduce_rms_norm_fp8_static_tensor,
    kNvfp4Dynamic: _allreduce_rms_norm_nvfp4,
}


def fused_ar_rms_norm_quant(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    norm: torch.nn.Module,
    consumer: torch.nn.Module | None,
    *,
    do_allreduce: bool,
) -> tuple[torch.Tensor | QuantizedActivation, torch.Tensor]:
    """Fused (optional AR +) residual-add + RMSNorm (+ input-quant for consumer).

    ``consumer`` is the downstream linear whose ``input_quant_key`` selects the
    fused quant impl; pass None at sites with no downstream Linear (final norm)
    or when the consumer's input is unquantized. Caller asserts whether the
    input is in per-rank partial state via ``do_allreduce``.
    """
    quant_key: QuantKey | None = (
        getattr(consumer, "input_quant_key", None) if consumer is not None else None
    )
    impl = (
        QUANT_IMPLS.get(quant_key, _allreduce_rms_norm)
        if quant_key is not None
        else _allreduce_rms_norm
    )
    max_token_num = (
        _max_token_num_for(hidden_states, get_tensor_model_parallel_world_size())
        if do_allreduce
        else 0
    )
    return impl(norm, hidden_states, residual, consumer, do_allreduce, max_token_num)
