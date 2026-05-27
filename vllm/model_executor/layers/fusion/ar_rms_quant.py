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
# Stay None on non-CUDA or CUDA-without-FlashInfer; the helper falls through
# to a plain tensor_model_parallel_all_reduce path.
ar_fusion_patterns = None
flashinfer_trtllm_fused_allreduce_norm = None
if current_platform.is_cuda():
    with contextlib.suppress(ImportError):
        from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
            ar_fusion_patterns,
            flashinfer_trtllm_fused_allreduce_norm,
        )

# Per-world-size workspace cap (MB) for the flashinfer fused AR kernel.
# Resolved at import time -- the lookup pulls in modules dynamo can't trace --
# so per-call sizing stays a pure int op.
_FLASHINFER_AR_MAX_SIZE_MB: dict[int, float] = (
    PassConfig.default_fi_allreduce_fusion_max_size_mb()
    if flashinfer_trtllm_fused_allreduce_norm is not None
    else {}
)


def _flashinfer_max_token_num(x: torch.Tensor, world_size: int) -> int:
    """Workspace-derived upper bound for the flashinfer fused-AR kernel
    (max_workspace_bytes / row_bytes). Mirrors FlashInferAllReduce._ensure_workspace.
    Returns 0 when this world_size isn't in the support table.
    """
    max_size_mb = _FLASHINFER_AR_MAX_SIZE_MB.get(world_size)
    if not max_size_mb:
        return 0
    return int(max_size_mb * 1024 * 1024) // (x.shape[-1] * x.element_size())


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


# Single-kernel AR + add + RMSNorm + activation-quant (flashinfer).
# Preconditions guaranteed by caller: do_allreduce, residual present, 2D, CUDA,
# flashinfer importable, world_size in _FLASHINFER_AR_MAX_SIZE_MB.


def _flashinfer_ar_rms_fp8(
    norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor,
    linear: torch.nn.Module,
    max_token_num: int,
) -> tuple[QuantizedActivation, torch.Tensor]:
    assert flashinfer_trtllm_fused_allreduce_norm is not None
    out_q = torch.empty(x.shape, dtype=current_platform.fp8_dtype(), device=x.device)
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


def _flashinfer_ar_rms_nvfp4(
    norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor,
    linear: torch.nn.Module,
    max_token_num: int,
) -> tuple[QuantizedActivation, torch.Tensor]:
    assert flashinfer_trtllm_fused_allreduce_norm is not None
    from vllm._custom_ops import create_fp4_output_tensors

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


_FUSED_AR_RMS_QUANT_IMPLS: dict[QuantKey, Callable] = {
    kFp8StaticTensorSym: _flashinfer_ar_rms_fp8,
    kNvfp4Dynamic: _flashinfer_ar_rms_nvfp4,
}


# Single-kernel AR + (add +) RMSNorm, no activation-quant (flashinfer).
# Downstream linear quantises its own input.


def _flashinfer_ar_rms(
    norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None,
    max_token_num: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert flashinfer_trtllm_fused_allreduce_norm is not None
    if residual is not None:
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

    # No residual: feed a zero residual + dedicated norm_out so the kernel's
    # AR'd input becomes the downstream residual.
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


# Single-kernel (add +) RMSNorm + activation-quant (vLLM C++ op).
# AR, if any, was already done by the caller.


def _rms_norm_fp8_static(
    norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None,
    linear: torch.nn.Module,
) -> tuple[QuantizedActivation, torch.Tensor]:
    out_q = torch.empty(x.shape, dtype=current_platform.fp8_dtype(), device=x.device)
    if residual is None:
        torch.ops._C.rms_norm_static_fp8_quant(
            out_q, x, norm.weight.data, linear.input_scale, norm.variance_epsilon
        )
        return _to_fp8_qa(out_q, x, linear), x
    torch.ops._C.fused_add_rms_norm_static_fp8_quant(
        out_q, x, residual, norm.weight.data, linear.input_scale, norm.variance_epsilon
    )
    return _to_fp8_qa(out_q, x, linear), residual


_FUSED_RMS_QUANT_IMPLS: dict[QuantKey, Callable] = {
    kFp8StaticTensorSym: _rms_norm_fp8_static,
}


# Plain RMSNorm (no AR, no quant). AR, if any, was done by the caller.


def _rms_norm(
    norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if current_platform.is_cuda():
        # Pin to the C++ kernel so deployment doesn't honour IrOpPriorityConfig
        # and silently codegen via inductor.
        if residual is None:
            out = torch.empty_like(x)
            torch.ops._C.rms_norm(out, x, norm.weight.data, norm.variance_epsilon)
            return out, x
        torch.ops._C.fused_add_rms_norm(
            x, residual, norm.weight.data, norm.variance_epsilon
        )
        return x, residual
    # ROCm / CPU / XPU / TPU: defer to RMSNorm.forward, which dispatches to
    # the platform's registered impl (aiter, native, etc.).
    if residual is None:
        return norm(x), x
    out, residual = norm(x, residual)
    return out, residual


def fused_ar_rms_norm_quant(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    norm: torch.nn.Module,
    consumer_linear: torch.nn.Module | None,
    *,
    do_allreduce: bool,
) -> tuple[torch.Tensor | QuantizedActivation, torch.Tensor]:
    """(Optional AR +) residual-add + RMSNorm, with the consumer linear's
    activation-quant folded in when a fused kernel makes that worthwhile.

    Returns a ``QuantizedActivation`` only when a fused kernel actually
    produced the pre-quantised output (paths A and C below). Otherwise
    returns a plain tensor and the downstream linear quantises its own
    input via the regular compile-time RMS+quant fusion pass.
    """
    world_size = get_tensor_model_parallel_world_size()
    quant_key: QuantKey | None = (
        getattr(consumer_linear, "input_quant_key", None)
        if consumer_linear is not None
        else None
    )
    has_residual = residual is not None

    # Can flashinfer fuse this AR with its post-AR RMSNorm? Requires CUDA + AR
    # happening + 2D input + flashinfer present + world_size in the workspace
    # table.
    can_fuse_ar = (
        do_allreduce
        and hidden_states.ndim == 2
        and flashinfer_trtllm_fused_allreduce_norm is not None
    )
    max_token_num = (
        _flashinfer_max_token_num(hidden_states, world_size) if can_fuse_ar else 0
    )
    can_fuse_ar = can_fuse_ar and max_token_num > 0

    fused_ar_rms_quant_impl = (
        _FUSED_AR_RMS_QUANT_IMPLS.get(quant_key)
        if can_fuse_ar and quant_key is not None
        else None
    )
    fused_rms_quant_impl = (
        _FUSED_RMS_QUANT_IMPLS.get(quant_key) if quant_key is not None else None
    )

    # A: single kernel for AR + add + RMSNorm + activation-quant.
    if fused_ar_rms_quant_impl is not None and has_residual:
        return fused_ar_rms_quant_impl(
            norm, hidden_states, residual, consumer_linear, max_token_num
        )

    # B: single kernel for AR + (add +) RMSNorm. Downstream linear quantises.
    if can_fuse_ar:
        return _flashinfer_ar_rms(norm, hidden_states, residual, max_token_num)

    # C: explicit AR (if any) + fused RMSNorm + activation-quant.
    if fused_rms_quant_impl is not None:
        if do_allreduce:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        return fused_rms_quant_impl(norm, hidden_states, residual, consumer_linear)

    # D: explicit AR (if any) + plain RMSNorm. Downstream linear quantises.
    if do_allreduce:
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)
    return _rms_norm(norm, hidden_states, residual)
