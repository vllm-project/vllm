# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP all-reduce + residual add + Gemma RMSNorm + consumer input quant.

Calls FlashInfer's ``kARResidualRMSNormOut{FP8,FP4}Quant`` patterns directly
(no torch.compile custom op). The kernel writes the bf16 normed activation
(GDN's ``in_proj_ba`` reads it) next to the static FP8 / NVFP4 codes; its
outputs are bitwise identical to FlashInfer AR+norm followed by
``scaled_fp8_quant`` / ``scaled_fp4_quant``.

Returns None when the fused kernel does not apply (FlashInfer or a supported
workspace unavailable, or a batch above the workspace budget); the caller
then runs an explicit all-reduce and the Triton norm-quant.
"""

import torch

from vllm.config.compilation import PassConfig
from vllm.distributed.device_communicators import flashinfer_all_reduce as fi_ar
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)

# Gemma RMSNorm scales by (1 + weight).
_GEMMA_WEIGHT_BIAS = 1.0

# Set when the quant workspace cannot be created for this TP group (e.g. the
# trtllm backend across nodes); get_fi_ar_quant_workspace raises on every call.
_workspace_unsupported = False


def _get_workspace(hidden_states: torch.Tensor, tp_size: int):
    """The FlashInfer quant workspace if it can take ``hidden_states``, else
    None."""
    global _workspace_unsupported
    if not fi_ar.fi_ar_available or _workspace_unsupported:
        return None
    if (
        not hidden_states.is_cuda
        or hidden_states.dim() != 2
        or not hidden_states.is_contiguous()
        or hidden_states.dtype not in (torch.bfloat16, torch.float16)
    ):
        return None

    # Same budget as the non-quant AR+norm workspace, which a matching backend
    # shares with this one.
    max_size_mb = PassConfig.default_fi_allreduce_fusion_max_size_mb().get(tp_size)
    if not max_size_mb:
        return None
    num_tokens, hidden_size = hidden_states.shape
    max_token_num = int(max_size_mb * 1024 * 1024) // (
        hidden_size * hidden_states.element_size()
    )
    if num_tokens > max_token_num:
        return None

    try:
        workspace = fi_ar.get_fi_ar_quant_workspace(
            world_size=tp_size,
            rank=get_tensor_model_parallel_rank(),
            max_token_num=max_token_num,
            hidden_dim=hidden_size,
            dtype=hidden_states.dtype,
            group=get_tp_group().cpu_group,
        )
    except ValueError as e:
        logger.warning_once("FlashInfer AR+norm+quant disabled: %s", e)
        _workspace_unsupported = True
        return None
    # mnnvl rotates through three Lamport buffers, so less than the whole
    # budget is usable per call.
    if workspace is None or not workspace.is_buffer_size_sufficient(
        tp_size=tp_size,
        num_tokens=num_tokens,
        hidden_dim=hidden_size,
        dtype=hidden_states.dtype,
    ):
        return None
    return workspace


def fused_allreduce_gemma_rms_norm_quant(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: torch.Tensor,
    key: QuantKey,
) -> tuple[QuantizedActivation, torch.Tensor, torch.Tensor] | None:
    """Fused ``gemma_rms_norm(all_reduce(hidden_states) + residual)`` + quant.

    ``hidden_states`` is the per-rank partial of a row-parallel linear;
    ``scale`` is the consumer's static FP8 input scale or NVFP4 global scale
    inverse. Returns the quantized normed activation, the normed activation in
    ``hidden_states``' dtype, and the new residual (written into the
    ``hidden_states`` buffer), or None when the fused kernel does not apply.
    """
    tp_size = get_tensor_model_parallel_world_size()
    workspace = _get_workspace(hidden_states, tp_size)
    if workspace is None:
        return None

    flashinfer_comm = fi_ar.flashinfer_comm
    patterns = flashinfer_comm.AllReduceFusionPattern
    nvfp4 = key == kNvfp4Dynamic
    num_tokens, hidden_size = hidden_states.shape
    norm_out = torch.empty_like(hidden_states)
    if nvfp4:
        pattern = patterns.kARResidualRMSNormOutFP4Quant
        quant_out = torch.empty(
            (num_tokens, hidden_size // 2),
            dtype=torch.uint8,
            device=hidden_states.device,
        )
        # 128x4-swizzled block scales, as scaled_fp4_quant lays them out.
        scale_out = torch.empty(
            (round_up(num_tokens, 128), round_up(hidden_size // 16, 4) // 4),
            dtype=torch.int32,
            device=hidden_states.device,
        ).view(current_platform.fp8_dtype())
    else:
        pattern = patterns.kARResidualRMSNormOutFP8Quant
        quant_out = torch.empty_like(hidden_states, dtype=current_platform.fp8_dtype())
        scale_out = None

    # use_oneshot=None leaves the one-shot/two-shot choice to FlashInfer, and
    # the default trigger_completion_at_end=True keeps the trtllm one-shot
    # kernel from signalling PDL dependents before its output is written.
    flashinfer_comm.allreduce_fusion(
        input=hidden_states,
        workspace=workspace,
        pattern=pattern,
        launch_with_pdl=True,
        residual_in=residual,
        residual_out=hidden_states,
        norm_out=norm_out,
        quant_out=quant_out,
        scale_out=scale_out,
        rms_gamma=weight,
        rms_eps=eps,
        scale_factor=scale,
        layout_code=flashinfer_comm.QuantizationSFLayout.SWIZZLED_128x4,
        use_oneshot=None,
        fp32_acc=True,
        weight_bias=_GEMMA_WEIGHT_BIAS,
    )
    qa = QuantizedActivation(
        data=quant_out,
        scale=scale_out if nvfp4 else scale,
        orig_dtype=hidden_states.dtype,
        orig_shape=hidden_states.shape,
        quant_key=key,
    )
    return qa, norm_out, hidden_states
