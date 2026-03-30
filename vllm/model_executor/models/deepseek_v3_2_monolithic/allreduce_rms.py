# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
AllReduce + RMSNorm for the monolithic DeepSeek V3.2.

When FlashInfer is available and the input fits the workspace,
uses flashinfer.comm.allreduce_fusion (kARResidualRMSNorm) to fuse
AllReduce + residual-add + RMSNorm into a single kernel launch.
Otherwise falls back to NCCL AllReduce followed by the Triton
fused_add_rms_norm kernel.
"""

from __future__ import annotations

from importlib.util import find_spec

import torch

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .ops import fused_add_rms_norm

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# FlashInfer availability check
# ---------------------------------------------------------------------------
_flashinfer_comm = None
if find_spec("flashinfer"):
    try:
        import flashinfer.comm as _fi_comm

        if hasattr(_fi_comm, "allreduce_fusion") and hasattr(
            _fi_comm, "create_allreduce_fusion_workspace"
        ):
            _flashinfer_comm = _fi_comm
    except ImportError:
        pass


def is_flashinfer_allreduce_available() -> bool:
    return _flashinfer_comm is not None


def should_use_allreduce_rms() -> bool:
    """Return True when TP > 1 (AllReduce is needed)."""
    return get_tensor_model_parallel_world_size() > 1


# ---------------------------------------------------------------------------
# Workspace parameter helper
# ---------------------------------------------------------------------------
class AllReduceRMSParams:
    """Pre-computed parameters for allreduce + RMS norm."""

    def __init__(self, vllm_config: VllmConfig, hidden_size: int) -> None:
        from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
            _FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB,
            FI_ALLREDUCE_FUSION_MAX_SIZE_MB,
        )

        tp_size = get_tensor_model_parallel_world_size()
        cap = current_platform.get_device_capability()
        device_cap = cap.to_int() if cap is not None else None

        # Use the same workspace size threshold as the compiler pass.
        # Inputs that exceed this fall back to NCCL + fused_add_rms_norm.
        max_size_map = FI_ALLREDUCE_FUSION_MAX_SIZE_MB.get(device_cap, {})
        max_size_mb = max_size_map.get(tp_size, 64)
        MiB = 1024 * 1024
        element_size = torch.tensor([], dtype=torch.get_default_dtype()).element_size()
        self.fi_max_token_num = int((max_size_mb * MiB) // (hidden_size * element_size))

        # One-shot size threshold.
        one_shot_map = _FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB.get(device_cap, {})
        self.fi_max_one_shot_size_mb: float | None = one_shot_map.get(tp_size, None)

        self.tp_size = tp_size
        self.hidden_size = hidden_size
        self.use_flashinfer = is_flashinfer_allreduce_available()


# ---------------------------------------------------------------------------
# Unified allreduce + residual-add + RMS norm
# ---------------------------------------------------------------------------
def allreduce_add_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    params: AllReduceRMSParams | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AllReduce + residual add + RMS norm.

    Uses FlashInfer fused kernel when available and the input fits;
    otherwise falls back to NCCL AllReduce + Triton fused_add_rms_norm.

    After the call:
      hidden_states = rms_norm(allreduce(hidden_states) + residual, weight)
      residual      = allreduce(hidden_states) + residual
    """
    num_tokens = hidden_states.shape[0]

    if (
        params is not None
        and params.use_flashinfer
        and num_tokens <= params.fi_max_token_num
        and _flashinfer_comm is not None
    ):
        return _fi_allreduce_add_rms_norm(hidden_states, residual, weight, eps, params)

    # Fallback: NCCL AllReduce + Triton fused_add_rms_norm.
    hidden_states = tensor_model_parallel_all_reduce(hidden_states)
    return fused_add_rms_norm(hidden_states, residual, weight, eps)


def _fi_allreduce_add_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    params: AllReduceRMSParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashInfer fused AllReduce + residual add + RMS norm."""
    assert _flashinfer_comm is not None
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        get_fi_ar_workspace,
    )

    workspace = get_fi_ar_workspace(
        world_size=params.tp_size,
        rank=get_tensor_model_parallel_rank(),
        max_token_num=params.fi_max_token_num,
        hidden_dim=params.hidden_size,
        dtype=hidden_states.dtype,
        group=get_tp_group().device_group,
    )
    if workspace is None:
        # Workspace init failed — fall back.
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        return fused_add_rms_norm(hidden_states, residual, weight, eps)

    MiB = 1024 * 1024
    current_size = (
        hidden_states.shape[0] * params.hidden_size * hidden_states.element_size()
    )
    use_oneshot = params.fi_max_one_shot_size_mb is None or (
        current_size <= params.fi_max_one_shot_size_mb * MiB
    )

    layout_code = None
    if workspace.backend == "trtllm":
        layout_code = _flashinfer_comm.QuantizationSFLayout.SWIZZLED_128x4

    _flashinfer_comm.allreduce_fusion(
        input=hidden_states,
        workspace=workspace,
        pattern=_flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
        launch_with_pdl=True,
        output=None,
        residual_out=residual,
        norm_out=hidden_states,
        quant_out=None,
        scale_out=None,
        residual_in=residual,
        rms_gamma=weight,
        rms_eps=eps,
        scale_factor=None,
        layout_code=layout_code,
        use_oneshot=use_oneshot,
        fp32_acc=True,
    )
    return hidden_states, residual
