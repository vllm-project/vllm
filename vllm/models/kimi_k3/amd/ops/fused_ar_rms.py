# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER fused all-reduce + RMSNorm for the Kimi-K3 AMD latent-MoE tail.

K3 serves on AMD with vLLM's own compilation -- ``CompilationMode.NONE`` plus
breakable CUDA graphs -- so the torch.compile AR+RMSNorm fusion pass
(``RocmAiterAllReduceFusionPass``) never runs. This calls the registered AITER
op directly, the same way ``amd/mla.py`` calls
``torch.ops.vllm.fused_mla_dual_rms_norm``: a plain launch on the model stream
that stream capture records into the surrounding graph segment.

The dispatch decision reads only shape, dtype, TP size and topology, so a
captured graph always replays the branch it was captured with.
"""

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.layernorm import RMSNorm

# AITER's fused kernel always takes a residual; the latent tail has none, so it
# gets zeros. Keyed per (shape, dtype, device) and never evicted: a single
# buffer that grew on a later shape would leave already-captured graphs
# pointing at freed memory.
_zero_residuals: dict[tuple[torch.Size, torch.dtype, torch.device], torch.Tensor] = {}


def _zero_residual(hidden_states: torch.Tensor) -> torch.Tensor:
    key = (hidden_states.shape, hidden_states.dtype, hidden_states.device)
    buf = _zero_residuals.get(key)
    if buf is None:
        buf = torch.zeros_like(hidden_states)
        _zero_residuals[key] = buf
    return buf


def can_fuse_allreduce_rms_norm(hidden_states: torch.Tensor) -> bool:
    """Whether AITER's one-stage fused AR+RMSNorm covers this tensor.

    Outside the one-stage gate AITER runs a two-stage reduce-scatter plus local
    norm, which is slower than an explicit all-reduce followed by RMSNorm, so
    the caller must fall back rather than fuse unconditionally.
    """
    if not rocm_aiter_ops.is_custom_all_reduce_enabled():
        return False
    if hidden_states.dim() != 2 or not hidden_states.is_contiguous():
        return False
    if hidden_states.dtype not in (torch.bfloat16, torch.float16):
        return False

    aiter_ar = rocm_aiter_ops.get_aiter_allreduce()
    if aiter_ar is None or aiter_ar.disabled:
        return False
    # Pre-0.1.12 builds template-specialize the launcher on hidden_dim and
    # silently no-op outside {512, 1024, 2048, 4096}; K3's latent is 3584.
    if not aiter_ar.supports_dynamic_hidden_dim:
        return False
    if not aiter_ar.should_custom_ar(hidden_states):
        return False
    return aiter_ar.use_1stage_fused_ar_rms(hidden_states)


def fused_allreduce_rms_norm_out(
    hidden_states: torch.Tensor,
    norm: RMSNorm,
) -> torch.Tensor:
    """All-reduce + RMSNorm with no residual add, fused when AITER admits it.

    Equivalent to ``norm(all_reduce(hidden_states))``.
    """
    if get_tensor_model_parallel_world_size() == 1:
        return norm(hidden_states)

    if not can_fuse_allreduce_rms_norm(hidden_states):
        return norm(tensor_model_parallel_all_reduce(hidden_states))

    norm_out, _ = rocm_aiter_ops.get_fused_allreduce_rmsnorm_op()(
        input_=hidden_states,
        residual=_zero_residual(hidden_states),
        weight=norm.weight,
        epsilon=norm.variance_epsilon,
        gemma_norm=False,
    )
    return norm_out
