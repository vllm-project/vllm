# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up Kimi-K3 Triton kernels on ROCm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention

logger = init_logger(__name__)


@torch.inference_mode()
def warmup_kda_triton_kernels(
    layer: KimiK3DeltaAttention, input_dtype: torch.dtype
) -> None:
    from vllm.models.kimi_k3.amd.ops.kda_prefill import (
        chunk_kda_prefill,
        get_gluon_chunk_kda_warmup_profiles,
    )

    if layer.kda_prefill_backend != "gluon":
        return
    kv_cache = layer.kv_cache
    if not isinstance(kv_cache, (list, tuple)) or len(kv_cache) < 2:
        return
    state = kv_cache[1]
    if not isinstance(state, torch.Tensor) or not state.numel():
        return

    logger.info("Warming up Kimi-K3 Gluon chunk KDA prefill kernels.")
    h, d = int(layer.local_num_heads), int(layer.head_dim)
    for num_seqs, seq_len in get_gluon_chunk_kda_warmup_profiles():
        num_tokens = num_seqs * seq_len
        shape = (1, num_tokens, h, d)
        chunk_kda_prefill(
            q=torch.zeros(shape, dtype=input_dtype, device=state.device),
            k=torch.zeros(shape, dtype=input_dtype, device=state.device),
            v=torch.zeros(shape, dtype=input_dtype, device=state.device),
            raw_g=torch.zeros(shape, dtype=input_dtype, device=state.device),
            raw_beta=torch.zeros(
                (1, num_tokens, h), dtype=input_dtype, device=state.device
            ),
            A_log=layer.A_log,
            g_bias=layer.dt_bias,
            lower_bound=layer.gate_lower_bound,
            state_cache=state,
            state_indices=torch.zeros(num_seqs, dtype=torch.int32, device=state.device),
            has_initial_state=torch.zeros(
                num_seqs, dtype=torch.bool, device=state.device
            ),
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=torch.arange(
                0, num_tokens + 1, seq_len, dtype=torch.int32, device=state.device
            ),
            backend="gluon",
            out=torch.empty(shape, dtype=input_dtype, device=state.device),
        )
