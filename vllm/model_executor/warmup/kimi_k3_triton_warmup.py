# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up Kimi-K3 Triton kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _get_kda_layer(worker: Worker) -> KimiK3DeltaAttention | None:
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention

    compilation_config = getattr(
        worker.model_runner,
        "compilation_config",
        None,
    )
    static_context = getattr(compilation_config, "static_forward_context", None)
    if not isinstance(static_context, dict):
        return None
    return next(
        (
            layer
            for layer in static_context.values()
            if isinstance(layer, KimiK3DeltaAttention)
        ),
        None,
    )


def _warm_attn_res(worker: Worker) -> None:
    from vllm.models.kimi_k3.nvidia.ops.attn_res import (
        attn_res,
        get_attn_res_triton_warmup_profiles,
    )

    config = worker.model_config.hf_text_config
    block_size = getattr(config, "attn_res_block_size", None)
    if block_size is None:
        return

    hidden_size = int(config.hidden_size)
    max_blocks = (int(config.num_hidden_layers) + block_size - 1) // block_size
    if max_blocks < 2:
        return

    dtype = worker.model_config.dtype
    device = torch.device("cuda")
    eps = float(config.rms_norm_eps)
    prefix = torch.zeros((1, hidden_size), dtype=dtype, device=device)
    delta = torch.zeros_like(prefix)
    blocks = torch.zeros(
        (1, max_blocks, hidden_size),
        dtype=dtype,
        device=device,
    )
    norm_weight = torch.zeros(hidden_size, dtype=dtype, device=device)
    qk_weight = torch.zeros_like(norm_weight)
    output_norm_weight = torch.zeros_like(norm_weight)

    for (
        num_blocks,
        has_delta,
        block_write_idx,
        apply_output_norm,
    ) in get_attn_res_triton_warmup_profiles(max_blocks):
        attn_res(
            prefix,
            delta if has_delta else None,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight if apply_output_norm else None,
            num_blocks=num_blocks,
            block_write_idx=block_write_idx,
            eps=eps,
            output_norm_eps=eps if apply_output_norm else 0.0,
        )


def _warm_recurrent_kda(
    layer: KimiK3DeltaAttention,
    input_dtype: torch.dtype,
) -> None:
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda.fused_recurrent import (
        fused_recurrent_kda,
        get_fused_recurrent_kda_fwd_warmup_profiles,
    )

    num_speculative_tokens = int(layer.num_spec)
    # fused_recurrent_kda_fwd_kernel is only used by speculative decode.
    if num_speculative_tokens <= 0:
        return

    kv_cache = layer.kv_cache
    if not isinstance(kv_cache, (list, tuple)) or len(kv_cache) < 2:
        return
    state = kv_cache[1]
    if not isinstance(state, torch.Tensor) or not state.numel():
        return

    logger.info("Warming up Kimi-K3 speculative KDA kernels.")
    h = int(layer.local_num_heads)
    d = int(layer.head_dim)
    tokens_per_sequence = num_speculative_tokens + 1
    for num_sequences in get_fused_recurrent_kda_fwd_warmup_profiles(h):
        num_tokens = num_sequences * tokens_per_sequence
        packed_qkv = torch.empty(
            (num_tokens, 3 * h * d),
            dtype=input_dtype,
            device=state.device,
        )
        q, k, v = (
            tensor.view(1, num_tokens, h, d)
            for tensor in packed_qkv.split(h * d, dim=-1)
        )
        fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            raw_g=torch.empty(
                (1, num_tokens, h, d),
                dtype=input_dtype,
                device=state.device,
            ),
            raw_beta=torch.empty(
                (1, num_tokens, h),
                dtype=input_dtype,
                device=state.device,
            ),
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            lower_bound=layer.gate_lower_bound,
            initial_state=state[:1],
            cu_seqlens=torch.arange(
                0,
                num_tokens + 1,
                tokens_per_sequence,
                dtype=torch.int32,
                device=state.device,
            ),
            ssm_state_indices=torch.zeros(
                (num_sequences, tokens_per_sequence),
                dtype=torch.int32,
                device=state.device,
            ),
            num_accepted_tokens=torch.ones(
                num_sequences,
                dtype=torch.int32,
                device=state.device,
            ),
            out=torch.empty(
                (1, num_tokens, h, d),
                dtype=input_dtype,
                device=state.device,
            ),
        )


@torch.inference_mode()
def kimi_k3_triton_warmup(worker: Worker) -> None:
    """Warm Kimi-K3 Triton kernels reachable by this server."""
    if not current_platform.is_cuda():
        return

    layer = _get_kda_layer(worker)
    if layer is None:
        return

    _warm_attn_res(worker)
    _warm_recurrent_kda(layer, worker.model_config.dtype)
