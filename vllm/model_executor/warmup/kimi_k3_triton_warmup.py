# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up Kimi-K3 Triton kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention as AMDKDA
    from vllm.models.kimi_k3.nvidia.kda import (
        KimiK3DeltaAttention as NvidiaKDA,
    )
    from vllm.v1.worker.gpu_worker import Worker

    KimiK3DeltaAttention = AMDKDA | NvidiaKDA

logger = init_logger(__name__)

# Keep prefill JIT warmup bounded, following the representative-profile style
# used by the NVIDIA Kimi-K3 warmups. This is also the primary profiled input
# size for the AITER FlashKDA integration.
_AITER_KDA_PREFILL_WARMUP_TOKENS = 4096


def _get_kda_layer(worker: Worker) -> KimiK3DeltaAttention | None:
    if current_platform.is_rocm():
        from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention
    else:
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


def _warm_aiter_kda_prefill(
    layer: KimiK3DeltaAttention,
    input_dtype: torch.dtype,
    max_num_batched_tokens: int,
) -> None:
    if getattr(layer, "kda_prefill_backend", None) != "flashkda":
        return

    from vllm.models.kimi_k3.amd.ops.kda_prefill import (
        aiter_causal_conv1d_prefill,
        aiter_kda_prefill,
    )

    kv_cache = layer.kv_cache
    if not isinstance(kv_cache, (list, tuple)) or len(kv_cache) < 2:
        return
    state = kv_cache[1]
    if not isinstance(state, torch.Tensor) or not state.numel():
        return

    logger.info("Warming up Kimi-K3 AITER FlashKDA prefill kernels.")
    num_tokens = min(
        max(32, int(max_num_batched_tokens)),
        _AITER_KDA_PREFILL_WARMUP_TOKENS,
    )
    h = int(layer.local_num_heads)
    d = int(layer.head_dim)
    projection_size = h * d
    conv_width = int(layer.conv_size)
    prefill_weight = getattr(layer, "prefill_conv1d_weight", None)
    if not isinstance(prefill_weight, torch.Tensor) or not prefill_weight.numel():
        return

    packed_qkv = torch.zeros(
        (num_tokens, 3 * projection_size),
        dtype=input_dtype,
        device=state.device,
    )
    conv_state = torch.zeros(
        (1, 3 * projection_size, conv_width - 1),
        dtype=input_dtype,
        device=state.device,
    )
    cu_seqlens = torch.tensor(
        [0, num_tokens],
        dtype=torch.int32,
        device=state.device,
    )
    q, k, v = aiter_causal_conv1d_prefill(
        x=packed_qkv,
        weight=prefill_weight,
        bias=layer.conv1d.bias,
        conv_state=conv_state,
        query_start_loc=cu_seqlens,
        projection_size=projection_size,
        cache_indices=torch.zeros(1, dtype=torch.int32, device=state.device),
        has_initial_state=torch.zeros(1, dtype=torch.bool, device=state.device),
        metadata=None,
    )
    q, k, v = (tensor.view(1, num_tokens, h, d) for tensor in (q, k, v))
    raw_gate = torch.zeros_like(q)
    raw_beta = torch.zeros(
        (1, num_tokens, h),
        dtype=input_dtype,
        device=state.device,
    )
    initial_state = torch.zeros(
        (1, h, d, d),
        dtype=state.dtype,
        device=state.device,
    )
    assert layer.gate_lower_bound is not None
    aiter_kda_prefill(
        q=q,
        k=k,
        v=v,
        raw_gate=raw_gate,
        raw_beta=raw_beta,
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        lower_bound=layer.gate_lower_bound,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
    )


@torch.inference_mode()
def kimi_k3_triton_warmup(worker: Worker) -> None:
    """Warm Kimi-K3 Triton kernels reachable by this server."""
    if not (current_platform.is_cuda() or current_platform.is_rocm()):
        return

    layer = _get_kda_layer(worker)
    if layer is None:
        return

    if current_platform.is_rocm():
        _warm_aiter_kda_prefill(
            layer,
            worker.model_config.dtype,
            worker.scheduler_config.max_num_batched_tokens,
        )
    else:
        _warm_attn_res(worker)
        _warm_recurrent_kda(layer, worker.model_config.dtype)
