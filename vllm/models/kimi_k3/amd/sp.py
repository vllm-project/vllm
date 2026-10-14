# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-sharded residual stream for Kimi-K3 prefill on ROCm.

With TP, every attention / MLP output all-reduce is replaced by a
reduce-scatter over tokens, and the (per-token) attention-residual mixing runs
on the local token shard only; the shard is all-gathered right before each
attention / MLP input. Communication volume is unchanged (AR = RS + AG), while
attn_res traffic and the block-residual bank shrink by the TP size.

In latent-MoE layers the router gate, latent down-projection and top-k also run
on the token shard (their outputs are all-gathered instead of being recomputed
on every rank), and the column-parallel up-projection tail reduce-scatters
its output instead of all-reducing it.

Enabled by ``VLLM_KIMI_K3_AMD_PREFILL_SP_MIN_TOKENS``; see :mod:`vllm.envs`.
"""

import contextlib
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.distributed import (
    get_dp_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape


def should_shard(num_tokens: int, has_residual: bool, has_aux_layers: bool) -> bool:
    """Whether this forward runs with a token-sharded residual stream.

    The residual stream is padded to a multiple of the TP size, so every step
    above the threshold shards; otherwise the unsharded path's larger
    activations would exceed the profiled peak.
    """
    min_tokens = envs.VLLM_KIMI_K3_AMD_PREFILL_SP_MIN_TOKENS
    return (
        min_tokens > 0
        and get_tensor_model_parallel_world_size() > 1
        and num_tokens >= min_tokens
        and not has_residual
        and not has_aux_layers
        and get_pp_group().is_last_rank
        and get_dp_group().world_size == 1
        and not torch.cuda.is_current_stream_capturing()
    )


def gather_tokens(shard: torch.Tensor, num_tokens: int | None = None) -> torch.Tensor:
    """All-gather the token shards, dropping the TP padding rows if
    ``num_tokens`` is given."""
    out = tensor_model_parallel_all_gather(shard, dim=0)
    return out if num_tokens is None else out[:num_tokens]


def gather_activation(
    shard: torch.Tensor | QuantizedActivation, num_tokens: int | None = None
) -> torch.Tensor | QuantizedActivation:
    """:func:`gather_tokens` that also accepts a per-token quantized
    activation, gathering its data and scale and rebuilding it."""
    if isinstance(shard, torch.Tensor):
        return gather_tokens(shard, num_tokens)
    key = shard.quant_key
    assert key.scale.group_shape == GroupShape.PER_TOKEN and key.scale2 is None
    data = gather_tokens(shard.data, num_tokens)
    return replace(
        shard,
        data=data,
        scale=gather_tokens(shard.scale, num_tokens),
        orig_shape=torch.Size((data.size(0), *shard.orig_shape[1:])),
    )


def scatter_tokens(partial: torch.Tensor) -> torch.Tensor:
    """Pad unpadded per-token partial sums back and reduce-scatter them."""
    pad = (-partial.size(0)) % get_tensor_model_parallel_world_size()
    if pad:
        partial = F.pad(partial, (0, 0, 0, pad))
    return tensor_model_parallel_reduce_scatter(partial, dim=0)


def all_gather_routing(
    topk_weights: torch.Tensor, topk_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-gather this rank's top-k over tokens in one collective."""
    w_bytes = topk_weights.contiguous().view(torch.uint8)
    packed = torch.cat([w_bytes, topk_ids.contiguous().view(torch.uint8)], dim=1)
    packed = tensor_model_parallel_all_gather(packed, dim=0)
    nw = w_bytes.size(1)
    return (
        packed[:, :nw].contiguous().view(topk_weights.dtype),
        packed[:, nw:].contiguous().view(topk_ids.dtype),
    )


@contextlib.contextmanager
def pre_routed(router: Any, topk: tuple[torch.Tensor, torch.Tensor]) -> Iterator[None]:
    """Make ``router.select_experts`` return ``topk`` for the duration."""
    shadowed = router.__dict__.get("select_experts")
    router.select_experts = lambda *args, **kwargs: topk
    try:
        yield
    finally:
        if shadowed is None:
            del router.select_experts
        else:
            router.select_experts = shadowed
