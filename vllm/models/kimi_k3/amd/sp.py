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
on every rank), and the latent is reduce-scattered instead of all-reduced.

Enabled by ``VLLM_KIMI_K3_AMD_PREFILL_SP_MIN_TOKENS``; see :mod:`vllm.envs`.
"""

import contextlib
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)

# True while the decoder stack runs token-sharded.
ACTIVE = False
# Unpadded token count of the sharded step. The residual stream is padded to a
# multiple of the TP size, so every step above the threshold shards; otherwise
# the unsharded path's larger activations would exceed the profiled peak.
NUM_TOKENS = 0


def should_shard(num_tokens: int, has_residual: bool, has_aux_layers: bool) -> bool:
    """Whether this forward runs with a token-sharded residual stream."""
    min_tokens = envs.VLLM_KIMI_K3_AMD_PREFILL_SP_MIN_TOKENS
    return (
        min_tokens > 0
        and get_tensor_model_parallel_world_size() > 1
        and num_tokens >= min_tokens
        and not has_residual
        and not has_aux_layers
        and get_pp_group().is_last_rank
        and not torch.cuda.is_current_stream_capturing()
    )


def gather_tokens(shard: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """All-gather the token shards and drop the TP padding rows."""
    return tensor_model_parallel_all_gather(shard, dim=0)[:num_tokens]


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
