# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`embedding/vocab_gather_common` (entry `vocab_gather_comm`): the target's token
embedding (and the DSpark drafter's block-id embedding, which aliases the same module)
as one peer-read gather kernel instead of mask + `F.embedding` + `masked_fill_` +
all-reduce.

The module's `forward` is replaced on the instance (`install`); ids the built gather
does not take (`vocab_gather_unsupported_reason`: not a contiguous 1-D int32/int64
tensor on the weight's device) and `inputs_embeds` paths go to the stock forward. Under
torch.compile (the release DSpark draft's backbone, which shares this module) the call
is the opaque op `vllm.kimi_akr_vocab_gather`, which makes the same choice when it runs.
The shard must stay unchanged and allocated after `install` (it is peer-mapped)."""

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.models.kimi_k3.amd import akr_nonlayer
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_state: dict[str, Any] = {}


def _lookup(ids: torch.Tensor) -> torch.Tensor:
    """The installed embedding of `ids` [...] -> [..., hidden]: the gather, or the stock
    forward for ids it does not take."""
    from amd_kernels.embedding.vocab_gather_common import (
        vocab_gather_unsupported_reason,
    )

    gather, stock = _state["gather"], _state["stock"]
    if ids.numel() > 0 and ids.is_contiguous():
        flat = ids.view(-1)
        if vocab_gather_unsupported_reason(gather, flat) is None:
            out = gather(flat)
            return out if ids.ndim == 1 else out.view(*ids.shape, -1)
    return stock(ids)


def _lookup_fake(ids: torch.Tensor) -> torch.Tensor:
    hidden, dtype = _state["out"]
    return ids.new_empty((*ids.shape, hidden), dtype=dtype)


direct_register_custom_op(
    op_name="kimi_akr_vocab_gather", op_func=_lookup, fake_impl=_lookup_fake
)


def unsupported_reason(embed) -> str | None:
    """vLLM-side gates (the shard layout the gather assumes); the kernel's shape gates
    are the entry's `vocab_gather_comm_unsupported_reason`."""
    g = akr_nonlayer.tp_group()
    if getattr(embed, "tp_size", 1) != g.world_size or embed.parallel_group is not None:
        return "embedding is not sharded over the TP group"
    if embed.num_embeddings != embed.org_vocab_size:
        return "padded or added vocabulary"
    if embed.org_vocab_size % g.world_size:
        return f"vocabulary {embed.org_vocab_size} not divisible by {g.world_size}"
    shard = embed.org_vocab_size // g.world_size
    if embed.weight.shape[0] != shard or not embed.reduce_results:
        return f"weight {tuple(embed.weight.shape)} or results not reduced"
    if embed.shard_indices.org_vocab_start_index != g.rank_in_group * shard:
        return "vocabulary shards are not in TP rank order"
    return None


def install(embed) -> bool:
    """Collective on the TP group (every rank, same order, after the weights load,
    before any CUDA graph capture). Returns whether the kernel path is on."""
    if not akr_nonlayer.switch("EMBED"):
        return False
    if akr_nonlayer.blocked("EMBED", unsupported_reason(embed)):
        return False
    from amd_kernels.embedding.vocab_gather_common import (
        vocab_gather_comm,
        vocab_gather_comm_unsupported_reason,
    )

    g = akr_nonlayer.tp_group()
    w = embed.weight.data
    if akr_nonlayer.blocked(
        "EMBED", vocab_gather_comm_unsupported_reason(w, g.rank_in_group, g.cpu_group)
    ):
        return False
    gather = vocab_gather_comm(w, g.rank_in_group, g.cpu_group)
    gather.warm_up()
    _state["gather"] = gather
    _state["stock"] = embed.forward
    _state["out"] = (w.shape[1], w.dtype)

    def forward(input_: torch.Tensor):
        if torch.compiler.is_compiling():
            return torch.ops.vllm.kimi_akr_vocab_gather(input_)
        return _lookup(input_)

    embed.forward = forward
    logger.info(
        "amd_kernels embedding: VocabGather on (weight %s)", tuple(embed.weight.shape)
    )
    return True


def close() -> None:
    _state.pop("stock", None)
    gather = _state.pop("gather", None)
    if gather is not None:
        gather.close()
