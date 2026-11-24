# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Orthogonal TP x PCP x DP vocabulary sharding plans and collectives."""

import weakref
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config_or_none
from vllm.distributed import (
    GroupCoordinator,
    get_dp_group,
    get_embed_exchange_group,
    get_lmhead_dp_group,
    get_pcp_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

VocabRole = Literal["auto", "embed_tokens", "lm_head", "standard"]

_EMBED_NAMES = frozenset(
    {"embed_tokens", "wte", "word_embeddings", "embedding", "tok_embeddings"}
)
_LMHEAD_NAMES = frozenset({"lm_head", "output", "embed_out", "head"})

_step_lmhead_capacity: ContextVar[int | None] = ContextVar(
    "step_lmhead_capacity", default=None
)
_embed_layer_counter: int = 0
_embed_layers: dict[str, weakref.ReferenceType["VocabParallelEmbedding"]] = {}


class VocabParallelMode(Enum):
    """Vocabulary-weight layout and collective execution strategy."""

    REPLICATED = auto()
    CUSTOM_GROUP = auto()
    STANDARD = auto()
    EXTENDED = auto()


@dataclass(frozen=True)
class VocabParallelPlan:
    """Shard coordinates and orthogonal process groups for a vocabulary layer.

    Extended sharding linearizes the `TP x PCP x FG_DP` grid with `TP` outer,
    `PCP` middle, and `FG_DP` inner:
        `shard_rank = (tp_rank * pcp_size + pcp_rank) * dp_size + dp_rank`
    This keeps `embed_tokens` and `lm_head` partitions identical when their
    configured sizes match, and allows `lm_head` to reconstruct natural
    token-ID order by concatenating `FG_DP`, `PCP`, and `TP` shards in sequence.
    """

    mode: VocabParallelMode
    role: str
    shard_rank: int
    shard_world_size: int
    dp_world_size: int = 1
    pcp_world_size: int = 1
    embedding_capacity: int = 1
    lmhead_capacity: int = 1
    token_exchange_group: GroupCoordinator | None = None
    output_reduce_group: GroupCoordinator | None = None
    dp_exchange_group: GroupCoordinator | None = None
    pcp_gather_group: GroupCoordinator | None = None
    tp_gather_group: GroupCoordinator | None = None


@contextmanager
def lmhead_step_capacity_context(capacity: int | None):
    """Publish the DP-agreed LM-head row count for the current step."""
    token = _step_lmhead_capacity.set(capacity)
    try:
        yield
    finally:
        _step_lmhead_capacity.reset(token)


def get_step_lmhead_capacity() -> int | None:
    return _step_lmhead_capacity.get()


def register_vocab_embed_layer(layer: "VocabParallelEmbedding") -> str:
    global _embed_layer_counter
    key = f"vocab_embed:{_embed_layer_counter}"
    _embed_layer_counter += 1
    _embed_layers[key] = weakref.ref(layer)
    return key


def _infer_vocab_role(
    *,
    vocab_role: VocabRole,
    is_lm_head_cls: bool,
    org_vocab_size: int,
    prefix: str,
    config: VllmConfig,
) -> Literal["embed_tokens", "lm_head", "standard"]:
    if vocab_role in ("embed_tokens", "lm_head", "standard"):
        return vocab_role

    parts = prefix.split(".") if prefix else []
    if "mtp" in parts:
        return "standard"

    if config.model_config is not None:
        hf_vocab = getattr(config.model_config.hf_text_config, "vocab_size", None)
        valid_vocabs = {config.model_config.get_vocab_size()}
        if hf_vocab is not None:
            valid_vocabs.add(hf_vocab)
        if org_vocab_size not in valid_vocabs:
            return "standard"

    leaf = parts[-1] if parts else ""
    if is_lm_head_cls:
        if not leaf or leaf in _LMHEAD_NAMES:
            return "lm_head"
        return "standard"

    if not leaf or leaf in _EMBED_NAMES:
        return "embed_tokens"
    return "standard"


def resolve_vocab_parallel_plan(
    *,
    org_vocab_size: int,
    prefix: str,
    disable_tp: bool,
    reduce_results: bool,
    parallel_group: GroupCoordinator | None,
    is_lm_head_cls: bool,
    vocab_role: VocabRole = "auto",
) -> VocabParallelPlan:
    """Build the shard layout and communication plan for a vocabulary layer."""
    if disable_tp:
        return VocabParallelPlan(
            mode=VocabParallelMode.REPLICATED,
            role="replicated",
            shard_rank=0,
            shard_world_size=1,
        )

    if parallel_group is not None:
        return VocabParallelPlan(
            mode=VocabParallelMode.CUSTOM_GROUP,
            role="standard",
            shard_rank=parallel_group.rank_in_group,
            shard_world_size=parallel_group.world_size,
            output_reduce_group=parallel_group if reduce_results else None,
        )

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    tp_group = get_tp_group() if tp_size > 1 else None

    config = get_current_vllm_config_or_none()
    if config is None or not config.parallel_config.fine_grained_tp_config.enabled:
        return VocabParallelPlan(
            mode=VocabParallelMode.STANDARD,
            role="lm_head" if is_lm_head_cls else "standard",
            shard_rank=tp_rank,
            shard_world_size=tp_size,
            output_reduce_group=tp_group if reduce_results else None,
            tp_gather_group=tp_group,
        )

    role = _infer_vocab_role(
        vocab_role=vocab_role,
        is_lm_head_cls=is_lm_head_cls,
        org_vocab_size=org_vocab_size,
        prefix=prefix,
        config=config,
    )
    if role == "standard":
        return VocabParallelPlan(
            mode=VocabParallelMode.STANDARD,
            role="standard",
            shard_rank=tp_rank,
            shard_world_size=tp_size,
            output_reduce_group=tp_group if reduce_results else None,
            tp_gather_group=tp_group,
        )

    fg_config = config.parallel_config.fine_grained_tp_config
    dp_size, pcp_size = (
        config.parallel_config.lmhead_dp_pcp_factors
        if role == "lm_head"
        else config.parallel_config.embedding_dp_pcp_factors
    )

    if pcp_size == 1 and dp_size == 1:
        return VocabParallelPlan(
            mode=VocabParallelMode.STANDARD,
            role=role,
            shard_rank=tp_rank,
            shard_world_size=tp_size,
            output_reduce_group=tp_group if reduce_results else None,
            tp_gather_group=tp_group,
        )

    if role == "embed_tokens" and not reduce_results:
        raise ValueError(
            "Extended vocabulary-parallel embedding requires reduce_results=True."
        )

    pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
    dp_rank = get_dp_group().rank_in_group if dp_size > 1 else 0
    shard_rank = (tp_rank * pcp_size + pcp_rank) * dp_size + dp_rank
    shard_world_size = tp_size * pcp_size * dp_size

    sched_max_tokens = config.scheduler_config.max_num_batched_tokens
    cg_max = config.compilation_config.max_cudagraph_capture_size or 0
    cg_sizes = config.compilation_config.cudagraph_capture_sizes
    cg_list_max = max(cg_sizes) if cg_sizes else 0
    embedding_capacity = max(1, sched_max_tokens, cg_max, cg_list_max)

    decode_query_len = 1 + getattr(config, "num_scheduled_spec_tokens", 0)
    lmhead_capacity = max(1, config.scheduler_config.max_num_seqs * decode_query_len)

    lmhead_dp, lmhead_pcp = config.parallel_config.lmhead_dp_pcp_factors
    can_score_lmhead = (
        role == "lm_head" or fg_config.lmhead_tp_size == fg_config.embedding_tp_size
    )
    dp_exchange_group = (
        get_lmhead_dp_group() if (can_score_lmhead and lmhead_dp > 1) else None
    )
    if dp_exchange_group is not None and role == "lm_head":
        assert dp_exchange_group.rank_in_group == dp_rank

    return VocabParallelPlan(
        mode=VocabParallelMode.EXTENDED,
        role=role,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
        dp_world_size=dp_size,
        pcp_world_size=pcp_size,
        embedding_capacity=embedding_capacity,
        lmhead_capacity=lmhead_capacity,
        token_exchange_group=(
            get_embed_exchange_group() if role == "embed_tokens" else None
        ),
        output_reduce_group=(
            tp_group if (role == "embed_tokens" and reduce_results) else None
        ),
        dp_exchange_group=dp_exchange_group,
        pcp_gather_group=(
            get_pcp_group() if (can_score_lmhead and lmhead_pcp > 1) else None
        ),
        tp_gather_group=tp_group if can_score_lmhead else None,
    )


def _resolve_embed_exchange_span(num_tokens: int, plan: VocabParallelPlan) -> int:
    if is_forward_context_available():
        fc = get_forward_context()
        if fc.cudagraph_runtime_mode != CUDAGraphMode.NONE or plan.dp_world_size == 1:
            return max(1, num_tokens)
        if fc.dp_metadata is not None:
            return max(
                1,
                num_tokens,
                int(fc.dp_metadata.num_tokens_across_dp_cpu.max().item()),
            )
    return plan.embedding_capacity


def _vocab_parallel_embed_exchange(
    flat_input: torch.Tensor, layer_key: str
) -> torch.Tensor:
    layer_ref = _embed_layers.get(layer_key)
    layer = layer_ref() if layer_ref is not None else None
    if layer is None:
        raise RuntimeError(f"Vocabulary embedding layer {layer_key} is no longer live.")

    plan = layer.vocab_parallel_plan
    exchange_group = plan.token_exchange_group
    assert exchange_group is not None

    num_tokens = flat_input.shape[0]
    exchange_tokens = _resolve_embed_exchange_span(num_tokens, plan)
    if num_tokens > exchange_tokens:
        raise ValueError(
            f"Embedding token count ({num_tokens}) exceeds exchange capacity "
            f"({exchange_tokens})."
        )

    padded_input = (
        F.pad(flat_input, (0, exchange_tokens - num_tokens))
        if num_tokens < exchange_tokens
        else flat_input
    )
    gathered_ids = exchange_group.all_gather(padded_input, dim=0)
    output_parallel, output_dtype = layer._lookup_local_shard(gathered_ids)
    scattered = exchange_group.reduce_scatter(output_parallel, dim=0)
    output = scattered[:num_tokens]

    reduce_group = plan.output_reduce_group
    if layer.reduce_results and reduce_group is not None and num_tokens > 0:
        output = reduce_group.all_reduce(output)

    return output.view(output_dtype)


def _vocab_parallel_embed_exchange_fake(
    flat_input: torch.Tensor, layer_key: str
) -> torch.Tensor:
    layer_ref = _embed_layers.get(layer_key)
    layer = layer_ref() if layer_ref is not None else None
    if layer is None:
        raise RuntimeError(f"Vocabulary embedding layer {layer_key} is no longer live.")
    return torch.empty(
        (flat_input.shape[0], layer.embedding_dim),
        dtype=layer.params_dtype,
        device=flat_input.device,
    )


direct_register_custom_op(
    op_name="vocab_parallel_embed_exchange",
    op_func=_vocab_parallel_embed_exchange,
    fake_impl=_vocab_parallel_embed_exchange_fake,
)


def forward_vocab_parallel_embedding(
    layer: "VocabParallelEmbedding", input_: torch.Tensor
) -> torch.Tensor:
    """Execute orthogonal `(PCP x FG_DP)` token exchange followed by `TP` sum."""
    assert layer._exchange_op_key is not None
    flat_input = input_ if input_.ndim == 1 else input_.reshape(-1)
    output = torch.ops.vllm.vocab_parallel_embed_exchange(
        flat_input, layer._exchange_op_key
    )
    if input_.ndim != 1:
        output = output.view(*input_.shape, layer.embedding_dim)
    return output


def lmhead_all_to_all(logits: torch.Tensor, group: GroupCoordinator) -> torch.Tensor:
    """Redistribute `[D * B, V_shard]` into `[B, D * V_shard]` across `group`."""
    world_size = group.world_size
    if world_size == 1:
        return logits
    if logits.shape[0] % world_size != 0:
        raise ValueError(
            f"LM head rows ({logits.shape[0]}) must be divisible by "
            f"group world_size ({world_size})."
        )
    vocab_per_partition = logits.shape[-1]
    batch_per_rank = logits.shape[0] // world_size
    exchanged = group.all_to_all_single(logits.contiguous())
    return (
        exchanged.view(world_size, batch_per_rank, vocab_per_partition)
        .permute(1, 0, 2)
        .contiguous()
        .view(batch_per_rank, world_size * vocab_per_partition)
    )


def compute_extended_lmhead_logits(
    hidden_states: torch.Tensor,
    lm_head: "VocabParallelEmbedding",
    apply_head_fn: Callable[[torch.Tensor], torch.Tensor],
    gather_tp_fn: Callable[[torch.Tensor], torch.Tensor | None],
    org_vocab_size: int,
) -> torch.Tensor | None:
    """Compute full-vocabulary logits over the `TP x PCP x FG_DP` shard grid."""
    plan = lm_head.vocab_parallel_plan
    leading_shape = hidden_states.shape[:-1]
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    num_rows = flat_hidden.shape[0]

    dp_group = plan.dp_exchange_group
    if dp_group is not None and dp_group.world_size > 1:
        capacity = get_step_lmhead_capacity()
        if capacity is None:
            capacity = plan.lmhead_capacity
        if num_rows > capacity:
            raise ValueError(
                f"LM head row count ({num_rows}) exceeds DP exchange capacity "
                f"({capacity})."
            )
        padded_hidden = (
            F.pad(flat_hidden, (0, 0, 0, capacity - num_rows))
            if num_rows < capacity
            else flat_hidden
        )
        gathered_hidden = dp_group.all_gather(padded_hidden, dim=0)
        partial_logits = apply_head_fn(gathered_hidden)
        logits = lmhead_all_to_all(partial_logits, dp_group)[:num_rows]
    else:
        logits = apply_head_fn(flat_hidden)

    if num_rows == 0:
        return logits.new_empty((*leading_shape, org_vocab_size))

    pcp_group = plan.pcp_gather_group
    if pcp_group is not None and pcp_group.world_size > 1:
        logits = pcp_group.all_gather(logits.contiguous(), dim=-1)

    final_logits: torch.Tensor | None = logits
    tp_group = plan.tp_gather_group
    if tp_group is not None and tp_group.world_size > 1:
        final_logits = gather_tp_fn(logits.contiguous())

    if final_logits is None:
        return None
    final_logits = final_logits[..., :org_vocab_size]
    return final_logits.view(*leading_shape, org_vocab_size)


def validate_vocab_parallel_model(model: nn.Module, vllm_config: VllmConfig) -> None:
    """Verify required vocabulary layers participate in configured sharding."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    emb_dp, emb_pcp = vllm_config.parallel_config.embedding_dp_pcp_factors
    lmhead_dp, lmhead_pcp = vllm_config.parallel_config.lmhead_dp_pcp_factors
    need_embed = get_pp_group().is_first_rank and (emb_dp > 1 or emb_pcp > 1)
    is_pooling = (
        vllm_config.model_config is not None
        and vllm_config.model_config.runner_type == "pooling"
    )
    need_lmhead = (
        get_pp_group().is_last_rank
        and not is_pooling
        and (lmhead_dp > 1 or lmhead_pcp > 1)
    )

    found_embed = False
    found_lmhead = False
    for name, layer in model.named_modules(remove_duplicate=False):
        if not isinstance(layer, VocabParallelEmbedding):
            continue
        if layer.vocab_parallel_plan.mode is not VocabParallelMode.EXTENDED:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if isinstance(layer, ParallelLMHead) or leaf in _LMHEAD_NAMES:
            found_lmhead = True
        if not isinstance(layer, ParallelLMHead):
            found_embed = True

    missing: list[str] = []
    if need_embed and not found_embed:
        missing.append("embed_tokens")
    if need_lmhead and not found_lmhead:
        missing.append("lm_head")
    if missing:
        raise ValueError(
            f"Model does not support extended vocabulary sharding for {missing}."
        )
