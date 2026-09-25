# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer-only context parallelism for MiniMax M3 (AMD MSA indexer).

Inverts the tensor-parallel split for the lightning indexer: instead of each
rank owning one index head and scoring every 128-token block, every rank
projects all index heads and scores ``1/P`` of the blocks, then one small
exchange gives each rank every shard's candidates for the heads it owns. Both
caches, the sparse attend, the block tables and the MoE are untouched -- this
shards indexer *work*, not state.

That is why it is not ``decode_context_parallel_size``, which shards the KV
cache itself (virtual block size, slot-mapping routing, LSE-merged output) and
is rejected below. The index cache is already replicated on every rank, so at
long context each rank re-reads the whole of it every step; sharding the block
axis is what turns that into ``1/P`` of the reads, and it is the only thing
here that scales with context.

Blocks go round-robin, ``owner(b) = b % P``, rather than in contiguous runs:
under causal masking contiguous shards would leave rank 0 with always-full
blocks and the last rank with mostly-empty tails, while round-robin keeps the
valid counts even at every sequence length.

Selection stays exact. Each rank cuts its shard to its own top-k -- the full
``k``, never ``k/P``, since the global winners may all live in one shard -- and
the merge takes the top-k of the ``P * k`` gathered candidates. Candidates
travel as the packed ``(score, global block id)`` sort keys the single-GPU
selector already uses to merge across chunks, so nothing on the wire has to say
which rank a candidate came from, and the payload is ``P * k * 8`` bytes per
token no matter how long the context is.
"""

import torch
import torch.distributed as dist

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.logger import init_logger

logger = init_logger(__name__)

# The block size the scoring and merge kernels are written against.
INDEXER_CP_SPARSE_BLOCK_SIZE = 128


def minimax_m3_indexer_cp_unsupported_reason(
    vllm_config: VllmConfig,
) -> str | None:
    """Return why this config cannot context-parallelize the indexer, or None.

    Deliberately includes the MSA indexer's own gate: the platform-neutral
    Triton indexer has no CP path, so a config that falls back to it must fall
    back on the head count too. Resolving that here rather than after the
    impl probe is what keeps the two from having to be re-decided against each
    other -- the probe below runs once, at whichever head count this returns.
    """
    from vllm.models.minimax_m3.amd.indexer_msa import (
        msa_indexer_unsupported_reason,
    )
    from vllm.models.minimax_m3.common.sparse_attention import (
        minimax_m3_use_aiter_sparse_pa,
    )

    if not vllm_config.attention_config.indexer_cp:
        return "not requested (--attention-config '{\"indexer_cp\": true}')"

    config = vllm_config.model_config.hf_text_config
    sparse_cfg = getattr(config, "sparse_attention_config", None)
    if sparse_cfg is None:
        return "the model has no sparse_attention_config (not MiniMax-M3)"

    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    if dcp_size > 1:
        # The two features both claim the context axis, but DCP claims it in
        # the cache: virtual block size, slot mappings that drop non-owned
        # tokens, an LSE-merged attend. Layering this on top would shard an
        # already-sharded context.
        return (
            f"decode_context_parallel_size={dcp_size} > 1 (KV-cache DCP owns "
            "the context axis; indexer CP replaces it, it does not extend it)"
        )

    total_index_heads = int(sparse_cfg["sparse_num_index_heads"])
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size > total_index_heads or total_index_heads % tp_size:
        # Every rank scores every head, so what the exchange routes is a
        # contiguous run of heads per rank; that run has to be the same width
        # on all of them, and it has to be the same run the tensor-parallel
        # split already gave this rank its KV heads from.
        return (
            f"needs sparse_num_index_heads divisible by tensor_parallel_size, "
            f"got tp_size={tp_size}, num_index_heads={total_index_heads}"
        )

    block_size = int(sparse_cfg["sparse_block_size"])
    if block_size != INDEXER_CP_SPARSE_BLOCK_SIZE:
        return (
            f"needs sparse_block_size={INDEXER_CP_SPARSE_BLOCK_SIZE}, got {block_size}"
        )

    # The MSA indexer is only kept if the AITER attend is also taken, since
    # the page table it emits is the one that attend reads; when it is not,
    # the model drops to the platform-neutral indexer and CP would have no
    # impl to run on. Same call the model makes, with the same argument -- a
    # MSA indexer is by definition one that emits the table.
    if not minimax_m3_use_aiter_sparse_pa(1, emits_sparse_block_table=True):
        return "the AITER sparse attend is not enabled, so the indexer is Triton's"

    # Probed at the CP head count, since that is the width this rank projects.
    # It is a weaker check than it was: both passes are Triton now, so what is
    # left is the shape contract, which CP does not change.
    reason = msa_indexer_unsupported_reason(
        topk_blocks=int(sparse_cfg["sparse_topk_blocks"]),
        sparse_block_size=block_size,
        num_index_heads=total_index_heads,
        index_head_dim=int(sparse_cfg["sparse_index_dim"]),
        indexer_kv_dtype=vllm_config.attention_config.resolve_indexer_kv_dtype("bf16"),
        max_model_len=vllm_config.model_config.max_model_len,
        score_type=sparse_cfg.get("sparse_score_type", "max"),
    )
    if reason is not None:
        return f"the MSA indexer is unusable at {total_index_heads} heads ({reason})"
    return None


def minimax_m3_indexer_cp_enabled(vllm_config: VllmConfig) -> bool:
    """Whether this config runs the indexer context-parallel.

    Recomputed rather than cached: every input is a config or topology value
    that is fixed for the process, so the answer is stable, and a module-level
    cache would only make it survive across the configs a test builds.
    """
    reason = minimax_m3_indexer_cp_unsupported_reason(vllm_config)
    if reason is not None:
        if vllm_config.attention_config.indexer_cp:
            logger.info_once("MiniMax M3 indexer CP: disabled (%s)", reason)
        return False
    world = get_tensor_model_parallel_world_size()
    logger.info_once(
        "MiniMax M3 indexer CP: enabled over %d ranks (each scores 1/%d of "
        "the blocks for every index head)",
        world,
        world,
    )
    return True


def get_indexer_cp_group():
    """The group the candidate exchange runs over.

    No ``new_group`` here: the gate requires ``tp_size`` to divide the index
    heads, so the CP group is exactly the TP group, and rank ``r``'s TP
    position already fixes the run of heads it owns. That is what lets the
    exchange skip a mapping table -- the all-gather keeps that run, and the
    all-to-all's implicit "chunk j goes to group rank j" lands it -- and it is
    also why the AITER allreduce object built for the TP group is the right one
    to borrow the IPC buffers from.
    """
    return get_tp_group()


def _aiter_all_gather(keys: torch.Tensor) -> torch.Tensor | None:
    """AITER's IPC all-gather over the CP group, or None if unavailable.

    Reaches for the tensor-parallel group's AITER allreduce object, which is
    the right one precisely because the gate forces the CP group to *be* the
    TP group. Returns None whenever anything about the build, the topology or
    this payload makes the custom path ineligible, leaving the caller on the
    portable collective.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    comm = rocm_aiter_ops.get_aiter_allreduce()
    if comm is None or comm.disabled:
        return None
    # Narrowed to int32 on the way in. A gather is a bitwise concatenation, so
    # the element type is only ever a label here -- but AITER relabels an
    # integer input as float of the same width before handing it to its pybind
    # layer, and that layer has no float64, so an int64 buffer dies there on a
    # dtype AITER chose itself. Two int32s per key over a contiguous last axis,
    # so this is a reinterpretation and not a copy.
    packed = keys.view(torch.int32)
    if not comm.should_custom_ag(packed):
        return None
    gathered = comm.custom_all_gather(packed, dim=0)
    if gathered is None:
        return None
    return gathered.view(torch.int64)


def exchange_candidates(keys: torch.Tensor) -> torch.Tensor:
    """Give this rank every shard's candidates for the heads it owns.

    ``keys`` is ``[heads, tokens, topk]`` packed sort keys, head-major, holding
    every index head because every rank scored every one of them over its own
    shard. Comes back as ``[world, owned, tokens, topk]``: entry ``r`` is rank
    ``r``'s candidates for the heads *this* rank owns, which are the same
    contiguous run the tensor-parallel split assigned it.

    The payload does not grow with the context -- it is a few hundred bytes
    per token however long the context is -- which is the whole reason the
    shards trade candidates rather than raw block scores. That also decides
    *which* collective: at this size neither one moves enough to matter and
    both cost only their launch, so the cheaper primitive wins even though it
    is the one that moves more. AITER's IPC all-gather hands every rank all
    ``P`` shards of all ``P`` heads and lets this rank keep the single column
    it owns, discarding ``P-1`` of every ``P`` rows; the all-to-all below
    sends only the rows that are actually wanted. Measured per layer per
    decode step at P=4, the wasteful one takes 5.8us and the tidy one 14.6us.

    Falls back to the all-to-all when the custom path is unavailable -- an
    AITER without it, a non-fully-connected topology, a payload outside its
    IPC buffer -- so this stays correct off the fast path, just slower.

    Neither collective is ever a capture's first, which would hang on the peer
    setup NCCL defers to it: every captured shape is run eagerly first
    (``cudagraph_num_of_warmups``, forced to 1 wherever capture happens), and
    the transport is a function of that shape, so the warmup takes the same
    branch the capture will.
    """
    group = get_indexer_cp_group()
    world = group.world_size
    heads = keys.shape[0]
    assert keys.is_contiguous(), "the candidate exchange needs a contiguous send buffer"
    assert heads % world == 0, (
        f"candidate exchange expects the index heads to divide over the group, "
        f"got {heads} heads over {world} ranks"
    )
    owned = heads // world
    lo = group.rank_in_group * owned

    gathered = _aiter_all_gather(keys)
    if gathered is not None:
        # [world, heads, tokens, topk], source rank major since the gather
        # concatenates along dim 0. Every rank holds every head, so this rank
        # keeps the run the tensor-parallel split gave it and drops the rest.
        # Left as the strided view it is: the merge takes the candidate strides
        # explicitly, so compacting it here would only add a copy.
        return gathered.view(world, *keys.shape)[:, lo : lo + owned]

    # The all-to-all needs no such slice: it splits dim 0 into `world` chunks
    # and sends chunk j to rank j, and chunk j is exactly rank j's run of
    # heads, so each rank is sent only what it keeps. What comes back is
    # source-rank major over that run.
    received = torch.empty_like(keys)
    dist.all_to_all_single(received, keys, group=group.device_group)
    return received.view(world, owned, *keys.shape[1:])
