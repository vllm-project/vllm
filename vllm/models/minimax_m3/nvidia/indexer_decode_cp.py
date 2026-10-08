# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Context-parallel MSA indexer decode for MiniMax-M3 (opt-in).

Default decode (``indexer_msa.py``): MiniMax-M3 has 4 index-query heads and one
index-K head. The index-K projection is replicated on every TP rank, so every
rank stores the full index-K cache. Each rank scores its own index head(s)
(sharded like the KV heads) over the whole context, then picks that head's
top-16 blocks. At TP4 every rank therefore reads all of the index-K once per
sparse layer per decode step, and that read grows with the context.

With ``VLLM_MINIMAX_M3_INDEXER_DECODE_CP=1`` the context is split across the
TP group instead of the heads. Per sparse layer, for decode-only batches:

1. all-gather the index queries, so every rank holds all index heads;
2. score: rank ``r`` scores only blocks ``j * tp + r`` for all heads
   (``index_decode_score.py``, ``cp_size``/``cp_rank``) into a compact
   ``[T, H, max_k_tiles / tp]`` buffer, so it reads ``1/tp`` of the index-K;
3. local top-16 per (token, head) over those blocks, as packed int64 keys;
4. all-gather the candidates (``16 * tp`` per (token, head));
5. each rank merges the candidates of its own head(s) into the top-16 block
   ids, ascending and ``-1`` padded: the ``sparse_topk_select`` output
   contract, written into the shared ``topk_indices_buffer``.

Exactness: the scores are bitwise the default ones (same kernel, same MMA
order, same block). Keys order by (score desc, block id asc); forced blocks
(``sparse_init_block`` / ``sparse_local_block``) get score FLT_MAX, as in
``sparse_topk_select``. Under a total order the top-16 of a union is the top-16
of the per-part top-16s, so the merge is the exact top-16. That equals the
default selection whenever no equal scores straddle the 16th place. On such
exact fp32 ties the default ``sparse_topk_select`` picks in an order that is
not specified; here the lower block id wins, deterministically.

Collectives: both all-gathers move raw bytes viewed as fp32 through
``sp_all_gather`` (the one-shot custom all-gather when available, else NCCL)
on the forward stream. Every rank takes the same branch: the gate depends only
on the env, the parallel layout and the batch's decode/prefill token counts.

CUDA graphs: fixed shapes per decode size; the buffers are persistent and
shared by all layers; there is no host sync.
"""

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.models.common.ops.sequence_parallel import sp_all_gather
from vllm.models.minimax_m3.nvidia.ops import minimax_m3_index_decode_score_cutedsl
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

TOPK = 16
# TP sizes served. Each rank's compact row (max_k_tiles / tp tiles) is split
# into NUM_GROUPS power-of-two groups for the two-level local top-k.
SUPPORTED_TP_SIZES = (4, 8)
NUM_GROUPS = 128
LOCAL_NUM_WARPS = 4
MERGE_NUM_WARPS = 1


@triton.jit
def _tile_keys(
    row_ptr,
    j,
    n_local,
    force_end_start,
    CP: tl.constexpr,
    CP_RANK: tl.constexpr,
    FORCE_BEGIN: tl.constexpr,
):
    """Packed keys of compact tiles ``j`` (global block ``j * CP + CP_RANK``).

    Key = ``1 << 48 | ordered(score) << 16 | (0xFFFF - block)``: a larger key
    is a larger score, then a lower block id. Tiles outside ``[0, n_local)``
    are not read and get key 0.
    """
    valid = j < n_local
    k = j * CP + CP_RANK
    score = tl.load(row_ptr + j, mask=valid, other=0.0)
    forced = valid & ((k < FORCE_BEGIN) | (k >= force_end_start))
    score = tl.where(forced, 3.4028234663852886e38, score)  # FLT_MAX
    bits = score.to(tl.uint32, bitcast=True)
    ordered = bits ^ tl.where((bits >> 31) != 0, 0xFFFFFFFF, 0x80000000)
    key = (1 << 48) | (ordered.to(tl.int64) << 16) | (0xFFFF - k).to(tl.int64)
    return tl.where(valid, key, 0)


@triton.jit
def _local_topk_keys_kernel(
    score_ptr,  # [T, H, W] fp32 compact scores, tile dim contiguous
    nvp_ptr,  # [T] int32 causal page count per token (global block ids)
    key_ptr,  # [T, H, TOPK] int64 out
    num_heads,
    max_k_tiles,
    stride_s_t,
    stride_s_h,
    stride_k_t,
    stride_k_h,
    CP: tl.constexpr,
    CP_RANK: tl.constexpr,
    FORCE_BEGIN: tl.constexpr,
    FORCE_END: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP: tl.constexpr,
):
    """Exact top-``TOPK`` packed keys of this rank's tiles, in two stages.

    Keys are unique (block id in the low bits), so the top-``TOPK`` keys of a
    row lie in the ``TOPK`` groups with the largest group maxima: a top key in
    a group ranked below ``TOPK`` other groups would have ``TOPK`` larger keys
    (those groups' maxima) above it. Stage 1 takes every group's max key and
    the top-``TOPK`` of those; stage 2 the top-``TOPK`` of the selected groups'
    tiles. Rows with fewer than ``TOPK`` valid tiles are padded with key 0.
    ``NUM_GROUPS * GROUP`` covers the whole compact row.
    """
    row = tl.program_id(0)  # row = t * H + h
    t = row // num_heads
    h = row - t * num_heads

    # Same clamp and forced window as sparse_topk_select.
    nvp = tl.load(nvp_ptr + t)
    nvp = tl.minimum(tl.maximum(nvp, 0), max_k_tiles)
    force_end_start = tl.where(nvp >= FORCE_END, nvp - FORCE_END, 0)
    # Owned blocks below nvp: j * CP + CP_RANK < nvp.
    n_local = (nvp + (CP - 1 - CP_RANK)) // CP

    row_ptr = score_ptr + t.to(tl.int64) * stride_s_t + h * stride_s_h
    g = tl.arange(0, NUM_GROUPS)
    s = tl.arange(0, GROUP)
    # Stage 1: the max key of every group of GROUP consecutive tiles.
    keys = _tile_keys(
        row_ptr,
        g[:, None] * GROUP + s[None, :],
        n_local,
        force_end_start,
        CP,
        CP_RANK,
        FORCE_BEGIN,
    )
    top_g = tl.topk(tl.max(keys, axis=1), TOPK)
    # Group of each selected max: block k = j * CP + CP_RANK, group j // GROUP.
    # A 0 max is an empty group; send it past the row so nothing is read.
    blk = (0xFFFF - (top_g & 0xFFFF)).to(tl.int32)
    grp = tl.where(top_g > 0, ((blk - CP_RANK) // CP) // GROUP, NUM_GROUPS)
    # Stage 2: the top-TOPK of the selected groups' tiles.
    cand = _tile_keys(
        row_ptr,
        grp[:, None] * GROUP + s[None, :],
        n_local,
        force_end_start,
        CP,
        CP_RANK,
        FORCE_BEGIN,
    )
    best = tl.topk(tl.reshape(cand, [TOPK * GROUP]), TOPK)

    out = key_ptr + t.to(tl.int64) * stride_k_t + h * stride_k_h + tl.arange(0, TOPK)
    tl.store(out, best)


@triton.jit
def _merge_topk_kernel(
    key_ptr,  # [CP, T, H, TOPK] int64 gathered keys
    out_ptr,  # [T, H_loc, TOPK] int32 block ids
    num_local_heads,
    head_base,
    stride_k_p,
    stride_k_t,
    stride_k_h,
    stride_o_t,
    stride_o_h,
    stride_o_k,
    CP: tl.constexpr,
    TOPK: tl.constexpr,
):
    row = tl.program_id(0)  # row = t * H_loc + h_loc
    t = row // num_local_heads
    hl = row - t * num_local_heads
    p = tl.arange(0, CP)
    i = tl.arange(0, TOPK)
    ptrs = (
        key_ptr
        + p[:, None].to(tl.int64) * stride_k_p
        + t.to(tl.int64) * stride_k_t
        + (head_base + hl) * stride_k_h
        + i[None, :]
    )
    keys = tl.load(ptrs)
    best = tl.topk(tl.reshape(keys, [CP * TOPK]), TOPK)
    # Unpack block ids; empty slots (key 0, only when nvp < TOPK) sort last.
    blk = tl.where(best > 0, 0xFFFF - (best & 0xFFFF), 0x7FFFFFFF).to(tl.int32)
    blk = tl.sort(blk)
    blk = tl.where(blk == 0x7FFFFFFF, -1, blk)
    out = out_ptr + t.to(tl.int64) * stride_o_t + hl * stride_o_h + i * stride_o_k
    tl.store(out, blk)


def _is_pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def _local_groups(width: int) -> tuple[int, int] | None:
    """(groups, tiles per group) covering a compact row of ``width``, or None."""
    num_groups = min(NUM_GROUPS, width)
    if not (_is_pow2(width) and num_groups >= TOPK):
        return None
    return num_groups, width // num_groups


def local_topk_keys(
    scores: torch.Tensor,
    num_valid_pages: torch.Tensor,
    keys: torch.Tensor,
    *,
    cp_size: int,
    cp_rank: int,
    max_k_tiles: int,
    force_begin: int,
    force_end: int,
) -> None:
    """Step 3: top-16 packed keys per (token, head) over this rank's blocks.

    ``scores`` is ``[T, H, >= max_k_tiles / cp_size]`` with compact column
    ``j`` holding global block ``j * cp_size + cp_rank``; only columns below
    the token's owned block count are read.
    """
    num_tokens, num_heads, _ = scores.shape
    assert scores.dtype == torch.float32 and scores.stride(2) == 1
    assert keys.dtype == torch.int64 and keys.stride(2) == 1
    assert tuple(keys.shape) == (num_tokens, num_heads, TOPK)
    assert num_valid_pages.dtype == torch.int32
    assert max_k_tiles <= 1 << 16  # 16-bit block id in the key
    width = max_k_tiles // cp_size  # n_local <= width
    assert width * cp_size == max_k_tiles and width <= scores.shape[2]
    groups = _local_groups(width)
    assert groups is not None, width
    _local_topk_keys_kernel[(num_tokens * num_heads,)](
        scores,
        num_valid_pages,
        keys,
        num_heads,
        max_k_tiles,
        scores.stride(0),
        scores.stride(1),
        keys.stride(0),
        keys.stride(1),
        CP=cp_size,
        CP_RANK=cp_rank,
        FORCE_BEGIN=force_begin,
        FORCE_END=force_end,
        TOPK=TOPK,
        NUM_GROUPS=groups[0],
        GROUP=groups[1],
        num_warps=LOCAL_NUM_WARPS,
    )


def merge_topk(
    gathered_keys: torch.Tensor,
    out: torch.Tensor,
    *,
    head_base: int,
) -> None:
    """Step 5: top-16 block ids of heads ``head_base + [0, H_loc)`` into out."""
    cp_size, num_tokens, _, topk = gathered_keys.shape
    assert topk == TOPK and gathered_keys.stride(3) == 1
    assert out.dtype == torch.int32 and out.shape[0] == num_tokens
    num_local_heads = out.shape[1]
    _merge_topk_kernel[(num_tokens * num_local_heads,)](
        gathered_keys,
        out,
        num_local_heads,
        head_base,
        gathered_keys.stride(0),
        gathered_keys.stride(1),
        gathered_keys.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        CP=cp_size,
        TOPK=TOPK,
        num_warps=MERGE_NUM_WARPS,
    )


def _score_q_view_ok(q: torch.Tensor) -> bool:
    """Whether the CuteDSL score kernel accepts this strided q without a copy.

    Its compiled signature (``make_fake_tensor(..., divisibility=16)``) needs a
    unit last-dim stride, the other strides divisible by 16 elements and a
    16-element-aligned base. The TMA load reads the same bytes either way.
    """
    return (
        q.stride(2) == 1
        and q.stride(0) % 16 == 0
        and q.stride(1) % 16 == 0
        and q.data_ptr() % (16 * q.element_size()) == 0
    )


class IndexerDecodeCP:
    """Per-process state shared by every sparse layer (buffers, layout, gate)."""

    def __init__(
        self,
        *,
        cp_size: int,
        cp_rank: int,
        num_heads: int,
        num_local_heads: int,
        max_rows: int,
        max_k_tiles: int,
        min_tokens: int,
        device: torch.device,
    ) -> None:
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.num_heads = num_heads
        self.num_local_heads = num_local_heads
        # TP > num index heads replicates each head on tp / num_heads ranks.
        self.head_replicas = max(1, cp_size // num_heads)
        self.head_base = (cp_rank // self.head_replicas) * num_local_heads
        # After dropping the replicas the gathered queries hold every head
        # exactly once (index_q is sharded like the KV heads).
        assert (cp_size // self.head_replicas) * num_local_heads == num_heads
        self.max_rows = max_rows
        self.max_k_tiles = max_k_tiles
        self.width = max_k_tiles // cp_size
        self.min_tokens = min_tokens
        # Unwritten tiles are never read (only j < n_local); -inf for safety.
        self.scores = torch.full(
            (max_rows, num_heads, self.width),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
        self.keys = torch.zeros(
            (max_rows, num_heads, TOPK), dtype=torch.int64, device=device
        )

    def applies(self, num_decode_tokens: int, max_decode_query_len: int) -> bool:
        return (
            self.min_tokens <= num_decode_tokens <= self.max_rows
            # CuteDSL score kernel's flattened Q tile (heads x query len).
            and self.num_heads * max_decode_query_len <= 32
        )

    def forward(
        self,
        index_q: torch.Tensor,  # [nd, H_loc, D], this rank's heads
        index_k_cache: torch.Tensor,
        decode_md,  # MiniMaxM3IndexerDecodeMetadata
        num_valid_pages: torch.Tensor,  # [nd] int32
        out: torch.Tensor,  # [nd, H_loc, TOPK] int32
        force_begin: int,
        force_end: int,
    ) -> None:
        nd, h_loc, dim = index_q.shape
        cp, nh = self.cp_size, self.num_heads

        # 1. All index-query heads on every rank (raw bytes, viewed as fp32).
        q_recv = sp_all_gather(index_q.reshape(nd, h_loc * dim).view(torch.float32))
        q_recv = q_recv.view(index_q.dtype).view(cp, nd, h_loc, dim)
        if self.head_replicas > 1:
            q_recv = q_recv[:: self.head_replicas]
        # [nd, H, D] as a strided view of the gathered bytes; the score kernel
        # reads it in place through its TMA descriptor when the strides allow.
        q_all = q_recv.permute(1, 0, 2, 3).reshape(nd, nh, dim)
        if not _score_q_view_ok(q_all):
            q_all = q_all.contiguous()

        # 2. Score this rank's blocks (j * cp + rank) for all heads.
        scores = self.scores[:nd]
        minimax_m3_index_decode_score_cutedsl(
            q_all,
            index_k_cache,
            decode_md.block_table,
            decode_md.seq_lens,
            decode_md.max_seq_len,
            force_begin,
            force_end,
            1,
            decode_md.decode_query_len,
            decode_md.max_decode_query_len,
            score_out=scores.transpose(0, 1),
            cp_size=cp,
            cp_rank=self.cp_rank,
        )

        # 3. Local top-16 candidates.
        keys = self.keys[:nd]
        local_topk_keys(
            scores,
            num_valid_pages,
            keys,
            cp_size=cp,
            cp_rank=self.cp_rank,
            max_k_tiles=self.max_k_tiles,
            force_begin=force_begin,
            force_end=force_end,
        )

        # 4. Every rank's candidates (raw bytes, viewed as fp32).
        k_recv = sp_all_gather(keys.view(nd, nh * TOPK).view(torch.float32))
        k_recv = k_recv.view(torch.int64).view(cp, nd, nh, TOPK)

        # 5. Exact top-16 of this rank's heads.
        merge_topk(k_recv, out, head_base=self.head_base)


_STATE: IndexerDecodeCP | None = None
_RESOLVED = False


def get_indexer_decode_cp(
    *, num_local_heads: int, max_k_tiles: int, device: torch.device | None = None
) -> IndexerDecodeCP | None:
    """The process's decode-CP state, or None (default path). Built once."""
    global _STATE, _RESOLVED
    if _RESOLVED:
        return _STATE
    _RESOLVED = True
    if not envs.VLLM_MINIMAX_M3_INDEXER_DECODE_CP:
        return None

    cp = get_tensor_model_parallel_world_size()
    vllm_config = get_current_vllm_config()
    hf_config = vllm_config.model_config.hf_config
    text_config = getattr(hf_config, "text_config", hf_config)
    sparse_cfg = text_config.sparse_attention_config
    num_heads = sparse_cfg["sparse_num_index_heads"]
    spec = vllm_config.speculative_config
    max_query_len = 1 + (spec.num_speculative_tokens if spec is not None else 0)
    sched = vllm_config.scheduler_config
    max_rows = min(sched.max_num_seqs * max_query_len, sched.max_num_batched_tokens)
    parallel = vllm_config.parallel_config

    why = None
    if cp not in SUPPORTED_TP_SIZES:
        why = f"tensor parallel size {cp} (needs one of {SUPPORTED_TP_SIZES})"
    elif parallel.decode_context_parallel_size > 1:
        why = "decode context parallelism is enabled"
    elif sparse_cfg["sparse_topk_blocks"] != TOPK:
        why = f"sparse_topk_blocks {sparse_cfg['sparse_topk_blocks']} != {TOPK}"
    elif (
        num_heads not in (1, 2, 4, 8)
        or (cp // max(1, cp // num_heads)) * num_local_heads != num_heads
    ):
        why = f"{num_heads} index heads, {num_local_heads} per rank"
    elif (
        max_k_tiles % cp
        or max_k_tiles > 1 << 16
        or _local_groups(max_k_tiles // cp) is None
    ):
        why = f"max_k_tiles {max_k_tiles}"
    elif num_heads * max_query_len > 32:
        why = f"{num_heads} index heads x query len {max_query_len} > 32"
    if why is not None:
        logger.warning(
            "VLLM_MINIMAX_M3_INDEXER_DECODE_CP=1 ignored: %s; using the default "
            "indexer decode path.",
            why,
        )
        return None

    min_tokens = max(1, envs.VLLM_MINIMAX_M3_INDEXER_DECODE_CP_MIN_TOKENS)
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    _STATE = IndexerDecodeCP(
        cp_size=cp,
        cp_rank=get_tensor_model_parallel_rank(),
        num_heads=num_heads,
        num_local_heads=num_local_heads,
        max_rows=max_rows,
        max_k_tiles=max_k_tiles,
        min_tokens=min_tokens,
        device=device,
    )
    logger.info_once(
        "MiniMax-M3 MSA indexer: decode context parallelism enabled: each of %d "
        "TP ranks scores 1/%d of the index-K blocks for all %d index heads, "
        "then the ranks merge the top-%d; decode-only batches of %d..%d tokens.",
        cp,
        cp,
        num_heads,
        TOPK,
        min_tokens,
        max_rows,
    )
    return _STATE
