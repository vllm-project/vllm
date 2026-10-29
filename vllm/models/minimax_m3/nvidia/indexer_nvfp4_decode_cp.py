# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Context-parallel decode for the MiniMax M3 NVFP4 indexer (opt-in, TP4).

At TP4 each rank owns one of the 4 index heads, but the index-K cache is
replicated, so every rank reads the whole cache per sparse layer. With
``--attention-config '{"indexer_decode_cp": true}'``, rank ``r``
instead:

1. exchanges index-q heads so it holds all 4 (``_q_exchange_kernel``);
2. scores all 4 heads over pages ``j * CP + r`` with the stock Q8KV4 kernel
   on a compact block table, reading 1/CP of the index-K;
3. takes the exact local top 16 per (query, head) and sends it to the head's
   owner (``_local_topk_push_kernel``);
4. merges the owner's ``16 * CP`` candidates into the indexer output: 15
   pages by (score desc, page asc), then the own page (``_merge_kernel``).

The merge is exact: the global top 16 lies in the union of per-rank top 16s.
The stock decode ranks 16-bit quantized scores, so the two may differ at ties.

The chunking is written against one contract, the MSA Q8KV4 decode scorer's:
eight query slots per request, ``(seq_len - 1) / 128`` pages scored, and only
the pages below a query's own page stored. The three places that depend on it
are ``_compact_block_table_kernel`` (the length it is handed),
``_local_topk_push_kernel`` (the cut that length drops) and ``_set_width``
(room for the page that length invents).

Both exchanges are NVLink stores into symmetric buffers whose payload is the
flag (``0xFF`` E4M3 NaN for queries, ``-1`` for keys); consumers restore it.
Buffers are per layer, shapes are fixed per batch size, and nothing syncs with
the host, so the path is CUDA-graph safe. It needs symmetric memory across the
tensor-parallel ranks and falls back to the stock decode without it.

The three Triton kernels are PDL-launched and wait for their producer, so
each starts its prologue in the tail of the kernel before it. Only the push
releases its dependent early, right after its own wait, which is what lets
the merge overlap it; the exchange must not, because the score kernel after
it reads the whole tile.
"""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops.cp_common import direct_cp_enabled

logger = init_logger(__name__)

PAGE_SIZE = 128
HEAD_DIM = 128
TOPK_BLOCKS = 16
DECODE_QUERY_LEN = 8  # query slots per request in a Q8KV4 decode tile
MAX_PAGES = 8192
NUM_INDEX_HEADS = 4
SUPPORTED_TP_SIZES = (4,)
# Rows with at most this many ranked pages take a single top-k.
TOPK_CHUNK = 256
# Groups of the two-level top-k over longer rows.
MAX_TOPK_GROUPS = 128

_TOPK = tl.constexpr(TOPK_BLOCKS)
_RANKED = tl.constexpr(TOPK_BLOCKS - 1)
_PAGE = tl.constexpr(PAGE_SIZE)
_SLOTS = tl.constexpr(DECODE_QUERY_LEN)
_WORDS = tl.constexpr(HEAD_DIM // 4)  # int32 words per index-q head


@triton.jit
def _peer_ptr(ptrs, rank, dtype: tl.constexpr):
    return tl.load(ptrs + rank).to(tl.pointer_type(dtype))


@triton.jit
def _compact_block_table_kernel(
    bt_ptr,  # [B, P] int32
    seq_lens_ptr,  # [B] int32
    cbt_ptr,  # [B, W] int32 out: compact page j -> global page j * CP + RANK
    cseq_ptr,  # [B] int32 out: lengths the score kernel sees
    seq_out_ptr,  # [B] int32 out: real lengths (stable buffer)
    P,
    W,
    stride_bt,
    stride_cbt,
    CP: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Compact page j = global page ``j * CP + RANK``, and the length the score
    kernel is handed for it.

    That length is a lie: the scorer stores only the pages below a query's own
    page, so ``n_local * 128 + 8`` puts all eight query slots one page past
    the chunk and stores every owned page, uncausally. The real cut comes back
    in ``_local_topk_push_kernel``. ``W`` keeps the invented page addressable,
    since a scorer that clamps the own page to its last score column would
    drop the chunk's last page."""
    b = tl.program_id(0)
    seq = tl.load(seq_lens_ptr + b)
    tl.store(seq_out_ptr + b, seq)
    hist = tl.minimum(tl.maximum((seq - 1) // _PAGE, 0), P)
    n_local = tl.minimum((hist + (CP - 1 - RANK)) // CP, W - 1)
    for start in range(0, W, BLOCK):
        j = start + tl.arange(0, BLOCK)
        src = tl.minimum(j * CP + RANK, P - 1)
        v = tl.load(bt_ptr + b.to(tl.int64) * stride_bt + src, mask=j < W)
        tl.store(cbt_ptr + b.to(tl.int64) * stride_cbt + j, v, mask=j < W)
    tl.store(cseq_ptr + b, n_local * _PAGE + _SLOTS)


@triton.jit
def _has_ff_byte(v):
    u = v.to(tl.uint32, bitcast=True)
    ff = (
        ((u & 0xFF) == 0xFF)
        | (((u >> 8) & 0xFF) == 0xFF)
        | (((u >> 16) & 0xFF) == 0xFF)
        | ((u >> 24) == 0xFF)
    )
    return tl.max(ff.to(tl.int32), axis=0) > 0


@triton.jit
def _q_exchange_kernel(
    q_ptr,  # [rows, 32] int32 view of this rank's e4m3 index-q head
    tile_ptrs,  # [CP] int64: q tile [B, 8, H, 128] bytes of every rank
    H: tl.constexpr,
    CP: tl.constexpr,
    RANK: tl.constexpr,
    HEAD: tl.constexpr,
    QLEN: tl.constexpr,
):
    """Store this rank's index-q head into every rank's tile, then wait for
    the other heads. The PDL wait is for index-q's producer."""
    tl.extra.cuda.gdc_wait()
    t = tl.program_id(0)
    b = t // QLEN
    slot = (_SLOTS - QLEN) + t % QLEN
    w = tl.arange(0, _WORDS)
    row = tl.load(q_ptr + t * _WORDS + w)
    dst_off = ((b * _SLOTS + slot) * H + HEAD) * _WORDS + w
    for dst in tl.static_range(CP):
        tl.store(_peer_ptr(tile_ptrs, dst, tl.int32) + dst_off, row)
    local = _peer_ptr(tile_ptrs, RANK, tl.int32)
    hw = tl.arange(0, H * _WORDS)
    off = (b * _SLOTS + slot) * H * _WORDS + hw
    v = tl.load(local + off, volatile=True)
    pending = _has_ff_byte(v)
    while pending:
        v = tl.load(local + off, volatile=True)
        pending = _has_ff_byte(v)


@triton.jit
def _page_keys(row_ptr, j, n, CP: tl.constexpr, RANK: tl.constexpr):
    """Key ``1 << 48 | ordered(score) << 16 | (0xFFFF - page)`` for ``j < n``,
    else 0. Unique; orders by score desc, then page asc."""
    valid = j < n
    page = j * CP + RANK
    score = tl.load(row_ptr + j, mask=valid, other=0.0)
    bits = score.to(tl.uint32, bitcast=True)
    ordered = bits ^ tl.where((bits >> 31) != 0, 0xFFFFFFFF, 0x80000000)
    key = (1 << 48) | (ordered.to(tl.int64) << 16) | (0xFFFF - page).to(tl.int64)
    return tl.where(valid, key, 0)


@triton.jit
def _local_topk_push_kernel(
    score_ptr,  # [B, 8 * H, W] fp32 compact scores (row = slot * H + head)
    seq_lens_ptr,  # [B] int32
    recv_ptrs,  # [CP] int64: candidates [CP, rows_max, 16] int64 of every rank
    rows_max,
    stride_sb,
    stride_sr,
    H: tl.constexpr,
    CP: tl.constexpr,
    RANK: tl.constexpr,
    QLEN: tl.constexpr,
    WIDTH: tl.constexpr,
    CHUNK: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP: tl.constexpr,
):
    """Per (query, head): exact top 16 of the owned pages, sent to the head's
    owner. Long rows use a two-level select (top 16 groups, then top 16).

    PDL-launched: the wait is for the scores, and releasing the merge only
    after it keeps the merge's own wait a guarantee that the score kernel has
    read the q tile."""
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0)
    h = tl.program_id(1)
    b = t // QLEN
    i = t % QLEN
    slot = (_SLOTS - QLEN) + i
    own = (tl.load(seq_lens_ptr + b) - QLEN + i) // _PAGE
    if own + 1 > _TOPK:
        # The causal cut the compact length dropped: the rank's pages below
        # the query's own page, bounded by the width like the scored pages (a
        # length past the page table ranks what it holds, as the stock TopK).
        n_rank = tl.minimum((own + (CP - 1 - RANK)) // CP, WIDTH - 1)
        row_ptr = score_ptr + b.to(tl.int64) * stride_sb + (slot * H + h) * stride_sr
        if n_rank <= CHUNK:
            j = tl.arange(0, CHUNK)
            best = tl.topk(_page_keys(row_ptr, j, n_rank, CP, RANK), _TOPK)
        else:
            g = tl.arange(0, NUM_GROUPS)
            s = tl.arange(0, GROUP)
            keys = _page_keys(
                row_ptr, g[:, None] * GROUP + s[None, :], n_rank, CP, RANK
            )
            top_g = tl.topk(tl.max(keys, axis=1), _TOPK)
            page = (0xFFFF - (top_g & 0xFFFF)).to(tl.int32)
            grp = tl.where(top_g > 0, ((page - RANK) // CP) // GROUP, NUM_GROUPS)
            cand = _page_keys(
                row_ptr, grp[:, None] * GROUP + s[None, :], n_rank, CP, RANK
            )
            best = tl.topk(tl.reshape(cand, [_TOPK * GROUP]), _TOPK)
        k = tl.arange(0, _TOPK)
        dst = _peer_ptr(recv_ptrs, h, tl.int64)
        tl.store(dst + (RANK * rows_max + t) * _TOPK + k, best)


@triton.jit
def _merge_kernel(
    seq_lens_ptr,  # [B] int32
    recv_ptrs,  # [CP] int64
    tile_ptrs,  # [CP] int64
    out_ptr,  # [rows, 1, 16] int32
    rows_max,
    out_stride,
    H: tl.constexpr,
    CP: tl.constexpr,
    RANK: tl.constexpr,
    QLEN: tl.constexpr,
):
    """Merge all ranks' candidates for this rank's head and restore sentinels.

    PDL-launched after every push program has started, so polling here never
    blocks a push."""
    t = tl.program_id(0)
    b = t // QLEN
    i = t % QLEN
    own = (tl.load(seq_lens_ptr + b) - QLEN + i) // _PAGE
    k = tl.arange(0, _TOPK)
    out = out_ptr + t.to(tl.int64) * out_stride + k
    if own + 1 <= _TOPK:
        tl.store(out, tl.where(k < own + 1, k, -1))
        # No push for this row: wait for the score before resetting the q tile.
        tl.extra.cuda.gdc_wait()
    else:
        recv = _peer_ptr(recv_ptrs, RANK, tl.int64)
        p = tl.arange(0, CP)
        off = (p[:, None] * rows_max + t) * _TOPK + k[None, :]
        keys = tl.load(recv + off, volatile=True)
        pending = tl.min(tl.min(keys, axis=1), axis=0) == -1
        while pending:
            keys = tl.load(recv + off, volatile=True)
            pending = tl.min(tl.min(keys, axis=1), axis=0) == -1
        best = tl.topk(tl.reshape(keys, [CP * _TOPK]), _TOPK)
        page = (0xFFFF - (best & 0xFFFF)).to(tl.int32)
        tl.store(out, tl.where(k == _RANKED, own, page))
        tl.store(recv + off, tl.full([CP, _TOPK], -1, tl.int64))
    # The score kernel has read this q tile row; re-arm it.
    slot = (_SLOTS - QLEN) + i
    tile = _peer_ptr(tile_ptrs, RANK, tl.int32)
    hw = tl.arange(0, H * _WORDS)
    tl.store(
        tile + (b * _SLOTS + slot) * H * _WORDS + hw,
        tl.full(hw.shape, -1, tl.int32),
    )


@dataclass
class Nvfp4DecodeCPPlan:
    """Per padded batch size: the 4-head score wrapper and its buffers."""

    # The wrapper owns the compact [B, 8 * H, width] fp32 scores.
    wrapper: Any  # BatchDecodeIndexerQ8KV4Wrapper(num_heads=4), CUDA-graph mode
    block_table: torch.Tensor  # [B, width] compact pages
    score_seq_lens: torch.Tensor  # [B] lengths the score kernel sees
    seq_lens: torch.Tensor  # [B] real lengths
    num_reqs: int


@dataclass
class Nvfp4DecodeCPLayer:
    """One sparse layer's symmetric buffers."""

    # Keep both alive: freeing one unmaps it while peers still write to it.
    tile: torch.Tensor  # local q tile, int32 words
    recv: torch.Tensor  # local candidates [CP, rows_max, 16] int64
    tile_ptrs: torch.Tensor  # [CP] int64
    recv_ptrs: torch.Tensor  # [CP] int64


class Nvfp4IndexerDecodeCP:
    """Per-process state shared by the builder and every sparse layer."""

    def __init__(self, *, max_reqs: int, device: torch.device) -> None:
        from vllm.third_party.fmha_sm100.sparse import BatchDecodeIndexerQ8KV4Wrapper

        self._wrapper_cls = BatchDecodeIndexerQ8KV4Wrapper
        self.cp = get_tensor_model_parallel_world_size()
        self.rank = get_tensor_model_parallel_rank()
        self.head = self.rank  # one index head per rank at TP4
        # Compact page width, fixed by the first plan() (see _set_width).
        self.width = 0
        self.num_groups = 0
        self.group = 0
        self.max_reqs = max_reqs
        self.rows_max = max_reqs * DECODE_QUERY_LEN
        self.device = device
        self._plans: dict[int, Nvfp4DecodeCPPlan] = {}
        self._tile_words = max_reqs * DECODE_QUERY_LEN * NUM_INDEX_HEADS * HEAD_DIM // 4
        self._recv_words = self.cp * self.rows_max * TOPK_BLOCKS

    def new_layer(self) -> Nvfp4DecodeCPLayer:
        """Collective: every TP rank builds its sparse layers in the same order."""
        import torch.distributed._symmetric_memory as symm_mem

        group_name = get_tp_group().device_group.group_name
        tile = symm_mem.empty(self._tile_words, dtype=torch.int32, device=self.device)
        tile.fill_(-1)
        recv = symm_mem.empty(self._recv_words, dtype=torch.int64, device=self.device)
        recv.fill_(-1)
        tile_ptrs = symm_mem.rendezvous(tile, group_name).buffer_ptrs
        recv_ptrs = symm_mem.rendezvous(recv, group_name).buffer_ptrs
        return Nvfp4DecodeCPLayer(
            tile=tile,
            recv=recv,
            tile_ptrs=torch.tensor(tile_ptrs, dtype=torch.int64, device=self.device),
            recv_ptrs=torch.tensor(recv_ptrs, dtype=torch.int64, device=self.device),
        )

    def _set_width(self, max_pages: int) -> None:
        """Fix the compact page width and the two-level top-k shape.

        One column past the pages a rank can own, so the page the compact
        length invents stays inside the score buffer and the chunk's last page
        is stored (see ``_compact_block_table_kernel``). Both levels of the
        top-k select over power-of-two tiles that cover the width.
        """
        width = cdiv(min(max_pages, MAX_PAGES), self.cp) + 1
        if self.width == 0:
            padded = 1 << (width - 1).bit_length()
            self.width = width
            self.group = max(1, padded // MAX_TOPK_GROUPS)
            self.num_groups = padded // self.group
        assert width <= self.width, "the block table grew after the first plan"

    def plan(
        self, block_table: torch.Tensor, seq_lens: torch.Tensor
    ) -> Nvfp4DecodeCPPlan:
        """Outside CUDA graph capture, once per step."""
        self._set_width(block_table.shape[1])
        num_reqs = seq_lens.shape[0]
        p = self._plans.get(num_reqs)
        if p is None:
            assert num_reqs <= self.max_reqs
            i32 = dict(dtype=torch.int32, device=self.device)
            bt = torch.empty(num_reqs, self.width, **i32)
            score_seq_lens = torch.empty(num_reqs, **i32)
            p = self._plans[num_reqs] = Nvfp4DecodeCPPlan(
                wrapper=self._wrapper_cls(
                    num_heads=NUM_INDEX_HEADS,
                    use_cuda_graph=True,
                    block_table_buffer=bt,
                    seq_lens_buffer=score_seq_lens,
                ),
                block_table=bt,
                score_seq_lens=score_seq_lens,
                seq_lens=torch.empty(num_reqs, **i32),
                num_reqs=num_reqs,
            )
        _compact_block_table_kernel[(num_reqs,)](
            block_table,
            seq_lens,
            p.block_table,
            p.score_seq_lens,
            p.seq_lens,
            min(block_table.shape[1], MAX_PAGES),
            self.width,
            block_table.stride(0),
            p.block_table.stride(0),
            CP=self.cp,
            RANK=self.rank,
            BLOCK=1024,
        )
        p.wrapper.plan(p.block_table, p.score_seq_lens)
        return p

    def forward(
        self,
        layer: Nvfp4DecodeCPLayer,
        plan: Nvfp4DecodeCPPlan,
        index_q: torch.Tensor,  # [rows, 1, 128] e4m3, this rank's head
        k_cache: torch.Tensor,
        out: torch.Tensor,  # [rows, 1, 16] int32
        query_len: int,
    ) -> None:
        rows = index_q.shape[0]
        assert rows == plan.num_reqs * query_len and out.shape[0] == rows
        _q_exchange_kernel[(rows,)](
            index_q.view(torch.int32).view(rows, HEAD_DIM // 4),
            layer.tile_ptrs,
            H=NUM_INDEX_HEADS,
            CP=self.cp,
            RANK=self.rank,
            HEAD=self.head,
            QLEN=query_len,
            num_warps=1,
            launch_pdl=True,
        )
        q_tile = layer.tile[
            : plan.num_reqs * DECODE_QUERY_LEN * NUM_INDEX_HEADS * HEAD_DIM // 4
        ]
        scores = plan.wrapper._run_scores(
            q_tile.view(torch.float8_e4m3fn).view(-1, NUM_INDEX_HEADS, HEAD_DIM),
            k_cache,
        )
        _local_topk_push_kernel[(rows, NUM_INDEX_HEADS)](
            scores,
            plan.seq_lens,
            layer.recv_ptrs,
            self.rows_max,
            scores.stride(0),
            scores.stride(1),
            H=NUM_INDEX_HEADS,
            CP=self.cp,
            RANK=self.rank,
            QLEN=query_len,
            WIDTH=self.width,
            CHUNK=TOPK_CHUNK,
            NUM_GROUPS=self.num_groups,
            GROUP=self.group,
            num_warps=4,
            launch_pdl=True,
        )
        _merge_kernel[(rows,)](
            plan.seq_lens,
            layer.recv_ptrs,
            layer.tile_ptrs,
            out,
            self.rows_max,
            out.stride(0),
            H=NUM_INDEX_HEADS,
            CP=self.cp,
            RANK=self.rank,
            QLEN=query_len,
            num_warps=1,
            launch_pdl=True,
        )


_STATE: Nvfp4IndexerDecodeCP | None = None
_RESOLVED = False


def get_nvfp4_indexer_decode_cp(
    *, num_index_heads: int, max_decode_query_len: int
) -> Nvfp4IndexerDecodeCP | None:
    """The process's decode-CP state, or None (stock decode). Built once."""
    global _STATE, _RESOLVED
    if _RESOLVED:
        return _STATE
    _RESOLVED = True
    vllm_config = get_current_vllm_config()
    if not vllm_config.attention_config.indexer_decode_cp:
        return None
    tp = get_tensor_model_parallel_world_size()
    text_config = vllm_config.model_config.hf_text_config
    total_heads = text_config.sparse_attention_config["sparse_num_index_heads"]
    why = None
    if tp not in SUPPORTED_TP_SIZES:
        why = f"tensor parallel size {tp} (needs one of {SUPPORTED_TP_SIZES})"
    elif total_heads != NUM_INDEX_HEADS or num_index_heads != 1:
        why = f"{total_heads} index heads, {num_index_heads} per rank"
    elif vllm_config.parallel_config.decode_context_parallel_size > 1:
        why = "decode context parallelism is enabled"
    elif max_decode_query_len > DECODE_QUERY_LEN:
        why = f"decode query length {max_decode_query_len} > {DECODE_QUERY_LEN}"
    elif not direct_cp_enabled(get_tp_group(), torch.int32, use_direct=None):
        why = "the tensor-parallel ranks have no symmetric memory between them"
    if why is not None:
        logger.warning(
            "indexer_decode_cp ignored: %s; using the stock NVFP4 indexer decode.",
            why,
        )
        return None
    sched = vllm_config.scheduler_config
    _STATE = Nvfp4IndexerDecodeCP(
        max_reqs=sched.max_num_seqs,
        device=torch.device("cuda", torch.accelerator.current_device_index()),
    )
    logger.info_once(
        "MiniMax M3 NVFP4 indexer: decode context parallelism over %d TP ranks "
        "(each scores 1/%d of the index-K pages for all %d index heads).",
        tp,
        tp,
        NUM_INDEX_HEADS,
    )
    return _STATE
