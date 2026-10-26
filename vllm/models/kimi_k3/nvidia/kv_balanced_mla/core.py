# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 MLA decode over the paged FP8 cache with a device-planned split-KV work list.

The kernel is TokenSpeed's FP8 MLA decode (vendored in ``kernel.py``): query rows
packed into 128-row tiles (heads x q tokens), one 2-CTA cluster per (query tile,
request, KV split), FP16 partials merged by its reduction kernel. A planner kernel
picks each request's split count from its KV length and lists exactly the planned
clusters; one wave of persistent clusters walks the list. Nothing is read on the
host, so a plan can be captured in a CUDA graph.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
import tvm_ffi
from cutlass import Float32, Int32

from vllm.triton_utils import tl, triton

from .helpers import get_mla_decode_arch, select_mla_decode_tilers
from .kernel import BlackwellMultiHeadLatentAttentionForwardFP8

LATENT, ROPE = 512, 64
KV_TILE = 128  # KV tokens per MMA tile
ROWS = 128  # query rows per tile (the 2-CTA M128 kernel)
# Most splits of one request; sizes the partial-output workspace.
CAPACITY = 8
# The kernel addresses the partials in Int32 bytes: rows x splits x LATENT x 2 B.
MAX_TILES = (1 << 31) // (ROWS * CAPACITY * LATENT * 2)
# Registers per thread of the softmax, correction and MMA / load warpgroups. The
# kernel's default (208, 256, 40) spills in the MMA warp, whose descriptor arithmetic
# then throttles the tensor core; the correction warps mostly wait and give some up.
REG_SPLIT = (200, 240, 64)
# (skip-correction threshold, P prescale), both log2. P is stored as E4M3 scaled by
# 2^prescale and the running row max lags the true one by at most 2^threshold, so
# probabilities down to 2^-(9 + prescale - threshold) of the row max survive the
# cast instead of flushing to zero as a split streams more keys. The largest scaled
# P, 2^(threshold + prescale), must stay within E4M3 (448).
P_SCALE = (2.0, 6.0)
MAX_WAVES = 8
MIN_SPLIT_TOKENS = 1024
OVERHEAD_TILES = 6


def query_tiles(heads: int, q: int) -> int:
    """Packed 128-row query tiles of one request."""
    return -(-heads * q // ROWS)


def max_planned_requests(heads: int, q: int) -> int:
    """Most requests whose partials the kernel can address."""
    return MAX_TILES // query_tiles(heads, q)


def workspace_bytes(heads: int, q: int, requests: int) -> int:
    """Split-KV scratch of ``requests``: FP16 partials and FP32 partial LSEs."""
    return requests * query_tiles(heads, q) * ROWS * CAPACITY * (LATENT * 2 + 4)


def work_list_len(heads: int, q: int, requests: int, clusters: int) -> int:
    """Entries of the work list: the planned clusters of up to ``MAX_WAVES`` waves
    (more if one split per request and query tile needs them, so no planned entry
    is ever dropped), then one wave of empty entries that ends every walk."""
    waves = max(MAX_WAVES, -(-requests * query_tiles(heads, q) // clusters))
    return (waves + 1) * clusters


def max_active_clusters(device: torch.device) -> int:
    with torch.cuda.device(device):
        return cutlass.utils.HardwareInfo().get_max_active_clusters(2)


@triton.jit
def _plan_work_list_kernel(
    seq_lens_ptr,
    counts_ptr,
    work_ptr,
    num_reqs,
    q_tiles,
    clusters,
    max_waves,
    capacity,
    min_tiles,
    overhead_tiles,
    list_len,
    KV_TILE: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    """Per-request split counts, then the work list.

    For each W = 1 .. ``max_waves`` waves of ``clusters`` (rows of a [W, B] search):
    the counts ceil(T_b / c) (T_b: KV tiles, at most ``capacity``) for the smallest
    chunk c >= ``min_tiles`` whose clusters fit W * clusters, as the kernel's
    nonempty splits; the estimated time is waves used x (longest chunk +
    ``overhead_tiles``). The fastest W wins (ties: fewer waves). The list holds each
    request's entries ``(q_tile * B + b) << 8 | split`` split-major (its query tiles
    adjacent, sharing the split's KV through L2), requests in order, then -1."""
    offs = tl.arange(0, BLOCK_B)
    mask = offs < num_reqs
    lens = tl.load(seq_lens_ptr + offs, mask, other=0)
    tiles_i = tl.where(mask, tl.maximum((lens + KV_TILE - 1) // KV_TILE, 1), 0)
    tiles = tiles_i.to(tl.float32)[None, :]
    waves_w = tl.arange(0, BLOCK_W) + 1
    budget = (waves_w * clusters).to(tl.float32)
    low = tl.zeros([BLOCK_W], tl.float32) + tl.maximum(min_tiles.to(tl.float32), 1.0)
    high = tl.maximum(tl.zeros([BLOCK_W], tl.float32) + tl.max(tiles, 1), low)
    for _ in tl.static_range(SEARCH_STEPS):
        mid = 0.5 * (low + high)
        n = tl.minimum(tl.ceil(tiles / mid[:, None]), capacity)
        used = tl.sum(tl.where(mask[None, :], n, 0.0), 1) * q_tiles
        fits = used <= budget
        high = tl.where(fits, mid, high)
        low = tl.where(fits, low, mid)
    n = tl.maximum(tl.minimum(tl.ceil(tiles / high[:, None]), capacity), 1.0)
    n = n.to(tl.int32)
    per = tl.maximum((tiles_i[None, :] + n - 1) // n, 1)  # the kernel's chunk
    n = tl.where(mask[None, :], (tiles_i[None, :] + per - 1) // per, 0)
    chunk = tl.max(tl.where(mask[None, :], per, 0), 1)
    used = tl.sum(n, 1) * q_tiles
    estimate = ((used + clusters - 1) // clusters) * (chunk + overhead_tiles)
    estimate = tl.where(waves_w <= max_waves, estimate, 1 << 30)
    best = tl.argmin(estimate, 0)
    rows = tl.arange(0, BLOCK_W)
    best_n = tl.sum(tl.where(rows[:, None] == best, n, 0), 0)
    tl.store(counts_ptr + offs, best_n, mask)

    work = best_n * q_tiles
    starts = tl.cumsum(work, 0) - work
    total = tl.sum(work, 0)
    for base in range(0, list_len, BLOCK_T):
        idx = base + tl.arange(0, BLOCK_T)
        tl.store(work_ptr + idx, -1, (idx >= total) & (idx < list_len))
    r = tl.arange(0, BLOCK_R)[None, :]
    split = r // q_tiles
    q_tile = r - split * q_tiles
    entry = ((q_tile * num_reqs + offs[:, None]) << 8) | split
    idx = starts[:, None] + r
    tl.store(
        work_ptr + idx, entry, mask[:, None] & (r < work[:, None]) & (idx < list_len)
    )


@cache
def _compile(
    heads: int,
    q: int,
    page_size: int,
    arch: tuple[int, int],
    clusters: int,
    causal: bool,
    lse: bool,
    cp: tuple[int, int, int],
):
    """The KV-balanced FP8 MLA decode for one shape: packed query rows, causal among a
    request's q rows unless ``causal`` is off, FP16 partials, BF16 latent output,
    and each row's log2 LSE with ``lse``. ``cp`` is (DCP world, rank, interleave):
    with a world above 1 the cache holds this rank's shard and the causal mask maps
    each row's global bound onto it."""
    qk_tiler, pv_tiler = select_mla_decode_tilers(
        heads, q, is_fp8=True, compute_capability=arch
    )
    assert qk_tiler == (ROWS, KV_TILE), qk_tiler
    kernel = BlackwellMultiHeadLatentAttentionForwardFP8(
        compute_capability=arch,
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=qk_tiler,
        mma_pv_tiler_mn=pv_tiler,
        max_active_clusters=clusters,
        page_size=page_size,
        skip_correction_threshold=P_SCALE[0],
        p_prescale_log2=P_SCALE[1],
        is_persistent=False,
        is_var_seq=True,
        is_var_split_kv=True,
        fold_sq_factor=1,
        is_causal=causal,
        num_heads=heads,
        seq_len_q=q,
        window_left=-1,
        cp_world=cp[0],
        cp_rank=cp[1],
        cp_interleave_size=cp[2],
        reducer_d_tiles=1,
        pack_q=True,
        reducer_max_splits=max(4, 1 << (CAPACITY - 1).bit_length()),
        partial_fp16=True,
        work_list=True,
        reg_split=REG_SPLIT,
    )
    fp8, sym = cutlass.Float8E4M3FN, cute.sym_int
    make, compact = cute.runtime.make_fake_tensor, cute.runtime.make_fake_compact_tensor
    batch, seq_q, num_heads = sym(), sym(), sym()
    latent, rope = sym(divisibility=16), sym(divisibility=16)
    pages, page_len, page_count = sym(), sym(), sym()

    def query(dim):
        return make(
            fp8,
            (batch, seq_q, num_heads, dim),
            stride=(sym(), sym(), sym(), 1),
            assumed_align=16,
        )

    def cache_view(dim):
        return make(
            fp8, (pages, page_len, dim), stride=(sym(), sym(), 1), assumed_align=16
        )

    lengths = compact(Int32, (batch,), assumed_align=4)
    args = (
        kernel,
        query(latent),
        query(rope),
        cache_view(latent),
        cache_view(rope),
        compact(Int32, (batch, page_count), stride_order=(1, 0), assumed_align=4),
        compact(
            cutlass.BFloat16,
            (batch, seq_q, num_heads, latent),
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        ),
        compact(
            Float32, (batch, seq_q, num_heads), stride_order=(2, 1, 0), assumed_align=4
        )
        if lse
        else None,
        compact(cutlass.Int8, (sym(),), assumed_align=32),  # workspace
        Int32(CAPACITY),
        lengths,  # cache lengths
        compact(Int32, (batch,), assumed_align=4),  # causal bound (global under DCP)
        compact(Int32, (batch,), assumed_align=4),  # split counts
        Float32(1.0),  # softmax scale
        Float32(1.0),  # output scale
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        False,  # PDL
        None,  # local visible lengths
        compact(Int32, (sym(),), assumed_align=4),  # work list
    )
    gpu_arch = get_mla_decode_arch(arch)
    return cute.compile(
        *args, options=f"--enable-tvm-ffi --opt-level 2 --gpu-arch={gpu_arch}a"
    )


def supports(heads: int, q: int, device: torch.device) -> bool:
    """Whether the kernel runs this shape: it needs the 2-CTA M128 tiles."""
    arch = torch.cuda.get_device_capability(device)
    tiler, _ = select_mla_decode_tilers(heads, q, is_fp8=True, compute_capability=arch)
    return tiler[0] == ROWS


class KvBalancedMla:
    """The KV-balanced MLA decode of one (heads, q, causal) shape for up to
    ``max_requests`` requests: ``plan`` once per forward, then ``decode`` per layer.

    The caller owns the plan buffers (``counts`` [max_requests], ``work``
    [list_len]) and the split-KV ``workspace`` [workspace_bytes]."""

    def __init__(
        self,
        heads: int,
        q: int,
        max_requests: int,
        causal: bool,
        device: torch.device,
    ):
        assert supports(heads, q, device)
        assert max_requests <= max_planned_requests(heads, q)
        self.heads, self.q, self.causal = heads, q, causal
        self.max_requests = max_requests
        self.arch = torch.cuda.get_device_capability(device)
        self.q_tiles = query_tiles(heads, q)
        self.clusters = max_active_clusters(device)
        self.list_len = work_list_len(heads, q, max_requests, self.clusters)
        self.max_waves = self.list_len // self.clusters - 1
        self.workspace_bytes = workspace_bytes(heads, q, max_requests)
        self.block_b = triton.next_power_of_2(max(16, max_requests))

    def plan(
        self, seq_lens: torch.Tensor, counts: torch.Tensor, work: torch.Tensor
    ) -> None:
        """Plan the split counts and the work list for ``seq_lens`` (this rank's KV
        lengths) into ``counts`` and ``work``."""
        requests = seq_lens.numel()
        assert requests <= self.max_requests
        assert counts.numel() >= self.max_requests and work.numel() >= self.list_len
        _plan_work_list_kernel[(1,)](
            seq_lens,
            counts,
            work,
            requests,
            self.q_tiles,
            self.clusters,
            self.max_waves,
            CAPACITY,
            MIN_SPLIT_TOKENS // KV_TILE,
            OVERHEAD_TILES,
            self.list_len,
            KV_TILE=KV_TILE,
            BLOCK_W=triton.next_power_of_2(self.max_waves),
            BLOCK_B=self.block_b,
            BLOCK_R=triton.next_power_of_2(CAPACITY * self.q_tiles),
            BLOCK_T=1024,
            SEARCH_STEPS=20,
        )

    def decode(
        self,
        query: torch.Tensor,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        softmax_scale: float,
        output_scale: float,
        counts: torch.Tensor,
        work: torch.Tensor,
        workspace: torch.Tensor,
        lse: torch.Tensor | None = None,
        causal_lens: torch.Tensor | None = None,
        cp: tuple[int, int, int] = (1, 0, 1),
    ) -> torch.Tensor:
        """BF16 latent [T, H, 512] of the FP8 ``query`` [T, H, 576] (``q`` rows per
        request) over the FP8 ``cache`` [blocks, page, 576] with the plan in
        ``counts`` and ``work``. With ``lse`` [requests, q, H] FP32 it also writes
        each row's log2 LSE. Under DCP (``cp`` = world, rank, interleave),
        ``seq_lens`` are this rank's shard lengths and ``causal_lens`` the global
        ones. An empty request gets a zero output and an LSE of -inf."""
        requests = seq_lens.numel()
        assert cache.shape[-1] == LATENT + ROPE, cache.shape
        assert workspace.numel() >= self.workspace_bytes
        query = query.view(requests, self.q, self.heads, LATENT + ROPE)
        out = query.new_empty(
            (requests, self.q, self.heads, LATENT), dtype=torch.bfloat16
        )
        cache = cache.view(torch.float8_e4m3fn)
        kernel = _compile(
            self.heads,
            self.q,
            cache.shape[1],
            self.arch,
            self.clusters,
            self.causal,
            lse is not None,
            cp,
        )
        with tvm_ffi.use_torch_stream():
            kernel(
                query[..., :LATENT],
                query[..., LATENT:],
                cache[:, :, :LATENT],
                cache[:, :, LATENT:],
                block_table,
                out,
                lse,
                workspace,
                Int32(CAPACITY),
                seq_lens,
                seq_lens if causal_lens is None else causal_lens,
                counts[:requests],
                Float32(softmax_scale),
                Float32(output_scale),
                None,
                work,
            )
        return out.view(-1, self.heads, LATENT)
