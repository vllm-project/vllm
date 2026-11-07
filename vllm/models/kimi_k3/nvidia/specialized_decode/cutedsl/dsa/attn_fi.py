# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer sparse MLA core of a K3.1 DSA layer (FLASHINFER_MLA_SPARSE): the FP8
query rows [B, H, 576] over the flat rows of the packed blocks (a per-tensor FP8 cache,
or on prims_ts only an nvfp4_ds_mla one), by FlashInfer's public
sparse MLA decodes, called directly with no allocation or extra kernel:

  convert     the top-k -> this rank's flat rows (a compacted prefix) and their
              counts, into the model's shared buffers on the compute stream (no
              index-group side stream or logical top-k event)
  prims_ts    BatchSparseMLADecodePagedTSWrapper on the stock decode's plan of the
              batch (the smallest of its row capacities that holds it: the plan
              fixes the split-KV schedule and so the BF16 partial roundings),
              planned at construction; the offsets are prepared, so a step
              launches the decode alone
  trtllm-gen  trtllm_batch_decode_with_kv_cache_mla (the stock workspace and
              multi-CTA KV counters)

Both write the latent straight into MlaValue's [B, H, 512] rows (TP) or DcpValue's
[B, 128, 512] rows and [B, 128] LSE (DCP: the q mailbox is already padded to the
kernel's 128 heads, so there is no head pad or unpad copy). A row this rank owns
none of (count 0) needs no masking: DcpValue reads the counts and drops it, whatever
its LSE (prims_ts returns -inf, trtllm-gen a finite garbage value).

The kernel is picked per (topology, rows) from `FI_KERNEL` among those that read
the cache (`fi_kernels`)."""

import bisect
import math

import torch

from ..common.policy import batch_table

KERNELS = ("trtllm", "prims_ts")
# Sparse MLA kernel per decode batch (rows = requests x q: 5..60 are DSpark k=4
# verify batches, 1..8, 16, 32 and 64 decode batches), per topology. GB300, FlashInfer
# 0.7.2.dev20261029, the core alone (conversion + decode, L2 flushed), us at 32K
# context:
#   TP (12 heads)   rows      1     5     10    15    20    30    35    40    60    64
#     trtllm-gen           12.8  13.7  14.9  16.1  18.0  21.4  21.3  24.2  31.0  44.0
#     prims_ts             12.7  13.5  15.4  15.8  19.2  21.2  22.2  30.1  34.0  36.7
#   DCP8 (128 heads)
#     trtllm-gen           14.3  17.4  18.5  19.5  21.8  22.9  23.7  16.8  17.8  18.4
#     prims_ts             12.1  12.7  13.0  13.0  13.1  13.3  13.5  13.6  14.4  14.9
# (prims_ts on wider plans than the stock decode's at 21-31 TP / 3-19 DCP8 rows; 1.5K
# and 128K contexts pick the same kernel at all but a few TP batches within 1 us.)
# Whole layer (TP8, default overlap): prims_ts' split-KV decode reads its rows
# before its PDL wait, so the kernel before it must not launch it early
# (PreparedDsa.held) and it starts a launch later than trtllm-gen, which waits.
# late_front, us at 8K/32K/128K/AgentX (the merged stack, one allocation):
#   q5 R1 (5 rows)    trtllm 35.1/39.0/49.5/35.1   prims_ts 36.4/41.0/50.2/37.3
#   q1 M2             trtllm 34.5/37.8/49.0/39.8   prims_ts 36.0/39.3/49.3/41.2
#   q1 M4             trtllm 36.6/40.9/60.3/49.0   prims_ts 37.3/42.2/60.2/49.6
#   q5 R2 (10 rows)   trtllm 41.0/45.4/59.3/48.1   prims_ts 41.4/45.3/58.1/47.5
#   q5 R3 (15 rows)   trtllm 41.9/48.5/66.2/56.2   prims_ts 41.7/47.5/65.4/54.8
#   q1 M16            trtllm 47.8/61.1/107.9/82.1  prims_ts 46.6/57.4/107.7/81.0
#   q5 R4 (20 rows)   trtllm 47.5/58.0/75.6/62.2   prims_ts 54.7/63.4/80.6/68.6
# so trtllm-gen up to 9 rows and from 17, prims_ts at 10-16; earlier sweeps: at
# 30-35 rows prims_ts by 0-2.3, from 40 rows trtllm-gen by 0-5.6 again (R8 fork
# 59.2-106.6 vs 64.9-111.5).
#
# Precision: both kernels compute alike (E4M3 P at 448 of the running row max, FP32
# softmax sums and accumulation, BF16 split-KV partials and output: equal to the BF16
# ulp on one KV tile, test_flashinfer_kernels_share_precision), so a batch may run
# either only where their KV split schedules also agree, which sets where the P
# quantization re-anchors and the partials round. At 128 heads (DCP8) trtllm-gen
# splits each row's KV across CTAs up to 39 rows and prims_ts only on plans of up to
# 19 rows: at 20-39 rows prims_ts' unsplit rows err up to 1.4x stock's (trtllm-gen)
# against exact attention, so those batches run trtllm-gen (GB300 core alone, 2048
# keys: prims_ts 19.0-19.3 us vs 21.3-23.8; ~256 keys: 8.9-9.1 vs 16.5-18.4). At 12
# heads (TP) the policy's prims_ts batches err as trtllm-gen's (0.91-1.0x).
FI_KERNEL = {
    "tp": "1-9:trtllm,10-16:prims_ts,17-29:trtllm,30-39:prims_ts,40-63:trtllm,"
    "64:prims_ts,65-320:trtllm",
    "dcp": "1-19:prims_ts,20-39:trtllm,40-320:prims_ts",
}
# prims_ts plans: a plan serves any batch up to its rows, and FlashInfer picks its
# split-KV schedule from them and from the top-k width, so a batch runs on the plan
# the stock decode gives it (bit-identical output). A wider plan can be faster (DCP8
# 3..19 rows on a 20-row plan: 12-13 us, against 13.5-16.8 on their own; nvfp4 at 12
# heads 6-64 rows on the 128-row plan: 35-36, against 35-81), but its split-KV
# schedule rounds different BF16 partials.


def dsa_on_flashinfer(attention) -> bool:
    """Whether a KimiK3DSAAttention runs FLASHINFER_MLA_SPARSE (per-tensor FP8 query;
    per-tensor FP8 or nvfp4_ds_mla cache)."""
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseImpl,
    )

    return isinstance(attention.mla_attn.mla_attn.impl, FlashInferMLASparseImpl)


def fi_kernels(kv_cache_dtype) -> tuple[str, ...]:
    """The sparse MLA kernels the specialized decode picks from per batch (FI_KERNEL):
    trtllm-gen and prims_ts on a per-tensor FP8 cache, prims_ts alone on an
    nvfp4_ds_mla one (no trtllm-gen kernel reads it); under DCP on a per-tensor
    FP8 cache also the specialized sparse decode ("sparse", attn_sparse) at the
    batches attn_sparse.SPARSE_DCP gives it."""
    if kv_cache_dtype == "nvfp4_ds_mla":
        return ("prims_ts",)
    return (*KERNELS, "sparse")


def kernel_for(rows, dcp, kernels, heads=None):
    """The sparse MLA kernel of one batch: the measured choice among `kernels`
    (`heads`: the query heads; trtllm-gen takes at most 64 or exactly 128, so
    unpadded 64 < H < 128 run prims_ts). Without DCP, 128 heads are TP1's 96
    head-padded (fi_heads): the stock decode's kernel, trtllm-gen on a per-tensor
    FP8 cache (its split-KV schedule; GB300 128K q5, 5 / 20 / 80 / 160 / 320 rows:
    17.7 / 26.0 / 51.0 / 73.1 / 113.6 us, prims_ts 21.2 / 27.4 / 50.7 / 72.4 /
    112.6), prims_ts on nvfp4_ds_mla."""
    kernels = tuple(k for k in kernels if k in KERNELS)  # FlashInfer's only
    if heads is not None and 64 < heads < 128:
        kernels = tuple(k for k in kernels if k != "trtllm") or kernels
    if heads == 128 and not dcp:
        return "trtllm" if "trtllm" in kernels else kernels[0]
    if len(kernels) == 1:
        return kernels[0]
    choice = batch_table(FI_KERNEL["dcp" if dcp else "tp"]).get(rows, "prims_ts")
    return choice if choice in kernels else kernels[0]


def plan_rows_for(rows, capacities=None):
    """The rows of the prims_ts plan that runs a batch of `rows`: the smallest of
    the stock decode's row `capacities` that holds it (None: its own)."""
    if not capacities:
        return rows
    capacity = bisect.bisect_left(capacities, rows)
    assert capacity < len(capacities), (rows, capacities)
    return capacities[capacity]


class _Plans:
    """prims_ts plans shared by every DSA layer: one per (device, rows, heads, cache
    format, top-k width, LSE), each planned for its exact row count (its schedule is
    the one FlashInfer picks for that many decode rows). The layers run one at a
    time on one stream, so the plans of a `plan` call share one workspace."""

    def __init__(self):
        self.plans: dict[tuple, object] = {}

    def plan(self, device, shapes, fmt, topk):
        """Plan every (plan rows, heads, lse) of `shapes` not yet planned."""
        from flashinfer.attention.prims_ts import (
            BatchSparseMLADecodePagedTSWrapper,
            get_prims_ts_sparse_mla_decode_workspace_size,
        )

        key = fmt, topk
        todo = [s for s in shapes if (device, *s, *key) not in self.plans]
        if not todo:
            return
        kwargs = dict(
            max_q_len=1,
            q_dtype=torch.float8_e4m3fn,
            max_topk=topk,
            packed_query=True,
            qk_rope_head_dim=64,
            kv_cache_format=fmt,
        )
        size = max(
            get_prims_ts_sparse_mla_decode_workspace_size(
                device, rows, heads, return_lse=lse, **kwargs
            )
            for rows, heads, lse in todo
        )
        workspace = torch.empty(size, dtype=torch.uint8, device=device)
        for rows, heads, lse in todo:
            plan = BatchSparseMLADecodePagedTSWrapper(workspace)
            plan.plan(device, rows, heads, return_lse=lse, **kwargs)
            self.plans[device, rows, heads, lse, *key] = plan

    def get(self, device, rows, heads, lse, fmt, topk):
        return self.plans[device, rows, heads, lse, fmt, topk]


def _offsets(capacity, rows, device):
    """The packed-query offsets of `rows` one-token rows in a plan of
    `capacity`: [0, 1, ..., rows, ..., rows]."""
    offsets = torch.arange(capacity + 1, dtype=torch.int32, device=device)
    return offsets.clamp_(max=rows)


PLANS = _Plans()


class FlashInferCore:
    """One layer's FlashInfer sparse core. `impl` / `layer`: its
    FlashInferMLASparseImpl and attention layer; `heads`: the query heads per
    row (the q mailbox's padded heads under DCP); `buffers`: the model's shared
    flat rows [R, K], their counts [R], the latent [R * heads * 512] (BF16) and,
    under DCP, the LSE [R * heads] (FP32); `kernels`: {rows: kernel}; `dcp`,
    `rank`: the KV cache's DCP sharding; `capacities`: the stock decode's prims_ts row
    capacities (`plan_rows_for`); `widths`: the index widths `decode` may get (default
    the buffers' top-k; e.g. also the DCP top-k merge's rows)."""

    def __init__(
        self,
        impl,
        layer,
        heads,
        buffers,
        kernels,
        dcp=1,
        rank=0,
        capacities=None,
        widths=(),
    ):
        self.impl, self.layer, self.heads = impl, layer, heads
        self.slots, self.valid, self.latent, self.lse = buffers
        self.kernels, self.dcp, self.rank = kernels, dcp, rank
        self.fmt = getattr(impl, "prims_ts_format", "native")
        self.topk = self.slots.shape[1]
        self.device = self.slots.device
        batches = [m for m, k in kernels.items() if k == "prims_ts"]
        plan_rows = {m: plan_rows_for(m, capacities) for m in batches}
        # {rows: (plan rows, packed-query offsets)}
        self.plan_rows = {
            m: (plan_rows[m], _offsets(plan_rows[m], m, self.device)) for m in batches
        }
        shapes = {(r, heads, dcp > 1) for r, _ in self.plan_rows.values()}
        self.widths = sorted({self.topk, *widths})
        for width in self.widths if shapes else ():
            PLANS.plan(self.device, sorted(shapes), self.fmt, width)

    def convert(
        self, topk, metadata, rows, stride_rows, interleave
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The top-k ids -> this rank's flat rows of the packed blocks (a compacted
        prefix; under DCP the global ids this rank owns, `interleave` tokens per
        rank in turn) and their counts."""
        from vllm.v1.attention.backends.mla.sparse_utils import (
            triton_convert_req_index_to_global_index,
        )

        slots, valid = self.slots[:rows], self.valid[:rows]
        triton_convert_req_index_to_global_index(
            metadata.req_id_per_token[:rows],
            metadata.block_table,
            topk[:rows],
            BLOCK_SIZE=metadata.block_size,
            BLOCK_STRIDE_ROWS=stride_rows,
            NUM_TOPK_TOKENS=self.topk,
            dcp_size=self.dcp,
            dcp_rank=self.rank,
            cp_kv_cache_interleave_size=interleave,
            return_valid_counts=True,
            out=slots,
            valid_counts_out=valid,
        )
        return slots, valid

    def run(self, query, cache, topk, metadata):
        """`query` [B, heads, 576] (FP8 bytes) over the packed `cache` at the
        top-k ids `topk` -> the latent [B, heads, 512] (and under DCP its LSE
        [B, heads] in the kernel's base)."""
        assert topk.shape[1] == self.topk
        stride_rows = self.rows(cache, metadata.block_size)[1]
        rows = query.shape[0]
        slots, valid = self.convert(topk, metadata, rows, stride_rows, 1)
        return self.decode(query, cache, slots, valid, metadata.block_size)

    def rows(self, cache, block_size):
        """The flat rows of the packed `cache` and the rows per block stride."""
        from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view

        view = torch.float8_e4m3fn if self.fmt == "native" else torch.uint8
        return flat_kv_row_view(cache.view(view), block_size)

    def decode(self, query, cache, slots, valid, block_size):
        """The sparse decode of `query` over this rank's flat rows `slots` of
        the packed `cache` (a compacted prefix of `valid` per row; e.g. from a
        fused top-k that emits them) -> as `run`."""
        rows, heads = query.shape[0], self.heads
        assert query.shape[1] == heads
        kv, stride_rows = self.rows(cache, block_size)
        query = query.view(torch.float8_e4m3fn)
        latent = self.latent[: rows * heads * 512].view(rows, heads, 512)
        lse = None
        if self.dcp > 1:
            lse = self.lse[: rows * heads].view(rows, heads)
        if self.kernels[rows] == "prims_ts":
            self.prims_ts(query, kv, slots, valid, latent, lse)
        else:
            self.trtllm(query, kv, stride_rows, slots, valid, latent, lse, block_size)
        return latent, lse

    def prims_ts(self, query, kv, slots, valid, latent, lse):
        layer = self.layer
        capacity, offsets = self.plan_rows[query.shape[0]]
        key = self.heads, lse is not None, self.fmt, slots.shape[1]
        plan = PLANS.get(self.device, capacity, *key)
        descale = layer._k_scale_float if self.fmt == "native" else 1.0
        plan.run(
            query,
            kv,
            slots,
            valid,
            qo_indptr=offsets,
            softmax_scale=self.impl.scale * layer._q_scale_float * descale,
            output_scale=descale,
            out=latent,
            lse=lse,
            validate=False,
        )

    def trtllm(self, query, kv, stride_rows, slots, valid, latent, lse, block_size):
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        impl, width = self.impl, slots.shape[1]
        impl._prepare_mqa_kernel(self.layer, query.device)
        # Block sizes and strides are aligned to 32 rows.
        page = math.gcd(stride_rows, block_size, 64)
        pages = kv.view(-1, page, kv.shape[-1])
        extra: dict[str, torch.Tensor] = {}
        if lse is not None:
            extra["lse"] = lse
        trtllm_batch_decode_with_kv_cache_mla(
            query=query.unsqueeze(1),
            kv_cache=pages.unsqueeze(1),
            workspace_buffer=impl._workspace_buffer,
            qk_nope_head_dim=impl.qk_nope_head_dim,
            kv_lora_rank=impl.kv_lora_rank,
            qk_rope_head_dim=impl.qk_rope_head_dim,
            block_tables=slots.unsqueeze(1),
            seq_lens=valid,
            max_seq_len=width,
            out=latent.unsqueeze(1),
            bmm1_scale=impl.bmm1_scale,
            bmm2_scale=impl.bmm2_scale,
            sparse_mla_top_k=width,
            return_lse=lse is not None,
            **extra,
        )
