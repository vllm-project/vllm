# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Query stage of a K3.1 DSA layer: MlaQuery (dsa) -> wq_b -> DsaIndex.

  MlaQuery (dsa)   the query rows (BF16 [B, 64 (padded heads), 576] for FlashMLA /
                   CuTe, per-tensor FP8 [B, H, 576] for FlashInfer), the MLA KV
                   insert and the normalized q_c rows; under DCP it also publishes
                   the query into the q mailbox
  wq_b             split-K projection of q_c -> [B, 32 * 128]
  DsaIndex         FP8 indexer query + head weights, indexer key insert

The packed projection row is [q_a 1536 | kv_a 576 | gate H*128 | wk 128 | wv 32]."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from vllm.utils.deep_gemm import mqa_logits_use_mx_scales

from ..common.host import fake, prepare_shared_carveout, stream, to_cute
from ..gemm.projection import PreparedProjection, weight_map
from ..mla.query import MlaQuery
from ..mla.query_wide import MlaAbsorb, MlaQuant, absorb_bmm, fi_heads, split_query
from .front import IndexFront
from .front_mma import IndexFrontMma
from .index import DIM, HEADS, DsaIndex
from .overlap import QueryNorm, QueryWait

QUERY_HEADS = 64  # FlashMLA sparse FP8 decode reads 64 or 128 query heads


def gate_columns(heads):
    return 2112, 2112 + heads * 128


def index_policies(batches):
    """Split-K tactics of the replicated wq_b projection (N 4096, K 1536) per padded
    batch (GB300 us, previous / these: B1-8 4.9/2.9, B32 3.7/3.4, B64 4.9/3.9);
    wider batches run 64-row N tiles."""
    tactics = {8: (128, 8, 3, 4), 16: (64, 16, 2, 6), 32: (64, 32, 1, 8)}
    tactics[64] = (64, 64, 1, 4)
    return {b: tactics[next((n for n in tactics if b <= n), 64)] for b in batches}


def q_b_policies(batches, heads=96):
    """Split-K tactics of the split query's Q_B projection (`q_b_wide_policies`
    from WIDE_HEADS heads; below, the TP8 shard: N = 12 * 192 = 2304, 64-row weight
    tiles split K four ways up to 32 rows, then two ways). GB300, 12 heads, L2
    flushed, us (SOL 0.94): 5 rows 5.3 (64,8,4,4; 6.8 with 128-row tiles), 40 rows
    7.3 (64,64,2,4; 9.1 split four ways), 80 rows 7.6 (64,64,2,4; 9.5 with one
    128-row batch tile), 320 rows 11.2 (64,160,2,3; 13.6 with 128-row tiles, 16.8
    with 128-row batch tiles)."""
    if heads >= 64:
        return q_b_wide_policies(batches)
    tactics = {8: (64, 8, 4, 4), 16: (64, 16, 4, 4), 32: (64, 32, 4, 4)}
    tactics.update({64: (64, 64, 2, 4), 128: (64, 64, 2, 4)})
    tactics[160] = (64, 160, 2, 3)
    return {b: tactics[next((n for n in tactics if b <= n), 160)] for b in batches}


def q_b_wide_policies(batches):
    """Split-K tactics of the split query's Q_B projection (N = H * 192, K 1536;
    query_wide): 128-row weight tiles (144 at 96 heads: one wave), no split, one
    batch tile up to 160 rows (each batch tile re-streams the weight). GB300,
    96 heads, L2 flushed, us: 5 rows 13.6 (8-row tiles; 14.0-36.6 for the
    others), 20 rows 14.2 (32), 80 rows 15.2 (128; 24.6 with 64-row tiles), 320
    rows 32.8 (160; 52.5 with 64, 37.8 with 128, 43.2 with 256)."""
    tactics = {8: (128, 8, 1, 4), 16: (128, 16, 1, 6), 32: (128, 32, 1, 5)}
    tactics.update({64: (128, 64, 1, 4), 128: (128, 128, 1, 3)})
    tactics[160] = (128, 160, 1, 3)
    return {b: tactics[next((n for n in tactics if b <= n), 160)] for b in batches}


def runs_split(heads, batch, dcp, fp8):
    """Whether the MLA query of `batch` rows runs split (mla.query_wide): at many
    heads always (FlashInfer only), at few heads (TP8 / DCP8) on FlashInfer's
    per-tensor FP8 query from SPLIT_FROM rows."""
    return split_query(heads, batch, dcp) and (fp8 or heads >= 64)


class WideQuery:
    """compile_query's result at query_wide heads: the q_a RMSNorm (QueryNorm, None
    when the overlap mode's main-stream QueryNorm writes the q_c rows), then the
    layer's Q_B projection (QueryStage.q_projection) and MlaAbsorb, or (`bmm`,
    query_wide.absorb_bmm) the W_UK bmm and MlaQuant."""

    def __init__(self, norm, absorb, bmm=False):
        self.norm, self.absorb, self.bmm = norm, absorb, bmm


def compile_query(
    batch,
    heads,
    pitch,
    fakes,
    device,
    dcp=0,
    q_heads=0,
    fp8=False,
    nvfp4=False,
    overlap="serial",
    cap=64,
    release=True,
    pdl=True,
):
    """MlaQuery (dsa) of one batch of rows. `fakes`: the fake weight and cache
    tensors (q_a norm, kv_a norm, Q_B, W_UK, the MLA cache). Under DCP the query
    also publishes into the q mailbox (`q_heads`: its padded heads). `nvfp4`:
    the FP8 query over an nvfp4_ds_mla cache (FlashInfer prims_ts). `cap`: the
    q mailbox rows (DcpBox.cap). `overlap` (overlap.MODES): "fork" and "front"
    leave the q_c rows to QueryNorm or IndexFront, and every mode but "serial" the
    mailbox poll to QueryWait.
    `release`: launch the next kernel early (PDL). `pdl`: MlaQuery itself
    launches with PDL (else it starts once the kernel it follows completes)."""
    qnorm, kvnorm, qb, uk, kv = fakes
    scalar = fake((1,), (1,), torch.float32, 4)
    if runs_split(heads, batch, dcp, fp8):
        # Under DCP QueryWait polls the q mailbox (MlaAbsorb only publishes).
        assert not dcp or overlap != "serial", "the split query does not poll"
        assert pdl, "the split query launches with PDL"
        # FlashMLA sparse's BF16 query rows hold QUERY_HEADS (64) heads.
        assert fp8, "the split query writes the FlashInfer (per-tensor FP8) query"
        norm = (
            None if overlap in ("fork", "late") else compile_norm(batch, heads, pitch)
        )
        box = (dcp, q_heads, cap) if dcp else None
        absorb = compile_absorb(batch, heads, pitch, fakes, fp8, nvfp4, release, box)
        return WideQuery(norm, absorb, absorb_bmm(heads, batch))
    if fp8:
        query = fake((batch, heads, 576), (heads * 576, 576, 1), torch.uint8)
    else:
        query = fake((batch, QUERY_HEADS, 576), (QUERY_HEADS * 576, 576, 1))
    qc = fake((batch, 1536), (1536, 1))
    projected = fake((batch, pitch), (pitch, 1))
    slots = fake((batch,), (1,), torch.int64, 8)
    mla_query = MlaQuery(
        heads,
        batch,
        q_b_cp_async=heads == 12 and batch == 1,
        dsa=True,
        fp8=fp8,
        nvfp4=nvfp4,
        dcp=dcp,
        q_heads=q_heads,
        store_qc=overlap in ("serial", "wait"),
        poll=overlap == "serial",
        late=overlap in ("late", "late_front"),
        release=release,
        pdl=pdl,
    )
    eps = cutlass.Float32(1e-6)
    q_args = (projected, qnorm, kvnorm, qb, uk, kv, slots, scalar, scalar, eps)
    publish: tuple = ()
    if dcp:  # this parity's q mailbox (bytes), its multicast address, rank
        mailbox = fake((cap, q_heads, 576 if fp8 else 1152), None, torch.uint8)
        publish = (mailbox, cutlass.Int64(0), cutlass.Int32(0))
    s = make_fake_stream()
    kernel = cute.compile(mla_query, *q_args, query, qc, *publish, s)
    if batch > 1:
        kernel = prepare_shared_carveout(kernel, device, 100)
    return kernel


def compile_absorb(batch, heads, pitch, fakes, fp8, nvfp4, release, box=None):
    """MlaAbsorb of one batch of rows (query_wide); `box`: (DCP size, the q
    mailbox's padded heads, its rows) to publish into under DCP."""
    qnorm, kvnorm, _, uk, kv = fakes
    scalar = fake((1,), (1,), torch.float32, 4)
    if fp8:  # the first ``heads`` of the head-padded FlashInfer query rows
        padded = fi_heads(heads) * 576
        query = fake((batch, heads, 576), (padded, 576, 1), torch.uint8)
    else:
        query = fake((batch, QUERY_HEADS, 576), (QUERY_HEADS * 576, 576, 1))
    projected = fake((batch, pitch), (pitch, 1))
    q_rows = fake((batch, heads * 192), (heads * 192, 1))
    slots = fake((batch,), (1,), torch.int64, 8)
    dcp, q_heads, cap = box or (0, 0, 0)
    kind = MlaAbsorb
    if fp8 and absorb_bmm(heads, batch):  # the W_UK bmm's latent in W_UK's place
        kind, uk = MlaQuant, fake((heads, batch, 512), (batch * 512, 512, 1))
    absorb = kind(
        heads,
        batch,
        dsa=True,
        fp8=fp8,
        nvfp4=nvfp4,
        release=release,
        dcp=dcp,
        q_heads=q_heads,
    )
    eps = cutlass.Float32(1e-6)
    args = (projected, qnorm, kvnorm, q_rows, uk, kv, slots, scalar, scalar, eps)
    publish: tuple = ()
    if dcp:  # this parity's q mailbox (bytes), its multicast address, rank
        mailbox = fake((cap, q_heads, 576 if fp8 else 1152), None, torch.uint8)
        publish = (mailbox, cutlass.Int64(0), cutlass.Int32(0))
    return cute.compile(absorb, *args, query, *publish, make_fake_stream())


@cache
def absorb_rows(device, rows, heads):
    """The W_UK bmm's BF16 latent [H * rows * 512] (query_wide.absorb_bmm), shared
    by every DSA layer of a device."""
    return torch.empty(heads * rows * 512, dtype=torch.bfloat16, device=device)


@cache
def q_rows_buffer(device, rows, columns):
    """The split query's Q_B rows, shared by every DSA layer of a device."""
    return torch.empty((rows, columns), dtype=torch.bfloat16, device=device)


@cache
def compile_norm(batch, heads, pitch):
    """QueryNorm of one batch of rows (shared by every DSA layer: no state)."""
    return cute.compile(
        QueryNorm(heads, batch),
        fake((batch, pitch), (pitch, 1)),
        fake((1536,), (1,)),
        fake((batch, 1536), (1536, 1)),
        cutlass.Float32(1e-6),
        make_fake_stream(),
    )


@cache
def compile_wait(batch, heads, dcp, q_heads, fp8, cap=64):
    """QueryWait of one batch of rows under DCP (else None; `cap`: the q mailbox
    rows). Shared by every DSA layer: the mailbox is a launch argument."""
    if not dcp:
        return None
    q_bytes = 1 if fp8 else 2
    mailbox = fake((cap, q_heads, 576 * q_bytes), None, torch.uint8)
    wait = QueryWait(batch, dcp * heads, q_heads, q_bytes)
    return cute.compile(wait, mailbox, make_fake_stream())


# The "front" overlap mode's kernel (from the packed rows): "cluster" (front.py,
# mma.sync, at most CLUSTER_FRONT_MAX rows: it stages every row in shared memory)
# up to FRONT_MMA_FROM - 1 rows, then "mma" (front_mma.py, tcgen05, any batch).
# FRONT_KERNEL forces one (tests, benchmarks).
FRONT_MMA_FROM = 41
CLUSTER_FRONT_MAX = 64
FRONT_KERNEL: str | None = None


def front_kernel(batch):
    """The "front" mode's kernel of `batch` rows ("cluster" or "mma")."""
    kind = FRONT_KERNEL or ("cluster" if batch < FRONT_MMA_FROM else "mma")
    if kind == "cluster" and batch > CLUSTER_FRONT_MAX:
        raise ValueError(f"IndexFront serves at most {CLUSTER_FRONT_MAX} rows")
    return kind


def compile_front(
    batch,
    heads,
    pitch,
    wq,
    index_kv,
    device,
    normalized=False,
):
    """The fused index front of one batch of rows (`wq`, `index_kv`: the
    layouts of the wq_b weight and the indexer cache): from the packed rows (q_a
    RMSNorm, wq_b, DsaIndex), or (`normalized`: IndexFrontMma only) from the q_c
    rows. Shared by every DSA layer of those layouts: the weight, cache and outputs
    are launch arguments."""
    kind, mx = front_kernel(batch), mqa_logits_use_mx_scales()
    args = batch, heads, pitch, wq, index_kv, device, normalized
    return _compile_front(*args, kind, mx)


@cache
def _compile_front(batch, heads, pitch, wq, index_kv, device, normalized, kind, mx):
    _, key = gate_columns(heads)
    wq, index_kv = fake(*wq), fake(*index_kv)
    projected = fake((batch, pitch), (pitch, 1))
    slots = fake((batch,), (1,), torch.int64, 8)
    norm = fake((DIM,), (1,), torch.float32, 4)
    q_out = fake((batch, HEADS, DIM), None, torch.uint8)
    w_out = fake((batch, HEADS), None, torch.bfloat16, 4)
    eps = cutlass.Float32(1e-6)
    tail = (norm, norm, index_kv, slots, q_out, w_out, eps, eps)
    if kind == "cluster" and not normalized:
        front: IndexFront | IndexFrontMma = IndexFront(batch, key, key + DIM, mx=mx)
        args: tuple = (projected, fake((1536,), (1,)), wq, *tail)
    else:
        front = IndexFrontMma(
            batch,
            key,
            key + DIM,
            mx=mx,
            normalized=normalized,
            row_pitch=1536 if normalized else pitch,
        )
        rows = fake((batch, 1536), (1536, 1)) if normalized else projected
        args = (projected, rows, fake((1536,), (1,)), cutlass.Int64(0), *tail)
    kernel = cute.compile(front, *args, make_fake_stream())
    return prepare_shared_carveout(kernel, device, 100)


def compile_index(batch, heads, pitch, index_kv):
    """DsaIndex of one batch of rows into the indexer cache `index_kv` (fake)."""
    _, key = gate_columns(heads)
    index = DsaIndex(key, key + DIM, mx=mqa_logits_use_mx_scales())
    projected = fake((batch, pitch), (pitch, 1))
    slots = fake((batch,), (1,), torch.int64, 8)
    iq = fake((batch, HEADS * DIM), (HEADS * DIM, 1))
    norm = fake((DIM,), (1,), torch.float32, 4)
    q_out = fake((batch, HEADS, DIM), None, torch.uint8)
    w_out = fake((batch, HEADS), None, torch.bfloat16, 4)
    eps = cutlass.Float32(1e-6)
    i_args = (projected, iq, norm, norm, index_kv, slots, q_out, w_out, eps)
    return cute.compile(index, *i_args, make_fake_stream())


class QueryStage:
    """One layer's query stage. `scratch`: the shared rows (BF16 query, q_c,
    indexer projection, FP8 indexer query, head weights); `fp8`: a per-tensor FP8
    query (FlashInfer), whose query rows are this layer's own, over a per-tensor FP8
    cache or (`nvfp4`) an nvfp4_ds_mla one, whose rows carry their own scales."""

    def __init__(self, attention, pitches, scratch, fp8, nvfp4=False, dcp=0):
        wrapper = attention.mla_attn
        layer, self.indexer = wrapper.mla_attn, attention.dsa_indexer
        self.heads, self.pitches = layer.W_UK_T.shape[0], pitches
        norms = wrapper.q_a_layernorm.weight, wrapper.kv_a_layernorm.weight
        self.weights: tuple = (*norms, wrapper.q_b_proj.weight, layer.W_UK_T)
        self.weights += (layer.W_UV,)
        self.device = device = layer.W_UK_T.device
        self.eps = cutlass.Float32(wrapper.q_a_layernorm.variance_epsilon)
        knorm = self.indexer.k_index_norm
        self.key_eps = cutlass.Float32(knorm.eps)
        self.key_norm = to_cute(knorm.weight, 4), to_cute(knorm.bias, 4)
        self.views = tuple(
            to_cute(w, 4 if w.dtype == torch.float32 else 16) for w in self.weights
        )
        self.unit = torch.ones(1, dtype=torch.float32, device=device)
        self.query, self.qc, self.iq, self.q_fp8, self.head_w = scratch
        # The FP8 query's inverse per-tensor scale and the KV row's scale (the
        # stock concat_and_cache_mla divides by it; FlashMLA: unused).
        self.scales = self.unit, self.unit
        if fp8:
            q_scale_inv = (1 / layer._q_scale.float()).reshape(1).to(device)
            k_scale = layer._k_scale.float().reshape(1).to(device)
            # nvfp4_ds_mla: the KV row is unscaled (prims_ts: kv_descale 1)
            self.scales = q_scale_inv, self.unit if nvfp4 else k_scale
            # FlashInfer's query heads (fi_heads): the pad heads stay zero.
            shape = (max(pitches), fi_heads(self.heads), 576)
            self.query = torch.zeros(shape, dtype=torch.uint8, device=device)
        policies = index_policies(sorted(pitches))
        wq = self.indexer.wq_index.weight
        self.index_projection = PreparedProjection(wq, policies).prepare()
        self.wq_view = to_cute(wq)
        # The fused index front's weight TMA map (128-row boxes: one head).
        self.wq_map = weight_map(wq, 128)
        # The split query's Q_B projection and its rows (split_query: TP1 at every
        # batch, TP8 at wide ones).
        self.q_projection = self.q_rows = None
        split = [b for b in pitches if runs_split(self.heads, b, dcp, fp8)]
        if split:
            q_b = self.weights[2]
            policies = q_b_policies(sorted(split), self.heads)
            self.q_projection = PreparedProjection(q_b, policies).prepare()
            self.q_rows = q_rows_buffer(device, max(pitches), q_b.shape[0])
            if any(absorb_bmm(self.heads, b) for b in split):
                absorb_rows(device, max(pitches), self.heads)

    def run_query(self, kernel, projected, cache, slots, publish=()):
        """MlaQuery: query rows, the KV insert at `slots` and q_c (`publish`:
        the q mailbox, its multicast address and this rank, under DCP)."""
        batch, s = projected.shape[0], stream(self.device)
        qnorm, kvnorm, qb, uk, _ = self.views
        kv = to_cute(cache.view(torch.uint8))
        q_scale_inv, k_scale = (to_cute(t, 4) for t in self.scales)
        x, q_slots = to_cute(projected), to_cute(slots[:batch], 8)
        scales = q_scale_inv, k_scale, self.eps
        if isinstance(kernel, WideQuery):
            assert self.q_projection is not None and self.q_rows is not None
            if kernel.norm is not None:
                self.run_norm(kernel.norm, projected)
            rows = self.q_projection.run(self.qc[:batch], self.q_rows[:batch])
            if kernel.bmm:  # W_UK as the stock decode's bmm over the heads
                heads = self.heads
                latent = absorb_rows(self.device, max(self.pitches), heads)
                latent = latent[: heads * batch * 512].view(heads, batch, 512)
                nope = rows.view(batch, heads, 192)[..., :128].transpose(0, 1)
                torch.bmm(nope, self.weights[3], out=latent)
                uk = to_cute(latent)
            q_args = (x, qnorm, kvnorm, to_cute(rows), uk, kv, q_slots, *scales)
            query = to_cute(self.query[:batch, : self.heads])
            kernel.absorb(*q_args, query, *publish, s)
            return
        q_args = (x, qnorm, kvnorm, qb, uk, kv, q_slots, *scales)
        query, qc = to_cute(self.query[:batch]), to_cute(self.qc[:batch])
        kernel(*q_args, query, qc, *publish, s)

    def run_norm(self, kernel, projected):
        """QueryNorm: the q_c rows of `projected` (query overlap)."""
        batch, s = projected.shape[0], stream(self.device)
        qnorm = self.views[0]
        kernel(to_cute(projected), qnorm, to_cute(self.qc[:batch]), self.eps, s)

    def run_front(self, kernel, projected, index_slots, normalized=False):
        """The fused index front: the FP8 indexer query and head weights straight
        from the packed rows (`normalized`: from the q_c rows), and the key into
        the indexer cache."""
        batch, s = projected.shape[0], stream(self.device)
        rows = to_cute(self.q_fp8[:batch]), to_cute(self.head_w[:batch], 4)
        key = to_cute(self.indexer.k_cache.kv_cache), to_cute(index_slots[:batch], 8)
        x, wq = to_cute(projected), self.wq_view
        args: tuple = (x,)
        if front_kernel(batch) != "cluster" or normalized:
            args += (to_cute(self.qc[:batch]) if normalized else x,)
            wq = cutlass.Int64(self.wq_map.data_ptr())
        args += (self.views[0], wq, *self.key_norm, *key, *rows)
        kernel(*args, self.eps, self.key_eps, s)

    def run_index(self, kernel, projected, index_slots):
        """wq_b, then the FP8 indexer query and head weights and the key into the
        indexer cache."""
        batch = projected.shape[0]
        iq = self.index_projection.run(self.qc[:batch], self.iq[:batch])
        rows = to_cute(self.q_fp8[:batch]), to_cute(self.head_w[:batch], 4)
        key = to_cute(self.indexer.k_cache.kv_cache), to_cute(index_slots[:batch], 8)
        s = stream(self.device)
        x = to_cute(projected)
        kernel(x, to_cute(iq), *self.key_norm, *key, *rows, self.key_eps, s)
