# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared MLA core: the packed input projection feeds the query kernel, then either
tokenspeed_mla_decode (core "tokenspeed") or FlashInfer's trtllm-gen MLA decode (core
"trtllm"; FP8 Q and KV as tokenspeed, FP32 softmax and accumulation) into the BF16
latent, and MlaValue (W_UV / gate). Under DCP (``dcp``: a DcpBox) the tokenspeed core
runs over the local KV shard for every DCP rank's heads between the query's q
publication and ``DcpValue``'s merge (see mla.dcp). Spec verify: ``q`` query tokens
per request (rows grouped by request, causal among them)."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ..common.host import fake, layout, prepare_shared_carveout, stream, to_cute
from .dcp import (
    compile_value,
    local_max,
    stock_decode,
)
from .query import MlaQuery
from .value import MlaValue

# tokenspeed (SM-filling splits) streams long contexts fast; trtllm-gen wins from B16.
TRTLLM_CORE_BATCH, CORES = 16, ("tokenspeed", "trtllm")
_FLASHINFER: dict[tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor]] = {}


@cache
def compile_core(batch, heads, pitch, specs, device, dcp=0, sms=148, q=1):
    """CuTe kernels (query, value) of one batch of rows (``q`` per request); under DCP
    (``dcp``: its size) the query publishes and the value is DcpValue."""
    qnorm, kvnorm, qb, uk, uv, qinv, kinv, kv = (
        fake(shape, stride, dtype, 4 if dtype == torch.float32 else 16)
        for shape, stride, dtype in specs
    )
    query = fake((batch, heads, 576), (heads * 576, 576, 1), torch.uint8)
    projected = fake((batch, 2112), (pitch, 1))
    slots = fake((batch,), (1,), torch.int64, 8)
    lengths = fake((batch // q,), (1,), torch.int32, 4)
    gate = fake((batch, heads * 128), (pitch, 1))
    output = fake((batch, heads * 128), (heads * 128, 1))
    latent = fake((batch, heads, 512))
    f32, s = cutlass.Float32, make_fake_stream()
    cp_async = heads == 12 and batch == 1
    mla_query = MlaQuery(heads, batch, q_b_cp_async=cp_async, dcp=dcp)
    q_args: tuple = (projected, qnorm, kvnorm, qb, uk, kv, slots, qinv, kinv)
    q_args += (f32(1e-6), query)
    if dcp:  # this parity's q mailbox, its multicast address, rank
        mailbox = fake((64, dcp * heads, 576), None, torch.uint8)
        q_args += (mailbox, cutlass.Int64(0), cutlass.Int32(0))
    with torch.accelerator.device_index(device):
        kernels = [cute.compile(mla_query, *q_args, s)]
        if batch > 1:
            kernels = [prepare_shared_carveout(kernels[0], device, 100)]
        if dcp:
            uv_spec = specs[4]
            kernels.append(
                compile_value(heads, dcp, batch, 64, uv_spec, pitch, device, sms, q)
            )
        else:
            v_args = (latent, uv, gate, lengths, output, s)
            value = MlaValue(heads, batch, q)
            kernels.append(cute.compile(value, *v_args))
    return tuple(kernels)


def core_for(batch, core="auto", q=1):
    """``core`` unless "auto": trtllm-gen from B16, else tokenspeed (spec verify,
    q > 1: tokenspeed, whose SM-filling splits stream long contexts far faster)."""
    if core == "auto":
        core = "trtllm" if batch >= TRTLLM_CORE_BATCH and q == 1 else "tokenspeed"
    if core not in CORES:
        raise ValueError(
            f"Kimi K3 MLA core {core!r}: expected auto, tokenspeed or trtllm"
        )
    return core


def pad_requests(batch, block_table, lengths, totals=None):
    """Request metadata for ``batch`` rows: PIECEWISE graphs pad the tokens but not
    the requests, so missing rows become empty requests (length 0 on page 0)."""
    rows = min(lengths.shape[0], block_table.shape[0])
    if rows >= batch:
        return block_table, lengths, totals
    pad = torch.nn.functional.pad
    block_table = pad(block_table[:rows], (0, 0, 0, batch - rows))
    lengths = pad(lengths[:rows], (0, batch - rows))
    totals = None if totals is None else pad(totals[:rows], (0, batch - rows))
    return block_table, lengths, totals


def trtllm_decode(query, cache, block_table, lengths, max_seq, scales, out):
    """FlashInfer trtllm-gen MLA decode of FP8 ``query`` [B, q, H, 576] (causal among
    the q rows) over the FP8 paged cache into BF16 ``out`` [B, q, H, 512], PDL chained;
    zeroed workspace and multi-CTA counter (self-resetting) owned here, one per (device,
    heads)."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
    from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

    device, heads = query.device, query.shape[2]
    if (device, heads) not in _FLASHINFER:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        count = get_trtllm_gen_multi_ctas_kv_counter_bytes(64, heads, sms)
        workspace = torch.zeros(128 << 20, dtype=torch.int8, device=device)
        counter = torch.zeros(count, dtype=torch.uint8, device=device)
        _FLASHINFER[device, heads] = workspace, counter
    workspace, counter = _FLASHINFER[device, heads]
    softmax, output_scale = scales
    trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=cache.view(torch.float8_e4m3fn).unsqueeze(1),
        workspace_buffer=workspace,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=block_table,
        seq_lens=lengths,
        max_seq_len=max_seq,
        out=out,
        bmm1_scale=softmax,
        bmm2_scale=output_scale,
        enable_pdl=True,
        backend="trtllm-gen",
        multi_ctas_kv_counter_buffer=counter,
    )


class PreparedMla:
    """``max_context`` (the deployment's longest sequence) bounds the split count;
    ``core`` forces "tokenspeed" or "trtllm". ``dcp``: the model's DcpBox, with this
    layer's mailbox ``parity``. ``queries``: the query lengths q (spec verify: 1 + the
    draft length) prepared for every batch of rows divisible by q."""

    def __init__(
        self,
        attention,
        pitches,
        query_scratch,
        *,
        max_context=None,
        core="auto",
        dcp=None,
        parity=0,
        queries=(1,),
    ):
        self.heads = attention.W_UK_T.shape[0]
        norms = attention.q_a_layernorm.weight, attention.kv_a_layernorm.weight
        self.weights: tuple = (*norms, attention.q_b_proj.weight, attention.W_UK_T)
        self.weights += (attention.W_UV, attention._q_scale_inv, attention._k_scale_inv)
        self.device, self.pitches = attention.q_b_proj.weight.device, pitches
        k_scale = attention._k_scale_float
        softmax = attention.scale * attention._q_scale_float * k_scale
        self.eps = cutlass.Float32(attention.rms_norm_eps)
        self.stock_scales = softmax, k_scale
        self.views = tuple(
            to_cute(w, 4 if w.dtype == torch.float32 else 16) for w in self.weights
        )
        self.query_scratch, self.kernels = query_scratch, {}
        shapes = [(b, q) for b in pitches for q in queries if b % q == 0]
        self.cores = {(b, q): core_for(b, core, q) for b, q in shapes}
        # The split count only depends on this bound; seq_lens bound the work.
        self.max_seq = max_context or 1 << 17
        self.dcp, self.parity = dcp, parity
        if dcp is not None:
            self.cores = dict.fromkeys(shapes, "tokenspeed")
            self.max_seq = local_max(self.max_seq, dcp.size)
        shape, bf16 = (max(pitches, default=0), self.heads, 512), torch.bfloat16
        self.latent = torch.empty(shape, dtype=bf16, device=self.device)

    def prepare_cache(self, kv):
        specs = tuple(layout(w) for w in (*self.weights, kv.view(torch.uint8)))
        for batch, q in self.cores:
            device = self.device.index
            dcp = self.dcp.size if self.dcp is not None else 0
            sms = torch.cuda.get_device_properties(self.device).multi_processor_count
            pitch = self.pitches[batch]
            args = batch, self.heads, pitch, specs, device, dcp, sms, q
            query = to_cute(self.query_scratch[:batch])
            latent = to_cute(self.latent[:batch])
            self.kernels[batch, q] = (*compile_core(*args), query, latent)
            if self.dcp is not None:
                self.prepare_dcp(batch)
            if kv.is_cuda:  # compile before graph capture
                table = torch.zeros(batch, 1, dtype=torch.int32, device=kv.device)
                ones = torch.full((batch,), q, dtype=torch.int32, device=kv.device)
                self.stock_core(kv, table, ones, batch, q, ones)
        return self

    def prepare_dcp(self, batch):
        """(MlaQuery's publish arguments, DcpValue's mailbox arguments, parity and
        rank) of one batch."""
        box, p = self.dcp, self.parity
        mailbox, ids = to_cute(box.q[p]), (cutlass.Int32(p), cutlass.Int32(box.rank))
        rows = to_cute(box.rows[:batch]), to_cute(box.rows_lse[:batch], 4)
        boxes = to_cute(box.partial[p]), to_cute(box.lse[p], 4)
        peers = to_cute(box.peers, 8), to_cute(box.lse_peers, 8)
        self.dcp_args = getattr(self, "dcp_args", {})
        self.dcp_args[batch] = (mailbox, box.mc(p), ids[1]), (rows, boxes, peers), ids

    def stock_core(self, cache, block_table, lengths, batch, q=1, totals=None):
        """tokenspeed_mla_decode or trtllm-gen into the BF16 latent (DCP: tokenspeed
        for every rank's heads over the local shard into the DcpBox rows + LSE; q > 1
        masks causally in global positions, ``totals``). tokenspeed: its own split
        count per batch (one wave, at most 32 splits); both PDL chained between our
        kernels."""
        box, reqs = self.dcp, batch // q
        query = self.query_scratch[:batch].view(torch.float8_e4m3fn)
        out, lse, cp = self.latent[:batch], None, {}
        if box is not None:
            query, out = box.q[self.parity, :batch].view(query.dtype), box.rows[:batch]
            lse = box.rows_lse[:batch].view(reqs, q, -1)
            if q > 1:  # D*H heads: packed (row, head) tiles stream the KV once per tile
                cp = dict(cp=(box.size, box.rank, box.interleave), totals=totals[:reqs])
                cp.update(pack=True)
        query, out = query.view(reqs, q, *query.shape[1:]), out.view(reqs, q, -1, 512)
        args = cache, block_table[:reqs], lengths[:reqs], self.max_seq
        if self.cores[batch, q] == "trtllm":
            trtllm_decode(query, *args, self.stock_scales, out)
            return
        stock_decode(query, *args, self.stock_scales, out, lse, **cp)

    def run(
        self, projected, out, *, cache, slots, block_table, lengths, totals=None, q=1
    ):
        """``lengths``/``block_table``: this rank's KV (the local shard under DCP);
        ``totals``: under DCP, the requests' full lengths. ``q`` rows per request (spec
        verify; ``slots`` per row). Rows of ``projected`` beyond the requests run as
        empty requests (zero output)."""
        batch, s = projected.shape[0], stream(self.device)
        reqs = batch // q
        block_table, lengths, totals = pad_requests(reqs, block_table, lengths, totals)
        kv = to_cute(cache.view(torch.uint8))
        mq, v, query, latent = self.kernels[batch, q]
        qnorm, kvnorm, qb, uk, uv, qinv, kinv = self.views
        q_slots, x = to_cute(slots[:batch], 8), to_cute(projected[:, :2112])
        q_args = (x, qnorm, kvnorm, qb, uk, kv, q_slots, qinv, kinv, self.eps, query)
        out_view, gate = to_cute(out), to_cute(projected[:, 2112:])
        lens = to_cute(lengths[:reqs], 4)
        if self.dcp is not None:  # DcpValue zeroes padded requests (global lengths)
            publish, ((rows, rows_lse), boxes, peers), ids = self.dcp_args[batch]
            mq(*q_args, *publish, s)
            self.stock_core(cache, block_table, lengths, batch, q, totals)
            v_args = (rows, rows_lse, lens, *boxes, *peers, publish[0], uv, gate)
            v(*v_args, to_cute(totals[:reqs], 4), out_view, *ids, s)
            return out
        mq(*q_args, s)
        self.stock_core(cache, block_table, lengths, batch, q)
        v(latent, uv, gate, lens, out_view, s)
        return out
