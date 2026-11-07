# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 DSpark Markov sampling on ROCm, sharded over the vocabulary
across TP ranks, all draft positions in one launch.

- Rank k owns markov_w2 rows [k V / TP, (k + 1) V / TP) = the draft lm_head's
  shard, so the base logits are its local output (no all-gather).
- Per position: each program does bias + base + Gumbel-max over its rows; the
  ranks exchange one tagged int64 record (value, epoch tag, token id) through
  uncached peer memory; the max record is the next token.
- The draft logits cache stays full-vocabulary on every rank (peer staging,
  copied at the end of the launch).
- One persistent launch when the programs are co-resident, else one launch a
  position.
- Same noise, rounding, cache contents and tie break as vLLM's loop; only the
  bias's fp32 sum order differs (last bf16 bit). Draft tokens are verified by
  the rejection sampler, so the target's output is unaffected.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import (
    _DRAFT_NOISE_SALT,
    _log1p_neg_stable,
    murmur3_uniform32,
    spec_step_seed,
)

logger = init_logger(__name__)

MAX_ROWS = 8  # requests a launch (the record vectors are MAX_ROWS wide)
MAX_ROWS_PERSISTENT = 4  # larger steps take vLLM's loop
MAX_STEPS = 8
MAX_RANKS = 8
MAX_PROGS = 256  # programs a launch (<= the CUs of MI355X)
# A rank's peer buffer (every rank maps every rank's):
MAIL_OFF = 0  # int64 [MAX_STEPS][MAX_ROWS][MAX_RANKS], the ranks' records
SLOT_OFF = 8192  # int64 [MAX_STEPS][MAX_ROWS][MAX_PROGS], this rank's programs'
STAGING_OFF = SLOT_OFF + MAX_STEPS * MAX_ROWS * MAX_PROGS * 8
# then the staging: bf16 [2 parities][MAX_ROWS][n_spec][vocab], the peers'
# logits shards
# rank-local control words (int64, device memory)
C_EPOCH = 0  # calls so far
C_ERR = 1  # set when a poll gave up
C_WORDS = 8
TAG_SHIFT = 18  # 18 token id bits (vocabulary < 2**18), 14 epoch tag bits
TAG_MASK = 0x3FFF
IDX_MASK = 0x3FFFF
# a wait gives up (and sets C_ERR) after this long, as the specialized
# decode's (mono/abi.py WAIT_SECONDS): the TP ranks reach a call seconds apart
# at worst, a bound on polls gave up within a second
WAIT_SECONDS = 10
WAIT_TICKS = WAIT_SECONDS * 100_000_000  # s_memrealtime ticks at 100 MHz
_MR = tl.constexpr(MAX_ROWS)
_MK = tl.constexpr(MAX_RANKS)
_MP = tl.constexpr(MAX_PROGS)
_MAIL = tl.constexpr(MAIL_OFF)
_SLOT = tl.constexpr(SLOT_OFF)
_STG = tl.constexpr(STAGING_OFF)
_CP = tl.constexpr(C_EPOCH)
_CR = tl.constexpr(C_ERR)
_TS = tl.constexpr(TAG_SHIFT)
_TM = tl.constexpr(TAG_MASK)
_IM = tl.constexpr(IDX_MASK)
_TAG_CLEAR = tl.constexpr(~(TAG_MASK << TAG_SHIFT))
_WAIT = tl.constexpr(WAIT_TICKS)


@triton.jit
def _noise(seed, pos, keys):
    u = murmur3_uniform32(seed, pos + _DRAFT_NOISE_SALT, keys)
    return -tl.log(-_log1p_neg_stable(u))


@triton.jit
def _now():
    """s_memrealtime: the 100 MHz device clock."""
    return tl.inline_asm_elementwise(
        "s_memrealtime $0\n s_waitcnt lgkmcnt(0)",
        "=s",
        [],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _drain():
    """This thread's outstanding memory operations completed (stores
    acknowledged, by a peer GPU too for uncached peer memory), then a
    workgroup barrier."""
    tl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)\n v_mov_b32 $0, 0",
        "=v",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tl.debug_barrier()


@triton.jit
def _record(value, idx, tag):
    bits = value.to(tl.int32, bitcast=True)
    ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    return (ordered.to(tl.int64) << 32) | (tag << _TS) | (_IM - idx.to(tl.int64))


@triton.jit
def _poll(ptr, offs, mask, tag, err_ptr):
    """Re-load the records at ``ptr + offs`` until every masked one carries
    ``tag``; return them with the tag cleared (INT64_MIN where masked)."""
    rec = tl.atomic_add(ptr + offs, 0, mask=mask, sem="relaxed", scope="sys")
    bad = tl.sum((mask & (((rec >> _TS) & _TM) != tag)).to(tl.int32))
    t0 = _now()
    late = False
    while (bad > 0) & (not late):
        rec = tl.atomic_add(ptr + offs, 0, mask=mask, sem="relaxed", scope="sys")
        bad = tl.sum((mask & (((rec >> _TS) & _TM) != tag)).to(tl.int32))
        late = _now() - t0 > _WAIT
    if bad > 0:
        tl.atomic_xchg(err_ptr, 1, sem="relaxed", scope="gpu")
    return tl.where(mask, rec & _TAG_CLEAR, -9223372036854775807 - 1)


@triton.jit
def _winners(mail_ptr, step, rv, tag, err_ptr, WORLD: tl.constexpr, WP2: tl.constexpr):
    """The tokens chosen at ``step`` for every row (one poll for all rows)."""
    j = tl.arange(0, WP2)
    ridx = tl.arange(0, _MR)
    rec = _poll(
        mail_ptr,
        (step * _MR + ridx[:, None]) * _MK + j[None, :],
        rv[:, None] & (j < WORLD)[None, :],
        tag,
        err_ptr,
    )
    return _IM - (tl.max(rec, axis=1) & _IM)


@triton.jit
def _shard_tile(
    wbits, z, e, base, v, vm, cache_ptr, valid, noise, temp, peerv, wj, rank,
    stage_off, HAS_CACHE: tl.constexpr, WORLD: tl.constexpr,
):  # fmt: skip
    """The logits of one register tile of this rank's vocabulary rows (token
    ids ``v``), stored into the local cache and every peer's staging; returns
    the tile's Gumbel-max."""
    w = ((wbits + z).to(tl.int32) << 16).to(tl.float32, bitcast=True)
    acc = tl.sum(w * e[None, :], axis=1)
    bias = acc.to(tl.bfloat16).to(tl.float32)
    logits = (base + bias).to(tl.bfloat16)
    if HAS_CACHE:
        cmask = vm & valid
        tl.store(cache_ptr + v, logits.to(cache_ptr.dtype.element_ty), mask=cmask)
        for p in tl.static_range(WORLD):
            if p != rank:
                pb = tl.sum(tl.where(wj == p, peerv, 0)) + stage_off
                tl.store(pb.to(tl.pointer_type(tl.bfloat16)) + v, logits, mask=cmask)
    lf = tl.where(vm, logits.to(tl.float32), float("-inf"))
    if temp != 0.0:
        lf = lf / temp
        lf = tl.where(vm, lf + noise, float("-inf"))
    return tl.max(lf, axis=0, return_indices=True)


@triton.jit
def _sharded_markov_kernel(
    input_ids_ptr, anchor_idx_ptr, w1_ptr, w2_ptr,
    base_ptr, base_s0, base_s1, base_col,
    cache_ptr, cache_s0, cache_s1,
    idx_map_ptr, idx_stride, pos_ptr, pos_stride, temp_ptr, seeds_ptr,
    tokens_ptr, tokens_stride, emb_ptr, emb_s0,
    ctrl_ptr, peers_ptr, zeros_ptr,
    rank, voff, vlocal, vocab, num_progs,
    S0: tl.constexpr, S1: tl.constexpr, N_SPEC: tl.constexpr,
    ROWS: tl.constexpr, RANK: tl.constexpr, TA: tl.constexpr, TB: tl.constexpr,
    WORLD: tl.constexpr, WP2: tl.constexpr, CH: tl.constexpr,
    HAS_CACHE: tl.constexpr, WRITE_EMB: tl.constexpr,
):  # fmt: skip
    """Positions [S0, S1) of one call on one rank (S0 = 0, S1 = N_SPEC: one
    persistent launch; S1 = S0 + 1: a launch a position). Each program owns
    TA + TB consecutive rows of this rank's vocabulary shard and stores its
    tagged best record a row into its slot; program 0 collects the slots and
    publishes the rank's record into every rank's mailbox; every program reads
    the next token from its own rank's mailbox. The launch with S1 == N_SPEC
    also copies the peers' cache shards out of the staging."""
    b = tl.program_id(0)
    k = tl.arange(0, RANK)
    ridx = tl.arange(0, _MR)
    lo = b * (TA + TB)
    lA = lo + tl.arange(0, TA)
    mA = lA < vlocal
    vA = voff + lA  # token ids
    wA = tl.load(
        w2_ptr + vA[:, None] * RANK + k[None, :], mask=mA[:, None], other=0.0
    ).to(tl.int16, bitcast=True)
    if TB > 0:
        lB = lo + TA + tl.arange(0, TB)
        mB = lB < vlocal
        vB = voff + lB
        wB = tl.load(
            w2_ptr + vB[:, None] * RANK + k[None, :], mask=mB[:, None], other=0.0
        ).to(tl.int16, bitcast=True)
    # the call's epoch: the S0 == 0 launch bumps it (program 0, once every
    # program has read it, i.e. after the first slot collection)
    epoch = tl.load(ctrl_ptr + _CP)
    if S0 == 0:
        epoch = epoch + 1
    tag = epoch & _TM
    parity = epoch & 1
    my_buf = tl.load(peers_ptr + rank)
    mail_ptr = (my_buf + _MAIL).to(tl.pointer_type(tl.int64))
    slot_ptr = (my_buf + _SLOT).to(tl.pointer_type(tl.int64))
    err_ptr = ctrl_ptr + _CR
    wj = tl.arange(0, WP2)
    peerv = tl.load(peers_ptr + wj, mask=wj < WORLD, other=0)
    rv = ridx < ROWS
    for step in tl.static_range(S0, S1):
        recs = tl.full([_MR], 0, tl.int64) | (tag << _TS)  # rows >= ROWS: dummies
        # every load of the position first (vectors over the rows): a load
        # issued after a peer store would wait for its acknowledgement
        reqv = tl.load(idx_map_ptr + ridx * idx_stride + step, mask=rv, other=-1).to(
            tl.int64
        )
        validv = reqv >= 0
        tempv = tl.load(temp_ptr + reqv, mask=validv, other=0.0).to(tl.float32)
        seedv = tl.load(seeds_ptr + reqv, mask=validv, other=0)
        if HAS_CACHE:
            seedv = spec_step_seed(seedv, step)
        posv = tl.load(pos_ptr + ridx * pos_stride + step, mask=rv, other=1) - 1
        # zeros the compiler cannot see through, added to the w2 tile's bits
        zv = tl.load(zeros_ptr + step * _MR + ridx).to(tl.int16)
        bptr = base_ptr + ridx[:, None] * base_s0 + step * base_s1 + base_col
        bA2 = tl.load(bptr + lA[None, :], mask=rv[:, None] & mA[None, :], other=0.0).to(
            tl.float32
        )
        if TB > 0:
            bB2 = tl.load(
                bptr + lB[None, :], mask=rv[:, None] & mB[None, :], other=0.0
            ).to(tl.float32)
        if step == 0:
            prevv = tl.load(
                input_ids_ptr + tl.load(anchor_idx_ptr + ridx, mask=rv, other=0),
                mask=rv,
                other=0,
            ).to(tl.int64)
        else:
            prevv = _winners(mail_ptr, step - 1, rv, tag, err_ptr, WORLD, WP2)
            if b == 0:
                tl.store(tokens_ptr + ridx * tokens_stride + (step - 1), prevv, mask=rv)
        e2 = tl.load(
            w1_ptr + prevv[:, None] * RANK + k[None, :], mask=rv[:, None], other=0.0
        )
        if WRITE_EMB:  # noqa: SIM102 (a constexpr, then a traced branch)
            if b == 0:
                tl.store(
                    emb_ptr + ridx[:, None] * emb_s0 + step * RANK + k[None, :],
                    e2,
                    mask=rv[:, None],
                )
        e2 = e2.to(tl.float32)
        for r in tl.static_range(ROWS):
            sel = ridx == r
            req = tl.sum(tl.where(sel, reqv, 0))
            valid = req >= 0
            temp = tl.sum(tl.where(sel, tempv, 0.0))
            seed = tl.sum(tl.where(sel, seedv, 0))
            pos = tl.sum(tl.where(sel, posv, 0))
            z = tl.sum(tl.where(sel, zv, 0)).to(tl.int16)
            e = tl.sum(tl.where(sel[:, None], e2, 0.0), axis=0)
            bA = tl.sum(tl.where(sel[:, None], bA2, 0.0), axis=0)
            nA = _noise(seed, pos, vA)
            cptr = cache_ptr + req * cache_s0 + step * cache_s1
            stage_off = _STG + (((parity * _MR + r) * N_SPEC + step) * vocab * 2)
            best_v, best_i = _shard_tile(
                wA, z, e, bA, vA, mA, cptr, valid, nA, temp, peerv, wj, rank,
                stage_off, HAS_CACHE, WORLD,
            )  # fmt: skip
            best_i = voff + lo + best_i
            if TB > 0:
                bB = tl.sum(tl.where(sel[:, None], bB2, 0.0), axis=0)
                nB = _noise(seed, pos, vB)
                vb, ib = _shard_tile(
                    wB, z, e, bB, vB, mB, cptr, valid, nB, temp, peerv, wj, rank,
                    stage_off, HAS_CACHE, WORLD,
                )  # fmt: skip
                take = vb > best_v
                best_v = tl.where(take, vb, best_v)
                best_i = tl.where(take, voff + lo + TA + ib, best_i)
            recs = tl.where(sel, _record(best_v, best_i, tag), recs)
        if HAS_CACHE and WORLD > 1 and step == N_SPEC - 1:
            # every staging store of this call acknowledged before the last
            # record: a peer's tail copy starts once it holds all of them
            _drain()
        tl.store(slot_ptr + (step * _MR + ridx) * _MP + b, recs)
        if b == 0:
            # collect this rank's programs, publish to every rank's mailbox
            pj = tl.arange(0, _MP)
            sl = _poll(
                slot_ptr,
                (step * _MR + ridx[:, None]) * _MP + pj[None, :],
                rv[:, None] & (pj < num_progs)[None, :],
                tag,
                err_ptr,
            )
            outs = (tl.max(sl, axis=1) & _TAG_CLEAR) | (tag << _TS)
            for p in tl.static_range(WORLD):
                pmb = tl.sum(tl.where(wj == p, peerv, 0)) + _MAIL
                pm = pmb.to(tl.pointer_type(tl.int64))
                tl.store(pm + (step * _MR + ridx) * _MK + rank, outs)
            if step == 0:
                tl.store(ctrl_ptr + _CP, epoch)
    if S1 == N_SPEC:
        # the last tokens; this program's share of the peers' cache shards
        tok = _winners(mail_ptr, N_SPEC - 1, rv, tag, err_ptr, WORLD, WP2)
        if b == 0:
            tl.store(tokens_ptr + ridx * tokens_stride + (N_SPEC - 1), tok, mask=rv)
        if HAS_CACHE and WORLD > 1:
            c = b * CH + tl.arange(0, CH)
            cm = (c < vocab) & ((c < voff) | (c >= voff + vlocal))
            stage = (my_buf + _STG).to(tl.pointer_type(tl.bfloat16))
            for r in tl.static_range(ROWS):
                for step in tl.static_range(N_SPEC):
                    req = tl.load(idx_map_ptr + r * idx_stride + step).to(tl.int64)
                    m = cm & (req >= 0)
                    x = tl.load(
                        stage + ((parity * _MR + r) * N_SPEC + step) * vocab + c,
                        mask=m,
                    )
                    tl.store(
                        cache_ptr + req * cache_s0 + step * cache_s1 + c,
                        x.to(cache_ptr.dtype.element_ty),
                        mask=m,
                    )


class ShardedMarkov:
    """Peer buffers + launcher on one TP rank. The constructor is collective
    over ``cpu_group`` (before graph capture); every rank calls ``sample``
    the same number of times with the same rows."""

    def __init__(
        self, w1, w2, n_spec, cpu_group, rank, world, device, vocab_shard=None
    ):
        from vllm.models.kimi_k3.amd.specialized_decode.runtime.peer import (
            PeerBuffer,
        )

        vocab, rk = w2.shape
        assert vocab <= IDX_MASK and 1 <= n_spec <= MAX_STEPS and world <= MAX_RANKS
        assert w1.dtype == w2.dtype == torch.bfloat16
        assert w1.is_contiguous() and w2.is_contiguous() and w1.shape[1] == rk
        self.w1, self.w2 = w1, w2
        self.vocab, self.rank_dim, self.n_spec = vocab, rk, n_spec
        self.rank, self.world = rank, world
        if vocab_shard is None:
            assert vocab % world == 0
            vocab_shard = (rank * vocab // world, vocab // world)
        self.voff, self.vlocal = vocab_shard
        ncu = torch.cuda.get_device_properties(device).multi_processor_count
        grid = min(ncu, MAX_PROGS)
        tok = triton.cdiv(self.vlocal, grid)
        # (rows of tile A, of tile B, programs, tail copy chunk) of each
        # launch mode; the persistent launch only if its programs are
        # co-resident (they wait on each other)
        self.geom = {}
        for mode in ("persistent", "round"):
            ta = triton.next_power_of_2(tok) // 2 if tok & (tok - 1) else tok
            tb = 0 if ta == tok else triton.next_power_of_2(tok - ta)
            if mode == "round" and tb:  # streamed a launch: one tile is enough
                ta, tb = triton.next_power_of_2(tok), 0
            nprog = triton.cdiv(self.vlocal, ta + tb)
            assert nprog <= MAX_PROGS
            if mode == "persistent" and nprog > ncu:
                continue
            ch = triton.next_power_of_2(triton.cdiv(vocab, nprog))
            self.geom[mode] = (ta, tb, nprog, ch)
        self.ctrl = torch.zeros(C_WORDS, dtype=torch.int64, device=device)
        self.zeros = torch.zeros(MAX_STEPS * MAX_ROWS, dtype=torch.int32, device=device)
        nbytes = STAGING_OFF + 2 * MAX_ROWS * n_spec * vocab * 2
        self.peer = PeerBuffer(nbytes, cpu_group, rank, world, device)
        self.peer.bytes.zero_()
        torch.accelerator.synchronize()
        if cpu_group is not None:
            torch.distributed.barrier(group=cpu_group)
        self.peers = self.peer.addresses

    def failed(self) -> bool:
        """Whether a wait gave up (a peer never published): a sync."""
        return bool(self.ctrl[C_ERR].item())

    def mode(self, rows: int) -> str:
        """Persistent at TP 8 or one row, else a launch a position."""
        m = "persistent" if rows == 1 or self.world >= 8 else "round"
        return m if m in self.geom else "round"

    def sample(
        self, base_logits, base_col, input_ids, anchor_idx, idx_map, sample_pos,
        temperature, seeds, draft_logits, draft_tokens, markov_embeds_out=None,
        mode=None, warmup=False,
    ):  # fmt: skip
        """``base_logits`` [rows, n_spec, Vb] bf16 holds this rank's vocabulary
        row ``voff`` at column ``base_col`` (``voff`` for the full logits, 0
        for the lm_head's local shard). Writes ``draft_tokens`` [rows, n_spec],
        the full-vocabulary ``draft_logits`` rows, and the Markov embeddings
        into ``markov_embeds_out`` [rows, n_spec, rank] if given. ``warmup``:
        only compile the launches' kernels (no launch, not collective)."""
        rows, n_spec, _ = base_logits.shape
        mode = mode or self.mode(rows)
        ta, tb, nprog, ch = self.geom[mode]
        assert n_spec == self.n_spec and 1 <= rows <= MAX_ROWS
        assert base_logits.stride(-1) == 1 and draft_tokens.stride(1) == 1
        has_cache = draft_logits is not None
        cache = draft_logits if has_cache else base_logits
        write_emb = markov_embeds_out is not None
        emb = markov_embeds_out if write_emb else self.w1
        common = dict(
            N_SPEC=n_spec,
            ROWS=rows,
            RANK=self.rank_dim,
            TA=ta,
            TB=tb,
            WORLD=self.world,
            WP2=max(2, triton.next_power_of_2(self.world)),
            CH=ch,
            HAS_CACHE=has_cache,
            WRITE_EMB=write_emb,
            num_warps=8,
        )
        args = (
            input_ids, anchor_idx, self.w1, self.w2,
            base_logits, base_logits.stride(0), base_logits.stride(1), base_col,
            cache, cache.stride(0) if has_cache else 0,
            cache.stride(1) if has_cache else 0,
            idx_map, idx_map.stride(0), sample_pos, sample_pos.stride(0),
            temperature, seeds, draft_tokens, draft_tokens.stride(0),
            emb, emb.stride(0) if write_emb else 0,
            self.ctrl, self.peers, self.zeros,
            self.rank, self.voff, self.vlocal, self.vocab, nprog,
        )  # fmt: skip
        spans = [(0, n_spec)] if mode == "persistent" else [
            (s, s + 1) for s in range(n_spec)
        ]  # fmt: skip
        for s0, s1 in spans:
            if warmup:
                _sharded_markov_kernel.warmup(*args, S0=s0, S1=s1, grid=(1,), **common)
            else:
                _sharded_markov_kernel[(nprog,)](*args, S0=s0, S1=s1, **common)


def _why_not(spec) -> str | None:
    """Why this speculator keeps vLLM's loop, or None. Configuration only, so
    the same on every rank."""
    model = spec.model
    head = getattr(getattr(model, "model", None), "markov_head", None)
    if head is None:
        return "no Markov head"
    if spec.draft_logits is None:
        return "greedy drafts"
    if spec.draft_logits.row_mapping is not None:
        return "a batch-sharded draft logits cache"
    if spec._draft_topk is not None:
        return "top-k drafting"
    if spec.draft_watermarker is not None:
        return "a draft watermarker"
    if spec._d2t_scatter_index is not None:
        return "a reduced draft vocabulary"
    if spec.acceptance_estimator is not None:
        return "the acceptance estimator"
    if hasattr(model, "apply_output_logit_transform"):
        return "an output logit transform"
    if spec.use_fp64_gumbel:
        return "fp64 Gumbel noise"
    w1, w2 = head.markov_w1.weight, head.markov_w2.weight
    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        return "Markov weights not bf16"
    if w1.shape[1] != w2.shape[1] or not (w1.is_contiguous() and w2.is_contiguous()):
        return "the Markov weights' layout"
    if head.markov_w2.quant_method.__class__.__name__ != "UnquantizedEmbeddingMethod":
        return "a quantized Markov head"
    lp = model.logits_processor
    if getattr(lp, "scale", 1.0) != 1.0 or getattr(lp, "soft_cap", None) is not None:
        return "a scaled or soft-capped logits processor"
    if spec.draft_logits.values.dtype != torch.bfloat16:
        return f"a {spec.draft_logits.values.dtype} draft logits cache"
    if w2.shape[0] > IDX_MASK:
        return "a vocabulary of 2**18 or more"
    if spec.num_speculative_steps > MAX_STEPS:
        return f"more than {MAX_STEPS} drafts"
    if _lm_head_shard(model) is None:
        return "no evenly sharded draft lm_head"
    return None


def _lm_head_shard(model) -> tuple[int, int] | None:
    """(first column, width) of this rank's draft lm_head logits, or None when
    the lm_head is not vocabulary-parallel without padding."""
    lm = getattr(model, "lm_head", None)
    si = getattr(lm, "shard_indices", None)
    if si is None or getattr(lm, "tp_size", 1) <= 1:
        return None
    if si.num_added_elements or si.num_org_elements != si.num_elements_padded:
        return None
    return int(si.org_vocab_start_index), int(si.num_org_elements)


class ShardedMarkovSampler:
    """The draft model's ``sample_sequential`` (the speculator asks it before
    its own loop): the sharded chain fed by the lm_head's local logits."""

    def __init__(self, spec, rank, world, cpu_group):
        from vllm.models.kimi_k3.amd.specialized_decode.runtime.guards import (
            FailStop,
        )

        model = spec.model
        head = model.model.markov_head
        w1, w2 = head.markov_w1.weight, head.markov_w2.weight
        self.sharded = ShardedMarkov(
            w1, w2, spec.num_speculative_steps, cpu_group, rank, world, w1.device,
            vocab_shard=_lm_head_shard(model),
        )  # fmt: skip
        # Compile every row count's kernels now: warm-up runs only some batch
        # sizes, and eager drafts (decode graphs not FULL) run the unpadded
        # one. A rank compiling mid-serving would leave its peers polling for
        # its records until they give up.
        # (the speculator's buffers bound the batch: max_num_seqs requests)
        max_reqs = spec.sample_idx_mapping.numel() // spec.num_speculative_steps
        for rows in range(1, min(MAX_ROWS_PERSISTENT, max_reqs) + 1):
            self._launch(spec, rows, None, warmup=True)
        torch.distributed.barrier(group=cpu_group)
        self.failstop = FailStop(
            self.sharded.ctrl[C_ERR : C_ERR + 1], rank, world, cpu_group,
            what="Kimi-K3 DSpark sharded Markov sampling", remedy="",
        )  # fmt: skip
        logger.info(
            "DSpark Markov sampling sharded over %d ranks: vocabulary rows "
            "[%d, +%d), launch geometry %s",
            world, self.sharded.voff, self.sharded.vlocal, self.sharded.geom,
        )  # fmt: skip

    @classmethod
    def create(cls, spec) -> ShardedMarkovSampler | None:
        """The sampler, or None for vLLM's loop. Collective over the TP group:
        every rank calls it at the same point, before CUDA graph capture."""
        from vllm.distributed import get_tp_group
        from vllm.models.kimi_k3.amd.specialized_decode.runtime.peer import tp_agree

        tp = get_tp_group()
        world, rank, cpu_group = tp.world_size, tp.rank_in_group, tp.cpu_group
        reason = "TP 1" if world <= 1 else _why_not(spec)
        if reason is None and world > MAX_RANKS:
            reason = f"TP above {MAX_RANKS}"
        if not tp_agree(reason is None, cpu_group):
            logger.info(
                "DSpark sharded Markov sampling off: %s", reason or "another rank"
            )
            return None
        return cls(spec, rank, world, cpu_group)

    def __call__(self, spec, num_reqs: int, head_hidden: torch.Tensor) -> bool:
        # num_reqs is the padded batch, the same on every TP rank, so every
        # rank takes the same (collective) path
        if not 1 <= num_reqs <= MAX_ROWS_PERSISTENT:
            return False
        n_spec = spec.num_speculative_steps
        num_sample = num_reqs * n_spec
        model = spec.model
        sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
        # this rank's vocabulary shard of the base logits: no all-gather
        base = model.logits_processor(model.lm_head, sample_hidden, skip_gather=True)
        if base.dtype != torch.bfloat16 or base.stride(-1) != 1:
            raise RuntimeError("DSpark sharded Markov sampling needs bf16 base logits")
        emb = self._launch(spec, num_reqs, base.view(num_reqs, n_spec, base.shape[-1]))
        if spec.use_confidence_head:
            assert emb is not None
            conf = model.compute_confidence(sample_hidden, emb.flatten(0, 1))
            spec.draft_token_confidence_probs[:num_reqs] = conf.view(num_reqs, n_spec)
        return True

    def _launch(self, spec, num_reqs, base, warmup=False):
        """The chain over ``base`` [num_reqs, n_spec, vlocal] (``warmup``: a
        stand-in of its layout, compile only); returns the Markov embeddings
        buffer, or None without the confidence head."""
        n_spec = spec.num_speculative_steps
        num_sample = num_reqs * n_spec
        device = spec.draft_tokens.device
        if base is None:
            base = torch.empty(
                num_reqs, n_spec, self.sharded.vlocal, dtype=torch.bfloat16,
                device=device,
            )  # fmt: skip
        emb = None
        if spec.use_confidence_head:
            w1 = spec.model.model.markov_head.markov_w1.weight
            emb = torch.empty(
                num_reqs, n_spec, w1.shape[1], dtype=w1.dtype, device=device
            )
        self.sharded.sample(
            base,
            0,
            spec.input_buffers.input_ids,
            spec._anchor_idx,
            spec.sample_idx_mapping[:num_sample].view(num_reqs, n_spec),
            spec.sample_pos[:num_sample].view(num_reqs, n_spec),
            spec.temperature,
            spec.seeds,
            spec.draft_logits.values,  # replicated: row = request state
            spec.draft_tokens,
            emb,
            warmup=warmup,
        )
        return emb

    def after_step(self) -> None:
        """Once per draft step on every TP rank (``DSparkSpeculator.propose``):
        raises on every rank once a rank gave up waiting for a peer's record.
        The draft tokens of that call are then unreliable (the rejection
        sampler keeps the output correct) and the ranks are out of step."""
        self.failstop.after_step()
