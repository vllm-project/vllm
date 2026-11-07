# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-round DCP decode top-k merge for the DSA indexer (CuTe DSL).

Under DCP with interleave 1, rank `r` owns global token `g` iff `g % W == r` and scores
only its local tokens `g // W`. The global top-K is merged from per-rank candidates:

1. Each rank takes its local top-`N` (`N` ~ 1.5x the fair share `K/W`)
   from the bf16 logits and packs every candidate into one 32-bit word,
   `(ord16(score) << 16) | (0xFFFF - local[15:0])`, plus a bit plane
   holding `local[16]` (one word per 32 candidates). Zero is padding.
2. The `[N + N/32]` int32 rows are all-gathered over the DCP group (`K / N` times fewer
   bytes than the 8-byte `(score, id)` pairs of the generic merge).
3. Every rank radix-selects the top-`K` of the `W * N` candidates under
   (bf16 score desc, global id asc) and writes their global ids.

The merge is exact: on every rank the selected scores equal, as a multiset, the top-`K`
of the union of all ranks' bf16 logits (equal scores may resolve to either token; the
choice is a function of the gathered words, so all ranks agree). Round 1 can only miss a
token of a *saturated* rank: one that sent `N` words, all strictly above the `K`-th
selected key `tau`. So the select also writes per row
`sat_word = (mask << 16) | ord16(tau)`, mask bit `r` = rank `r` is saturated (identical
on every rank). If the mask is 0, round 1 is exact. Otherwise (at most `(K - 1) // N`
saturated ranks), each saturated rank pushes its local top-`K` of the row into every
rank's symmetric receive slot (P2P stores, then a release flag) from inside its select,
which never waits; a persistent rescue kernel then re-selects the row over the other
ranks' round-1 words at or above `tau` plus the pushed lists. The second round is exact
without a third: every token it cannot see scores at most the `K`-th of what it sees.
Unsaturated rows cost the rescue one mask load.

The rescue state (`DCPTopkRescueWorkspace`) is built once per DCP group by
`init_dcp_topk_merge`; the entry points refuse to run without it. Every DCP
rank must call every merge, the merges of a step must be stream-ordered, and
no rank may start the round-1 exchange of call `n + 1` before its rescue of
call `n` has finished (true for the all-gather here).

Non-finite logits never cause an out-of-bounds access: the pack drops local ids outside
`[0, min(row_end, S, 2**17))` (DeepSelect returns arbitrary ids for a row holding a NaN)
and candidates with a NaN score, and the select never pushes past a row's rescue slots;
+-inf scores are ordinary candidates (+inf first, -inf last, ties by global id). A row
holding a NaN is outside the exactness contract: every rank still selects the same set,
but it may miss top tokens of the rank whose local top-N the NaN disturbed.
"""

import functools
from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Int32, Int64, Uint16, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import recast_val
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import (
    CuTeDSLLaunchSpec,
    VllmCuTeDSLJitKernel,
)
from vllm.models.kimi_k3.nvidia.ops.dcp_topk_merge import (
    LOCAL_INDEX_BITS,
    packed_row_words,
    rescue_slots,
)

_WARP = 32


@cute.jit
def _block_scan_inclusive(
    val: Int32, lane: Int32, warp: Int32, warp_totals: cute.Tensor, num_warps: int
) -> Int32:
    """Inclusive block-wide sum; `warp_totals` is smem [num_warps]."""
    for i in cutlass.range_constexpr(5):
        partial = cute.arch.shuffle_sync_up(val, offset=1 << i, mask_and_clamp=0)
        if lane >= Int32(1 << i):
            val += partial
    if lane == Int32(_WARP - 1):
        warp_totals[warp] = val
    cute.arch.sync_threads()
    if warp == Int32(0):
        total = Int32(0)
        if lane < Int32(num_warps):
            total = warp_totals[lane]
        prefix = total
        for i in cutlass.range_constexpr(5):
            partial = cute.arch.shuffle_sync_up(prefix, offset=1 << i, mask_and_clamp=0)
            if lane >= Int32(1 << i):
                prefix += partial
        if lane < Int32(num_warps):
            warp_totals[lane] = prefix - total
    cute.arch.sync_threads()
    return val + warp_totals[warp]


class DCPTopkPackKernel(VllmCuTeDSLJitKernel["DCPTopkPackKernel.CompileKey"]):
    """Local top-N indices (-1 padded) + bf16 logits -> packed candidate rows.

    An index outside `[0, min(row_end, S, 2**17))` (`row_end` from
    `row_ends` when given) or a NaN score packs as an empty slot."""

    max_tb_size = 512

    @dataclass(frozen=True)
    class CompileKey:
        local_topk: int
        has_row_ends: bool = False

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        local_topk = compile_key.local_topk
        has_row_ends = compile_key.has_row_ends
        assert local_topk % _WARP == 0
        tb_size = min(local_topk, DCPTopkPackKernel.max_tb_size)
        num_chunks = -(-local_topk // tb_size)

        @cute.kernel
        def device_kernel(
            logits: cute.Tensor,
            local_idx: cute.Tensor,
            packed: cute.Tensor,
            row_ends: cute.Tensor,
        ):
            row, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            # Ids past the row (DeepSelect's output for a NaN row) are dropped.
            end = Int32(logits.shape[1])
            if end > Int32(1 << LOCAL_INDEX_BITS):
                end = Int32(1 << LOCAL_INDEX_BITS)
            if cutlass.const_expr(has_row_ends):
                row_end = row_ends[row]
                if row_end < end:
                    end = row_end
            for c in cutlass.range_constexpr(num_chunks):
                # local_topk is warp-aligned, so whole warps take this branch.
                j = tid + Int32(c * tb_size)
                if j < Int32(local_topk):
                    local = local_idx[row, j]
                    word = Uint32(0)
                    high = False
                    if local >= Int32(0) and local < end:
                        bits = Uint32(recast_val(logits[row, local], Uint16))
                        # NaN scores are dropped (no order).
                        if (bits & Uint32(0x7FFF)) <= Uint32(0x7F80):
                            # Order-preserving bf16 key: flip negatives, set
                            # the sign bit of non-negatives.
                            ordered = bits | Uint32(0x8000)
                            if (bits & Uint32(0x8000)) != Uint32(0):
                                ordered = ~bits & Uint32(0xFFFF)
                            local_u = recast_val(local, Uint32)
                            word = (ordered << Uint32(16)) | (
                                Uint32(0xFFFF) - (local_u & Uint32(0xFFFF))
                            )
                            high = (local_u >> Uint32(16)) != Uint32(0)
                    packed[row, j] = recast_val(word, Int32)
                    high_bits = cute.arch.vote_ballot_sync(high)
                    if j % Int32(_WARP) == Int32(0):
                        packed[row, Int32(local_topk) + j // Int32(_WARP)] = high_bits

        @cute.jit
        def host_entrypoint(
            logits: cute.Tensor,
            local_idx: cute.Tensor,
            packed: cute.Tensor,
            row_ends: cute.Tensor,
            stream,
        ):
            # Without has_row_ends ``row_ends`` is a placeholder.
            device_kernel(logits, local_idx, packed, row_ends).launch(
                grid=(packed.shape[0], 1, 1),
                block=(tb_size, 1, 1),
                stream=stream,
            )

        return host_entrypoint

    def dispatch(  # type: ignore[override]
        self, *, local_topk: int, has_row_ends: bool = False
    ) -> CompileKey:
        return self.CompileKey(local_topk=local_topk, has_row_ends=has_row_ends)

    def get_warmup_keys(
        self, *, local_topk: int, has_row_ends: bool = False
    ) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(
            local_topk=local_topk, has_row_ends=has_row_ends
        )

    def warmup_inputs(self, compile_key: CompileKey) -> tuple[Any, ...]:
        num_rows = cute.sym_int()
        local_idx = make_fake_tensor(
            Int32, (num_rows, compile_key.local_topk), divisibility=1
        )
        return (
            make_fake_tensor(BFloat16, (num_rows, cute.sym_int()), divisibility=1),
            local_idx,
            make_fake_tensor(
                Int32,
                (num_rows, packed_row_words(compile_key.local_topk)),
                divisibility=1,
            ),
            make_fake_tensor(Int32, (cute.sym_int(),))
            if compile_key.has_row_ends
            else local_idx,
        )

    @kernel_launcher
    def __call__(
        self,
        logits: torch.Tensor,
        local_idx: torch.Tensor,
        packed: torch.Tensor,
        row_ends: torch.Tensor | None = None,
    ) -> CuTeDSLLaunchSpec[CompileKey]:
        compile_key = self.dispatch(
            local_topk=local_idx.shape[1], has_row_ends=row_ends is not None
        )
        return compile_key, (
            logits,
            local_idx,
            packed,
            local_idx if row_ends is None else row_ends,
        )


# Device helpers for the exact-merge rescue (raw-address loads/stores with
# explicit memory semantics; the peer buffers are symmetric memory).


def _asm(result_type, operands, asm: str, constraints: str, *, loc, ip):
    return llvm.inline_asm(
        result_type,
        [operand.ir_value(loc=loc, ip=ip) for operand in operands],
        asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


def _asm_void(operands, asm: str, constraints: str, *, loc, ip) -> None:
    # A dummy result keeps the side-effecting asm in the DSL's value graph.
    _asm(
        T.i32(),
        operands,
        "mov.u32 $0, 0;\n\t" + asm,
        ",".join(("=r", constraints)) if constraints else "=r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_i64(addr: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(
        _asm(T.i64(), [Int64(addr)], "ld.global.u64 $0, [$1];", "=l,l", loc=loc, ip=ip)
    )


@dsl_user_op
def _ld_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(T.i32(), [Int64(addr)], "ld.global.u32 $0, [$1];", "=r,l", loc=loc, ip=ip)
    )


@dsl_user_op
def _st_v4(addr: Int64, v0, v1, v2, v3, *, loc=None, ip=None) -> None:
    _asm_void(
        [Int64(addr), *(Int32(v) for v in (v0, v1, v2, v3))],
        "st.global.v4.u32 [$1], {$2, $3, $4, $5};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_acquire_sys(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr)],
            "ld.acquire.sys.global.u32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_release_sys(addr: Int64, value: Int32, *, loc=None, ip=None) -> None:
    _asm_void(
        [Int64(addr), Int32(value)],
        "st.release.sys.global.u32 [$1], $2;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _globaltimer(*, loc=None, ip=None) -> Int64:
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc=loc, ip=ip))


@dsl_user_op
def _nanosleep(*, loc=None, ip=None) -> None:
    _asm_void([], "nanosleep.u32 64;", "", loc=loc, ip=ip)


@dsl_user_op
def _trap(*, loc=None, ip=None) -> None:
    _asm_void([], "trap;", "", loc=loc, ip=ip)


_TB_SIZE = 512
_RADIX_BITS = 9
_NUM_BINS = 1 << _RADIX_BITS
# A rescue waits on peers' selects, which may run late (host stalls, a JIT
# compile in eager mode); only a peer that never pushes (a bug) reaches this.
_SPIN_TIMEOUT_NS = 600 * 10**9


def _gid_bits(world_size: int) -> int:
    return (world_size * (1 << LOCAL_INDEX_BITS) - 1).bit_length()


@cute.jit
def _popc64(bits: Uint64) -> Int32:
    return recast_val(
        cute.arch.popc(Uint32(bits & Uint64(0xFFFFFFFF)))
        + cute.arch.popc(Uint32(bits >> Uint64(32))),
        Int32,
    )


@cute.jit
def _ord16(bits: Uint32) -> Uint32:
    """Order-preserving bf16 key: flip negatives, set the sign bit of
    non-negatives."""
    ordered = bits | Uint32(0x8000)
    if (bits & Uint32(0x8000)) != Uint32(0):
        ordered = ~bits & Uint32(0xFFFF)
    return ordered


def _make_radix_select(
    *,
    topk: int,
    world_size: int,
    keys_per_thread: int,
    emit_ids: bool,
    block_size: int,
) -> Any:
    """Top-`topk` of the CTA's register keys `(ord16 << gid_bits) | ~gid`
    (0 = empty) under (score desc, global id asc): writes the ids and/or this
    rank's owned physical rows for `row` and returns the thread's smallest
    selected ord16 (0x10000 when none). The select and the rescue share it, so
    both emit bit-identical rows for the same candidate set."""
    emit_rows = block_size > 0
    # One selected-key bit per register key, in 64-bit words.
    num_sel_words = -(-keys_per_thread // 64)
    tb_size = _TB_SIZE
    radix_bits = _RADIX_BITS
    num_bins = _NUM_BINS
    num_warps = tb_size // _WARP
    gid_bits = _gid_bits(world_size)
    gid_mask = (1 << gid_bits) - 1
    num_passes = -(-(16 + gid_bits) // radix_bits)

    @cute.jit
    def radix_select(
        keys: cute.Tensor,
        row: Int32,
        tid: Int32,
        lane: Int32,
        warp: Int32,
        hist: cute.Tensor,
        warp_totals: cute.Tensor,
        scalars: cute.Tensor,
        prefix_ptr: cute.Pointer,
        out: cute.Tensor,
        req_id: cute.Tensor,
        block_table: cute.Tensor,
        rows_out: cute.Tensor,
        valid_out: cute.Tensor,
        dcp_rank: Int32,
        stride_rows: Int32,
    ) -> Int32:
        # scalars: [committed, owned, threshold, take_all]
        sel_min = Int32(0x10000)
        # Bit i % 64 of word i // 64: key i is selected. The ids and rows are
        # written after the passes, in (thread, key) order at block-scan
        # offsets, so they come out in the same order on every run (warp-order
        # atomics would vary it, and with it the downstream sparse attention's
        # summation order).
        sel_bits = cute.make_rmem_tensor((num_sel_words,), Uint64)
        for w in cutlass.range_constexpr(num_sel_words):
            sel_bits[w] = Uint64(0)
        if tid == Int32(0):
            scalars[0] = Int32(0)
            scalars[1] = Int32(0)
            scalars[3] = Int32(0)
            prefix_ptr.store(Uint64(0))
        cute.arch.sync_threads()

        for p in cutlass.range_constexpr(num_passes):
            shift = (num_passes - 1 - p) * radix_bits
            if scalars[3] == Int32(0):
                prefix = prefix_ptr.load()
                hist[tid] = Int32(0)
                cute.arch.sync_threads()
                for i in cutlass.range_constexpr(keys_per_thread):
                    key = keys[i]
                    if (
                        key != Uint64(0)
                        and (key >> Uint64(shift + radix_bits)) == prefix
                    ):
                        b = Int32((key >> Uint64(shift)) & Uint64(num_bins - 1))
                        cute.arch.atomic_add(
                            hist.iterator + b, Int32(1), sem="relaxed", scope="cta"
                        )
                cute.arch.sync_threads()

                # Bins high to low: tid t scans bin num_bins - 1 - t.
                need = Int32(topk) - scalars[0]
                count = hist[Int32(num_bins - 1) - tid]
                incl = _block_scan_inclusive(count, lane, warp, warp_totals, num_warps)
                # The committed count grows by the keys selected below: the
                # bins above the threshold (and the threshold bin when done).
                # Every thread read scalars[0] before the scan's barriers.
                if count > Int32(0) and incl >= need and incl - count < need:
                    scalars[2] = Int32(num_bins - 1) - tid
                    if incl == need:
                        scalars[3] = Int32(1)
                        scalars[0] = scalars[0] + incl
                    else:
                        scalars[0] = scalars[0] + incl - count
                if tid == Int32(tb_size - 1) and incl < need:
                    # Fewer matching candidates than needed: keep them all.
                    scalars[2] = Int32(0)
                    scalars[3] = Int32(1)
                    scalars[0] = scalars[0] + incl
                cute.arch.sync_threads()

                thr = scalars[2]
                done = scalars[3] != Int32(0)
                for i in cutlass.range_constexpr(keys_per_thread):
                    key = keys[i]
                    selected = False
                    if (
                        key != Uint64(0)
                        and (key >> Uint64(shift + radix_bits)) == prefix
                    ):
                        b = Int32((key >> Uint64(shift)) & Uint64(num_bins - 1))
                        selected = b > thr or (done and b == thr)
                    if selected:
                        score = Int32(key >> Uint64(gid_bits))
                        if score < sel_min:
                            sel_min = score
                        sel_bits[i // 64] = sel_bits[i // 64] | Uint64(1 << (i % 64))
                cute.arch.sync_threads()
                if tid == Int32(0) and not done:
                    prefix_ptr.store((prefix << Uint64(radix_bits)) | Uint64(thr))
                cute.arch.sync_threads()

        if cutlass.const_expr(emit_ids):
            count = Int32(0)
            for w in cutlass.range_constexpr(num_sel_words):
                count += _popc64(sel_bits[w])
            incl = _block_scan_inclusive(count, lane, warp, warp_totals, num_warps)
            pos = incl - count
            for i in cutlass.range_constexpr(keys_per_thread):
                if ((sel_bits[i // 64] >> Uint64(i % 64)) & Uint64(1)) != Uint64(0):
                    gid = Uint32(gid_mask) - Uint32(keys[i] & Uint64(gid_mask))
                    out[row, pos] = recast_val(gid, Int32)
                    pos += Int32(1)
            for i in cutlass.range(scalars[0] + tid, topk, tb_size, unroll=1):
                out[row, i] = Int32(-1)
            if cutlass.const_expr(emit_rows):
                cute.arch.sync_threads()  # warp_totals is reused below
        if cutlass.const_expr(emit_rows):
            # This rank's owned selected keys (a block past the table dropped).
            owned_bits = cute.make_rmem_tensor((num_sel_words,), Uint64)
            for w in cutlass.range_constexpr(num_sel_words):
                owned_bits[w] = Uint64(0)
            for i in cutlass.range_constexpr(keys_per_thread):
                if ((sel_bits[i // 64] >> Uint64(i % 64)) & Uint64(1)) != Uint64(0):
                    gid = Uint32(gid_mask) - Uint32(keys[i] & Uint64(gid_mask))
                    local = recast_val(gid // Uint32(world_size), Int32)
                    block = local // Int32(block_size)
                    if gid % Uint32(world_size) == recast_val(
                        dcp_rank, Uint32
                    ) and block < Int32(block_table.shape[1]):
                        owned_bits[i // 64] = owned_bits[i // 64] | Uint64(
                            1 << (i % 64)
                        )
            count = Int32(0)
            for w in cutlass.range_constexpr(num_sel_words):
                count += _popc64(owned_bits[w])
            incl = _block_scan_inclusive(count, lane, warp, warp_totals, num_warps)
            if tid == Int32(tb_size - 1):
                scalars[1] = incl
            pos = incl - count
            for i in cutlass.range_constexpr(keys_per_thread):
                if ((owned_bits[i // 64] >> Uint64(i % 64)) & Uint64(1)) != Uint64(0):
                    gid = Uint32(gid_mask) - Uint32(keys[i] & Uint64(gid_mask))
                    local = recast_val(gid // Uint32(world_size), Int32)
                    block = local // Int32(block_size)
                    rows_out[row, pos] = block_table[
                        req_id[row], block
                    ] * stride_rows + (local - block * Int32(block_size))
                    pos += Int32(1)
            cute.arch.sync_threads()
            num_owned = scalars[1]
            for i in cutlass.range(
                num_owned + tid, rows_out.shape[1], tb_size, unroll=1
            ):
                rows_out[row, i] = Int32(-1)
            if tid == Int32(0):
                valid_out[row] = num_owned
        return sel_min

    return radix_select


@cute.jit
def _decode_key(
    word: Uint32, high: Uint32, j: Int32, owner: Uint32, world_size: int, gid_bits: int
) -> Uint64:
    """B' word (+ its bit-plane word) of candidate slot `j` -> select key."""
    key = Uint64(0)
    if word != Uint32(0):
        local = (Uint32(0xFFFF) - (word & Uint32(0xFFFF))) | (
            ((high >> recast_val(j & Int32(31), Uint32)) & Uint32(1)) << Uint32(16)
        )
        gid = local * Uint32(world_size) + owner
        key = (Uint64(word >> Uint32(16)) << Uint64(gid_bits)) | Uint64(
            Uint32((1 << gid_bits) - 1) - gid
        )
    return key


class DCPTopkSelectKernel(VllmCuTeDSLJitKernel["DCPTopkSelectKernel.CompileKey"]):
    """Gathered packed rows `[W, T, N + N/32]` -> the global top-K of each
    row as global ids `[T, K]` and/or this rank's owned physical KV rows.

    One CTA per row. Candidates become `(ord16 << gid_bits) | ~gid` keys (unique, so the
    order is total) and a 9-bit radix select over the `16 + gid_bits` key bits finds the
    K-th key; ties in the bf16 score are resolved toward the lower global id. Rows with
    fewer than K candidates keep all of them.

    `emit_ids`: global ids, selected first (in a fixed order: the same ids in
    the same order on every run), then -1.
    `emit_rows`: the selected tokens owned by `dcp_rank` (`gid % W`) as
    physical rows `block_table[req_id[t], local // B] * stride_rows +
    local % B` (`local = gid // W`), compacted to the front (in a fixed
    order: the same rows in the same order on every run),
    then -1; plus their count. A block id past the block table is dropped.

    `exact`: also derive the row's saturation word `(mask << 16) | tau16`
    (`tau16` the ord16 of the K-th selected key, 0 when fewer than K were
    selected; bit `r` of `mask` set iff rank `r` sent N words, all with
    ord16 > tau16) and, when this rank's bit is set, push its local top-K
    (from `logits`) to every rank's rescue slot `popc(mask & ((1 << rank)
    - 1))` and release `flags[rank][t] = seq2[t] + 1`. Never waits.
    """

    tb_size = _TB_SIZE
    radix_bits = _RADIX_BITS
    num_bins = _NUM_BINS

    @dataclass(frozen=True)
    class CompileKey:
        topk: int
        local_topk: int
        world_size: int
        emit_ids: bool = True
        # Physical-row output; block_size 0 disables it.
        block_size: int = 0
        exact: bool = False

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        emit_ids = compile_key.emit_ids
        block_size = compile_key.block_size
        exact = compile_key.exact
        assert emit_ids or block_size > 0
        tb_size = _TB_SIZE
        num_bins = _NUM_BINS
        num_warps = tb_size // _WARP
        topk = compile_key.topk
        local_topk = compile_key.local_topk
        world_size = compile_key.world_size
        assert world_size <= 16
        num_keys = world_size * local_topk
        keys_per_thread = -(-num_keys // tb_size)
        gid_bits = _gid_bits(world_size)
        num_slots = rescue_slots(topk, local_topk)
        recv_words = packed_row_words(topk)
        radix_select = _make_radix_select(
            topk=topk,
            world_size=world_size,
            keys_per_thread=keys_per_thread,
            emit_ids=emit_ids,
            block_size=block_size,
        )

        @cute.struct
        class SharedStorage:
            hist: cute.struct.MemRange[Int32, num_bins]
            warp_totals: cute.struct.MemRange[Int32, num_warps]
            scalars: cute.struct.MemRange[Int32, 8]
            notsat: cute.struct.MemRange[Int32, 16]
            prefix: cute.struct.Align[cute.struct.MemRange[Uint64, 1], 8]
            push: cute.struct.Align[
                cute.struct.MemRange[Int32, recv_words if exact else 4], 16
            ]

        @cute.jit
        def push_local_topk(
            row: Int32,
            tid: Int32,
            lane: Int32,
            warp: Int32,
            hist: cute.Tensor,
            warp_totals: cute.Tensor,
            scalars: cute.Tensor,
            buf: cute.Tensor,
            logits: cute.Tensor,
            row_ends: cute.Tensor,
            slot: Int32,
            dcp_rank: Int32,
            seq2: cute.Tensor,
            peer_recv: Int64,
            peer_flags: Int64,
            max_rows: Int32,
        ):
            """This rank's local top-min(K, valid) of `logits[row]` in the B'
            layout -> every rank's `recv[row, slot]`; then the flag."""
            end = Int32(logits.shape[1])
            if end > Int32(1 << LOCAL_INDEX_BITS):
                end = Int32(1 << LOCAL_INDEX_BITS)
            if row_ends[row] < end:
                end = row_ends[row]
            # scalars[4..7]: threshold bin, count above it, take-all, eq total
            # Pass 1: histogram of ord16 >> 7.
            hist[tid] = Int32(0)
            if tid == Int32(0):
                scalars[4] = Int32(-1)
                scalars[5] = Int32(0)
            cute.arch.sync_threads()
            for loc_idx in cutlass.range(tid, end, tb_size, unroll=1):
                bits = Uint32(recast_val(logits[row, loc_idx], Uint16))
                if (bits & Uint32(0x7FFF)) <= Uint32(0x7F80):
                    b = Int32(_ord16(bits) >> Uint32(7))
                    cute.arch.atomic_add(
                        hist.iterator + b, Int32(1), sem="relaxed", scope="cta"
                    )
            cute.arch.sync_threads()
            count = hist[Int32(num_bins - 1) - tid]
            incl = _block_scan_inclusive(count, lane, warp, warp_totals, num_warps)
            if count > Int32(0) and incl >= Int32(topk) and incl - count < Int32(topk):
                scalars[4] = Int32(num_bins - 1) - tid
                scalars[5] = incl - count
            cute.arch.sync_threads()
            b1 = scalars[4]
            above = scalars[5]
            v = Uint32(0)
            c_gt = Int32(0)
            if b1 >= Int32(0):
                # Pass 2: low 7 bits inside the boundary bin.
                hist[tid] = Int32(0)
                cute.arch.sync_threads()
                for loc_idx in cutlass.range(tid, end, tb_size, unroll=1):
                    bits = Uint32(recast_val(logits[row, loc_idx], Uint16))
                    if (bits & Uint32(0x7FFF)) <= Uint32(0x7F80):
                        o = _ord16(bits)
                        if Int32(o >> Uint32(7)) == b1:
                            cute.arch.atomic_add(
                                hist.iterator + Int32(o & Uint32(127)),
                                Int32(1),
                                sem="relaxed",
                                scope="cta",
                            )
                cute.arch.sync_threads()
                count2 = hist[Int32(num_bins - 1) - tid]
                incl2 = _block_scan_inclusive(
                    count2, lane, warp, warp_totals, num_warps
                )
                need2 = Int32(topk) - above
                if count2 > Int32(0) and incl2 >= need2 and incl2 - count2 < need2:
                    scalars[6] = Int32(num_bins - 1) - tid
                    scalars[7] = above + incl2 - count2
                cute.arch.sync_threads()
                v = (Uint32(b1) << Uint32(7)) | Uint32(scalars[6])
                c_gt = scalars[7]
            # Emit: > v anywhere, == v in ascending l up to topk - c_gt.
            for i in cutlass.range(tid, recv_words, tb_size, unroll=1):
                buf[i] = Int32(0)
            if tid == Int32(0):
                scalars[5] = Int32(0)
            cute.arch.sync_threads()
            eq_run = Int32(0)
            gt_run = Int32(0)
            for base in cutlass.range(0, end, tb_size, unroll=1):
                loc_idx = base + tid
                o = Uint32(0)
                if loc_idx < end:
                    bits = Uint32(recast_val(logits[row, loc_idx], Uint16))
                    if (bits & Uint32(0x7FFF)) <= Uint32(0x7F80):
                        o = _ord16(bits)
                is_eq = o == v and o != Uint32(0)
                # One scan counts both, in a fixed order (the rescue round's
                # keys, and so its rows, then come out the same on every run):
                # keys above v in the high half, keys equal to it in the low.
                flag = Int32(0)
                if is_eq:
                    flag = Int32(1)
                if o > v:
                    flag = Int32(1 << 16)
                incl = _block_scan_inclusive(flag, lane, warp, warp_totals, num_warps)
                if tid == Int32(tb_size - 1):
                    scalars[6] = incl
                cute.arch.sync_threads()
                eq_incl = incl & Int32(0xFFFF)
                pos = Int32(-1)
                if o > v:
                    pos = gt_run + (incl >> Int32(16)) - Int32(1)
                if is_eq:
                    idx = eq_run + eq_incl - Int32(1)
                    if idx < Int32(topk) - c_gt:
                        pos = c_gt + idx
                if pos >= Int32(0):
                    l_u = recast_val(loc_idx, Uint32)
                    buf[pos] = recast_val(
                        (o << Uint32(16)) | (Uint32(0xFFFF) - (l_u & Uint32(0xFFFF))),
                        Int32,
                    )
                    if (l_u >> Uint32(16)) != Uint32(0):
                        cute.arch.atomic_or(
                            buf.iterator + (Int32(topk) + (pos >> Int32(5))),
                            Int32(1) << (pos & Int32(31)),
                            sem="relaxed",
                            scope="cta",
                        )
                eq_run += scalars[6] & Int32(0xFFFF)
                gt_run += scalars[6] >> Int32(16)
                cute.arch.sync_threads()
            # Store to every rank's slot, then release the flag.
            row_offset = (Int64(row) * Int64(num_slots) + Int64(slot)) * Int64(
                recv_words * 4
            )
            for p in cutlass.range_constexpr(world_size):
                dst = _ld_i64(peer_recv + Int64(p * 8)) + row_offset
                for vi in cutlass.range(tid, recv_words // 4, tb_size, unroll=1):
                    _st_v4(
                        dst + Int64(vi) * Int64(16),
                        buf[vi * 4],
                        buf[vi * 4 + 1],
                        buf[vi * 4 + 2],
                        buf[vi * 4 + 3],
                    )
            cute.arch.sync_threads()
            if tid == Int32(0):
                cute.arch.fence_acq_rel_sys()
                flag = seq2[row] + Int32(1)
                flag_offset = (Int64(dcp_rank) * Int64(max_rows) + Int64(row)) * Int64(
                    4
                )
                for p in cutlass.range_constexpr(world_size):
                    _st_release_sys(
                        _ld_i64(peer_flags + Int64(p * 8)) + flag_offset, flag
                    )

        @cute.kernel
        def device_kernel(
            gathered: cute.Tensor,
            out: cute.Tensor,
            req_id: cute.Tensor,
            block_table: cute.Tensor,
            rows_out: cute.Tensor,
            valid_out: cute.Tensor,
            dcp_rank: Int32,
            stride_rows: Int32,
            sat_word: cute.Tensor,
            seq2: cute.Tensor,
            logits: cute.Tensor,
            row_ends: cute.Tensor,
            peer_recv: Int64,
            peer_flags: Int64,
            max_rows: Int32,
        ):
            row, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            lane = cute.arch.lane_idx()
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            smem = cutlass.memory.SmemAllocator()
            storage = smem.allocate(SharedStorage, 16)
            hist = storage.hist.get_tensor(cute.make_layout((num_bins,)))
            warp_totals = storage.warp_totals.get_tensor(cute.make_layout((num_warps,)))
            scalars = storage.scalars.get_tensor(cute.make_layout((8,)))
            notsat = storage.notsat.get_tensor(cute.make_layout((16,)))
            prefix_ptr = storage.prefix.data_ptr()

            keys = cute.make_rmem_tensor((keys_per_thread,), Uint64)
            for i in cutlass.range_constexpr(keys_per_thread):
                flat = tid + Int32(i * tb_size)
                key = Uint64(0)
                if flat < Int32(num_keys):
                    rank = flat // Int32(local_topk)
                    j = flat - rank * Int32(local_topk)
                    key = _decode_key(
                        recast_val(gathered[rank, row, j], Uint32),
                        recast_val(
                            gathered[rank, row, Int32(local_topk) + (j >> Int32(5))],
                            Uint32,
                        ),
                        j,
                        recast_val(rank, Uint32),
                        world_size,
                        gid_bits,
                    )
                keys[i] = key
            if cutlass.const_expr(exact):  # noqa: SIM102
                if tid < Int32(16):
                    notsat[tid] = Int32(0)

            sel_min = radix_select(
                keys,
                row,
                tid,
                lane,
                warp,
                hist,
                warp_totals,
                scalars,
                prefix_ptr,
                out,
                req_id,
                block_table,
                rows_out,
                valid_out,
                dcp_rank,
                stride_rows,
            )

            if cutlass.const_expr(exact):
                # tau16: the K-th selected key's ord16 (0 below K selected).
                wmin = cute.arch.warp_redux_sync(sel_min, "min")
                hist[tid] = Int32(0x10000)
                cute.arch.sync_threads()
                if lane == Int32(0):
                    hist[warp] = wmin
                cute.arch.sync_threads()
                tau = Int32(0x10000)
                for w in cutlass.range_constexpr(num_warps):
                    if hist[w] < tau:
                        tau = hist[w]
                if scalars[0] < Int32(topk):
                    tau = Int32(0)
                # Rank r is saturated iff all its N slots hold ord16 > tau16
                # (empty slots are 0, so it sent N words).
                for i in cutlass.range_constexpr(keys_per_thread):
                    flat = tid + Int32(i * tb_size)
                    if flat < Int32(num_keys):
                        key = keys[i]
                        above = (
                            key != Uint64(0) and Int32(key >> Uint64(gid_bits)) > tau
                        )
                        ballot = recast_val(cute.arch.vote_ballot_sync(above), Uint32)
                        if lane == Int32(0) and ballot != Uint32(0xFFFFFFFF):
                            notsat[flat // Int32(local_topk)] = Int32(1)
                cute.arch.sync_threads()
                mask = Int32(0)
                for r in cutlass.range_constexpr(world_size):
                    if notsat[r] == Int32(0):
                        mask = mask | Int32(1 << r)
                # Distinct keys bound the saturated ranks by S = (K - 1) // N.
                # Repeated keys (out of contract: DeepSelect's ids on a NaN
                # row) can leave fewer than K selected (tau16 = 0) and more
                # full ranks than slots; keep round 1 then (no push or wait).
                num_sat = recast_val(cute.arch.popc(recast_val(mask, Uint32)), Int32)
                if num_sat > Int32(num_slots):
                    mask = Int32(0)
                if tid == Int32(0):
                    sat_word[row] = (mask << Int32(16)) | tau
                if ((mask >> dcp_rank) & Int32(1)) != Int32(0):
                    slot = recast_val(
                        cute.arch.popc(
                            recast_val(
                                mask & ((Int32(1) << dcp_rank) - Int32(1)), Uint32
                            )
                        ),
                        Int32,
                    )
                    push_buf = storage.push.get_tensor(cute.make_layout((recv_words,)))
                    push_local_topk(
                        row,
                        tid,
                        lane,
                        warp,
                        hist,
                        warp_totals,
                        scalars,
                        push_buf,
                        logits,
                        row_ends,
                        slot,
                        dcp_rank,
                        seq2,
                        peer_recv,
                        peer_flags,
                        max_rows,
                    )

        @cute.jit
        def host_entrypoint(
            gathered: cute.Tensor,
            out: cute.Tensor,
            req_id: cute.Tensor,
            block_table: cute.Tensor,
            rows_out: cute.Tensor,
            valid_out: cute.Tensor,
            dcp_rank: Int32,
            stride_rows: Int32,
            sat_word: cute.Tensor,
            seq2: cute.Tensor,
            logits: cute.Tensor,
            row_ends: cute.Tensor,
            peer_recv: Int64,
            peer_flags: Int64,
            max_rows: Int32,
            stream,
        ):
            # Unused outputs / exact-only inputs are placeholders.
            device_kernel(
                gathered,
                out,
                req_id,
                block_table,
                rows_out,
                valid_out,
                dcp_rank,
                stride_rows,
                sat_word,
                seq2,
                logits,
                row_ends,
                peer_recv,
                peer_flags,
                max_rows,
            ).launch(
                grid=(gathered.shape[1], 1, 1),
                block=(tb_size, 1, 1),
                stream=stream,
            )

        return host_entrypoint

    def dispatch(  # type: ignore[override]
        self,
        *,
        topk: int,
        local_topk: int,
        world_size: int,
        emit_ids: bool = True,
        block_size: int = 0,
        exact: bool = False,
    ) -> CompileKey:
        return self.CompileKey(
            topk=topk,
            local_topk=local_topk,
            world_size=world_size,
            emit_ids=emit_ids,
            block_size=block_size,
            exact=exact,
        )

    def get_warmup_keys(self, **kwargs) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(**kwargs)

    def warmup_inputs(self, compile_key: CompileKey) -> tuple[Any, ...]:
        return _select_fake_inputs(
            compile_key.world_size,
            compile_key.local_topk,
            compile_key.topk,
            compile_key.block_size,
        )

    @kernel_launcher
    def __call__(
        self,
        gathered: torch.Tensor,
        out: torch.Tensor | None,
        *,
        local_topk: int,
        topk: int,
        rows: "_RowsOutput | None" = None,
        exact: "_ExactArgs | None" = None,
    ) -> CuTeDSLLaunchSpec[CompileKey]:
        compile_key = self.dispatch(
            topk=topk,
            local_topk=local_topk,
            world_size=gathered.shape[0],
            emit_ids=out is not None,
            block_size=0 if rows is None else rows.block_size,
            exact=exact is not None,
        )
        return compile_key, (
            *_select_args(gathered, out, rows, 0 if exact is None else exact.rank),
            *_exact_args(exact, gathered.device),
        )


def _select_fake_inputs(
    world_size: int, local_topk: int, topk: int, block_size: int
) -> tuple[Any, ...]:
    num_rows = cute.sym_int()
    row_words = packed_row_words(local_topk)
    gathered = cute.runtime.make_fake_tensor(
        Int32,
        (world_size, num_rows, row_words),
        stride=(cute.sym_int64(), row_words, 1),
        assumed_align=4,
    )

    def vec() -> Any:
        return make_fake_tensor(Int32, (cute.sym_int(),))

    def mat() -> Any:
        return make_fake_tensor(Int32, (num_rows, cute.sym_int()), divisibility=1)

    block_table = cute.runtime.make_fake_tensor(
        Int32,
        (cute.sym_int(), cute.sym_int()),
        stride=(cute.sym_int(), 1),
        assumed_align=4,
    )
    logits = cute.runtime.make_fake_tensor(
        BFloat16,
        (cute.sym_int(), cute.sym_int()),
        stride=(cute.sym_int(), 1),
        assumed_align=2,
    )
    del topk, block_size
    return (
        gathered,
        mat(),
        vec(),
        block_table,
        mat(),
        vec(),
        Int32(0),
        Int32(1),
        vec(),
        vec(),
        logits,
        vec(),
        Int64(0),
        Int64(0),
        Int32(1),
    )


def _select_args(
    gathered: torch.Tensor,
    out: torch.Tensor | None,
    rows: "_RowsOutput | None",
    rank: int,
) -> tuple[Any, ...]:
    if rows is None:
        assert out is not None
        dummy = _dummy_i32(gathered.device)
        return (
            gathered,
            out,
            dummy,
            _dummy_table(gathered.device),
            out,
            dummy,
            rank,
            0,
        )
    return (
        gathered,
        rows.rows_out if out is None else out,
        rows.req_id,
        rows.block_table,
        rows.rows_out,
        rows.valid_out,
        rows.dcp_rank,
        rows.stride_rows,
    )


_DUMMIES: dict[Any, torch.Tensor] = {}


def _dummy(device: torch.device, name: str, shape, dtype) -> torch.Tensor:
    key = (device, name)
    t = _DUMMIES.get(key)
    if t is None:
        t = torch.zeros(shape, dtype=dtype, device=device)
        _DUMMIES[key] = t
    return t


def _dummy_i32(device: torch.device) -> torch.Tensor:
    return _dummy(device, "i32", (1,), torch.int32)


def _dummy_table(device: torch.device) -> torch.Tensor:
    return _dummy(device, "table", (1, 1), torch.int32)


@dataclass(frozen=True)
class _ExactArgs:
    """Rescue state for one exact merge call (one ubatch of a workspace)."""

    sat_word: torch.Tensor
    seq2: torch.Tensor
    logits: torch.Tensor
    row_ends: torch.Tensor
    recv: torch.Tensor  # local [T_max, S, K + K/32]
    flags: torch.Tensor  # local [W, T_max]
    peer_recv: torch.Tensor  # int64 [W]
    peer_flags: torch.Tensor  # int64 [W]
    max_rows: int
    rank: int
    topk: int
    local_topk: int
    world_size: int


def _exact_args(exact: _ExactArgs | None, device: torch.device) -> tuple[Any, ...]:
    if exact is None:
        dummy = _dummy_i32(device)
        logits = _dummy(device, "bf16", (1, 1), torch.bfloat16)
        return (dummy, dummy, logits, dummy, 0, 0, 1)
    return (
        exact.sat_word,
        exact.seq2,
        exact.logits,
        exact.row_ends,
        exact.peer_recv.data_ptr(),
        exact.peer_flags.data_ptr(),
        exact.max_rows,
    )


class DCPTopkRescueKernel(VllmCuTeDSLJitKernel["DCPTopkRescueKernel.CompileKey"]):
    """Second round of the exact merge, launched after every exact select.

    Persistent grid (G = min(T, #SMs) CTAs strided over rows). A row whose saturation
    word has an empty mask is skipped (the fast path). Otherwise the CTA waits for the
    pushes of the mask's ranks, re-selects the top-K of the round-1 words of the other
    ranks (only ord16 >= tau16) plus the pushed local top-K lists under the same tie
    rule, overwrites the row's outputs and bumps `seq2[row]`.
    """

    tb_size = _TB_SIZE

    @dataclass(frozen=True)
    class CompileKey:
        topk: int
        local_topk: int
        world_size: int
        emit_ids: bool = True
        block_size: int = 0

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        emit_ids = compile_key.emit_ids
        block_size = compile_key.block_size
        tb_size = _TB_SIZE
        num_bins = _NUM_BINS
        num_warps = tb_size // _WARP
        topk = compile_key.topk
        local_topk = compile_key.local_topk
        world_size = compile_key.world_size
        num_slots = rescue_slots(topk, local_topk)
        recv_words = packed_row_words(topk)
        num_r1 = world_size * local_topk
        num_keys = num_r1 + num_slots * topk
        keys_per_thread = -(-num_keys // tb_size)
        gid_bits = _gid_bits(world_size)
        radix_select = _make_radix_select(
            topk=topk,
            world_size=world_size,
            keys_per_thread=keys_per_thread,
            emit_ids=emit_ids,
            block_size=block_size,
        )

        @cute.struct
        class SharedStorage:
            hist: cute.struct.MemRange[Int32, num_bins]
            warp_totals: cute.struct.MemRange[Int32, num_warps]
            scalars: cute.struct.MemRange[Int32, 8]
            owner: cute.struct.MemRange[Int32, 16]
            prefix: cute.struct.Align[cute.struct.MemRange[Uint64, 1], 8]

        @cute.kernel
        def device_kernel(
            gathered: cute.Tensor,
            out: cute.Tensor,
            req_id: cute.Tensor,
            block_table: cute.Tensor,
            rows_out: cute.Tensor,
            valid_out: cute.Tensor,
            dcp_rank: Int32,
            stride_rows: Int32,
            sat_word: cute.Tensor,
            seq2: cute.Tensor,
            recv: Int64,
            flags: Int64,
            max_rows: Int32,
        ):
            cta, _, _ = cute.arch.block_idx()
            num_ctas, _, _ = cute.arch.grid_dim()
            tid, _, _ = cute.arch.thread_idx()
            lane = cute.arch.lane_idx()
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            smem = cutlass.memory.SmemAllocator()
            storage = smem.allocate(SharedStorage, 16)
            hist = storage.hist.get_tensor(cute.make_layout((num_bins,)))
            warp_totals = storage.warp_totals.get_tensor(cute.make_layout((num_warps,)))
            scalars = storage.scalars.get_tensor(cute.make_layout((8,)))
            owner = storage.owner.get_tensor(cute.make_layout((16,)))
            prefix_ptr = storage.prefix.data_ptr()
            num_rows = gathered.shape[1]

            for row in cutlass.range(cta, num_rows, num_ctas, unroll=1):
                word = recast_val(sat_word[row], Uint32)
                mask = Int32(word >> Uint32(16))
                if mask != Int32(0):
                    tau = Int32(word & Uint32(0xFFFF))
                    num_sat = recast_val(
                        cute.arch.popc(recast_val(mask, Uint32)), Int32
                    )
                    if num_sat > Int32(num_slots):
                        _trap()
                    expected = seq2[row] + Int32(1)
                    if tid == Int32(0):
                        k = Int32(0)
                        for r in cutlass.range_constexpr(world_size):
                            if ((mask >> Int32(r)) & Int32(1)) != Int32(0):
                                owner[k] = Int32(r)
                                k += Int32(1)
                    if (
                        warp == Int32(0)
                        and lane < Int32(world_size)
                        and ((mask >> lane) & Int32(1)) != Int32(0)
                    ):
                        flag_addr = flags + (
                            Int64(lane) * Int64(max_rows) + Int64(row)
                        ) * Int64(4)
                        start = _globaltimer()
                        while _ld_acquire_sys(flag_addr) != expected:
                            _nanosleep()
                            if _globaltimer() - start > Int64(_SPIN_TIMEOUT_NS):
                                cute.printf(
                                    "DCP top-k rescue timeout row=%d rank=%d seq=%d\n",
                                    row,
                                    lane,
                                    expected,
                                )
                                _trap()
                    cute.arch.sync_threads()

                    keys = cute.make_rmem_tensor((keys_per_thread,), Uint64)
                    row_base = recv + Int64(row) * Int64(num_slots * recv_words * 4)
                    for i in cutlass.range_constexpr(keys_per_thread):
                        flat = tid + Int32(i * tb_size)
                        key = Uint64(0)
                        if flat < Int32(num_r1):
                            rank = flat // Int32(local_topk)
                            if ((mask >> rank) & Int32(1)) == Int32(0):
                                j = flat - rank * Int32(local_topk)
                                key = _decode_key(
                                    recast_val(gathered[rank, row, j], Uint32),
                                    recast_val(
                                        gathered[
                                            rank,
                                            row,
                                            Int32(local_topk) + (j >> Int32(5)),
                                        ],
                                        Uint32,
                                    ),
                                    j,
                                    recast_val(rank, Uint32),
                                    world_size,
                                    gid_bits,
                                )
                                if Int32(key >> Uint64(gid_bits)) < tau:
                                    key = Uint64(0)
                        elif flat < Int32(num_keys):
                            s = (flat - Int32(num_r1)) // Int32(topk)
                            if s < num_sat:
                                j = flat - Int32(num_r1) - s * Int32(topk)
                                base = row_base + Int64(s) * Int64(recv_words * 4)
                                key = _decode_key(
                                    _ld_u32(base + Int64(j) * Int64(4)),
                                    _ld_u32(
                                        base
                                        + Int64(Int32(topk) + (j >> Int32(5)))
                                        * Int64(4)
                                    ),
                                    j,
                                    recast_val(owner[s], Uint32),
                                    world_size,
                                    gid_bits,
                                )
                        keys[i] = key

                    radix_select(
                        keys,
                        row,
                        tid,
                        lane,
                        warp,
                        hist,
                        warp_totals,
                        scalars,
                        prefix_ptr,
                        out,
                        req_id,
                        block_table,
                        rows_out,
                        valid_out,
                        dcp_rank,
                        stride_rows,
                    )
                    cute.arch.sync_threads()
                    if tid == Int32(0):
                        seq2[row] = expected

        @cute.jit
        def host_entrypoint(
            gathered: cute.Tensor,
            out: cute.Tensor,
            req_id: cute.Tensor,
            block_table: cute.Tensor,
            rows_out: cute.Tensor,
            valid_out: cute.Tensor,
            dcp_rank: Int32,
            stride_rows: Int32,
            sat_word: cute.Tensor,
            seq2: cute.Tensor,
            recv: Int64,
            flags: Int64,
            max_rows: Int32,
            num_ctas: Int32,
            stream,
        ):
            device_kernel(
                gathered,
                out,
                req_id,
                block_table,
                rows_out,
                valid_out,
                dcp_rank,
                stride_rows,
                sat_word,
                seq2,
                recv,
                flags,
                max_rows,
            ).launch(
                grid=(num_ctas, 1, 1),
                block=(tb_size, 1, 1),
                stream=stream,
            )

        return host_entrypoint

    def dispatch(  # type: ignore[override]
        self,
        *,
        topk: int,
        local_topk: int,
        world_size: int,
        emit_ids: bool = True,
        block_size: int = 0,
    ) -> CompileKey:
        return self.CompileKey(
            topk=topk,
            local_topk=local_topk,
            world_size=world_size,
            emit_ids=emit_ids,
            block_size=block_size,
        )

    def get_warmup_keys(self, **kwargs) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(**kwargs)

    def warmup_inputs(self, compile_key: CompileKey) -> tuple[Any, ...]:
        fake = _select_fake_inputs(
            compile_key.world_size,
            compile_key.local_topk,
            compile_key.topk,
            compile_key.block_size,
        )
        return (*fake[:10], Int64(0), Int64(0), Int32(1), Int32(1))

    @kernel_launcher
    def __call__(
        self,
        gathered: torch.Tensor,
        out: torch.Tensor | None,
        *,
        local_topk: int,
        topk: int,
        exact: _ExactArgs,
        rows: "_RowsOutput | None" = None,
    ) -> CuTeDSLLaunchSpec[CompileKey]:
        compile_key = self.dispatch(
            topk=topk,
            local_topk=local_topk,
            world_size=gathered.shape[0],
            emit_ids=out is not None,
            block_size=0 if rows is None else rows.block_size,
        )
        num_rows = gathered.shape[1]
        num_ctas = max(1, min(num_rows, _num_sms(gathered.device)))
        return compile_key, (
            *_select_args(gathered, out, rows, exact.rank),
            exact.sat_word,
            exact.seq2,
            exact.recv.data_ptr(),
            exact.flags.data_ptr(),
            exact.max_rows,
            num_ctas,
        )


@functools.cache
def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


@dataclass(frozen=True)
class _RowsOutput:
    req_id: torch.Tensor
    block_table: torch.Tensor
    rows_out: torch.Tensor
    valid_out: torch.Tensor
    dcp_rank: int
    block_size: int
    stride_rows: int


_DCP_TOPK_PACK_KERNEL = DCPTopkPackKernel()
_DCP_TOPK_SELECT_KERNEL = DCPTopkSelectKernel()
_DCP_TOPK_RESCUE_KERNEL = DCPTopkRescueKernel()


class DCPTopkRescueWorkspace:
    """Rescue state of the exact merge: per ubatch a symmetric receive buffer
    `[T_max, S, K + K/32]` int32 (S = `rescue_slots(K, N)` pushed local top-K lists per
    row), symmetric flags `[W, T_max]`, and local `seq2` / `sat_word` `[T_max]`.

    `peer_recv` / `peer_flags` hold every rank's buffer address (this
    rank's included)."""

    def __init__(
        self,
        group: Any,
        device: torch.device,
        max_rows: int,
        topk: int,
        local_topk: int,
    ) -> None:
        from vllm.v1.attention.ops.cp_common import DirectCPWorkspace

        base = DirectCPWorkspace(group, device, 1)
        self.device_group = group
        self.world_size = base.world_size
        self.rank = base.rank
        num_slots = rescue_slots(topk, local_topk)
        recv, recv_ptrs = base._allocate(
            (1, max_rows, num_slots, packed_row_words(topk)), torch.int32
        )
        flags, flag_ptrs = base._allocate((1, self.world_size, max_rows), torch.int32)
        self._base = base
        self._init(
            recv[0],
            flags[0],
            recv_ptrs[0].contiguous(),
            flag_ptrs[0].contiguous(),
            max_rows,
            topk,
            local_topk,
        )

    def _init(self, recv, flags, peer_recv, peer_flags, max_rows, topk, local_topk):
        self.recv = recv
        self.flags = flags
        self.peer_recv = peer_recv
        self.peer_flags = peer_flags
        self.max_rows = max_rows
        self.topk = topk
        self.local_topk = local_topk
        device = recv.device
        self.seq2 = torch.zeros(max_rows, dtype=torch.int32, device=device)
        self.sat_word = torch.zeros(max_rows, dtype=torch.int32, device=device)

    def args(self, logits: torch.Tensor, row_ends: torch.Tensor) -> _ExactArgs:
        return _ExactArgs(
            sat_word=self.sat_word,
            seq2=self.seq2,
            logits=logits,
            row_ends=row_ends,
            recv=self.recv,
            flags=self.flags,
            peer_recv=self.peer_recv,
            peer_flags=self.peer_flags,
            max_rows=self.max_rows,
            rank=self.rank,
            topk=self.topk,
            local_topk=self.local_topk,
            world_size=self.world_size,
        )


_RESCUE_WORKSPACES: dict[tuple[int, int], DCPTopkRescueWorkspace] = {}


def init_dcp_topk_rescue_workspace(
    max_rows: int, topk: int, local_topk: int, group: Any = None
) -> DCPTopkRescueWorkspace:
    """Allocate (once) the exact merge's rescue workspace for (K, N) on the
    DCP group. Must run on every DCP rank, outside CUDA-graph capture."""
    key = (topk, local_topk)
    ws = _RESCUE_WORKSPACES.get(key)
    if ws is None or ws.max_rows < max_rows:
        if group is None:
            from vllm.distributed import get_dcp_group

            group = get_dcp_group()
        ws = DCPTopkRescueWorkspace(
            group.device_group,
            torch.device("cuda", torch.accelerator.current_device_index()),
            max_rows,
            topk,
            local_topk,
        )
        _RESCUE_WORKSPACES[key] = ws
    return ws


def get_dcp_topk_rescue_workspace(
    topk: int, local_topk: int
) -> DCPTopkRescueWorkspace | None:
    return _RESCUE_WORKSPACES.get((topk, local_topk))


def register_dcp_topk_merge_warmup(
    topk: int,
    local_topk: int,
    world_size: int,
    *,
    emit_ids: bool = True,
    block_size: int = 0,
    exact: bool = True,
) -> None:
    """Compile the pack kernel and one select (+ rescue) variant at JIT
    warmup.

    The stock indexer uses the defaults (global ids); a caller of `dcp_topk_merge_rows`
    passes its `block_size` and whether it also wants the global ids."""
    _DCP_TOPK_PACK_KERNEL.register_warmup(local_topk=local_topk, has_row_ends=True)
    _DCP_TOPK_SELECT_KERNEL.register_warmup(
        topk=topk,
        local_topk=local_topk,
        world_size=world_size,
        emit_ids=emit_ids,
        block_size=block_size,
        exact=exact,
    )
    if exact:
        _DCP_TOPK_RESCUE_KERNEL.register_warmup(
            topk=topk,
            local_topk=local_topk,
            world_size=world_size,
            emit_ids=emit_ids,
            block_size=block_size,
        )


def dcp_topk_pack_indices(
    logits: torch.Tensor,
    local_idx: torch.Tensor,
    out: torch.Tensor,
    row_ends: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack a local top-N (`local_idx` `[T, N]` int32, -1 padded, N a multiple of 32) of
    the bf16 `logits` `[T, S]` into `out` `[T, N + N/32]` int32 (the B' layout, see
    `__init__`). Indices at or past `row_ends` (`[T]` int32, optional) or `S` and NaN
    scores become empty slots."""
    _DCP_TOPK_PACK_KERNEL(logits, local_idx, out, row_ends)
    return out


def _bound_logits(logits: torch.Tensor, max_local_len: int | None) -> torch.Tensor:
    if max_local_len is not None and max_local_len < logits.shape[1]:
        return logits[:, :max_local_len]
    return logits


def dcp_topk_pack(
    logits: torch.Tensor,
    row_ends: torch.Tensor,
    local_topk: int,
    out: torch.Tensor | None = None,
    max_local_len: int | None = None,
) -> torch.Tensor:
    """Local top-N (DeepSelect) of the bf16 `logits` as packed
    `[T, N + N/32]` rows (into `out` when given).

    `max_local_len` bounds every `row_ends` entry. DeepGEMM sizes the
    logits by the global `max_model_len`; DeepSelect picks its cluster
    variant for <= 6 rows at >= 512K columns (~21 us against ~6 us), so the
    columns past the local bound are cut off before it sees them.
    """
    from vllm.model_executor.layers.indexer_topk import deep_select_topk

    logits = _bound_logits(logits, max_local_len)
    num_rows = logits.shape[0]
    local_idx = deep_select_topk(logits, local_topk, end=row_ends)
    if out is None:
        out = torch.empty(
            (num_rows, packed_row_words(local_topk)),
            dtype=torch.int32,
            device=logits.device,
        )
    return dcp_topk_pack_indices(logits, local_idx, out, row_ends)


def assert_packed_merge_contract(
    packed: torch.Tensor,
    logits: torch.Tensor,
    row_ends: torch.Tensor,
    local_topk: int,
) -> None:
    """Check a packed-mode producer against the contract of `dcp_topk_merge_rows`
    (raises `AssertionError`; syncs, for tests and debugging): per row, exactly
    `min(N, v)` words (`v` the non-NaN tokens below `row_ends`) of distinct local ids
    below `row_ends`, each word's score the logit's bf16 bits, and no unsent non-NaN
    token scoring above the lowest sent one."""
    num_rows = packed.shape[0]
    assert packed.dtype == torch.int32
    assert packed.shape[1] == packed_row_words(local_topk), packed.shape
    words = packed[:, :local_topk].long() & 0xFFFFFFFF
    plane = packed[:, local_topk:].long() & 0xFFFFFFFF
    j = torch.arange(local_topk, device=packed.device)
    local = (0xFFFF - (words & 0xFFFF)) | (
        ((plane[:, j // _WARP] >> (j % _WARP)) & 1) << 16
    )
    bits = logits[:num_rows].contiguous().view(torch.int16).long() & 0xFFFF
    ordered = torch.where(bits >= 0x8000, ~bits & 0xFFFF, bits | 0x8000)
    is_nan = (bits & 0x7FFF) > 0x7F80
    for row in range(num_rows):
        end = min(int(row_ends[row]), logits.shape[1])
        sent = words[row] != 0
        ids = local[row][sent]
        valid = ~is_nan[row, :end]
        num_valid = int(valid.sum())
        assert ids.numel() == min(local_topk, num_valid), (
            f"row {row}: {ids.numel()} words, expected {min(local_topk, num_valid)}"
        )
        if ids.numel() == 0:
            continue
        assert int(ids.max()) < end, f"row {row}: local id past row_end {end}"
        assert ids.unique().numel() == ids.numel(), f"row {row}: repeated ids"
        assert not bool(is_nan[row, ids].any()), f"row {row}: NaN score sent"
        sent_ord = words[row][sent] >> 16
        assert torch.equal(sent_ord, ordered[row, ids]), (
            f"row {row}: a word's score is not its logit"
        )
        unsent = valid.clone()
        unsent[ids] = False
        if bool(unsent.any()):
            assert int(ordered[row, :end][unsent].max()) <= int(sent_ord.min()), (
                f"row {row}: not a top-{local_topk} by score"
            )


def dcp_topk_select(
    gathered: torch.Tensor,
    local_topk: int,
    topk_indices: torch.Tensor | None,
    *,
    topk: int | None = None,
    rows: _RowsOutput | None = None,
    exact: _ExactArgs | None = None,
) -> None:
    """Exact global top-K of the gathered `[W, T, N + N/32]` rows: the
    select, then the rescue (second round) of saturated rows."""
    if topk is None:
        assert topk_indices is not None
        topk = topk_indices.shape[1]
    if exact is None and local_topk < topk:
        raise ValueError("the DCP top-k select needs its rescue state (exact)")
    if exact is not None:
        _check_exact(gathered, topk, local_topk, rows, exact)
    _DCP_TOPK_SELECT_KERNEL(
        gathered,
        topk_indices,
        local_topk=local_topk,
        topk=topk,
        rows=rows,
        exact=exact,
    )
    if exact is not None:
        _DCP_TOPK_RESCUE_KERNEL(
            gathered,
            topk_indices,
            local_topk=local_topk,
            topk=topk,
            exact=exact,
            rows=rows,
        )


def _check_exact(
    gathered: torch.Tensor,
    topk: int,
    local_topk: int,
    rows: _RowsOutput | None,
    exact: _ExactArgs,
) -> None:
    """Host checks of an exact select call: the kernels index the rescue
    workspace (and peers' symmetric memory) by row, rank and slot unchecked,
    and CuTe accepts mismatched tensors silently."""
    world_size, num_rows = gathered.shape[0], gathered.shape[1]
    if (topk, local_topk, world_size) != (
        exact.topk,
        exact.local_topk,
        exact.world_size,
    ):
        raise ValueError(
            f"DCP top-k select (K={topk}, N={local_topk}, W={world_size}) does "
            f"not match its rescue workspace (K={exact.topk}, "
            f"N={exact.local_topk}, W={exact.world_size})"
        )
    if (
        gathered.dtype != torch.int32
        or gathered.dim() != 3
        or gathered.shape[2] != packed_row_words(local_topk)
        or gathered.stride(2) != 1
        or gathered.stride(1) != gathered.shape[2]
    ):
        raise ValueError(
            "gathered must be int32 [W, T, N + N/32] with contiguous rows, got "
            f"{gathered.dtype} {tuple(gathered.shape)} {gathered.stride()}"
        )
    if num_rows > exact.max_rows:
        raise ValueError(
            f"DCP top-k merge of {num_rows} rows exceeds its rescue workspace "
            f"({exact.max_rows} rows)"
        )
    if rows is not None and rows.dcp_rank != exact.rank:
        raise ValueError(
            f"dcp_rank {rows.dcp_rank} is not the rescue workspace's rank {exact.rank}"
        )
    logits, row_ends = exact.logits, exact.row_ends
    if (
        logits.dtype != torch.bfloat16
        or logits.dim() != 2
        or logits.stride(1) != 1
        or logits.shape[0] < num_rows
    ):
        raise ValueError(
            f"logits must be bf16 [>= {num_rows}, S] with stride(1) == 1, got "
            f"{logits.dtype} {tuple(logits.shape)} {logits.stride()}"
        )
    if (
        row_ends.dtype != torch.int32
        or row_ends.dim() != 1
        or row_ends.shape[0] < num_rows
        or row_ends.stride(0) != 1
    ):
        raise ValueError(
            f"row_ends must be contiguous int32 [>= {num_rows}], got "
            f"{row_ends.dtype} {tuple(row_ends.shape)} {row_ends.stride()}"
        )


def _all_gather_packed(packed: torch.Tensor, group: Any) -> torch.Tensor:
    if group is None:
        from vllm.distributed import get_dcp_group

        group = get_dcp_group()
    return group.all_gather(packed, dim=0).view(group.world_size, *packed.shape)


def _exact_args_for(
    topk: int,
    local_topk: int,
    logits: torch.Tensor | None,
    row_ends: torch.Tensor | None,
    group: Any = None,
) -> _ExactArgs:
    """Rescue state of an exact merge call. There is no lossy fallback: a
    missing workspace is an error (see `init_dcp_topk_merge`). The call
    itself is checked by `dcp_topk_select`."""
    assert local_topk < topk, (local_topk, topk)
    ws = get_dcp_topk_rescue_workspace(topk, local_topk)
    if ws is None:
        raise RuntimeError(
            f"DCP top-k merge (K={topk}, N={local_topk}) has no rescue "
            "workspace; call init_dcp_topk_merge at model construction"
        )
    if logits is None or row_ends is None:
        raise ValueError(
            "the exact DCP top-k merge needs this rank's bf16 logits and "
            "row_ends (a saturated rank re-reads its row)"
        )
    device_group = getattr(group, "device_group", None)
    if device_group is not None and ws.device_group is not None:  # noqa: SIM102
        if device_group is not ws.device_group:
            raise ValueError(
                "the DCP top-k merge's group is not the one its rescue "
                "workspace was built on"
            )
    return ws.args(logits, row_ends)


def dcp_topk_merge(
    logits: torch.Tensor,
    row_ends: torch.Tensor,
    topk_indices: torch.Tensor,
    local_topk: int,
    max_local_len: int | None = None,
) -> None:
    """Exact global top-K ids over the DCP group from this rank's bf16
    `logits` (the stock indexer hook). Collective over the DCP group.

    `logits` is `[T, S]` bf16 (row stride 1024-byte aligned for DeepSelect), `row_ends`
    the per-row local lengths (at most `max_local_len`, see `dcp_topk_pack`),
    `topk_indices` the `[T, K]` int32 output (global token ids, -1 padded).
    """
    logits = _bound_logits(logits, max_local_len)
    packed = dcp_topk_pack(logits, row_ends, local_topk)
    topk = topk_indices.shape[1]
    dcp_topk_select(
        _all_gather_packed(packed, None),
        local_topk,
        topk_indices,
        exact=_exact_args_for(topk, local_topk, logits, row_ends),
    )


def dcp_topk_merge_rows(
    *,
    logits: torch.Tensor | None = None,
    row_ends: torch.Tensor | None = None,
    packed: torch.Tensor | None = None,
    local_topk: int,
    topk: int,
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    stride_rows: int,
    dcp_rank: int,
    rows_out: torch.Tensor,
    valid_out: torch.Tensor,
    ids_out: torch.Tensor | None = None,
    group: Any = None,
    max_local_len: int | None = None,
) -> None:
    """Exact DCP top-K merge straight to this rank's physical KV rows.

    Input: `logits` `[T, S]` bf16 (row stride 1024-byte aligned, as DeepGEMM emits it)
    and `row_ends` `[T]` int32 local lengths, plus optionally `packed` `[T, N + N/32]`
    int32 (this rank's local top-N already in the B' layout); without `packed` the merge
    takes the local top-N itself (DeepSelect).

    Packed mode still requires `logits` and `row_ends`: a saturated rank re-reads its
    row's logits for the second round. The producer guarantees that `packed[t]` holds
    `min(N, v_t)` distinct local ids below `row_ends[t]` (`v_t` its non-NaN tokens),
    that they are a top-N by score of `logits[t, :row_ends[t]]`, that each word's score
    is that bf16 logit bit for bit, and that `logits` stays unmodified until this call
    returns (in stream order).

    Output (caller buffers, written for rows `[0, T)`): `rows_out` `[T, >= topk]` int32
    owned physical rows (compacted, -1 after; a rank may own up to `topk` of a row),
    `valid_out` `[T]` their count, `ids_out` `[T, topk]` the global ids (optional).

    Collective over the DCP group (see the module docstring).
    """
    if logits is None or row_ends is None:
        raise ValueError(
            "dcp_topk_merge_rows needs logits= and row_ends= (also with packed=)"
        )
    logits = _bound_logits(logits, max_local_len)
    if packed is None:
        packed = dcp_topk_pack(logits, row_ends, local_topk)
    else:
        if (
            packed.dtype != torch.int32
            or packed.dim() != 2
            or packed.shape[1] != packed_row_words(local_topk)
        ):
            raise ValueError(
                f"packed must be int32 [T, {packed_row_words(local_topk)}], got "
                f"{packed.dtype} {tuple(packed.shape)}"
            )
    num_rows = packed.shape[0]
    exact = _exact_args_for(topk, local_topk, logits, row_ends, group)
    if rows_out.shape[0] != num_rows or rows_out.shape[1] < topk:
        # A rank may own up to K entries of a saturated row.
        raise ValueError(
            f"rows_out must be [{num_rows}, >= {topk}], got {tuple(rows_out.shape)}"
        )
    if valid_out.shape[0] != num_rows or req_id.shape[0] < num_rows:
        raise ValueError("valid_out must have T rows and req_id at least T")
    if ids_out is not None and ids_out.shape != (num_rows, topk):
        raise ValueError(f"ids_out must be [{num_rows}, {topk}]")
    if block_table.stride(1) != 1:
        raise ValueError("block_table rows must be contiguous")
    rows = _RowsOutput(
        req_id=req_id,
        block_table=block_table,
        rows_out=rows_out,
        valid_out=valid_out,
        dcp_rank=dcp_rank,
        block_size=block_size,
        stride_rows=stride_rows,
    )
    dcp_topk_select(
        _all_gather_packed(packed, group),
        local_topk,
        ids_out,
        topk=topk,
        rows=rows,
        exact=exact,
    )
