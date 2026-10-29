# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# CuTeDSL and Triton JIT signatures cannot use regular Python type annotations.
# ruff: noqa: ANN001, ANN201, ANN202, ANN204, ANN205

"""CuTeDSL stable top-k selector for the direct DCP top-k candidate exchange."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Int64, Uint32
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import recast_val


def stable_topk_from_pingpong_candidates_cutedsl(
    inboxes: torch.Tensor,
    write_seq: torch.Tensor,
    topk: int,
    out: torch.Tensor,
    use_pdl: bool,
    score_bits: int = 32,
    id_bits: int = 32,
) -> None:
    """`score_bits`: high score-key bits that rank (16 for upcast BF16 scores).
    `id_bits`: low bits that hold every token id."""
    StableTopKFromPingPongCandidatesKernel.compile(
        topk,
        inboxes.shape[2],
        inboxes.shape[1],
        use_pdl,
        score_bits,
        id_bits,
    )(inboxes, write_seq, out)


@cute.jit
def _warp_scan_inclusive_i32(val: Int32, lane: Int32) -> Int32:
    for i in cutlass.range_constexpr(cute.arch.WARP_SIZE.bit_length() - 1):
        offset = 1 << i
        partial = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            val += partial
    return val


@cute.jit
def _block_scan_inclusive_i32(
    val: Int32,
    lane: Int32,
    warp_id: Int32,
    warp_scratch: cute.Tensor,
    warps_per_block: int,
) -> Int32:
    prefix = _warp_scan_inclusive_i32(val, lane)
    if lane == Int32(cute.arch.WARP_SIZE - 1):
        warp_scratch[0, warp_id] = prefix
    cute.arch.sync_threads()

    if warp_id == Int32(0):
        warp_total = Int32(0)
        if lane < Int32(warps_per_block):
            warp_total = warp_scratch[0, lane]
        warp_prefix = _warp_scan_inclusive_i32(warp_total, lane)
        if lane < Int32(warps_per_block):
            warp_scratch[0, lane] = warp_prefix - warp_total
    cute.arch.sync_threads()

    return prefix + warp_scratch[0, warp_id]


class StableTopKFromPingPongCandidatesKernel:
    tb_size = 512
    hist_bins = 2048
    radix_bits = (hist_bins - 1).bit_length()
    assert hist_bins == 1 << radix_bits
    key_bits = Uint32.width
    hist_chunks = (hist_bins + tb_size - 1) // tb_size
    warps_per_block = tb_size // cute.arch.WARP_SIZE

    def __init__(
        self,
        topk: int,
        num_candidates: int,
        use_pdl: bool,
        score_bits: int = 32,
        id_bits: int = 32,
    ):
        assert num_candidates % self.tb_size == 0, (
            "Stable top-k requires candidate count "
            f"to be a multiple of {self.tb_size}, got {num_candidates}"
        )
        self.topk = topk
        self.keys_per_thread = num_candidates // self.tb_size
        self.use_pdl = use_pdl
        assert 0 < score_bits <= self.key_bits and 0 < id_bits <= self.key_bits
        self.score_bits = score_bits
        self.id_bits = id_bits
        # The high token-id bits fill the score key's spare low bits, so score
        # ties are broken in the score passes; only the remaining low id bits
        # (none when they all fit) need the token-id passes.
        self.fold_bits = min(id_bits, self.key_bits - score_bits)
        self.rest_id_bits = id_bits - self.fold_bits
        self.ranked_bits = score_bits + self.fold_bits
        self.score_passes, self.score_final_bits = self._passes(self.ranked_bits)
        self.id_passes, self.id_final_bits = self._passes(max(self.rest_id_bits, 1))

        @cute.struct
        class SharedStorage:
            hist: cute.struct.MemRange[Int32, self.hist_bins]
            committed_count: cute.struct.MemRange[Int32, 1]
            running_count: cute.struct.MemRange[Int32, 1]
            threshold_bin: cute.struct.MemRange[Int32, 1]
            threshold_found: cute.struct.MemRange[Int32, 1]
            include_threshold_bin: cute.struct.MemRange[Int32, 1]
            needs_tie_break: cute.struct.MemRange[Int32, 1]
            prefix_s: cute.struct.MemRange[Uint32, 1]
            warp_totals: cute.struct.MemRange[Int32, self.warps_per_block]

        self.shared_storage = SharedStorage

    @cute.jit
    def _score_key(self, score: Float32, token_id: Int32) -> Uint32:
        bits = recast_val(score, Uint32)
        mask = Uint32(0x80000000)
        if (bits & Uint32(0x80000000)) != Uint32(0):
            mask = Uint32(0xFFFFFFFF)
        key = bits ^ mask
        if cutlass.const_expr(self.score_bits < self.key_bits):
            # Only the high score bits rank (the rest are zero for upcast BF16,
            # all ones once a negative score is flipped).
            shift = Uint32(self.key_bits - self.score_bits)
            key = (key >> shift) << shift
        if cutlass.const_expr(self.fold_bits > 0):
            reversed_id = ~Uint32(token_id)
            if cutlass.const_expr(self.id_bits < self.key_bits):
                reversed_id = reversed_id & Uint32((1 << self.id_bits) - 1)
            key = key | (
                (reversed_id >> Uint32(self.rest_id_bits))
                << Uint32(self.key_bits - self.ranked_bits)
            )
        if token_id < Int32(0):
            key = Uint32(0)
        return key

    @classmethod
    def _passes(cls, bits: int) -> tuple[int, int]:
        """Radix passes over the top `bits` key bits, and the final pass width."""
        passes = (bits + cls.radix_bits - 1) // cls.radix_bits
        return passes, bits - cls.radix_bits * (passes - 1)

    @cute.jit
    def _token_id_key(self, token_id: Int32) -> Uint32:
        # Smaller ids rank first; the id bits not folded into the score key
        # go to the top of the key.
        key = ~Uint32(token_id)
        if cutlass.const_expr(0 < self.rest_id_bits < self.key_bits):
            key = key << Uint32(self.key_bits - self.rest_id_bits)
        return key

    @cute.jit
    def _prefix_matches(
        self,
        key: Uint32,
        prefix: Uint32,
        prefix_bits: Int32,
    ):
        matches = prefix_bits == Int32(0)
        if prefix_bits != Int32(0):
            shift = Int32(self.key_bits) - prefix_bits
            matches = (key >> Uint32(shift)) == (prefix >> Uint32(shift))
        return matches

    @cute.jit
    def _radix_pass(
        self,
        score_keys: cute.Tensor,
        token_ids: cute.Tensor,
        output: cute.Tensor,
        storage,
        tid: Int32,
        step: Int32,
        bits: int,
        is_final_pass: bool,
        use_token_id_key: bool,
        score_threshold: Uint32,
    ):
        hist_smem = storage.hist.get_tensor(cute.make_layout((self.hist_bins,)))
        committed_count_smem = storage.committed_count.data_ptr()
        running_count_smem = storage.running_count.data_ptr()
        threshold_bin_smem = storage.threshold_bin.data_ptr()
        threshold_found_smem = storage.threshold_found.data_ptr()
        include_threshold_bin_smem = storage.include_threshold_bin.data_ptr()
        needs_tie_break_smem = storage.needs_tie_break.data_ptr()
        prefix_smem = storage.prefix_s.data_ptr()
        warp_totals_smem = storage.warp_totals.get_tensor(
            cute.make_layout((1, self.warps_per_block))
        )

        prefix_bits = step * Int32(self.radix_bits)
        num_bins = 1 << bits
        # Narrow final passes still scan whole tb_size slices; bins at or above
        # num_bins stay zero.
        scan_bins = max(num_bins, self.tb_size)
        block_scan_iterations = scan_bins // self.tb_size
        shift = Int32(self.key_bits) - prefix_bits - Int32(bits)
        bin_mask = Uint32(num_bins - 1)
        prefix = prefix_smem.load()

        for chunk in cutlass.range_constexpr(self.hist_chunks):
            hist_smem[tid + Int32(chunk * self.tb_size)] = Int32(0)
        if tid == Int32(0):
            running_count_smem.store(committed_count_smem.load())
            include_threshold_bin_smem.store(Int32(0))
            if cutlass.const_expr(not use_token_id_key):
                needs_tie_break_smem.store(Int32(0))
            threshold_found_smem.store(Int32(0))
        cute.arch.sync_threads()

        for key_idx in cutlass.range_constexpr(self.keys_per_thread):
            token_id = token_ids[key_idx]
            key = score_keys[key_idx]
            valid = token_id >= Int32(0)
            if cutlass.const_expr(use_token_id_key):
                valid = valid and key == score_threshold
                key = self._token_id_key(token_id)
            if valid and self._prefix_matches(key, prefix, prefix_bits):
                bin_idx = Int32((key >> Uint32(shift)) & bin_mask)
                cute.arch.atomic_add(
                    hist_smem.iterator + bin_idx,
                    Int32(1),
                    sem="relaxed",
                    scope="cta",
                )
        cute.arch.sync_threads()

        lane = cute.arch.lane_idx()
        warp_id = cute.arch.warp_idx()
        # Each iteration scans one tb_size-wide slice of bins, high to low.
        iter = Int32(0)
        threshold_found = threshold_found_smem.load()
        while threshold_found == Int32(0) and iter < Int32(block_scan_iterations):
            bin_idx = Int32(scan_bins - 1) - (iter * Int32(self.tb_size) + tid)
            count = hist_smem[bin_idx]
            chunk_inclusive = _block_scan_inclusive_i32(
                count,
                lane,
                warp_id,
                warp_totals_smem,
                self.warps_per_block,
            )
            running_count = running_count_smem.load()
            prior_in_scan_slice = chunk_inclusive - count
            remaining = Int32(self.topk) - running_count - prior_in_scan_slice
            if count > Int32(0) and remaining > Int32(0) and remaining <= count:
                threshold_bin_smem.store(bin_idx)
                if count <= remaining or cutlass.const_expr(
                    is_final_pass and use_token_id_key
                ):
                    include_threshold_bin_smem.store(Int32(1))
                elif cutlass.const_expr(is_final_pass and not use_token_id_key):
                    needs_tie_break_smem.store(Int32(1))
                threshold_found_smem.store(Int32(1))
            # Barrier: every thread must finish reading running_count for this
            # slice before tb_size-1 advances it, else a warp racing ahead to
            # the store makes a lagging thread double-count the slice total
            # (-> remaining too small -> threshold too high -> under-fill).
            cute.arch.sync_threads()
            if tid == Int32(self.tb_size - 1):
                running_count_smem.store(running_count + chunk_inclusive)
            cute.arch.sync_threads()

            threshold_found = threshold_found_smem.load()
            iter += Int32(1)

        threshold = threshold_bin_smem.load()
        should_include_threshold = include_threshold_bin_smem.load() != Int32(0)
        for key_idx in cutlass.range_constexpr(self.keys_per_thread):
            token_id = token_ids[key_idx]
            key = score_keys[key_idx]
            valid = token_id >= Int32(0)
            if cutlass.const_expr(use_token_id_key):
                valid = valid and key == score_threshold
                key = self._token_id_key(token_id)
            if valid and self._prefix_matches(key, prefix, prefix_bits):
                bin_idx = Int32((key >> Uint32(shift)) & bin_mask)
                selected = bin_idx > threshold
                if should_include_threshold:
                    selected = selected or bin_idx == threshold
                if selected:
                    dst = cute.arch.atomic_add(
                        committed_count_smem,
                        Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                    if dst < Int32(self.topk):
                        output[dst] = token_id
        cute.arch.sync_threads()

        pass_finished = include_threshold_bin_smem.load()
        needs_tie_break = Int32(0)
        if cutlass.const_expr(not use_token_id_key):
            needs_tie_break = needs_tie_break_smem.load()
            if needs_tie_break != Int32(0):
                pass_finished = Int32(1)
        if tid == Int32(0) and (
            pass_finished == Int32(0) or needs_tie_break != Int32(0)
        ):
            prefix_smem.store(prefix | (Uint32(threshold) << Uint32(shift)))
        cute.arch.sync_threads()
        return pass_finished

    @cute.jit
    def _run_radix_passes(
        self,
        score_keys: cute.Tensor,
        token_ids: cute.Tensor,
        output: cute.Tensor,
        storage,
        tid: Int32,
        use_token_id_key: bool,
        score_threshold: Uint32,
    ):
        step = Int32(0)
        finished = Int32(0)
        radix_passes, final_radix_bits = self.score_passes, self.score_final_bits
        if cutlass.const_expr(use_token_id_key):
            radix_passes, final_radix_bits = self.id_passes, self.id_final_bits
        while finished == Int32(0) and step < Int32(radix_passes - 1):
            finished = self._radix_pass(
                score_keys,
                token_ids,
                output,
                storage,
                tid,
                step,
                self.radix_bits,
                False,
                use_token_id_key,
                score_threshold,
            )
            step += Int32(1)

        if finished == Int32(0):
            self._radix_pass(
                score_keys,
                token_ids,
                output,
                storage,
                tid,
                Int32(radix_passes - 1),
                final_radix_bits,
                True,
                use_token_id_key,
                score_threshold,
            )

    @cute.jit
    def _select_row(
        self,
        input_row: cute.Tensor,
        output_row: cute.Tensor,
        tid: Int32,
    ):
        score_keys = cute.make_rmem_tensor((self.keys_per_thread,), Uint32)
        token_ids = cute.make_rmem_tensor((self.keys_per_thread,), Int32)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage, 8)
        committed_count_smem = storage.committed_count.data_ptr()
        running_count_smem = storage.running_count.data_ptr()
        prefix_smem = storage.prefix_s.data_ptr()
        needs_tie_break_smem = storage.needs_tie_break.data_ptr()
        warp_totals_smem = storage.warp_totals.get_tensor(
            cute.make_layout((1, self.warps_per_block))
        )
        valid_count = Int32(0)
        for key_idx in cutlass.range_constexpr(self.keys_per_thread):
            col = tid + Int32(key_idx * self.tb_size)
            token_id = Int32(input_row[col, 1])
            token_ids[key_idx] = token_id
            if token_id >= Int32(0):
                valid_count += Int32(1)

        if tid == Int32(0):
            committed_count_smem.store(Int32(0))
            prefix_smem.store(Uint32(0))
            needs_tie_break_smem.store(Int32(0))
        cute.arch.sync_threads()

        lane = cute.arch.lane_idx()
        warp_id = cute.arch.warp_idx()
        valid_prefix = _block_scan_inclusive_i32(
            valid_count,
            lane,
            warp_id,
            warp_totals_smem,
            self.warps_per_block,
        )
        if tid == Int32(self.tb_size - 1):
            running_count_smem.store(valid_prefix)
        cute.arch.sync_threads()

        total_valid = running_count_smem.load()
        if total_valid <= Int32(self.topk):
            if total_valid < Int32(self.topk):
                for i in range(tid, self.topk, self.tb_size):
                    if i >= total_valid:
                        output_row[i] = Int32(-1)
            dst = valid_prefix - valid_count
            for key_idx in cutlass.range_constexpr(self.keys_per_thread):
                token_id = token_ids[key_idx]
                if token_id >= Int32(0):
                    output_row[dst] = token_id
                    dst += Int32(1)
        else:
            for key_idx in cutlass.range_constexpr(self.keys_per_thread):
                col = tid + Int32(key_idx * self.tb_size)
                score = Float32(input_row[col, 0])
                score_keys[key_idx] = self._score_key(score, token_ids[key_idx])

            self._run_radix_passes(
                score_keys,
                token_ids,
                output_row,
                storage,
                tid,
                False,
                Uint32(0),
            )
            # Only exact score ties at the top-k boundary need token-id radix.
            if needs_tie_break_smem.load() != Int32(0):
                score_threshold = prefix_smem.load()
                # Every warp must retain the threshold before the shared slot is
                # reused as the token-id radix prefix.
                cute.arch.sync_threads()
                if tid == Int32(0):
                    prefix_smem.store(Uint32(0))
                cute.arch.sync_threads()

                self._run_radix_passes(
                    score_keys,
                    token_ids,
                    output_row,
                    storage,
                    tid,
                    True,
                    score_threshold,
                )

    @cute.jit
    def __call__(
        self,
        inboxes: cute.Tensor,
        write_seq: cute.Tensor,
        out: cute.Tensor,
        stream: CUstream,
    ):
        grid = (out.shape[0], 1, 1)
        self.kernel(inboxes, write_seq, out).launch(
            grid=grid,
            block=(self.tb_size, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        inboxes: cute.Tensor,
        write_seq: cute.Tensor,
        out: cute.Tensor,
    ):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        # The predecessor releases this grid after all inbox slices are ready.
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        epoch = Int32(write_seq[0])
        slot = (epoch - Int32(1)) & Int32(1)
        input_row = inboxes[slot, row, None, None]
        output_row = out[row, None]
        self._select_row(input_row, output_row, tid)

    @cache
    @staticmethod
    def compile(
        topk: int,
        num_candidates: int,
        max_rows: int,
        use_pdl: bool,
        score_bits: int = 32,
        id_bits: int = 32,
    ):
        num_rows = cute.sym_int()
        inboxes = cute.runtime.make_fake_tensor(
            Float32,
            (2, max_rows, num_candidates, 2),
            stride=(
                max_rows * num_candidates * 2,
                num_candidates * 2,
                2,
                1,
            ),
            assumed_align=8,
        )
        write_seq = make_fake_tensor(Int64, (1,), divisibility=1)
        out = make_fake_tensor(Int32, (num_rows, topk), divisibility=1)

        kernel = StableTopKFromPingPongCandidatesKernel(
            topk, num_candidates, use_pdl, score_bits, id_bits
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            inboxes,
            write_seq,
            out,
            stream,
            options="--enable-tvm-ffi",
        )
