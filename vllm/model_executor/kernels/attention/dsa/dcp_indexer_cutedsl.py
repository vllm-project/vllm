# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Uint32, Uint64
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import recast_val
from vllm.model_executor.warmup.jit_warmup import (
    VllmJitKernel,
)
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import compile_cutedsl
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor
from vllm.triton_utils import tl, triton


def stable_topk_from_gathered_candidates_cutedsl(
    gathered: torch.Tensor,
    topk: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(
            (gathered.shape[0], topk),
            dtype=torch.int32,
            device=gathered.device,
        )
    _STABLE_TOPK_FROM_GATHERED_CANDIDATES_KERNEL(gathered, out, topk=topk)
    return out


def pack_dcp_topk_candidates_cutedsl(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    packed: torch.Tensor,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
    row_starts: torch.Tensor | None,
) -> None:
    topk = topk_indices.shape[1]
    row_starts_arg = row_starts if row_starts is not None else topk_indices
    _PACK_DCP_TOPK_CANDIDATES_KERNEL(
        logits,
        topk_indices,
        packed,
        row_starts_arg,
        logits_stride0=logits.stride(0),
        logits_stride1=logits.stride(1),
        topk_stride0=topk_indices.stride(0),
        topk_stride1=topk_indices.stride(1),
        packed_stride0=packed.stride(0),
        packed_stride1=packed.stride(1),
        packed_stride2=packed.stride(2),
        num_cols=logits.shape[1],
        dcp_rank=dcp_rank,
        dcp_world_size=dcp_world_size,
        cp_interleave=cp_interleave,
        has_row_starts=row_starts is not None,
        topk=topk,
        block_size=512,
    )


class PackDCPTopkCandidatesKernel(
    VllmJitKernel["PackDCPTopkCandidatesKernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        dcp_rank: int
        dcp_world_size: int
        cp_interleave: int
        has_row_starts: bool
        topk: int
        block_size: int

    @staticmethod
    @triton.jit(
        do_not_specialize=[
            "logits_stride0",
            "logits_stride1",
            "topk_stride0",
            "topk_stride1",
            "packed_stride0",
            "packed_stride1",
            "packed_stride2",
            "num_cols",
        ]
    )
    def kernel(
        logits,
        topk_indices,
        packed,
        row_starts,
        logits_stride0,
        logits_stride1,
        topk_stride0,
        topk_stride1,
        packed_stride0,
        packed_stride1,
        packed_stride2,
        num_cols,
        DCP_RANK: tl.constexpr,
        DCP_WORLD_SIZE: tl.constexpr,
        CP_INTERLEAVE: tl.constexpr,
        HAS_ROW_STARTS: tl.constexpr,
        TOPK: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        cols = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < TOPK

        local_idx = tl.load(
            topk_indices + row * topk_stride0 + cols * topk_stride1,
            mask=mask,
            other=-1,
        )
        valid = local_idx >= 0
        safe_local_idx = tl.maximum(local_idx, 0)

        row_start = 0
        if HAS_ROW_STARTS:
            row_start = tl.load(row_starts + row)

        score_col = safe_local_idx + row_start
        score_col = tl.minimum(score_col, tl.maximum(num_cols - 1, 0))
        score = tl.load(
            logits + row * logits_stride0 + score_col * logits_stride1,
            mask=mask & valid,
            other=-float("inf"),
        )

        global_id = (
            (safe_local_idx // CP_INTERLEAVE) * (DCP_WORLD_SIZE * CP_INTERLEAVE)
            + DCP_RANK * CP_INTERLEAVE
            + safe_local_idx % CP_INTERLEAVE
        )
        global_id = tl.where(valid, global_id, -1)

        packed_base = packed + row * packed_stride0 + cols * packed_stride1
        tl.store(packed_base, score, mask=mask)
        tl.store(packed_base + packed_stride2, global_id.to(tl.float32), mask=mask)

    def dispatch(  # type: ignore[override]
        self,
        *,
        dcp_rank: int,
        dcp_world_size: int,
        cp_interleave: int,
        has_row_starts: bool,
        topk: int,
        block_size: int,
    ) -> CompileKey:
        return self.CompileKey(
            dcp_rank=dcp_rank,
            dcp_world_size=dcp_world_size,
            cp_interleave=cp_interleave,
            has_row_starts=has_row_starts,
            topk=topk,
            block_size=block_size,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        if dcp_world_size <= 1:
            return []
        cp_interleave = vllm_config.parallel_config.cp_kv_cache_interleave_size
        topk = vllm_config.model_config.hf_config.index_topk
        if topk <= 0:
            return []

        try:
            from vllm.distributed.parallel_state import get_dcp_group

            dcp_rank = get_dcp_group().rank_in_group
        except Exception:
            dcp_rank = 0

        return self._trace_dispatch(self.dispatch)(
            dcp_rank=dcp_rank,
            dcp_world_size=dcp_world_size,
            cp_interleave=cp_interleave,
            has_row_starts=(False, True),
            topk=topk,
            block_size=512,
        )

    def compile(self, compile_key: CompileKey) -> None:
        warmup = getattr(self.kernel, "warmup", None)
        assert warmup is not None
        fp32_ptr = TritonWarmupTensor(torch.float32)
        int32_ptr = TritonWarmupTensor(torch.int32)
        warmup(
            fp32_ptr,
            int32_ptr,
            fp32_ptr,
            int32_ptr,
            1,  # do not specialize logits_stride0
            1,  # do not specialize logits_stride1
            1,  # do not specialize topk_stride0
            1,  # do not specialize topk_stride1
            1,  # do not specialize packed_stride0
            1,  # do not specialize packed_stride1
            1,  # do not specialize packed_stride2
            1,  # do not specialize num_cols
            DCP_RANK=compile_key.dcp_rank,
            DCP_WORLD_SIZE=compile_key.dcp_world_size,
            CP_INTERLEAVE=compile_key.cp_interleave,
            HAS_ROW_STARTS=compile_key.has_row_starts,
            TOPK=compile_key.topk,
            BLOCK_SIZE=compile_key.block_size,
            grid=(1, 1),
            num_warps=8,
        )

    def __call__(
        self,
        logits: torch.Tensor,
        topk_indices: torch.Tensor,
        packed: torch.Tensor,
        row_starts_arg: torch.Tensor,
        *,
        logits_stride0: int,
        logits_stride1: int,
        topk_stride0: int,
        topk_stride1: int,
        packed_stride0: int,
        packed_stride1: int,
        packed_stride2: int,
        num_cols: int,
        dcp_rank: int,
        dcp_world_size: int,
        cp_interleave: int,
        has_row_starts: bool,
        topk: int,
        block_size: int,
    ) -> None:
        compile_key = self.dispatch(
            dcp_rank=dcp_rank,
            dcp_world_size=dcp_world_size,
            cp_interleave=cp_interleave,
            has_row_starts=has_row_starts,
            topk=topk,
            block_size=block_size,
        )
        self._guard_warmup_call(compile_key)
        grid = (topk_indices.shape[0], triton.cdiv(topk, block_size))
        self.kernel[grid](
            logits,
            topk_indices,
            packed,
            row_starts_arg,
            logits_stride0,
            logits_stride1,
            topk_stride0,
            topk_stride1,
            packed_stride0,
            packed_stride1,
            packed_stride2,
            num_cols,
            DCP_RANK=dcp_rank,
            DCP_WORLD_SIZE=dcp_world_size,
            CP_INTERLEAVE=cp_interleave,
            HAS_ROW_STARTS=has_row_starts,
            TOPK=topk,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )


class StableTopKFromGatheredCandidatesKernel(
    VllmJitKernel["StableTopKFromGatheredCandidatesKernel.CompileKey"]
):
    tb_size = 512
    hist_bins = 2048
    radix_bits = (hist_bins - 1).bit_length()
    key_bits = Uint64.width
    radix_passes = (key_bits + radix_bits - 1) // radix_bits
    final_radix_bits = key_bits - radix_bits * (radix_passes - 1)
    hist_chunks = (hist_bins + tb_size - 1) // tb_size
    warps_per_block = tb_size // cute.arch.WARP_SIZE

    @dataclass(frozen=True)
    class CompileKey:
        topk: int
        num_candidates: int

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        tb_size = StableTopKFromGatheredCandidatesKernel.tb_size
        hist_bins = StableTopKFromGatheredCandidatesKernel.hist_bins
        radix_bits = StableTopKFromGatheredCandidatesKernel.radix_bits
        key_bits = StableTopKFromGatheredCandidatesKernel.key_bits
        radix_passes = StableTopKFromGatheredCandidatesKernel.radix_passes
        final_radix_bits = StableTopKFromGatheredCandidatesKernel.final_radix_bits
        hist_chunks = StableTopKFromGatheredCandidatesKernel.hist_chunks
        warps_per_block = StableTopKFromGatheredCandidatesKernel.warps_per_block
        topk = compile_key.topk
        assert hist_bins == 1 << radix_bits
        assert compile_key.num_candidates % tb_size == 0, (
            "StableTopKFromGatheredCandidatesKernel requires candidate count "
            f"to be a multiple of {tb_size}, got {compile_key.num_candidates}"
        )
        keys_per_thread = compile_key.num_candidates // tb_size

        @cute.struct
        class SharedStorage:
            hist: cute.struct.MemRange[Int32, hist_bins]
            committed_count: cute.struct.MemRange[Int32, 1]
            running_count: cute.struct.MemRange[Int32, 1]
            threshold_bin: cute.struct.MemRange[Int32, 1]
            threshold_found: cute.struct.MemRange[Int32, 1]
            include_threshold_bin: cute.struct.MemRange[Int32, 1]
            prefix_s: cute.struct.Align[cute.struct.MemRange[Uint64, 1], 8]
            warp_totals: cute.struct.MemRange[Int32, warps_per_block]

        shared_storage = SharedStorage

        @cute.jit
        def warp_scan_inclusive_i32(val: Int32, lane: Int32) -> Int32:
            for i in cutlass.range_constexpr(cute.arch.WARP_SIZE.bit_length() - 1):
                offset = 1 << i
                partial = cute.arch.shuffle_sync_up(
                    val, offset=offset, mask_and_clamp=0
                )
                if lane >= offset:
                    val += partial
            return val

        @cute.jit
        def block_scan_inclusive_i32(
            val: Int32,
            lane: Int32,
            warp_id: Int32,
            warp_scratch: cute.Tensor,
            warps_per_block: int,
        ) -> Int32:
            prefix = warp_scan_inclusive_i32(val, lane)
            if lane == Int32(cute.arch.WARP_SIZE - 1):
                warp_scratch[0, warp_id] = prefix
            cute.arch.sync_threads()

            if warp_id == Int32(0):
                warp_total = Int32(0)
                if lane < Int32(warps_per_block):
                    warp_total = warp_scratch[0, lane]
                warp_prefix = warp_scan_inclusive_i32(warp_total, lane)
                if lane < Int32(warps_per_block):
                    warp_scratch[0, lane] = warp_prefix - warp_total
            cute.arch.sync_threads()

            return prefix + warp_scratch[0, warp_id]

        @cute.jit
        def stable_key(score: Float32, token_id: Int32) -> Uint64:
            bits = recast_val(score, Uint32)
            mask = Uint32(0x80000000)
            if (bits & Uint32(0x80000000)) != Uint32(0):
                mask = Uint32(0xFFFFFFFF)
            score_key = Uint64(bits ^ mask) << Uint64(32)
            id_key = Uint64(~Uint32(token_id))
            key = score_key | id_key
            if token_id < Int32(0):
                key = Uint64(0)
            return key

        @cute.jit
        def prefix_matches(
            key: Uint64,
            prefix: Uint64,
            prefix_bits: Int32,
        ):
            matches = prefix_bits == Int32(0)
            if prefix_bits != Int32(0):
                shift = Int32(key_bits) - prefix_bits
                matches = (key >> Uint64(shift)) == (prefix >> Uint64(shift))
            return matches

        @cute.jit
        def radix_pass(
            keys: cute.Tensor,
            output: cute.Tensor,
            storage,
            tid: Int32,
            step: Int32,
            bits: int,
            is_final_pass: bool,
        ):
            hist_smem = storage.hist.get_tensor(cute.make_layout((hist_bins,)))
            committed_count_smem = storage.committed_count.data_ptr()
            running_count_smem = storage.running_count.data_ptr()
            threshold_bin_smem = storage.threshold_bin.data_ptr()
            threshold_found_smem = storage.threshold_found.data_ptr()
            include_threshold_bin_smem = storage.include_threshold_bin.data_ptr()
            prefix_smem = storage.prefix_s.data_ptr()
            warp_totals_smem = storage.warp_totals.get_tensor(
                cute.make_layout((1, warps_per_block))
            )

            prefix_bits = step * Int32(radix_bits)
            num_bins = 1 << bits
            block_scan_iterations = (num_bins + tb_size - 1) // tb_size
            shift = Int32(key_bits) - prefix_bits - Int32(bits)
            bin_mask = Uint64(num_bins - 1)
            prefix = prefix_smem.load()

            for chunk in cutlass.range_constexpr(hist_chunks):
                hist_smem[tid + Int32(chunk * tb_size)] = Int32(0)
            if tid == Int32(0):
                running_count_smem.store(committed_count_smem.load())
                include_threshold_bin_smem.store(Int32(0))
                threshold_found_smem.store(Int32(0))
            cute.arch.sync_threads()

            for key_idx in cutlass.range_constexpr(keys_per_thread):
                key = keys[key_idx]
                if prefix_matches(key, prefix, prefix_bits):
                    bin_idx = Int32((key >> Uint64(shift)) & bin_mask)
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
                bin_idx = Int32(num_bins - 1) - (iter * Int32(tb_size) + tid)
                count = hist_smem[bin_idx]
                chunk_inclusive = block_scan_inclusive_i32(
                    count,
                    lane,
                    warp_id,
                    warp_totals_smem,
                    warps_per_block,
                )
                running_count = running_count_smem.load()
                prior_in_scan_slice = chunk_inclusive - count
                remaining = Int32(topk) - running_count - prior_in_scan_slice
                if count > Int32(0) and remaining > Int32(0) and remaining <= count:
                    threshold_bin_smem.store(bin_idx)
                    if count <= remaining or cutlass.const_expr(is_final_pass):
                        include_threshold_bin_smem.store(Int32(1))
                    threshold_found_smem.store(Int32(1))
                # Barrier: every thread must finish reading running_count for this
                # slice before tb_size-1 advances it, else a warp racing ahead to
                # the store makes a lagging thread double-count the slice total
                # (-> remaining too small -> threshold too high -> under-fill).
                cute.arch.sync_threads()
                if tid == Int32(tb_size - 1):
                    running_count_smem.store(running_count + chunk_inclusive)
                cute.arch.sync_threads()

                threshold_found = threshold_found_smem.load()
                iter += Int32(1)

            threshold = threshold_bin_smem.load()
            should_include_threshold = include_threshold_bin_smem.load() != Int32(0)
            for key_idx in cutlass.range_constexpr(keys_per_thread):
                key = keys[key_idx]
                if prefix_matches(key, prefix, prefix_bits):
                    bin_idx = Int32((key >> Uint64(shift)) & bin_mask)
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
                        if dst < Int32(topk):
                            output[dst] = recast_val(~Uint32(key), Int32)
            cute.arch.sync_threads()

            pass_finished = include_threshold_bin_smem.load()
            if tid == Int32(0) and pass_finished == Int32(0):
                prefix_smem.store(prefix | (Uint64(threshold) << Uint64(shift)))
            cute.arch.sync_threads()
            return pass_finished

        @cute.kernel
        def device_kernel(input: cute.Tensor, out: cute.Tensor):
            row, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            input_row = input[row, None, None]
            output_row = out[row, None]
            keys = cute.make_rmem_tensor((keys_per_thread,), Uint64)

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(shared_storage, 8)
            committed_count_smem = storage.committed_count.data_ptr()
            prefix_smem = storage.prefix_s.data_ptr()
            for i in range(tid, topk, tb_size):
                output_row[i] = Int32(-1)

            for key_idx in cutlass.range_constexpr(keys_per_thread):
                col = tid + Int32(key_idx * tb_size)
                score = Float32(input_row[col, 0])
                token_id = Int32(input_row[col, 1])
                keys[key_idx] = stable_key(score, token_id)

            if tid == Int32(0):
                committed_count_smem.store(Int32(0))
                prefix_smem.store(Uint64(0))
            cute.arch.sync_threads()

            step = Int32(0)
            finished = Int32(0)
            while finished == Int32(0) and step < Int32(radix_passes - 1):
                finished = radix_pass(
                    keys,
                    output_row,
                    storage,
                    tid,
                    step,
                    radix_bits,
                    False,
                )
                step += Int32(1)

            if finished == Int32(0):
                radix_pass(
                    keys,
                    output_row,
                    storage,
                    tid,
                    Int32(radix_passes - 1),
                    final_radix_bits,
                    True,
                )

        @cute.jit
        def host_entrypoint(
            gathered: cute.Tensor,
            out: cute.Tensor,
            stream: CUstream,
        ):
            grid = (gathered.shape[0], 1, 1)
            device_kernel(gathered, out).launch(
                grid=grid,
                block=(tb_size, 1, 1),
                stream=stream,
            )

        return host_entrypoint

    def dispatch(  # type: ignore[override]
        self,
        *,
        topk: int,
        num_candidates: int,
    ) -> CompileKey:
        return self.CompileKey(topk=topk, num_candidates=num_candidates)

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        if dcp_world_size <= 1:
            return []
        topk = vllm_config.model_config.hf_config.index_topk
        if topk <= 0:
            return []
        return self._trace_dispatch(self.dispatch)(
            topk=topk,
            num_candidates=topk * dcp_world_size,
        )

    def compile(self, compile_key: CompileKey) -> None:
        cache_key = (compile_key.topk, compile_key.num_candidates)
        if self._compiled_cache_contains(
            compile_key,
            cache_key=cache_key,
        ):
            return

        num_rows = cute.sym_int()
        gathered = cute.runtime.make_fake_tensor(
            Float32,
            (num_rows, compile_key.num_candidates, 2),
            stride=(cute.sym_int64(divisibility=2), 2, 1),
            assumed_align=8,
        )
        out = make_fake_tensor(
            Int32,
            (num_rows, compile_key.topk),
            divisibility=1,
        )
        self._compiled_cache[cache_key] = compile_cutedsl(
            self.kernel(compile_key),
            gathered,
            out,
        )

    def __call__(self, gathered: torch.Tensor, out: torch.Tensor, *, topk: int) -> Any:
        compile_key = self.dispatch(topk=topk, num_candidates=gathered.shape[1])
        self._guard_warmup_call(compile_key)
        cache_key = (compile_key.topk, compile_key.num_candidates)
        compiled = self._get_compiled_from_cache(
            compile_key,
            cache_key=cache_key,
            runtime_context={
                "gathered_shape": tuple(gathered.shape),
                "out_shape": tuple(out.shape),
                "topk": topk,
            },
        )
        return compiled(gathered, out)


_PACK_DCP_TOPK_CANDIDATES_KERNEL = PackDCPTopkCandidatesKernel()
_STABLE_TOPK_FROM_GATHERED_CANDIDATES_KERNEL = StableTopKFromGatheredCandidatesKernel()
