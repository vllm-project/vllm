# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/jit/csrc/gemm/tiny_gemm.cuh
"""Kimi-K3 TP8 skinny GEMMs for the KDA F_A/beta and F_B projections."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import T, dsl_user_op

_FAB_N = 144
_FAB_K = 7168
_FAB_VECTOR_WIDTH = 16
_FAB_BLOCK_SIZE = _FAB_K // _FAB_VECTOR_WIDTH
_FAB_NUM_WARPS = _FAB_BLOCK_SIZE // cute.arch.WARP_SIZE

_FB_N = 1536
_FB_K = 128
_FB_VECTOR_WIDTH = 8
_FB_K_LANES = _FB_K // _FB_VECTOR_WIDTH

SKINNY_N_SPLIT_N = {
    2: 1,
    3: 2,
    4: 3,
    5: 3,
    6: 3,
    7: 3,
    8: 3,
    9: 2,
    10: 2,
    11: 2,
    12: 2,
    13: 2,
    14: 1,
    15: 1,
    16: 1,
}

SKINNY_K_SPLIT_N = {
    2: 8,
    3: 16,
    4: 16,
    5: 24,
    6: 24,
    7: 24,
    8: 24,
    9: 12,
    10: 16,
    11: 16,
    12: 12,
    13: 8,
    14: 8,
    15: 8,
    16: 8,
}


@dsl_user_op
def _fma_f32_bf16(
    a: BFloat16,
    b: BFloat16,
    acc: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    a_bits = llvm.bitcast(T.i16(), a.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    b_bits = llvm.bitcast(T.i16(), b.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    result = llvm.inline_asm(
        T.f32(),
        [a_bits, b_bits, acc.ir_value(loc=loc, ip=ip)],
        "fma.rn.f32.bf16 $0, $1, $2, $3;",
        "=f,h,h,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Float32(result)


class _KdaSkinnyNGemm:
    def __init__(self, num_rows: int, split_n: int) -> None:
        if _FAB_N % split_n:
            raise ValueError("split_n must divide the FAB output size")
        if num_rows * split_n > _FAB_BLOCK_SIZE:
            raise ValueError("the FAB output tile must fit in one block")
        self.num_rows = num_rows
        self.split_n = split_n

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        copy_a = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            BFloat16,
            num_bits_per_copy=_FAB_VECTOR_WIDTH * BFloat16.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        copy_weight = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            BFloat16,
            num_bits_per_copy=_FAB_VECTOR_WIDTH * BFloat16.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        self.kernel(a, weight, output, copy_a, copy_weight).launch(
            grid=[_FAB_N // self.split_n, 1, 1],
            block=[_FAB_BLOCK_SIZE, 1, 1],
            smem=_FAB_NUM_WARPS * self.num_rows * self.split_n * 4,
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        a: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        copy_a: cute.CopyAtom,
        copy_weight: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()
        num_rows: cutlass.Constexpr = self.num_rows
        split_n: cutlass.Constexpr = self.split_n

        a_vectors = cute.logical_divide(a, (None, _FAB_VECTOR_WIDTH))
        weight_vectors = cute.logical_divide(weight, (None, _FAB_VECTOR_WIDTH))
        a_tiles = cute.logical_divide(
            a_vectors,
            (None, (None, _FAB_BLOCK_SIZE)),
        )
        weight_tiles = cute.logical_divide(
            weight_vectors,
            (None, (None, _FAB_BLOCK_SIZE)),
        )
        a_tile = a_tiles[None, (None, (tidx, None))]
        n_base = block_idx * split_n

        a_regs = cute.make_rmem_tensor(
            cute.make_layout(
                (num_rows, _FAB_VECTOR_WIDTH),
                stride=(_FAB_VECTOR_WIDTH, 1),
            ),
            BFloat16,
        )
        weight_regs = cute.make_rmem_tensor(
            cute.make_layout(
                (split_n, _FAB_VECTOR_WIDTH),
                stride=(_FAB_VECTOR_WIDTH, 1),
            ),
            BFloat16,
        )
        acc = cute.make_rmem_tensor(
            cute.make_layout((num_rows, split_n), stride=(split_n, 1)),
            Float32,
        )
        acc.fill(0.0)

        for ni in cutlass.range_constexpr(split_n):
            weight_tile = weight_tiles[
                n_base + ni,
                (None, (tidx, None)),
            ]
            cute.copy(copy_weight, weight_tile[None, 0], weight_regs[ni, None])

        cute.arch.griddepcontrol_wait()

        for mi in cutlass.range_constexpr(num_rows):
            cute.copy(copy_a, a_tile[mi, None, 0], a_regs[mi, None])
        for mi in cutlass.range_constexpr(num_rows):
            for ni in cutlass.range_constexpr(split_n):
                for vi in cutlass.range_constexpr(_FAB_VECTOR_WIDTH):
                    acc[mi, ni] = _fma_f32_bf16(
                        a_regs[mi, vi],
                        weight_regs[ni, vi],
                        acc[mi, ni],
                    )
                acc[mi, ni] = cute.arch.warp_reduction_sum(acc[mi, ni])

        partials_layout = cute.make_layout(
            (_FAB_NUM_WARPS, num_rows, split_n),
            stride=(num_rows * split_n, split_n, 1),
        )
        smem = cutlass.utils.SmemAllocator()
        partials = smem.allocate_tensor(
            Float32,
            partials_layout,
            byte_alignment=16,
        )
        with cute.arch.elect_one():
            for mi in cutlass.range_constexpr(num_rows):
                for ni in cutlass.range_constexpr(split_n):
                    partials[warp_idx, mi, ni] = acc[mi, ni]

        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.sync_threads()

        if tidx < num_rows * split_n:
            mi = tidx // split_n
            ni = tidx % split_n
            total = (
                partials[None, mi, ni]
                .load()
                .reduce(
                    cute.ReductionOp.ADD,
                    init_val=Float32(0.0),
                    reduction_profile=0,
                )
            )
            output[mi, n_base + ni] = Float32(total).to(BFloat16)


class _KdaSkinnyKGemm:
    def __init__(self, num_rows: int, split_n: int) -> None:
        if _FB_N % split_n:
            raise ValueError("split_n must divide the FB output size")
        if split_n * _FB_K_LANES > 1024:
            raise ValueError("the FB output tile exceeds the block-size limit")
        self.num_rows = num_rows
        self.split_n = split_n

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            BFloat16,
            num_bits_per_copy=_FB_VECTOR_WIDTH * BFloat16.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        self.kernel(a, weight, output, copy_atom).launch(
            grid=[_FB_N // self.split_n, 1, 1],
            block=[self.split_n * _FB_K_LANES, 1, 1],
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        a: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        copy_atom: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        num_rows: cutlass.Constexpr = self.num_rows
        split_n: cutlass.Constexpr = self.split_n
        k_lane = tidx % _FB_K_LANES
        n_idx = block_idx * split_n + tidx // _FB_K_LANES

        a_vectors = cute.logical_divide(a, (None, _FB_VECTOR_WIDTH))
        weight_vectors = cute.logical_divide(weight, (None, _FB_VECTOR_WIDTH))
        a_tiles = cute.logical_divide(
            a_vectors,
            (None, (None, _FB_K_LANES)),
        )
        weight_tiles = cute.logical_divide(
            weight_vectors,
            (None, (None, _FB_K_LANES)),
        )
        a_tile = a_tiles[None, (None, (k_lane, None))]
        weight_tile = weight_tiles[n_idx, (None, (k_lane, None))]
        weight_regs = cute.make_rmem_tensor(_FB_VECTOR_WIDTH, BFloat16)
        input_regs = cute.make_rmem_tensor(
            cute.make_layout(
                (num_rows, _FB_VECTOR_WIDTH),
                stride=(_FB_VECTOR_WIDTH, 1),
            ),
            BFloat16,
        )
        cute.copy(
            copy_atom,
            weight_tile[None, 0],
            weight_regs,
        )

        cute.arch.griddepcontrol_wait()

        for mi in cutlass.range_constexpr(num_rows):
            cute.copy(
                copy_atom,
                a_tile[mi, None, 0],
                input_regs[mi, None],
            )
        for mi in cutlass.range_constexpr(num_rows):
            acc = Float32(0.0)
            for vi in cutlass.range_constexpr(_FB_VECTOR_WIDTH):
                acc = _fma_f32_bf16(
                    input_regs[mi, vi],
                    weight_regs[vi],
                    acc,
                )
            for offset in (8, 4, 2, 1):
                acc += cute.arch.shuffle_sync_bfly(acc, offset)
            output[mi, n_idx] = acc.to(BFloat16)

        cute.arch.griddepcontrol_launch_dependents()


class KdaSkinnyGemm:
    def __init__(self) -> None:
        self._compiled_n: dict[tuple[int, int], Callable[..., None]] = {}
        self._compiled_k: dict[tuple[int, int], Callable[..., None]] = {}
        self._warmup_n: set[int] = set()
        self._warmup_k: set[int] = set()
        self._warmup_registered = False

    def _compile_n(self, num_rows: int) -> None:
        device = torch.accelerator.current_device_index()
        key = (device, num_rows)
        if key in self._compiled_n:
            return
        split_n = SKINNY_N_SPLIT_N[num_rows]
        self._compiled_n[key] = cute.compile(
            _KdaSkinnyNGemm(num_rows, split_n),
            make_fake_tensor(
                BFloat16,
                (num_rows, _FAB_K),
                stride=(_FAB_K, 1),
                assumed_align=32,
            ),
            make_fake_tensor(
                BFloat16,
                (_FAB_N, _FAB_K),
                stride=(_FAB_K, 1),
                assumed_align=32,
            ),
            make_fake_tensor(
                BFloat16,
                (num_rows, _FAB_N),
                stride=(_FAB_N, 1),
                assumed_align=32,
            ),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi --ptxas-options -maxrregcount=255",
        )

    def _compile_k(self, num_rows: int) -> None:
        device = torch.accelerator.current_device_index()
        key = (device, num_rows)
        if key in self._compiled_k:
            return
        split_n = SKINNY_K_SPLIT_N[num_rows]
        self._compiled_k[key] = cute.compile(
            _KdaSkinnyKGemm(num_rows, split_n),
            make_fake_tensor(
                BFloat16,
                (num_rows, _FB_K),
                stride=(_FAB_N, 1),
                assumed_align=32,
            ),
            make_fake_tensor(
                BFloat16,
                (_FB_N, _FB_K),
                stride=(_FB_K, 1),
                assumed_align=32,
            ),
            make_fake_tensor(
                BFloat16,
                (num_rows, _FB_N),
                stride=(_FB_N, 1),
                assumed_align=32,
            ),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi --ptxas-options -maxrregcount=255",
        )

    def request_warmup(self, n_tokens: set[int], k_tokens: set[int]) -> None:
        if not n_tokens <= SKINNY_N_SPLIT_N.keys():
            raise ValueError("unsupported skinny-N warmup token count")
        if not k_tokens <= SKINNY_K_SPLIT_N.keys():
            raise ValueError("unsupported skinny-K warmup token count")
        self._warmup_n.update(n_tokens)
        self._warmup_k.update(k_tokens)
        if self._warmup_registered:
            return
        from vllm.model_executor.warmup.cutedsl_warmup import (
            register_cutedsl_warmup_provider,
        )

        register_cutedsl_warmup_provider(self)
        self._warmup_registered = True

    def get_cutedsl_warmup_compile_units(self):
        from vllm.model_executor.warmup.cutedsl_warmup import CuTeDSLCompileUnit

        n_units = tuple(
            CuTeDSLCompileUnit(
                name="Kimi-K3 KDA skinny-N GEMM",
                key=("kimi-k3-kda-skinny-n", num_rows),
                compile=partial(self._compile_n, num_rows),
            )
            for num_rows in sorted(self._warmup_n)
        )
        k_units = tuple(
            CuTeDSLCompileUnit(
                name="Kimi-K3 KDA skinny-K GEMM",
                key=("kimi-k3-kda-skinny-k", num_rows),
                compile=partial(self._compile_k, num_rows),
            )
            for num_rows in sorted(self._warmup_k)
        )
        return n_units + k_units

    @staticmethod
    def _validate(
        a: torch.Tensor,
        weight: torch.Tensor,
        *,
        a_shape: tuple[int, int],
        weight_shape: tuple[int, int],
    ) -> None:
        if a.shape != a_shape or weight.shape != weight_shape:
            raise ValueError(
                f"expected input {a_shape} and weight {weight_shape}, got "
                f"{tuple(a.shape)} and {tuple(weight.shape)}"
            )
        if a.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise ValueError("KDA skinny GEMMs require BF16 input and weight")
        if not a.is_cuda or not weight.is_cuda or a.device != weight.device:
            raise ValueError("KDA skinny GEMM operands must share a CUDA device")
        if not a.is_contiguous() or not weight.is_contiguous():
            raise ValueError("KDA skinny GEMM operands must be contiguous")
        if a.data_ptr() % 32 or weight.data_ptr() % 32:
            raise ValueError("KDA skinny GEMM operands must be 32-byte aligned")

    def run_n(self, a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        num_rows = a.shape[0]
        if num_rows not in SKINNY_N_SPLIT_N:
            raise ValueError("skinny-N requires M in [2, 16]")
        self._validate(
            a,
            weight,
            a_shape=(num_rows, _FAB_K),
            weight_shape=(_FAB_N, _FAB_K),
        )
        output = torch.empty(
            (num_rows, _FAB_N),
            dtype=a.dtype,
            device=a.device,
        )
        with torch.accelerator.device_index(a.device.index):
            self._compile_n(num_rows)
            key = (a.device.index, num_rows)
            self._compiled_n[key](
                a,
                weight,
                output,
            )
        return output

    def run_k(self, packed_fab: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        num_rows = packed_fab.shape[0]
        if num_rows not in SKINNY_K_SPLIT_N:
            raise ValueError("skinny-K requires M in [2, 16]")
        self._validate(
            packed_fab,
            weight,
            a_shape=(num_rows, _FAB_N),
            weight_shape=(_FB_N, _FB_K),
        )
        output = torch.empty(
            (num_rows, _FB_N),
            dtype=packed_fab.dtype,
            device=packed_fab.device,
        )
        with torch.accelerator.device_index(packed_fab.device.index):
            self._compile_k(num_rows)
            key = (packed_fab.device.index, num_rows)
            self._compiled_k[key](
                packed_fab[:, :_FB_K],
                weight,
                output,
            )
        return output


kda_skinny_gemm = KdaSkinnyGemm()
