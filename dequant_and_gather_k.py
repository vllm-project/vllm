# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# once we have more CuteDSL kernels in vLLM, we can refactor small helper functions
# to a separate file
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.compile_utils import make_fake_tensor


def dequant_and_gather_k_cutedsl(
    out: torch.Tensor,  # [num_reqs, max_num_tokens, head_size]
    k_cache: torch.Tensor,  # [num_blocks, block_size, head_bytes]
    seq_lens: torch.Tensor,  # [num_reqs]
    gather_lens: torch.Tensor | None,  # [num_reqs]
    block_table: torch.Tensor,  # [num_reqs, max_blocks_per_seq]
    block_size: int,
    offset: int,
) -> None:
    _, block_size, _ = k_cache.shape
    DequantGatherKCacheKernel.compile(block_size=block_size)(
        out, k_cache, seq_lens, gather_lens, block_table, offset
    )


@dsl_user_op
def _fp32x2_to_bf16x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint32(out)


@dsl_user_op
def _bf16x2_to_fp32(data: Uint32, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [data.ir_value(loc=loc, ip=ip)],
        "shl.b32 $0, $2, 16;\n\tand.b32 $1, $2, 0xFFFF0000;\n",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _bf16x2_abs(a: Uint32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip)],
        "abs.bf16x2 $0, $1;",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint32(out)


@dsl_user_op
def _bf16x2_max(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "max.bf16x2 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint32(out)


@dsl_user_op
def _fp32x8_to_fp4x8(
    vals: cute.Tensor,
    offset: cutlass.Constexpr[int],
    *,
    loc=None,
    ip=None,
) -> Uint32:
    # Pack eight scaled FP32 values into four E2M1x2 bytes, returned as one b32.
    assert vals.element_type is Float32
    out = llvm.inline_asm(
        T.i32(),
        [vals[offset + i].ir_value(loc=loc, ip=ip) for i in range(8)],
        "{\n\t"
        ".reg .b8 x0, x1, x2, x3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 x0, $2, $1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 x1, $4, $3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 x2, $6, $5;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 x3, $8, $7;\n\t"
        "mov.b32 $0, {x0, x1, x2, x3};\n\t"
        "}\n",
        "=r,f,f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint32(out)


@dsl_user_op
def _fp8x4_to_bf16x4(x: Uint32, *, loc=None, ip=None) -> cute.TensorSSA:
    # there is only fp8->fp16, no fp8->bf16,
    # so we have this monster here
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [x.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 x0, x1;\n\t"
        ".reg .b16 t00, t01, t10, t11;\n\t"
        "mov.b32 {x0, x1}, $2;\n\t"
        "cvt.rn.f16x2.e4m3x2 $0, x0;\n\t"
        "cvt.rn.f16x2.e4m3x2 $1, x1;\n\t"
        "mov.b32 {t00, t01}, $0;\n\t"
        "mov.b32 {t10, t11}, $1;\n\t"
        "cvt.rn.bf16.f16 t00, t00;\n\t"
        "cvt.rn.bf16.f16 t01, t01;\n\t"
        "cvt.rn.bf16.f16 t10, t10;\n\t"
        "cvt.rn.bf16.f16 t11, t11;\n\t"
        "mov.b32 $0, {t00, t01};\n\t"
        "mov.b32 $1, {t10, t11};\n\t"
        "}\n",
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    vec = vector.from_elements(
        ir.VectorType.get([2], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip) for i in range(2)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 2, Uint32)


@dsl_user_op
def _bf16x2_mul(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "mul.rn.bf16x2 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint32(out)


# Custom vectorized load to support cache modifiers. For some reason,
# cute.autovec_copy() does not currently emit the requested modifiers.
# tensor and coord is only used to select the base pointer. actual load
# is done using out_dtype
@dsl_user_op
def _ldg_vec(
    tensor: cute.Tensor,
    coord: cute.Coord,
    vec_size: cutlass.Constexpr[int],
    modifier: cutlass.Constexpr[str] = "",
    ld_type: cutlass.Constexpr[type[cutlass.Numeric] | None] = None,
    *,
    loc=None,
    ip=None,
) -> cute.TensorSSA:
    if ld_type is None:
        ld_type = tensor.element_type
    if const_expr(ld_type is Float32):
        ptx_ty = "f32"
        constraint = "=f"
    elif const_expr(ld_type is Uint32):
        ptx_ty = "u32"
        constraint = "=r"
    else:
        raise TypeError(f"_ldg_vec only supports Uint32 and Float32, got {ld_type}")

    # compute base pointer
    base_ptr = (
        tensor.iterator + cute.crd2idx(coord, tensor.layout, loc=loc, ip=ip)
    ).toint()

    # build PTX string
    ptx_str = f"ld.global{modifier}.v{vec_size}.{ptx_ty}"
    ptx_str += "{" + ", ".join(f"${i}" for i in range(vec_size)) + "}"
    ptx_str += f", [${vec_size}];"

    out = llvm.inline_asm(
        llvm.StructType.get_literal([ld_type.mlir_type] * vec_size),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        ptx_str,
        ",".join([constraint] * vec_size + ["l"]),
        has_side_effects=False,
        is_align_stack=False,
    )
    vec = vector.from_elements(
        ir.VectorType.get([vec_size], ld_type.mlir_type, loc=loc),
        [
            llvm.extractvalue(ld_type.mlir_type, out, [i], loc=loc, ip=ip)
            for i in range(vec_size)
        ],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, vec_size, ld_type)


@dsl_user_op
def _stg_vec(
    tensor: cute.Tensor,
    coord: cute.Coord,
    values: cute.Tensor,
    vec_size: cutlass.Constexpr[int],
    modifier: cutlass.Constexpr[str] = "",
    *,
    loc=None,
    ip=None,
) -> None:
    # NOTE: st_type is derived from values tensor
    st_type = values.element_type
    if const_expr(st_type is Float32):
        ptx_ty = "f32"
        constraint = "f"
    elif const_expr(st_type is Uint32):
        ptx_ty = "u32"
        constraint = "r"
    else:
        raise TypeError(f"_stg_vec only supports Uint32 and Float32, got {st_type}")

    # compute base pointer
    base_ptr = (
        tensor.iterator + cute.crd2idx(coord, tensor.layout, loc=loc, ip=ip)
    ).toint()

    # build PTX string
    ptx_str = f"st.global{modifier}.v{vec_size}.{ptx_ty} [$0], "
    ptx_str += "{" + ", ".join(f"${i + 1}" for i in range(vec_size)) + "};"

    llvm.inline_asm(
        None,
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)]
        + [values[i].ir_value(loc=loc, ip=ip) for i in range(vec_size)],
        ptx_str,
        ",".join(["l"] + [constraint] * vec_size),
        has_side_effects=True,
        is_align_stack=False,
    )


class DequantGatherKCacheKernel:
    # hard-coded for DSv4
    head_dim = 512
    group_size = 64  # 1 scale per 64 elems

    def __init__(self, fp8_dim: int = 448, block_size: int = 64):
        self.fp8_dim = fp8_dim
        self.bf16_dim = self.head_dim - fp8_dim
        self.block_size = block_size

        self.num_warps = 4
        self.tb_size = self.num_warps * 32

    @cute.jit
    def __call__(
        self,
        out: cute.Tensor,  # [num_reqs, max_num_tokens, head_size]
        k_cache: cute.Tensor,  # [num_blocks, block_size, head_bytes]
        seq_lens: cute.Tensor,  # [num_reqs]
        gather_lens: cute.Tensor | None,  # [num_reqs]
        block_table: cute.Tensor,  # [num_reqs, max_blocks_per_req]
        offset: Int32,
        stream: CUstream,
    ):
        grid = (out.shape[0], 512, 1)
        self.kernel(
            out,
            k_cache,
            seq_lens,
            gather_lens,
            block_table,
            offset,
        ).launch(grid=grid, block=(self.tb_size, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        out: cute.Tensor,  # [num_reqs, max_num_tokens, head_size]
        k_cache: cute.Tensor,  # [num_blocks, block_size, head_bytes]
        seq_lens: cute.Tensor,  # [num_reqs]
        gather_lens: cute.Tensor | None,  # [num_reqs]
        block_table: cute.Tensor,  # [num_reqs, max_blocks_per_req]
        offset: Int32,
    ):
        req_id, worker_id, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        _, num_workers, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32

        # split k_cache into k_data and k_scale
        # each [block_size, head_bytes] block is actually a concat of
        # [block_size, fp8_dim + bf16_dim * 2] and [block_size, 8]
        data_dim = cutlass.const_expr(self.fp8_dim + self.bf16_dim * 2)
        k_data = cute.make_tensor(
            k_cache.iterator,
            layout=cute.make_layout(
                (k_cache.shape[0], self.block_size, data_dim),
                stride=(k_cache.stride[0], data_dim, 1),
            ),
        )
        k_scale = cute.make_tensor(
            k_cache.iterator + (self.block_size * data_dim),
            layout=cute.make_layout(
                (k_cache.shape[0], self.block_size, 8),
                stride=(k_cache.stride[0], 8, 1),
            ),
        )

        seq_len = seq_lens[req_id]
        gather_len = seq_len
        if cutlass.const_expr(gather_lens is not None):
            gather_len = gather_lens[req_id]
        start_pos = seq_len - gather_len

        # at each position, we have 448 FP8 values + 64 BF16 values.
        # to make our lives easier, we will use 16B loads for FP8,
        # then 32B stores for the dequantized BF16.
        # hence 1 warp (32 threads) can cover 1 position/token exactly.
        #
        # the first 28 threads do 16B loads = 448 FP8 values.
        # the last 4 threads do 32B loads = 64 BF16 values.
        # then the whole warp do 32B stores = 512 BF16 values.

        for i in range(
            worker_id * self.num_warps + warp_id,
            gather_len,
            num_workers * self.num_warps,
        ):
            pos = start_pos + i
            page_id = block_table[req_id, pos // self.block_size]

            # we don't do bounds check here to avoid warp divergence.
            # the last 4 threads will load and dequantize to garbage.
            k_block_offset = pos % self.block_size
            coord = (page_id, k_block_offset, lane_id * 16)
            data = _ldg_vec(k_data, coord, 4, "", Uint32)
            scale = k_scale[page_id, k_block_offset, lane_id * 16 // self.group_size]

            # convert to bf16x2 via bit manipulation
            scale_u32 = Uint32(scale)
            scale_bf16x2 = (scale_u32 << Uint32(23)) | (scale_u32 << Uint32(7))

            # cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 requires PTX 9.2 (CUDA 13.2)
            dequant = cute.make_rmem_tensor(8, Uint32)
            for j in cutlass.range_constexpr(4):
                tmp = _fp8x4_to_bf16x4(data[j])

                # bf16 multiply is safe
                dequant[j * 2] = _bf16x2_mul(tmp[0], scale_bf16x2)
                dequant[j * 2 + 1] = _bf16x2_mul(tmp[1], scale_bf16x2)

            # the last 4 threads load BF16 data
            if lane_id * 16 >= self.fp8_dim:
                coord = (page_id, k_block_offset, lane_id * 32 - self.fp8_dim)
                dequant.store(_ldg_vec(k_data, coord, 8, "", Uint32))

            coord = (req_id, offset + i, lane_id * 16)
            _stg_vec(out, coord, dequant, 8)

    @cache
    @staticmethod
    def compile(fp8_dim: int = 448, block_size: int = 64):
        num_reqs = cute.sym_int()
        head_dim = DequantGatherKCacheKernel.head_dim
        head_bytes = fp8_dim + (head_dim - fp8_dim) * 2 + 8

        out = make_fake_tensor(BFloat16, (num_reqs, cute.sym_int(), head_dim), 16)
        k_cache = make_fake_tensor(Uint8, (cute.sym_int(), block_size, head_bytes))
        seq_lens = make_fake_tensor(Int32, (num_reqs,))
        gather_lens = make_fake_tensor(Int32, (num_reqs,))
        block_table = make_fake_tensor(Int32, (num_reqs, cute.sym_int()))

        kernel = DequantGatherKCacheKernel(fp8_dim, block_size)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            out,
            k_cache,
            seq_lens,
            gather_lens,
            block_table,
            Int32(0),
            stream,
            options="--enable-tvm-ffi",
        )
