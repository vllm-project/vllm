# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe-DSL `redux.sync` + PDL top-6 router for DeepSeek-V4.

Uses a 4-warp (128-thread) two-stage `warp_redux_sync` kernel when the token
count is below the GPU's warp-scheduler saturation point
(`M <= 4 * sm_count`) to maximize per-token parallelism, and a packed
1-warp-per-token (4 tokens/CTA, zero-barrier, SMEM-staged score) kernel for
larger prefill/mixed batches to reach 100% SM warp occupancy.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.platforms import current_platform

_TOPK = 6
_WARP_SCHEDULERS_PER_SM = 4
_SINGLE_WARP_TOKENS_PER_CTA = 4
_SUPPORTED_NUM_EXPERTS = (256, 384)
_SmemAllocator = getattr(cutlass, "memory", cutlass.utils).SmemAllocator


@cache
def _device_supports_pdl(device_index: int) -> bool:
    return current_platform.has_device_capability(90, device_id=device_index)


@cache
def _multi_warp_max_tokens(device_index: int) -> int:
    """Switch from 4-warp to 1-warp once 1-warp CTAs saturate SM schedulers."""
    sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
    return _WARP_SCHEDULERS_PER_SM * sm_count


def _asm(
    result,
    operands: list,
    text: str,
    constraints: str,
    side_effects: bool,
    loc,
    ip,
):
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _f32(val, loc, ip):
    return Float32(val).ir_value(loc=loc, ip=ip)


@dsl_user_op
def _load_f32_nc(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.f32 $0, [$1];",
            "=f,l",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _load_s32_nc(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.s32 $0, [$1];",
            "=r,l",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _load_s64_nc(addr: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(
        _asm(
            T.i64(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.s64 $0, [$1];",
            "=l,l",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _load_f32_ordered(addr: Int64, *, loc=None, ip=None) -> Float32:
    # Coherent, side-effecting load so ptxas cannot hoist PDL-dependent logits
    # across griddepcontrol_wait() or read stale non-coherent cache lines.
    return Float32(
        _asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _load_u8(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "{ .reg .u16 t; ld.global.u8 t, [$1]; cvt.u32.u16 $0, t; }",
            "=r,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _store_f32(addr: Int64, val: Float32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), _f32(val, loc, ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _store_s32(addr: Int64, val: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.global.s32 [$0], $1;",
        "l,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _store_s64(addr: Int64, val: Int64, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int64(val).ir_value(loc=loc, ip=ip),
        ],
        "st.global.s64 [$0], $1;",
        "l,l",
        True,
        loc,
        ip,
    )


# Match Triton's _dsv4_topk_kernel arithmetic instruction-for-instruction:
# exp(x) = ex2.approx.f32(x * log2(e)), sqrt = sqrt.approx.ftz.f32,
# NaN ranked as -1e30 (0xF149F2CA), -0.0 canonicalized to +0.0, and
# float32 mapped to an order-preserving uint32 key (> 0 for all valid floats).
_SOFTPLUS_ARG_PTX = (
    "{\n"
    ".reg .f32 a;\n"
    "mul.f32 a, $1, 0f3FB8AA3B;\n"
    "ex2.approx.f32 a, a;\n"
    "add.f32 $0, a, 0f3F800000;\n"
    "}"
)

_SCORE_KEY_PTX = (
    "{\n"
    ".reg .pred p;\n"
    ".reg .f32 l, s, v;\n"
    ".reg .b32 b, n;\n"
    "setp.gt.f32 p, $2, 0f41A00000;\n"
    "selp.f32 l, $2, $3, p;\n"
    "sqrt.approx.ftz.f32 s, l;\n"
    "add.f32 v, s, $4;\n"
    "setp.num.f32 p, v, v;\n"
    "selp.f32 v, v, 0fF149F2CA, p;\n"
    "setp.eq.f32 p, v, 0f00000000;\n"
    "selp.f32 v, 0f00000000, v, p;\n"
    "mov.b32 b, v;\n"
    "setp.lt.s32 p, b, 0;\n"
    "not.b32 n, b;\n"
    "or.b32 b, b, 0x80000000;\n"
    "selp.b32 $1, n, b, p;\n"
    "mov.f32 $0, s;\n"
    "}"
)


@dsl_user_op
def _score_and_key(logit, bias, *, loc=None, ip=None):
    arg = Float32(
        _asm(
            T.f32(),
            [_f32(logit, loc, ip)],
            _SOFTPLUS_ARG_PTX,
            "=f,f",
            False,
            loc,
            ip,
        )
    )
    log_arg = cute.math.log(arg, fastmath=False)
    res = _asm(
        llvm.StructType.get_literal([T.f32(), T.i32()]),
        [_f32(logit, loc, ip), _f32(log_arg, loc, ip), _f32(bias, loc, ip)],
        _SCORE_KEY_PTX,
        "=f,=r,f,f,f",
        False,
        loc,
        ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _add_f32(a, b, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_f32(a, loc, ip), _f32(b, loc, ip)],
            "add.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _normalized_weight(scale, total, weight, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_f32(scale, loc, ip), _f32(total, loc, ip), _f32(weight, loc, ip)],
            "{\n"
            ".reg .pred p;\n"
            ".reg .f32 d;\n"
            "setp.gt.f32 p, $2, 0f00000000;\n"
            "selp.f32 d, $2, 0f3F800000, p;\n"
            "div.full.f32 d, $1, d;\n"
            "mul.f32 $0, d, $3;\n"
            "}",
            "=f,f,f,f",
            False,
            loc,
            ip,
        )
    )


class _Dsv4TopkCutlass:
    """`redux.sync` + PDL top-6 router kernel (E in {256, 384}, 1 or 4 warps)."""

    def __init__(
        self,
        num_experts: int,
        num_warps: int,
        has_padding: bool,
        has_vl: bool,
        input_ids_is_int64: bool,
        is_int64: bool,
        use_pdl: bool,
    ) -> None:
        self.num_experts = num_experts
        self.num_warps = num_warps
        self.threads = num_warps * 32
        self.experts_per_thread = num_experts // self.threads
        self.num_merge_candidates = num_warps * _TOPK
        self.single_warp_threads = _SINGLE_WARP_TOKENS_PER_CTA * 32
        self.single_warp_smem_elems = _SINGLE_WARP_TOKENS_PER_CTA * num_experts
        self.has_padding = has_padding
        self.has_vl = has_vl
        self.input_ids_is_int64 = input_ids_is_int64
        self.is_int64 = is_int64
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        gating_output: Int64,
        correction_bias: Int64,
        is_padding: Int64,
        topk_weights: Int64,
        topk_ids: Int64,
        input_ids: Int64,
        bias_vl: Int64,
        num_tokens: Int32,
        grid_x: Int32,
        image_sentinel_lo: Int32,
        routed_scaling_factor: Float32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.num_warps == 1):

            @cute.struct
            class ScoreStorage:
                scores: cute.struct.MemRange[Float32, self.single_warp_smem_elems]

            self.single_warp_kernel(
                gating_output,
                correction_bias,
                is_padding,
                topk_weights,
                topk_ids,
                input_ids,
                bias_vl,
                num_tokens,
                image_sentinel_lo,
                routed_scaling_factor,
                ScoreStorage,
            ).launch(
                grid=(grid_x, 1, 1),
                block=[self.single_warp_threads, 1, 1],
                stream=stream,
                use_pdl=self.use_pdl,
            )
        else:

            @cute.struct
            class Storage:
                merge_keys: cute.struct.MemRange[Uint32, self.num_merge_candidates]
                merge_ids: cute.struct.MemRange[Int32, self.num_merge_candidates]
                merge_scores: cute.struct.MemRange[Float32, self.num_merge_candidates]

            self.multi_warp_kernel(
                gating_output,
                correction_bias,
                is_padding,
                topk_weights,
                topk_ids,
                input_ids,
                bias_vl,
                image_sentinel_lo,
                routed_scaling_factor,
                Storage,
            ).launch(
                grid=(grid_x, 1, 1),
                block=[self.threads, 1, 1],
                stream=stream,
                use_pdl=self.use_pdl,
            )

    @cute.jit
    def _select_bias_ptr(
        self,
        correction_bias: Int64,
        input_ids: Int64,
        bias_vl: Int64,
        row: Int32,
        image_sentinel_lo: Int32,
    ) -> Int64:
        bias_ptr = correction_bias
        if cutlass.const_expr(self.has_vl):
            if cutlass.const_expr(self.input_ids_is_int64):
                tok = _load_s64_nc(input_ids + Int64(row) * 8)
            else:
                tok = Int64(_load_s32_nc(input_ids + Int64(row) * 4))
            lo = Int64(image_sentinel_lo)
            if tok >= lo and tok < lo + 5:
                bias_ptr = bias_vl
        return bias_ptr

    @cute.jit
    def _find_best_local(self, key: cute.Tensor):
        best_key = key[0]
        best_local_idx = Int32(0)
        for j in cutlass.range_constexpr(1, self.experts_per_thread):
            if key[j] > best_key:
                best_key = key[j]
                best_local_idx = Int32(j)
        return best_key, best_local_idx

    @cute.jit
    def _normalize_and_store(
        self,
        slot_weight: Float32,
        slot_id: Int32,
        routed_scaling_factor: Float32,
        is_padding: Int64,
        topk_weights: Int64,
        topk_ids: Int64,
        row: Int32,
        lane: Int32,
    ) -> None:
        total = slot_weight
        for shift in cutlass.range_constexpr(2, -1, -1):
            total = _add_f32(total, cute.arch.shuffle_sync_bfly(total, 1 << shift))

        if lane < _TOPK:
            final_w = _normalized_weight(routed_scaling_factor, total, slot_weight)
            final_id = slot_id
            if cutlass.const_expr(self.has_padding):  # noqa: SIM102
                if _load_u8(is_padding + Int64(row)) != 0:
                    final_w = Float32(0.0)
                    final_id = Int32(-1)
            out_idx = Int64(row) * _TOPK + lane
            _store_f32(topk_weights + out_idx * 4, final_w)
            if cutlass.const_expr(self.is_int64):
                _store_s64(topk_ids + out_idx * 8, Int64(final_id))
            else:
                _store_s32(topk_ids + out_idx * 4, final_id)

    @cute.kernel
    def single_warp_kernel(
        self,
        gating_output: Int64,
        correction_bias: Int64,
        is_padding: Int64,
        topk_weights: Int64,
        topk_ids: Int64,
        input_ids: Int64,
        bias_vl: Int64,
        num_tokens: Int32,
        image_sentinel_lo: Int32,
        routed_scaling_factor: Float32,
        ScoreStorage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        warp = tid // 32
        lane = tid % 32
        row = bx * _SINGLE_WARP_TOKENS_PER_CTA + warp
        experts_per_thread = self.experts_per_thread

        smem = _SmemAllocator()
        storage = smem.allocate(ScoreStorage)
        smem_scores = storage.scores.get_tensor(
            cute.make_layout(self.single_warp_smem_elems)
        )
        warp_offset = warp * self.num_experts

        bias_ptr = correction_bias
        if cutlass.const_expr(self.has_vl):  # noqa: SIM102
            if row < num_tokens:
                bias_ptr = self._select_bias_ptr(
                    correction_bias, input_ids, bias_vl, row, image_sentinel_lo
                )

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        slot_weight = Float32(0.0)
        slot_id = Int32(0)
        if row < num_tokens:
            row_base = gating_output + Int64(row) * (self.num_experts * 4)
            key = cute.make_rmem_tensor(experts_per_thread, Uint32)
            for chunk in cutlass.range_constexpr(experts_per_thread // 4):
                j0 = chunk * 4
                b0 = _load_f32_nc(bias_ptr + Int64((j0 + 0) * 32 + lane) * 4)
                b1 = _load_f32_nc(bias_ptr + Int64((j0 + 1) * 32 + lane) * 4)
                b2 = _load_f32_nc(bias_ptr + Int64((j0 + 2) * 32 + lane) * 4)
                b3 = _load_f32_nc(bias_ptr + Int64((j0 + 3) * 32 + lane) * 4)
                x0 = _load_f32_ordered(row_base + Int64((j0 + 0) * 32 + lane) * 4)
                x1 = _load_f32_ordered(row_base + Int64((j0 + 1) * 32 + lane) * 4)
                x2 = _load_f32_ordered(row_base + Int64((j0 + 2) * 32 + lane) * 4)
                x3 = _load_f32_ordered(row_base + Int64((j0 + 3) * 32 + lane) * 4)
                s0, k0 = _score_and_key(x0, b0)
                s1, k1 = _score_and_key(x1, b1)
                s2, k2 = _score_and_key(x2, b2)
                s3, k3 = _score_and_key(x3, b3)
                smem_scores[warp_offset + (j0 + 0) * 32 + lane] = s0
                smem_scores[warp_offset + (j0 + 1) * 32 + lane] = s1
                smem_scores[warp_offset + (j0 + 2) * 32 + lane] = s2
                smem_scores[warp_offset + (j0 + 3) * 32 + lane] = s3
                key[j0 + 0] = k0
                key[j0 + 1] = k1
                key[j0 + 2] = k2
                key[j0 + 3] = k3

            best_key, best_local_idx = self._find_best_local(key)

            for slot in cutlass.range_constexpr(_TOPK):
                top = cute.arch.warp_redux_sync(best_key, "umax")
                tied_expert_id = Uint32(0xFFFFFFFF)
                if best_key == top:
                    tied_expert_id = Uint32(lane + 32 * best_local_idx)
                winner = Int32(cute.arch.warp_redux_sync(tied_expert_id, "umin"))
                if winner == lane + 32 * best_local_idx:
                    for j in cutlass.range_constexpr(experts_per_thread):
                        if best_local_idx == j:
                            key[j] = Uint32(0)
                    best_key, best_local_idx = self._find_best_local(key)
                if lane == slot:
                    slot_weight = smem_scores[warp_offset + winner]
                    slot_id = winner

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        if row < num_tokens:
            self._normalize_and_store(
                slot_weight,
                slot_id,
                routed_scaling_factor,
                is_padding,
                topk_weights,
                topk_ids,
                row,
                lane,
            )

    @cute.kernel
    def multi_warp_kernel(
        self,
        gating_output: Int64,
        correction_bias: Int64,
        is_padding: Int64,
        topk_weights: Int64,
        topk_ids: Int64,
        input_ids: Int64,
        bias_vl: Int64,
        image_sentinel_lo: Int32,
        routed_scaling_factor: Float32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp = tid // 32
        lane = tid % 32
        experts_per_thread = self.experts_per_thread
        warp_base = warp * (32 * experts_per_thread)

        smem = _SmemAllocator()
        storage = smem.allocate(Storage)
        smem_keys = storage.merge_keys.get_tensor(
            cute.make_layout(self.num_merge_candidates)
        )
        smem_ids = storage.merge_ids.get_tensor(
            cute.make_layout(self.num_merge_candidates)
        )
        smem_scores = storage.merge_scores.get_tensor(
            cute.make_layout(self.num_merge_candidates)
        )

        bias_ptr = self._select_bias_ptr(
            correction_bias, input_ids, bias_vl, row, image_sentinel_lo
        )
        expert_bias = cute.make_rmem_tensor(experts_per_thread, Float32)
        for j in cutlass.range_constexpr(experts_per_thread):
            e = warp_base + j * 32 + lane
            expert_bias[j] = _load_f32_nc(bias_ptr + Int64(e) * 4)

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        score = cute.make_rmem_tensor(experts_per_thread, Float32)
        key = cute.make_rmem_tensor(experts_per_thread, Uint32)
        row_base = gating_output + Int64(row) * (self.num_experts * 4)
        for j in cutlass.range_constexpr(experts_per_thread):
            e = warp_base + j * 32 + lane
            logit = _load_f32_ordered(row_base + Int64(e) * 4)
            score[j], key[j] = _score_and_key(logit, expert_bias[j])

        # Stage 1: each warp independently selects its top-6 via redux.sync.
        best_key, best_local_idx = self._find_best_local(key)

        out_key = Uint32(0)
        out_id = Int32(-1)
        out_score = Float32(0.0)
        for slot in cutlass.range_constexpr(_TOPK):
            top = cute.arch.warp_redux_sync(best_key, "umax")
            tied_expert_id = Uint32(0xFFFFFFFF)
            if best_key == top:
                tied_expert_id = Uint32(warp_base + lane + 32 * best_local_idx)
            winner = Int32(cute.arch.warp_redux_sync(tied_expert_id, "umin"))
            won = Float32(0.0)
            if winner - warp_base == lane + 32 * best_local_idx:
                for j in cutlass.range_constexpr(experts_per_thread):
                    if best_local_idx == j:
                        won = score[j]
                        key[j] = Uint32(0)
                best_key, best_local_idx = self._find_best_local(key)
            weight = cute.arch.shuffle_sync(won, (winner - warp_base) & 31)
            if lane == slot:
                out_key = top
                out_id = winner
                out_score = weight

        if lane < _TOPK:
            idx = warp * _TOPK + lane
            smem_keys[idx] = out_key
            smem_ids[idx] = out_id
            smem_scores[idx] = out_score

        cute.arch.barrier()

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        # Stage 2: warp 0 merges the warp top-6 candidates into the final top-6.
        if warp == 0:
            cand_key = Uint32(0)
            cand_id = Int32(0x7FFFFFFF)
            cand_score = Float32(0.0)
            if lane < self.num_merge_candidates:
                cand_key = smem_keys[lane]
                cand_id = smem_ids[lane]
                cand_score = smem_scores[lane]

            slot_weight = Float32(0.0)
            slot_id = Int32(0)
            for slot in cutlass.range_constexpr(_TOPK):
                top = cute.arch.warp_redux_sync(cand_key, "umax")
                tied_expert_id = Uint32(0xFFFFFFFF)
                if cand_key == top:
                    tied_expert_id = Uint32(cand_id)
                chosen = Int32(cute.arch.warp_redux_sync(tied_expert_id, "umin"))
                holders = cute.arch.vote_ballot_sync(cand_id == chosen)
                first = cute.arch.popc((holders & (Uint32(0) - holders)) - 1)
                weight = cute.arch.shuffle_sync(cand_score, first)
                if cand_id == chosen:
                    cand_key = Uint32(0)
                if lane == slot:
                    slot_weight = weight
                    slot_id = chosen

            self._normalize_and_store(
                slot_weight,
                slot_id,
                routed_scaling_factor,
                is_padding,
                topk_weights,
                topk_ids,
                row,
                lane,
            )


@cache
def _compiled_dsv4_topk(
    device_index: int,
    num_experts: int,
    num_warps: int,
    has_padding: bool,
    has_vl: bool,
    input_ids_is_int64: bool,
    is_int64: bool,
    use_pdl: bool,
):
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            _Dsv4TopkCutlass(
                num_experts=num_experts,
                num_warps=num_warps,
                has_padding=has_padding,
                has_vl=has_vl,
                input_ids_is_int64=input_ids_is_int64,
                is_int64=is_int64,
                use_pdl=use_pdl,
            ),
            *(Int64(0),) * 7,
            Int32(1),
            Int32(1),
            Int32(0),
            Float32(1.0),
            make_fake_stream(),
        )


def dsv4_topk_cutedsl(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    indices_dtype: torch.dtype,
    routed_scaling_factor: float,
    is_padding: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    bias_vl: torch.Tensor | None = None,
    image_sentinel_lo: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the CuTe-DSL `redux.sync` + PDL DSv4 top-6 kernel."""
    num_tokens, num_experts = gating_output.shape
    assert num_experts in _SUPPORTED_NUM_EXPERTS
    has_padding = is_padding is not None
    has_vl = bias_vl is not None and image_sentinel_lo > 0
    input_ids_is_int64 = (
        has_vl and input_ids is not None and (input_ids.dtype == torch.int64)
    )
    is_int64 = indices_dtype == torch.int64

    shape = (num_tokens, _TOPK)
    topk_weights = gating_output.new_empty(shape, dtype=torch.float32)
    topk_ids = gating_output.new_empty(shape, dtype=indices_dtype)
    if num_tokens > 0:
        device_index = gating_output.device.index
        num_warps = 4 if num_tokens <= _multi_warp_max_tokens(device_index) else 1
        grid_x = (
            (num_tokens + _SINGLE_WARP_TOKENS_PER_CTA - 1)
            // _SINGLE_WARP_TOKENS_PER_CTA
            if num_warps == 1
            else num_tokens
        )
        compiled = _compiled_dsv4_topk(
            device_index=device_index,
            num_experts=num_experts,
            num_warps=num_warps,
            has_padding=has_padding,
            has_vl=has_vl,
            input_ids_is_int64=input_ids_is_int64,
            is_int64=is_int64,
            use_pdl=_device_supports_pdl(device_index),
        )
        stream = torch.accelerator.current_stream(gating_output.device)
        stream_handle = getattr(stream, "native_handle", None)
        if stream_handle is None:
            stream_handle = stream.cuda_stream  # type: ignore[attr-defined]
        compiled(
            Int64(gating_output.data_ptr()),
            Int64(correction_bias.data_ptr()),
            Int64(is_padding.data_ptr() if is_padding is not None else 0),
            Int64(topk_weights.data_ptr()),
            Int64(topk_ids.data_ptr()),
            Int64(input_ids.data_ptr() if has_vl and input_ids is not None else 0),
            Int64(bias_vl.data_ptr() if has_vl and bias_vl is not None else 0),
            Int32(num_tokens),
            Int32(grid_x),
            Int32(image_sentinel_lo),
            Float32(routed_scaling_factor),
            cuda.CUstream(stream_handle),
        )
    return topk_weights, topk_ids
