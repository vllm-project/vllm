# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Taken from https://github.com/ModelTC/LightLLM/blob/8ed97c74c18f11505b048b1ba00ba5c0cef8bff6/lightllm/common/fused_moe/deepep_scatter_gather.py
and updated to fit vllm needs and terminology.
"""

from dataclasses import dataclass
from typing import Any

import math

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.utils import count_expert_num_tokens
from vllm.model_executor.warmup.jit_warmup import (
    VllmJitKernel,
    WarmupIntRange,
    zip_inputs,
)
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    triton_scalar_specialization_rep,
)
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    get_mk_alignment_for_contiguous_layout,
    get_theoretical_mk_alignment_for_contiguous_layout,
)
from vllm.utils.math_utils import round_up


def expert_num_tokens_round_up_and_sum(
    expert_num_tokens: torch.Tensor, alignment: int
) -> int:
    # Round up each element in expert_num_tokens to the nearest multiple of
    # alignment.
    ent = (expert_num_tokens.to(torch.int64) + (alignment - 1)) // alignment * alignment
    return torch.sum(ent).item()


def compute_aligned_M_and_alignment(
    M: int,
    num_topk: int,
    local_num_experts: int,
    alignment: int,
    expert_tokens_meta: mk.ExpertTokensMetadata | None,
) -> tuple[int, int]:
    """Return (M_sum, alignment_used).

    `alignment_used` may be smaller than the caller-supplied `alignment` on
    SM100/SM120 when DeepGEMM can JIT a smaller BLOCK_M for the per-call
    expected_m. Callers that index by block size (e.g. ``M_sum // block_m``)
    or assert workspace alignment must use the returned `alignment_used`,
    not their original `alignment` argument.

    Prefer this over the int-returning :func:`compute_aligned_M` when the
    GEMM call site needs to wrap itself in ``mk_alignment_scope`` or
    otherwise reason about the actual per-expert padding.
    """
    if (expert_tokens_meta is not None) and (
        expert_tokens_meta.expert_num_tokens_cpu is not None
    ):
        return (
            expert_num_tokens_round_up_and_sum(
                expert_tokens_meta.expert_num_tokens_cpu, alignment=alignment
            ),
            alignment,
        )

    # expert_num_tokens not on cpu. Cap padding by min(M*num_topk,
    # local_num_experts) — at batch=1 decode only `num_topk` experts can be
    # active, so the worst-case `local_num_experts*(align-1)` is too loose.
    # Also shrink `alignment` to DeepGEMM's per-call theoretical BLOCK_M on
    # SM100/SM120 when smaller.
    expected_m = M * num_topk
    try:
        from vllm.utils.deep_gemm import (
            get_theoretical_mk_alignment_for_contiguous_layout,
        )

        # num_groups=local_num_experts so the helper recovers per-expert em;
        # omitting it over-picks BLOCK_M on SM120 (heuristic assumes em is
        # already per-expert).
        per_call_align = get_theoretical_mk_alignment_for_contiguous_layout(
            expected_m=expected_m,
            num_groups=local_num_experts,
        )
        if per_call_align and per_call_align <= alignment:
            alignment = per_call_align
    except Exception:
        pass

    max_active_experts = min(M * num_topk, local_num_experts)
    M_sum = (M * num_topk) + max_active_experts * (alignment - 1)
    M_sum = round_up(M_sum, alignment)
    return M_sum, alignment


def compute_aligned_M(
    M: int,
    num_topk: int,
    local_num_experts: int,
    alignment: int,
    expert_tokens_meta: mk.ExpertTokensMetadata | None,
) -> int:
    """Return ``M_sum`` only (backward-compat wrapper).

    Equivalent to :func:`compute_aligned_M_and_alignment`'s first return
    value. Existing downstream callers and the warmup path that only size
    a workspace use this. Call sites that need the actual per-expert
    alignment (to wrap GEMMs in ``mk_alignment_scope``) should use
    :func:`compute_aligned_M_and_alignment` instead.
    """
    M_sum, _ = compute_aligned_M_and_alignment(
        M, num_topk, local_num_experts, alignment, expert_tokens_meta
    )
    return M_sum


@triton.jit
def apply_expert_map(expert_id, expert_map):
    if expert_id != -1:
        expert_id = tl.load(expert_map + expert_id).to(expert_id.dtype)
    return expert_id


class DeepGemmEPScatterStartKernel(
    VllmJitKernel["DeepGemmEPScatterStartKernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        num_experts: int
        block_e: int
        block_expert_num: int
        align_m: int

    @staticmethod
    @triton.jit
    def kernel(
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_EXPERT_NUM: tl.constexpr,
        ALIGN_M: tl.constexpr,
    ):
        cur_expert = tl.program_id(0)

        offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
        tokens_per_expert = tl.load(
            num_recv_tokens_per_expert + offset_cumsum,
            mask=offset_cumsum < num_experts,
            other=0,
        )
        # Round up to ALIGN_M so cumsum matches the workspace's per-expert slices.
        tokens_per_expert = ((tokens_per_expert + ALIGN_M - 1) // ALIGN_M) * ALIGN_M
        cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert

        # Extract this block's offset from the register vector (warp shuffle,
        # no global memory round-trip) then write it once to expert_start_loc.
        cur_expert_start = tl.sum(
            tl.where(offset_cumsum == cur_expert, cumsum, tl.zeros_like(cumsum))
        )
        tl.store(expert_start_loc + cur_expert, cur_expert_start)
        cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)

        m_indices_start_ptr = m_indices + cur_expert_start
        off_expert = tl.arange(0, BLOCK_E)

        # any rows in the per-expert aligned region that do not correspond to
        # real tokens are left untouched here and should remain initialized to
        # -1 so DeepGEMM can skip them
        for start_m in tl.range(0, cur_expert_token_num, BLOCK_E):
            offs = start_m + off_expert
            mask = offs < cur_expert_token_num
            tl.store(
                m_indices_start_ptr + offs,
                cur_expert,
                mask=mask,
            )

    def dispatch(  # type: ignore[override]
        self,
        *,
        num_experts: int,
        align_m: int | None,
        num_tokens: int = 0,
        top_k: int = 0,
        max_align_m: int = 0,
    ) -> CompileKey:
        resolved_align_m = (
            align_m
            if align_m is not None
            else min(
                get_theoretical_mk_alignment_for_contiguous_layout(
                    expected_m=num_tokens * top_k,
                    num_groups=num_experts,
                )
                or max_align_m,
                max_align_m,
            )
        )
        return self.CompileKey(
            num_experts=num_experts,
            # BLOCK_E is the m_indices fill-loop tile (masked), independent of align_m.
            block_e=128,
            block_expert_num=triton.next_power_of_2(num_experts),
            align_m=resolved_align_m,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        hf_config = vllm_config.model_config.hf_config
        if getattr(hf_config, "model_type", None) != "deepseek_v4":
            return []

        num_experts = hf_config.n_routed_experts
        parallel_config = vllm_config.parallel_config
        eplb_config = getattr(parallel_config, "eplb_config", None)
        num_experts += int(getattr(eplb_config, "num_redundant_experts", 0) or 0)
        if num_experts > 0 and parallel_config.enable_expert_parallel:
            try:
                from vllm.distributed.parallel_state import get_ep_group

                world_size = get_ep_group().world_size
            except Exception:
                world_size = int(getattr(parallel_config, "data_parallel_size", 1) or 1)
            num_experts //= world_size if world_size >= 1 else 1
        if num_experts <= 0:
            return []

        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        top_k = hf_config.num_experts_per_tok
        if max_tokens <= 0 or top_k <= 0:
            return []

        max_align_m, _ = get_mk_alignment_for_contiguous_layout()
        return self._trace_dispatch(self.dispatch)(
            num_experts=num_experts,
            align_m=None,
            num_tokens=WarmupIntRange(1, max_tokens + 1),
            top_k=top_k,
            max_align_m=max_align_m,
        )

    def compile(self, compile_key: CompileKey) -> None:
        warmup = getattr(self.kernel, "warmup", None)
        assert warmup is not None
        int32_ptr = TritonWarmupTensor(torch.int32)
        warmup(
            int32_ptr,
            int32_ptr,
            int32_ptr,
            num_experts=compile_key.num_experts,
            BLOCK_E=compile_key.block_e,
            BLOCK_EXPERT_NUM=compile_key.block_expert_num,
            ALIGN_M=compile_key.align_m,
            grid=(compile_key.num_experts,),
            num_warps=8,
        )

    def __call__(
        self,
        num_recv_tokens_per_expert: torch.Tensor,
        expert_start_loc: torch.Tensor,
        m_indices: torch.Tensor,
        *,
        align_m: int,
    ) -> None:
        num_experts = num_recv_tokens_per_expert.shape[0]
        compile_key = self.dispatch(
            num_experts=num_experts,
            align_m=align_m,
        )
        self.kernel[(num_experts,)](
            num_recv_tokens_per_expert,
            expert_start_loc,
            m_indices,
            num_experts=num_experts,
            num_warps=8,
            BLOCK_E=compile_key.block_e,
            BLOCK_EXPERT_NUM=compile_key.block_expert_num,
            ALIGN_M=align_m,
        )


class DeepGemmEPScatterCopyKernel(
    VllmJitKernel["DeepGemmEPScatterCopyKernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        total_token_num: int
        topk_num: int
        has_expert_map: bool
        hidden_size: int
        hidden_size_pad: int
        scale_hidden_size: int
        scale_hidden_size_pad: int
        pack_ue8m0: bool
        scale_packed_size: int
        scale_packed_size_pad: int

    @staticmethod
    @triton.jit
    def kernel(
        total_token_num,
        expert_start_loc,
        recv_x,
        recv_x_stride0,
        recv_x_stride1,
        recv_x_scale,
        recv_x_scale_stride0,
        recv_x_scale_stride1,
        recv_topk,
        recv_topk_stride0,
        recv_topk_stride1,
        output_tensor,
        output_tensor_stride0,
        output_tensor_stride1,
        output_tensor_scale,
        output_tensor_scale_stride0,
        output_tensor_scale_stride1,
        output_index,
        output_index_stride0,
        output_index_stride1,
        topk_num: tl.constexpr,
        expert_map,
        HAS_EXPERT_MAP: tl.constexpr,
        HIDDEN_SIZE: tl.constexpr,
        HIDDEN_SIZE_PAD: tl.constexpr,
        SCALE_HIDDEN_SIZE: tl.constexpr,
        SCALE_HIDDEN_SIZE_PAD: tl.constexpr,
        PACK_UE8M0: tl.constexpr,
        SCALE_PACKED_SIZE: tl.constexpr,
        SCALE_PACKED_SIZE_PAD: tl.constexpr,
    ):
        start_token_id = tl.program_id(0)
        grid_num = tl.num_programs(0)

        offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
        mask = offset_in < HIDDEN_SIZE

        output_tensor_stride0 = output_tensor_stride0.to(tl.int64)

        if PACK_UE8M0:
            # One int32 per 4 consecutive 32-wide UE8M0 groups, stored MN-major.
            offs_pk = tl.arange(0, SCALE_PACKED_SIZE_PAD)
            mask_pk = offs_pk < SCALE_PACKED_SIZE
        else:
            offset_in_s = tl.arange(0, SCALE_HIDDEN_SIZE_PAD)
            mask_s = offset_in_s < SCALE_HIDDEN_SIZE

        for token_id in range(start_token_id, total_token_num, grid_num):
            to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)

            if PACK_UE8M0:
                # Pack 4 UE8M0 bytes into one int32 (byte j = group 4*pk+j).
                base_s = recv_x_scale + token_id * recv_x_scale_stride0
                g0, g1 = offs_pk * 4, offs_pk * 4 + 1
                g2, g3 = offs_pk * 4 + 2, offs_pk * 4 + 3
                b0 = tl.load(
                    base_s + g0 * recv_x_scale_stride1, mask=g0 < SCALE_HIDDEN_SIZE
                )
                b1 = tl.load(
                    base_s + g1 * recv_x_scale_stride1, mask=g1 < SCALE_HIDDEN_SIZE
                )
                b2 = tl.load(
                    base_s + g2 * recv_x_scale_stride1, mask=g2 < SCALE_HIDDEN_SIZE
                )
                b3 = tl.load(
                    base_s + g3 * recv_x_scale_stride1, mask=g3 < SCALE_HIDDEN_SIZE
                )
                packed_s = (
                    b0.to(tl.int32)
                    | (b1.to(tl.int32) << 8)
                    | (b2.to(tl.int32) << 16)
                    | (b3.to(tl.int32) << 24)
                )
            else:
                to_copy_s = tl.load(
                    recv_x_scale + token_id * recv_x_scale_stride0 + offset_in_s,
                    mask=mask_s,
                )

            for topk_index in tl.range(0, topk_num, 1, num_stages=4):
                expert_id = tl.load(
                    recv_topk + token_id * recv_topk_stride0 + topk_index
                )

                if HAS_EXPERT_MAP:
                    expert_id = apply_expert_map(expert_id, expert_map)

                if expert_id >= 0:
                    dest_token_index = tl.atomic_add(expert_start_loc + expert_id, 1)
                    dest_token_index_i64 = dest_token_index.to(tl.int64)
                    tl.store(
                        output_index + token_id * output_index_stride0 + topk_index,
                        dest_token_index,
                    )
                    output_tensor_ptr = (
                        output_tensor + dest_token_index_i64 * output_tensor_stride0
                    )
                    tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)

                    output_tensor_scale_ptr = (
                        output_tensor_scale
                        + dest_token_index * output_tensor_scale_stride0
                    )
                    if PACK_UE8M0:
                        tl.store(
                            output_tensor_scale_ptr
                            + offs_pk * output_tensor_scale_stride1,
                            packed_s,
                            mask=mask_pk,
                        )
                    else:
                        tl.store(
                            output_tensor_scale_ptr + offset_in_s,
                            to_copy_s,
                            mask=mask_s,
                        )

    def dispatch(  # type: ignore[override]
        self,
        *,
        total_token_num: int,
        hidden_size: int,
        topk_num: int,
        has_expert_map: bool,
        block_size: int,
        pack_ue8m0: bool,
    ) -> CompileKey:
        scale_hidden_size = hidden_size // block_size
        # pack_ue8m0: scatter packs 4 UE8M0 bytes per int32; else copies scales as-is.
        scale_packed_size = (scale_hidden_size + 3) // 4 if pack_ue8m0 else 1
        return self.CompileKey(
            total_token_num=triton_scalar_specialization_rep(total_token_num),
            topk_num=topk_num,
            has_expert_map=has_expert_map,
            hidden_size=hidden_size,
            hidden_size_pad=triton.next_power_of_2(hidden_size),
            scale_hidden_size=scale_hidden_size,
            scale_hidden_size_pad=triton.next_power_of_2(scale_hidden_size),
            pack_ue8m0=pack_ue8m0,
            scale_packed_size=scale_packed_size,
            scale_packed_size_pad=triton.next_power_of_2(scale_packed_size),
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        hf_config = vllm_config.model_config.hf_config
        if getattr(hf_config, "model_type", None) != "deepseek_v4":
            return []

        hidden_size = hf_config.hidden_size
        topk_num = hf_config.num_experts_per_tok
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        if hidden_size <= 0 or topk_num <= 0 or max_tokens <= 0:
            return []

        _, block_k = get_mk_alignment_for_contiguous_layout()
        return self._trace_dispatch(self.dispatch)(
            zip_inputs(
                dict(block_size=block_k, pack_ue8m0=False),
                dict(block_size=32, pack_ue8m0=True),
            ),
            total_token_num=WarmupIntRange(1, max_tokens + 1),
            hidden_size=hidden_size,
            topk_num=topk_num,
            has_expert_map=(False, True),
        )

    def compile(self, compile_key: CompileKey) -> None:
        warmup = getattr(self.kernel, "warmup", None)
        assert warmup is not None
        int32_ptr = TritonWarmupTensor(torch.int32)
        fp8_ptr = TritonWarmupTensor(torch.float8_e4m3fn)
        scale_ptr = TritonWarmupTensor(
            torch.int32 if compile_key.pack_ue8m0 else torch.float32
        )
        expert_map = int32_ptr if compile_key.has_expert_map else None
        hidden_stride = triton_scalar_specialization_rep(compile_key.hidden_size)
        scale_stride = triton_scalar_specialization_rep(
            compile_key.scale_hidden_size
        )
        topk_stride = triton_scalar_specialization_rep(compile_key.topk_num)
        output_scale_stride0, output_scale_stride1 = (
            (1, 16) if compile_key.pack_ue8m0 else (scale_stride, 1)
        )
        warmup(
            compile_key.total_token_num,
            int32_ptr,
            fp8_ptr,
            hidden_stride,
            1,
            scale_ptr,
            scale_stride,
            1,
            int32_ptr,
            topk_stride,
            1,
            fp8_ptr,
            hidden_stride,
            1,
            scale_ptr,
            output_scale_stride0,
            output_scale_stride1,
            int32_ptr,
            topk_stride,
            1,
            topk_num=compile_key.topk_num,
            expert_map=expert_map,
            HAS_EXPERT_MAP=compile_key.has_expert_map,
            HIDDEN_SIZE=compile_key.hidden_size,
            HIDDEN_SIZE_PAD=compile_key.hidden_size_pad,
            SCALE_HIDDEN_SIZE=compile_key.scale_hidden_size,
            SCALE_HIDDEN_SIZE_PAD=compile_key.scale_hidden_size_pad,
            PACK_UE8M0=compile_key.pack_ue8m0,
            SCALE_PACKED_SIZE=compile_key.scale_packed_size,
            SCALE_PACKED_SIZE_PAD=compile_key.scale_packed_size_pad,
            grid=(1,),
            num_warps=8,
        )

    def __call__(
        self,
        recv_x: torch.Tensor,
        recv_x_scale: torch.Tensor,
        recv_topk: torch.Tensor,
        expert_map: torch.Tensor | None,
        expert_start_loc: torch.Tensor,
        output_tensor: torch.Tensor,
        output_tensor_scale: torch.Tensor,
        output_index: torch.Tensor,
        *,
        block_size: int,
        pack_ue8m0: bool,
    ) -> None:
        hidden_size = recv_x.shape[1]
        compile_key = self.dispatch(
            total_token_num=recv_x.shape[0],
            hidden_size=hidden_size,
            topk_num=recv_topk.shape[1],
            has_expert_map=expert_map is not None,
            block_size=block_size,
            pack_ue8m0=pack_ue8m0,
        )
        self.kernel[(min(recv_topk.shape[0], 1024 * 8),)](
            recv_topk.shape[0],
            expert_start_loc,
            recv_x,
            recv_x.stride(0),
            recv_x.stride(1),
            recv_x_scale,
            recv_x_scale.stride(0),
            recv_x_scale.stride(1),
            recv_topk,
            recv_topk.stride(0),
            recv_topk.stride(1),
            output_tensor,
            output_tensor.stride(0),
            output_tensor.stride(1),
            output_tensor_scale,
            output_tensor_scale.stride(0),
            output_tensor_scale.stride(1),
            output_index,
            output_index.stride(0),
            output_index.stride(1),
            topk_num=compile_key.topk_num,
            expert_map=expert_map,
            HAS_EXPERT_MAP=compile_key.has_expert_map,
            num_warps=8,
            HIDDEN_SIZE=compile_key.hidden_size,
            HIDDEN_SIZE_PAD=compile_key.hidden_size_pad,
            SCALE_HIDDEN_SIZE=compile_key.scale_hidden_size,
            SCALE_HIDDEN_SIZE_PAD=compile_key.scale_hidden_size_pad,
            PACK_UE8M0=compile_key.pack_ue8m0,
            SCALE_PACKED_SIZE=compile_key.scale_packed_size,
            SCALE_PACKED_SIZE_PAD=compile_key.scale_packed_size_pad,
        )


class DeepGemmEPScatterKernel:
    def __init__(
        self,
        *,
        start: DeepGemmEPScatterStartKernel,
        copy: DeepGemmEPScatterCopyKernel,
    ) -> None:
        self.start = start
        self.copy = copy

    def __call__(
        self,
        recv_x: torch.Tensor,
        recv_x_scale: torch.Tensor,
        recv_topk: torch.Tensor,
        num_recv_tokens_per_expert: torch.Tensor,
        expert_map: torch.Tensor | None,
        expert_start_loc: torch.Tensor,
        output_tensor: torch.Tensor,
        output_tensor_scale: torch.Tensor,
        m_indices: torch.Tensor,
        output_index: torch.Tensor,
        align_m: int,
        block_size: int,
        pack_ue8m0: bool,
    ) -> None:
        self.start(
            num_recv_tokens_per_expert,
            expert_start_loc,
            m_indices,
            align_m=align_m,
        )
        self.copy(
            recv_x,
            recv_x_scale,
            recv_topk,
            expert_map,
            expert_start_loc,
            output_tensor,
            output_tensor_scale,
            output_index,
            block_size=block_size,
            pack_ue8m0=pack_ue8m0,
        )


class DeepGemmEPGatherKernel(VllmJitKernel["DeepGemmEPGatherKernel.CompileKey"]):
    @dataclass(frozen=True)
    class CompileKey:
        dtype: torch.dtype
        total_token_num: int
        topk_num: int
        has_expert_map: bool
        block_d: int
        hidden_stride: int
        topk_stride: int

    @staticmethod
    @triton.jit
    def kernel(
        total_token_num,
        input_tensor,
        input_tensor_stride0,
        input_tensor_stride1,
        recv_topk_ids,
        recv_topk_ids_stride0,
        recv_topk_ids_stride1,
        recv_topk_weight,
        recv_topk_weight_stride0,
        recv_topk_weight_stride1,
        input_index,
        input_index_stride0,
        input_index_stride1,
        output_tensor,
        output_tensor_stride0,
        output_tensor_stride1,
        topk_num: tl.constexpr,
        expert_map,
        HAS_EXPERT_MAP: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        cur_block = tl.program_id(0)
        start_cur_token = tl.program_id(1)
        grid_num = tl.num_programs(1)

        for cur_token in range(start_cur_token, total_token_num, grid_num):
            off_d = tl.arange(0, BLOCK_D)
            accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
            for topk_index in range(0, topk_num):
                expert_id = tl.load(
                    recv_topk_ids + cur_token * recv_topk_ids_stride0 + topk_index
                )

                if HAS_EXPERT_MAP:
                    expert_id = apply_expert_map(expert_id, expert_map)

                if expert_id >= 0:
                    source_token_index = tl.load(
                        input_index + cur_token * input_index_stride0 + topk_index
                    )
                    acc_weight = tl.load(
                        recv_topk_weight
                        + cur_token * recv_topk_weight_stride0
                        + topk_index
                    )
                    tmp = tl.load(
                        input_tensor
                        + source_token_index * input_tensor_stride0
                        + cur_block * BLOCK_D
                        + off_d
                    )
                    accumulator += tmp.to(tl.float32) * acc_weight

            tl.store(
                output_tensor
                + cur_token * output_tensor_stride0
                + cur_block * BLOCK_D
                + off_d,
                accumulator.to(output_tensor.dtype.element_ty),
            )

    def dispatch(  # type: ignore[override]
        self,
        *,
        dtype: torch.dtype,
        total_token_num: int,
        hidden_size: int,
        topk_num: int,
        has_expert_map: bool,
    ) -> CompileKey:
        return self.CompileKey(
            dtype=dtype,
            total_token_num=triton_scalar_specialization_rep(total_token_num),
            topk_num=topk_num,
            has_expert_map=has_expert_map,
            block_d=math.gcd(hidden_size, 1024),
            hidden_stride=triton_scalar_specialization_rep(hidden_size),
            topk_stride=triton_scalar_specialization_rep(topk_num),
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        hidden_size = vllm_config.model_config.hf_config.hidden_size
        topk_num = vllm_config.model_config.hf_config.num_experts_per_tok
        dtype = vllm_config.model_config.dtype
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        if hidden_size <= 0 or topk_num <= 0 or max_tokens <= 0:
            return []

        return self._trace_dispatch(self.dispatch)(
            dtype=dtype,
            total_token_num=WarmupIntRange(1, max_tokens + 1),
            hidden_size=hidden_size,
            topk_num=topk_num,
            has_expert_map=(False, True),
        )

    def compile(self, compile_key: CompileKey) -> None:
        warmup = getattr(self.kernel, "warmup", None)
        assert warmup is not None

        value_ptr = TritonWarmupTensor(compile_key.dtype)
        int32_ptr = TritonWarmupTensor(torch.int32)
        warmup(
            compile_key.total_token_num,
            value_ptr,
            compile_key.hidden_stride,
            1,
            int32_ptr,
            compile_key.topk_stride,
            1,
            TritonWarmupTensor(torch.float32),
            compile_key.topk_stride,
            1,
            int32_ptr,
            compile_key.topk_stride,
            1,
            value_ptr,
            compile_key.hidden_stride,
            1,
            topk_num=compile_key.topk_num,
            expert_map=(int32_ptr if compile_key.has_expert_map else None),
            HAS_EXPERT_MAP=compile_key.has_expert_map,
            BLOCK_D=compile_key.block_d,
            grid=(1, 1),
            num_warps=2,
        )

    def __call__(
        self,
        input_tensor: torch.Tensor,
        recv_topk_ids: torch.Tensor,
        recv_topk_weight: torch.Tensor,
        input_index: torch.Tensor,
        expert_map: torch.Tensor | None,
        output_tensor: torch.Tensor,
    ) -> None:
        num_warps = 2
        num_tokens = output_tensor.shape[0]
        hidden_size = input_tensor.shape[1]
        block_d = math.gcd(hidden_size, 1024)
        assert hidden_size % block_d == 0
        grid = (triton.cdiv(hidden_size, block_d), min(num_tokens, 1024))

        self.kernel[grid](
            num_tokens,
            input_tensor,
            input_tensor.stride(0),
            input_tensor.stride(1),
            recv_topk_ids,
            recv_topk_ids.stride(0),
            recv_topk_ids.stride(1),
            recv_topk_weight,
            recv_topk_weight.stride(0),
            recv_topk_weight.stride(1),
            input_index,
            input_index.stride(0),
            input_index.stride(1),
            output_tensor,
            output_tensor.stride(0),
            output_tensor.stride(1),
            topk_num=recv_topk_ids.shape[1],
            expert_map=expert_map,
            HAS_EXPERT_MAP=expert_map is not None,
            num_warps=num_warps,
            BLOCK_D=block_d,
        )


@torch.no_grad()
def ep_scatter(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_map: torch.Tensor | None,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
    align_m: int = 128,
    block_size: int = 128,
    pack_ue8m0: bool = False,
):
    block_d = block_size  # block size of activation-scale quantization
    num_experts = num_recv_tokens_per_expert.shape[0]

    assert m_indices.shape[0] % align_m == 0
    assert expert_start_loc.shape[0] == num_experts

    _DEEPGEMM_EP_SCATTER_KERNEL(
        recv_x,
        recv_x_scale,
        recv_topk,
        num_recv_tokens_per_expert,
        expert_map,
        expert_start_loc,
        output_tensor,
        output_tensor_scale,
        m_indices,
        output_index,
        align_m,
        block_d,
        pack_ue8m0,
    )
    return


@torch.no_grad()
def ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
    expert_map: torch.Tensor | None,
    output_tensor: torch.Tensor,
):
    _DEEPGEMM_EP_GATHER_KERNEL(
        input_tensor,
        recv_topk_ids,
        recv_topk_weight,
        input_index,
        expert_map,
        output_tensor,
    )
    return


def deepgemm_moe_permute(
    aq: torch.Tensor,
    aq_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    local_num_experts: int,
    expert_map: torch.Tensor | None,
    expert_tokens_meta: mk.ExpertTokensMetadata | None,
    aq_out: torch.Tensor | None = None,
    block_size: int | None = None,
):
    assert aq.ndim == 2
    assert topk_ids.dtype.is_signed, "The kernel uses -1 to represent invalid topk_ids"
    H = aq.size(1)
    device = aq.device

    block_m, block_k = get_mk_alignment_for_contiguous_layout()
    # The activation-scale group size may differ from the M/K tile alignment
    # (e.g. MXFP8 uses a 32-element scale group while block_k stays 128).
    if block_size is not None:
        block_k = block_size

    M_sum, align_used = compute_aligned_M_and_alignment(
        M=topk_ids.size(0),
        num_topk=topk_ids.size(1),
        local_num_experts=local_num_experts,
        alignment=block_m,
        expert_tokens_meta=expert_tokens_meta,
    )

    expert_start_loc = torch.empty(
        (local_num_experts), device=device, dtype=torch.int32
    )

    assert aq_out is None or aq_out.shape == (M_sum, H)
    if aq_out is None:
        aq_out = torch.empty((M_sum, H), device=device, dtype=aq.dtype)

    # uint8 UE8M0 (MXFP8) -> scatter packs into DeepGEMM's int32 MN-major
    # TMA-aligned layout; float32 (FP8/FP4) scattered row-major as-is.
    pack_ue8m0 = aq_scale.dtype == torch.uint8
    sf_k = H // block_k
    if pack_ue8m0:
        packed_sf_k = (sf_k + 3) // 4
        tma_aligned_mn = round_up(M_sum, 4)
        aq_scale_out = torch.empty_strided(
            (M_sum, packed_sf_k),
            (1, tma_aligned_mn),
            device=device,
            dtype=torch.int32,
        )
    else:
        aq_scale_out = torch.zeros((M_sum, sf_k), device=device, dtype=torch.float32)

    # DeepGEMM uses negative values in m_indices (here expert_ids) to mark
    # completely invalid / padded blocks that should be skipped. We always
    # initialize expert_ids to -1 so any row that is not explicitly written
    # by the scatter kernel will be treated as invalid and skipped by
    # DeepGEMM's scheduler.
    expert_ids = torch.full(
        (M_sum,),
        fill_value=-1,
        device=device,
        dtype=torch.int32,
    )
    inv_perm = torch.empty(topk_ids.shape, device=device, dtype=torch.int32)

    expert_num_tokens = None
    if expert_tokens_meta is not None:
        expert_num_tokens = expert_tokens_meta.expert_num_tokens
    else:
        expert_num_tokens = count_expert_num_tokens(
            topk_ids, local_num_experts, expert_map
        )

    ep_scatter(
        recv_x=aq,
        recv_x_scale=aq_scale,
        recv_topk=topk_ids,
        num_recv_tokens_per_expert=expert_num_tokens,
        expert_start_loc=expert_start_loc,
        expert_map=expert_map,
        output_tensor=aq_out,
        output_tensor_scale=aq_scale_out,
        m_indices=expert_ids,
        output_index=inv_perm,
        align_m=align_used,
        block_size=block_k,
        pack_ue8m0=pack_ue8m0,
    )

    return aq_out, aq_scale_out, expert_ids, inv_perm, align_used


def deepgemm_unpermute_and_reduce(
    a: torch.Tensor,  # Grouped gemm output
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_perm: torch.Tensor,
    expert_map: torch.Tensor | None,
    output: torch.Tensor,
):
    return ep_gather(
        input_tensor=a,
        recv_topk_ids=topk_ids,
        recv_topk_weight=topk_weights,
        input_index=inv_perm,
        expert_map=expert_map,
        output_tensor=output,
    )


_DEEPGEMM_EP_SCATTER_START_KERNEL = DeepGemmEPScatterStartKernel()
_DEEPGEMM_EP_SCATTER_COPY_KERNEL = DeepGemmEPScatterCopyKernel()
_DEEPGEMM_EP_SCATTER_KERNEL = DeepGemmEPScatterKernel(
    start=_DEEPGEMM_EP_SCATTER_START_KERNEL,
    copy=_DEEPGEMM_EP_SCATTER_COPY_KERNEL,
)
_DEEPGEMM_EP_GATHER_KERNEL = DeepGemmEPGatherKernel()
