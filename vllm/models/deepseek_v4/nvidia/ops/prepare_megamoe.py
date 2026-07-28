# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton input-staging kernel for DeepSeek V4 MegaMoE.

Quantizes hidden states to fp8 with E8M0 group scales and repacks the
routing top-k tensors into the int64/float32 layout that the DeepGEMM
MegaMoE kernels consume.
"""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import (
    VllmJitKernel,
    zip_inputs,
)
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2


class PrepareMegaMoeInputsKernel(
    VllmJitKernel["PrepareMegaMoeInputsKernel.CompileKey"]
):
    def __init__(self) -> None:
        self.block_k = 128
        self.group_k = 32
        super().__init__()

    @dataclass(frozen=True)
    class CompileKey:
        hidden_size: int
        top_k: int
        block_topk: int
        has_padding: bool
        has_shared_x_sf: bool
        shared_block_m: int

    @staticmethod
    @triton.jit
    def kernel(
        hidden_states,
        x_fp8,
        x_sf,
        shared_x_sf,
        topk_ids,
        topk_weights,
        is_padding,
        topk_idx_out,
        topk_weights_out,
        hidden_stride_m: tl.constexpr,
        hidden_stride_k: tl.constexpr,
        x_stride_m: tl.constexpr,
        x_stride_k: tl.constexpr,
        x_sf_stride_m: tl.constexpr,
        x_sf_stride_k: tl.constexpr,
        shared_x_sf_stride_m: tl.constexpr,
        shared_x_sf_stride_k: tl.constexpr,
        topk_ids_stride_m: tl.constexpr,
        topk_ids_stride_k: tl.constexpr,
        topk_weights_stride_m: tl.constexpr,
        topk_weights_stride_k: tl.constexpr,
        is_padding_stride_m: tl.constexpr,
        topk_idx_stride_m: tl.constexpr,
        topk_idx_stride_k: tl.constexpr,
        topk_weights_out_stride_m: tl.constexpr,
        topk_weights_out_stride_k: tl.constexpr,
        hidden_size: tl.constexpr,
        top_k: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_K: tl.constexpr,
        BLOCK_TOPK: tl.constexpr,
        SHARED_BLOCK_M: tl.constexpr,
    ) -> None:
        token_id = tl.program_id(0)
        k_block_id = tl.program_id(1)

        k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < hidden_size
        hidden = tl.load(
            hidden_states + token_id * hidden_stride_m + k_offsets * hidden_stride_k,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)

        num_groups: tl.constexpr = BLOCK_K // GROUP_K
        hidden_groups = tl.reshape(tl.abs(hidden), [num_groups, GROUP_K])
        amax = tl.max(hidden_groups, axis=1)
        amax = tl.maximum(amax, 1.0e-4)

        scale = amax / 448.0
        scale_bits = scale.to(tl.uint32, bitcast=True)
        scale_exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(
            tl.uint32
        )
        scale_exp = tl.minimum(tl.maximum(scale_exp, 1), 254)
        rounded_scale = (scale_exp << 23).to(tl.float32, bitcast=True)

        hidden_groups = tl.reshape(hidden, [num_groups, GROUP_K])
        scaled = hidden_groups * (1.0 / rounded_scale)[:, None]
        scaled = tl.reshape(scaled, [BLOCK_K])
        fp8 = scaled.to(tl.float8e4nv)
        tl.store(
            x_fp8 + token_id * x_stride_m + k_offsets * x_stride_k,
            fp8,
            mask=k_mask,
        )

        scale_offsets = tl.arange(0, num_groups)
        packed_scale = tl.sum(scale_exp << (scale_offsets * 8), axis=0).to(tl.int32)
        tl.store(
            x_sf + token_id * x_sf_stride_m + k_block_id * x_sf_stride_k,
            packed_scale,
        )

        # Shared-expert TMA consumes the same scales in an MN-major layout.
        if shared_x_sf is not None:
            m_block_id = token_id // SHARED_BLOCK_M
            m_in_block = token_id % SHARED_BLOCK_M
            aligned_block_m: tl.constexpr = triton.cdiv(SHARED_BLOCK_M, 128) * 128
            transposed_m = (
                (m_in_block // 128) * 128
                + (m_in_block % 32) * 4
                + (m_in_block % 128) // 32
            )
            shared_row = m_block_id * aligned_block_m + transposed_m
            tl.store(
                shared_x_sf
                + shared_row * shared_x_sf_stride_m
                + k_block_id * shared_x_sf_stride_k,
                packed_scale,
            )

        if k_block_id == 0:
            topk_offsets = tl.arange(0, BLOCK_TOPK)
            topk_mask = topk_offsets < top_k
            token_is_padding = False
            if is_padding is not None:
                token_is_padding = tl.load(is_padding + token_id * is_padding_stride_m)

            ids = tl.load(
                topk_ids
                + token_id * topk_ids_stride_m
                + topk_offsets * topk_ids_stride_k,
                mask=topk_mask,
                other=0,
            ).to(tl.int64)
            ids = tl.where(token_is_padding, -1, ids)
            tl.store(
                topk_idx_out
                + token_id * topk_idx_stride_m
                + topk_offsets * topk_idx_stride_k,
                ids,
                mask=topk_mask,
            )

            weights = tl.load(
                topk_weights
                + token_id * topk_weights_stride_m
                + topk_offsets * topk_weights_stride_k,
                mask=topk_mask,
                other=0.0,
            )
            weights = tl.where(token_is_padding, 0.0, weights)
            tl.store(
                topk_weights_out
                + token_id * topk_weights_out_stride_m
                + topk_offsets * topk_weights_out_stride_k,
                weights,
                mask=topk_mask,
            )

    def dispatch(  # type: ignore[override]
        self,
        *,
        hidden_size: int,
        top_k: int,
        has_padding: bool,
        has_shared_x_sf: bool,
        shared_block_m: int,
    ) -> CompileKey:
        block_topk = next_power_of_2(top_k)
        return self.CompileKey(
            hidden_size=hidden_size,
            top_k=top_k,
            block_topk=block_topk,
            has_padding=has_padding,
            has_shared_x_sf=has_shared_x_sf,
            shared_block_m=shared_block_m if has_shared_x_sf else 1,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        hf_config = vllm_config.model_config.hf_config
        if getattr(hf_config, "model_type", None) != "deepseek_v4":
            return []

        hidden_size = vllm_config.model_config.hf_config.hidden_size
        top_k = vllm_config.model_config.hf_config.num_experts_per_tok
        has_shared_experts = getattr(
            vllm_config.model_config.hf_config, "n_shared_experts", None
        ) is not None
        if hidden_size <= 0 or top_k <= 0:
            return []

        # DeepGEMM's MegaMoE scheduler selects from these BLOCK_M candidates.
        shared_inputs = zip_inputs(
            dict(has_shared_x_sf=False, shared_block_m=1),
            dict(has_shared_x_sf=has_shared_experts, shared_block_m=8),
            dict(has_shared_x_sf=has_shared_experts, shared_block_m=16),
            dict(has_shared_x_sf=has_shared_experts, shared_block_m=32),
            dict(has_shared_x_sf=has_shared_experts, shared_block_m=64),
            dict(has_shared_x_sf=has_shared_experts, shared_block_m=96),
            dict(has_shared_x_sf=has_shared_experts, shared_block_m=128),
            dict(has_shared_x_sf=has_shared_experts, shared_block_m=192),
        )
        return self._trace_dispatch(self.dispatch)(
            shared_inputs,
            hidden_size=hidden_size,
            top_k=top_k,
            has_padding=(False, True),
        )

    def compile(self, compile_key: CompileKey) -> None:
        warmup = getattr(self.kernel, "warmup", None)
        assert warmup is not None

        hidden_size = compile_key.hidden_size
        top_k = compile_key.top_k
        block_k = self.block_k
        group_k = self.group_k
        # Scale groups are packed into one int32 per BLOCK_K-wide hidden block.
        x_scale_width = hidden_size // block_k

        hidden_ptr = TritonWarmupTensor(torch.bfloat16, shape=(1, hidden_size))
        fp8_ptr = TritonWarmupTensor(torch.float8_e4m3fn, shape=(1, hidden_size))
        int32_ptr = TritonWarmupTensor(torch.int32)
        shared_scale_ptr = (
            TritonWarmupTensor(torch.int32)
            if compile_key.has_shared_x_sf
            else None
        )
        topk_int32_ptr = TritonWarmupTensor(torch.int32, shape=(1, top_k))
        topk_int64_ptr = TritonWarmupTensor(torch.int64, shape=(1, top_k))
        topk_float32_ptr = TritonWarmupTensor(torch.float32, shape=(1, top_k))
        padding_ptr = (
            TritonWarmupTensor(torch.bool) if compile_key.has_padding else None
        )

        warmup(
            hidden_ptr,
            fp8_ptr,
            int32_ptr,
            shared_scale_ptr,
            topk_int32_ptr,
            topk_float32_ptr,
            padding_ptr,
            topk_int64_ptr,
            topk_float32_ptr,
            hidden_size,
            1,
            hidden_size,
            1,
            x_scale_width,
            1,
            1 if compile_key.has_shared_x_sf else 0,
            1 if compile_key.has_shared_x_sf else 0,
            top_k,
            1,
            top_k,
            1,
            1 if compile_key.has_padding else 0,
            top_k,
            1,
            top_k,
            1,
            hidden_size,
            top_k,
            BLOCK_K=block_k,
            GROUP_K=group_k,
            BLOCK_TOPK=compile_key.block_topk,
            SHARED_BLOCK_M=compile_key.shared_block_m,
            num_warps=4,
            grid=(1, triton.cdiv(hidden_size, block_k)),
        )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        x_fp8: torch.Tensor,
        x_sf: torch.Tensor,
        topk_idx_out: torch.Tensor,
        topk_weights_out: torch.Tensor,
        is_padding: torch.Tensor | None = None,
        shared_x_sf: torch.Tensor | None = None,
        shared_block_m: int | None = None,
    ) -> None:
        num_tokens, hidden_size = hidden_states.shape
        if num_tokens == 0:
            return
        if hidden_size % 128 != 0:
            raise ValueError(
                "DeepSeek V4 MegaMoE input staging requires hidden_size to be "
                "a multiple of 128."
            )
        top_k = topk_ids.shape[1]
        if topk_weights.shape != topk_ids.shape:
            raise ValueError(
                "DeepSeek V4 MegaMoE input staging requires topk_weights and "
                "topk_ids to have the same shape."
            )
        if (shared_x_sf is None) != (shared_block_m is None):
            raise ValueError(
                "DeepSeek V4 MegaMoE shared input staging requires both "
                "shared_x_sf and shared_block_m."
            )
        if shared_x_sf is not None:
            assert shared_block_m is not None
            if shared_block_m <= 0:
                raise ValueError("MegaMoE shared_block_m must be positive.")
            expected_sf_k = hidden_size // self.block_k
            if shared_x_sf.ndim != 2 or shared_x_sf.shape[1] != expected_sf_k:
                raise ValueError(
                    "MegaMoE shared_x_sf must have shape "
                    f"(*, {expected_sf_k}), got {tuple(shared_x_sf.shape)}."
                )
            aligned_block_m = triton.cdiv(shared_block_m, 128) * 128
            required_rows = triton.cdiv(num_tokens, shared_block_m) * aligned_block_m
            if shared_x_sf.shape[0] < required_rows:
                raise ValueError(
                    "MegaMoE shared_x_sf has insufficient rows: requires "
                    f"{required_rows}, got {shared_x_sf.shape[0]}."
                )

        block_k = self.block_k
        block_topk = triton.next_power_of_2(top_k)
        grid = (num_tokens, triton.cdiv(hidden_size, block_k))
        padding_stride_m = is_padding.stride(0) if is_padding is not None else 0
        self.kernel[grid](
            hidden_states,
            x_fp8,
            x_sf,
            shared_x_sf,
            topk_ids,
            topk_weights,
            is_padding,
            topk_idx_out,
            topk_weights_out,
            hidden_states.stride(0),
            hidden_states.stride(1),
            x_fp8.stride(0),
            x_fp8.stride(1),
            x_sf.stride(0),
            x_sf.stride(1),
            shared_x_sf.stride(0) if shared_x_sf is not None else 0,
            shared_x_sf.stride(1) if shared_x_sf is not None else 0,
            topk_ids.stride(0),
            topk_ids.stride(1),
            topk_weights.stride(0),
            topk_weights.stride(1),
            padding_stride_m,
            topk_idx_out.stride(0),
            topk_idx_out.stride(1),
            topk_weights_out.stride(0),
            topk_weights_out.stride(1),
            hidden_size,
            top_k,
            BLOCK_K=block_k,
            GROUP_K=self.group_k,
            BLOCK_TOPK=block_topk,
            SHARED_BLOCK_M=shared_block_m or 1,
            num_warps=4,
        )


_PREPARE_MEGAMOE_INPUTS_KERNEL = PrepareMegaMoeInputsKernel()
