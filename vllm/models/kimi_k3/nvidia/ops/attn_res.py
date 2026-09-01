# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code adapted from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
from dataclasses import dataclass
from typing import Any

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.warmup.jit_warmup import WarmupIntRange, zip_inputs
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
    kernel_launcher,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


class AttnResKernel(VllmTritonJitKernel["AttnResKernel.CompileKey"]):
    @dataclass(frozen=True)
    class CompileKey:
        dtype: torch.dtype
        stride_prefix_m: int
        stride_delta_m: int
        stride_block_m: int
        stride_block_r: int
        stride_output_m: int
        num_blocks: int
        hidden_size: int
        block_write_idx: int
        eps: float
        output_norm_eps: float
        has_delta: bool
        write_block: bool
        apply_output_norm: bool
        block_l: int
        block_d: int
        num_warps: int
        num_stages: int
        launch_pdl: bool

    @staticmethod
    @triton.jit
    def kernel(
        prefix_ptr,
        delta_ptr,
        blocks_ptr,
        norm_weight_ptr,
        qk_weight_ptr,
        output_norm_weight_ptr,
        output_ptr,
        stride_prefix_m: tl.constexpr,
        stride_delta_m: tl.constexpr,
        stride_block_m: tl.constexpr,
        stride_block_r: tl.constexpr,
        stride_output_m: tl.constexpr,
        num_blocks: tl.constexpr,
        hidden_size: tl.constexpr,
        block_write_idx: tl.constexpr,
        eps: tl.constexpr,
        output_norm_eps: tl.constexpr,
        HAS_DELTA: tl.constexpr,
        WRITE_BLOCK: tl.constexpr,
        APPLY_OUTPUT_NORM: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_D: tl.constexpr,
        launch_pdl: tl.constexpr,
    ):
        row_idx = tl.program_id(0).to(tl.int64)
        d_offsets = tl.max_contiguous(tl.arange(0, BLOCK_D), BLOCK_D)
        d_mask = d_offsets < hidden_size

        if launch_pdl:
            tl.extra.cuda.gdc_wait()

        updated_prefix = tl.load(
            prefix_ptr + row_idx * stride_prefix_m + d_offsets,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        if HAS_DELTA:
            delta = tl.load(
                delta_ptr + row_idx * stride_delta_m + d_offsets,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            updated_prefix += delta
            # Match the BF16 prefix-add result before using it as a residual source.
            updated_prefix = updated_prefix.to(prefix_ptr.dtype.element_ty).to(tl.float32)
            tl.store(
                prefix_ptr + row_idx * stride_prefix_m + d_offsets,
                updated_prefix,
                mask=d_mask,
            )
        if WRITE_BLOCK:
            tl.store(
                blocks_ptr
                + row_idx * stride_block_m
                + block_write_idx * stride_block_r
                + d_offsets,
                updated_prefix,
                mask=d_mask,
            )
        # With only the prefix source, the AttnRes softmax is exactly one.
        if num_blocks == 0:
            mixed = updated_prefix
        else:
            # Reloading avoids keeping the full prefix vector live across the loop.
            if HAS_DELTA:
                tl.debug_barrier()
            input_qk_weight = tl.load(
                norm_weight_ptr + d_offsets, mask=d_mask, other=0.0
            ).to(tl.float32) * tl.load(
                qk_weight_ptr + d_offsets, mask=d_mask, other=0.0
            ).to(tl.float32)
            max_logit = tl.full((), -float("inf"), tl.float32)
            denominator = tl.zeros((), tl.float32)
            mixed = tl.zeros((BLOCK_D,), tl.float32)

            num_sources = num_blocks + 1
            for source_tile in range(tl.cdiv(num_sources, BLOCK_L)):
                source_offsets = source_tile * BLOCK_L + tl.arange(0, BLOCK_L)
                source_mask = source_offsets < num_sources
                is_prefix = source_offsets == num_blocks
                block_ptrs = (
                    blocks_ptr
                    + row_idx * stride_block_m
                    + source_offsets[:, None] * stride_block_r
                    + d_offsets[None, :]
                )
                prefix_ptrs = (
                    prefix_ptr
                    + row_idx * stride_prefix_m
                    + source_offsets[:, None] * 0
                    + d_offsets[None, :]
                )
                value_ptrs = tl.where(is_prefix[:, None], prefix_ptrs, block_ptrs)
                values = tl.load(
                    value_ptrs,
                    mask=source_mask[:, None] & d_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                reciprocal_std = tl.rsqrt(
                    tl.sum(values * values, axis=1) * (1.0 / hidden_size) + eps
                )
                logits = tl.sum(values * input_qk_weight[None, :], axis=1) * reciprocal_std
                scores = tl.where(source_mask, logits, -float("inf"))

                new_max_logit = tl.maximum(max_logit, tl.max(scores, axis=0))
                old_scale = tl.exp(max_logit - new_max_logit)
                block_scales = tl.exp(scores - new_max_logit)
                denominator = denominator * old_scale + tl.sum(block_scales, axis=0)
                mixed = mixed * old_scale + tl.sum(block_scales[:, None] * values, axis=0)
                max_logit = new_max_logit

            mixed /= denominator
        output = mixed

        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()

        if APPLY_OUTPUT_NORM:
            output_reciprocal_std = tl.rsqrt(
                tl.sum(tl.where(d_mask, mixed * mixed, 0.0), axis=0) * (1.0 / hidden_size)
                + output_norm_eps
            )
            output_norm_weight = tl.load(
                output_norm_weight_ptr + d_offsets, mask=d_mask, other=0.0
            ).to(tl.float32)
            output = mixed * output_reciprocal_std * output_norm_weight
        tl.store(
            output_ptr + row_idx * stride_output_m + d_offsets,
            output,
            mask=d_mask,
        )

    def dispatch(  # type: ignore[override]
        self,
        *,
        dtype: torch.dtype,
        num_tokens: int,
        num_blocks: int,
        hidden_size: int,
        max_blocks: int,
        block_write_idx: int,
        eps: float,
        output_norm_eps: float,
        has_delta: bool,
        apply_output_norm: bool,
        launch_pdl: bool,
    ) -> CompileKey:
        block_l = 1 if num_tokens >= 256 or num_blocks <= 1 else 4
        return self.CompileKey(
            dtype=dtype,
            stride_prefix_m=hidden_size,
            stride_delta_m=hidden_size if has_delta else 0,
            stride_block_m=max_blocks * hidden_size,
            stride_block_r=hidden_size,
            stride_output_m=hidden_size,
            num_blocks=num_blocks,
            hidden_size=hidden_size,
            block_write_idx=block_write_idx,
            eps=eps,
            output_norm_eps=output_norm_eps if apply_output_norm else 0.0,
            has_delta=has_delta,
            write_block=block_write_idx >= 0,
            apply_output_norm=apply_output_norm,
            block_l=block_l,
            block_d=triton.next_power_of_2(hidden_size),
            num_warps=4 if block_l == 1 else 8,
            num_stages=2,
            launch_pdl=launch_pdl,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        config = vllm_config.model_config.hf_text_config
        attn_res_block_size = config.attn_res_block_size
        if attn_res_block_size is None:
            return []

        max_blocks = triton.cdiv(config.num_hidden_layers, attn_res_block_size)
        return self._trace_dispatch(self.dispatch)(
            zip_inputs(
                dict(apply_output_norm=False, output_norm_eps=0.0),
                dict(
                    apply_output_norm=True,
                    output_norm_eps=config.rms_norm_eps,
                ),
            ),
            dtype=vllm_config.model_config.dtype,
            num_tokens=(1, 256),
            num_blocks=WarmupIntRange(0, max_blocks + 1),
            hidden_size=config.hidden_size,
            max_blocks=max_blocks,
            block_write_idx=WarmupIntRange(-1, max_blocks),
            eps=config.rms_norm_eps,
            has_delta=(False, True),
            launch_pdl=current_platform.is_arch_support_pdl(),
            _when=lambda *, num_blocks, block_write_idx, apply_output_norm: (
                (block_write_idx == -1)
                or (apply_output_norm and block_write_idx == num_blocks)
            ),
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        max_blocks = compile_key.stride_block_m // compile_key.hidden_size
        num_tokens = (
            256 if compile_key.block_l == 1 and compile_key.num_blocks > 1 else 1
        )
        data = TritonWarmupTensor(
            compile_key.dtype,
            shape=(num_tokens, compile_key.hidden_size),
        )
        return dict(
            prefix=data,
            delta=data if compile_key.has_delta else None,
            blocks=TritonWarmupTensor(
                compile_key.dtype,
                shape=(num_tokens, max_blocks, compile_key.hidden_size),
            ),
            norm_weight=TritonWarmupTensor(
                compile_key.dtype,
                shape=(compile_key.hidden_size,),
            ),
            qk_weight=TritonWarmupTensor(
                compile_key.dtype,
                shape=(compile_key.hidden_size,),
            ),
            output_norm_weight=(
                TritonWarmupTensor(
                    compile_key.dtype,
                    shape=(compile_key.hidden_size,),
                )
                if compile_key.apply_output_norm
                else None
            ),
            output=data,
            num_blocks=compile_key.num_blocks,
            block_write_idx=compile_key.block_write_idx,
            eps=compile_key.eps,
            output_norm_eps=compile_key.output_norm_eps,
        )

    @kernel_launcher
    def __call__(
        self,
        prefix: torch.Tensor,
        delta: torch.Tensor | None,
        blocks: torch.Tensor,
        norm_weight: torch.Tensor,
        qk_weight: torch.Tensor,
        output_norm_weight: torch.Tensor | None,
        output: torch.Tensor,
        num_blocks: int,
        block_write_idx: int,
        eps: float,
        output_norm_eps: float,
    ) -> LaunchSpec:
        num_tokens, hidden_size = prefix.shape
        compile_key = self.dispatch(
            dtype=prefix.dtype,
            num_tokens=num_tokens,
            num_blocks=num_blocks,
            hidden_size=hidden_size,
            max_blocks=blocks.shape[1],
            block_write_idx=block_write_idx,
            eps=eps,
            output_norm_eps=output_norm_eps,
            has_delta=delta is not None,
            apply_output_norm=output_norm_weight is not None,
            launch_pdl=current_platform.is_arch_support_pdl(),
        )
        return (num_tokens,), dict(
            stride_prefix_m=prefix.stride(0),
            stride_delta_m=0 if delta is None else delta.stride(0),
            stride_block_m=blocks.stride(0),
            stride_block_r=blocks.stride(1),
            stride_output_m=output.stride(0),
            num_blocks=num_blocks,
            hidden_size=hidden_size,
            block_write_idx=block_write_idx,
            eps=eps,
            output_norm_eps=output_norm_eps,
            HAS_DELTA=delta is not None,
            WRITE_BLOCK=block_write_idx >= 0,
            APPLY_OUTPUT_NORM=output_norm_weight is not None,
            BLOCK_L=compile_key.block_l,
            BLOCK_D=compile_key.block_d,
            num_warps=compile_key.num_warps,
            num_stages=compile_key.num_stages,
            launch_pdl=compile_key.launch_pdl,
        )



def attn_res(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
) -> torch.Tensor:
    num_tokens, hidden_size = prefix.shape
    assert prefix.stride(-1) == 1
    assert delta is None or delta.stride(-1) == 1
    assert blocks.stride(-1) == 1
    assert norm_weight.stride(-1) == 1
    assert qk_weight.stride(-1) == 1
    assert output_norm_weight is None or output_norm_weight.stride(-1) == 1
    # The in-tree NVIDIA kernel covers the common fused-add + output-norm path;
    # Triton handles block boundaries and final pre-norm output. The native op
    # is only compiled for SM100 under CUDA >= 13, so a device check alone is
    # not enough to know it exists.
    if (
        hidden_size == 7168
        and delta is not None
        and output_norm_weight is not None
        and num_blocks > 0
        and block_write_idx < 0
        and current_platform.is_device_capability_family(100)
        and hasattr(torch.ops._C, "kimi_k3_attn_res")
    ):
        return ops.kimi_k3_attn_res(
            prefix,
            delta,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight,
            num_blocks,
            eps,
            output_norm_eps,
        )
    output = prefix.new_empty(prefix.shape)
    _ATTN_RES_KERNEL(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        output,
        num_blocks,
        block_write_idx,
        eps,
        output_norm_eps,
    )
    return output


_ATTN_RES_KERNEL = AttnResKernel()
