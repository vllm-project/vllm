# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile the independent backend's decode variants without allocating KV."""

from collections.abc import Iterator
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.models.deepseek_v41.nvidia.ops.small_head_query_union import _pairs
from vllm.models.deepseek_v41.nvidia.ops.small_head_sparse_decode import (
    _merge_splits_kernel,
    _pick_config,
    _small_head_sparse_decode_kernel,
)
from vllm.triton_utils import triton
from vllm.v1.attention.backends.mla.compressor_utils import get_dspark_swa_index_width


def _decode_warmup_inputs(
    vllm_config: VllmConfig, block_size: int
) -> Iterator[dict[str, Any]]:
    config = vllm_config.model_config.hf_text_config
    heads = config.num_attention_heads // (
        vllm_config.parallel_config.tensor_parallel_size
    )
    widths = {config.sliding_window}
    spec = vllm_config.speculative_config
    if spec is not None and spec.use_dspark():
        widths.add(
            get_dspark_swa_index_width(
                config.sliding_window, spec.num_speculative_tokens
            )
        )
    ratios = sorted(set(getattr(config, "compress_ratios", None) or (0,)))
    int_ptr = TritonWarmupTensor(torch.int32)
    cache_ptr = TritonWarmupTensor(torch.uint8)
    partial_ptr = TritonWarmupTensor(torch.float32)
    for width in sorted(widths):
        for ratio in ratios:
            extra_width = config.index_topk if ratio else width
            extra_page = block_size // ratio if ratio else block_size
            # Token count only changes the launch configuration, not the JIT key.
            for tokens in (1, 4, 8):
                splits, block_n, warps, stages = _pick_config(
                    tokens, width + (extra_width if ratio else 0)
                )
                yield dict(
                    grid=(1,),
                    q_ptr=TritonWarmupTensor(torch.bfloat16),
                    q_stride_t=heads * 512,
                    q_stride_h=512,
                    swa_cache_ptr=cache_ptr,
                    swa_page_stride=block_size * 584,
                    swa_page_size=block_size,
                    swa_indices_ptr=int_ptr,
                    swa_indices_stride=width,
                    swa_lens_ptr=int_ptr,
                    extra_cache_ptr=cache_ptr,
                    extra_page_stride=extra_page * 584,
                    extra_page_size=extra_page,
                    extra_indices_ptr=int_ptr,
                    extra_indices_stride=extra_width,
                    extra_lens_ptr=int_ptr,
                    part_o_ptr=partial_ptr,
                    part_lse_ptr=partial_ptr,
                    num_heads=heads,
                    sm_scale=512**-0.5,
                    HAS_EXTRA=bool(ratio),
                    BLOCK_H=16,
                    BLOCK_N=block_n,
                    NUM_SPLITS=splits,
                    num_warps=warps,
                    num_stages=stages,
                )


def _merge_warmup_inputs(
    vllm_config: VllmConfig, block_size: int
) -> Iterator[dict[str, Any]]:
    config = vllm_config.model_config.hf_text_config
    heads = config.num_attention_heads // (
        vllm_config.parallel_config.tensor_parallel_size
    )
    splits = {
        case["NUM_SPLITS"] for case in _decode_warmup_inputs(vllm_config, block_size)
    }
    float_ptr = TritonWarmupTensor(torch.float32)
    for num_splits in sorted(splits):
        yield dict(
            grid=(1,),
            part_o_ptr=float_ptr,
            part_lse_ptr=float_ptr,
            sink_ptr=float_ptr,
            out_ptr=TritonWarmupTensor(torch.bfloat16),
            out_stride_t=heads * 512,
            out_stride_h=512,
            BLOCK_H=16,
            NUM_SPLITS=num_splits,
            num_warps=4,
        )


def _pair_warmup_inputs(vllm_config: VllmConfig) -> Iterator[dict[str, Any]]:
    config = vllm_config.model_config.hf_text_config
    if (
        config.num_attention_heads // vllm_config.parallel_config.tensor_parallel_size
        != 8
    ):
        return
    spec = vllm_config.speculative_config
    drafts = spec.num_speculative_tokens if spec is not None else 0
    capacity = min(
        vllm_config.scheduler_config.max_num_batched_tokens,
        vllm_config.scheduler_config.max_num_seqs * (drafts + 1),
    )
    int_ptr = TritonWarmupTensor(torch.int32)
    for power in range(5, triton.next_power_of_2(capacity).bit_length()):
        yield dict(
            grid=(1,),
            req=int_ptr,
            valid=TritonWarmupTensor(torch.bool),
            rows=int_ptr,
            count=int_ptr,
            T=min(capacity, 1 << power),
            B=1 << power,
            num_warps=4,
        )


# Warm the same JIT objects used at runtime.
DECODE_WARMUP = triton_kernel_dispatcher_with_warmup(
    warmup_inputs=_decode_warmup_inputs
)(_small_head_sparse_decode_kernel)
MERGE_WARMUP = triton_kernel_dispatcher_with_warmup(warmup_inputs=_merge_warmup_inputs)(
    _merge_splits_kernel
)
PAIR_WARMUP = triton_kernel_dispatcher_with_warmup(warmup_inputs=_pair_warmup_inputs)(
    _pairs
)


def register_small_head_decode_warmup(vllm_config: VllmConfig, block_size: int) -> None:
    DECODE_WARMUP.register_warmup(vllm_config=vllm_config, block_size=block_size)
    MERGE_WARMUP.register_warmup(vllm_config=vllm_config, block_size=block_size)
    PAIR_WARMUP.register_warmup(vllm_config=vllm_config)
