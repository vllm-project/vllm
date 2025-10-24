# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform


def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
) -> None:
    # NOTE(DefTruth): Currently, custom merge_attn_states CUDA kernel
    # is not support for FP8 dtype, fallback to use Triton kernel.
    def supported_dtypes(o: torch.Tensor) -> bool:
        return o.dtype in [torch.float32, torch.half, torch.bfloat16]

    # NOTE(DefTruth): Currently, custom merge_attn_states CUDA
    # kernel load/store 128b(16 bytes) per memory issue within
    # thread. Namely, the headsize(headdim) must be multiple of
    # pack_size (float32 -> 4, half/bfloat16 -> 8).
    def supported_headdim(o: torch.Tensor) -> bool:
        headdim = o.shape[2]  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
        if o.dtype == torch.float32:
            return headdim % 4 == 0
        return headdim % 8 == 0

    if (
        current_platform.is_cuda()
        and supported_dtypes(output)
        and supported_headdim(output)
    ):
        from vllm._custom_ops import merge_attn_states

        return merge_attn_states(
            output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse
        )
    else:
        from vllm.attention.ops.triton_merge_attn_states import merge_attn_states

        return merge_attn_states(
            output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse
        )
