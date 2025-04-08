# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: Optional[torch.Tensor] = None,
) -> None:
    # NOTE(DefTruth): Currently, custom merge_attn_states CUDA kernel
    # is not support for FP8 dtype, fallback to use Triton kernel.
    supported_dtypes = [torch.float32, torch.half, torch.bfloat16]
    if current_platform.is_cuda() and (output.dtype in supported_dtypes):
        from vllm._custom_ops import merge_attn_states
        return merge_attn_states(output, prefix_output, prefix_lse,
                                 suffix_output, suffix_lse, output_lse)
    else:
        from vllm.attention.ops.triton_merge_attn_states import (
            merge_attn_states)
        return merge_attn_states(output, prefix_output, prefix_lse,
                                 suffix_output, suffix_lse, output_lse)
