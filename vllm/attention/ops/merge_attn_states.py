# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch

from vllm.platforms import current_platform


def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: Optional[torch.Tensor] = None,
) -> None:
    if current_platform.is_cuda():
        from vllm._custom_ops import merge_attn_states
        return merge_attn_states(output, prefix_output, prefix_lse,
                                 suffix_output, suffix_lse, output_lse)
    else:
        from vllm.attention.ops.triton_merge_attn_states import (
            merge_attn_states)
        return merge_attn_states(output, prefix_output, prefix_lse,
                                 suffix_output, suffix_lse, output_lse)
