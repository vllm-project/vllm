# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""合并注意力状态操作模块。

本模块实现了合并前后缀注意力状态的操作，负责：
- 合并 prefix 和 suffix 的注意力输出和 LSE
- 根据平台和数据类型选择 CUDA 或 Triton 实现

主要函数：
- merge_attn_states: 合并注意力状态
"""

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
    """合并注意力状态（prefix 和 suffix）。

    根据平台和数据类型选择合适的实现：
    - CUDA 平台且数据类型支持：使用自定义 CUDA kernel
    - 其他情况：使用 Triton kernel

    支持的 dtype: float32, half, bfloat16
    支持的 head_size:
    - float32: 必须是 4 的倍数
    - half/bfloat16: 必须是 8 的倍数

    Args:
        output: 输出张量
        prefix_output: prefix 输出
        prefix_lse: prefix LSE
        suffix_output: suffix 输出
        suffix_lse: suffix LSE
        output_lse: 输出 LSE（可选）
    """
    # NOTE(DefTruth): Currently, custom merge_attn_states CUDA kernel
    # does not support FP8 dtype, fallback to use Triton kernel.
    def supported_dtypes(o: torch.Tensor) -> bool:
        """检查数据类型是否支持。

        Args:
            o: 张量

        Returns:
            是否支持
        """
        return o.dtype in [torch.float32, torch.half, torch.bfloat16]

    # NOTE(DefTruth): Currently, custom merge_attn_states CUDA
    # kernel load/store 128b(16 bytes) per memory issue within
    # thread. Namely, the headsize(headdim) must be multiple of
    # pack_size (float32 -> 4, half/bfloat16 -> 8).
    def supported_headdim(o: torch.Tensor) -> bool:
        """检查头大小是否支持。

        Args:
            o: 张量

        Returns:
            是否支持
        """
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
        from vllm.v1.attention.ops.triton_merge_attn_states import merge_attn_states

        return merge_attn_states(
            output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse
        )
