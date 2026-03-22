# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""上下文并行工具函数模块。

本模块提供上下文并行（CP/DCP）相关的辅助函数，负责：
- 准备 DCP 本地序列长度缓冲区
- 使用 Triton 内核计算每个 rank 的本地序列长度

主要函数：
- prepare_dcp_local_seq_lens: 填充 DCP 本地 seq_lens 缓冲区
- _dcp_local_seq_lens_kernel: Triton 内核计算本地序列长度
"""
import torch

from vllm.triton_utils import tl, triton


def prepare_dcp_local_seq_lens(
    dcp_local_seq_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    num_reqs: int,
    dcp_size: int,
    dcp_rank: int,
    cp_interleave: int,
) -> None:
    """填充持久的 DCP 本地 seq_lens 缓冲区（CUDA 图安全）。

    在上下文并行中，KV 缓存以轮转方式分布在不同的 ranks 之间。
    此函数计算每个 rank 需要处理的本地序列长度。

    Args:
        dcp_local_seq_lens: DCP 本地序列长度缓冲区
        seq_lens: 序列长度张量
        num_reqs: 请求数量
        dcp_size: DCP 组大小
        dcp_rank: DCP rank ID
        cp_interleave: CP 交错因子
    """
    if dcp_size == 1:
        return

    max_num_reqs = dcp_local_seq_lens.shape[0]
    BLOCK_SIZE = 128
    num_blocks = triton.cdiv(max_num_reqs, BLOCK_SIZE)
    _dcp_local_seq_lens_kernel[(num_blocks,)](
        dcp_local_seq_lens,
        seq_lens,
        dcp_size,
        dcp_rank,
        cp_interleave,
        num_reqs,
        max_num_reqs,
        BLOCK_SIZE,
    )


@triton.jit
def _dcp_local_seq_lens_kernel(
    out_ptr,
    seq_lens_ptr,
    dcp_size,
    dcp_rank,
    cp_interleave,
    num_reqs,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton 内核：计算 DCP 本地序列长度。

    以轮转方式在不同 ranks 之间分配 KV 缓存。
    计算公式：
    - rounds = seq_lens // (dcp_size * cp_interleave)
    - remainder = seq_lens % (dcp_size * cp_interleave)
    - local_seq_lens = rounds * cp_interleave + min(max(remainder - dcp_rank * cp_interleave, 0), cp_interleave)

    Args:
        out_ptr: 输出缓冲区指针
        seq_lens_ptr: 序列长度指针
        dcp_size: DCP 组大小
        dcp_rank: DCP rank ID
        cp_interleave: CP 交错因子
        num_reqs: 请求数量
        max_num_reqs: 最大请求数量
        BLOCK_SIZE: 块大小（编译时常量）
    """
    pid = tl.program_id(0)
    block = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    seq_lens = tl.load(seq_lens_ptr + block, mask=block < num_reqs)

    # 以轮转方式在不同 ranks 之间分配 KV 缓存
    rounds = seq_lens // (dcp_size * cp_interleave)
    remainder = seq_lens % (dcp_size * cp_interleave)

    remainder = tl.maximum(remainder - dcp_rank * cp_interleave, 0)
    remainder = tl.minimum(remainder, cp_interleave)
    local_seq_lens = rounds * cp_interleave + remainder

    # 对于 [num_reqs, max_num_reqs) 范围，填充 0
    local_seq_lens = tl.where(block < num_reqs, local_seq_lens, 0)
    tl.store(out_ptr + block, local_seq_lens, mask=block < max_num_reqs)
