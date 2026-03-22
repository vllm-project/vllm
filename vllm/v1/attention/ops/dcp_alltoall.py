# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP All-to-All 注意力通信后端模块。

本模块为 Decode Context Parallel (DCP) 提供 All-to-All (A2A) 通信能力，
作为 AllGather + ReduceScatter (AG+RS) 的替代方案。

主要特点：
- 不聚合完整的 Q 张量和分散部分输出，而是跨 rank 交换部分注意力输出和 LSE 值
- 使用精确的 LSE 加权约减组合部分结果
- 将每层注意力的 NCCL 调用次数从 3 次减少到 2 次，降低通信开销

使用方式：
    vllm serve model --tp 16 --dcp 16 --dcp-comm-backend a2a

参考资料：https://arxiv.org/abs/2507.07120
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.attention.ops.common import CPTritonContext


def _lse_weighted_combine(
    outputs: torch.Tensor,
    lses: torch.Tensor,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """LSE 加权组合的 CPU 参考实现。

    这是纯 PyTorch 实现，用于测试和验证。
    对于 GPU 执行，请使用 dcp_lse_combine_triton。

    Args:
        outputs: 部分注意力输出 [N, B, H, D]
                 N = KV 分片数（rank 数）
                 B = 批次大小（token 数）
                 H = 每个 rank 的头数
                 D = 头维度
        lses: Log-sum-exp 值 [N, B, H]
        return_lse: 如果为 True，同时返回全局 LSE
        is_lse_base_on_e: 如果为 True，LSE 以 e 为底；如果为 False，以 2 为底

    Returns:
        组合后的输出 [B, H, D]，以及可选的全局 LSE [B, H]
    """
    N, B, H, D = outputs.shape

    # Handle NaN and inf in LSEs
    lses = torch.where(
        torch.isnan(lses) | torch.isinf(lses),
        torch.tensor(float("-inf"), device=lses.device, dtype=lses.dtype),
        lses,
    )

    # Compute max LSE for numerical stability
    lse_max, _ = lses.max(dim=0)  # [B, H]
    lse_max = torch.where(
        lse_max == float("-inf"),
        torch.zeros_like(lse_max),
        lse_max,
    )

    # Compute weights: softmax over the N dimension
    if is_lse_base_on_e:
        weights = torch.exp(lses - lse_max.unsqueeze(0))  # [N, B, H]
    else:
        weights = torch.pow(2.0, lses - lse_max.unsqueeze(0))  # [N, B, H]

    # Handle NaN weights
    weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)

    # Normalize weights
    weight_sum = weights.sum(dim=0, keepdim=True)  # [1, B, H]
    weights = weights / weight_sum.clamp(min=1e-10)  # [N, B, H]

    # Weighted combination: sum over N dimension
    result = (outputs * weights.unsqueeze(-1)).sum(dim=0)  # [B, H, D]

    if return_lse:
        if is_lse_base_on_e:
            global_lse = torch.log(weight_sum.squeeze(0)) + lse_max  # [B, H]
        else:
            global_lse = torch.log2(weight_sum.squeeze(0)) + lse_max  # [B, H]
        return result, global_lse

    return result


@triton.jit
def _dcp_lse_combine_kernel(
    # Input pointers
    recv_output_ptr,
    recv_lse_ptr,
    # Output pointers
    out_ptr,
    out_lse_ptr,
    # Strides for recv_output [N, B, H_local, D]
    ro_stride_N,
    ro_stride_B,
    ro_stride_H,
    ro_stride_D,
    # Strides for recv_lse [N, B, H_local]
    rl_stride_N,
    rl_stride_B,
    rl_stride_H,
    # Strides for output [B, H_local, D]
    o_stride_B,
    o_stride_H,
    o_stride_D,
    # Constants
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    """DCP LSE 加权组合的 Triton kernel。

    All-to-All 之后，每个 rank 拥有：
    - recv_output [N, B, H_local, D]: 来自所有 KV 分片的部分输出
    - recv_lse [N, B, H_local]: 来自所有 KV 分片的部分 LSE

    该 kernel 在本地计算加权组合（无需通信）。

    Grid: (B, H_local)
    每个程序处理一个 (batch, head) 并处理所有 D 元素。
    """
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)

    # Base offset for this (batch, head)
    base_lse_offset = batch_idx * rl_stride_B + head_idx * rl_stride_H
    base_out_offset = batch_idx * ro_stride_B + head_idx * ro_stride_H

    # First pass: find max LSE for numerical stability
    lse_max = -float("inf")
    for n in tl.static_range(N):
        lse_offset = n * rl_stride_N + base_lse_offset
        lse_val = tl.load(recv_lse_ptr + lse_offset)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        lse_max = tl.maximum(lse_max, lse_val)

    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)

    # Second pass: compute sum of exp(lse - max)
    lse_sum = 0.0
    for n in tl.static_range(N):
        lse_offset = n * rl_stride_N + base_lse_offset
        lse_val = tl.load(recv_lse_ptr + lse_offset)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            lse_sum += tl.exp(lse_val - lse_max)
        else:
            lse_sum += tl.exp2(lse_val - lse_max)

    # Compute global LSE
    if IS_BASE_E:  # noqa: SIM108
        global_lse = tl.log(lse_sum) + lse_max
    else:
        global_lse = tl.log2(lse_sum) + lse_max

    # Third pass: weighted combination across D dimension
    d_offsets = tl.arange(0, HEAD_DIM)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for n in tl.static_range(N):
        lse_offset = n * rl_stride_N + base_lse_offset
        lse_val = tl.load(recv_lse_ptr + lse_offset)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            weight = tl.exp(lse_val - global_lse)
        else:
            weight = tl.exp2(lse_val - global_lse)
        weight = tl.where(weight != weight, 0.0, weight)

        out_offsets = n * ro_stride_N + base_out_offset + d_offsets * ro_stride_D
        out_vals = tl.load(recv_output_ptr + out_offsets)
        acc += out_vals.to(tl.float32) * weight

    # Store result
    final_offsets = (
        batch_idx * o_stride_B + head_idx * o_stride_H + d_offsets * o_stride_D
    )
    tl.store(out_ptr + final_offsets, acc)

    if RETURN_LSE:
        tl.store(out_lse_ptr + base_lse_offset, global_lse)


def dcp_lse_combine_triton(
    recv_output: torch.Tensor,
    recv_lse: torch.Tensor,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """DCP A2A 的 Triton 加速 LSE 加权组合。

    Args:
        recv_output: [N, B, H_local, D] - 来自所有 KV 分片的部分输出
        recv_lse: [N, B, H_local] - 来自所有 KV 分片的部分 LSE
        return_lse: 如果为 True，同时返回全局 LSE
        is_lse_base_on_e: 如果为 True，LSE 以 e 为底；如果为 False，以 2 为底

    Returns:
        组合后的输出 [B, H_local, D]
        如果 return_lse=True，同时返回 global_lse [B, H_local]
    """
    N, B, H_local, D = recv_output.shape

    out = torch.empty(
        (B, H_local, D), device=recv_output.device, dtype=recv_output.dtype
    )

    if return_lse:
        out_lse = torch.empty(
            (B, H_local), device=recv_lse.device, dtype=recv_lse.dtype
        )
    else:
        out_lse = torch.empty(1, device=recv_lse.device, dtype=recv_lse.dtype)

    ro_stride_N, ro_stride_B, ro_stride_H, ro_stride_D = recv_output.stride()
    rl_stride_N, rl_stride_B, rl_stride_H = recv_lse.stride()
    o_stride_B, o_stride_H, o_stride_D = out.stride()

    grid = (B, H_local, 1)

    _dcp_lse_combine_kernel[grid](
        recv_output,
        recv_lse,
        out,
        out_lse,
        ro_stride_N,
        ro_stride_B,
        ro_stride_H,
        ro_stride_D,
        rl_stride_N,
        rl_stride_B,
        rl_stride_H,
        o_stride_B,
        o_stride_H,
        o_stride_D,
        N=N,
        HEAD_DIM=D,
        IS_BASE_E=is_lse_base_on_e,
        RETURN_LSE=return_lse,
    )

    if return_lse:
        return out, out_lse
    return out


def dcp_a2a_lse_reduce(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """使用 All-to-All 组合 DCP rank 间的部分注意力输出。

    每个 rank 拥有所有头的注意力输出，但只拥有 KV 缓存的本地分片。该函数：
    1. 通过 All-to-All 跨 rank 交换部分输出
    2. 通过 All-to-All 交换 LSE 值
    3. 使用精确的 LSE 加权约减组合它们（Triton kernel）

    张量流程：
        输入：cp_attn_out [B, H, D] - 所有头，本地 KV 分片
        重塑：[N, B, H/N, D] - 跨 rank 分割头
        A2A: 两次 all_to_all_single 调用（输出和 LSE）
        组合：接收 [N, B, H/N, D] + lse [N, B, H/N] -> [B, H/N, D]

    Args:
        cp_attn_out: [B, H, D] 其中 B=num_tokens, H=total_heads, D=head_dim
        cp_attn_lse: [B, H] log-sum-exp 值 (fp32)
        cp_group: DCP 通信的 GroupCoordinator
        ctx: CPTritonContext（未使用，用于签名兼容性）
        return_lse: 如果为 True，同时返回组合后的全局 LSE
        is_lse_base_on_e: 如果为 True，LSE 以 e 为底；如果为 False，以 2 为底

    Returns:
        组合后的输出 [B, H/N, D]（头分散）
        如果 return_lse=True，同时返回 global_lse [B, H/N]
    """
    world_size = cp_group.world_size

    if world_size == 1:
        if return_lse:
            return cp_attn_out, cp_attn_lse
        return cp_attn_out

    local_output = cp_attn_out.contiguous()
    local_lse = cp_attn_lse.contiguous()

    B, H, D = local_output.shape
    H_per_rank = H // world_size

    # Reshape for All-to-All: [B, H, D] -> [N, B, H/N, D]
    # Split heads into N chunks, each destined for a different rank
    send_output = (
        local_output.view(B, world_size, H_per_rank, D).permute(1, 0, 2, 3).contiguous()
    )
    recv_output = torch.empty_like(send_output)

    # Same for LSE: [B, H] -> [N, B, H/N]
    send_lse = local_lse.view(B, world_size, H_per_rank).permute(1, 0, 2).contiguous()
    recv_lse = torch.empty_like(send_lse)

    # All-to-All for partial attention outputs and LSE values (async overlap)
    work_output = dist.all_to_all_single(
        recv_output.view(-1),
        send_output.view(-1),
        group=cp_group.device_group,
        async_op=True,
    )
    work_lse = dist.all_to_all_single(
        recv_lse.view(-1),
        send_lse.view(-1),
        group=cp_group.device_group,
        async_op=True,
    )
    work_output.wait()
    work_lse.wait()

    # LSE-weighted combination via Triton kernel (local, no communication)
    return dcp_lse_combine_triton(
        recv_output,
        recv_lse,
        return_lse=return_lse,
        is_lse_base_on_e=is_lse_base_on_e,
    )
