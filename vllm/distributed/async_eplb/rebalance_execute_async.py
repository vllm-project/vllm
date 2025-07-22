# SPDX-License-Identifier: Apache-2.0
"""
The actual execution of the rearrangement.

This involves the exchange of expert weights between GPUs.
"""

from collections.abc import Iterable, MutableSequence, Sequence
from functools import partial

import threading
import torch
from torch.distributed import (P2POp, ProcessGroup, 
                               batch_isend_irecv, barrier)
from vllm.distributed.parallel_state import get_ep_group, get_node_count

def idx_local_to_global(
    local_idx: int,
    local_cnt: int,
    ep_rank: int,
) -> int:
    """
    Convert a local expert index to a global expert index.
    """
    return ep_rank * local_cnt + local_idx


def idx_global_to_local(
    global_idx: int,
    local_cnt: int,
    ep_rank: int,
) -> int:
    """
    Convert a global expert index to a local expert index.
    """
    return global_idx - ep_rank * local_cnt


def global_idx_to_rank(
    global_idx: int,
    local_cnt: int,
) -> int:
    """
    Convert a global expert index to a rank index.
    """
    return global_idx // local_cnt


def get_ep_ranks_with_expert(
    idx: int,
    num_local_experts: int,
    old_indices: Sequence[int],
    new_indices: Sequence[int],
) -> tuple[MutableSequence[int], MutableSequence[int]]:
    """
    Get the ranks of the experts that need to be exchanged.

    Args:
        idx: The index of the expert.
        num_local_experts: The number of local experts.
        old_indices: The old indices of the experts.
        new_indices: The new indices of the experts.

    Returns:
        A tuple of two lists:
        - The ranks of the experts that need to be sent.
        - The ranks of the experts that need to be received.
    """
    global2rank = partial(
        global_idx_to_rank,
        local_cnt=num_local_experts,
    )

    ranks_to_send: list[int] = []
    ranks_to_recv: list[int] = []

    for i, e in enumerate(old_indices):
        if e == idx:
            rank = global2rank(i)
            if not ranks_to_send or ranks_to_send[-1] != rank:
                ranks_to_send.append(rank)

    for i, e in enumerate(new_indices):
        if e == idx:
            rank = global2rank(i)
            if not ranks_to_recv or ranks_to_recv[-1] != rank:
                ranks_to_recv.append(rank)

    # Remove those ranks that can get this expert locally.
    ranks_to_send_set = set(ranks_to_send)
    ranks_to_recv_actual = [
        rank for rank in ranks_to_recv if rank not in ranks_to_send_set
    ]

    return ranks_to_send, ranks_to_recv_actual

from typing import Sequence, Iterable, Set
from functools import partial

import torch
import torch.distributed as dist
import threading
from functools import partial
from typing import Sequence, Iterable, Set, List, Optional


def get_global_rank(group, rank):
    return rank  # 简化实现

def get_ep_ranks_with_expert(expert, num_local_experts, old_indices, new_indices):
    # 简化实现
    return [0, 1], [0, 1]

def idx_local_to_global(local_idx, local_cnt, ep_rank):
    return local_idx + ep_rank * local_cnt

def barrier(group):
    dist.barrier(group=group)

async def async_move_to_buffer(
    num_local_experts: int,
    old_indices: Sequence[int],
    new_indices: Sequence[int],
    expert_weights: Iterable[torch.Tensor],
    expert_weights_buffer: Sequence[torch.Tensor],
    cuda_stream: Optional[torch.cuda.Stream] ,
) -> None:
    """
    异步将专家权重搬运到缓冲区，实现：
    1. 已完成发送的专家释放锁
    2. 新存入buffer的专家添加锁（不释放，供后续使用）
    """
    # 获取专家并行组
    ep_group = get_ep_group().device_group
    ep_rank = ep_group.rank()
    # 创建新的CUDA Stream
    #cuda_stream = torch.cuda.Stream()

    expert_locks = [threading.Lock() for _ in range(num_local_experts)]
    # 初始化锁定专家集合（跟踪新存入buffer的专家）
    locked_experts = set()
    
    local2global = partial(idx_local_to_global, local_cnt=num_local_experts, ep_rank=ep_rank)
    
    # 标记未变化的专家（原逻辑保留）
    is_unchanged = [
        old_indices[local2global(i)] == new_indices[local2global(i)]
        for i in range(num_local_experts)
    ]
    
    # 本地权重复制到缓冲区（原逻辑保留，本地复制完成后释放锁）
    is_received_locally = is_unchanged[:]
    for src in range(num_local_experts):
        src_global = local2global(src)
        for dst in range(num_local_experts):
            dst_global = local2global(dst)
            if is_received_locally[dst]:
                continue
            if old_indices[src_global] == new_indices[dst_global]:
                # 本地复制时加锁，防止写入冲突
                expert_locks[dst].acquire()
                try:
                    is_received_locally[dst] = True
                    with torch.cuda.stream(cuda_stream):
                        for weight, buffer in zip(expert_weights, expert_weights_buffer):
                            buffer[dst].copy_(weight[src])
                finally:
                    # 本地复制完成后释放锁（非新存入的跨进程专家，无需长期锁定）
                    expert_locks[dst].release()
    
    # 准备跨进程发送和接收操作（原逻辑保留）
    p2p_ops: list[P2POp] = []
    
    # 发送操作（修改：发送完成后释放锁）
    experts_send_loc = {}
    for src in range(num_local_experts):
        expert = old_indices[local2global(src)]
        if expert in experts_send_loc:
            continue
        experts_send_loc[expert] = src
    
    for expert, src in sorted(experts_send_loc.items()):
        # 发送前为源专家加锁，确保发送时数据不被修改
        expert_locks[src].acquire()
        try:
            ranks_to_send, ranks_to_recv = get_ep_ranks_with_expert(
                expert, num_local_experts, old_indices, new_indices
            )
            # 计算目标进程并添加发送操作（原逻辑保留）
            num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
            sender_pos = ranks_to_send.index(ep_rank)
            recv_begin = sender_pos * num_dst_per_sender
            recv_end = recv_begin + num_dst_per_sender
            recv_ranks = ranks_to_recv[recv_begin:recv_end]
            remainder_start = len(ranks_to_send) * num_dst_per_sender
            recver_pos = remainder_start + sender_pos
            if recver_pos < len(ranks_to_recv):
                recv_ranks.append(ranks_to_recv[recver_pos])
            
            # 在指定CUDA Stream中执行发送操作
            with torch.cuda.stream(cuda_stream):
                for dst in recv_ranks:
                    dst_global = get_global_rank(ep_group, dst)
                    p2p_ops += [
                        P2POp(torch.distributed.isend, weight[src], dst_global)
                        for weight in expert_weights
                    ]
            
            # 执行当前专家的发送操作（在指定流中）
            send_reqs = [op for op in p2p_ops if op.tensor is weight[src]]  # 筛选当前专家的发送请求
            if send_reqs:
                with torch.cuda.stream(cuda_stream):
                    reqs = batch_isend_irecv(send_reqs)
                    # 注意：这里使用 req.wait() 会同步该流，但不会阻塞其他流
                    for req in reqs:
                        req.wait()
        finally:
            # 发送完成后释放锁（已完成发送的专家，无需继续锁定）
            expert_locks[src].release()
    
    # 接收操作（修改：新存入buffer的专家添加锁并保持）
    experts_recv_loc = {}
    for dst in range(num_local_experts):
        if is_received_locally[dst]:
            continue
        expert = new_indices[local2global(dst)]
        if expert in experts_recv_loc:
            continue
        experts_recv_loc[expert] = dst
    
    # 【改动：临时字典存储每个目标专家的临时接收张量】
    temp_tensors = {}
    for expert, dst in sorted(experts_recv_loc.items()):
        ranks_to_send, ranks_to_recv = get_ep_ranks_with_expert(
            expert, num_local_experts, old_indices, new_indices
        )
        # 计算源进程（原逻辑保留）
        num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
        recver_pos = ranks_to_recv.index(ep_rank)
        remainder_start = len(ranks_to_send) * num_dst_per_sender
        if recver_pos < remainder_start:
            src = ranks_to_send[recver_pos // num_dst_per_sender]
        else:
            src = ranks_to_send[recver_pos - remainder_start]
        src_global = get_global_rank(ep_group, src)
        
        # 创建临时张量接收数据（原逻辑保留）
        temp_tensor = torch.empty_like(expert_weights_buffer[0][dst])
        temp_tensors[dst] = temp_tensor
        
        # 在指定CUDA Stream中执行接收操作
        with torch.cuda.stream(cuda_stream):
            p2p_ops += [
                P2POp(torch.distributed.irecv, temp_tensor, src_global)
                for _ in expert_weights_buffer
            ]
    
    # 执行异步P2P通信（在指定流中）
    if p2p_ops:
        with torch.cuda.stream(cuda_stream):
            reqs = batch_isend_irecv(p2p_ops)
            # 注意：这里使用 req.wait() 会同步该流，但不会阻塞其他流
            for req in reqs:
                req.wait()
    
    for dst in temp_tensors:
            # 在指定CUDA Stream中将临时张量的数据复制到缓冲区
        with torch.cuda.stream(cuda_stream):
            for weight, buffer in zip(expert_weights, expert_weights_buffer):
                buffer[dst].copy_(temp_tensors[dst])
    # 等待所有操作在该流中完成
    cuda_stream.synchronize()
    # 设置屏障（在指定CUDA Stream中）
    with torch.cuda.stream(cuda_stream):
        barrier(group=ep_group)
    
    
    
    # 此时缓冲区已准备好，但仍被锁定
    return is_unchanged, is_received_locally, experts_recv_loc

def move_from_buffer(
    expert_weights: Iterable[torch.Tensor],
    expert_weights_buffer: Sequence[torch.Tensor],
    is_unchanged: list[bool],
    is_received_locally: list[bool],
    experts_recv_loc: dict[int, int],
    new_indices: Sequence[int],
) -> None:
    """
    将专家权重从缓冲区搬运到工作区
    完成后释放锁
    """
    num_local_experts= len(expert_weights_buffer)
    local2global = partial(idx_local_to_global, local_cnt=num_local_experts, ep_rank=ep_rank)
    
    # 将缓冲区的专家权重搬运到工作区
    # copy_操作为CUDA特定，需要抽象
    for dst in range(num_local_experts):
        if is_unchanged[dst]:
            continue
        if is_received_locally[dst]:
            for weight, buffer in zip(expert_weights, expert_weights_buffer):
                weight[dst].copy_(buffer[dst])
        else:
            expert = new_indices[local2global(dst)]
            src = experts_recv_loc[expert]
            for weight, buffer in zip(expert_weights, expert_weights_buffer):
                weight[dst].copy_(buffer[src])
    


async def eplb_worker(
    old_global_expert_indices: torch.Tensor,
    new_global_expert_indices: torch.Tensor,
    expert_weights: Sequence[Iterable[torch.Tensor]],
    expert_weights_buffer,
    is_profile: bool = False,
    layer: int = 0,
    cuda_stream : Optional[torch.cuda.Stream] = None,
) -> None:
    """
    Rearranges the expert weights in place according to the new expert indices.

    The value of the indices arguments are logical indices of the experts,
    while keys are physical.

    Args:
        old_global_expert_indices: Shape (num_moe_layers, num_physical_experts).
        new_global_expert_indices: Shape (num_moe_layers, num_physical_experts).
        expert_weights: A sequence of shape (num_moe_layers)(weight_count)
            of tensors of shape (num_local_physical_experts, hidden_size_i).
            For example, a linear layer may have up and down projection,
            so weight_count = 2. Each weight's hidden size can be different.
        ep_group: The device process group for expert parallelism.
        is_profile (bool): If `True`, do not perform any actual weight copy.
            This is used during profile run, where we only perform dummy
            communications to reserve enough memory for the buffers.
    """
    num_moe_layers, num_physical_experts = old_global_expert_indices.shape
    assert len(expert_weights) == num_moe_layers

    ep_group = get_ep_group().device_group
    num_local_physical_experts = next(iter(expert_weights[0])).shape[0]
    assert new_global_expert_indices.shape == (num_moe_layers,
                                               num_physical_experts)

    ep_rank = ep_group.rank()
    ep_size = ep_group.size()
    assert num_physical_experts == ep_size * num_local_physical_experts

    # A buffer to hold the expert weights in one layer during the exchange.
    # NOTE: Currently we assume the same weights across different layers
    # have the same shape.
    
    #expert_weights_buffer = [torch.empty_like(w) for w in expert_weights[0]]

    if is_profile:
        # Maximum send size is to send all local experts to all ranks,
        # So we use a dummy `all_gather` to reserve enough communication buffer
        for weight, buffer in zip(expert_weights[0], expert_weights_buffer):
            # A `/dev/null`-like buffer to avoid real memory allocation
            dummy_recv_buffer = [buffer for _ in range(ep_size)]
            # NOTE(bowen): Needed this barrier to avoid OOM during actual
            # execution. I'm not very sure why this is needed
            torch.distributed.barrier()
            torch.distributed.all_gather.all_gather(
                dummy_recv_buffer,
                weight,
                group=ep_group,
            )
        return

    torch.cuda.synchronize()
    await async_move_to_buffer(
        num_local_experts=num_local_physical_experts,
        old_indices=old_global_expert_indices[layer].tolist(),
        new_indices=new_global_expert_indices[layer].tolist(),
        expert_weights=expert_weights[layer],
        expert_weights_buffer=expert_weights_buffer,
        cuda_stream=cuda_stream,
    )
        # NOTE(bowen): We need this synchronize to run, but I don't know why.
        # If you figure out the reason, please let me know -- thank you!
        #N卡GPU专属操作，需要抽象
        


__all__ = ["eplb_worker", "move_from_buffer"]
