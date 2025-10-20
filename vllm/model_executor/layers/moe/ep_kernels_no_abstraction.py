# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# torchrun --nproc_per_node=2 vllm/model_executor/layers/moe/ep_kernels_no_abstraction.py # noqa: E501

# type: ignore
import os

import torch
import torch.distributed as dist
from deep_ep import Buffer, EventOverlap

# Communication buffer (will allocate at runtime)
_buffer: Buffer | None = None

# Set the number of SMs to use
# NOTES: this is a static variable
Buffer.set_num_sms(24)


def get_buffer(group: dist.ProcessGroup, hidden_bytes: int) -> Buffer:
    global _buffer

    # NOTES: you may also replace `get_*_config` with your auto-tuned results via all
    # the tests
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Allocate a buffer if not existed or not enough buffer size
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


def get_hidden_bytes(x: torch.Tensor) -> int:
    t = x[0] if isinstance(x, tuple) else x
    return t.size(1) * max(t.element_size(), 2)


def dispatch_forward(
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    previous_event: EventOverlap | None = None,
) -> tuple[
    torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    list,
    tuple,
    EventOverlap,
]:
    # NOTES: an optional `previous_event` means a CUDA event captured that you want to
    # make it as a dependency of the dispatch kernel, it may be useful with
    # communication-computation overlap. For more information, please
    # refer to the docs of `Buffer.dispatch`
    global _buffer
    assert _buffer is not None

    # Calculate layout before actual dispatch
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        previous_event,
    ) = _buffer.get_dispatch_layout(
        topk_idx,
        num_experts,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=previous_event is not None,
    )
    # Do MoE dispatch
    # NOTES: the CPU will wait for GPU's signal to arrive, so this is not compatible
    # with CUDA graph. Unless you specify `num_worst_tokens`, but this flag is
    # for intranode only. For more advanced usages, please refer to the docs of
    # the `dispatch` function
    (
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        num_recv_tokens_per_expert_list,
        handle,
        event,
    ) = _buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    # For event management, please refer to the docs of the `EventOverlap` class
    return (
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        num_recv_tokens_per_expert_list,
        handle,
        event,
    )


if __name__ == "__main__":
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
    )
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(local_rank)

    group = dist.group.WORLD
    num_experts = 8
    local_batch_size = 4
    batch_size = local_batch_size * world_size
    hidden_size = 128
    local_num_experts = num_experts // group.size()
    x = torch.randn(local_batch_size, hidden_size, device="cuda", dtype=torch.bfloat16)
    hidden_bytes = get_hidden_bytes(x)
    get_buffer(group, hidden_bytes)
    topk = 2

    expert_weights = torch.randn(
        local_batch_size,
        num_experts,
        dtype=torch.float32,
        device="cuda",
    )
    topk_weights, topk_idx = torch.topk(expert_weights, topk, dim=1)

    # Dispatch
    (
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        num_recv_tokens_per_expert_list,
        handle,
        event,
    ) = dispatch_forward(
        x,
        topk_idx,
        topk_weights,
        num_experts,
    )
    # print(f"rank {rank} recv_x: {recv_x.shape=}")
    recv_topk_global_idx = recv_topk_idx + torch.where(
        recv_topk_idx == -1,
        0,
        rank * local_num_experts,
    )

    # Dispatch naive
    all_x = [torch.empty_like(x) for _ in range(world_size)]
    all_topk_idx = [torch.empty_like(topk_idx) for _ in range(world_size)]
    all_topk_weights = [torch.empty_like(topk_weights) for _ in range(world_size)]

    dist.all_gather(all_x, x)
    dist.all_gather(all_topk_idx, topk_idx)
    dist.all_gather(all_topk_weights, topk_weights)

    all_x = torch.cat(all_x, dim=0)
    all_topk_idx = torch.cat(all_topk_idx, dim=0)
    all_topk_weights = torch.cat(all_topk_weights, dim=0)

    expert_range_start = rank * local_num_experts
    expert_range_end = (rank + 1) * local_num_experts
    recv_i = -1
    expert_inputs_ground_truth = []
    # Verification
    for i in range(batch_size):
        activated_on_this_rank = False
        for j in range(topk):
            if expert_range_start <= all_topk_idx[i, j] < expert_range_end:
                if not activated_on_this_rank:
                    activated_on_this_rank = True
                    recv_i += 1
                    for prev in range(j):
                        # Assert previous tokens are not activated on this rank
                        assert recv_topk_idx[recv_i, prev] == -1
                        assert recv_topk_weights[recv_i, prev] == 0.0, (
                            f"{recv_topk_weights[recv_i, prev]=}"
                        )
                    expert_inputs_ground_truth.append(all_x[i])

                assert (
                    recv_topk_idx[recv_i, j] == all_topk_idx[i, j] - expert_range_start
                )
                assert recv_topk_weights[recv_i, j] == all_topk_weights[i, j]
            else:
                if activated_on_this_rank:
                    assert recv_topk_idx[recv_i, j] == -1
                    assert recv_topk_weights[recv_i, j] == 0.0

    expert_inputs_ground_truth = torch.stack(expert_inputs_ground_truth, dim=0)
    torch.testing.assert_close(expert_inputs_ground_truth, recv_x, atol=0.0, rtol=0.0)
