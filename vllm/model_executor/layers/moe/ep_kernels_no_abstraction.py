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


def combine_forward(
    x: torch.Tensor, handle: tuple, previous_event: EventOverlap | None = None
) -> tuple[torch.Tensor, EventOverlap]:
    global _buffer

    # Do MoE combine
    # For more advanced usages, please refer to the docs of the `combine` function
    combined_x, _, event = _buffer.combine(
        x,
        handle,
        async_finish=True,
        previous_event=previous_event,
        allocate_on_comm_stream=previous_event is not None,
    )

    # For event management, please refer to the docs of the `EventOverlap` class
    return combined_x, event


def run_high_throughput():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    group = dist.group.WORLD
    num_experts = 8
    local_batch_size = 4
    batch_size = local_batch_size * world_size
    hidden_size = 32
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

    # Verification
    expert_range_start = rank * local_num_experts
    expert_range_end = (rank + 1) * local_num_experts
    recv_i = -1
    expert_inputs_ground_truth = []
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
                        assert recv_topk_weights[recv_i, prev] == 0.0
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

    # NOTE: This dispatch result cannot be directly fed into the GEMM kernel, it
    # needs another shuffuling/expansion process to cluster the input to each expert.
    # The results to combine will be reduced on the local rank first.

    # Combine
    results = recv_x
    combined_results, event = combine_forward(results, handle, event)

    for i in range(local_batch_size):
        appear_count = 0.0
        for j in range(world_size):
            expert_range_start = j * local_num_experts
            expert_range_end = (j + 1) * local_num_experts
            if any(
                (expert_range_start <= topk_idx[i]) & (topk_idx[i] < expert_range_end)
            ):
                appear_count += 1.0
        torch.testing.assert_close(combined_results[i], x[i] * appear_count)


# You may call this function at the framework initialization
def low_latency_get_buffer(
    group: dist.ProcessGroup,
    num_max_dispatch_tokens_per_rank: int,
    hidden: int,
    num_experts: int,
) -> Buffer:
    # NOTES: the low-latency mode will consume much more space than the normal mode
    # So we recommend that `num_max_dispatch_tokens_per_rank` (the actual batch size
    # in the decoding engine) should be less than 256
    global _buffer
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank, hidden, group.size(), num_experts
    )

    # Allocate a buffer if not existed or not enough buffer size
    if (
        _buffer is None
        or _buffer.group != group
        or not _buffer.low_latency_mode
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        # NOTES: for the best performance, the QP number **must** be equal to the
        # number of the local experts
        assert num_experts % group.size() == 0
        _buffer = Buffer(
            group,
            0,
            num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_experts // group.size(),
        )
    return _buffer


def low_latency_dispatch(
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    num_max_dispatch_tokens_per_rank: int,
    num_experts: int,
):
    global _buffer

    # Do MoE dispatch, compatible with CUDA graph (but you may restore some buffer
    # status once you replay)
    recv_hidden_states, recv_expert_count, handle, event, hook = (
        _buffer.low_latency_dispatch(
            hidden_states,
            topk_idx,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            async_finish=False,
            return_recv_hook=True,
            use_fp8=False,
        )
    )

    # NOTES: the actual tensor will not be received only if you call `hook()`,
    # it is useful for double-batch overlapping, but **without any SM occupation**
    # If you don't want to overlap, please set `return_recv_hook=False`
    # Later, you can use our GEMM library to do the computation with this specific
    # format
    return recv_hidden_states, recv_expert_count, handle, event, hook


def low_latency_combine(
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    handle: tuple,
):
    global _buffer

    # Do MoE combine, compatible with CUDA graph (but you may restore some buffer
    # status once you replay)
    combined_hidden_states, event_overlap, hook = _buffer.low_latency_combine(
        hidden_states,
        topk_idx,
        topk_weights,
        handle,
        async_finish=False,
        return_recv_hook=True,
    )

    # NOTES: the same behavior as described in the dispatch kernel
    return combined_hidden_states, event_overlap, hook


def run_low_latency():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    group = dist.group.WORLD
    num_experts = 8
    local_batch_size = 4
    batch_size = local_batch_size * world_size
    hidden_size = 2048
    local_num_experts = num_experts // group.size()
    x = torch.randn(local_batch_size, hidden_size, device="cuda", dtype=torch.bfloat16)
    topk = 2
    num_max_dispatch_tokens_per_rank = local_batch_size
    low_latency_get_buffer(
        group, num_max_dispatch_tokens_per_rank, hidden_size, num_experts
    )

    expert_weights = torch.randn(
        local_batch_size,
        num_experts,
        dtype=torch.float32,
        device="cuda",
    )
    topk_weights, topk_idx = torch.topk(expert_weights, topk, dim=1)

    recv_hidden_states, recv_expert_count, handle, event, hook = low_latency_dispatch(
        x, topk_idx, num_max_dispatch_tokens_per_rank, num_experts
    )
    hook()

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

    expert_tok_ids = [[] for _ in range(local_num_experts)]

    expert_range_start = rank * local_num_experts
    for i, cnt in enumerate(recv_expert_count):
        for j in range(cnt):
            for k in range(batch_size):
                if torch.allclose(
                    recv_hidden_states[i, j], all_x[k], rtol=0.0, atol=0.0
                ):
                    expert_tok_ids[i].append(k)
                    assert i + expert_range_start in all_topk_idx[k]
                    break

    assert [len(tok_ids) for tok_ids in expert_tok_ids] == recv_expert_count.tolist()


if __name__ == "__main__":
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
    )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    run_high_throughput()
    run_low_latency()
