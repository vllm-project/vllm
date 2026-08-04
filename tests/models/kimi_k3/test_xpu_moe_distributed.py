# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from multiprocessing.queues import SimpleQueue

import pytest
import torch
import torch.multiprocessing as mp

from tests.models.kimi_k3.test_xpu_moe_reference import _make_config
from tests.utils import get_open_port
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.models.kimi_k3.xpu.linear import KimiMoE
from vllm.platforms import current_platform
from vllm.v1.worker.workspace import init_workspace_manager

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(),
    reason="Distributed KimiMoE regression requires XPU devices",
)


def _parallel_sizes(mode: str) -> tuple[int, int, bool]:
    if mode == "tp1":
        return 1, 1, False
    if mode == "tp2":
        return 2, 1, False
    if mode == "ep2":
        return 2, 1, True
    if mode == "tp2_ep4":
        return 2, 2, True
    raise ValueError(f"Unknown parallel mode: {mode}")


def _distributed_worker(
    rank: int,
    mode: str,
    world_size: int,
    port: int,
    result_queue: SimpleQueue,
) -> None:
    tensor_parallel_size, data_parallel_size, enable_expert_parallel = (
        _parallel_sizes(mode)
    )
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.xpu.set_device(rank)
    device = torch.device("xpu", rank)
    parallel_config = ParallelConfig(
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        enable_expert_parallel=enable_expert_parallel,
    )
    if data_parallel_size > 1:
        parallel_config.data_parallel_rank = rank // tensor_parallel_size
        torch.distributed.init_process_group(backend="xccl")
    vllm_config = VllmConfig(parallel_config=parallel_config)

    try:
        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=tensor_parallel_size,
                rank=rank % tensor_parallel_size,
                distributed_init_method="env://",
                local_rank=rank,
                backend="xccl",
            )
            initialize_model_parallel(tensor_parallel_size)
            init_workspace_manager(device)

            previous_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            try:
                with torch.device(device):
                    moe = KimiMoE(_make_config())
            finally:
                torch.set_default_dtype(previous_dtype)

            for parameter in moe.parameters():
                parameter.data.fill_(0.02)
            routed_experts = moe.experts.routed_experts
            routed_intermediate_size = 32 // moe.experts.moe_config.tp_size
            w13_weight = routed_experts.w13_weight
            w13_partition_size = w13_weight.shape[1] // 2
            w13_weight.data.zero_()
            w13_weight.data[:, :routed_intermediate_size].fill_(0.02)
            w13_weight.data[
                :,
                w13_partition_size : w13_partition_size
                + routed_intermediate_size,
            ].fill_(0.02)
            w2_weight = routed_experts.w2_weight
            w2_weight.data.zero_()
            w2_weight.data[:, :, :routed_intermediate_size].fill_(0.02)
            selected_experts = torch.tensor(
                [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27],
                device=device,
            )
            correction_bias = moe.gate.e_score_correction_bias
            correction_bias.data.zero_()
            correction_bias.data[selected_experts] = torch.linspace(
                1.0,
                0.85,
                selected_experts.numel(),
                dtype=correction_bias.dtype,
                device=device,
            )
            moe.experts.routed_experts.quant_method.process_weights_after_loading(
                moe.experts.routed_experts
            )

            expected_routed_tp_size = (
                1 if enable_expert_parallel else tensor_parallel_size
            )
            expected_routed_ep_size = world_size if enable_expert_parallel else 1
            assert moe.experts.moe_config.tp_size == expected_routed_tp_size
            assert moe.experts.moe_config.ep_size == expected_routed_ep_size

            expert_map = moe.experts.expert_map
            if enable_expert_parallel:
                local_num_experts = 32 // world_size
                local_start = rank * local_num_experts
                expected_map = torch.full(
                    (32,), -1, dtype=torch.int32, device=device
                )
                expected_map[local_start : local_start + local_num_experts] = (
                    torch.arange(local_num_experts, dtype=torch.int32, device=device)
                )
                assert expert_map is not None
                torch.testing.assert_close(expert_map, expected_map, rtol=0, atol=0)
            else:
                assert expert_map is None

            torch.manual_seed(41)
            hidden_states = torch.randn(
                4, 64, dtype=torch.bfloat16, device=device
            )
            torch.distributed.broadcast(hidden_states, src=0)
            router_logits, _ = moe.gate(hidden_states)
            _, topk_ids = moe.experts.router.select_experts(
                hidden_states, router_logits
            )
            if expert_map is None:
                local_routed_slots = topk_ids.numel()
            else:
                local_routed_slots = int(
                    (expert_map[topk_ids.to(torch.int64)] >= 0).sum()
                )
            assert local_routed_slots > 0

            with set_forward_context(
                {}, vllm_config, num_tokens=hidden_states.shape[0]
            ):
                output = moe(hidden_states)
            torch.xpu.synchronize()
            assert bool(torch.isfinite(output).all())

            gathered_outputs = [torch.empty_like(output) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_outputs, output)
            for rank_output in gathered_outputs[1:]:
                torch.testing.assert_close(
                    gathered_outputs[0], rank_output, rtol=0, atol=0
                )
            if rank == 0:
                result_queue.put(output.float().cpu().tolist())
    finally:
        cleanup_dist_env_and_memory()


def _run_parallel_mode(mode: str) -> torch.Tensor:
    tensor_parallel_size, data_parallel_size, _ = _parallel_sizes(mode)
    world_size = tensor_parallel_size * data_parallel_size
    context = mp.get_context("spawn")
    result_queue = context.SimpleQueue()
    mp.spawn(
        _distributed_worker,
        args=(mode, world_size, get_open_port(), result_queue),
        nprocs=world_size,
        join=True,
    )
    return torch.tensor(result_queue.get())


@pytest.mark.parametrize(
    ("mode", "required_devices"),
    [("tp2", 2), ("ep2", 2), ("tp2_ep4", 4)],
)
def test_xpu_kimi_moe_distributed_matches_tp1(
    mode: str, required_devices: int
) -> None:
    if torch.xpu.device_count() < required_devices:
        pytest.skip(f"{mode} requires {required_devices} XPU devices")

    reference = _run_parallel_mode("tp1")
    actual = _run_parallel_mode(mode)
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-3)
