# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.utils import ensure_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.models.kimi_k3.nvidia.ops.ag_gemm import AgGemm
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

_SHAPES = (
    (129, 512),
    (257, 768),
    (1023, 511),
    (1024, 512),
    (8191, 512),
)
_K = 768


def _reference(
    local_input: torch.Tensor,
    weight: torch.Tensor,
    global_M: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    gathered = torch.empty(
        (world_size * local_input.shape[0], local_input.shape[1]),
        dtype=local_input.dtype,
        device=local_input.device,
    )
    dist.all_gather_single(gathered, local_input, group=group)
    return torch.mm(gathered[:global_M], weight.T)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, global_M: int) -> None:
    torch.testing.assert_close(
        actual[:global_M],
        expected,
        rtol=5e-2,
        atol=4.0,
    )


def _worker(local_rank: int, world_size: int, master_port: int) -> None:
    device = torch.device("cuda", local_rank)
    torch.accelerator.set_device_index(device)
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
        }
    )

    init_distributed_environment()
    with ensure_current_vllm_config():
        initialize_model_parallel(tensor_model_parallel_size=world_size)

    tp_group = get_tp_group()
    group = tp_group.device_group
    ag_gemm = AgGemm(max_global_tokens=max(M for M, _ in _SHAPES), hidden_size=_K)

    weight_generator = torch.Generator(device=device)
    input_generator = torch.Generator(device=device)
    weights = {}
    for N in {N for _, N in _SHAPES}:
        weight_generator.manual_seed(1000 + local_rank * 10 + N)
        weights[N] = torch.randn(
            N,
            _K,
            dtype=torch.bfloat16,
            device=device,
            generator=weight_generator,
        )

    # Alternating shapes exercise workspace and signal reuse. Odd global token
    # counts also verify that callers can discard sequence-parallel padding.
    for global_M, N in (*_SHAPES, *_SHAPES[::-1]):
        local_M = (global_M + world_size - 1) // world_size
        input_generator.manual_seed(2000 + local_rank * 10 + global_M + N)
        local_input = torch.randn(
            local_M,
            _K,
            dtype=torch.bfloat16,
            device=device,
            generator=input_generator,
        )
        expected = _reference(local_input, weights[N], global_M, group)
        actual = ag_gemm(local_input, weights[N])
        torch.accelerator.synchronize(device)
        _assert_close(actual, expected, global_M)
        dist.barrier(group=group)

    # Exercise AG-GEMM under CUDA graph capture and repeated replay.
    graph_M, graph_N = 1025, 512
    local_M = (graph_M + world_size - 1) // world_size
    input_generator.manual_seed(3000 + local_rank)
    graph_input = torch.randn(
        local_M,
        _K,
        dtype=torch.bfloat16,
        device=device,
        generator=input_generator,
    )
    graph_expected = _reference(graph_input, weights[graph_N], graph_M, group)

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    for _ in range(3):
        with torch.cuda.stream(capture_stream):
            graph_output = ag_gemm(graph_input, weights[graph_N])
        capture_stream.synchronize()
        dist.barrier(group=group)
    dist.barrier(group=tp_group.cpu_group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_output = ag_gemm(graph_input, weights[graph_N])
    capture_stream.synchronize()
    dist.barrier(group=group)

    for _ in range(3):
        graph.replay()
        torch.accelerator.synchronize(device)
        _assert_close(graph_output, graph_expected, graph_M)
        dist.barrier(group=group)

    dist.barrier(group=tp_group.cpu_group)
    del ag_gemm
    cleanup_dist_env_and_memory()


@pytest.mark.distributed(num_gpus=2)
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="Kimi-K3 AG-GEMM requires SM100",
)
def test_kimi_k3_ag_gemm(monkeypatch: pytest.MonkeyPatch) -> None:
    world_size = 2
    if torch.accelerator.device_count() < world_size:
        pytest.skip("AG-GEMM requires two GPUs")

    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "1")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "1")
    try:
        mp.spawn(
            _worker,
            args=(world_size, get_open_port()),
            nprocs=world_size,
        )
    finally:
        cleanup_dist_env_and_memory()
