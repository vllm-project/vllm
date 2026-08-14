# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.utils import ensure_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.models.kimi_k3.nvidia.ops.cute_dsl.gemm_rs import GemmRS
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

_SHAPES = (
    (129, 768),
    (257, 768),
    (1023, 4224),
    (1024, 4224),
    (8191, 4224),
)
_N = 512


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    M = x.shape[0]
    padded_M = (M + world_size - 1) // world_size * world_size
    partial = torch.empty(
        (padded_M, weight.shape[0]),
        dtype=x.dtype,
        device=x.device,
    )
    torch.mm(x, weight.T, out=partial[:M])
    if padded_M > M:
        partial[M:].zero_()

    output = torch.empty(
        (padded_M // world_size, weight.shape[0]),
        dtype=x.dtype,
        device=x.device,
    )
    dist.reduce_scatter_single(output, partial, group=group)
    return output


def _assert_valid_rows_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    M: int,
    rank: int,
) -> None:
    rank_start = rank * actual.shape[0]
    valid_rows = min(max(M - rank_start, 0), actual.shape[0])
    torch.testing.assert_close(
        actual[:valid_rows],
        expected[:valid_rows],
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

    group = dist.group.WORLD
    rank = dist.get_rank(group)
    gemm_rs = GemmRS(max_M=max(M for M, _ in _SHAPES), N=_N)

    weight_generator = torch.Generator(device=device)
    input_generator = torch.Generator(device=device)
    weights = {}
    for K in {K for _, K in _SHAPES}:
        weight_generator.manual_seed(1000 + rank * 10 + K)
        weights[K] = torch.randn(
            _N,
            K,
            dtype=torch.bfloat16,
            device=device,
            generator=weight_generator,
        )

    # Alternating shapes exercise producer-flag reuse across different grids,
    # CTA-group choices, and both BN=128 and BN=256 dispatches.
    for M, K in (*_SHAPES, *_SHAPES[::-1]):
        input_generator.manual_seed(2000 + M + K)
        x = torch.randn(
            M,
            K,
            dtype=torch.bfloat16,
            device=device,
            generator=input_generator,
        )
        expected = _reference(x, weights[K], world_size, group)
        actual = gemm_rs(x, weights[K])
        torch.accelerator.synchronize(device)
        _assert_valid_rows_close(actual, expected, M, rank)

    # Exercise GEMM-RS under CUDA graph capture and replay.
    graph_M, graph_K = 1025, 4224
    input_generator.manual_seed(3000)
    graph_x = torch.randn(
        graph_M,
        graph_K,
        dtype=torch.bfloat16,
        device=device,
        generator=input_generator,
    )
    graph_expected = _reference(graph_x, weights[graph_K], world_size, group)
    graph_output = torch.empty_like(graph_expected)

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            graph_output.copy_(gemm_rs(graph_x, weights[graph_K]))
    capture_stream.synchronize()
    dist.barrier(group=group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_output.copy_(gemm_rs(graph_x, weights[graph_K]))
    torch.cuda.current_stream().wait_stream(capture_stream)
    dist.barrier(group=group)

    for _ in range(3):
        graph.replay()
        torch.accelerator.synchronize(device)
        _assert_valid_rows_close(graph_output, graph_expected, graph_M, rank)

    dist.barrier(group=group)
    del gemm_rs
    cleanup_dist_env_and_memory()


@pytest.mark.distributed(num_gpus=2)
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="Kimi-K3 GEMM-RS requires SM100",
)
def test_kimi_k3_gemm_rs(monkeypatch: pytest.MonkeyPatch) -> None:
    world_size = 2
    if torch.accelerator.device_count() < world_size:
        pytest.skip("GEMM-RS requires two GPUs")

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
