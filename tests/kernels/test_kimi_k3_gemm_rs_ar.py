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
    all_reduce: bool,
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

    if all_reduce:
        dist.all_reduce(partial, group=group)
        return partial[:M]

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
    all_reduce: bool,
) -> None:
    if all_reduce:
        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=4.0)
        return
    rank_start = rank * actual.shape[0]
    valid_rows = min(max(M - rank_start, 0), actual.shape[0])
    torch.testing.assert_close(
        actual[:valid_rows],
        expected[:valid_rows],
        rtol=5e-2,
        atol=4.0,
    )


def _run_mode(
    *,
    all_reduce: bool,
    device: torch.device,
    group: dist.ProcessGroup,
    rank: int,
    world_size: int,
    weights: dict[int, torch.Tensor],
) -> None:
    # cute_dsl is unavailable off CUDA, so import it only inside the GPU worker.
    from vllm.models.kimi_k3.nvidia.ops.cute_dsl.gemm_rs_ar import GemmRsAr

    gemm_rs_ar = GemmRsAr(
        max_M=max(M for M, _ in _SHAPES),
        N=_N,
        all_reduce=all_reduce,
    )
    input_generator = torch.Generator(device=device)

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
        expected = _reference(x, weights[K], world_size, group, all_reduce)
        actual = gemm_rs_ar(x, weights[K])
        torch.accelerator.synchronize(device)
        _assert_valid_rows_close(actual, expected, M, rank, all_reduce)

    if all_reduce:
        # AR outputs must remain valid after the symmetric workspace is reused.
        lifetime_M, lifetime_K = 257, 768
        input_generator.manual_seed(2500)
        lifetime_x = torch.randn(
            lifetime_M,
            lifetime_K,
            dtype=torch.bfloat16,
            device=device,
            generator=input_generator,
        )
        first_output = gemm_rs_ar(lifetime_x, weights[lifetime_K])
        first_snapshot = first_output.clone()
        second_output = gemm_rs_ar(-lifetime_x, weights[lifetime_K])
        torch.accelerator.synchronize(device)
        assert first_output.data_ptr() != second_output.data_ptr()
        torch.testing.assert_close(first_output, first_snapshot, rtol=0, atol=0)

    graph_M, graph_K = 1025, 4224
    input_generator.manual_seed(3000)
    graph_x = torch.randn(
        graph_M,
        graph_K,
        dtype=torch.bfloat16,
        device=device,
        generator=input_generator,
    )
    graph_expected = _reference(
        graph_x,
        weights[graph_K],
        world_size,
        group,
        all_reduce,
    )
    graph_output = torch.empty_like(graph_expected)

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            graph_output.copy_(gemm_rs_ar(graph_x, weights[graph_K]))
    capture_stream.synchronize()
    dist.barrier(group=group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_output.copy_(gemm_rs_ar(graph_x, weights[graph_K]))
    torch.cuda.current_stream().wait_stream(capture_stream)
    dist.barrier(group=group)

    for _ in range(3):
        graph.replay()
        torch.accelerator.synchronize(device)
        _assert_valid_rows_close(
            graph_output,
            graph_expected,
            graph_M,
            rank,
            all_reduce,
        )

    dist.barrier(group=group)
    del gemm_rs_ar, capture_stream, graph, graph_output


def _worker(local_rank: int, world_size: int, master_port: int) -> None:
    # This module pulls in cute_dsl, which is unavailable off CUDA, so importing
    # it at module level would fail collection. Import it where it is used, as
    # `kda.py` does.

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

    weight_generator = torch.Generator(device=device)
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

    # Production binds one mode per worker; exercise both mode-bound instances
    # sequentially in this test without implying that both are initialized.
    for all_reduce in (False, True):
        _run_mode(
            all_reduce=all_reduce,
            device=device,
            group=group,
            rank=rank,
            world_size=world_size,
            weights=weights,
        )
    cleanup_dist_env_and_memory()


@pytest.mark.distributed(num_gpus=2)
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="Kimi-K3 GEMM-RS/AR requires SM100",
)
def test_kimi_k3_gemm_rs_ar(monkeypatch: pytest.MonkeyPatch) -> None:
    world_size = 2
    if torch.accelerator.device_count() < world_size:
        pytest.skip("GEMM-RS/AR requires two GPUs")

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
