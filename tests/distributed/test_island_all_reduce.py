# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing as mp
import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.device_communicators.island_all_reduce import (
    find_two_islands,
)
from vllm.distributed.parallel_state import get_tp_group, graph_capture
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

from ..utils import init_test_distributed_environment

WORLD_SIZE = 8
HIDDEN = 6144
# Exit code of a worker whose GPUs do not use the island all-reduce.
NOT_TWO_ISLANDS = 77
# Rank -> GPU mapping that alternates between the islands of an 8x MI210 node.
INTERLEAVED_GPUS = [0, 4, 1, 5, 2, 6, 3, 7]


def _fully_connected_within_islands(ids: list[int]) -> bool:
    return len({i // 4 for i in ids}) == 1


@pytest.mark.parametrize(
    "ids,connected,expected",
    [
        (
            list(range(8)),
            _fully_connected_within_islands,
            ([0, 1, 2, 3], [4, 5, 6, 7]),
        ),
        # A fully connected node (e.g. 8x MI300X) uses the flat custom all-reduce.
        (list(range(8)), lambda ids: True, None),
        # PCIe-only GPUs have no islands.
        (list(range(8)), lambda ids: len(ids) == 1, None),
        (list(range(4)), _fully_connected_within_islands, None),
        # The TP rank order need not follow the islands.
        (
            INTERLEAVED_GPUS,
            _fully_connected_within_islands,
            ([0, 2, 4, 6], [1, 3, 5, 7]),
        ),
    ],
)
def test_find_two_islands(ids, connected, expected):
    with patch.object(
        current_platform, "is_fully_connected", side_effect=connected, create=True
    ):
        assert find_two_islands(ids) == expected


def _rank_input(rank: int, tokens: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 * seed + rank)
    return (torch.randn(tokens, HIDDEN, generator=g) * (1 + rank)).to(dtype)


def _check(
    out: torch.Tensor | None,
    tokens: int,
    dtype: torch.dtype,
    seed: int,
    world_size: int = WORLD_SIZE,
):
    assert out is not None
    parts = [_rank_input(r, tokens, dtype, seed).float() for r in range(world_size)]
    ref = torch.stack(parts).sum(0).to(out.device)
    err = (out.float() - ref).abs().max() / ref.abs().max()
    assert err < 2e-2, f"tokens={tokens} rel err {err.item():.2e}"


def _island_worker(rank: int, port: int, case: str, dtype_name: str) -> None:
    if case == "custom_off":
        os.environ["VLLM_ROCM_ISLAND_AR_CUSTOM_MAX_BYTES"] = "0"
    from vllm.distributed.device_communicators.island_all_reduce import (
        IslandAllReduce,
    )

    gpu = INTERLEAVED_GPUS[rank] if case == "interleaved" else rank
    device = torch.device(f"cuda:{gpu}")
    torch.accelerator.set_device_index(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD_SIZE,
    )
    iar = IslandAllReduce(dist.group.WORLD, device, list(range(WORLD_SIZE)))
    if iar.disabled:
        raise SystemExit(NOT_TWO_ISLANDS)
    dtype = getattr(torch, dtype_name)

    def check(out: torch.Tensor | None, tokens: int) -> None:
        _check(out, tokens, dtype, 0)

    try:
        if case == "small":
            assert len(iar.ca_stages) == 2
            for tokens in (1, 3, 64):
                check(
                    iar.all_reduce(_rank_input(rank, tokens, dtype, 0).to(device)),
                    tokens,
                )
        elif case == "large":
            assert iar.hierarchical
            for tokens in (256, 1024, 4096):
                check(
                    iar.all_reduce(_rank_input(rank, tokens, dtype, 0).to(device)),
                    tokens,
                )
        elif case == "interleaved":
            # One size per path; the hierarchical one depends on the pairing.
            assert len(iar.ca_stages) == 2 and iar.hierarchical
            for tokens in (3, 256):
                check(
                    iar.all_reduce(_rank_input(rank, tokens, dtype, 0).to(device)),
                    tokens,
                )
        elif case == "custom_off":
            assert not iar.ca_stages and iar.hierarchical
            check(iar.all_reduce(_rank_input(rank, 256, dtype, 0).to(device)), 256)
            assert (
                iar.all_reduce(torch.ones(8, HIDDEN, dtype=dtype, device=device))
                is None
            )
        elif case == "declined":
            # 8194 bytes: not a 16-byte multiple and below the hierarchical
            # threshold, so the caller must fall back.
            odd = torch.ones(4097, dtype=torch.bfloat16, device=device)
            assert iar.all_reduce(odd) is None
        else:
            raise ValueError(case)
    finally:
        iar.destroy()
        dist.destroy_process_group()


def _tp_group_worker(rank: int, port: int) -> None:
    """Backend selection, dispatch and graph capture through the TP group."""
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(WORLD_SIZE, 1, rank, str(port))
    iar = get_tp_group().device_communicator.island_ar_comm
    if iar is None or iar.disabled:
        raise SystemExit(NOT_TWO_ISLANDS)
    dtype = torch.bfloat16
    # One size per path: custom (8 tokens) and hierarchical (256 tokens).
    for tokens in (8, 256):
        static = _rank_input(rank, tokens, dtype, 0).to(device)
        tensor_model_parallel_all_reduce(static)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with (
            graph_capture(device=device) as capture_context,
            torch.cuda.graph(graph, stream=capture_context.stream),
        ):
            graph_out = tensor_model_parallel_all_reduce(static)
        for seed in range(1, 6):
            new = _rank_input(rank, tokens, dtype, seed).to(device)
            static.copy_(new)
            graph.replay()
            torch.accelerator.synchronize()
            _check(graph_out, tokens, dtype, seed)
            # Same kernels in the same order, eager and replayed.
            assert torch.equal(iar.all_reduce(new), graph_out)


def _one_island_worker(rank: int, port: int, n: int) -> None:
    """A TP group of n inside one island: custom all-reduce, one stage."""
    from vllm.distributed.device_communicators.island_all_reduce import (
        IslandAllReduce,
    )

    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    dist.init_process_group(
        backend="gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=n
    )
    iar = IslandAllReduce(dist.group.WORLD, device, list(range(n)))
    if iar.disabled:
        raise SystemExit(NOT_TWO_ISLANDS)
    try:
        assert len(iar.ca_stages) == 1 and not iar.hierarchical
        for tokens in (1, 3, 64):
            x = _rank_input(rank, tokens, dtype, 0).to(device)
            _check(iar.all_reduce(x), tokens, dtype, 0, n)
        # At the size limit the caller falls back to RCCL.
        assert iar.all_reduce(_rank_input(rank, 256, dtype, 0).to(device)) is None

        tokens = 8
        static = _rank_input(rank, tokens, dtype, 0).to(device)
        iar.all_reduce(static)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with iar.capture(), torch.cuda.graph(graph):
            graph_out = iar.all_reduce(static)
        for seed in range(1, 6):
            new = _rank_input(rank, tokens, dtype, seed).to(device)
            static.copy_(new)
            graph.replay()
            torch.accelerator.synchronize()
            _check(graph_out, tokens, dtype, seed, n)
            assert torch.equal(iar.all_reduce(new), graph_out)
    finally:
        iar.destroy()
        dist.destroy_process_group()


def _run_workers(target, *args, world_size: int = WORLD_SIZE) -> None:
    ctx = mp.get_context("spawn")
    port = get_open_port()
    procs = [
        ctx.Process(target=target, args=(rank, port, *args))
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=300)
    codes = [proc.exitcode for proc in procs]
    if all(code == NOT_TWO_ISLANDS for code in codes):
        pytest.skip("GPU topology does not use the island all-reduce")
    assert codes == [0] * world_size, f"worker exit codes: {codes}"


requires_8_rocm_gpus = pytest.mark.skipif(
    not current_platform.is_rocm() or current_platform.device_count() < WORLD_SIZE,
    reason="requires 8 ROCm GPUs",
)


@requires_8_rocm_gpus
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
def test_island_all_reduce_small_tensors_match_fp32_sum(dtype_name):
    _run_workers(_island_worker, "small", dtype_name)


@requires_8_rocm_gpus
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
def test_island_all_reduce_large_tensors_match_fp32_sum(dtype_name):
    _run_workers(_island_worker, "large", dtype_name)


@requires_8_rocm_gpus
def test_island_all_reduce_rank_order_interleaving_islands():
    _run_workers(_island_worker, "interleaved", "bfloat16")


@requires_8_rocm_gpus
def test_island_all_reduce_declines_unsupported_size():
    _run_workers(_island_worker, "declined", "bfloat16")


@requires_8_rocm_gpus
def test_island_all_reduce_tp_group_cudagraph_replay_matches_eager():
    _run_workers(_tp_group_worker)


@requires_8_rocm_gpus
def test_island_all_reduce_custom_path_disabled_by_zero_limit():
    _run_workers(_island_worker, "custom_off", "bfloat16")


@pytest.mark.parametrize("n", [2, 4])
def test_island_all_reduce_single_island_group(n):
    if not current_platform.is_rocm() or current_platform.device_count() < n:
        pytest.skip(f"requires {n} ROCm GPUs")
    _run_workers(_one_island_worker, n, world_size=n)
