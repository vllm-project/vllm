# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing as mp
from unittest.mock import patch

import pytest
import torch

from vllm.config import KVTransferConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.device_communicators.xgmi_island_all_reduce import (
    IslandAllReduce,
    find_two_islands,
)
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    graph_capture,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

from ..utils import init_test_distributed_environment

HIDDEN = 6144
# Exit code of a worker whose GPUs are not two XGMI islands.
NOT_TWO_ISLANDS = 77
# Rank -> GPU mapping that alternates between the islands of an 8x MI210 node.
INTERLEAVED_GPUS = [0, 4, 1, 5, 2, 6, 3, 7]
# Tokens per case, with sizes on both sides of the switch where it matters.
EAGER_TOKENS = {
    "small": (1, 3, 64),
    "large": (256, 1024, 4096),
    "interleaved": (3, 256),
    "host_staged_kv": (8, 256),
    "gpu_direct_kv": (8,),
}
# kv_buffer_device per case; NIXL registers GPU memory for RDMA unless it is "cpu".
KV_BUFFER_DEVICE = {"host_staged_kv": "cpu", "gpu_direct_kv": "cuda"}


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
    world_size: int,
    tokens: int,
    dtype: torch.dtype,
    seed: int,
):
    assert out is not None
    parts = [_rank_input(r, tokens, dtype, seed).float() for r in range(world_size)]
    ref = torch.stack(parts).sum(0).to(out.device)
    err = (out.float() - ref).abs().max() / ref.abs().max()
    assert err < 2e-2, f"tokens={tokens} rel err {err.item():.2e}"


def _worker(rank: int, world_size: int, port: int, case: str, dtype_name: str) -> None:
    """Runs one case through the TP group, as the model does."""
    gpu = INTERLEAVED_GPUS[rank] if case == "interleaved" else rank
    device = torch.device(f"cuda:{gpu}")
    torch.accelerator.set_device_index(device)
    kv_config = None
    if case in KV_BUFFER_DEVICE:
        kv_config = KVTransferConfig(
            kv_connector="NixlConnector",
            kv_role="kv_both",
            kv_buffer_device=KV_BUFFER_DEVICE[case],
        )
    with set_current_vllm_config(VllmConfig(kv_transfer_config=kv_config)):
        init_test_distributed_environment(
            world_size, 1, rank, str(port), local_rank=gpu
        )
    assert get_tp_group().device_communicator.device == device
    iar = get_tp_group().device_communicator.island_ar_comm
    if iar is None or iar.disabled:
        # Skip only if the topology explains it; otherwise the wiring is broken.
        ids = [
            current_platform.visible_device_id_to_physical_device_id(i)
            for i in range(world_size)
        ]
        assert find_two_islands(ids) is None, "two islands found but no island backend"
        raise SystemExit(NOT_TWO_ISLANDS)
    assert iar.ca_island is not None
    assert (iar.nccl_island is not None) == (case != "gpu_direct_kv")
    dtype = getattr(torch, dtype_name)

    if case == "declined":
        # 8194 bytes: not a 16-byte multiple and below the hierarchical
        # threshold, so the communicator must use its default path.
        odd = torch.ones(4097, dtype=dtype, device=device)
        assert iar.all_reduce(odd) is None
        out = tensor_model_parallel_all_reduce(odd)
        assert torch.equal(out, torch.full_like(out, world_size))
    elif case == "graph":
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
                _check(graph_out, world_size, tokens, dtype, seed)
                # Same kernels in the same order, eager and replayed.
                assert torch.equal(iar.all_reduce(new), graph_out)
    else:
        for tokens in EAGER_TOKENS[case]:
            input_ = _rank_input(rank, tokens, dtype, 0).to(device)
            out = tensor_model_parallel_all_reduce(input_)
            _check(out, world_size, tokens, dtype, 0)
            # RCCL also sums correctly, so this is what shows the communicator
            # picked the island path: only the same kernels match bit for bit.
            assert torch.equal(iar.all_reduce(input_), out)
        if case == "gpu_direct_kv":
            # Above the switch the island path declines, and RCCL still sums.
            input_ = _rank_input(rank, 256, dtype, 0).to(device)
            assert iar.all_reduce(input_) is None
            _check(tensor_model_parallel_all_reduce(input_), world_size, 256, dtype, 0)
    # Teardown goes through the communicator and frees the island's groups.
    destroy_model_parallel()
    assert iar.disabled and not iar.groups
    destroy_distributed_environment()


def _has_cdna2_gpus(count: int) -> bool:
    if not current_platform.is_rocm() or current_platform.device_count() < count:
        return False
    from vllm.platforms.rocm import get_cdna_version

    return get_cdna_version() == 2


@pytest.mark.parametrize("world_size", IslandAllReduce.SUPPORTED_WORLD_SIZES)
@pytest.mark.parametrize(
    "case,dtype_name",
    [
        ("small", "bfloat16"),
        ("small", "float16"),
        ("large", "bfloat16"),
        ("large", "float16"),
        ("interleaved", "bfloat16"),
        ("declined", "bfloat16"),
        ("graph", "bfloat16"),
        ("host_staged_kv", "bfloat16"),
        ("gpu_direct_kv", "bfloat16"),
    ],
)
def test_xgmi_island_all_reduce(case, dtype_name, world_size):
    if not _has_cdna2_gpus(world_size):
        pytest.skip(f"requires {world_size} CDNA2 GPUs")
    ctx = mp.get_context("spawn")
    port = get_open_port()
    procs = [
        ctx.Process(target=_worker, args=(rank, world_size, port, case, dtype_name))
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=300)
    codes = [proc.exitcode for proc in procs]
    if all(code == NOT_TWO_ISLANDS for code in codes):
        pytest.skip("GPUs are not two XGMI islands")
    assert codes == [0] * world_size, f"worker exit codes: {codes}"
