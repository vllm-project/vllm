# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_CUSTOM_ALL_GATHER_SMALL: small all-gathers through the one-shot
CUDA-IPC all-gather must be bit-identical to the NCCL path."""

import pytest
import ray
import torch

from vllm.distributed.device_communicators import custom_all_reduce as car
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.parallel_state import get_tp_group, graph_capture

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)


def _fake_custom_allreduce(**overrides) -> car.CustomAllreduce:
    communicator = car.CustomAllreduce.__new__(car.CustomAllreduce)
    communicator.disabled = False
    communicator.world_size = 4
    communicator.fully_connected = True
    communicator.max_all_gather_size = 2 * 1024 * 1024
    communicator.mnnvl_multicast_ptr = 0
    for name, value in overrides.items():
        setattr(communicator, name, value)
    return communicator


@pytest.mark.parametrize(
    ("overrides", "inp", "expected"),
    [
        ({}, torch.empty(8, 1024, dtype=torch.bfloat16), True),
        ({}, torch.empty(8, 16, dtype=torch.float32), True),
        ({}, torch.empty(4, 1024, dtype=torch.float16), True),
        ({}, torch.empty(16, dtype=torch.int32), False),
        ({}, torch.empty(0, 16, dtype=torch.float32), False),
        ({}, torch.empty(3, dtype=torch.float16), False),  # not 16B multiple
        ({}, torch.empty(16, 32, dtype=torch.float32)[:, :16], False),
        ({}, torch.empty(2 * 1024 * 1024 // 2 + 8, dtype=torch.bfloat16), False),
        ({"disabled": True}, torch.empty(8, 16, dtype=torch.float32), False),
        ({"world_size": 16}, torch.empty(8, 16, dtype=torch.float32), False),
        ({"fully_connected": False}, torch.empty(8, 16, dtype=torch.float32), False),
        # Multicast availability does not matter: the Lamport kernel is never
        # selected by ipc_all_gather.
        ({"mnnvl_multicast_ptr": 1}, torch.empty(8, 16, dtype=torch.float32), True),
    ],
)
def test_should_ipc_all_gather(monkeypatch, overrides, inp, expected):
    monkeypatch.setattr(car.current_platform, "is_cuda", lambda: True)
    communicator = _fake_custom_allreduce(**overrides)
    assert communicator.should_ipc_all_gather(inp) is expected


class _FakeIpcAllGather:
    """Stands in for CustomAllreduce: rank r contributes `inp + 100 * r`."""

    disabled = False

    def __init__(self, world_size: int):
        self.world_size = world_size
        self.calls = 0

    def ipc_all_gather(self, inp: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.cat([inp + 100 * r for r in range(self.world_size)], dim=0)


@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 4, 8)])
def test_custom_all_gather_small_layout(dim, shape):
    world_size = 4
    comm = CudaCommunicator.__new__(CudaCommunicator)
    comm.world_size = world_size
    comm.ca_comm = _FakeIpcAllGather(world_size)
    comm.custom_all_gather_small_max_bytes = 64 * 1024

    inp = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
    d = dim + inp.dim() if dim < 0 else dim
    out = comm._custom_all_gather_small(inp, d)
    expected = torch.cat([inp + 100 * r for r in range(world_size)], dim=d)
    assert out is not None
    assert torch.equal(out, expected)

    comm.custom_all_gather_small_max_bytes = inp.nbytes - 1
    assert comm._custom_all_gather_small(inp, d) is None
    assert comm.ca_comm.calls == 1


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int16 if x.element_size() == 2 else torch.int32)


def _special_values(x: torch.Tensor) -> torch.Tensor:
    flat = x.view(-1)
    flat[0::7] = -0.0
    flat[1::11] = float("nan")
    flat[2::13] = float("inf")
    flat[3::17] = float("-inf")
    flat[4::19] = 1e-40 if x.dtype == torch.float32 else 1e-39
    return x


# (shape, dtype, dim): drafter eh_proj / logits gathers and the 64 B per-row
# argmax candidates, plus one size above the default 64 KiB threshold.
_CASES = [
    ((1, 1024), torch.bfloat16, -1),
    ((6, 1024), torch.bfloat16, -1),
    ((48, 1024), torch.bfloat16, -1),
    ((1, 32768), torch.bfloat16, -1),
    ((1, 16), torch.float32, 0),
    ((8, 16), torch.float32, 0),
    ((8, 32768), torch.bfloat16, -1),  # 512 KiB per rank: NCCL fallback
]


@ray.remote(num_gpus=1, max_calls=1)
def small_all_gather_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.setenv("VLLM_CUSTOM_ALL_GATHER_SMALL", "1")
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        comm = get_tp_group().device_communicator
        assert isinstance(comm, CudaCommunicator)
        assert comm.use_custom_all_gather_small
        ca = comm.ca_comm
        # Without a fully connected custom communicator (e.g. PCIe-only GPUs)
        # everything falls back to NCCL; the comparisons below still hold.
        ipc_available = ca is not None and not ca.disabled and ca.fully_connected

        def gather(x, dim, custom):
            comm.use_custom_all_gather_small = custom
            try:
                return comm.all_gather(x, dim)
            finally:
                comm.use_custom_all_gather_small = True

        gen = torch.Generator(device=device).manual_seed(1234 + rank)
        for shape, dtype, dim in _CASES:
            small = shape[0] * shape[1] * torch.finfo(dtype).bits // 8 <= (
                comm.custom_all_gather_small_max_bytes
            )
            x = torch.randn(shape, generator=gen, device=device).to(dtype)
            if ipc_available:
                assert ca.should_ipc_all_gather(x) or not small
            for inp in (x, _special_values(x.clone())):
                new = gather(inp, dim, custom=True)
                ref = gather(inp, dim, custom=False)
                assert new.shape == ref.shape
                assert torch.equal(_bits(new), _bits(ref)), (shape, dtype, dim)

        # CUDA graph capture and replay with refreshed inputs.
        for shape, dtype, dim in _CASES[:5]:
            static = torch.randn(shape, generator=gen, device=device).to(dtype)
            with graph_capture(device=device) as graph_capture_context:
                torch.accelerator.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                    out = comm.all_gather(static, dim)
            for i in range(20):
                fresh = torch.randn(shape, generator=gen, device=device).to(dtype)
                static.copy_(_special_values(fresh) if i % 4 == 0 else fresh)
                graph.replay()
                ref = gather(static, dim, custom=False)
                assert torch.equal(_bits(out), _bits(ref)), (shape, dtype, dim, i)
        torch.accelerator.synchronize()


@pytest.mark.parametrize("tp_size", [2, 4])
def test_custom_all_gather_small_matches_nccl(monkeypatch: pytest.MonkeyPatch, tp_size):
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    multi_process_parallel(monkeypatch, tp_size, 1, small_all_gather_worker)
