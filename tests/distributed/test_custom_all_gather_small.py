# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_CUSTOM_ALL_GATHER_SMALL: small TP all-gathers through
`custom_all_gather` must match the NCCL path, eagerly and under CUDA graphs.

On multicast-capable GPUs `custom_all_gather` uses the MNNVL Lamport kernel,
which reserves the 32-bit word 0x80000000 as its "not written yet" marker and
stores such words as 0, i.e. it turns -0.0 into +0.0. Values are compared with
signed zeros canonicalized; everything else (NaN payloads, inf, subnormals)
must match bit for bit.
"""

import pytest
import ray
import torch

from vllm.distributed.device_communicators import cuda_communicator as cc
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.parallel_state import get_tp_group, graph_capture

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)


@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 4, 8)])
def test_custom_all_gather_small_routing_and_layout(monkeypatch, dim, shape):
    world_size = 4
    calls = []

    def fake_custom_all_gather(inp):
        # Rank r contributes inp + 100 * r, gathered along dim 0.
        calls.append(inp.shape)
        return torch.cat([inp + 100 * r for r in range(world_size)], dim=0)

    def fake_nccl_all_gather(self, inp, dim):
        return "nccl"

    monkeypatch.setattr(cc, "should_nccl_symm_mem_ag_rs", lambda: False)
    monkeypatch.setattr(DeviceCommunicatorBase, "all_gather", fake_nccl_all_gather)
    comm = CudaCommunicator.__new__(CudaCommunicator)
    comm.world_size = world_size
    comm.pynccl_comm = None
    comm.custom_all_gather = fake_custom_all_gather

    inp = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
    d = dim + inp.dim() if dim < 0 else dim
    expected = torch.cat([inp + 100 * r for r in range(world_size)], dim=d)

    comm.custom_all_gather_max_bytes = inp.nbytes
    assert torch.equal(comm.all_gather(inp, dim), expected)
    assert calls == [inp.shape]

    # Above the threshold, or disabled (0), the NCCL path is used.
    for max_bytes in (inp.nbytes - 1, 0):
        comm.custom_all_gather_max_bytes = max_bytes
        assert comm.all_gather(inp, dim) == "nccl"
    assert len(calls) == 1

    # custom_all_gather declining the input falls back to NCCL as well.
    comm.custom_all_gather_max_bytes = inp.nbytes
    comm.custom_all_gather = lambda inp: None
    assert comm.all_gather(inp, dim) == "nccl"


def _canonical_bits(x: torch.Tensor) -> torch.Tensor:
    x = torch.where(x == 0, torch.zeros_like(x), x)
    return x.view(torch.int16 if x.element_size() == 2 else torch.int32)


def _special_values(x: torch.Tensor) -> torch.Tensor:
    flat = x.view(-1)
    flat[0::7] = -0.0
    flat[1::11] = float("nan")
    flat[2::13] = float("inf")
    flat[3::17] = float("-inf")
    flat[4::19] = 1e-40 if x.dtype == torch.float32 else 1e-39
    return x


# (shape, dtype, dim): drafter eh_proj / logits gathers and the per-row argmax
# candidates, plus one size above the default 64 KiB threshold.
_CASES = [
    ((1, 1024), torch.bfloat16, -1),
    ((6, 1024), torch.bfloat16, -1),
    ((48, 1024), torch.bfloat16, -1),
    ((1, 32768), torch.bfloat16, -1),
    ((1, 16), torch.float32, 0),
    ((8, 16), torch.float32, 0),
    ((8, 2), torch.float32, -1),
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
        max_bytes = comm.custom_all_gather_max_bytes
        assert max_bytes > 0

        def gather(x, dim, custom):
            comm.custom_all_gather_max_bytes = max_bytes if custom else 0
            try:
                return comm.all_gather(x, dim)
            finally:
                comm.custom_all_gather_max_bytes = max_bytes

        gen = torch.Generator(device=device).manual_seed(1234 + rank)
        for shape, dtype, dim in _CASES:
            x = torch.randn(shape, generator=gen, device=device).to(dtype)
            for inp in (x, _special_values(x.clone())):
                new = gather(inp, dim, custom=True)
                ref = gather(inp, dim, custom=False)
                assert new.shape == ref.shape
                assert torch.equal(_canonical_bits(new), _canonical_bits(ref)), (
                    shape,
                    dtype,
                    dim,
                )

        # CUDA graph capture and replay with refreshed inputs.
        for shape, dtype, dim in _CASES[:7]:
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
                assert torch.equal(_canonical_bits(out), _canonical_bits(ref)), (
                    shape,
                    dtype,
                    dim,
                    i,
                )
        torch.accelerator.synchronize()


@pytest.mark.parametrize("tp_size", [2, 4])
def test_custom_all_gather_small_matches_nccl(monkeypatch: pytest.MonkeyPatch, tp_size):
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    multi_process_parallel(monkeypatch, tp_size, 1, small_all_gather_worker)
