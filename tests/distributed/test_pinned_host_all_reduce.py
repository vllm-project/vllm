# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.all_reduce_utils import can_p2p
from vllm.distributed.device_communicators.pinned_host_all_reduce import (
    PinnedHostAllReduce,
)
from vllm.distributed.parallel_state import get_tp_group, graph_capture
from vllm.platforms import current_platform

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

# (tokens, hidden) of decode batches and prefill chunks; in bfloat16 they span
# the LL (<= 16 KiB), flag (<= 48 KiB) and copy-engine (<= 2 MiB) modes.
SHAPES = [(1, 8), (1, 4096), (1, 5120), (4, 4096), (8, 5120), (64, 4096), (100, 5120)]
DTYPES = [torch.bfloat16, torch.float16, torch.float32]


def _inputs(rank, device, seed, dtypes=DTYPES):
    gen = torch.Generator(device=device).manual_seed(2 * seed + rank)
    return [
        torch.randn(shape, generator=gen, device=device).to(dtype)
        for dtype in dtypes
        for shape in SHAPES
    ]


def _nccl(inputs, group):
    refs = [x.clone() for x in inputs]
    for ref in refs:
        dist.all_reduce(ref, group=group)
    return refs


def _delay(device, n=4096, reps=1):
    """Queue GPU work so the next call on this rank starts late."""
    a = torch.ones(n, n, device=device, dtype=torch.bfloat16)
    for _ in range(reps):
        a = a @ a


@ray.remote(num_gpus=1, max_calls=1)
def pinned_host_allreduce(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm import _custom_ops as ops

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.setenv("VLLM_ALLREDUCE_USE_PINNED_HOST", "1")
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        tp_group = get_tp_group()
        nccl_group = tp_group.device_group
        comm = tp_group.device_communicator.pinned_host_ar_comm
        # Selected only without peer access, by the gate's check; elsewhere
        # custom allreduce wins.
        peer_access = can_p2p(rank, range(2))
        assert (comm is not None) != peer_access
        if comm is None:
            comm = PinnedHostAllReduce(tp_group.cpu_group, device)
        assert not comm.disabled

        # Two-rank sums are exact by construction (one fp32 add, rounded once),
        # so every result must equal NCCL's bit for bit.
        inputs = _inputs(rank, device, seed=0)
        assert all(comm.should_use(x) for x in inputs)
        refs = _nccl(inputs, nccl_group)
        # Back to back without host sync, rank 1 delayed now and then so that
        # each mode also runs with the peer one call behind.
        outs = []
        for i, x in enumerate(3 * inputs):
            if rank == 1 and i % 5 == 0:
                _delay(device)
            outs.append(comm.all_reduce(x))
        for out, ref in zip(outs, 3 * refs):
            assert torch.equal(out, ref)

        # CUDA graph: one capture (one call chained on the previous output),
        # replayed with fresh inputs.
        static = _inputs(rank, device, seed=0, dtypes=[torch.bfloat16])
        with graph_capture(device=device) as ctx:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=ctx.stream):
                outs = [comm.all_reduce(x) for x in static]
                outs.append(comm.all_reduce(outs[-1]))
        for seed in (1, 2):
            for x, new in zip(static, _inputs(rank, device, seed, [torch.bfloat16])):
                x.copy_(new)
            refs = _nccl(static, nccl_group)
            refs.append(_nccl([refs[-1]], nccl_group)[0])
            graph.replay()
            for out, ref in zip(outs, refs):
                assert torch.equal(out, ref)

        # Sequence numbers crossing 2^32: a flag, an LL and a copy-engine call,
        # each with the peer late, on buffers and flags never written before.
        wrap = PinnedHostAllReduce(tp_group.cpu_group, device, start_seq=2**32 - 2)
        assert not wrap.disabled
        for numel in (12 * 1024, 2 * 1024, 64 * 1024):
            x = torch.full((numel,), rank + 1.0, device=device, dtype=torch.bfloat16)
            if rank == 1:
                _delay(device, reps=8)
            out = wrap.all_reduce(x)
            assert torch.equal(out, torch.full_like(x, 3.0)), numel
        wrap.close()

        too_large = torch.empty(comm.max_size // 2 + 8, device=device).bfloat16()
        assert not comm.should_use(too_large)
        assert not comm.should_use(inputs[1][:, :-1])  # not a multiple of 16 B
        assert not comm.should_use(inputs[3].t())

        # Inputs at an 8-byte offset are staged, so the path never depends on
        # the address; the op itself still rejects them.
        gen = torch.Generator(device=device).manual_seed(rank)
        for numel in (1024, 16384, 65536):  # LL, flag, copy engine
            x = torch.randn(numel + 4, generator=gen, device=device).bfloat16()[4:]
            assert x.data_ptr() % 16 and comm.should_use(x)
            assert torch.equal(comm.all_reduce(x), _nccl([x], nccl_group)[0]), numel
        with pytest.raises(RuntimeError, match="16-byte aligned"):
            ops.pinned_host_all_reduce(comm._ptr, x, torch.empty_like(x))


@pytest.mark.skipif(
    not current_platform.is_cuda() or torch.accelerator.device_count() < 2,
    reason="Needs two CUDA GPUs.",
)
def test_pinned_host_allreduce(monkeypatch: pytest.MonkeyPatch):
    multi_process_parallel(monkeypatch, 2, 1, pinned_host_allreduce)
