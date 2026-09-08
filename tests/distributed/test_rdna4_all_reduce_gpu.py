# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import multiprocessing as mp

import pytest
import torch
import torch.distributed as dist

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_rdna4
from vllm.utils.network_utils import get_open_port

from ..utils import multi_gpu_test

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm()
    or not on_rdna4()
    or importlib.util.find_spec("flydsl") is None,
    reason="RDNA4 FlyDSL all-reduce tests require RDNA4 GPUs and FlyDSL",
)


def _worker(rank, world_size, port, element_counts):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    from vllm.distributed.device_communicators.rdna4_all_reduce import (
        RDNA4AllReduce,
    )

    communicator = RDNA4AllReduce(
        group=dist.group.WORLD,
        device=device,
        max_size=max(element_counts) * torch.bfloat16.itemsize,
    )
    try:
        assert not communicator.disabled
        expected = world_size * (world_size + 1) / 2
        for numel in element_counts:
            inp = torch.full((numel,), rank + 1, dtype=torch.bfloat16, device=device)
            out = torch.empty_like(inp)
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize(device)
            dist.barrier()
            with communicator.capture(), torch.cuda.graph(graph):
                result = communicator.custom_all_reduce(inp, out=out)
                assert result is out
            for _ in range(16):
                graph.replay()
            torch.cuda.synchronize(device)
            torch.testing.assert_close(
                out,
                torch.full_like(out, expected),
                rtol=0,
                atol=0,
            )
    finally:
        torch.cuda.synchronize(device)
        dist.barrier()
        communicator.close()
        dist.destroy_process_group()


def _run(world_size, element_counts):
    context = mp.get_context("spawn")
    port = get_open_port()
    processes = [
        context.Process(
            target=_worker,
            args=(rank, world_size, port, element_counts),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=180)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
            pytest.fail("RDNA4 all-reduce worker timed out")
        assert process.exitcode == 0, (
            f"RDNA4 all-reduce worker exited with code {process.exitcode}"
        )


@multi_gpu_test(num_gpus=2)
def test_rdna4_all_reduce_tp2():
    _run(2, (8, 32_768))


@multi_gpu_test(num_gpus=4)
def test_rdna4_all_reduce_tp4():
    _run(4, (8, 65_536, 524_288))


@multi_gpu_test(num_gpus=8)
def test_rdna4_all_reduce_tp8():
    _run(8, (8, 65_536, 196_640))
