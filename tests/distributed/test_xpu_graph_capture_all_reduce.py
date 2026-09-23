# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test: an eager large all-reduce after XPU graph capture must not
leave the caller's stream recording into a command graph.

oneCCL chains each collective on the previous one's completion event. After a
capture that event belongs to the captured graph, and oneCCL's large-message
path (above roughly 4-8 MB) submits with an explicit dependency on it. That
used to pull the caller's queue into the recording, so every later graph
replay failed ("Cannot prepare for replay during capturing stage ...
xpuStreamCaptureStatus: Recording"). GroupCoordinator.graph_capture() now
resets the chain on exit via XpuCommunicator.reset_after_graph_capture().
"""

import pytest
import torch
import torch.multiprocessing as mp

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    graph_capture,
    init_distributed_environment,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.v1.worker.xpu_model_runner import _torch_cuda_wrapper

HIDDEN = 5120
SMALL_ROWS = 40  # 0.4 MB bf16: like a captured decode-size all-reduce
LARGE_ROWS = 2000  # 20 MB bf16: like the embed all-reduce of a 2k-token prefill


def _worker(rank: int, world_size: int, port: str) -> None:
    # graph_capture() uses torch.cuda.*; the XPU model runner aliases those to
    # torch.xpu.* the same way.
    with _torch_cuda_wrapper():
        _run(rank, world_size, port)


def _run(rank: int, world_size: int, port: str) -> None:
    torch.xpu.set_device(rank)
    device = torch.device("xpu", rank)
    # Not tests.utils.init_test_distributed_environment: that uses the default
    # (NCCL) backend, while XPU workers run on current_platform.dist_backend.
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://localhost:{port}",
            local_rank=rank,
            backend=current_platform.dist_backend,
        )
        ensure_model_parallel_initialized(world_size, 1)

    small = torch.ones(SMALL_ROWS, HIDDEN, dtype=torch.bfloat16, device=device)
    large = torch.ones(LARGE_ROWS, HIDDEN, dtype=torch.bfloat16, device=device)

    graph = torch.xpu.XPUGraph()
    with graph_capture(device=device) as ctx:
        tensor_model_parallel_all_reduce(small)  # warm up on the capture stream
        torch.xpu.synchronize()
        with torch.xpu.graph(graph, stream=ctx.stream):
            captured_out = tensor_model_parallel_all_reduce(small)
    torch.xpu.synchronize()

    run_stream = torch.xpu.Stream(device=device)
    with torch.xpu.stream(run_stream):
        # First eager collective after capture, and a large one.
        large_out = tensor_model_parallel_all_reduce(large)
        assert not torch.xpu.is_current_stream_capturing(), (
            "eager all-reduce after graph capture left the stream recording"
        )
        graph.replay()
    torch.xpu.synchronize()

    torch.testing.assert_close(large_out, large * world_size)
    torch.testing.assert_close(captured_out, small * world_size)


@pytest.mark.skipif(
    not current_platform.is_xpu() or torch.xpu.device_count() < 2,
    reason="needs 2 XPU devices",
)
def test_eager_large_all_reduce_after_graph_capture():
    world_size = 2
    mp.spawn(_worker, args=(world_size, str(get_open_port())), nprocs=world_size)
