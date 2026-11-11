# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stream identity under the Confidential Computing copy patches.

`torch.cuda.Stream()` hands out streams round-robin from a fixed pool of 32
per device and priority. The CC patches key the event redirect on the raw
handle of the D2H copy streams, so a pool stream taken later (e.g. by
`graph_capture()` or an aux stream) must never share that handle: a capture
would fail and a cross-stream join outside capture would be silently dropped.
`new_d2h_stream()` therefore creates the copy streams outside the pool.
"""

import pytest
import torch

from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.platforms import current_platform
from vllm.v1 import conf_compute_utils
from vllm.v1.worker.gpu.async_utils import new_d2h_stream

# Confidential Computing is NVIDIA-only: ROCm never takes these paths.
pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and torch.cuda.is_available()),
    reason="needs NVIDIA CUDA",
)

POOL_WRAP = 64  # > 2 x the 32-stream pool, enough to see every pool handle

# Everything conf_compute_utils.install() replaces.
_PATCHED_BY_INSTALL = [
    (torch.Tensor, "copy_"),
    (torch.Tensor, "to"),
    (torch.Tensor, "cuda"),
    (torch.cuda.Event, "record"),
    (torch.cuda.Event, "synchronize"),
    (torch.cuda.Event, "query"),
    (torch, "Event"),
    (torch.cuda.Stream, "synchronize"),
    (torch.cuda.Stream, "wait_stream"),
    (torch.cuda.Stream, "query"),
    (torch.cuda, "synchronize"),
    (torch.accelerator, "synchronize"),
]


def _set_cc(monkeypatch, enabled: bool) -> None:
    monkeypatch.setattr(
        type(current_platform),
        "is_confidential_compute",
        classmethod(lambda cls: enabled),
    )


@pytest.fixture
def d2h_stream(monkeypatch):
    _set_cc(monkeypatch, True)
    # Record the originals so the patches don't outlive the test: the other tests
    # in this pytest process must not run with the CC copy paths installed.
    for obj, name in _PATCHED_BY_INSTALL:
        monkeypatch.setattr(obj, name, getattr(obj, name))
    monkeypatch.setattr(conf_compute_utils, "_d2h_executor", None)
    monkeypatch.setattr(conf_compute_utils, "_d2h_streams", set())
    monkeypatch.setattr(conf_compute_utils, "_producer_of", {})
    conf_compute_utils.install()
    yield new_d2h_stream(torch.device("cuda", torch.accelerator.current_device_index()))
    assert conf_compute_utils._d2h_executor is not None
    conf_compute_utils._d2h_executor.shutdown()


def _pool_streams(device: torch.device) -> list[torch.cuda.Stream]:
    return [torch.cuda.Stream(device) for _ in range(POOL_WRAP)]


def test_new_d2h_stream_is_outside_the_pool(d2h_stream):
    pool = {s.cuda_stream for s in _pool_streams(d2h_stream.device)}
    assert d2h_stream.cuda_stream not in pool
    assert d2h_stream.cuda_stream in conf_compute_utils._d2h_streams


def test_new_d2h_stream_without_cc_is_a_pool_stream(monkeypatch):
    _set_cc(monkeypatch, False)
    stream = new_d2h_stream(torch.accelerator.current_device_index())
    assert not isinstance(stream, torch.cuda.ExternalStream)
    assert stream.cuda_stream not in conf_compute_utils._d2h_streams


def test_capture_on_pool_stream_after_d2h_stream(d2h_stream):
    """A graph captured on any pool stream can fork/join through an event, as
    the speculator's capture does, however many pool streams were taken after
    the D2H stream was created."""
    device = d2h_stream.device
    capture_stream = _pool_streams(device)[-1]
    x = torch.ones(1 << 16, device=device)
    side = torch.cuda.Stream(device)
    graph = torch.cuda.CUDAGraph()

    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with (
        torch.cuda.stream(capture_stream),
        torch.cuda.graph(graph, stream=capture_stream),
    ):
        y = x * 2
        fork = torch.cuda.Event()
        fork.record()
        side.wait_event(fork)
        with torch.cuda.stream(side):
            z = y + 1
        capture_stream.wait_stream(side)
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(z, torch.full_like(x, 3))


def test_join_on_pool_stream_after_d2h_stream(d2h_stream):
    """`cur.wait_stream(aux)` must order `cur` after work on `aux` for any
    pool stream taken after the D2H stream was created."""
    device = d2h_stream.device
    aux = _pool_streams(device)[-1]
    cur = torch.cuda.current_stream(device)
    aux.wait_stream(cur)
    with torch.cuda.stream(aux):
        torch.cuda._sleep(500_000_000)
    cur.wait_stream(aux)
    done = torch.cuda.Event()
    done.record(cur)
    done.synchronize()
    assert aux.query(), "compute stream ran ahead of the aux stream it joined"
    torch.accelerator.synchronize()


def test_exported_functions_resolve_on_rocm():
    """`CudaRTLibrary()` resolves all of these on every platform, so one without
    a HIP name would break the library on ROCm."""
    exported = {f.name for f in CudaRTLibrary.exported_functions}
    assert exported <= CudaRTLibrary.cuda_to_hip_mapping.keys()


def test_inplace_target_without_cc_leaves_streams_alone(monkeypatch):
    """Non-CC callers, e.g. a NonUvaBuffer, must not reach the staging stream."""
    _set_cc(monkeypatch, False)
    monkeypatch.setattr(
        conf_compute_utils,
        "_staging_stream",
        lambda _: pytest.fail("took the staging stream without CC"),
    )
    target = torch.zeros(1, device=torch.accelerator.current_accelerator())
    conf_compute_utils.register_inplace_h2d_target(target)
    assert (
        target.untyped_storage().data_ptr()
        not in conf_compute_utils._inplace_h2d_storages
    )
