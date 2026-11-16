# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IPC metadata, ACK ownership, and failure isolation without requiring CUDA.

The two optional CUDA cases additionally check cross-process GPU tensor copies.
"""

import contextlib
import multiprocessing
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
import torch

from vllm.v1.spec_decode.remote import cuda_ipc as ipc

_UUID = "GPU-00000000-0000-0000-0000-000000000001"


@pytest.mark.parametrize("raw_uuid", [_UUID, _UUID.removeprefix("GPU-")])
def test_device_uuid_normalizes_pytorch_and_nvml_formats(raw_uuid):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_device_properties=lambda device: SimpleNamespace(uuid=raw_uuid)
        )
    )
    assert ipc._device_uuid(fake_torch, "cuda:0") == _UUID


def _descriptor(**changes):
    values = dict(
        buffer_id=1,
        generation=1,
        device_uuid=_UUID,
        memory_handle=b"m" * 64,
        ready_event_handle=b"e" * 64,
        nbytes=32,
        capacity_bytes=1024,
        offset_bytes=256,
    )
    values.update(changes)
    return ipc.CudaIpcDescriptor(**values)


class _Tensor:
    is_cuda = True
    device = torch.device("cuda:0")
    dtype = torch.float32

    def __init__(self, size=8, ptr=4096):
        self.size, self.ptr = size, ptr

    def numel(self):
        return self.size

    def element_size(self):
        return 4

    def contiguous(self):
        return self

    def data_ptr(self):
        return self.ptr

    def record_stream(self, stream):
        pass


class _Runtime:
    def __init__(self):
        self.calls = []
        self.next_event = 10
        self.fail_wait = False

    def malloc(self, size):
        self.calls.append(("malloc", size))
        return 65536

    def free(self, ptr):
        self.calls.append(("free", ptr))

    def memory_handle(self, ptr):
        return b"m" * 64

    def open_memory(self, handle):
        self.calls.append(("open_memory", handle))
        return 131072

    def close_memory(self, ptr):
        self.calls.append(("close_memory", ptr))

    def event(self, *, interprocess=False):
        self.next_event += 1
        self.calls.append(("event", self.next_event, interprocess))
        return self.next_event

    def event_handle(self, event):
        return event.to_bytes(64, "little")

    def open_event(self, handle):
        self.calls.append(("open_event", handle))
        return 100

    def destroy_event(self, event):
        self.calls.append(("destroy_event", event))

    def record(self, event, stream):
        self.calls.append(("record", event, stream))

    def wait(self, event, stream):
        self.calls.append(("wait", event, stream))

    def copy(self, dst, src, size, stream):
        self.calls.append(("copy", dst, src, size, stream))

    def synchronize(self, event, timeout):
        self.calls.append(("synchronize", event, timeout))
        if self.fail_wait:
            raise ipc.CudaIpcError("copy completion timed out")


@pytest.fixture
def runtime(monkeypatch):
    runtime = _Runtime()
    monkeypatch.setattr(ipc, "_CudaRuntime", lambda: runtime)
    monkeypatch.setattr(ipc, "_device_uuid", lambda *_: _UUID)
    monkeypatch.setattr(
        torch.accelerator, "device_index", lambda *_: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda *_: SimpleNamespace(cuda_stream=3)
    )
    monkeypatch.setattr(torch, "empty", lambda *_, **__: _Tensor())
    monkeypatch.setattr(ipc, "_QUARANTINE", [])
    return runtime


@pytest.mark.parametrize(
    "changes",
    [
        {"nbytes": 31},
        {"capacity_bytes": 16},
        {"offset_bytes": -1},
        {"offset_bytes": 1000},
        {"memory_handle": b"short"},
        {"ready_event_handle": b"short"},
        {"device_uuid": "cuda:0"},
        {"generation": 0},
        {"buffer_id": 0},
    ],
)
def test_descriptor_rejects_invalid_ranges_and_opaque_handles(changes):
    with pytest.raises(ipc.CudaIpcError):
        ipc.validate_descriptor(_descriptor(**changes), "float32", (2, 4))


def test_descriptor_wire_rejects_unknown_fields_without_unpickling():
    encoded = msgspec.msgpack.encode(
        {**msgspec.to_builtins(_descriptor()), "code": "bad"}
    )
    with pytest.raises(msgspec.ValidationError):
        msgspec.msgpack.decode(encoded, type=ipc.CudaIpcDescriptor)


@pytest.mark.parametrize("operation", ["copy", "record"])
def test_first_export_failure_keeps_uncertain_storage_quarantined(
    runtime, monkeypatch, operation
):
    exporter = ipc.CudaIpcExporter("cuda:0", max_bytes=1024)

    def fail(*args):
        raise ipc.CudaIpcError("injected export failure")

    monkeypatch.setattr(runtime, operation, fail)
    with pytest.raises(ipc.CudaIpcError, match="injected export failure"):
        exporter.export(_Tensor())
    assert exporter.closed
    assert len(ipc._QUARANTINE) == 1
    calls = runtime.calls.copy()
    exporter.close()
    assert runtime.calls == calls
    assert not any(call[0] in ("free", "destroy_event") for call in calls)


def test_arena_is_reserved_before_first_export_and_reused_after_ack(runtime):
    exporter = ipc.CudaIpcExporter("cuda:0", max_bytes=1024)
    assert runtime.calls == [("malloc", 1024)]
    assert exporter.reserved_bytes == 1024
    first = exporter.export(_Tensor())
    exporter.release(first)
    second = exporter.export(_Tensor(size=16))
    assert first.memory_handle == second.memory_handle
    assert first.ready_event_handle == second.ready_event_handle
    assert second.generation == first.generation + 1
    with pytest.raises(ipc.CudaIpcError, match="stale"):
        exporter.release(first)
    assert len([c for c in runtime.calls if c[0] == "malloc"]) == 1
    exporter.release(second)
    exporter.close(peer_closed=True)
    assert runtime.calls[-1] == ("free", 65536)


def test_export_registration_failure_releases_unpublished_event(runtime):
    class NoGrowth(list):
        def append(self, value):
            raise MemoryError("export registration failed")

    exporter = ipc.CudaIpcExporter("cuda:0", max_bytes=1024)
    exporter.buffers = NoGrowth()
    with pytest.raises(MemoryError, match="registration failed"):
        exporter.export(_Tensor())
    assert ("destroy_event", 11) in runtime.calls
    assert not any(call[0] == "copy" for call in runtime.calls)
    assert not exporter.buffers and not ipc._QUARANTINE
    exporter.buffers = []
    lease = exporter.export(_Tensor())
    exporter.release(lease)
    exporter.close(peer_closed=True)
    assert runtime.calls.count(("destroy_event", 11)) == 1
    assert ("destroy_event", 12) in runtime.calls


@pytest.mark.parametrize("failure", ["record", "registration"])
def test_import_registration_failure_releases_unpublished_event(
    runtime, monkeypatch, failure
):
    class NoGrowth(dict):
        def __setitem__(self, key, value):
            raise MemoryError("import registration failed")

    def no_record(*args):
        raise MemoryError("import record failed")

    receiver = ipc.CudaIpcReceiver("cuda:0", max_bytes=1024)
    with monkeypatch.context() as patch:
        if failure == "record":
            patch.setattr(ipc, "_ImportBuffer", no_record)
        else:
            patch.setattr(receiver, "buffers", NoGrowth())
        with pytest.raises(MemoryError, match="import .* failed"):
            receiver.receive(_descriptor(), "float32", (2, 4))
    assert runtime.calls.count(("destroy_event", 100)) == 1
    assert not any(call[0] == "copy" for call in runtime.calls)
    assert not receiver.buffers and not ipc._QUARANTINE
    receiver.receive(_descriptor(), "float32", (2, 4))
    receiver.close()
    # The fake runtime returns 100 again for the successfully reopened event.
    assert runtime.calls.count(("destroy_event", 100)) == 2
    assert ("close_memory", 131072) in runtime.calls


def test_outstanding_leases_have_disjoint_ranges_and_bounded_storage(runtime):
    exporter = ipc.CudaIpcExporter("cuda:0", max_bytes=512, max_buffers=2)
    first = exporter.export(_Tensor())
    second = exporter.export(_Tensor())
    assert first.offset_bytes + first.nbytes <= second.offset_bytes
    with pytest.raises(ipc.CudaIpcError, match="exhausted"):
        exporter.export(_Tensor())
    exporter.release(first)
    third = exporter.export(_Tensor())
    assert third.offset_bytes == first.offset_bytes
    exporter.close()
    assert not any(c[0] == "free" for c in runtime.calls)


def test_even_acknowledged_arena_survives_disconnect_until_peer_closed(runtime):
    exporter = ipc.CudaIpcExporter("cuda:0", max_bytes=1024)
    descriptor = exporter.export(_Tensor())
    exporter.release(descriptor)
    exporter.close()
    assert len(ipc._QUARANTINE) == 1
    assert not any(c[0] == "free" for c in runtime.calls)
    with pytest.raises(ipc.CudaIpcError, match="closed"):
        exporter.export(_Tensor())


@pytest.mark.parametrize("fail_operation", ["destroy_event", "free"])
def test_exporter_cleanup_attempts_all_resources_and_quarantines_failures(
    runtime, monkeypatch, fail_operation
):
    exporter = ipc.CudaIpcExporter("cuda:0", max_bytes=1024)
    first = exporter.export(_Tensor())
    second = exporter.export(_Tensor())
    exporter.release(first)
    exporter.release(second)
    events = [buffer.event for buffer in exporter.buffers]
    original = getattr(runtime, fail_operation)

    def fail_once(handle):
        original(handle)
        if fail_operation == "free" or handle == events[0]:
            raise ipc.CudaIpcError("injected release failure")

    monkeypatch.setattr(runtime, fail_operation, fail_once)
    runtime.calls.clear()
    with pytest.raises(ipc.CudaIpcError, match="injected release failure"):
        exporter.close(peer_closed=True)
    assert runtime.calls == [
        *(("destroy_event", event) for event in events),
        ("free", 65536),
    ]
    assert exporter.closed
    failed_handle = events[0] if fail_operation == "destroy_event" else 65536
    assert [
        (runtime, exporter.device, [(fail_operation, failed_handle)])
    ] == ipc._QUARANTINE
    # CUDA release may have partially succeeded; never retry an uncertain handle.
    calls = runtime.calls.copy()
    exporter.close(peer_closed=True)
    assert runtime.calls == calls


@pytest.mark.parametrize("fail_operation", ["destroy_event", "close_memory"])
def test_receiver_cleanup_attempts_all_resources_and_quarantines_failures(
    runtime, monkeypatch, fail_operation
):
    receiver = ipc.CudaIpcReceiver("cuda:0", max_bytes=1024)
    receiver.receive(_descriptor(), "float32", (2, 4))
    original = getattr(runtime, fail_operation)

    def fail_once(handle):
        original(handle)
        if fail_operation == "close_memory" or handle == 100:
            raise ipc.CudaIpcError("injected release failure")

    monkeypatch.setattr(runtime, fail_operation, fail_once)
    runtime.calls.clear()
    with pytest.raises(ipc.CudaIpcError, match="injected release failure"):
        receiver.close()
    assert runtime.calls == [
        ("destroy_event", 100),
        ("close_memory", 131072),
        ("destroy_event", receiver.done_event),
    ]
    assert receiver.closed
    failed_handle = 100 if fail_operation == "destroy_event" else 131072
    assert [
        (runtime, receiver.device, [(fail_operation, failed_handle)])
    ] == ipc._QUARANTINE
    calls = runtime.calls.copy()
    receiver.close()
    assert runtime.calls == calls


def test_receiver_cleanup_reports_multiple_failures_after_attempting_every_release(
    runtime, monkeypatch
):
    receiver = ipc.CudaIpcReceiver("cuda:0", max_bytes=1024)
    receiver.receive(_descriptor(), "float32", (2, 4))

    def fail_event(event):
        runtime.calls.append(("destroy_event", event))
        raise ipc.CudaIpcError("event release failed")

    def fail_memory(ptr):
        runtime.calls.append(("close_memory", ptr))
        raise ipc.CudaIpcError("memory release failed")

    monkeypatch.setattr(runtime, "destroy_event", fail_event)
    monkeypatch.setattr(runtime, "close_memory", fail_memory)
    runtime.calls.clear()
    with pytest.raises(ipc.CudaIpcError) as error:
        receiver.close()
    assert "event release failed" in str(error.value)
    assert "memory release failed" in str(error.value)
    assert runtime.calls == [
        ("destroy_event", 100),
        ("close_memory", 131072),
        ("destroy_event", receiver.done_event),
    ]
    assert [(runtime, receiver.device, runtime.calls)] == ipc._QUARANTINE


@pytest.mark.parametrize("owner", ["exporter", "receiver"])
def test_cleanup_device_context_failure_quarantines_unreleased_resources(
    runtime, monkeypatch, owner
):
    pool: ipc.CudaIpcExporter | ipc.CudaIpcReceiver
    if owner == "exporter":
        pool = ipc.CudaIpcExporter("cuda:0", max_bytes=1024)
    else:
        pool = ipc.CudaIpcReceiver("cuda:0", max_bytes=1024)
        pool.receive(_descriptor(), "float32", (2, 4))

    @contextlib.contextmanager
    def unavailable_device(*_):
        raise ipc.CudaIpcError("device unavailable during close")
        yield

    monkeypatch.setattr(torch.accelerator, "device_index", unavailable_device)
    runtime.calls.clear()
    with pytest.raises(ipc.CudaIpcError, match="device unavailable"):
        pool.close()
    assert runtime.calls == []
    assert pool.closed
    assert len(ipc._QUARANTINE) == 1


def test_receive_copies_into_independent_storage_and_reuses_imports(runtime):
    receiver = ipc.CudaIpcReceiver("cuda:0", max_bytes=1024)
    first = receiver.receive(_descriptor(), "float32", (2, 4))
    second = receiver.receive(
        _descriptor(generation=2, offset_bytes=0), "float32", (2, 4)
    )
    assert first is not second
    assert len([c for c in runtime.calls if c[0] == "open_memory"]) == 1
    assert len([c for c in runtime.calls if c[0] == "open_event"]) == 1
    copies = [c for c in runtime.calls if c[0] == "copy"]
    assert copies == [
        ("copy", 4096, 131072 + 256, 32, 3),
        ("copy", 4096, 131072, 32, 3),
    ]
    names = [c[0] for c in runtime.calls]
    assert names.index("wait") < names.index("copy") < names.index("synchronize")
    receiver.close()
    assert ("close_memory", 131072) in runtime.calls


def test_receiver_rejects_replayed_leases_and_changed_arena(runtime):
    receiver = ipc.CudaIpcReceiver("cuda:0", max_bytes=1024)
    receiver.receive(_descriptor(), "float32", (2, 4))
    with pytest.raises(ipc.CudaIpcError, match="replayed"):
        receiver.receive(_descriptor(), "float32", (2, 4))
    with pytest.raises(ipc.CudaIpcError, match="arena identity"):
        receiver.receive(
            _descriptor(generation=2, memory_handle=b"x" * 64), "float32", (2, 4)
        )
    receiver.close()


def test_receiver_timeout_quarantines_inflight_copy_and_never_unmaps(runtime):
    receiver = ipc.CudaIpcReceiver("cuda:0", max_bytes=1024)
    runtime.fail_wait = True
    with pytest.raises(ipc.CudaIpcError, match="timed out"):
        receiver.receive(_descriptor(), "float32", (2, 4))
    receiver.close()
    assert len(ipc._QUARANTINE) == 1
    assert not any(c[0] in ("close_memory", "destroy_event") for c in runtime.calls)


def test_receiver_quarantine_after_fatal_context_makes_no_cuda_calls(runtime):
    receiver = ipc.CudaIpcReceiver("cuda:0", max_bytes=1024)
    receiver.receive(_descriptor(), "float32", (2, 4))
    buffers = receiver.buffers
    calls = list(runtime.calls)
    receiver.quarantine()
    receiver.close()
    assert receiver.closed and not receiver.buffers
    assert runtime.calls == calls
    assert len(ipc._QUARANTINE) == 1
    assert ipc._QUARANTINE[0][3] is buffers


@pytest.mark.parametrize("code", [226, 700, 710, 810])
@pytest.mark.parametrize("operation", ["call", "query"])
def test_runtime_errors_preserve_fatal_status(monkeypatch, code, operation):
    from vllm.v1.spec_decode.remote.errors import cuda_context_fatal

    library = Mock()
    library.cudaEventDestroy.return_value = code
    library.cudaEventQuery.return_value = code
    library.cudaGetErrorString.return_value = b"opaque vendor diagnostic"
    monkeypatch.setenv("VLLM_CUDART_SO_PATH", "libcudart-test.so")
    monkeypatch.setattr(ipc.ctypes, "CDLL", lambda _: library)
    api = ipc._CudaRuntime()
    with pytest.raises(ipc.CudaIpcError) as caught:
        if operation == "call":
            api.destroy_event(10)
        else:
            api.synchronize(10, 0.1)
    assert caught.value.cuda_error_code == code
    assert cuda_context_fatal(caught.value)


def test_fatal_cleanup_stops_at_first_failure_and_retains_unattempted_resources(
    runtime, monkeypatch
):
    error = ipc.CudaIpcError("opaque CUDA failure", cuda_error_code=810)

    def destroy(handle):
        runtime.calls.append(("destroy_event", handle))
        raise error

    monkeypatch.setattr(runtime, "destroy_event", destroy)
    resources = [("destroy_event", 10), ("destroy_event", 11), ("free", 4096)]
    with pytest.raises(ipc.CudaIpcError) as caught:
        ipc._close_resources(torch, torch.device("cuda:0"), runtime, resources)
    assert caught.value is error
    assert runtime.calls == [("destroy_event", 10)]
    assert ipc._QUARANTINE[0][2] == resources


def test_receiver_uses_remaining_budget_and_expiry_before_submit_is_safe(
    runtime, monkeypatch
):
    now = [10.0]
    monkeypatch.setattr(ipc.time, "monotonic", lambda: now[0])
    receiver = ipc.CudaIpcReceiver("cuda:0", max_bytes=1024)
    copy = runtime.copy

    def copy_and_advance(*args):
        copy(*args)
        now[0] += 0.02

    monkeypatch.setattr(runtime, "copy", copy_and_advance)
    receiver.receive(_descriptor(), "float32", (2, 4), deadline=10.03)
    assert runtime.calls[-1][0] == "synchronize"
    assert runtime.calls[-1][2] == pytest.approx(0.01)
    calls = list(runtime.calls)
    with pytest.raises(ipc.CudaIpcError, match="before submission"):
        receiver.receive(_descriptor(generation=2), "float32", (2, 4), deadline=10)
    assert runtime.calls == calls
    assert not ipc._QUARANTINE
    receiver.close()


def test_completion_unknown_fences_new_sessions_and_bounds_quarantine(
    runtime, monkeypatch
):
    from vllm.v1.spec_decode.remote.server import (
        FakeDraftAdapter,
        RemoteDraftServer,
        _ServerSession,
    )
    from vllm.v1.spec_decode.remote.transport import DataFrame, TransportError

    adapter = FakeDraftAdapter(vocab_size=32)
    monkeypatch.setattr(adapter, "cuda_ipc_device", lambda: torch.device("cuda:0"))
    server = RemoteDraftServer(adapter)
    runtime.fail_wait = True
    frame = DataFrame(0, "float32", (2, 4), b"", cuda_ipc=_descriptor())
    for index in range(4):
        session = _ServerSession(server, Mock())
        session.selected_transport = "cuda_ipc"
        expected = ipc.CudaIpcCompletionError if index == 0 else TransportError
        with pytest.raises(expected), server.adapter_turn():
            session._resolve_features([frame])
        session.cleanup()
        assert not server.healthy
        assert len(ipc._QUARANTINE) == 1
    assert not any(
        call[0] in ("free", "destroy_event", "close_memory") for call in runtime.calls
    )


def test_unsupported_physical_peer_fails_before_opening_ipc_memory(
    runtime, monkeypatch
):
    def unsupported(*args):
        raise ipc.CudaIpcError("peer access unavailable")

    monkeypatch.setattr(ipc, "_require_peer_access", unsupported)
    receiver = ipc.CudaIpcReceiver("cuda:0", max_bytes=1024)
    with pytest.raises(ipc.CudaIpcError, match="peer access unavailable"):
        receiver.receive(_descriptor(), "float32", (2, 4))
    assert not any(c[0] == "open_memory" for c in runtime.calls)


@pytest.mark.parametrize("status", [0, 3])
def test_peer_capability_uses_physical_uuids_without_cuda_visibility(
    status, monkeypatch
):
    from vllm.third_party import pynvml

    destination = "GPU-00000000-0000-0000-0000-000000000002"
    calls = []
    monkeypatch.setattr(pynvml, "nvmlInit", lambda: calls.append("init"))
    monkeypatch.setattr(pynvml, "nvmlShutdown", lambda: calls.append("shutdown"))
    monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByUUID", lambda uuid: uuid)

    def p2p_status(dst, src, capability):
        calls.append((dst, src, capability))
        return status

    monkeypatch.setattr(pynvml, "nvmlDeviceGetP2PStatus", p2p_status)
    if status:
        with pytest.raises(ipc.CudaIpcError, match="peer access unavailable"):
            ipc._require_peer_access(_UUID, destination)
    else:
        ipc._require_peer_access(_UUID, destination)
    assert calls == [
        "init",
        (destination, _UUID, 0),
        (destination, _UUID, 1),
        "shutdown",
    ]


def _cuda_sender(connection):
    exporter = ipc.CudaIpcExporter("cuda:0", max_bytes=1 << 20)
    try:
        for step in range(3):
            tensor = torch.arange(1024, device="cuda:0", dtype=torch.float32) + step
            descriptor = exporter.export(tensor)
            connection.send_bytes(msgspec.msgpack.encode(descriptor))
            assert connection.recv() == "ack"
            exporter.release(descriptor)
        assert connection.recv() == "closed"
        exporter.close(peer_closed=True)
    finally:
        connection.close()


def _stop_cuda_sender(process):
    if process.pid is not None:
        if process.is_alive():
            process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join(10)
            assert not process.is_alive(), "CUDA exporter did not exit after kill"
    process.close()


@pytest.mark.parametrize("destination", [0, 1])
def test_cuda_cross_process_copy_preserves_previous_generation(destination):
    if torch.version.hip is not None:
        pytest.skip("CUDA IPC requires NVIDIA CUDA")
    if not torch.cuda.is_available() or torch.accelerator.device_count() <= destination:
        pytest.skip("required CUDA devices unavailable")
    source_uuid = ipc._device_uuid(torch, torch.device("cuda:0"))
    destination_uuid = ipc._device_uuid(torch, torch.device(f"cuda:{destination}"))
    try:
        ipc._require_peer_access(source_uuid, destination_uuid)
    except ipc.CudaIpcError as exc:
        pytest.skip(str(exc))
    context = multiprocessing.get_context("spawn")
    with contextlib.ExitStack() as cleanup:
        parent, child = context.Pipe()
        cleanup.callback(parent.close)
        cleanup.callback(child.close)
        process = context.Process(target=_cuda_sender, args=(child,))
        cleanup.callback(_stop_cuda_sender, process)
        process.start()
        child.close()
        receiver = ipc.CudaIpcReceiver(f"cuda:{destination}", max_bytes=1 << 20)
        cleanup.callback(receiver.close)
        retained = []
        for _ in range(3):
            assert parent.poll(60), "CUDA exporter did not produce a descriptor"
            descriptor = msgspec.msgpack.decode(
                parent.recv_bytes(), type=ipc.CudaIpcDescriptor
            )
            retained.append(receiver.receive(descriptor, "float32", (1024,)))
            parent.send("ack")
        receiver.close()
        parent.send("closed")
        for step, tensor in enumerate(retained):
            torch.testing.assert_close(tensor.cpu(), torch.arange(1024).float() + step)
        process.join(30)
        assert process.exitcode == 0
