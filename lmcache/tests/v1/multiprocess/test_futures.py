# SPDX-License-Identifier: Apache-2.0
# Standard
import multiprocessing as mp
import threading
import time

# Third Party
import pytest

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.multiprocess.futures import CUDAMessagingFuture, MessagingFuture


def _event_ipc_supported_for_active_device() -> bool:
    """Return True when the active device runtime supports event IPC."""
    # First Party
    from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend

    try:
        backend = get_event_ipc_backend(torch_device_type)
        backend.check_event_support(0)
        return True
    except Exception:
        return False


REQUIRES_EVENT_IPC = pytest.mark.skipif(
    not _event_ipc_supported_for_active_device(),
    reason=(
        "Event IPC is not supported for the active runtime device; "
        "skip IPC-dependent CUDAMessagingFuture tests."
    ),
)

# ==============================================================================
# Helper Functions for CUDAMessagingFuture Tests
# ==============================================================================


def _create_cuda_event_in_process(event_queue: mp.Queue, delay: float = 0.0):
    """Helper process that creates a CUDA event and sends the IPC handle."""
    torch_dev.init()
    if delay > 0:
        time.sleep(delay)

    # Create and record a CUDA event with interprocess flag
    event = torch_dev.Event(interprocess=True)
    event.record()
    event_bytes = event.ipc_handle()

    # Send the event handle to the main process
    event_queue.put(event_bytes)


def test_messaging_future_basic_usage():
    """Test basic usage of MessagingFuture: set result and retrieve it."""
    future = MessagingFuture[int]()

    # Initially, future should not be done
    assert not future.query(), "Future should not be done initially"

    # Set result
    future.set_result(42)

    # Future should now be done
    assert future.query(), "Future should be done after setting result"

    # Get result (should be immediate)
    result = future.result(timeout=1)
    assert result == 42, f"Expected result 42, got {result}"


def test_messaging_future_with_thread():
    """Test MessagingFuture with result set from another thread."""
    future = MessagingFuture[str]()

    def set_future_result():
        time.sleep(0.5)
        future.set_result("Hello from thread")

    # Start thread that will set the result
    thread = threading.Thread(target=set_future_result)
    thread.start()

    # Initially should not be done
    assert not future.query(), "Future should not be done before thread sets result"

    # Wait for result
    result = future.result(timeout=2)
    assert result == "Hello from thread", f"Expected 'Hello from thread', got {result}"

    # Should be done now
    assert future.query(), "Future should be done after getting result"

    thread.join()


def test_messaging_future_wait_success():
    """Test wait method when result becomes available."""
    future = MessagingFuture[int]()

    def set_future_result():
        time.sleep(0.3)
        future.set_result(100)

    thread = threading.Thread(target=set_future_result)
    thread.start()

    # Wait should return True when result is set
    success = future.wait(timeout=1)
    assert success, "Wait should return True when result is available"
    assert future.query(), "Future should be done after wait returns True"

    thread.join()


def test_messaging_future_wait_timeout():
    """Test wait method when timeout is reached."""
    future = MessagingFuture[int]()

    # Wait with short timeout (result never set)
    start_time = time.time()
    success = future.wait(timeout=0.2)
    elapsed = time.time() - start_time

    assert not success, "Wait should return False on timeout"
    assert not future.query(), "Future should not be done after timeout"
    assert 0.15 < elapsed < 0.3, f"Wait should respect timeout, elapsed: {elapsed}"


def test_messaging_future_result_timeout():
    """Test result method raises TimeoutError when timeout is reached."""
    future = MessagingFuture[int]()

    # Try to get result with timeout (result never set)
    with pytest.raises(
        TimeoutError, match="Future result not available within timeout"
    ):
        future.result(timeout=0.2)

    assert not future.query(), "Future should not be done after timeout"


def test_messaging_future_wait_no_timeout():
    """Test wait method without timeout (waits indefinitely until result is set)."""
    future = MessagingFuture[float]()

    def set_future_result():
        time.sleep(0.3)
        future.set_result(3.14)

    thread = threading.Thread(target=set_future_result)
    thread.start()

    # Wait without timeout should wait until result is available
    success = future.wait()  # No timeout parameter
    assert success, "Wait should return True when result is set"
    assert future.result() == 3.14, "Result should be accessible after wait"

    thread.join()


def test_messaging_future_multiple_result_calls():
    """Test that result can be retrieved multiple times after being set."""
    future = MessagingFuture[str]()
    future.set_result("persistent value")

    # Get result multiple times
    result1 = future.result(timeout=0.1)
    result2 = future.result(timeout=0.1)
    result3 = future.result(timeout=0.1)

    assert result1 == result2 == result3 == "persistent value", (
        "Result should be retrievable multiple times"
    )


def test_messaging_future_complex_type():
    """Test MessagingFuture with complex types like lists and dicts."""
    future = MessagingFuture[dict]()

    complex_data = {"key1": [1, 2, 3], "key2": {"nested": "value"}, "key3": 42}

    def set_future_result():
        time.sleep(0.2)
        future.set_result(complex_data)

    thread = threading.Thread(target=set_future_result)
    thread.start()

    result = future.result(timeout=1)
    assert result == complex_data, "Complex types should be preserved"

    thread.join()


# ==============================================================================
# CUDAMessagingFuture Tests
# ==============================================================================


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason=f"requires available {torch_device_type} runtime",
)
@REQUIRES_EVENT_IPC
def test_cuda_messaging_future_basic_usage():
    """Test basic usage of CUDAMessagingFuture: create, wait, and get result."""
    torch_dev.init()

    # Create CUDA event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_cuda_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=30)
    process.join(timeout=2)

    # Create the raw future that will return (event_bytes, result_value)
    raw_future = MessagingFuture[tuple[bytes, int]]()

    # Create CUDAMessagingFuture from raw future
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future)

    # Initially, future should not be done
    assert not cuda_future.query(), "CUDAMessagingFuture should not be done initially"

    # Set result in raw future
    raw_future.set_result((event_bytes, 42))

    # Wait for CUDA future to complete
    success = cuda_future.wait()
    assert success, "Wait should return True when result is available"

    # Get result
    result = cuda_future.result()
    assert result == 42, f"Expected result 42, got {result}"

    # Query should return True after completion
    assert cuda_future.query(), "CUDAMessagingFuture should be done after wait"


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason=f"requires available {torch_device_type} runtime",
)
@REQUIRES_EVENT_IPC
def test_cuda_messaging_future_with_thread():
    """Test CUDAMessagingFuture with result set from another thread."""
    torch_dev.init()

    # Create CUDA event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_cuda_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=30)
    process.join(timeout=2)

    raw_future = MessagingFuture[tuple[bytes, str]]()
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future)

    def set_future_result():
        time.sleep(0.5)
        raw_future.set_result((event_bytes, "Hello CUDA"))

    # Start thread that will set the result
    thread = threading.Thread(target=set_future_result)
    thread.start()

    # Initially should not be done
    assert not cuda_future.query(), (
        "Future should not be done before thread sets result"
    )

    # Wait for result
    result = cuda_future.result()
    assert result == "Hello CUDA", f"Expected 'Hello CUDA', got {result}"

    # Should be done now
    assert cuda_future.query(), "Future should be done after getting result"

    thread.join()


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason=f"requires available {torch_device_type} runtime",
)
@REQUIRES_EVENT_IPC
def test_cuda_messaging_future_wait_no_timeout():
    """Test wait method without timeout (waits indefinitely
    until result is set).
    """
    torch_dev.init()

    # Create CUDA event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_cuda_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=30)
    process.join(timeout=2)

    raw_future = MessagingFuture[tuple[bytes, float]]()
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future)

    def set_future_result():
        time.sleep(0.3)
        raw_future.set_result((event_bytes, 3.14))

    thread = threading.Thread(target=set_future_result)
    thread.start()

    # Wait without timeout should wait until result is available
    success = cuda_future.wait()  # No timeout parameter
    assert success, "Wait should return True when result is set"
    assert cuda_future.result() == 3.14, "Result should be accessible after wait"

    thread.join()


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason=f"requires available {torch_device_type} runtime",
)
@REQUIRES_EVENT_IPC
def test_cuda_messaging_future_wait_with_timeout_success():
    """Test that wait method works correctly with timeout when result is available."""
    torch_dev.init()

    # Create CUDA event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_cuda_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=30)
    process.join(timeout=2)

    raw_future = MessagingFuture[tuple[bytes, int]]()
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future)

    def set_future_result():
        time.sleep(0.3)
        raw_future.set_result((event_bytes, 123))

    thread = threading.Thread(target=set_future_result)
    thread.start()

    # Wait with timeout should return True when result is available
    success = cuda_future.wait(timeout=2.0)
    assert success, "Wait with timeout should return True when result is available"
    assert cuda_future.result() == 123, "Result should be accessible after wait"

    thread.join()


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason=f"requires available {torch_device_type} runtime",
)
@REQUIRES_EVENT_IPC
def test_cuda_messaging_future_wait_timeout_reached():
    """Test that wait method returns False when timeout is reached."""
    torch_dev.init()

    raw_future = MessagingFuture[tuple[bytes, int]]()
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future)

    # Wait with short timeout (result never set)
    start_time = time.time()
    success = cuda_future.wait(timeout=0.2)
    elapsed = time.time() - start_time

    assert not success, "Wait should return False on timeout"
    assert not cuda_future.query(), "Future should not be done after timeout"
    assert 0.15 < elapsed < 0.4, f"Wait should respect timeout, elapsed: {elapsed}"


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason=f"requires available {torch_device_type} runtime",
)
@REQUIRES_EVENT_IPC
def test_cuda_messaging_future_result_with_timeout_success():
    """Test that result method works correctly with timeout when result is available."""
    torch_dev.init()

    # Create CUDA event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_cuda_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=30)
    process.join(timeout=2)

    raw_future = MessagingFuture[tuple[bytes, int]]()
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future)

    def set_future_result():
        time.sleep(0.3)
        raw_future.set_result((event_bytes, 456))

    thread = threading.Thread(target=set_future_result)
    thread.start()

    # Get result with timeout should succeed when result is available
    result = cuda_future.result(timeout=2.0)
    assert result == 456, f"Expected result 456, got {result}"

    thread.join()


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason=f"requires available {torch_device_type} runtime",
)
@REQUIRES_EVENT_IPC
def test_cuda_messaging_future_result_timeout_reached():
    """Test that result method raises TimeoutError when timeout is reached."""
    torch_dev.init()

    raw_future = MessagingFuture[tuple[bytes, int]]()
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future)

    # Try to get result with timeout (result never set)
    with pytest.raises(
        TimeoutError, match="DeviceMessagingFuture result not available within timeout"
    ):
        cuda_future.result(timeout=0.2)


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason=f"requires available {torch_device_type} runtime",
)
@REQUIRES_EVENT_IPC
def test_cuda_messaging_future_multiple_result_calls():
    """Test that result can be retrieved multiple times after being set."""
    torch_dev.init()

    # Create CUDA event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_cuda_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=30)
    process.join(timeout=2)

    raw_future = MessagingFuture[tuple[bytes, str]]()
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future)

    raw_future.set_result((event_bytes, "persistent cuda value"))

    # Get result multiple times
    result1 = cuda_future.result()
    result2 = cuda_future.result()
    result3 = cuda_future.result()

    assert result1 == result2 == result3 == "persistent cuda value", (
        "Result should be retrievable multiple times"
    )


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason=f"requires available {torch_device_type} runtime",
)
@REQUIRES_EVENT_IPC
def test_cuda_messaging_future_query_before_and_after():
    """Test query method returns False before completion and True after."""
    torch_dev.init()

    # Create CUDA event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_cuda_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=30)
    process.join(timeout=2)

    raw_future = MessagingFuture[tuple[bytes, int]]()
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future)

    # Query before setting result
    assert not cuda_future.query(), "Query should return False before result is set"

    # Set result
    raw_future.set_result((event_bytes, 100))

    # Wait for completion
    cuda_future.wait()

    # Query after setting result
    assert cuda_future.query(), (
        "Query should return True after result is set and waited"
    )


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason=f"requires available {torch_device_type} runtime",
)
@REQUIRES_EVENT_IPC
def test_cuda_messaging_future_complex_type():
    """Test CUDAMessagingFuture with complex types like lists and dicts."""
    torch_dev.init()

    # Create CUDA event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_cuda_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=30)
    process.join(timeout=2)

    complex_data = {"key1": [1, 2, 3], "key2": {"nested": "value"}, "key3": 42}

    raw_future = MessagingFuture[tuple[bytes, dict]]()
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future)

    def set_future_result():
        time.sleep(0.2)
        raw_future.set_result((event_bytes, complex_data))

    thread = threading.Thread(target=set_future_result)
    thread.start()

    result = cuda_future.result()
    assert result == complex_data, "Complex types should be preserved"

    thread.join()


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason=f"requires available {torch_device_type} runtime",
)
@REQUIRES_EVENT_IPC
def test_messaging_future_to_device_future():
    """Test converting MessagingFuture to DeviceMessagingFuture
    using to_device_future method.
    """
    torch_dev.init()

    # Create CUDA event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_cuda_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=30)
    process.join(timeout=2)

    raw_future = MessagingFuture[tuple[bytes, int]]()

    # Convert to device future
    device_future = raw_future.to_device_future()

    # Verify it's a CUDAMessagingFuture instance
    assert isinstance(device_future, CUDAMessagingFuture), (
        "to_device_future should return DeviceMessagingFuture instance"
    )

    # Set result and verify it works
    raw_future.set_result((event_bytes, 999))

    result = device_future.result()
    assert result == 999, f"Expected result 999, got {result}"


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason=f"requires available {torch_device_type} runtime",
)
@REQUIRES_EVENT_IPC
def test_cuda_messaging_future_with_explicit_device():
    """Test CUDAMessagingFuture with explicit device parameter."""
    torch_dev.init()

    # Create CUDA event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_cuda_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=30)
    process.join(timeout=2)

    device = torch_dev.current_device()
    raw_future = MessagingFuture[tuple[bytes, str]]()

    # Create CUDA future with explicit device
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future, device=device)

    # Set result
    raw_future.set_result((event_bytes, "explicit device"))

    # Get result
    result = cuda_future.result()
    assert result == "explicit device", f"Expected 'explicit device', got {result}"


# ==============================================================================
# Device-neutral future wiring (no CUDA required)
# ==============================================================================


def test_cuda_future_is_alias_of_device_future():
    # First Party
    from lmcache.v1.multiprocess.futures import (
        CUDAMessagingFuture,
        DeviceMessagingFuture,
    )

    assert CUDAMessagingFuture is DeviceMessagingFuture


def test_to_cuda_future_returns_device_future_instance():
    # First Party
    from lmcache.v1.multiprocess.futures import DeviceMessagingFuture

    raw = MessagingFuture()
    with pytest.warns(DeprecationWarning, match="to_device_future"):
        fut = raw.to_cuda_future(device="cpu")
    assert isinstance(fut, DeviceMessagingFuture)


def test_device_future_stub_end_to_end():
    # First Party
    from lmcache.v1.multiprocess.futures import DeviceMessagingFuture

    raw = MessagingFuture[tuple[bytes, int]]()
    fut = DeviceMessagingFuture.FromMessagingFuture(raw, device="cpu")
    assert not fut.query()
    raw.set_result((b"stub_ipc_handle", 7))
    assert fut.wait() is True
    assert fut.result() == 7
    assert fut.query() is True


def test_device_future_delegates_to_backend(monkeypatch):
    # First Party
    from lmcache.v1.multiprocess.futures import DeviceMessagingFuture

    calls = []

    class _FakeBackend:
        device_type = "fake"

        def check_event_support(self, device):
            calls.append(("check", device))

        def import_event(self, handle, device):
            calls.append(("import", handle, device))
            return "EVT"

        def synchronize_event(self, event, device):
            calls.append(("sync", event, device))

        def query_event(self, event):
            calls.append(("query", event))
            return True

    monkeypatch.setattr(
        "lmcache.v1.multiprocess.futures.get_event_ipc_backend",
        lambda device=None: _FakeBackend(),
    )

    raw = MessagingFuture[tuple[bytes, int]]()
    fut = DeviceMessagingFuture.FromMessagingFuture(raw, device="dev")
    raw.set_result((b"h", 99))
    assert fut.wait() is True
    assert fut.result() == 99
    assert ("check", "dev") in calls
    assert ("import", b"h", "dev") in calls
    assert any(c[0] == "sync" for c in calls)


def test_device_future_checks_backend_support_during_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported event IPC fails before the future enters the wait path."""
    # First Party
    from lmcache.v1.multiprocess.futures import DeviceMessagingFuture

    class _UnsupportedBackend:
        device_type = "fake"

        def check_event_support(self, device: object) -> None:
            raise RuntimeError("event IPC unavailable")

    monkeypatch.setattr(
        "lmcache.v1.multiprocess.futures.get_event_ipc_backend",
        lambda device=None: _UnsupportedBackend(),
    )

    with pytest.raises(RuntimeError, match="event IPC unavailable"):
        DeviceMessagingFuture.FromMessagingFuture(
            MessagingFuture[tuple[bytes, int]](), device="dev"
        )
