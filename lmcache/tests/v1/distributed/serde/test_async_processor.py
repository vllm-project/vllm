# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for AsyncSerdeProcessor.

Verifies the async, notifier-based contract:
- submit returns a task id immediately.
- The corresponding notifier fd is signaled on completion.
- query_result returns the bool outcome, None before completion and after
  being consumed (non-idempotent).
- Failing serialize/deserialize produces result=False and still signals fd.
- Serialize and deserialize use distinct event fds.
"""

# Standard
from typing import Callable, Optional
import select
import time

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde import (
    AsyncSerdeProcessor,
    Deserializer,
    Serializer,
)
from lmcache.v1.platform import consume_fd

_TEST_KEY = ObjectKey(chunk_hash=b"\x00" * 32, model_name="test", kv_rank=0)


class _FakeSerializer(Serializer):
    def __init__(self, transform: Optional[Callable[[int], None]] = None) -> None:
        self._transform = transform
        self.calls = 0

    def serialize(self, src, dst, key) -> int:  # type: ignore[no-untyped-def]
        if self._transform is not None:
            self._transform(self.calls)
        self.calls += 1
        return 1

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        return 1


class _FakeDeserializer(Deserializer):
    def __init__(self, transform: Optional[Callable[[int], None]] = None) -> None:
        self._transform = transform
        self.calls = 0

    def deserialize(self, src, dst, key) -> None:  # type: ignore[no-untyped-def]
        if self._transform is not None:
            self._transform(self.calls)
        self.calls += 1


def _wait_for_fd(fd: int, timeout_s: float = 2.0) -> bool:
    """Wait until ``fd`` is readable or timeout. Drains the pending signal."""
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining_ms = int(max(0, (deadline - time.monotonic()) * 1000))
        if poller.poll(remaining_ms):
            try:
                consume_fd(fd)
            except OSError:
                pass
            return True
    return False


def test_serialize_and_deserialize_fds_are_distinct() -> None:
    processor = AsyncSerdeProcessor(_FakeSerializer(), _FakeDeserializer())
    try:
        assert (
            processor.get_serialize_event_fd() != processor.get_deserialize_event_fd()
        )
    finally:
        processor.close()


def test_serialize_signals_fd_and_result_is_true() -> None:
    serializer = _FakeSerializer()
    processor = AsyncSerdeProcessor(serializer, _FakeDeserializer())
    try:
        task_id = processor.submit_serialize([object()], [object()], [_TEST_KEY])  # type: ignore[list-item]
        assert _wait_for_fd(processor.get_serialize_event_fd()), "fd never signaled"
        assert processor.query_serialize_result(task_id) is True
        # Non-idempotent: second query returns None.
        assert processor.query_serialize_result(task_id) is None
        assert serializer.calls == 1
    finally:
        processor.close()


def test_deserialize_signals_fd_and_result_is_true() -> None:
    deserializer = _FakeDeserializer()
    processor = AsyncSerdeProcessor(_FakeSerializer(), deserializer)
    try:
        task_id = processor.submit_deserialize([object()], [object()], [_TEST_KEY])  # type: ignore[list-item]
        assert _wait_for_fd(processor.get_deserialize_event_fd()), "fd never signaled"
        assert processor.query_deserialize_result(task_id) is True
        assert processor.query_deserialize_result(task_id) is None
        assert deserializer.calls == 1
    finally:
        processor.close()


def test_serialize_failure_reports_false() -> None:
    """If the sync serializer raises, query_serialize_result returns False."""

    def _boom(_i: int) -> None:
        raise RuntimeError("serialize failed")

    processor = AsyncSerdeProcessor(_FakeSerializer(_boom), _FakeDeserializer())
    try:
        task_id = processor.submit_serialize([object()], [object()], [_TEST_KEY])  # type: ignore[list-item]
        assert _wait_for_fd(processor.get_serialize_event_fd()), "fd never signaled"
        assert processor.query_serialize_result(task_id) is False
    finally:
        processor.close()


def test_query_returns_none_before_completion() -> None:
    """Querying an unknown/not-yet-completed task yields None, not an error."""
    processor = AsyncSerdeProcessor(_FakeSerializer(), _FakeDeserializer())
    try:
        # No task submitted with id 42 — should be None, not raise.
        assert processor.query_serialize_result(42) is None
        assert processor.query_deserialize_result(42) is None
    finally:
        processor.close()


def test_estimate_serialized_size_delegates_to_serializer() -> None:
    serializer = _FakeSerializer()
    processor = AsyncSerdeProcessor(serializer, _FakeDeserializer())
    try:
        layout = MemoryLayoutDesc(shapes=[], dtypes=[])
        assert processor.estimate_serialized_size(layout) == 1
    finally:
        processor.close()


# ---------------------------------------------------------------------------
# set_used_size propagation: after serialize returns ``n``, the processor
# narrows the destination's logical size so downstream L2 adapters that
# read ``obj.get_size()`` / ``obj.byte_array`` see exactly ``n`` bytes.
# ---------------------------------------------------------------------------


class _RecordingSerializer(Serializer):
    """A serializer that always claims to have written ``n_bytes`` so
    tests can assert the processor narrows the destination accordingly.
    """

    def __init__(self, n_bytes: int) -> None:
        self._n_bytes = n_bytes

    def serialize(self, src, dst, key) -> int:  # type: ignore[no-untyped-def]
        return self._n_bytes

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        return self._n_bytes


class _NarrowableDst:
    """A minimal MemoryObj-like destination that records every
    ``set_used_size`` call.  Not a real ``MemoryObj`` -- the test only
    needs the part of the interface the processor touches."""

    def __init__(self) -> None:
        self.used_sizes: list[int] = []

    def set_used_size(self, n: int) -> None:
        self.used_sizes.append(n)


def test_serialize_narrows_dst_to_actual_bytes_written() -> None:
    """After a successful serialize that returns ``n``, the processor
    calls ``dst.set_used_size(n)`` so the over-allocated destination
    buffer reports the actual written length, not its allocated size."""
    serializer = _RecordingSerializer(n_bytes=213)
    processor = AsyncSerdeProcessor(serializer, _FakeDeserializer())
    try:
        dst = _NarrowableDst()
        task_id = processor.submit_serialize([object()], [dst], [_TEST_KEY])  # type: ignore[list-item]
        assert _wait_for_fd(processor.get_serialize_event_fd()), "fd never signaled"
        assert processor.query_serialize_result(task_id) is True
        assert dst.used_sizes == [213]
    finally:
        processor.close()


def test_serialize_failure_does_not_narrow_dst() -> None:
    """If the serializer raises, the processor must NOT narrow the
    destination -- failure cleanup typically reuses the original
    allocation and the next attempt depends on the layout-derived
    size."""

    class _BoomSerializer(Serializer):
        def serialize(self, src, dst, key) -> int:  # type: ignore[no-untyped-def]
            raise RuntimeError("serialize failed")

        def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
            return 1

    processor = AsyncSerdeProcessor(_BoomSerializer(), _FakeDeserializer())
    try:
        dst = _NarrowableDst()
        task_id = processor.submit_serialize([object()], [dst], [_TEST_KEY])  # type: ignore[list-item]
        assert _wait_for_fd(processor.get_serialize_event_fd()), "fd never signaled"
        assert processor.query_serialize_result(task_id) is False
        assert dst.used_sizes == [], "narrowing must not happen on failure"
    finally:
        processor.close()


def test_serialize_works_when_dst_lacks_set_used_size() -> None:
    """Backward compatibility: a duck-typed destination that does NOT
    implement ``set_used_size`` (e.g. a bare ``object()`` in older
    tests) must still produce a successful task; the processor guards
    the narrowing call with ``hasattr``."""
    processor = AsyncSerdeProcessor(_FakeSerializer(), _FakeDeserializer())
    try:
        task_id = processor.submit_serialize([object()], [object()], [_TEST_KEY])  # type: ignore[list-item]
        assert _wait_for_fd(processor.get_serialize_event_fd()), "fd never signaled"
        # Success despite dst lacking set_used_size.
        assert processor.query_serialize_result(task_id) is True
    finally:
        processor.close()


def test_serialize_skips_narrowing_when_serializer_returns_non_int() -> None:
    """A serializer that does not honor the ``serialize() -> int``
    contract and returns ``None`` must not crash the processor.  The
    narrowing call is skipped (``isinstance(n, int)`` guard) rather than
    passing ``None`` into ``set_used_size``, where the range check would
    raise ``TypeError: '<' not supported between NoneType and int``."""

    class _NoneReturningSerializer(Serializer):
        def serialize(self, src, dst, key):  # type: ignore[no-untyped-def]
            return None  # type: ignore[return-value]

        def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
            return 1

    processor = AsyncSerdeProcessor(_NoneReturningSerializer(), _FakeDeserializer())
    try:
        dst = _NarrowableDst()
        task_id = processor.submit_serialize([object()], [dst], [_TEST_KEY])  # type: ignore[list-item]
        assert _wait_for_fd(processor.get_serialize_event_fd()), "fd never signaled"
        # Task still succeeds; narrowing was skipped, not attempted.
        assert processor.query_serialize_result(task_id) is True
        assert dst.used_sizes == [], "must not call set_used_size with a non-int"
    finally:
        processor.close()
