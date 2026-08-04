# SPDX-License-Identifier: Apache-2.0
"""Tests for LookupHashLoggingSubscriber and LookupHashLogConfig."""

# Standard
from pathlib import Path
import json
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.logging.lookup_hash import (
    LookupHashLogConfig,
    LookupHashLoggingSubscriber,
    _format_timestamp,
)


class TestLookupHashLogConfig:
    def test_disabled_by_default(self) -> None:
        config = LookupHashLogConfig()
        assert not config.enabled
        assert config.output_dir == ""

    def test_enabled_when_output_dir_set(self) -> None:
        config = LookupHashLogConfig(output_dir="/tmp/test")
        assert config.enabled

    def test_defaults(self) -> None:
        config = LookupHashLogConfig()
        assert config.rotation_interval_sec == 6 * 3600
        assert config.rotation_max_size == 100 * 1024 * 1024
        assert config.max_files == 100


class TestFormatTimestamp:
    def test_known_timestamp(self) -> None:
        # 2026-04-01 14:30:25 UTC
        ts = 1775053825.0
        result = _format_timestamp(ts)
        assert result == "20260401_143025"

    def test_returns_string(self) -> None:
        result = _format_timestamp(time.time())
        assert isinstance(result, str)
        assert len(result) == 15  # YYYYMMDD_HHMMSS


def _make_event(
    request_id: str = "req-001",
    chunk_hashes: list | None = None,
    model_name: str = "test-model",
    chunk_size: int = 256,
    seq_len: int = 1024,
    dtypes: list[str] | None = None,
    shapes: list | None = None,
) -> Event:
    """Helper to create a lookup event with metadata."""
    return Event(
        event_type=EventType.MP_LOOKUP,
        session_id=request_id,
        metadata={
            "request_id": request_id,
            "chunk_hashes": chunk_hashes or [b"\xab\xcd"],
            "model_name": model_name,
            "chunk_size": chunk_size,
            "seq_len": seq_len,
            "dtypes": dtypes or [],
            "shapes": shapes or [],
        },
    )


def _publish_and_drain(bus: EventBus, event: Event) -> None:
    """Publish an event and synchronously drain it."""
    bus.publish(event)
    bus._drain_all()


class TestLookupHashLoggingSubscriber:
    @pytest.fixture
    def log_dir(self, tmp_path: Path) -> Path:
        """Provide a temporary log directory."""
        return tmp_path / "lookup_hashes"

    @pytest.fixture
    def config(self, log_dir: Path) -> LookupHashLogConfig:
        """Provide a config with small limits for testing."""
        return LookupHashLogConfig(
            output_dir=str(log_dir),
            rotation_interval_sec=3600,
            rotation_max_size=100 * 1024 * 1024,
            max_files=100,
        )

    @pytest.fixture
    def bus(self) -> EventBus:
        """Create an EventBus (no background thread — we drain manually)."""
        return EventBus(EventBusConfig(enabled=True))

    def test_creates_output_dir(self, config: LookupHashLogConfig) -> None:
        sub = LookupHashLoggingSubscriber(config)
        assert Path(config.output_dir).is_dir()
        sub.shutdown()

    def test_log_and_shutdown_writes_file(
        self,
        config: LookupHashLogConfig,
        log_dir: Path,
        bus: EventBus,
    ) -> None:
        sub = LookupHashLoggingSubscriber(config)
        bus.register_subscriber(sub)

        event = _make_event(
            chunk_hashes=[b"\xab\xcd", b"\x12\x34"],
        )
        _publish_and_drain(bus, event)
        sub.shutdown()

        files = list(log_dir.glob("lookup_hashes_*.jsonl"))
        assert len(files) == 1

        lines = files[0].read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1

        data = json.loads(lines[0])
        assert data["request_id"] == "req-001"
        assert data["model_name"] == "test-model"
        assert data["chunk_hashes"] == ["0xabcd", "0x1234"]
        assert "timestamp" in data

    def test_multiple_entries(
        self,
        config: LookupHashLogConfig,
        log_dir: Path,
        bus: EventBus,
    ) -> None:
        sub = LookupHashLoggingSubscriber(config)
        bus.register_subscriber(sub)

        for i in range(10):
            event = _make_event(
                request_id=f"req-{i:03d}",
                chunk_hashes=[b"\xaa"],
                model_name=f"model-{i}",
            )
            _publish_and_drain(bus, event)
        sub.shutdown()

        files = list(log_dir.glob("lookup_hashes_*.jsonl"))
        assert len(files) == 1

        lines = files[0].read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 10

    def test_disabled_config_not_enabled(self) -> None:
        config = LookupHashLogConfig()  # output_dir=""
        assert not config.enabled
        # Just verify the config property works; init_observability
        # checks config.enabled before creating the subscriber.

    def test_shutdown_is_idempotent(
        self,
        config: LookupHashLogConfig,
        bus: EventBus,
    ) -> None:
        sub = LookupHashLoggingSubscriber(config)
        bus.register_subscriber(sub)

        _publish_and_drain(
            bus,
            _make_event(chunk_hashes=[b"\xaa"]),
        )
        sub.shutdown()
        # Second shutdown should not raise
        sub.shutdown()

    def test_size_based_rotation(self, log_dir: Path, bus: EventBus) -> None:
        config = LookupHashLogConfig(
            output_dir=str(log_dir),
            rotation_interval_sec=999999,  # won't trigger
            rotation_max_size=200,  # very small, triggers quickly
            max_files=100,
        )
        sub = LookupHashLoggingSubscriber(config)
        bus.register_subscriber(sub)

        for i in range(20):
            _publish_and_drain(
                bus,
                _make_event(
                    request_id=f"req-{i:03d}",
                    chunk_hashes=[b"\xaa\xbb\xcc\xdd"],
                ),
            )
        sub.shutdown()

        files = list(log_dir.glob("lookup_hashes_*.jsonl"))
        assert len(files) > 1, "Should have rotated due to size"

    def test_max_files_limit(self, log_dir: Path, bus: EventBus) -> None:
        config = LookupHashLogConfig(
            output_dir=str(log_dir),
            rotation_interval_sec=999999,
            rotation_max_size=50,  # tiny, forces many rotations
            max_files=3,
        )
        sub = LookupHashLoggingSubscriber(config)
        bus.register_subscriber(sub)

        for i in range(30):
            _publish_and_drain(
                bus,
                _make_event(
                    request_id=f"req-{i:03d}",
                    chunk_hashes=[b"\xaa\xbb\xcc\xdd"],
                ),
            )
        sub.shutdown()

        files = list(log_dir.glob("lookup_hashes_*.jsonl"))
        assert len(files) <= 3

    def test_existing_files_discovered_on_init(
        self, log_dir: Path, bus: EventBus
    ) -> None:
        """Files from previous runs are counted toward max_files."""
        log_dir.mkdir(parents=True, exist_ok=True)
        # Create 3 pre-existing files
        for i in range(3):
            f = log_dir / f"lookup_hashes_20260101_000000_{i:06d}.jsonl"
            f.write_text("{}\n", encoding="utf-8")
            # Stagger mtime so sorting is deterministic
            time.sleep(0.01)

        config = LookupHashLogConfig(
            output_dir=str(log_dir),
            rotation_interval_sec=999999,
            rotation_max_size=50,
            max_files=4,  # 3 existing + 1 new = at limit
        )
        sub = LookupHashLoggingSubscriber(config)
        bus.register_subscriber(sub)

        # Write enough to trigger at least 2 rotations
        for i in range(20):
            _publish_and_drain(
                bus,
                _make_event(
                    request_id=f"req-{i:03d}",
                    chunk_hashes=[b"\xaa\xbb\xcc\xdd"],
                ),
            )
        sub.shutdown()

        files = list(log_dir.glob("lookup_hashes_*.jsonl"))
        assert len(files) <= 4

    def test_json_output_is_valid(
        self,
        config: LookupHashLogConfig,
        log_dir: Path,
        bus: EventBus,
    ) -> None:
        sub = LookupHashLoggingSubscriber(config)
        bus.register_subscriber(sub)

        _publish_and_drain(
            bus,
            _make_event(
                request_id="req-001",
                chunk_hashes=[b"\xff"],
                model_name="model-a",
            ),
        )
        _publish_and_drain(
            bus,
            _make_event(
                request_id="req-002",
                chunk_hashes=[b"\x00" * 16],
                model_name="model-b",
            ),
        )
        sub.shutdown()

        files = list(log_dir.glob("lookup_hashes_*.jsonl"))
        for f in files:
            for line in f.read_text(encoding="utf-8").strip().split("\n"):
                data = json.loads(line)  # Should not raise
                assert isinstance(data["timestamp"], float)
                assert isinstance(data["request_id"], str)
                assert isinstance(data["model_name"], str)
                assert isinstance(data["shapes"], list)
                assert isinstance(data["chunk_hashes"], list)

    def test_integer_hashes_handled(
        self,
        config: LookupHashLogConfig,
        log_dir: Path,
        bus: EventBus,
    ) -> None:
        """Verify integer chunk hashes (not bytes) are also handled."""
        sub = LookupHashLoggingSubscriber(config)
        bus.register_subscriber(sub)

        _publish_and_drain(
            bus,
            _make_event(
                chunk_hashes=[255, 65536],  # type: ignore[list-item]
            ),
        )
        sub.shutdown()

        files = list(log_dir.glob("lookup_hashes_*.jsonl"))
        lines = files[0].read_text(encoding="utf-8").strip().split("\n")
        data = json.loads(lines[0])
        assert data["chunk_hashes"] == ["0xff", "0x10000"]
