# SPDX-License-Identifier: Apache-2.0

"""Tests for ObservabilityConfig CLI parsing and extra-logging registration."""

# Standard
import argparse
import logging
import time

# Third Party
import pytest

# First Party
from lmcache import torch_device_type
from lmcache.v1.mp_observability.config import (
    ObservabilityConfig,
    add_observability_args,
    init_observability,
    parse_args_to_observability_config,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBusConfig, init_event_bus

_EXTRA_LOGGER = "lmcache.v1.mp_observability.subscribers.logging.extra_stats"


def _parse(argv: list[str]) -> ObservabilityConfig:
    parser = argparse.ArgumentParser()
    add_observability_args(parser)
    return parse_args_to_observability_config(parser.parse_args(argv))


class TestExtraLoggingArgs:
    def test_defaults(self):
        config = _parse([])
        assert config.extra_logging_enabled is False
        assert config.extra_logging_interval == 10.0

    def test_flags(self):
        config = _parse(["--enable-extra-logging", "--extra-logging-interval", "2.5"])
        assert config.extra_logging_enabled is True
        assert config.extra_logging_interval == 2.5

    def test_conflicts_with_disabled_observability(self):
        with pytest.raises(ValueError, match="enable-extra-logging"):
            _parse(["--enable-extra-logging", "--disable-observability"])

    def test_non_positive_interval_rejected(self):
        with pytest.raises(ValueError, match="extra-logging-interval"):
            _parse(["--enable-extra-logging", "--extra-logging-interval", "0"])


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class TestExtraLoggingRegistration:
    def _drive_bus(self, config: ObservabilityConfig) -> list[str]:
        handler = _CaptureHandler()
        lg = logging.getLogger(_EXTRA_LOGGER)
        old_level = lg.level
        lg.addHandler(handler)
        lg.setLevel(logging.INFO)
        try:
            bus = init_observability(config, start_prometheus_http_server=False)
            bus.publish(
                Event(
                    event_type=EventType.MP_STORE_END,
                    session_id="req-1",
                    metadata={
                        "device": f"{torch_device_type}:0",
                        "total_bytes": 10**9,
                        "num_tokens": 512,
                    },
                )
            )
            time.sleep(0.06)
            bus.publish(Event(event_type=EventType.L1_EVICTION_LOOP_TICK))
            time.sleep(0.15)
            bus.stop()
        finally:
            lg.removeHandler(handler)
            lg.setLevel(old_level)
            init_event_bus(EventBusConfig(enabled=False))
        return [r.getMessage() for r in handler.records]

    def test_registered_when_enabled(self):
        config = ObservabilityConfig(
            enabled=True,
            metrics_enabled=False,
            logging_enabled=False,
            tracing_enabled=False,
            extra_logging_enabled=True,
            extra_logging_interval=0.05,
        )
        messages = self._drive_bus(config)
        assert any("L0<->L1 stats" in m for m in messages)

    def test_not_registered_when_disabled(self):
        config = ObservabilityConfig(
            enabled=True,
            metrics_enabled=False,
            logging_enabled=False,
            tracing_enabled=False,
        )
        messages = self._drive_bus(config)
        assert messages == []
