# SPDX-License-Identifier: Apache-2.0

"""Tests for register_gauge in otel_init."""

# Standard
from unittest.mock import MagicMock, patch

# First Party
from lmcache.v1.mp_observability.otel_init import register_gauge


class TestRegisterGauge:
    """Tests for the register_gauge convenience wrapper."""

    def test_creates_observable_gauge(self):
        """Gauge is created on the correct meter."""
        # Standard
        import sys

        mock_meter = MagicMock()
        mock_otel = MagicMock()
        mock_otel.get_meter.return_value = mock_meter
        mock_otel.metrics = mock_otel

        saved = {
            k: sys.modules.get(k) for k in ("opentelemetry", "opentelemetry.metrics")
        }
        sys.modules["opentelemetry"] = mock_otel
        sys.modules["opentelemetry.metrics"] = mock_otel
        try:
            register_gauge(
                "test.meter",
                "test.gauge",
                "A test gauge",
                lambda: 42,
            )
            mock_otel.get_meter.assert_called_once_with("test.meter")
            mock_meter.create_observable_gauge.assert_called_once()
            call_kwargs = mock_meter.create_observable_gauge.call_args
            assert call_kwargs[0][0] == "test.gauge"
            assert call_kwargs[1]["description"] == "A test gauge"
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_callback_invokes_func(self):
        """The OTel callback delegates to the user-provided func."""
        mock_meter = MagicMock()
        mock_otel = MagicMock()
        mock_otel.get_meter.return_value = mock_meter
        mock_otel.Observation = lambda v: v

        # Standard
        import sys

        saved = {
            k: sys.modules.get(k) for k in ("opentelemetry", "opentelemetry.metrics")
        }
        sys.modules["opentelemetry"] = mock_otel
        sys.modules["opentelemetry.metrics"] = mock_otel
        mock_otel.metrics = mock_otel
        try:
            counter = {"value": 0}

            def my_func():
                counter["value"] += 1
                return 99

            register_gauge("m", "g", "desc", my_func)

            # Extract the callback that was passed
            cb_list = mock_meter.create_observable_gauge.call_args[1]["callbacks"]
            assert len(cb_list) == 1
            result = cb_list[0](None)
            assert result == [99]
            assert counter["value"] == 1
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_no_otel_logs_debug(self):
        """When opentelemetry is missing, log debug and skip."""
        # Standard
        import sys

        saved = {
            k: sys.modules.pop(k, None)
            for k in list(sys.modules)
            if k.startswith("opentelemetry")
        }
        try:
            # Force ImportError by removing the module
            with patch(
                "builtins.__import__",
                side_effect=ImportError("no otel"),
            ):
                # Should not raise
                register_gauge("m", "g", "desc", lambda: 0)
        finally:
            sys.modules.update({k: v for k, v in saved.items() if v is not None})

    def test_register_multiple_gauges(self):
        """Multiple gauges can be registered without error."""
        mock_meter = MagicMock()
        mock_otel = MagicMock()
        mock_otel.get_meter.return_value = mock_meter

        # Standard
        import sys

        saved = {
            k: sys.modules.get(k) for k in ("opentelemetry", "opentelemetry.metrics")
        }
        sys.modules["opentelemetry"] = mock_otel
        sys.modules["opentelemetry.metrics"] = mock_otel
        mock_otel.metrics = mock_otel
        try:
            register_gauge("m", "g1", "d1", lambda: 1)
            register_gauge("m", "g2", "d2", lambda: 2)
            assert mock_meter.create_observable_gauge.call_count == 2
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
