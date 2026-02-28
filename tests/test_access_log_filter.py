# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the UvicornAccessLogFilter class.
"""

import logging
from argparse import Namespace

from vllm.entrypoints.openai.server_utils import (
    _get_excluded_paths,
    get_uvicorn_log_config,
)
from vllm.logging_utils.access_log_filter import (
    UvicornAccessLogFilter,
    create_uvicorn_log_config,
)


class TestUvicornAccessLogFilter:
    """Test cases for UvicornAccessLogFilter."""

    def test_filter_allows_all_when_no_excluded_paths(self):
        """Filter should allow all logs when no paths are excluded."""
        filter = UvicornAccessLogFilter(excluded_paths=[])

        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "GET", "/v1/completions", "1.1", 200),
            exc_info=None,
        )

        assert filter.filter(record) is True

    def test_filter_allows_all_when_excluded_paths_is_none(self):
        """Filter should allow all logs when excluded_paths is None."""
        filter = UvicornAccessLogFilter(excluded_paths=None)

        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "GET", "/health", "1.1", 200),
            exc_info=None,
        )

        assert filter.filter(record) is True

    def test_filter_excludes_health_endpoint(self):
        """Filter should exclude /health endpoint when configured."""
        filter = UvicornAccessLogFilter(excluded_paths=["/health"])

        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "GET", "/health", "1.1", 200),
            exc_info=None,
        )

        assert filter.filter(record) is False

    def test_filter_excludes_metrics_endpoint(self):
        """Filter should exclude /metrics endpoint when configured."""
        filter = UvicornAccessLogFilter(excluded_paths=["/metrics"])

        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "GET", "/metrics", "1.1", 200),
            exc_info=None,
        )

        assert filter.filter(record) is False

    def test_filter_allows_non_excluded_endpoints(self):
        """Filter should allow endpoints not in the excluded list."""
        filter = UvicornAccessLogFilter(excluded_paths=["/health", "/metrics"])

        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "POST", "/v1/completions", "1.1", 200),
            exc_info=None,
        )

        assert filter.filter(record) is True

    def test_filter_excludes_multiple_endpoints(self):
        """Filter should exclude multiple configured endpoints."""
        filter = UvicornAccessLogFilter(excluded_paths=["/health", "/metrics", "/ping"])

        # Test /health
        record_health = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "GET", "/health", "1.1", 200),
            exc_info=None,
        )
        assert filter.filter(record_health) is False

        # Test /metrics
        record_metrics = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "GET", "/metrics", "1.1", 200),
            exc_info=None,
        )
        assert filter.filter(record_metrics) is False

        # Test /ping
        record_ping = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "GET", "/ping", "1.1", 200),
            exc_info=None,
        )
        assert filter.filter(record_ping) is False

    def test_filter_with_query_parameters(self):
        """Filter should exclude endpoints even with query parameters."""
        filter = UvicornAccessLogFilter(excluded_paths=["/health"])

        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "GET", "/health?verbose=true", "1.1", 200),
            exc_info=None,
        )

        assert filter.filter(record) is False

    def test_filter_different_http_methods(self):
        """Filter should exclude endpoints regardless of HTTP method."""
        filter = UvicornAccessLogFilter(excluded_paths=["/ping"])

        # Test GET
        record_get = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "GET", "/ping", "1.1", 200),
            exc_info=None,
        )
        assert filter.filter(record_get) is False

        # Test POST
        record_post = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:12345", "POST", "/ping", "1.1", 200),
            exc_info=None,
        )
        assert filter.filter(record_post) is False

    def test_filter_with_different_status_codes(self):
        """Filter should exclude endpoints regardless of status code."""
        filter = UvicornAccessLogFilter(excluded_paths=["/health"])

        for status_code in [200, 500, 503]:
            record = logging.LogRecord(
                name="uvicorn.access",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg='%s - "%s %s HTTP/%s" %d',
                args=("127.0.0.1:12345", "GET", "/health", "1.1", status_code),
                exc_info=None,
            )
            assert filter.filter(record) is False


class TestCreateUvicornLogConfig:
    """Test cases for create_uvicorn_log_config function."""

    def test_creates_valid_config_structure(self):
        """Config should have required logging configuration keys."""
        config = create_uvicorn_log_config(excluded_paths=["/health"])

        assert "version" in config
        assert config["version"] == 1
        assert "disable_existing_loggers" in config
        assert "formatters" in config
        assert "handlers" in config
        assert "loggers" in config
        assert "filters" in config

    def test_config_includes_access_log_filter(self):
        """Config should include the access log filter."""
        config = create_uvicorn_log_config(excluded_paths=["/health", "/metrics"])

        assert "access_log_filter" in config["filters"]
        filter_config = config["filters"]["access_log_filter"]
        assert filter_config["()"] == UvicornAccessLogFilter
        assert filter_config["excluded_paths"] == ["/health", "/metrics"]

    def test_config_applies_filter_to_access_handler(self):
        """Config should apply the filter to the access handler."""
        config = create_uvicorn_log_config(excluded_paths=["/health"])

        assert "access" in config["handlers"]
        assert "filters" in config["handlers"]["access"]
        assert "access_log_filter" in config["handlers"]["access"]["filters"]

    def test_config_with_custom_log_level(self):
        """Config should respect custom log level."""
        config = create_uvicorn_log_config(
            excluded_paths=["/health"], log_level="debug"
        )

        assert config["loggers"]["uvicorn"]["level"] == "DEBUG"
        assert config["loggers"]["uvicorn.access"]["level"] == "DEBUG"
        assert config["loggers"]["uvicorn.error"]["level"] == "DEBUG"

    def test_config_with_empty_excluded_paths(self):
        """Config should work with empty excluded paths."""
        config = create_uvicorn_log_config(excluded_paths=[])

        assert config["filters"]["access_log_filter"]["excluded_paths"] == []

    def test_config_with_none_excluded_paths(self):
        """Config should work with None excluded paths."""
        config = create_uvicorn_log_config(excluded_paths=None)

        assert config["filters"]["access_log_filter"]["excluded_paths"] == []


class TestIntegration:
    """Integration tests for the access log filter."""

    def test_filter_with_real_logger(self):
        """Test filter works with a real Python logger simulating uvicorn."""
        # Create a logger with our filter (simulating uvicorn.access)
        logger = logging.getLogger("uvicorn.access")
        logger.setLevel(logging.INFO)

        # Clear any existing handlers
        logger.handlers = []

        # Create a custom handler that tracks messages
        logged_messages: list[str] = []

        class TrackingHandler(logging.Handler):
            def emit(self, record):
                logged_messages.append(record.getMessage())

        handler = TrackingHandler()
        handler.setLevel(logging.INFO)
        filter = UvicornAccessLogFilter(excluded_paths=["/health", "/metrics"])
        handler.addFilter(filter)
        logger.addHandler(handler)

        # Log using uvicorn's format with args tuple
        # Format: '%s - "%s %s HTTP/%s" %d'
        logger.info(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1:12345",
            "GET",
            "/health",
            "1.1",
            200,
        )
        logger.info(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1:12345",
            "GET",
            "/v1/completions",
            "1.1",
            200,
        )
        logger.info(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1:12345",
            "GET",
            "/metrics",
            "1.1",
            200,
        )
        logger.info(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1:12345",
            "POST",
            "/v1/chat/completions",
            "1.1",
            200,
        )

        # Verify only non-excluded endpoints were logged
        assert len(logged_messages) == 2
        assert "/v1/completions" in logged_messages[0]
        assert "/v1/chat/completions" in logged_messages[1]

    def test_filter_allows_non_uvicorn_access_logs(self):
        """Test filter allows logs from non-uvicorn.access loggers."""
        filter = UvicornAccessLogFilter(excluded_paths=["/health"])

        # Log record from a different logger name
        record = logging.LogRecord(
            name="uvicorn.error",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Some error message about /health",
            args=(),
            exc_info=None,
        )

        # Should allow because it's not from uvicorn.access
        assert filter.filter(record) is True

    def test_filter_handles_malformed_args(self):
        """Test filter handles log records with unexpected args format."""
        filter = UvicornAccessLogFilter(excluded_paths=["/health"])

        # Log record with insufficient args
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Some message",
            args=("only", "two"),
            exc_info=None,
        )

        # Should allow because args doesn't have expected format
        assert filter.filter(record) is True

    def test_filter_handles_non_tuple_args(self):
        """Test filter handles log records with non-tuple args."""
        filter = UvicornAccessLogFilter(excluded_paths=["/health"])

        # Log record with None args
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Some message without args",
            args=None,
            exc_info=None,
        )

        # Should allow because args is None
        assert filter.filter(record) is True


class TestGetExcludedPaths:
    """Test cases for the _get_excluded_paths helper."""

    def test_no_flags_set(self):
        """Returns empty list when neither flag is set."""
        args = Namespace(
            disable_uvicorn_metrics_access_log=False,
            disable_access_log_for_endpoints=None,
        )
        assert _get_excluded_paths(args) == []

    def test_metrics_flag_only(self):
        """Returns /health and /metrics when shorthand flag is set."""
        args = Namespace(
            disable_uvicorn_metrics_access_log=True,
            disable_access_log_for_endpoints=None,
        )
        assert _get_excluded_paths(args) == ["/health", "/metrics"]

    def test_endpoints_flag_only(self):
        """Returns parsed paths when only endpoints flag is set."""
        args = Namespace(
            disable_uvicorn_metrics_access_log=False,
            disable_access_log_for_endpoints="/ping,/ready",
        )
        assert _get_excluded_paths(args) == ["/ping", "/ready"]

    def test_both_flags_merged_no_duplicates(self):
        """Merges both flags and removes duplicate paths."""
        args = Namespace(
            disable_uvicorn_metrics_access_log=True,
            disable_access_log_for_endpoints="/metrics,/ping",
        )
        paths = _get_excluded_paths(args)
        # /health and /metrics from shorthand, /ping from endpoints,
        # /metrics should not be duplicated
        assert paths == ["/health", "/metrics", "/ping"]

    def test_endpoints_flag_with_whitespace(self):
        """Handles whitespace around comma-separated paths."""
        args = Namespace(
            disable_uvicorn_metrics_access_log=False,
            disable_access_log_for_endpoints=" /health , /ping , ",
        )
        assert _get_excluded_paths(args) == ["/health", "/ping"]


class TestGetUvicornLogConfigWithMetricsFlag:
    """Test get_uvicorn_log_config with --disable-uvicorn-metrics-access-log."""

    def test_returns_none_when_no_flags(self):
        """Returns None when no filtering or log config is requested."""
        args = Namespace(
            log_config_file=None,
            disable_uvicorn_metrics_access_log=False,
            disable_access_log_for_endpoints=None,
            uvicorn_log_level="info",
        )
        assert get_uvicorn_log_config(args) is None

    def test_returns_config_with_metrics_flag(self):
        """Returns a log config filtering /health and /metrics."""
        args = Namespace(
            log_config_file=None,
            disable_uvicorn_metrics_access_log=True,
            disable_access_log_for_endpoints=None,
            uvicorn_log_level="info",
        )
        config = get_uvicorn_log_config(args)
        assert config is not None
        assert "filters" in config
        filter_cfg = config["filters"]["access_log_filter"]
        assert "/health" in filter_cfg["excluded_paths"]
        assert "/metrics" in filter_cfg["excluded_paths"]

    def test_metrics_flag_combined_with_endpoints(self):
        """Merges metrics shorthand with explicit endpoint list."""
        args = Namespace(
            log_config_file=None,
            disable_uvicorn_metrics_access_log=True,
            disable_access_log_for_endpoints="/ping",
            uvicorn_log_level="info",
        )
        config = get_uvicorn_log_config(args)
        assert config is not None
        excluded = config["filters"]["access_log_filter"]["excluded_paths"]
        assert excluded == ["/health", "/metrics", "/ping"]
