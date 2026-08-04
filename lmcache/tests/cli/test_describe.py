# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache describe kvcache``."""

# Standard
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest.mock import patch
import json

# Third Party
import pytest

# First Party
from lmcache.cli.commands.describe import (
    DescribeError,
    fetch_health,
    fetch_json,
    fetch_running_requests,
    fmt_health,
    normalize_url,
    safe_get,
)

# ---------------------------------------------------------------------------
# Sample /status payload
# ---------------------------------------------------------------------------

SAMPLE_STATUS = {
    "is_healthy": True,
    "engine_type": "MPCacheServer",
    "chunk_size": 256,
    "hash_algorithm": "sha256",
    "registered_gpu_ids": [0],
    "cache_context_meta": {
        "0": {
            "model_name": "llama",
            "world_size": 1,
            "kv_cache_layout": {
                "num_layers": 32,
                "num_blocks": 2048,
                "cache_size_per_token": 163840,
                "kernel_groups": [
                    {
                        "kernel_group_idx": 0,
                        "engine_group_idx": 0,
                        "object_group_idx": 0,
                        "num_layers": 32,
                        "layer_indices": list(range(32)),
                        "tokens_per_block": 16,
                        "slots_per_block": 16,
                        "dtype": "torch.float16",
                        "engine_kv_concrete_shape": "32 x [2, 2048, 16, 8, 128]",
                        "is_mla": False,
                        "engine_kv_format": "NL_X_TWO_NB_BS_NH_HS",
                        "engine_kv_shape": "NL x [2, NB, BS, NH, HS]",
                        "attention_backend": "vLLM non-MLA flash attention",
                    },
                ],
            },
        },
    },
    "active_sessions": 3,
    "storage_manager": {
        "is_healthy": True,
        "l1_manager": {
            "is_healthy": True,
            "total_object_count": 1024,
            "write_locked_count": 0,
            "read_locked_count": 0,
            "temporary_count": 0,
            "memory_used_bytes": 45_415_895_859,  # ~42.30 GB
            "memory_total_bytes": 64_424_509_440,  # ~60.00 GB
            "memory_usage_ratio": 0.705,
            "write_ttl_seconds": 10,
            "read_ttl_seconds": 10,
        },
        "store_controller": {"is_healthy": True},
        "prefetch_controller": {"is_healthy": True},
        "eviction_controller": {
            "is_healthy": True,
            "thread_alive": True,
            "eviction_policy": "LRU",
            "trigger_watermark": 0.9,
            "eviction_ratio": 0.2,
        },
        "l2_adapters": [],
        "num_l2_adapters": 0,
    },
}


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestNormalizeUrl:
    def test_adds_http(self):
        assert normalize_url("localhost:8000") == "http://localhost:8000"

    def test_preserves_http(self):
        assert normalize_url("http://localhost:8000") == "http://localhost:8000"

    def test_preserves_https(self):
        assert normalize_url("https://host:443") == "https://host:443"

    def test_strips_trailing_slash(self):
        assert normalize_url("http://host:8000/") == "http://host:8000"

    def test_strips_multiple_trailing_slashes(self):
        assert normalize_url("http://host:8000///") == "http://host:8000"


class TestFmtHealth:
    def test_healthy(self):
        assert fmt_health(True) == "OK"

    def test_unhealthy(self):
        assert fmt_health(False) == "UNHEALTHY"

    def test_none(self):
        assert fmt_health(None) is None


class TestSafeGet:
    def test_nested(self):
        d = {"a": {"b": {"c": 42}}}
        assert safe_get(d, "a", "b", "c") == 42

    def test_missing_key(self):
        d = {"a": {"b": 1}}
        assert safe_get(d, "a", "x") is None

    def test_missing_key_with_default(self):
        d = {"a": 1}
        assert safe_get(d, "a", "b", default="N/A") == "N/A"

    def test_non_dict_intermediate(self):
        d = {"a": 5}
        assert safe_get(d, "a", "b") is None


# ---------------------------------------------------------------------------
# Field extraction integration test
# ---------------------------------------------------------------------------


class TestDescribeKvcacheFields:
    """Test that ``_describe_kvcache`` extracts fields correctly from a
    sample ``/status`` response."""

    def test_field_extraction(self):
        """Verify metrics are populated from the sample status dict."""
        # First Party
        from lmcache.cli.commands.describe import DescribeCommand

        cmd = DescribeCommand()

        class FakeArgs:
            target = "kvcache"
            url = "http://localhost:8000"
            format = "json"
            output = None

        # Patch fetch_json to return our sample data
        with patch(
            "lmcache.cli.commands.describe.fetch_json",
            return_value=SAMPLE_STATUS,
        ):
            # Capture the JSON output
            # Standard
            import io

            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd.execute(FakeArgs())

            output = json.loads(buf.getvalue())

        m = output["metrics"]
        assert m["health"] == "OK"
        assert m["url"] == "http://localhost:8000"
        assert m["engine_type"] == "MPCacheServer"
        assert m["chunk_size"] == 256
        assert m["l1_capacity_gb"] == 60.0
        assert m["l1_used_gb"] == "42.30 (70.5%)"
        assert m["eviction_policy"] == "LRU"
        assert m["cached_objects"] == 1024
        assert m["active_sessions"] == 3

        # Per-model section (list)
        assert "models" in m
        model = m["models"][0]
        assert model["model"] == "llama"
        assert model["world_size"] == 1
        assert model["gpu_ids"] == "0"
        assert model["num_layers"] == 32
        assert model["num_blocks"] == 2048
        assert model["cache_size_per_token"] == 163840

        # Per-kernel-group section (list)
        assert "kernel_groups" in m
        kg = m["kernel_groups"][0]
        assert kg["model"] == "llama"
        assert kg["kernel_group_idx"] == 0
        assert kg["engine_group_idx"] == 0
        assert kg["object_group_idx"] == 0
        assert kg["num_layers"] == 32
        assert kg["slots_per_block"] == 16
        assert kg["dtype"] == "torch.float16"
        assert kg["is_mla"] is False
        assert kg["attention_backend"] == "vLLM non-MLA flash attention"
        assert kg["engine_kv_shape"] == "NL x [2, NB, BS, NH, HS]"
        assert kg["engine_kv_concrete_shape"] == "32 x [2, 2048, 16, 8, 128]"

    def test_unhealthy(self):
        """Verify health shows UNHEALTHY when is_healthy is False."""
        # First Party
        from lmcache.cli.commands.describe import DescribeCommand

        unhealthy_data = {**SAMPLE_STATUS, "is_healthy": False}
        cmd = DescribeCommand()

        class FakeArgs:
            target = "kvcache"
            url = "http://localhost:8000"
            format = "json"
            output = None

        with patch(
            "lmcache.cli.commands.describe.fetch_json",
            return_value=unhealthy_data,
        ):
            # Standard
            import io

            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd.execute(FakeArgs())

            output = json.loads(buf.getvalue())

        assert output["metrics"]["health"] == "UNHEALTHY"

    def test_missing_fields_show_na(self):
        """Verify missing nested fields render as None (N/A in terminal)."""
        # First Party
        from lmcache.cli.commands.describe import DescribeCommand

        minimal_data = {"is_healthy": True, "engine_type": "MPCacheServer"}
        cmd = DescribeCommand()

        class FakeArgs:
            target = "kvcache"
            url = "http://localhost:8000"
            format = "json"
            output = None

        with patch(
            "lmcache.cli.commands.describe.fetch_json",
            return_value=minimal_data,
        ):
            # Standard
            import io

            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd.execute(FakeArgs())

            output = json.loads(buf.getvalue())

        m = output["metrics"]
        assert m["l1_capacity_gb"] is None
        assert m["l1_used_gb"] is None
        assert m["eviction_policy"] is None
        assert m["cached_objects"] is None
        assert m["active_sessions"] is None


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestDescribeErrors:
    def test_connection_refused(self):
        """Verify sys.exit(1) on connection error."""
        # First Party
        from lmcache.cli.commands.describe import DescribeCommand

        cmd = DescribeCommand()

        class FakeArgs:
            target = "kvcache"
            url = "http://localhost:19999"
            format = None
            output = None

        with pytest.raises(SystemExit) as exc_info:
            cmd.execute(FakeArgs())

        assert exc_info.value.code == 1

    def test_503_error(self):
        """Verify sys.exit(1) on HTTP 503."""
        # First Party
        from lmcache.cli.commands.describe import DescribeCommand

        cmd = DescribeCommand()

        class FakeArgs:
            target = "kvcache"
            url = "http://localhost:8000"
            format = None
            output = None

        with patch(
            "lmcache.cli.commands.describe.fetch_json",
            side_effect=DescribeError("Server unhealthy: engine not initialized"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cmd.execute(FakeArgs())

            assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# fetch_json with a real HTTP server
# ---------------------------------------------------------------------------


class _MockHandler(BaseHTTPRequestHandler):
    """Minimal handler that serves a canned JSON response."""

    response_body: bytes = b"{}"
    response_code: int = 200

    def do_GET(self):
        self.send_response(self.response_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.response_body)

    def log_message(self, format, *args):
        pass  # suppress stderr noise


class TestFetchJson:
    def test_success(self):
        handler = type(
            "_H",
            (_MockHandler,),
            {
                "response_body": json.dumps({"ok": True}).encode(),
                "response_code": 200,
            },
        )
        server = HTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        t = Thread(target=server.handle_request, daemon=True)
        t.start()
        try:
            result = fetch_json(f"http://127.0.0.1:{port}/status")
            assert result == {"ok": True}
        finally:
            server.server_close()

    def test_503(self):
        handler = type(
            "_H",
            (_MockHandler,),
            {
                "response_body": json.dumps(
                    {"error": "engine not initialized"}
                ).encode(),
                "response_code": 503,
            },
        )
        server = HTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        t = Thread(target=server.handle_request, daemon=True)
        t.start()
        try:
            with pytest.raises(DescribeError, match="Server unhealthy"):
                fetch_json(f"http://127.0.0.1:{port}/status")
        finally:
            server.server_close()


class TestFetchHealth:
    def test_ok(self):
        handler = type(
            "_H",
            (_MockHandler,),
            {"response_body": b"", "response_code": 200},
        )
        server = HTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        t = Thread(target=server.handle_request, daemon=True)
        t.start()
        try:
            assert fetch_health(f"http://127.0.0.1:{port}/health") is True
        finally:
            server.server_close()

    def test_non_200(self):
        handler = type(
            "_H",
            (_MockHandler,),
            {"response_body": b"", "response_code": 503},
        )
        server = HTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        t = Thread(target=server.handle_request, daemon=True)
        t.start()
        try:
            assert fetch_health(f"http://127.0.0.1:{port}/health") is False
        finally:
            server.server_close()

    def test_connection_refused(self):
        assert fetch_health("http://127.0.0.1:19999/health") is False


class TestFetchRunningRequests:
    def _serve(self, body: bytes, code: int = 200):
        handler = type(
            "_H",
            (_MockHandler,),
            {"response_body": body, "response_code": code},
        )
        server = HTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        t = Thread(target=server.handle_request, daemon=True)
        t.start()
        return server, port

    def test_single_series(self):
        body = (
            b"# HELP vllm:num_requests_running Number of running requests.\n"
            b"# TYPE vllm:num_requests_running gauge\n"
            b'vllm:num_requests_running{model_name="m"} 3.0\n'
        )
        server, port = self._serve(body)
        try:
            assert fetch_running_requests(f"http://127.0.0.1:{port}/metrics") == 3
        finally:
            server.server_close()

    def test_sums_multiple_series(self):
        body = (
            b'vllm:num_requests_running{model_name="a"} 3.0\n'
            b'vllm:num_requests_running{model_name="b"} 1.0\n'
        )
        server, port = self._serve(body)
        try:
            assert fetch_running_requests(f"http://127.0.0.1:{port}/metrics") == 4
        finally:
            server.server_close()

    def test_metric_absent(self):
        body = b"# only other metrics here\nvllm:num_requests_waiting 5.0\n"
        server, port = self._serve(body)
        try:
            assert fetch_running_requests(f"http://127.0.0.1:{port}/metrics") is None
        finally:
            server.server_close()

    def test_connection_refused(self):
        assert fetch_running_requests("http://127.0.0.1:19999/metrics") is None


# ---------------------------------------------------------------------------
# describe engine
# ---------------------------------------------------------------------------

SAMPLE_MODELS = {
    "object": "list",
    "data": [
        {
            "id": "meta-llama/Llama-3.1-8B-Instruct",
            "object": "model",
            "owned_by": "vllm",
            "max_model_len": 131072,
        }
    ],
}


class TestDescribeEngineFields:
    """Test that ``_describe_engine`` extracts fields from a sample
    ``/v1/models`` response plus a ``/health`` result."""

    def _run(self, models, is_healthy):
        """Execute ``describe engine`` with mocked HTTP and return metrics dict."""
        # First Party
        from lmcache.cli.commands.describe import DescribeCommand

        cmd = DescribeCommand()

        class FakeArgs:
            target = "engine"
            url = "http://localhost:8000"
            format = "json"
            output = None

        with (
            patch(
                "lmcache.cli.commands.describe.fetch_json",
                return_value=models,
            ),
            patch(
                "lmcache.cli.commands.describe.fetch_health",
                return_value=is_healthy,
            ),
            patch(
                "lmcache.cli.commands.describe.fetch_running_requests",
                return_value=3,
            ),
        ):
            # Standard
            import io

            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd.execute(FakeArgs())
            return json.loads(buf.getvalue())

    def test_field_extraction(self):
        """Verify model, context, status, and running requests are populated."""
        output = self._run(SAMPLE_MODELS, is_healthy=True)
        assert output["title"] == "Inference Engine"
        m = output["metrics"]
        assert m["model"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert m["max_context"] == 131072
        assert m["status"] == "OK"
        assert m["running_requests"] == 3

    def test_unhealthy(self):
        """Verify status shows UNHEALTHY when /health is not OK."""
        output = self._run(SAMPLE_MODELS, is_healthy=False)
        assert output["metrics"]["status"] == "UNHEALTHY"

    def test_missing_model(self):
        """Verify empty model list renders model/context as None."""
        output = self._run({"object": "list", "data": []}, is_healthy=True)
        m = output["metrics"]
        assert m["model"] is None
        assert m["max_context"] is None


class TestDescribeEngineDefaultUrl:
    def test_engine_default_url(self):
        """Verify url=None resolves to the engine default (localhost:8000)."""
        # First Party
        from lmcache.cli.commands.describe import DescribeCommand

        cmd = DescribeCommand()

        class FakeArgs:
            target = "engine"
            url = None
            format = "json"
            output = None

        captured: dict[str, str] = {}

        def fake_fetch(url, *args, **kwargs):
            captured["url"] = url
            return SAMPLE_MODELS

        with (
            patch(
                "lmcache.cli.commands.describe.fetch_json",
                side_effect=fake_fetch,
            ),
            patch(
                "lmcache.cli.commands.describe.fetch_health",
                return_value=True,
            ),
            patch(
                "lmcache.cli.commands.describe.fetch_running_requests",
                return_value=3,
            ),
        ):
            # Standard
            import io

            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd.execute(FakeArgs())

        assert "8000" in captured["url"]


class TestDescribeEngineErrors:
    def test_error_exits_1(self):
        """Verify sys.exit(1) when the engine cannot be reached."""
        # First Party
        from lmcache.cli.commands.describe import DescribeCommand

        cmd = DescribeCommand()

        class FakeArgs:
            target = "engine"
            url = "http://localhost:8000"
            format = None
            output = None

        with patch(
            "lmcache.cli.commands.describe.fetch_json",
            side_effect=DescribeError("Cannot connect"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cmd.execute(FakeArgs())
            assert exc_info.value.code == 1
