# SPDX-License-Identifier: Apache-2.0
"""Tests for the BenchCommand CLI wiring and orchestrator."""

# Standard
from unittest.mock import AsyncMock, MagicMock, patch
import argparse
import io
import json
import sys
import time

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench import BenchCommand
from lmcache.cli.commands.bench.engine_bench.command import (
    _emit_final_metrics,
    _resolve_args,
    run_engine_bench,
)
from lmcache.cli.commands.bench.engine_bench.config import EngineBenchConfig
from lmcache.cli.commands.bench.engine_bench.stats import (
    FinalStats,
    RequestResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        bench_target="engine",
        engine_url="http://localhost:8000",
        lmcache_url=None,
        model="test-model",
        workload="long-doc-qa",
        kv_cache_volume=100.0,
        tokens_per_gb_kvcache=50000,
        seed=42,
        output_dir=".",
        no_csv=False,
        json=False,
        quiet=True,
        ignore_eos=False,
        ldqa_document_length=100,
        ldqa_query_per_document=1,
        ldqa_shuffle_policy="tile",
        ldqa_num_inflight_requests=1,
        ldqa_max_output_length=128,
        mrc_shared_prompt_length=2000,
        mrc_chat_history_length=10000,
        mrc_user_input_length=50,
        mrc_output_length=200,
        mrc_qps=1.0,
        mrc_duration=60.0,
        rp_request_length=10000,
        rp_num_requests=50,
        format=None,
        output=None,
        config=None,
        no_interactive=False,
        export_config=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_config(**overrides) -> EngineBenchConfig:
    defaults = dict(
        engine_url="http://localhost:8000",
        model="test-model",
        workload="long-doc-qa",
        kv_cache_volume_gb=100.0,
        tokens_per_gb_kvcache=50000,
        seed=42,
        output_dir=".",
        export_csv=True,
        export_json=False,
        quiet=True,
    )
    defaults.update(overrides)
    return EngineBenchConfig(**defaults)  # type: ignore[arg-type]


def _make_final_stats(**overrides) -> FinalStats:
    defaults: dict[str, int | float] = dict(
        total_requests=10,
        successful_requests=10,
        failed_requests=0,
        elapsed_time=5.0,
        mean_ttft_ms=300.0,
        mean_decode_speed=48.0,
        mean_request_latency_ms=2000.0,
        input_throughput=20000.0,
        output_throughput=256.0,
        total_input_tokens=100000,
        total_output_tokens=1280,
        p50_ttft_ms=280.0,
        p90_ttft_ms=450.0,
        p99_ttft_ms=600.0,
        p50_decode_speed=47.0,
        p90_decode_speed=42.0,
        p99_decode_speed=38.0,
        p50_request_latency_ms=1900.0,
        p90_request_latency_ms=2500.0,
        p99_request_latency_ms=3000.0,
    )
    defaults.update(overrides)
    return FinalStats(**defaults)  # type: ignore[arg-type]


def _make_result(request_id: str = "req_0") -> RequestResult:
    now = time.time()
    return RequestResult(
        request_id=request_id,
        successful=True,
        ttft=0.3,
        request_latency=2.0,
        num_input_tokens=100,
        num_output_tokens=10,
        decode_speed=25.0,
        submit_time=now,
        first_token_time=now + 0.3,
        finish_time=now + 2.0,
        error="",
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestBenchCommandRegistration:
    def test_bench_in_all_commands(self) -> None:
        # First Party
        from lmcache.cli.commands import ALL_COMMANDS

        names = [cmd.name() for cmd in ALL_COMMANDS]
        assert "bench" in names

    def test_engine_subparser_exists(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        cmd = BenchCommand()
        cmd.register(subparsers)

        args = parser.parse_args(
            [
                "bench",
                "engine",
                "--engine-url",
                "http://localhost:8000",
                "--workload",
                "long-doc-qa",
                "--tokens-per-gb-kvcache",
                "50000",
            ]
        )
        assert args.bench_target == "engine"
        assert args.engine_url == "http://localhost:8000"
        assert args.workload == "long-doc-qa"
        assert args.tokens_per_gb_kvcache == 50000

    def test_optional_args_parse_without_engine_url(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        cmd = BenchCommand()
        cmd.register(subparsers)

        # engine-url and workload are now optional (for interactive mode)
        args = parser.parse_args(["bench", "engine"])
        assert args.engine_url is None
        assert args.workload is None

    def test_config_flag_accepted(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        cmd = BenchCommand()
        cmd.register(subparsers)

        args = parser.parse_args(["bench", "engine", "--config", "my_config.json"])
        assert args.config == "my_config.json"

    def test_default_values(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        cmd = BenchCommand()
        cmd.register(subparsers)

        args = parser.parse_args(
            [
                "bench",
                "engine",
                "--engine-url",
                "http://localhost:8000",
                "--workload",
                "long-doc-qa",
                "--tokens-per-gb-kvcache",
                "50000",
            ]
        )
        assert args.kv_cache_volume == 100.0
        assert args.seed == 42
        assert args.output_dir == "."
        assert args.ldqa_document_length == 10000
        assert args.ldqa_query_per_document == 2
        assert args.ldqa_shuffle_policy == "random"
        assert args.ldqa_num_inflight_requests == 3
        assert args.quiet is False


# ---------------------------------------------------------------------------
# --no-interactive
# ---------------------------------------------------------------------------


class TestNoInteractive:
    def test_no_interactive_flag_accepted(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        cmd = BenchCommand()
        cmd.register(subparsers)

        args = parser.parse_args(["bench", "engine", "--no-interactive"])
        assert args.no_interactive is True

    def test_no_interactive_missing_engine_url(self) -> None:
        args = _make_args(
            engine_url=None,
            workload="long-doc-qa",
            tokens_per_gb_kvcache=6553,
            no_interactive=True,
        )
        with pytest.raises(SystemExit, match="--engine-url"):
            _resolve_args(args)

    def test_no_interactive_missing_workload(self) -> None:
        args = _make_args(
            engine_url="http://localhost:8000",
            workload=None,
            tokens_per_gb_kvcache=6553,
            no_interactive=True,
        )
        with pytest.raises(SystemExit, match="--workload"):
            _resolve_args(args)

    def test_no_interactive_missing_tokens_and_lmcache(self) -> None:
        args = _make_args(
            engine_url="http://localhost:8000",
            workload="long-doc-qa",
            tokens_per_gb_kvcache=None,
            lmcache_url=None,
            no_interactive=True,
        )
        with pytest.raises(
            SystemExit, match="--tokens-per-gb-kvcache or --lmcache-url"
        ):
            _resolve_args(args)

    def test_no_interactive_passes_with_all_args(self) -> None:
        args = _make_args(no_interactive=True)
        result = _resolve_args(args)
        assert result is args

    def test_no_interactive_passes_with_lmcache_url(self) -> None:
        args = _make_args(
            tokens_per_gb_kvcache=None,
            lmcache_url="http://localhost:8080",
            no_interactive=True,
        )
        result = _resolve_args(args)
        assert result is args


# ---------------------------------------------------------------------------
# --export-config
# ---------------------------------------------------------------------------


class TestExportConfig:
    def test_export_config_flag_accepted(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        cmd = BenchCommand()
        cmd.register(subparsers)

        args = parser.parse_args(
            [
                "bench",
                "engine",
                "--engine-url",
                "http://localhost:8000",
                "--workload",
                "long-doc-qa",
                "--tokens-per-gb-kvcache",
                "6553",
                "--export-config",
                "out.json",
            ]
        )
        assert args.export_config == "out.json"

    def test_export_config_errors_when_missing_args(self) -> None:
        args = _make_args(
            engine_url=None,
            export_config="out.json",
        )
        with pytest.raises(SystemExit, match="--engine-url"):
            _resolve_args(args)

    def test_export_config_writes_json(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        export_path = str(tmp_path / "exported.json")
        args = _make_args(
            export_config=export_path,
            quiet=True,
        )

        cmd = BenchCommand()
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            run_engine_bench(cmd, args)
        finally:
            sys.stdout = old_stdout

        with open(export_path) as f:
            data = json.load(f)

        assert "engine_url" not in data
        assert data["workload"] == "long-doc-qa"
        assert data["tokens_per_gb_kvcache"] == 50000
        assert "lmcache_url" not in data

    def test_max_output_length_rejected_for_unsupported_workload(
        self,
        tmp_path,  # type: ignore[no-untyped-def]
    ) -> None:
        # Setting a non-default max output length for a workload without that
        # parameter is rejected.
        args = _make_args(
            workload="random-prefill",
            ldqa_max_output_length=512,
            export_config=str(tmp_path / "exported.json"),
        )
        with pytest.raises(ValueError, match="max output length cannot be specified"):
            run_engine_bench(BenchCommand(), args)

    def test_export_config_excludes_lmcache_url(
        self,
        tmp_path,  # type: ignore[no-untyped-def]
    ) -> None:
        export_path = str(tmp_path / "exported.json")
        args = _make_args(
            lmcache_url="http://localhost:8080",
            export_config=export_path,
            quiet=True,
        )

        cmd = BenchCommand()
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            run_engine_bench(cmd, args)
        finally:
            sys.stdout = old_stdout

        with open(export_path) as f:
            data = json.load(f)

        assert "lmcache_url" not in data
        assert data["tokens_per_gb_kvcache"] == 50000

    def test_export_config_includes_workload_args(
        self,
        tmp_path,  # type: ignore[no-untyped-def]
    ) -> None:
        export_path = str(tmp_path / "exported.json")
        args = _make_args(
            workload="long-doc-qa",
            ldqa_document_length=5000,
            ldqa_query_per_document=4,
            export_config=export_path,
            quiet=True,
        )

        cmd = BenchCommand()
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            run_engine_bench(cmd, args)
        finally:
            sys.stdout = old_stdout

        with open(export_path) as f:
            data = json.load(f)

        assert data["ldqa_document_length"] == 5000
        assert data["ldqa_query_per_document"] == 4


# ---------------------------------------------------------------------------
# Final metrics emission
# ---------------------------------------------------------------------------


class TestBenchCommandEmitMetrics:
    def test_emit_final_metrics_terminal(self) -> None:
        cmd = BenchCommand()
        config = _make_config()
        final = _make_final_stats()
        args = _make_args(quiet=False)

        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            _emit_final_metrics(cmd, config, final, args)
        finally:
            sys.stdout = old_stdout

        output = buf.getvalue()
        assert "Engine Benchmark Result" in output
        assert "Configuration" in output
        assert "Time to First Token" in output
        assert "Decoding Speed" in output
        assert "test-model" in output

    def test_emit_final_metrics_json(self) -> None:
        cmd = BenchCommand()
        config = _make_config()
        final = _make_final_stats()
        args = _make_args(quiet=False, format="json")

        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            _emit_final_metrics(cmd, config, final, args)
        finally:
            sys.stdout = old_stdout

        data = json.loads(buf.getvalue())
        assert "metrics" in data
        assert "ttft" in data["metrics"]
        assert "decode" in data["metrics"]
        assert data["metrics"]["config"]["model"] == "test-model"
        assert data["metrics"]["results"]["successful"] == 10


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class TestBenchCommandOrchestrator:
    @patch(
        "lmcache.cli.commands.bench.engine_bench.config.resolve_tokens_per_gb",
        return_value=6553,
    )
    @patch(
        "lmcache.cli.commands.bench.engine_bench.request_sender.AsyncOpenAI",
    )
    def test_lmcache_url_resolves(
        self,
        mock_openai_cls,
        mock_resolve,
        tmp_path,
    ) -> None:
        """When --lmcache-url is set, tokens_per_gb is resolved from server."""
        # Third Party
        from openai.types import CompletionUsage
        from openai.types.chat import ChatCompletionChunk
        from openai.types.chat.chat_completion_chunk import (
            Choice,
            ChoiceDelta,
        )

        usage = CompletionUsage(
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
        )

        async def _fake_stream(*_args, **_kwargs):
            yield ChatCompletionChunk(
                id="c1",
                choices=[
                    Choice(
                        delta=ChoiceDelta(content="Hi"),
                        index=0,
                    )
                ],
                created=0,
                model="test-model",
                object="chat.completion.chunk",
            )
            yield ChatCompletionChunk(
                id="c1",
                choices=[],
                created=0,
                model="test-model",
                object="chat.completion.chunk",
                usage=usage,
            )

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            side_effect=lambda **kw: _fake_stream(),
        )
        mock_client.close = AsyncMock()

        args = _make_args(
            lmcache_url="http://localhost:8080",
            tokens_per_gb_kvcache=None,
            kv_cache_volume=0.001,
            ldqa_document_length=100,
            ldqa_query_per_document=1,
            ldqa_num_inflight_requests=1,
            output_dir=str(tmp_path),
            no_csv=True,
        )

        cmd = BenchCommand()
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            run_engine_bench(cmd, args)
        finally:
            sys.stdout = old_stdout

        mock_resolve.assert_called_once_with(
            "http://localhost:8080",
            "test-model",
        )

    @patch(
        "lmcache.cli.commands.bench.engine_bench.request_sender.AsyncOpenAI",
    )
    def test_bench_engine_wiring(
        self,
        mock_openai_cls,
        tmp_path,
    ) -> None:
        """End-to-end orchestrator test with mocked OpenAI client."""
        # Third Party
        from openai.types import CompletionUsage
        from openai.types.chat import ChatCompletionChunk
        from openai.types.chat.chat_completion_chunk import (
            Choice,
            ChoiceDelta,
        )

        def _make_chunk(content="", usage=None):
            choices = []
            if content:
                choices.append(
                    Choice(
                        delta=ChoiceDelta(content=content),
                        index=0,
                    )
                )
            return ChatCompletionChunk(
                id="c1",
                choices=choices,
                created=0,
                model="test-model",
                object="chat.completion.chunk",
                usage=usage,
            )

        usage = CompletionUsage(
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
        )

        async def _fake_stream(*_args, **_kwargs):
            yield _make_chunk(content="Hello")
            yield _make_chunk(content=" world")
            yield _make_chunk(usage=usage)

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            side_effect=lambda **kw: _fake_stream(),
        )
        mock_client.close = AsyncMock()

        output_dir = str(tmp_path)
        args = _make_args(
            kv_cache_volume=0.001,
            tokens_per_gb_kvcache=1000,
            ldqa_document_length=100,
            ldqa_query_per_document=1,
            ldqa_num_inflight_requests=1,
            output_dir=output_dir,
            no_csv=False,
            json=True,
            quiet=True,
        )

        cmd = BenchCommand()
        # Suppress stdout for metrics emission
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            run_engine_bench(cmd, args)
        finally:
            sys.stdout = old_stdout

        # Verify CSV was written
        csv_path = tmp_path / "bench_results.csv"
        assert csv_path.exists()

        # Verify JSON was written
        json_path = tmp_path / "bench_summary.json"
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert "config" in data
        assert "results" in data
        assert data["results"]["total_requests"] > 0

    @patch(
        "lmcache.cli.commands.bench.engine_bench.request_sender.AsyncOpenAI",
    )
    def test_config_file_loading(
        self,
        mock_openai_cls,
        tmp_path,
    ) -> None:
        """Benchmark runs correctly from a --config JSON file."""
        # Third Party
        from openai.types import CompletionUsage
        from openai.types.chat import ChatCompletionChunk
        from openai.types.chat.chat_completion_chunk import (
            Choice,
            ChoiceDelta,
        )

        def _make_chunk(content="", usage=None):
            choices = []
            if content:
                choices.append(
                    Choice(
                        delta=ChoiceDelta(content=content),
                        index=0,
                    )
                )
            return ChatCompletionChunk(
                id="c1",
                choices=choices,
                created=0,
                model="test-model",
                object="chat.completion.chunk",
                usage=usage,
            )

        usage = CompletionUsage(
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
        )

        async def _fake_stream(*_args, **_kwargs):
            yield _make_chunk(content="Hello")
            yield _make_chunk(usage=usage)

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            side_effect=lambda **kw: _fake_stream(),
        )
        mock_client.close = AsyncMock()

        # Write a config JSON file
        config_data = {
            "engine_url": "http://localhost:8000",
            "model": "test-model",
            "workload": "long-doc-qa",
            "tokens_per_gb_kvcache": 1000,
            "kv_cache_volume": 0.001,
            "ldqa_document_length": 100,
            "ldqa_query_per_document": 1,
            "ldqa_shuffle_policy": "tile",
            "ldqa_num_inflight_requests": 1,
        }
        config_path = tmp_path / "test_config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        args = _make_args(
            config=str(config_path),
            engine_url=None,
            workload=None,
            tokens_per_gb_kvcache=None,
            output_dir=str(tmp_path),
            no_csv=True,
            quiet=True,
        )

        cmd = BenchCommand()
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            run_engine_bench(cmd, args)
        finally:
            sys.stdout = old_stdout

        # Verify benchmark ran — sender was called
        assert mock_client.chat.completions.create.call_count > 0
