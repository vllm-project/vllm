# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache query`` CLI command."""

# Standard
from unittest.mock import MagicMock, patch
import argparse

# Third Party
import pytest

# First Party
from lmcache.cli.commands.query import QueryCommand


def _engine_metric_map(
    model_id: str = "facebook/opt-125m",
) -> dict[str, tuple[str, object]]:
    """Metric-map portion of :meth:`lmcache.cli.request.Request.send_request`."""
    return {
        "model": ("Model", model_id),
        "prompt_tokens": ("Input tokens", 10),
        "output_tokens": ("Output tokens", 5),
        "ttft_ms": ("TTFT (ms)", 1.0),
        "tpot_ms_per_token": ("TPOT (ms/token)", 2.0),
        "total_latency_ms": ("Total latency (ms)", 100.0),
        "throughput_tokens_per_s": ("Throughput (tokens/s)", 50.0),
    }


@pytest.fixture
def cmd() -> QueryCommand:
    return QueryCommand()


@pytest.fixture
def parser(cmd: QueryCommand) -> argparse.ArgumentParser:
    """An :class:`~argparse.ArgumentParser` with ``QueryCommand`` registered."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    cmd.register(sub)
    return p


class TestQueryCommandMetadata:
    def test_name(self, cmd: QueryCommand) -> None:
        assert cmd.name() == "query"

    def test_help(self, cmd: QueryCommand) -> None:
        assert "inference" in cmd.help().lower()


class TestQueryCommandArguments:
    def test_registers_subcommand(self, parser: argparse.ArgumentParser) -> None:
        """The ``query engine`` subcommand should be parseable."""
        args = parser.parse_args(
            [
                "query",
                "engine",
                "--url",
                "http://localhost:8000/v1",
                "--prompt",
                "hello",
            ],
        )
        assert hasattr(args, "func")

    def test_engine_args_registered(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(
            [
                "query",
                "engine",
                "--url",
                "http://host:9/v1",
                "--prompt",
                "{ffmpeg} test",
                "--model",
                "m",
                "--max-tokens",
                "64",
                "--timeout",
                "5",
                "--documents",
                "a=/tmp/x",
                "--completions",
                "--chat-first",
                "--format",
                "json",
                "--output",
                "/tmp/out",
            ],
        )
        assert args.url == "http://host:9/v1"
        assert args.prompt == "{ffmpeg} test"
        assert args.model == "m"
        assert args.max_tokens == 64
        assert args.timeout == 5.0
        assert args.documents == ["a=/tmp/x"]
        assert args.completions is True
        assert args.chat_first is True
        assert args.format == "json"
        assert args.output == "/tmp/out"

    def test_default_values(self, parser: argparse.ArgumentParser) -> None:
        """Required args only — everything else should get defaults."""
        args = parser.parse_args(
            [
                "query",
                "engine",
                "--url",
                "http://localhost:8000/v1",
                "--prompt",
                "hi",
            ],
        )
        assert args.model is None
        assert args.max_tokens == 128
        assert args.timeout == 30.0
        assert args.documents == []
        assert args.completions is False
        assert args.chat_first is False
        assert args.format is None
        assert args.output is None


class TestQueryCommandExecute:
    def test_func_bound_to_execute(
        self, cmd: QueryCommand, parser: argparse.ArgumentParser
    ) -> None:
        """``parse_args`` should bind ``func`` to the subcommand's execute."""
        args = parser.parse_args(
            [
                "query",
                "engine",
                "--url",
                "http://localhost:8000/v1",
                "--prompt",
                "hello",
                "--model",
                "m",
            ],
        )
        # func is bound to the discovered EngineCommand's execute
        assert callable(args.func)
        assert args.query_target == "engine"

    @patch("lmcache.cli.commands.query.engine_command.Request")
    def test_execute_calls_request_send_request(
        self,
        mock_request_cls: MagicMock,
        cmd: QueryCommand,
        parser: argparse.ArgumentParser,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """query_engine should call Request.send_request with the expanded prompt."""
        mock_instance = MagicMock()
        mock_instance.send_request.return_value = ("Paris.", _engine_metric_map())
        mock_request_cls.return_value = mock_instance

        args = parser.parse_args(
            [
                "query",
                "engine",
                "--url",
                "http://localhost:8000/v1",
                "--prompt",
                "hello",
                "--model",
                "facebook/opt-125m",
            ],
        )
        cmd.execute(args)

        mock_request_cls.assert_called_once_with(
            base="http://localhost:8000/v1",
            model="facebook/opt-125m",
            max_tokens=128,
            timeout=30.0,
            completions_only=False,
            chat_first=False,
        )
        mock_instance.send_request.assert_called_once_with("hello")

        out = capsys.readouterr().out
        assert "Query Engine" in out
        assert "facebook/opt-125m" in out
        assert "Input tokens" in out
        assert "Prompt tokens" not in out
        assert "Answer" in out
        assert "Paris." in out

    @patch("lmcache.cli.commands.query.engine_command.Request")
    def test_execute_uses_engine_model_when_cli_model_omitted(
        self,
        mock_request_cls: MagicMock,
        cmd: QueryCommand,
        parser: argparse.ArgumentParser,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """With no ``--model``, the report uses the model id from engine stats."""
        mock_instance = MagicMock()
        mock_instance.send_request.return_value = (
            "answer",
            _engine_metric_map(model_id="listed-model"),
        )
        mock_request_cls.return_value = mock_instance

        args = parser.parse_args(
            [
                "query",
                "engine",
                "--url",
                "http://localhost:8000/v1",
                "--prompt",
                "x",
            ],
        )
        cmd.execute(args)
        assert "listed-model" in capsys.readouterr().out

    def test_execute_invalid_prompt_exits(
        self,
        cmd: QueryCommand,
        parser: argparse.ArgumentParser,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Unknown ``{placeholder}`` without ``--documents`` raises and exits."""
        args = parser.parse_args(
            [
                "query",
                "engine",
                "--url",
                "http://localhost:8000/v1",
                "--prompt",
                "{unknown_corpus}",
                "--model",
                "m",
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd.execute(args)
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "Unknown documents" in err
        assert "unknown_corpus" in err
