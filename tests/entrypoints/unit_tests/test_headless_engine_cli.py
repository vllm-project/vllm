# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for `vllm.entrypoints.cli.headless_engine`."""

from unittest.mock import patch

import pytest

from vllm.entrypoints.cli.headless_engine import main


@pytest.fixture
def run_headless_mock():
    with patch("vllm.entrypoints.cli.serve.run_headless") as mock:
        yield mock


def test_main_calls_run_headless_with_parsed_engine_args(run_headless_mock):
    main(
        [
            "--model",
            "Qwen/Qwen3-0.6B",
            "--data-parallel-address",
            "127.0.0.1",
            "--data-parallel-rpc-port",
            "1234",
            "--data-parallel-size",
            "2",
            "--max-model-len",
            "512",
        ]
    )

    run_headless_mock.assert_called_once()
    args = run_headless_mock.call_args[0][0]
    assert args.model == "Qwen/Qwen3-0.6B"
    assert args.data_parallel_address == "127.0.0.1"
    assert args.data_parallel_rpc_port == 1234
    assert args.data_parallel_size == 2
    # Real argparse type coercion: parsed as int, not the raw string.
    assert args.max_model_len == 512


def test_main_forces_api_server_count_to_zero(run_headless_mock):
    main(["--model", "Qwen/Qwen3-0.6B"])

    args = run_headless_mock.call_args[0][0]
    assert args.api_server_count == 0


def test_main_rejects_unknown_args(run_headless_mock):
    # e.g. a typo'd flag, or a genuinely Frontend-only flag that doesn't
    # apply to a headless engine -- both should fail fast rather than
    # silently launch with the wrong (default) config.
    with pytest.raises(SystemExit):
        main(
            [
                "--model",
                "Qwen/Qwen3-0.6B",
                "--chat-template",
                "some-template.jinja",
            ]
        )

    run_headless_mock.assert_not_called()


def test_main_applies_flexible_argument_parser_preprocessing(run_headless_mock):
    # Underscore spelling and dotted-JSON option syntax are both handled by
    # `FlexibleArgumentParser.parse_args`'s preprocessing, not by argparse
    # itself -- this only works if `main` calls `parse_args`, not
    # `parse_known_args` directly on the raw argv.
    main(
        [
            "--model",
            "Qwen/Qwen3-0.6B",
            "--max_model_len",  # underscores, not dashes
            "512",
        ]
    )

    args = run_headless_mock.call_args[0][0]
    assert args.max_model_len == 512


def test_main_defaults_match_async_engine_args_defaults(run_headless_mock):
    from vllm.engine.arg_utils import AsyncEngineArgs

    main(["--model", "Qwen/Qwen3-0.6B"])

    args = run_headless_mock.call_args[0][0]
    defaults = AsyncEngineArgs()
    assert args.enable_log_requests == defaults.enable_log_requests
    assert args.disable_log_stats == defaults.disable_log_stats
