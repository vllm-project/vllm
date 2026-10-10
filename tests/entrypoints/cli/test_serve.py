# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ``vllm serve`` CLI subcommand."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from vllm.entrypoints.cli.serve import run_headless


class _StopAfterEngineConfig(Exception):
    pass


def test_headless_imports_reasoning_parser_plugin_before_engine_config():
    plugin_path = "/tmp/custom_reasoning_parser.py"
    args = argparse.Namespace(
        api_server_count=0,
        reasoning_parser_plugin=plugin_path,
        data_parallel_hybrid_lb=False,
    )
    engine_args = MagicMock()

    with (
        patch(
            "vllm.entrypoints.cli.serve.vllm.AsyncEngineArgs.from_cli_args",
            return_value=engine_args,
        ),
        patch(
            "vllm.reasoning.ReasoningParserManager.import_reasoning_parser"
        ) as import_plugin,
    ):

        def create_engine_config(**kwargs):
            import_plugin.assert_called_once_with(plugin_path)
            raise _StopAfterEngineConfig

        engine_args.create_engine_config.side_effect = create_engine_config

        with pytest.raises(_StopAfterEngineConfig):
            run_headless(args)


def test_headless_rejects_hybrid_lb_before_building_engine_config():
    args = argparse.Namespace(
        api_server_count=0,
        reasoning_parser_plugin=None,
        data_parallel_hybrid_lb=True,
    )
    engine_args = MagicMock()

    with patch(
        "vllm.entrypoints.cli.serve.vllm.AsyncEngineArgs.from_cli_args",
        return_value=engine_args,
    ):
        with pytest.raises(ValueError, match="hybrid_lb is not applicable"):
            run_headless(args)

    # The invalid combination must be rejected before any engine config work,
    # which otherwise downloads the HF config and detects the platform first.
    engine_args.create_engine_config.assert_not_called()
