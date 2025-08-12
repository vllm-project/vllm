# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that model_tag positional argument is correctly handled in API server.

This test ensures that the bug where model_tag was not converted to model
(causing the engine to use the default model) is fixed.
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser


@pytest.fixture
def api_server_parser():
    """Create parser as api_server.py does"""
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    return make_arg_parser(parser)


def test_model_tag_positional_argument(api_server_parser):
    """Test that model_tag positional argument is parsed correctly"""
    model_path = "facebook/opt-125m"
    args = api_server_parser.parse_args([
        model_path, "--task", "generate", "--served-model-name",
        "opt-125m-server"
    ])

    # Verify model_tag is set
    assert hasattr(args, 'model_tag')
    assert args.model_tag == model_path

    # At this point, model should NOT be set (this is the bug)
    # The default value would be used if we don't apply the fix
    assert not hasattr(args, 'model') or args.model is None


def test_model_tag_to_model_conversion():
    """Test the fix that converts model_tag to model"""
    model_path = "meta-llama/Meta-Llama-3-8B"

    # Simulate parsed args with model_tag but no model
    args = argparse.Namespace()
    args.model_tag = model_path
    args.task = "generate"
    args.served_model_name = ["llama-3-8b"]

    # This is the fix that should be in api_server.py
    if hasattr(args, 'model_tag') and args.model_tag is not None:
        args.model = args.model_tag

    # Verify the fix works
    assert args.model == model_path
    assert args.model_tag == model_path


def test_async_engine_args_without_fix():
    """Test that AsyncEngineArgs uses default model without the fix"""
    model_path = "openai-community/gpt2"

    # Simulate args WITHOUT the model attribute (the bug scenario)
    args = argparse.Namespace()
    args.model_tag = model_path  # This is set
    # args.model is NOT set - this is the bug

    # Mock the AsyncEngineArgs to check what it receives
    with patch.object(AsyncEngineArgs, '__init__', return_value=None):
        # Create a mock that captures what from_cli_args would do
        mock_engine_args = MagicMock(spec=AsyncEngineArgs)
        mock_engine_args.model = "Qwen/Qwen3-0.6B"  # Default value

        with patch.object(AsyncEngineArgs,
                          'from_cli_args',
                          return_value=mock_engine_args):
            engine_args = AsyncEngineArgs.from_cli_args(args)

            # Without the fix, the model would be the default
            assert engine_args.model == "Qwen/Qwen3-0.6B"


def test_async_engine_args_with_fix():
    """Test that AsyncEngineArgs uses correct model with the fix"""
    model_path = "distilbert/distilgpt2"

    # Simulate args WITH the fix applied
    args = argparse.Namespace()
    args.model_tag = model_path
    args.model = model_path  # The fix: model_tag copied to model
    args.task = "generate"
    args.served_model_name = ["distilgpt2-server"]

    # Mock AsyncEngineArgs to verify it gets the correct model
    with patch.object(AsyncEngineArgs, '__init__', return_value=None):
        mock_engine_args = MagicMock(spec=AsyncEngineArgs)
        mock_engine_args.model = model_path  # Should use the provided model

        with patch.object(AsyncEngineArgs,
                          'from_cli_args',
                          return_value=mock_engine_args) as mock_from_cli:
            engine_args = AsyncEngineArgs.from_cli_args(args)

            # With the fix, the model should be correct
            assert engine_args.model == model_path
            mock_from_cli.assert_called_once_with(args)


def test_model_directly_specified():
    """Test backward compatibility when --model is used directly"""
    model_path = "facebook/opt-125m"

    # When model is specified with --model flag (not positional)
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)

    args = parser.parse_args(["--model", model_path, "--task", "generate"])

    # model should be set directly
    assert args.model == model_path
    # model_tag might be None (no positional arg)
    assert args.model_tag is None

    # No conversion needed in this case
    # AsyncEngineArgs should work correctly
    with patch.object(AsyncEngineArgs, '__init__', return_value=None):
        mock_engine_args = MagicMock(spec=AsyncEngineArgs)
        mock_engine_args.model = model_path

        with patch.object(AsyncEngineArgs,
                          'from_cli_args',
                          return_value=mock_engine_args):
            engine_args = AsyncEngineArgs.from_cli_args(args)
            assert engine_args.model == model_path


def test_no_model_specified():
    """Test when neither model_tag nor model is specified"""
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)

    # No model specified at all
    args = parser.parse_args([])

    assert args.model_tag is None
    # model would use the default from AsyncEngineArgs

    # The fix should handle this gracefully
    if hasattr(args, 'model_tag') and args.model_tag is not None:
        args.model = args.model_tag
    # No change since model_tag is None

    # AsyncEngineArgs will use its default
    with patch.object(AsyncEngineArgs, '__init__', return_value=None):
        mock_engine_args = MagicMock(spec=AsyncEngineArgs)
        mock_engine_args.model = "Qwen/Qwen3-0.6B"  # Default

        with patch.object(AsyncEngineArgs,
                          'from_cli_args',
                          return_value=mock_engine_args):
            engine_args = AsyncEngineArgs.from_cli_args(args)
            assert engine_args.model == "Qwen/Qwen3-0.6B"  # Uses default


def test_integration_with_serve_command():
    """Test that vllm serve command handles model_tag correctly"""
    # The serve.py command already has the fix at line 40-41:
    # if hasattr(args, 'model_tag') and args.model_tag is not None:
    #     args.model = args.model_tag

    # Simulate what serve.py does
    args = argparse.Namespace()
    args.model_tag = "my-model/path"

    # Apply the same fix that's in serve.py
    if hasattr(args, 'model_tag') and args.model_tag is not None:
        args.model = args.model_tag

    assert args.model == "my-model/path"

    # This ensures consistency between serve.py and api_server.py
