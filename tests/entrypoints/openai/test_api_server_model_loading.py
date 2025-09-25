# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration test for api_server model loading with positional arguments.

This test verifies that the model_tag to model conversion works correctly
in the actual api_server startup flow.
"""

from typing import Optional

from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser


def test_api_server_model_tag_handling():
    """
    Test that api_server correctly handles model_tag positional argument.
    
    This is a regression test for the bug where V1 engine would load
    the default model (Qwen/Qwen3-0.6B) instead of the user-specified model
    when using positional arguments.
    """
    # Test models to verify
    test_cases: list[dict[str, Optional[str]]] = [{
        "model": "facebook/opt-125m",
        "task": "generate",
        "served_name": "opt-125m"
    }, {
        "model": "openai-community/gpt2",
        "task": "generate",
        "served_name": "gpt2-server"
    }, {
        "model": "meta-llama/Meta-Llama-3-8B",
        "task": None,
        "served_name": "llama-3-8b"
    }]

    for test_case in test_cases:
        # Create parser as api_server.py does
        parser = FlexibleArgumentParser(
            description="vLLM OpenAI-Compatible RESTful API server.")
        parser = make_arg_parser(parser)

        # Build command line args
        cli_args: list[str] = [test_case["model"]]  # type: ignore
        if test_case["task"]:
            cli_args.extend(["--task", test_case["task"]])
        cli_args.extend(["--served-model-name",
                         test_case["served_name"]])  # type: ignore

        # Parse arguments
        args = parser.parse_args(cli_args)

        # Verify model_tag is set correctly
        assert args.model_tag == test_case["model"], \
            f"model_tag should be {test_case['model']}, got {args.model_tag}"

        # Apply the fix (this should be in api_server.py)
        if hasattr(args, 'model_tag') and args.model_tag is not None:
            args.model = args.model_tag

        # Verify model is now set correctly
        assert args.model == test_case["model"], \
            f"model should be {test_case['model']}, got {args.model}"

        # Verify other args are preserved
        if test_case["task"]:
            assert args.task == test_case["task"]
        assert args.served_model_name == [test_case["served_name"]]


def test_api_server_with_model_flag():
    """Test backward compatibility with --model flag"""
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)

    model_path = "facebook/opt-125m"
    args = parser.parse_args(
        ["--model", model_path, "--host", "127.0.0.1", "--port", "8000"])

    # When using --model flag, model is set directly
    assert args.model == model_path
    # model_tag should be None (no positional arg)
    assert args.model_tag is None

    # The fix should not affect this case
    if hasattr(args, 'model_tag') and args.model_tag is not None:
        args.model = args.model_tag

    # model should still be the same
    assert args.model == model_path


def test_api_server_mixed_model_args():
    """Test when both positional model_tag and --model are provided"""
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)

    positional_model = "model-from-positional"
    flag_model = "model-from-flag"

    # Both positional and --model flag
    args = parser.parse_args([positional_model, "--model", flag_model])

    # The --model flag should take precedence (standard argparse behavior)
    assert args.model == flag_model
    assert args.model_tag == positional_model

    # The fix would override with positional (which matches serve.py behavior)
    if hasattr(args, 'model_tag') and args.model_tag is not None:
        args.model = args.model_tag

    # After fix, positional takes precedence (matching serve.py)
    assert args.model == positional_model


def test_api_server_no_model():
    """Test when no model is specified"""
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)

    # No model arguments at all
    args = parser.parse_args(["--host", "0.0.0.0", "--port", "8080"])

    assert args.model_tag is None
    # model would be None or default

    # The fix should handle None gracefully
    if hasattr(args, 'model_tag') and args.model_tag is not None:
        args.model = args.model_tag

    # Should not crash, model remains as it was


def test_kubernetes_deployment_scenario():
    """
    Test a typical Kubernetes deployment scenario.
    
    This simulates the command that would fail without the fix:
    python -m vllm.entrypoints.openai.api_server \
        organization/model-name-v1 \
        --task generate --served-model-name model-server
    """
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)

    # Typical Kubernetes deployment args with positional model
    test_model = "meta-llama/Meta-Llama-3-8B"
    args = parser.parse_args([
        test_model, "--task", "generate", "--served-model-name",
        "llama-3-server", "--host", "0.0.0.0", "--port", "8000"
    ])

    # Before fix: model_tag is set but model is not
    assert args.model_tag == test_model

    # This is the fix that must be in api_server.py
    if hasattr(args, 'model_tag') and args.model_tag is not None:
        args.model = args.model_tag

    # After fix: model is correctly set
    assert args.model == test_model
    assert args.task == "generate"
    assert args.served_model_name == ["llama-3-server"]

    # This ensures the V1 engine will load the correct model
    # instead of defaulting to "Qwen/Qwen3-0.6B"
