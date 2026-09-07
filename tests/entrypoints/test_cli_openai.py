# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from unittest.mock import Mock, patch

import pytest

from vllm.entrypoints.cli.openai import ChatCommand, CompleteCommand


@pytest.mark.parametrize("command", [ChatCommand, CompleteCommand])
@pytest.mark.parametrize("quick", [None, "", "hello"])
def test_quick_prompt(command, quick):
    parser = command.add_cli_args(argparse.ArgumentParser())
    args = parser.parse_args([] if quick is None else ["--quick", quick])
    client = Mock()
    create = (
        client.chat.completions.create
        if command is ChatCommand
        else client.completions.create
    )
    create.return_value = []

    with (
        patch(
            "vllm.entrypoints.cli.openai._interactive_cli",
            return_value=("test", client),
        ),
        patch("builtins.input", side_effect=["interactive", EOFError]) as mock_input,
    ):
        command.cmd(args)

    create.assert_called_once()
    expected_prompt = "interactive" if quick is None else quick
    if command is ChatCommand:
        assert create.call_args.kwargs["messages"][0] == {
            "role": "user",
            "content": expected_prompt,
        }
    else:
        assert create.call_args.kwargs["prompt"] == expected_prompt
    assert mock_input.call_count == (2 if quick is None else 0)


@pytest.mark.parametrize("quick", [False, True])
@pytest.mark.parametrize("max_tokens", [None, 0, -1, 3])
def test_completion_max_tokens(quick, max_tokens):
    parser = CompleteCommand.add_cli_args(argparse.ArgumentParser())
    options = ["--quick", "hello"] if quick else []
    if max_tokens is not None:
        options.extend(["--max-tokens", str(max_tokens)])
    args = parser.parse_args(options)
    client = Mock()
    client.completions.create.return_value = []

    with (
        patch(
            "vllm.entrypoints.cli.openai._interactive_cli",
            return_value=("test", client),
        ),
        patch("builtins.input", side_effect=["hello", EOFError]),
    ):
        CompleteCommand.cmd(args)

    expected_kwargs = {"model": "test", "prompt": "hello", "stream": True}
    if max_tokens is not None:
        expected_kwargs["max_tokens"] = max_tokens
    client.completions.create.assert_called_once_with(**expected_kwargs)
