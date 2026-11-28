# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.entrypoints.cli import openai as cli


@pytest.mark.parametrize("command", [cli.ChatCommand, cli.CompleteCommand])
@pytest.mark.parametrize("quick", [True, False], ids=["quick", "interactive"])
@pytest.mark.parametrize("header_delay", [0.0, 2.0])
def test_stats_include_request_setup(monkeypatch, capsys, command, quick, header_delay):
    now = 10.0
    monkeypatch.setattr(cli, "time", SimpleNamespace(perf_counter=lambda: now))

    def chunks():
        nonlocal now
        # The first event has no token (e.g. a role-only chat chunk).
        yield SimpleNamespace(choices=[], usage=None)
        now += 1.0
        choice = SimpleNamespace(text="hello", delta=SimpleNamespace(content="hello"))
        yield SimpleNamespace(choices=[choice], usage=None)
        now += 1.0
        yield SimpleNamespace(choices=[], usage=SimpleNamespace(completion_tokens=4))

    def create(**kwargs):
        nonlocal now
        now += header_delay
        return chunks()

    client = Mock()
    endpoint = (
        client.chat.completions if command is cli.ChatCommand else client.completions
    )
    endpoint.create.side_effect = create
    monkeypatch.setattr(cli, "_interactive_cli", lambda args: ("model", client))
    monkeypatch.setattr("builtins.input", Mock(side_effect=["hi", EOFError]))
    args = argparse.Namespace(
        quick="hi" if quick else None, stats=True, system_prompt=None, max_tokens=None
    )
    command.cmd(args)

    output = capsys.readouterr().out
    assert f"TTFT: {(header_delay + 1) * 1000:.2f} ms" in output
    duration = header_delay + 2
    assert f"{4 / duration:.2f} tokens/s (4 tokens in {duration:.2f}s)" in output
