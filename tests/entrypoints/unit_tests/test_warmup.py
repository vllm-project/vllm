# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
from argparse import Namespace
from collections import Counter
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.entrypoints.launchers.launcher import validate_api_server_args
from vllm.entrypoints.warmup import (
    WarmupConfig,
    WarmupPrompt,
    load_warmup_config,
    warmup_engine,
)

GENERATE_CONFIG = {
    "prompts": [
        {"prompt": "Tell me about AI", "max_tokens": 8},
        {"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 4},
        {"prompt": "Explain quantum computing"},
    ],
    "concurrency": [1, 2, 5],
    "request_params": {"temperature": 0.0},
}


def test_load_from_json_string_file_and_dict(tmp_path):
    path = tmp_path / "warmup.json"
    path.write_text(json.dumps(GENERATE_CONFIG))

    from_str = load_warmup_config(json.dumps(GENERATE_CONFIG))
    from_file = load_warmup_config(str(path))
    from_dict = load_warmup_config(GENERATE_CONFIG)

    assert from_str == from_file == from_dict
    assert from_dict is not None
    assert from_dict.task == "generate"
    assert from_dict.concurrency == [1, 2, 5]
    assert from_dict.request_params == {"temperature": 0.0}
    assert from_dict.prompts[0] == WarmupPrompt(prompt="Tell me about AI", max_tokens=8)
    assert from_dict.prompts[2].max_tokens == 256


def test_load_defaults_and_passthrough():
    assert load_warmup_config(None) is None

    config = load_warmup_config({"task": "embed", "prompts": [{"input": "hi"}]})
    assert config is not None
    assert config.concurrency == [1]
    assert config.request_params == {}
    assert load_warmup_config(config) is config

    assert load_warmup_config({"prompts": [{"prompt": "x"}], "concurrency": 4}) == (
        WarmupConfig(prompts=[WarmupPrompt(prompt="x")], concurrency=[4])
    )


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"prompts": [{"prompt": "x"}], "task": "embedding"}, "Invalid warmup task"),
        ({"prompts": [{"prompt": "x"}], "prompt": "typo"}, "Unknown warmup config"),
        ({"prompts": [{"text": "x"}]}, "Unknown warmup prompt"),
        ({"prompts": []}, "non-empty list"),
        ({}, "non-empty list"),
        ({"prompts": ["x"]}, "must be a JSON object"),
        ({"prompts": [{}]}, "exactly one"),
        ({"prompts": [{"prompt": "x", "messages": []}]}, "exactly one"),
        (
            {"prompts": [{"prompt": "x", "messages": [{"role": "user"}]}]},
            "exactly one",
        ),
        ({"prompts": [{"messages": []}]}, "must be non-empty"),
        ({"prompts": [{"prompt": ""}]}, "must be non-empty"),
        ({"prompts": [{"input": "x"}]}, "requires task 'embed'"),
        (
            {"task": "embed", "prompts": [{"messages": [{"role": "user"}]}]},
            "not supported for task 'embed'",
        ),
        ({"task": "embed", "prompts": [{"input": ["a", 1]}]}, "list of strings"),
        ({"prompts": [{"prompt": 1}]}, "'prompt' must be a string"),
        ({"prompts": [{"prompt": "x", "max_tokens": 0}]}, "positive int"),
        ({"prompts": [{"prompt": "x"}], "concurrency": 0}, "concurrency"),
        ({"prompts": [{"prompt": "x"}], "concurrency": []}, "concurrency"),
        ({"prompts": [{"prompt": "x"}], "concurrency": [1, "2"]}, "concurrency"),
        (
            {"prompts": [{"prompt": "x"}], "request_params": {"max_tokens": 4}},
            "max_tokens",
        ),
        (
            {"prompts": [{"prompt": "x"}], "request_params": {"not_a_param": 0}},
            "request_params",
        ),
        (
            {"prompts": [{"prompt": "x"}], "request_params": {"temperature": -1}},
            "request_params",
        ),
    ],
)
def test_load_rejects_invalid_config(config, match):
    with pytest.raises(ValueError, match=match):
        load_warmup_config(config)


def test_load_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_warmup_config(str(tmp_path / "missing.json"))


def test_validate_api_server_args_parses_warmup_config():
    args = Namespace(
        enable_auto_tool_choice=False,
        tool_call_parser=None,
        structured_outputs_config=SimpleNamespace(reasoning_parser=None),
        warmup_config=json.dumps(GENERATE_CONFIG),
    )
    validate_api_server_args(args)
    assert args.warmup_config == load_warmup_config(GENERATE_CONFIG)

    args.warmup_config = json.dumps({"task": "embedding", "prompts": [{"input": "x"}]})
    with pytest.raises(ValueError, match="Invalid warmup task"):
        validate_api_server_args(args)


class FakeEngineClient:
    """Records each request and the peak number of in-flight requests."""

    def __init__(self):
        self.calls: list[tuple[str, Any, Any]] = []
        self.in_flight = 0
        self.peak_in_flight = 0

    async def _stream(self):
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            for _ in range(3):
                await asyncio.sleep(0)
                yield None
        finally:
            self.in_flight -= 1

    def generate(self, prompt, sampling_params, request_id):
        self.calls.append((request_id, prompt, sampling_params))
        return self._stream()

    def encode(self, prompt, pooling_params, request_id):
        self.calls.append((request_id, prompt, pooling_params))
        return self._stream()


class FakeOnlineRenderer:
    async def render_completion(self, request):
        return [{"rendered": request.prompt}]

    async def render_chat(self, request):
        content = request.messages[0]["content"]
        return [], [{"rendered": f"chat:{content}"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 2, 3, 5])
async def test_warmup_engine_runs_every_prompt_at_each_level(concurrency):
    config = load_warmup_config({**GENERATE_CONFIG, "concurrency": concurrency})
    assert config is not None
    engine = FakeEngineClient()

    await warmup_engine(engine, FakeOnlineRenderer(), config)  # type: ignore[arg-type]

    expected = ["Tell me about AI", "chat:Hello!", "Explain quantum computing"]
    rendered = Counter(prompt["rendered"] for _, prompt, _ in engine.calls)
    assert set(rendered) == set(expected)
    assert sum(rendered.values()) == max(concurrency, len(expected))
    assert engine.peak_in_flight == min(concurrency, sum(rendered.values()))
    assert len({request_id for request_id, _, _ in engine.calls}) == len(engine.calls)

    max_tokens = {
        prompt["rendered"]: params.max_tokens for _, prompt, params in engine.calls
    }
    assert max_tokens == {
        "Tell me about AI": 8,
        "chat:Hello!": 4,
        "Explain quantum computing": 256,
    }
    assert all(params.temperature == 0.0 for _, _, params in engine.calls)


@pytest.mark.asyncio
async def test_warmup_engine_sweeps_concurrency_levels():
    config = load_warmup_config(GENERATE_CONFIG)
    assert config is not None
    engine = FakeEngineClient()

    await warmup_engine(engine, FakeOnlineRenderer(), config)  # type: ignore[arg-type]

    # Levels 1, 2, 5 with 3 prompts -> 3 + 3 + 5 requests.
    assert len(engine.calls) == 11
    assert engine.peak_in_flight == 5


@pytest.mark.asyncio
async def test_warmup_engine_embed_expands_list_input():
    config = load_warmup_config(
        {
            "task": "embed",
            "prompts": [{"input": ["a", "b", "c"]}, {"input": "d"}, {"prompt": "e"}],
            "concurrency": [2],
            "request_params": {"use_activation": False},
        }
    )
    assert config is not None
    engine = FakeEngineClient()

    await warmup_engine(engine, FakeOnlineRenderer(), config)  # type: ignore[arg-type]

    assert sorted(prompt for _, prompt, _ in engine.calls) == ["a", "b", "c", "d", "e"]
    assert engine.peak_in_flight == 2
    for _, _, pooling_params in engine.calls:
        assert pooling_params.task == "embed"
        assert pooling_params.use_activation is False
