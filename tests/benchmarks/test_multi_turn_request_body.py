# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skip_global_cleanup

REPO_ROOT = Path(__file__).parents[2]
MULTI_TURN_DIR = REPO_ROOT / "benchmarks" / "multi_turn"
sys.path.insert(0, str(MULTI_TURN_DIR))
SPEC = importlib.util.spec_from_file_location(
    "benchmark_serving_multi_turn",
    MULTI_TURN_DIR / "benchmark_serving_multi_turn.py",
)
assert SPEC is not None
benchmark_serving_multi_turn = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark_serving_multi_turn)


def test_parse_extra_request_body_requires_json_object():
    with pytest.raises(ValueError, match="JSON object"):
        benchmark_serving_multi_turn.parse_extra_request_body("[1, 2, 3]")


def test_disable_thinking_adds_reasoning_controls():
    args = argparse.Namespace(disable_thinking=True, extra_request_body=None)

    extra_request_body = benchmark_serving_multi_turn.get_extra_request_body(args)

    assert extra_request_body == {
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "low",
    }


def test_extra_request_body_merges_with_disable_thinking_defaults():
    args = argparse.Namespace(
        disable_thinking=True,
        extra_request_body='{"chat_template_kwargs":{"foo":"bar"},"metadata":{"x":1}}',
    )

    extra_request_body = benchmark_serving_multi_turn.get_extra_request_body(args)

    assert extra_request_body == {
        "chat_template_kwargs": {
            "enable_thinking": False,
            "foo": "bar",
        },
        "reasoning_effort": "low",
        "metadata": {"x": 1},
    }


def test_build_request_payload_merges_extra_request_body():
    payload = benchmark_serving_multi_turn.build_request_payload(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        model="Qwen/Qwen3-32B-FP8",
        stream=True,
        min_tokens=1,
        max_tokens=63,
        conversation_id="conv-1",
        extra_request_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "low",
        },
    )

    assert payload["model"] == "Qwen/Qwen3-32B-FP8"
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": False}
    assert payload["min_tokens"] == 1
    assert payload["max_tokens"] == 63
    assert payload["conversation_id"] == "conv-1"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["reasoning_effort"] == "low"
