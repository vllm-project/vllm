# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
from pathlib import Path

import pytest

from vllm.benchmarks.datasets import get_samples


class _RecordingTokenizer:
    """Minimal tokenizer stub that records the kwargs forwarded to
    apply_chat_template, so we can assert chat_template_kwargs propagation
    without loading a real model/template."""

    def __init__(self) -> None:
        self.captured_kwargs: dict | None = None
        self.chat_template = "dummy-template"

    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        **kwargs,
    ) -> str:
        self.captured_kwargs = kwargs
        return conversation[0]["content"]

    def __call__(self, text: str):
        return argparse.Namespace(input_ids=list(range(len(text.split()))))


def _args(dataset_path: str, chat_template_kwargs) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_name="custom",
        dataset_path=dataset_path,
        disable_shuffle=True,
        num_prompts=1,
        custom_output_len=32,
        skip_chat_template=False,
        chat_template_kwargs=chat_template_kwargs,
        no_oversample=False,
        seed=0,
        request_id_prefix="",
    )


def _write_one(path: Path) -> None:
    path.write_text(json.dumps({"prompt": "hello world"}) + "\n")


@pytest.mark.benchmark
def test_chat_template_kwargs_forwarded(tmp_path: Path) -> None:
    """--chat-template-kwargs must reach the client-side apply_chat_template."""
    jsonl = tmp_path / "data.jsonl"
    _write_one(jsonl)

    tok = _RecordingTokenizer()
    get_samples(_args(str(jsonl), {"thinking": True}), tok)

    assert tok.captured_kwargs == {"thinking": True}


@pytest.mark.benchmark
def test_chat_template_kwargs_default_is_noop(tmp_path: Path) -> None:
    """When not provided, no extra kwargs are passed (existing behavior)."""
    jsonl = tmp_path / "data.jsonl"
    _write_one(jsonl)

    tok = _RecordingTokenizer()
    get_samples(_args(str(jsonl), None), tok)

    assert tok.captured_kwargs == {}
