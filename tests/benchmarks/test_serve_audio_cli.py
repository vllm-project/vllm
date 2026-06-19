#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio

import pytest

import vllm.benchmarks.serve as serve_module


def _parse_serve_args(*extra_args: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    serve_module.add_cli_args(parser)
    return parser.parse_args(
        [
            "--backend",
            "openai-audio",
            "--endpoint",
            "/v1/audio/transcriptions",
            "--model",
            "openai/whisper-large-v3",
            "--dataset-name",
            "sonnet",
            "--num-prompts",
            "1",
            "--skip-tokenizer-init",
            *extra_args,
        ]
    )


@pytest.mark.benchmark
def test_bench_serve_enable_vad_sets_audio_request_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parse_serve_args("--enable-vad")
    captured: dict[str, object] = {}

    async def fake_benchmark(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(serve_module, "get_samples", lambda args, tokenizer: [])
    monkeypatch.setattr(serve_module, "check_goodput_args", lambda args: {})
    monkeypatch.setattr(serve_module, "freeze_gc_heap", lambda: None)
    monkeypatch.setattr(serve_module, "benchmark", fake_benchmark)

    result = asyncio.run(serve_module.main_async(args))

    assert captured["endpoint_type"] == "openai-audio"
    assert captured["extra_body"] == {"vad_config.enabled": True}
    assert result["enable_vad"] is True


@pytest.mark.benchmark
def test_bench_serve_enable_vad_requires_audio_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parse_serve_args(
        "--backend",
        "openai",
        "--endpoint",
        "/v1/completions",
        "--enable-vad",
    )

    monkeypatch.setattr(serve_module, "get_samples", lambda args, tokenizer: [])
    monkeypatch.setattr(serve_module, "check_goodput_args", lambda args: {})

    with pytest.raises(ValueError, match="openai-audio"):
        asyncio.run(serve_module.main_async(args))
