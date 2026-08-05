# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for cached-token metrics in `vllm bench serve`.

The server only reports usage.prompt_tokens_details.cached_tokens when
launched with --enable-prompt-tokens-details; without it the new metrics
must be omitted entirely.
"""

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

import vllm.benchmarks.serve as serve_module
from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncInput,
    RequestFuncOutput,
    async_request_openai_completions,
)
from vllm.benchmarks.serve import calculate_metrics


def _output(cached_tokens: int | None) -> RequestFuncOutput:
    return RequestFuncOutput(
        success=True,
        generated_text="hello",
        output_tokens=8,
        prompt_len=16,
        cached_tokens=cached_tokens,
        latency=1.0,
        ttft=0.1,
        itl=[0.01] * 7,
        start_time=0.0,
    )


def _calculate_metrics(outputs: list[RequestFuncOutput]):
    return calculate_metrics(
        input_requests=[],
        outputs=outputs,
        dur_s=10.0,
        tokenizer=None,
        selected_percentiles=[99.0],
        goodput_config_dict={},
    )


def test_cached_token_metrics_reported():
    metrics, _ = _calculate_metrics([_output(12), _output(4), _output(0)])
    assert metrics.total_cached == 16


def test_cached_token_metrics_omitted_when_server_does_not_report():
    metrics, _ = _calculate_metrics([_output(None), _output(None)])
    assert metrics.total_cached is None


async def _fake_completions_server(usage: dict):
    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        chunk = json.dumps({"choices": [{"text": "ok"}]})
        await resp.write(f"data: {chunk}\n\n".encode())
        await resp.write(f"data: {json.dumps({'usage': usage})}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp

    app = web.Application()
    app.router.add_post("/v1/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, runner.addresses[0][1]


async def _request_once(port: int) -> RequestFuncOutput:
    async with aiohttp.ClientSession() as session:
        return await async_request_openai_completions(
            RequestFuncInput(
                prompt="hello",
                api_url=f"http://127.0.0.1:{port}/v1/completions",
                prompt_len=4,
                output_len=8,
                model="test-model",
            ),
            session,
        )


@pytest.mark.asyncio
async def test_request_func_parses_cached_tokens():
    usage = {
        "prompt_tokens": 4,
        "completion_tokens": 8,
        "prompt_tokens_details": {"cached_tokens": 3},
    }
    runner, port = await _fake_completions_server(usage)
    try:
        output = await _request_once(port)
    finally:
        await runner.cleanup()
    assert output.success
    assert output.cached_tokens == 3


@pytest.mark.asyncio
async def test_request_func_cached_tokens_none_without_details():
    usage = {"prompt_tokens": 4, "completion_tokens": 8}
    runner, port = await _fake_completions_server(usage)
    try:
        output = await _request_once(port)
    finally:
        await runner.cleanup()
    assert output.success
    assert output.cached_tokens is None


def _serve_args(
    dataset_path: str, base_url: str, result_dir: str
) -> argparse.Namespace:
    return argparse.Namespace(
        # dataset
        dataset_name="custom",
        dataset_path=dataset_path,
        disable_shuffle=True,
        num_prompts=3,
        custom_output_len=8,
        skip_chat_template=True,
        chat_template_kwargs=None,
        no_oversample=False,
        seed=0,
        request_id_prefix="bench-",
        # model / tokenizer
        model="test-model",
        served_model_name=None,
        tokenizer=None,
        tokenizer_mode="auto",
        trust_remote_code=False,
        skip_tokenizer_init=True,
        # backend / endpoint
        backend="openai",
        base_url=base_url,
        host=None,
        port=None,
        endpoint="/v1/completions",
        header=None,
        insecure=False,
        # traffic
        request_rate=float("inf"),
        burstiness=1.0,
        max_concurrency=None,
        probe_request_rate=0.0,
        # misc serve args read by main_async
        plot_timeline=False,
        plot_dataset_stats=False,
        self_timed=None,
        metadata=None,
        label=None,
        logprobs=None,
        use_beam_search=False,
        ignore_eos=False,
        goodput=None,
        percentile_metrics="ttft,tpot,itl,e2el",
        metric_percentiles="99",
        save_result=True,
        append_result=False,
        result_dir=result_dir,
        result_filename="cached_result.json",
        num_warmups=0,
        profile=False,
        disable_tqdm=True,
        lora_modules=None,
        lora_assignment="random",
        ramp_up_strategy=None,
        ramp_up_start_rps=None,
        ramp_up_end_rps=None,
        ready_check_timeout_sec=0,
        extra_body=None,
        top_p=None,
        top_k=None,
        min_p=None,
        temperature=None,
        frequency_penalty=None,
        presence_penalty=None,
        repetition_penalty=None,
        save_detailed=False,
        input_len=None,
        output_len=None,
    )


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_total_cached_tokens_saved_in_result_json(tmp_path: Path) -> None:
    """End to end: cached tokens reported per request must be summed and
    written to the result JSON as total_cached_tokens."""
    usage = {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "prompt_tokens_details": {"cached_tokens": 3},
    }
    runner, port = await _fake_completions_server(usage)

    dataset_path = tmp_path / "prompts.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps({"prompt": f"hello {i}"}) for i in range(3)) + "\n"
    )
    args = _serve_args(str(dataset_path), f"http://127.0.0.1:{port}", str(tmp_path))

    try:
        result = await asyncio.wait_for(serve_module.main_async(args), timeout=30)
    finally:
        await runner.cleanup()

    assert result["total_cached_tokens"] == 9

    saved = json.loads((tmp_path / "cached_result.json").read_text())
    assert saved["total_cached_tokens"] == 9
