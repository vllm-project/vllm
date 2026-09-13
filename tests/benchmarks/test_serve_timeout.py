# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for timeout handling in `vllm bench serve` metric calculation."""

import argparse
import asyncio
import json
import warnings
from pathlib import Path

import pytest
from aiohttp import web

import vllm.benchmarks.serve as serve_module
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
from vllm.benchmarks.serve import calculate_metrics

TIMEOUT_ERROR = (
    "Traceback (most recent call last):\n"
    '  File "endpoint_request_func.py", line 230, in async_request_openai_completions\n'
    "    async for chunk_bytes in response.content:\n"
    "aiohttp.client_exceptions.ServerTimeoutError: "
    "Timeout on reading data from socket\n"
)


def _successful_output() -> RequestFuncOutput:
    return RequestFuncOutput(
        success=True,
        generated_text="hello world",
        output_tokens=8,
        prompt_len=16,
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


def test_timed_out_request_emits_warning():
    outputs = [
        _successful_output(),
        RequestFuncOutput(success=False, error=TIMEOUT_ERROR),
    ]
    with pytest.warns(UserWarning, match="timed out"):
        metrics, _ = _calculate_metrics(outputs)
    assert metrics.completed == 1
    assert metrics.failed == 1


def test_non_timeout_failure_does_not_warn():
    outputs = [
        _successful_output(),
        RequestFuncOutput(success=False, error="Internal Server Error"),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics, _ = _calculate_metrics(outputs)
    assert metrics.completed == 1
    assert metrics.failed == 1


STALL_SENTINEL = "STALL"


async def _completions_handler(request: web.Request) -> web.StreamResponse:
    payload = await request.json()
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    chunk = json.dumps({"choices": [{"text": "ok"}]})
    await resp.write(f"data: {chunk}\n\n".encode())
    if STALL_SENTINEL in payload["prompt"]:
        # Stall for longer than the VLLM_BENCH_SOCK_READ timeout set by the
        # test (until released at teardown), so the client's per-chunk read
        # timeout fires and marks this request as failed.
        await request.app["stall_release"].wait()
    else:
        usage = json.dumps(
            {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2}}
        )
        await resp.write(f"data: {usage}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
    return resp


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
        result_filename="hang_result.json",
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
async def test_hung_stream_still_writes_result_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One stalled stream must not block the run: with VLLM_BENCH_SOCK_READ
    set, the hung request fails on the read timeout, the benchmark finishes,
    and the result JSON is written with the straggler counted as failed."""
    monkeypatch.setenv("VLLM_BENCH_SOCK_READ", "0.5")

    app = web.Application()
    app["stall_release"] = asyncio.Event()
    app.router.add_post("/v1/completions", _completions_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0, shutdown_timeout=0.1)
    await site.start()
    port = runner.addresses[0][1]

    dataset_path = tmp_path / "prompts.jsonl"
    dataset_path.write_text(
        "\n".join(
            json.dumps({"prompt": p})
            for p in ["hello one", f"hello {STALL_SENTINEL}", "hello three"]
        )
        + "\n"
    )
    args = _serve_args(str(dataset_path), f"http://127.0.0.1:{port}", str(tmp_path))

    try:
        # wait_for bounds the run: without the sock_read timeout this hangs.
        result = await asyncio.wait_for(serve_module.main_async(args), timeout=30)
    finally:
        app["stall_release"].set()
        await runner.cleanup()

    assert result["completed"] == 2
    assert result["failed"] == 1

    saved = json.loads((tmp_path / "hang_result.json").read_text())
    assert saved["completed"] == 2
    assert saved["failed"] == 1
