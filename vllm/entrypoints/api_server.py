# SPDX-License-Identifier: Apache-2.0
"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""
import asyncio
import json
import ssl
import time
from argparse import Namespace
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Optional

import orjson
import uvloop
from fastapi import FastAPI, Request, APIRouter
from fastapi.responses import JSONResponse, Response, StreamingResponse, ORJSONResponse
from vllm.entrypoints.openai.api_server import build_async_engine_client, create_server_socket
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.utils import with_cancellation
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, random_uuid, set_ulimit, is_valid_ipv6_address
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger("vllm.entrypoints.api_server")

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None
backend_process = None
request_decode_length_map = {}
start_time = time.time()


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/simple_schedule_trace")
async def simple_status() -> Response:
    """simpel status check with metrics collection only"""
    assert engine is not None
    scheduler_trace = await engine.get_scheduler_trace()
    scheduler_trace_flattened = {}
    total_requests_count = 0
    total_running_requests_count = 0
    total_required_waiting_blocks = 0
    free_gpu_blocks = 0
    num_preempted = 0
    for i in scheduler_trace.keys():
        for key in scheduler_trace[i].keys():
            if key == "free_gpu_blocks":
                free_gpu_blocks += scheduler_trace[i][key]
            elif key == "num_preempted":
                num_preempted += scheduler_trace[i][key]
            else:
                total_requests_count += len(scheduler_trace[i][key])
                if key == "running":
                    total_running_requests_count += len(scheduler_trace[i][key])
                elif key == "waiting":
                    for request_info in scheduler_trace[i][key]:
                        total_required_waiting_blocks += request_info["required_prompted_blocks"]
    scheduler_trace_flattened["total_requests_count"] = total_requests_count
    scheduler_trace_flattened["free_gpu_blocks"] = free_gpu_blocks
    scheduler_trace_flattened["num_preempted"] = num_preempted
    scheduler_trace_flattened["current_running_requests_count"] = total_running_requests_count
    scheduler_trace_flattened["adjusted_free_gpu_blocks"] = free_gpu_blocks - total_required_waiting_blocks
    return Response(content=orjson.dumps(scheduler_trace_flattened),
                    media_type="application/json")


@app.get("/schedule_trace")
async def status() -> Response:
    """Status check."""
    assert engine is not None
    scheduler_trace = await engine.get_scheduler_trace()
    scheduler_trace_count = 0
    scheduler_trace_flattened = {}
    free_gpu_blocks = 0
    num_preempted = 0
    for i in scheduler_trace.keys():
        for key in scheduler_trace[i].keys():
            if key == "free_gpu_blocks":
                free_gpu_blocks += scheduler_trace[i][key]
            elif key == "num_preempted":
                num_preempted += scheduler_trace[i][key]
            else:
                scheduler_trace_flattened[key] = []
                for request_info in scheduler_trace[i][key]:
                    request_id = int(request_info['request_id'])
                    arrival_time = request_info['arrival_time'] - start_time
                    total_output_length = request_info["seq_total_output_length"]
                    prompt_length = request_info["seq_prompts_length"]
                    computed_length = request_info["seq_computed_length"]
                    is_prefill = 1 if request_info["is_prefill"] else 0
                    if request_id in request_decode_length_map:
                        expected_length = request_decode_length_map[request_id]
                    else:
                        expected_length = 0
                    scheduler_trace_flattened[key].extend([request_id, arrival_time,
                                                           total_output_length, prompt_length,
                                                           computed_length, is_prefill, expected_length])
                    scheduler_trace_count += 1
    scheduler_trace_flattened["free_gpu_blocks"] = free_gpu_blocks
    scheduler_trace_flattened["num_preempted"] = num_preempted
    encoded_scheduler_trace = orjson.dumps(scheduler_trace_flattened)
    return Response(content=encoded_scheduler_trace,
                    media_type="application/json")


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    return await _generate(request_dict, raw_request=request)


@with_cancellation
async def _generate(request_dict: dict, raw_request: Request) -> Response:
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    assert engine is not None
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            assert prompt is not None
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\n").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    try:
        async for request_output in results_generator:
            final_output = request_output
    except asyncio.CancelledError:
        return Response(status_code=499)

    assert final_output is not None
    prompt = final_output.prompt
    assert prompt is not None
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)


@app.post("/generate_benchmark")
async def generate_benchmark(request: Request) -> Response:
    request_start_time = time.time()
    request_dict = await request.json()
    return await _generate_benchmark(request_dict, request, request_start_time)


@with_cancellation
async def _generate_benchmark(request_dict, request: Request, request_start_time) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    # Add some benchmark-related codes comparing to the generate API.
    prompt = request_dict.pop("prompt")
    _ = request_dict.pop("stream", False)
    request_id = request_dict.pop("request_id")
    request_decode_length_map[int(request_id)] = request_dict.pop('num_predicted_tokens')
    sampling_params = SamplingParams(**request_dict)

    assert engine is not None
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Non-streaming case
    final_output = None
    per_token_latency = []
    start = time.time()
    try:
        async for request_output in results_generator:
            now = time.time()
            per_token_latency.append([now, (now - start) * 1000])
            start = now
            final_output = request_output
    except asyncio.CancelledError:
        print("Cancelled request for request_id: {}".format(request_id))
        return Response(status_code=499)

    generation = final_output.outputs[0].text
    num_output_tokens = len(final_output.outputs[0].token_ids)
    num_input_tokens = len(final_output.prompt_token_ids)
    expected_resp_len = request_dict['max_tokens']
    if not max(expected_resp_len, 1) == max(num_output_tokens, 1):
        "request_id={}, expected_resp_len={}, num_output_tokens={}, num_input_tokens={}".format(
            request_id, expected_resp_len, num_output_tokens, num_input_tokens)
    ret = {
        'request_id': request_id,
        'generated_text': generation,
        'num_output_tokens_cf': num_output_tokens,
        'per_token_latency': per_token_latency,
    }
    if final_output.metrics:
        if final_output.metrics.time_in_queue:
            ret['waiting_latency'] = final_output.metrics.time_in_queue * 1000
        if final_output.metrics.model_execute_time:
            ret['inference_latency'] = final_output.metrics.model_execute_time * 1000
        if final_output.metrics.first_token_time:
            ret['ttft'] = (final_output.metrics.first_token_time - final_output.metrics.arrival_time) * 1000
    end_time = time.time()
    ret["time_on_backend"] = (end_time - request_start_time) * 1000

    return JSONResponse(ret)


def build_app(args: Namespace) -> FastAPI:
    global app, start_time

    app.root_path = args.root_path
    start_time = time.time()
    return app


@asynccontextmanager
async def get_engine_client(args: Namespace) -> AsyncLLMEngine:
    engine_args = AsyncEngineArgs.from_cli_args(args)
    if args.disable_frontend_multiprocessing:
        engine_client = AsyncLLMEngine.from_engine_args(engine_args, usage_context=UsageContext.API_SERVER)
        yield engine_client
    else:
        async with build_async_engine_client(args, False, engine_args) as engine_client:
            is_sleeping = await engine_client.is_sleeping()
            print("Engine is sleeping: {}".format(is_sleeping))
            get_scheduler_trace = await engine_client.get_scheduler_trace()
            print("Scheduler trace: {}".format(get_scheduler_trace))
            yield engine_client


async def run_server(args: Namespace = None,
                     **uvicorn_kwargs: Any) -> None:
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    set_ulimit()
    app = build_app(args)
    async with get_engine_client(args) as engine_client:
        global engine
        engine = engine_client

        if not args.disable_frontend_multiprocessing:
            sock_addr = (args.host or "", args.port)
            sock = create_server_socket(sock_addr)

            def _listen_addr(a: str) -> str:
                if is_valid_ipv6_address(a):
                    return '[' + a + ']'
                return a or "0.0.0.0"

            is_ssl = args.ssl_keyfile and args.ssl_certfile
            logger.info("Starting vLLM API server on http%s://%s:%d",
                        "s" if is_ssl else "", _listen_addr(sock_addr[0]),
                        sock_addr[1])
        else:
            sock = None

        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            workers=args.workers,
            **uvicorn_kwargs,
        )

        await shutdown_task
        if sock:
            sock.close()


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=parser.check_port, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--enable-ssl-refresh",
        action="store_true",
        default=False,
        help="Refresh SSL Context when SSL certificate files change")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument("--log-level", type=str, default="debug")
    parser.add_argument("--disable-frontend-multiprocessing", action="store_true")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    logger.info("Starting server with args: %s", str(args))
    uvloop.run(run_server(args))
