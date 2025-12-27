# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
from argparse import Namespace
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

import vllm.envs as envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.utils import with_cancellation
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.system_utils import set_ulimit
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger("vllm.entrypoints.api_server")

app = FastAPI()
engine = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


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


@app.post("/fault_tolerance/apply")
async def process_fault_tolerance_instruction(request: Request) -> Response:
    """Apply fault tolerance instructions to the engine.

    This endpoint handles fault recovery operations such as retrying operations.

    The request should be a JSON object with the following fields:
    - fault_tolerance_instruction: The name of fault tolerance method.
    - fault_tolerance_timeout: Timeout in seconds for the operation to complete.
    - fault_tolerance_params: dict, optional. Additional dynamic parameters for
    the fault tolerance operation.
    """
    request_dict = await request.json()

    fault_tolerance_instruction = request_dict.get("fault_tolerance_instruction")
    fault_tolerance_timeout = request_dict.get("fault_tolerance_timeout")
    kwargs = request_dict.get("fault_tolerance_params", {})
    assert engine is not None
    success = await engine.handle_fault(
        fault_tolerance_instruction, fault_tolerance_timeout, **kwargs
    )
    if success:
        return JSONResponse(
            status_code=200,
            content={"message": "Instruction executed successfully."},
        )

    logger.error("Fault tolerance operation failed. Shutting down the engine.")
    engine.shutdown()
    raise HTTPException(
        status_code=400,
        detail="Instruction execution failed.",
    )


@app.get("/fault_tolerance/status")
async def get_fault_info() -> Response:
    """Health check."""
    assert engine is not None
    engine_status_dict = await engine.get_fault_info()
    return Response(json.dumps(engine_status_dict), status_code=200)


@with_cancellation
async def _generate(request_dict: dict, raw_request: Request) -> Response:
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    # Since SamplingParams is created fresh per request, safe to skip clone
    sampling_params = SamplingParams(**request_dict, skip_clone=True)
    request_id = random_uuid()

    assert engine is not None
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            assert prompt is not None
            text_outputs = [prompt + output.text for output in request_output.outputs]
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


def build_app(args: Namespace) -> FastAPI:
    global app

    app.root_path = args.root_path
    return app


async def init_app(
    args: Namespace,
    llm_engine: AsyncLLMEngine | None = None,
) -> FastAPI:
    app = build_app(args)

    global engine

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = (
        llm_engine
        if llm_engine is not None
        else AsyncLLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.API_SERVER
        )
    )
    app.state.engine_client = engine
    app.state.args = args
    return app


async def run_server(
    args: Namespace, llm_engine: AsyncLLMEngine | None = None, **uvicorn_kwargs: Any
) -> None:
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    set_ulimit()

    app = await init_app(args, llm_engine)
    assert engine is not None

    shutdown_task = await serve_http(
        app,
        sock=None,
        enable_ssl_refresh=args.enable_ssl_refresh,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        **uvicorn_kwargs,
    )

    await shutdown_task


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=parser.check_port, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument(
        "--ssl-ca-certs", type=str, default=None, help="The CA certificates file"
    )
    parser.add_argument(
        "--enable-ssl-refresh",
        action="store_true",
        default=False,
        help="Refresh SSL Context when SSL certificate files change",
    )
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)",
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy",
    )
    parser.add_argument("--log-level", type=str, default="debug")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    asyncio.run(run_server(args))
