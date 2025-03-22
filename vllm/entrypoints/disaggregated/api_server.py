# SPDX-License-Identifier: Apache-2.0

import uvicorn
import uvloop

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.disaggregated.pd_engine import PDEngine
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                    OpenAIServingModels)
from vllm.entrypoints.openai.protocol import CompletionRequest
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.openai.protocol import (
    CompletionResponse, ErrorResponse)

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger('vllm.entrypoints.disaggregated.api_server')

app = FastAPI()

@app.get("/v1/models")
async def show_available_models(raw_request: Request):
    handler: OpenAIServingModels = raw_request.app.state.openai_serving_models
    models_ = await handler.show_available_models()
    return JSONResponse(content=models_.model_dump())

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    handler: OpenAIServingCompletion = raw_request.app.state.openai_serving_completion
    generator = await handler.create_completion(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, CompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")

@asynccontextmanager
async def pd_engine_client_ctx_manager(
    model_name: str,
    prefill_addr: str,
    decode_addr: str,
    connector_addr: str) -> AsyncIterator[PDEngine]:
    engine = PDEngine(model_name, prefill_addr, decode_addr, connector_addr)
    yield engine
    engine.shutdown()

async def main(args, **uvicorn_kwargs):
    logger.info("vLLM Disaggregate Connector Start %s %s", args,
                uvicorn_kwargs)
    
    prefill_addr = f"ipc://{args.prefill_addr}"
    decode_addr = f"ipc://{args.decode_addr}"
    connector_addr = f"ipc://{args.connector_addr}"

    with pd_engine_client_ctx_manager(
        args.model, prefill_addr, decode_addr, connector_addr) as engine_client:

        model_config = await engine_client.get_model_config()

        # Models.
        app.state.openai_serving_models = OpenAIServingModels(
            engine_client=engine_client,
            model_config=model_config,
            base_model_paths=[BaseModelPath(
                name=args.served_model_name or args.model,
                model_path=args.model)
            ],
        )

        # Completions.
        app.state.openai_serving_completion = OpenAIServingCompletion(
            engine_client=engine_client,
            model_config=model_config,
            models=app.state.openai_serving_models,
            request_logger=None,
        )

    # Init Uvicorn Server. Server.
    config = uvicorn.Config(app, host="0.0.0.0", port=args.port)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible P/D Server.")
    parser.add_argument("--host",
                        type=str,
                        default="0.0.0.0",
                        help="The host of the HTTP server.")
    parser.add_argument("--port",
                        type=int,
                        default=8001,
                        help="The port of the HTTP server.")
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="The path to the model.")
    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The served name of the model.")
    parser.add_argument("--connector-addr",
                        type=str,
                        required=True,
                        help="The zmq ipc connector address")
    parser.add_argument("--prefill-addr",
                        type=str,
                        required=True,
                        help="The zmq ipc prefill address")
    parser.add_argument("--decode-addr",
                        type=str,
                        required=True,
                        help="The zmq ipc decode address")
    args = parser.parse_args()
    uvloop.run(main(args))
