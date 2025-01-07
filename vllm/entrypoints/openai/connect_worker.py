import json
from typing import Optional
import zmq
import zmq.asyncio
import tempfile
import uuid
import httpx
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (CompletionRequest,
                                              CompletionRequest,
                                              CompletionResponse,
                                              ErrorResponse)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.openai.serving_tokenization import OpenAIServingTokenization
from vllm.logger import init_logger
import traceback

prometheus_multiproc_dir: tempfile.TemporaryDirectory

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger('vllm.entrypoints.openai.connect_worker')

def base(app: FastAPI) -> OpenAIServing:
    # Reuse the existing instance
    return tokenization(app)


def models(app: FastAPI) -> OpenAIServingModels:
    return app.state.openai_serving_models


def chat(app: FastAPI) -> Optional[OpenAIServingChat]:
    return app.state.openai_serving_chat


def completion(app: FastAPI) -> Optional[OpenAIServingCompletion]:
    return app.state.openai_serving_completion

def tokenization(app: FastAPI) -> OpenAIServingTokenization:
    return app.state.openai_serving_tokenization


def bytes_to_headers(bytes_data: bytes) -> httpx.Headers:
    headers_dict = json.loads(bytes_data.decode())
    return httpx.Headers(headers_dict)

async def worker_routine(worker_url: str, app: FastAPI,
                   context: zmq.asyncio.Context = None, i: int = 0):
    """Worker routine"""
    try:
        # Socket to talk to dispatcher
        socket = context.socket(zmq.DEALER)
        worker_identity = f"worker-{i}-{uuid.uuid4()}"
        socket.setsockopt(zmq.IDENTITY, worker_identity.encode())
        socket.connect(worker_url)
        logger.info(f"{worker_identity} started at {worker_url}")
        while True:
            identity, url, header, body  = await socket.recv_multipart()
            logger.info(f"worker-{i} Received request identity: [{identity} ]")
            url = url.decode()
            logger.info(f"worker-{i} Received request url: [{url} ]")
            header = bytes_to_headers(header)
            logger.info(f"worker-{i} Received request headers: [{header} ]")
            body = json.loads(body.decode())
            logger.info(f"worker-{i} Received request body: [{body} ]")
            logger.info(f"worker-{i} Calling OpenAI API")
            completionRequest = CompletionRequest(**body)
            createRequest = create_request(url, "POST", body, header)
            generator = await create_completion(app, completionRequest, createRequest)
            logger.info(f"worker-{i} Received response: [{generator} ]")
            if isinstance(generator, ErrorResponse):
                content = generator.model_dump_json()
                context = json.loads(content)
                context.append("status_code", generator.code)
                await socket.send_multipart([identity,  b"application/json", json.dumps(context).encode()])
            elif isinstance(generator, CompletionResponse):
                await socket.send_multipart([identity,  b"application/json", JSONResponse.render(content=generator.model_dump())])
            else:
                async for chunk in generator:
                    logger.info(f"worker-{i} Sending response chunk: [{chunk} ]")
                    await socket.send_multipart([identity,  b"text/event-stream", chunk.encode()])
    except Exception as e:
        logger.error(f"Error in worker routine: {e} worker-{i}")
        logger.error(traceback.format_exc())

async def create_completion(app: FastAPI, request: CompletionRequest, raw_request: Request):
    handler =  completion(app)
    logger.info(f"zmq requset post: {request}")
    if handler is None:
        return base(app).create_error_response(
            message="The model does not support Completions API")

    generator = await handler.create_completion(request, raw_request)
    logger.info(f"zmq requset end post: {generator}")
    return generator


def create_request(path: str, method: str, body: bytes, headers: dict = None):
    scope = {
        'type': 'http',
        'http_version': '1.1',
        'method': method,
        'path': path,
        'headers': list(headers.items()) if headers else [],
    }
    if body:
        scope['body'] = json.dumps(body).encode('utf-8')
    async def receive():
        return {
            'type': 'http.request',
            'body': scope.get('body', b''),
        }
    async def send(message):
        pass 
    return Request(scope, receive=receive, send=send)


if __name__ == "__main__":
    print(bytes_to_headers(b'{"Content-Type": "application/json"}'))
    
