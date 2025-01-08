# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import traceback
import uuid
from typing import Optional

import httpx
import zmq
import zmq.asyncio
from fastapi import FastAPI, Request

# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (CompletionRequest,
                                              CompletionResponse,
                                              ErrorResponse)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.openai.serving_tokenization import (
    OpenAIServingTokenization)
from vllm.logger import init_logger

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
                   context: zmq.asyncio.Context, i: int = 0):
    """Worker routine"""
    try:
        # Socket to talk to dispatcher
        socket = context.socket(zmq.DEALER)
        worker_identity = f"worker-{i}-{uuid.uuid4()}"
        socket.setsockopt(zmq.IDENTITY, worker_identity.encode())
        socket.connect(worker_url)
        logger.info("%s started at %s", worker_identity, worker_url)
        while True:
            identity, url, header, body  = await socket.recv_multipart()
            logger.info("worker-%d Received request identity: [ %s ]",
                        i, identity.decode())
            url_str = url.decode()
            logger.info("worker-%d Received request url: [ %s ]",
                        i, url_str)
            headers = bytes_to_headers(header)
            logger.info("worker-%d Received request headers: [ %s ]",
                         i, headers)
            body_json = json.loads(body.decode())
            logger.info("worker-%d  Received request body: [ %s ]",
                        i, body_json)
            logger.info("worker-%d Calling OpenAI API", i)
            completionRequest = CompletionRequest(**body_json)
            createRequest = create_request(url_str, "POST", body_json, headers)
            generator = await create_completion(app, completionRequest,
                                                createRequest)
            logger.info("worker-%d Received response: [ %s ]", i, generator)
            if isinstance(generator, ErrorResponse):
                content = generator.model_dump_json()
                context_json = json.loads(content)
                context_json.append("status_code", generator.code)
                await socket.send_multipart([identity,  b"application/json",
                                             json.dumps(context_json).encode('utf-8')])
            elif isinstance(generator, CompletionResponse):
                await socket.send_multipart([identity,  b"application/json",
                                             json.dumps(generator.model_dump()).encode('utf-8')])
            else:
                async for chunk in generator:
                    logger.info("worker-%d Sending response chunk: [ %s ]",
                                i, chunk)
                    await socket.send_multipart([identity,
                                                 b"text/event-stream",
                                                 chunk.encode('utf-8')])
    except Exception as e:
        logger.error("Error in worker routine: %s worker-%d", e, i)
        logger.error(traceback.format_exc())

async def create_completion(app: FastAPI, request: CompletionRequest,
                            raw_request: Request):
    handler =  completion(app)
    logger.info("zmq request post: %s", request)
    if handler is None:
        return base(app).create_error_response(
            message="The model does not support Completions API")

    generator = await handler.create_completion(request, raw_request)
    logger.info("zmq request end post: %s", generator)
    return generator


def create_request(path: str, method: str, body: dict,
                   headers: httpx.Headers) -> Request:
    scope = {
        'type': 'http',
        'http_version': '1.1',
        'method': method,
        'path': path,
        'headers': list(headers.items()) if headers else [],
    }
    if body:
        scope['body'] = json.dumps(body)
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
