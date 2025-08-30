import logging
import os
import uuid
from typing import Union

import aiohttp
import ray
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from ray import actor, serve

from dllm.constants import (BALANCER_ACTOR_NAME, DLLM_NAMESPACE,
                            ENDPOINT_APPLICATION_NAME,
                            ENDPOINT_PROXY_DEPLOYMENT_NAME)
from dllm.entities import DispatchResult

logger = logging.getLogger(__name__)

app = FastAPI()


@serve.deployment(
    name=ENDPOINT_PROXY_DEPLOYMENT_NAME,
    num_replicas=1,
    max_ongoing_requests=4096,
)
@serve.ingress(app)
class ProxyDeployment:
    #: the balancer handle
    _balancer_handle: Union[actor.ActorHandle, None]

    def __init__(self):
        self._balancer_handle = None

    @staticmethod
    async def record_exception_info(e):
        """
        record exception info
        Args:
            e: exception info
        """
        import sys
        import traceback
        exc_info = sys.exc_info()
        logger.info("Error occurred in disagg prefill proxy server")
        logger.info(e)
        logger.info("".join(traceback.format_exception(*exc_info)))

    async def forward_request(self, url: str, headers: dict, data: dict):
        """
        Send request to the inference instance, return the AsyncGenerator reading the content
        Args:
            url: request url
            headers: request header
            data: request data
        Returns:
            AsyncGenerator: the first iteration is the status code, and subsequent iterations are the response content
        """
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(
                total=6 * 60 * 60)) as session:
            async with session.post(url=url, json=data,
                                    headers=headers) as response:
                # Return status code in advance
                yield response.status
                if response.status == 200:
                    async for chunk_bytes in response.content.iter_chunked(
                            1024):
                        yield chunk_bytes
                else:
                    content = await response.read()
                    yield content

    async def forward_request_without_yield(self, url: str, headers: dict,
                                            data: dict):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(
                total=6 * 60 * 60)) as session:
            async with session.post(url=url, json=data,
                                    headers=headers) as response:
                content = await response.read()
                return response.status, content

    async def schedule(self, prompt: str) -> DispatchResult:
        if self._balancer_handle is None:
            self._balancer_handle = ray.get_actor(name=BALANCER_ACTOR_NAME,
                                                  namespace=DLLM_NAMESPACE)
        dispatch_result = await self._balancer_handle.dispatch_request.remote(  # type: ignore # ray remote call
        )
        return dispatch_result

    @app.post("/health")
    async def health(self, request: Request):
        return Response(status_code=200, content="healthy")

    @app.post("/v1/completions")
    async def openai_completions(self, raw_request: Request):
        """
        https://github.com/vllm-project/vllm/blob/main/benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py
        """
        import pydantic
        from vllm.entrypoints.openai.protocol import CompletionRequest

        request_body = await raw_request.json()
        headers = {
            "Authorization":
            f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id":
            raw_request.headers.get("X-Request-Id") or str(uuid.uuid4())
        }

        try:
            request = CompletionRequest(**request_body)
        except pydantic.ValidationError as e:
            return Response(status_code=500, content={"error": str(e)})

        assert isinstance(request.prompt,
                          str), "currently only support one prompt at a time"

        dispatch_result = await self.schedule(request.prompt)
        logger.info(
            f"({headers['X-Request-Id']}) recv request: {request.prompt}, "
            f"prefill to: {dispatch_result.prefill_uri},"
            f"decode to {dispatch_result.decode_uri}")

        try:
            prefill_request = request_body.copy()
            prefill_request["max_tokens"] = 1
            if dispatch_result.prefill_uri:
                status_code, prefill_result = await self.forward_request_without_yield(
                    f"{dispatch_result.prefill_uri}/v1/completions",
                    headers=headers,
                    data=prefill_request,
                )
                if status_code != 200:
                    logger.error(
                        f"prefill request failed, status code:{status_code}, content:{prefill_result}"
                    )
                    return Response(content=prefill_result,
                                    status_code=status_code)

            # return decode
            decode_token_generator = self.forward_request(
                f"{dispatch_result.decode_uri}/v1/completions",
                headers=headers,
                data=request_body,
            )
            status_code = 200
            # Only iterate once, get the status code and transmit it transparently
            async for status in decode_token_generator:
                status_code = status
                break
            return StreamingResponse(
                decode_token_generator,  # type: ignore
                status_code=status_code,  # type: ignore
                media_type="application/octet-stream",
            )
        except Exception as e:
            await self.record_exception_info(e)
            raise


def deploy_endpoint_to_cluster(host: str = "0.0.0.0", port: int = 8000):
    serve.start(http_options=serve.HTTPOptions(host=host, port=port))
    serve.run(ProxyDeployment.bind(), name=ENDPOINT_APPLICATION_NAME)
