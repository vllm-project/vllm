import argparse
import asyncio
import json
import logging
import traceback
from http import HTTPStatus
from typing import AsyncGenerator, AsyncIterable, List, Optional, Union

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Extra, Field, PositiveInt

from vllm import AsyncEngineArgs, AsyncLLMEngine, RequestOutput, SamplingParams
from vllm.logger import init_logger
from vllm.sequence import PromptLogprobs, SampleLogprobs
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import random_uuid

TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
engine = None
tokenizer = None
max_model_len = None
logger = init_logger(__name__)


class GenerateRequestDetails(BaseModel):

    class Config:
        extra = Extra.forbid

    prompt_token_ids: bool = False
    prompt_text: bool = False
    output_token_ids: bool = False
    output_text: bool = False


class GenerateRequest(BaseModel):
    """
    vllm/sampling_params.py
    """

    class Config:
        extra = Extra.forbid

    request_id: str = Field(default_factory=random_uuid)
    prompt: str
    stream: bool = False
    return_full_text: bool = True
    details: GenerateRequestDetails = Field(
        default_factory=GenerateRequestDetails)

    # TODO: It would be nice if `SamplingParams` itself was a Pydantic class - wouldn't need to duplicate params

    # SamplingParams
    n: PositiveInt = 1
    best_of: Optional[PositiveInt] = 1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: int = 0.0
    use_beam_search: bool = False
    length_penalty: float = 1.0
    early_stopping: Union[bool, str] = False
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    ignore_eos: bool = False
    max_tokens: int = 16
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True

    @property
    def sampling_params(self) -> SamplingParams:
        sampling_kwargs = self.dict().copy()
        for key in (
                "request_id",
                "prompt",
                "stream",
                "return_full_text",
                "details",
        ):
            sampling_kwargs.pop(key, None)
        sampling_params = SamplingParams(**sampling_kwargs)
        return sampling_params


class PromptDetails(BaseModel):
    num_tokens: int
    token_ids: Optional[List[int]] = None
    text: Optional[str] = None
    logprobs: Optional[PromptLogprobs] = None


class OutputDetails(BaseModel):
    num_tokens: int
    finish_reason: Optional[str] = None
    text: Optional[str] = None
    token_ids: Optional[List[int]] = None
    logprobs: Optional[SampleLogprobs] = None


class GenerateResponseDetails(BaseModel):
    prompt: PromptDetails
    outputs: List[OutputDetails]


class GenerateResponse(BaseModel):
    request_id: str
    text: List[str]
    details: GenerateResponseDetails


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


def _add_details(generate_request: GenerateRequest,
                 final_output: RequestOutput) -> GenerateResponse:
    prompt = final_output.prompt
    prompt_detail = PromptDetails(
        num_tokens=len(final_output.prompt_token_ids))
    if generate_request.details.prompt_token_ids:
        prompt_detail.token_ids = final_output.prompt_token_ids
    if generate_request.details.prompt_text:
        prompt_detail.text = final_output.prompt
    if generate_request.prompt_logprobs:
        prompt_detail.logprobs = final_output.prompt_logprobs
    text_outputs = []
    output_details = []
    final_output.outputs.sort(key=lambda _o: _o.index)
    for output in final_output.outputs:
        if generate_request.return_full_text:
            text_outputs.append(prompt + output.text)
        else:
            text_outputs.append(output.text)
        output_detail = OutputDetails(
            num_tokens=len(output.token_ids),
            finish_reason=output.finish_reason,
        )
        if generate_request.details.output_token_ids:
            output_detail.token_ids = output.token_ids
        if generate_request.details.output_text:
            output_detail.text = output.text
        if generate_request.logprobs:
            output_detail.logprobs = output.logprobs
        output_details.append(output_detail)

    response = GenerateResponse(
        request_id=final_output.request_id,
        text=text_outputs,
        details=GenerateResponseDetails(prompt=prompt_detail,
                                        outputs=output_details),
    )
    return response


async def check_length(request: GenerateRequest, ) -> Optional[JSONResponse]:
    global tokenizer, max_model_len
    assert (
        tokenizer is not None and max_model_len is not None
    ), "tokenizer and max model length should ne not None to validate length"
    input_ids = tokenizer(request.prompt).input_ids
    token_num = len(input_ids)
    if request.max_tokens is None:
        request.max_tokens = max_model_len - token_num
    if token_num + request.max_tokens > max_model_len:
        error = (
            f"This model's maximum context length is {max_model_len} tokens. "
            f"However, you requested {request.max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{request.max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.")
        response = {"request_id": request.request_id, "error": error}
        return JSONResponse(response, status_code=HTTPStatus.BAD_REQUEST)


async def stream_results(
    generate_request: GenerateRequest,
    results_generator: AsyncIterable[RequestOutput],
) -> AsyncGenerator[bytes, None]:
    request_id = generate_request.request_id
    try:
        final_output = None
        previous_lengths = [0] * generate_request.n
        async for request_output in results_generator:
            if generate_request.return_full_text:
                prompt = request_output.prompt
                text_outputs = [
                    prompt + output.text for output in request_output.outputs
                ]
            else:
                text_outputs = []
                for i, output in enumerate(request_output.outputs):
                    text_outputs.append(output.text[previous_lengths[i]:])
                    previous_lengths[i] = len(output.text)
            response = {"text": text_outputs}
            yield (json.dumps(response) + "\0").encode("utf-8")
            # TODO: Get feedback from vLLM authors if we want to switch to this as this is backwards incompatible. Can we have both?
            # yield (f"data: {json.dumps(response)}\n\n").encode("utf-8")
            final_output = request_output
        assert final_output is not None

        # TODO: Get feedback from vLLM authors if we want to switch to this as this is backwards incompatible. Can we have both?
        # response = _add_details(generate_request=generate_request, final_output=final_output)
        # response = response.dict(exclude_none=True)
        # response.pop("text", None)  # We do not want to return complete text at the end for now
        # yield (f"data: {json.dumps(response)}\n\n").encode("utf-8")
    except Exception:
        await engine.abort(request_id=request_id)
        response = {
            "request_id": request_id,
            "error": "Aborted due to internal server error",
        }
        exc_metadata = response.copy()
        exc_metadata["traceback"] = traceback.format_exc()
        logger.error(f"Aborting request {json.dumps(exc_metadata)}")
    # TODO: Get feedback from vLLM authors if we want to switch to this as this is backwards incompatible. Can we have both?
    # yield (f"data: {json.dumps(response)}\n\n").encode("utf-8")
    # finally:
    #     yield "data: [DONE]\n\n"


@app.post("/generate")
async def generate(generate_request: GenerateRequest,
                   request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_id = generate_request.request_id
    prompt = generate_request.prompt
    stream = generate_request.stream
    try:
        sampling_params = generate_request.sampling_params
    except ValueError as e:
        response = {"request_id": request_id, "error": str(e)}
        logger.error(
            f"Validation error in sampling params {json.dumps(response)}")
        return JSONResponse(response, status_code=400)

    if tokenizer and max_model_len:
        error_response = await check_length(request=generate_request)
        if error_response:
            logger.error(f"Rejecting request {json.dumps(error_response)}")
            return error_response

    results_generator: AsyncIterable[RequestOutput] = engine.generate(
        prompt=prompt, sampling_params=sampling_params, request_id=request_id)

    # Streaming case
    if stream:
        return StreamingResponse(stream_results(
            generate_request=generate_request,
            results_generator=results_generator),
                                 media_type="text/event-stream")

    # Non-streaming case
    final_output = None
    try:
        async for request_output in results_generator:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await engine.abort(request_id)
                return Response(status_code=499)
            final_output = request_output
    except Exception:
        await engine.abort(request_id=request_id)
        response = {
            "request_id": request_id,
            "error": "Aborted due to internal server error",
        }
        exc_metadata = response.copy()
        exc_metadata["traceback"] = traceback.format_exc()
        logger.error(f"Aborting request {json.dumps(exc_metadata)}")
        return JSONResponse(response, status_code=500)

    assert final_output is not None
    response = _add_details(generate_request=generate_request,
                            final_output=final_output)
    response = response.dict(exclude_none=True)
    return JSONResponse(response)


def main():
    global logger, tokenizer, engine, max_model_len
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", type=str, default="info")
    parser.add_argument("--allow-credentials",
                        action="store_true",
                        help="allow credentials")
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=["*"],
                        help="allowed origins")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=["*"],
                        help="allowed methods")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=["*"],
                        help="allowed headers")
    parser.add_argument("--timeout-keep-alive", type=int, default=5)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)

    log_level = getattr(logging, args.log_level.upper())
    for handler in logger.handlers:
        handler.setLevel(log_level)

    logger.info(f"Starting server with args: {args}")

    # Auto detect gpu count
    if engine_args.tensor_parallel_size == -1:
        import torch

        if torch.cuda.is_available():
            engine_args.tensor_parallel_size = max(torch.cuda.device_count(),
                                                   1)
            logger.info(
                f"Setting tensor_parallel_size automatically to {engine_args.tensor_parallel_size}"
            )

    engine = AsyncLLMEngine.from_engine_args(engine_args)

    engine_model_config = asyncio.run(engine.get_model_config())
    max_model_len = engine_model_config.max_model_len

    tokenizer = get_tokenizer(
        engine_model_config.tokenizer,
        tokenizer_mode=engine_model_config.tokenizer_mode,
        trust_remote_code=engine_model_config.trust_remote_code)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=args.timeout_keep_alive)


if __name__ == '__main__':
    main()
