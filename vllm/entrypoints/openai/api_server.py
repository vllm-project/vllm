# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/serve/openai_api_server.py

import argparse
import asyncio
import json
import time
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union

import fastapi
import uvicorn
from fastapi import BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from packaging import version

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    CompletionRequest, CompletionResponse, CompletionResponseChoice,
    CompletionResponseStreamChoice, CompletionStreamResponse,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    LogProbs, ModelCard, ModelList, ModelPermission, UsageInfo)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import random_uuid

try:
    import fastchat
    from fastchat.conversation import Conversation, SeparatorStyle
    from fastchat.model.model_adapter import get_conversation_template
    _fastchat_available = True
except ImportError:
    _fastchat_available = False

TIMEOUT_KEEP_ALIVE = 5  # seconds

logger = init_logger(__name__)
served_model = None
app = fastapi.FastAPI()
engine = None


def create_error_response(status_code: HTTPStatus,
                          message: str) -> JSONResponse:
    return JSONResponse(ErrorResponse(message=message,
                                      type="invalid_request_error").dict(),
                        status_code=status_code.value)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):  # pylint: disable=unused-argument
    return create_error_response(HTTPStatus.BAD_REQUEST, str(exc))


async def check_model(request) -> Optional[JSONResponse]:
    if request.model == served_model:
        return
    ret = create_error_response(
        HTTPStatus.NOT_FOUND,
        f"The model `{request.model}` does not exist.",
    )
    return ret


async def get_gen_prompt(request) -> str:
    if not _fastchat_available:
        raise ModuleNotFoundError(
            "fastchat is not installed. Please install fastchat to use "
            "the chat completion and conversation APIs: `$ pip install fschat`"
        )
    if version.parse(fastchat.__version__) < version.parse("0.2.23"):
        raise ImportError(
            f"fastchat version is low. Current version: {fastchat.__version__} "
            "Please upgrade fastchat to use: `$ pip install -U fschat`")

    conv = get_conversation_template(request.model)
    conv = Conversation(
        name=conv.name,
        system_template=conv.system_template,
        system_message=conv.system_message,
        roles=conv.roles,
        messages=list(conv.messages),  # prevent in-place modification
        offset=conv.offset,
        sep_style=SeparatorStyle(conv.sep_style),
        sep=conv.sep,
        sep2=conv.sep2,
        stop_str=conv.stop_str,
        stop_token_ids=conv.stop_token_ids,
    )

    if isinstance(request.messages, str):
        prompt = request.messages
    else:
        for message in request.messages:
            msg_role = message["role"]
            if msg_role == "system":
                conv.system_message = message["content"]
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    return prompt


async def check_length(
    request: Union[ChatCompletionRequest, CompletionRequest],
    prompt: Optional[str] = None,
    prompt_ids: Optional[List[int]] = None
) -> Tuple[List[int], Optional[JSONResponse]]:
    assert (not (prompt is None and prompt_ids is None)
            and not (prompt is not None and prompt_ids is not None)
            ), "Either prompt or prompt_ids should be provided."
    if prompt_ids is not None:
        input_ids = prompt_ids
    else:
        input_ids = tokenizer(prompt).input_ids
    token_num = len(input_ids)

    if token_num + request.max_tokens > max_model_len:
        return input_ids, create_error_response(
            HTTPStatus.BAD_REQUEST,
            f"This model's maximum context length is {max_model_len} tokens. "
            f"However, you requested {request.max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{request.max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.",
        )
    else:
        return input_ids, None


@app.get("/v1/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    model_cards = [
        ModelCard(id=served_model,
                  root=served_model,
                  permission=[ModelPermission()])
    ]
    return ModelList(data=model_cards)


def create_logprobs(
    token_ids: List[int],
    token_logprobs: List[float],
    top_logprobs: Optional[List[Dict[int, float]]] = None,
    initial_text_offset: int = 0,
) -> LogProbs:
    """Create OpenAI-style logprobs."""
    logprobs = LogProbs()
    last_token_len = 0
    if top_logprobs:
        logprobs.top_logprobs = []
    for i, (token_id,
            token_logprob) in enumerate(zip(token_ids, token_logprobs)):
        token = tokenizer.convert_ids_to_tokens(token_id)
        logprobs.tokens.append(token)
        logprobs.token_logprobs.append(token_logprob)
        if len(logprobs.text_offset) == 0:
            logprobs.text_offset.append(initial_text_offset)
        else:
            logprobs.text_offset.append(logprobs.text_offset[-1] +
                                        last_token_len)
        last_token_len = len(token)

        if top_logprobs:
            logprobs.top_logprobs.append({
                tokenizer.convert_ids_to_tokens(i): p
                for i, p in top_logprobs[i].items()
            } if top_logprobs[i] else None)
    return logprobs


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
        - logit_bias (to be supported by vLLM engine)
    """
    logger.info(f"Received chat completion request: {request}")

    if not engine.is_running:
        engine.start_background_loop()

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    if request.logit_bias is not None:
        # TODO: support logit_bias in vLLM engine.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "logit_bias is not currently supported")

    prompt = await get_gen_prompt(request)
    token_ids, error_check_ret = await check_length(request, prompt=prompt)
    if error_check_ret is not None:
        return error_check_ret

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.time())
    try:
        sampling_params = SamplingParams(
            n=request.n,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            max_tokens=request.max_tokens,
            best_of=request.best_of,
            top_k=request.top_k,
            ignore_eos=request.ignore_eos,
            use_beam_search=request.use_beam_search,
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    result_generator = engine.generate(prompt, sampling_params, request_id,
                                       token_ids)

    async def abort_request() -> None:
        await engine.abort(request_id)

    def create_stream_response_json(
        index: int,
        text: str,
        finish_reason: Optional[str] = None,
    ) -> str:
        choice_data = ChatCompletionResponseStreamChoice(
            index=index,
            delta=DeltaMessage(content=text),
            finish_reason=finish_reason,
        )
        response = ChatCompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
        )
        response_json = response.json(ensure_ascii=False)

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        # First chunk with role
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(id=request_id,
                                                 choices=[choice_data],
                                                 model=model_name)
            data = chunk.json(exclude_unset=True, ensure_ascii=False)
            yield f"data: {data}\n\n"

        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                delta_text = output.text[len(previous_texts[i]):]
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)
                response_json = create_stream_response_json(
                    index=i,
                    text=delta_text,
                )
                yield f"data: {response_json}\n\n"
                if output.finish_reason is not None:
                    response_json = create_stream_response_json(
                        index=i,
                        text="",
                        finish_reason=output.finish_reason,
                    )
                    yield f"data: {response_json}\n\n"
        yield "data: [DONE]\n\n"

    # Streaming response
    if request.stream:
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(completion_stream_generator(),
                                 media_type="text/event-stream",
                                 background=background_tasks)

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await abort_request()
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "Client disconnected")
        final_res = res
    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        choice_data = ChatCompletionResponseChoice(
            index=output.index,
            message=ChatMessage(role="assistant", content=output.text),
            finish_reason=output.finish_reason,
        )
        choices.append(choice_data)

    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(
        len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    if request.stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_stream_generator(),
                                 media_type="text/event-stream")

    return response


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    """Completion API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/completions/create
    for the API specification. This API mimics the OpenAI Completion API.

    NOTE: Currently we do not support the following features:
        - echo (since the vLLM engine does not currently support
          getting the logprobs of prompt tokens)
        - suffix (the language models we currently support do not support
          suffix)
        - logit_bias (to be supported by vLLM engine)
    """
    logger.info(f"Received completion request: {request}")

    if not engine.is_running:
        engine.start_background_loop()

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    # OpenAI API supports echoing the prompt when max_tokens is 0.
    echo_self = request.echo and request.max_tokens == 0

    if request.suffix is not None:
        # The language models we currently support do not support suffix.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "suffix is not currently supported")

    if request.logit_bias is not None:
        # TODO: support logit_bias in vLLM engine.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "logit_bias is not currently supported")

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"

    use_token_ids = False
    if isinstance(request.prompt, list):
        if len(request.prompt) == 0:
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "please provide at least one prompt")
        first_element = request.prompt[0]
        if isinstance(first_element, int):
            use_token_ids = True
            prompt = request.prompt
        elif isinstance(first_element, (str, list)):
            # TODO: handles multiple prompt case in list[list[int]]
            if len(request.prompt) > 1:
                return create_error_response(
                    HTTPStatus.BAD_REQUEST,
                    "multiple prompts in a batch is not currently supported")
            use_token_ids = not isinstance(first_element, str)
            prompt = request.prompt[0]
    else:
        prompt = request.prompt

    if use_token_ids:
        _, error_check_ret = await check_length(request, prompt_ids=prompt)
    else:
        token_ids, error_check_ret = await check_length(request, prompt=prompt)
    if error_check_ret is not None:
        return error_check_ret

    created_time = int(time.time())
    try:
        sampling_params = SamplingParams(
            n=request.n,
            best_of=request.best_of,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop,
            ignore_eos=request.ignore_eos,
            max_tokens=request.max_tokens if not echo_self else 1,
            logprobs=request.logprobs,
            use_beam_search=request.use_beam_search,
            echo=request.echo,
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    if use_token_ids:
        result_generator = engine.generate(None,
                                           sampling_params,
                                           request_id,
                                           prompt_token_ids=prompt)
    else:
        result_generator = engine.generate(prompt, sampling_params, request_id,
                                           token_ids)

    # Similar to the OpenAI API, when n != best_of, we do not stream the
    # results. In addition, we do not stream the results when use beam search.
    stream = (request.stream
              and (request.best_of is None or request.n == request.best_of)
              and not request.use_beam_search)

    async def abort_request() -> None:
        await engine.abort(request_id)

    def create_stream_response_json(
        index: int,
        text: str,
        logprobs: Optional[LogProbs] = None,
        finish_reason: Optional[str] = None,
    ) -> str:
        choice_data = CompletionResponseStreamChoice(
            index=index,
            text=text,
            logprobs=logprobs,
            finish_reason=finish_reason,
        )
        response = CompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
        )
        response_json = response.json(ensure_ascii=False)

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        echo_self_ends = [False] * request.n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                if echo_self and echo_self_ends[i]:
                    continue
                delta_text = output.text[len(previous_texts[i]):]
                if request.logprobs is not None:
                    logprobs = create_logprobs(
                        token_ids=output.token_ids[previous_num_tokens[i]:],
                        token_logprobs=output.
                        logprobs[previous_num_tokens[i]:],
                        top_logprobs=output.
                        top_logprobs[previous_num_tokens[i]:]
                        if request.logprobs > 0 else None,
                        initial_text_offset=len(previous_texts[i]),
                    )
                else:
                    logprobs = None
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)
                response_json = create_stream_response_json(
                    index=i,
                    text=delta_text,
                    logprobs=logprobs,
                )
                yield f"data: {response_json}\n\n"
                echo_self_ends[i] = echo_self and len(
                    previous_texts[i]) == len(res.prompt_token_ids)
                if echo_self and echo_self_ends[i]:
                    output.finish_reason = "length"
                if output.finish_reason is not None:
                    logprobs = (LogProbs()
                                if request.logprobs is not None else None)
                    response_json = create_stream_response_json(
                        index=i,
                        text="",
                        logprobs=logprobs,
                        finish_reason=output.finish_reason,
                    )
                    yield f"data: {response_json}\n\n"
        yield "data: [DONE]\n\n"

    # Streaming response
    if stream:
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(completion_stream_generator(),
                                 media_type="text/event-stream",
                                 background=background_tasks)

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await abort_request()
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "Client disconnected")
        final_res = res

    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        if request.logprobs is not None:
            token_ids = (output.token_ids
                         if not echo_self else output.token_ids[:-1])
            token_logprobs = (output.logprobs
                              if not echo_self else output.logprobs[:-1])
            if request.logprobs == 0:
                top_logprobs = None
            else:
                top_logprobs = (output.top_logprobs
                                if not echo_self else output.top_logprobs[:-1])
            logprobs = create_logprobs(
                token_ids=token_ids,
                token_logprobs=token_logprobs,
                top_logprobs=top_logprobs,
            )
        else:
            logprobs = None
        if echo_self:
            output_text = tokenizer.decode(output.token_ids[:-1],
                                           skip_special_tokens=True)
        else:
            output_text = output.text
        choice_data = CompletionResponseChoice(
            index=output.index,
            text=output_text,
            logprobs=logprobs,
            finish_reason=output.finish_reason,
        )
        choices.append(choice_data)

    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(
        len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = CompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    if request.stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_stream_generator(),
                                 media_type="text/event-stream")

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser.add_argument("--host",
                        type=str,
                        default="localhost",
                        help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
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
    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. If not "
                        "specified, the model name will be the same as "
                        "the huggingface name.")

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    logger.info(f"args: {args}")

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args,
                                             start_engine_loop=False)
    engine_model_config = asyncio.run(engine.get_model_config())
    max_model_len = engine_model_config.get_max_model_len()

    # A separate tokenizer to map token IDs to strings.
    tokenizer = get_tokenizer(engine_args.tokenizer,
                              tokenizer_mode=engine_args.tokenizer_mode,
                              trust_remote_code=engine_args.trust_remote_code)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
