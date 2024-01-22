import time
from fastapi import Request
from typing import AsyncGenerator, AsyncIterator
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.engine.async_llm_engine import AsyncLLMEngine
from .protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    LogProbs,
    UsageInfo,
)
from vllm.outputs import RequestOutput
from vllm.entrypoints.openai.serving_engine import OpenAIServing

logger = init_logger(__name__)


async def completion_stream_generator(
        request: CompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        echo_without_generation, create_logprobs_fn, request_id, created_time,
        model_name) -> AsyncGenerator[str, None]:
    previous_texts = [""] * request.n
    previous_num_tokens = [0] * request.n
    has_echoed = [False] * request.n

    async for res in result_generator:
        # TODO: handle client disconnect for streaming
        for output in res.outputs:
            i = output.index
            delta_text = output.text[len(previous_texts[i]):]
            token_ids = output.token_ids[previous_num_tokens[i]:]
            if request.logprobs is not None:
                top_logprobs = output.logprobs[previous_num_tokens[i]:]
            else:
                top_logprobs = None
            offsets = len(previous_texts[i])
            if request.echo and not has_echoed[i]:
                if not echo_without_generation:
                    delta_text = res.prompt + delta_text
                    token_ids = res.prompt_token_ids + token_ids
                    if top_logprobs:
                        top_logprobs = res.prompt_logprobs + top_logprobs
                else:  # only just return the prompt
                    delta_text = res.prompt
                    token_ids = res.prompt_token_ids
                    if top_logprobs:
                        top_logprobs = res.prompt_logprobs
                has_echoed[i] = True
            if request.logprobs is not None:
                logprobs = create_logprobs_fn(
                    token_ids=token_ids,
                    top_logprobs=top_logprobs,
                    num_output_top_logprobs=request.logprobs,
                    initial_text_offset=offsets,
                )
            else:
                logprobs = None
            previous_texts[i] = output.text
            previous_num_tokens[i] = len(output.token_ids)
            finish_reason = output.finish_reason
            response_json = CompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=model_name,
                choices=[
                    CompletionResponseStreamChoice(
                        index=i,
                        text=delta_text,
                        logprobs=logprobs,
                        finish_reason=finish_reason,
                    )
                ]).model_dump_json(exclude_unset=True)
            yield f"data: {response_json}\n\n"

            if output.finish_reason is not None:
                logprobs = LogProbs() if request.logprobs is not None else None
                prompt_tokens = len(res.prompt_token_ids)
                completion_tokens = len(output.token_ids)
                final_usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )
                response_json = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[
                        CompletionResponseStreamChoice(
                            index=i,
                            text="",
                            logprobs=logprobs,
                            finish_reason=output.finish_reason,
                        )
                    ],
                    usage=final_usage,
                ).model_dump_json(exclude_unset=True)
                yield f"data: {response_json}\n\n"

    yield "data: [DONE]\n\n"


def parse_prompt_format(prompt) -> tuple[bool, list]:
    # get the prompt, openai supports the following
    # "a string, array of strings, array of tokens, or array of token arrays."
    prompt_is_tokens = False
    prompts = [prompt]  # case 1: a string
    if isinstance(prompt, list):
        if len(prompt) == 0:
            raise ValueError("please provide at least one prompt")
        elif isinstance(prompt[0], str):
            prompt_is_tokens = False
            prompts = prompt  # case 2: array of strings
        elif isinstance(prompt[0], int):
            prompt_is_tokens = True
            prompts = [prompt]  # case 3: array of tokens
        elif isinstance(prompt[0], list) and isinstance(prompt[0][0], int):
            prompt_is_tokens = True
            prompts = prompt  # case 4: array of token arrays
        else:
            raise ValueError(
                "prompt must be a string, array of strings, array of tokens, or array of token arrays"
            )
    return prompt_is_tokens, prompts


def request_output_to_completion_response(final_res: RequestOutput, request,
                                          echo_without_generation,
                                          create_logprobs_fn, request_id,
                                          created_time,
                                          model_name) -> CompletionResponse:
    assert final_res is not None
    choices = []
    prompt_token_ids = final_res.prompt_token_ids
    prompt_logprobs = final_res.prompt_logprobs
    prompt_text = final_res.prompt
    for output in final_res.outputs:
        if request.logprobs is not None:
            if not echo_without_generation:
                token_ids = output.token_ids
                top_logprobs = output.logprobs
                if request.echo:
                    token_ids = prompt_token_ids + token_ids
                    top_logprobs = prompt_logprobs + top_logprobs
            else:
                token_ids = prompt_token_ids
                top_logprobs = prompt_logprobs
            logprobs = create_logprobs_fn(
                token_ids=token_ids,
                top_logprobs=top_logprobs,
                num_output_top_logprobs=request.logprobs,
            )
        else:
            logprobs = None
        if not echo_without_generation:
            output_text = output.text
            if request.echo:
                output_text = prompt_text + output_text
        else:
            output_text = prompt_text
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

    return CompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )


class OpenAIServingCompletion(OpenAIServing):

    def __init__(self, engine: AsyncLLMEngine, served_model: str):
        super().__init__(engine=engine, served_model=served_model)

    async def create_completion(self, request: CompletionRequest,
                                raw_request: Request):
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/completions/create
        for the API specification. This API mimics the OpenAI Completion API.

        NOTE: Currently we do not support the following features:
            - suffix (the language models we currently support do not support
            suffix)
            - logit_bias (to be supported by vLLM engine)
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # OpenAI API supports echoing the prompt when max_tokens is 0.
        echo_without_generation = request.echo and request.max_tokens == 0

        # Return error for unsupported features.
        if request.suffix is not None:
            return self.create_error_response(
                "suffix is not currently supported")
        if request.logit_bias is not None and len(request.logit_bias) > 0:
            return self.create_error_response(
                "logit_bias is not currently supported")

        model_name = request.model
        request_id = f"cmpl-{random_uuid()}"
        created_time = int(time.monotonic())

        # Schedule the request and get the result generator.
        try:
            sampling_params = request.to_sampling_params()

            prompt_is_tokens, prompts = parse_prompt_format(request.prompt)

            if len(prompts) > 1:
                raise ValueError(
                    "Batching in completion API is not supported.")
            prompt = prompts[0]

            if prompt_is_tokens:
                input_ids = self._validate_prompt_and_tokenize(
                    request, prompt_ids=prompt)
            else:
                input_ids = self._validate_prompt_and_tokenize(request,
                                                               prompt=prompt)

            result_generator = self.engine.generate(None,
                                                    sampling_params,
                                                    request_id,
                                                    prompt_token_ids=input_ids)
        except ValueError as e:
            return self.create_error_response(str(e))

        # Similar to the OpenAI API, when n != best_of, we do not stream the
        # results. In addition, we do not stream the results when use beam search.
        stream = (request.stream
                  and (request.best_of is None or request.n == request.best_of)
                  and not request.use_beam_search)

        # Streaming response
        if stream:
            return completion_stream_generator(request, result_generator,
                                               echo_without_generation,
                                               self._create_logprobs,
                                               request_id, created_time,
                                               model_name)

        # Non-streaming response
        final_res: RequestOutput = None
        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return self.create_error_response("Client disconnected")
            final_res = res
        response = request_output_to_completion_response(
            final_res, request, echo_without_generation, self._create_logprobs,
            request_id, created_time, model_name)

        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        if request.stream:
            response_json = response.model_dump_json()

            async def fake_stream_generator() -> AsyncGenerator[str, None]:
                yield f"data: {response_json}\n\n"
                yield "data: [DONE]\n\n"

            return fake_stream_generator()

        return response
