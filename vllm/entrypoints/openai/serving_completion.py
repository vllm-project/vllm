import asyncio
import time
from fastapi import Request
from typing import AsyncGenerator, AsyncIterator, Callable, List, Optional, Dict, Tuple
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    LogProbs,
    UsageInfo,
)
from vllm.outputs import RequestOutput
from vllm.entrypoints.openai.serving_engine import OpenAIServing, LoRA
from vllm.model_executor.guided_decoding import get_guided_decoding_logits_processor

logger = init_logger(__name__)

TypeTokenIDs = List[int]
TypeTopLogProbs = List[Optional[Dict[int, float]]]
TypeCreateLogProbsFn = Callable[
    [TypeTokenIDs, TypeTopLogProbs, Optional[int], int], LogProbs]


async def completion_stream_generator(
    request: CompletionRequest,
    raw_request: Request,
    on_abort,
    result_generator: AsyncIterator[Tuple[int, RequestOutput]],
    create_logprobs_fn: TypeCreateLogProbsFn,
    request_id: str,
    created_time: int,
    model_name: str,
    num_prompts: int,
) -> AsyncGenerator[str, None]:
    previous_texts = [""] * request.n * num_prompts
    previous_num_tokens = [0] * request.n * num_prompts
    has_echoed = [False] * request.n * num_prompts

    async for prompt_idx, res in result_generator:

        # Abort the request if the client disconnects.
        if await raw_request.is_disconnected():
            await on_abort(f"{request_id}-{prompt_idx}")
            raise StopAsyncIteration()

        for output in res.outputs:
            i = output.index + prompt_idx * request.n
            # TODO(simon): optimize the performance by avoiding full text O(n^2) sending.

            if request.echo and request.max_tokens == 0:
                # only return the prompt
                delta_text = res.prompt
                delta_token_ids = res.prompt_token_ids
                top_logprobs = res.prompt_logprobs
                has_echoed[i] = True
            elif request.echo and request.max_tokens > 0 and not has_echoed[i]:
                # echo the prompt and first token
                delta_text = res.prompt + output.text
                delta_token_ids = res.prompt_token_ids + output.token_ids
                top_logprobs = res.prompt_logprobs + (output.logprobs or [])
                has_echoed[i] = True
            else:
                # return just the delta
                delta_text = output.text[len(previous_texts[i]):]
                delta_token_ids = output.token_ids[previous_num_tokens[i]:]
                top_logprobs = output.logprobs[
                    previous_num_tokens[i]:] if output.logprobs else None

            if request.logprobs is not None:
                assert top_logprobs is not None, "top_logprobs must be provided when logprobs is requested"
                logprobs = create_logprobs_fn(
                    token_ids=delta_token_ids,
                    top_logprobs=top_logprobs,
                    num_output_top_logprobs=request.logprobs,
                    initial_text_offset=len(previous_texts[i]),
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
                ]).model_dump_json()
            yield f"data: {response_json}\n\n"

            if output.finish_reason is not None:  # return final usage
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
                ).model_dump_json()
                yield f"data: {response_json}\n\n"

    yield "data: [DONE]\n\n"


def parse_prompt_format(prompt) -> Tuple[bool, list]:
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


def request_output_to_completion_response(
    final_res_batch: List[RequestOutput],
    request: CompletionRequest,
    create_logprobs_fn: TypeCreateLogProbsFn,
    request_id: str,
    created_time: int,
    model_name: str,
) -> CompletionResponse:
    choices = []
    num_prompt_tokens = 0
    num_generated_tokens = 0
    for final_res in final_res_batch:
        assert final_res is not None
        prompt_token_ids = final_res.prompt_token_ids
        prompt_logprobs = final_res.prompt_logprobs
        prompt_text = final_res.prompt

        for output in final_res.outputs:
            if request.echo and request.max_tokens == 0:
                token_ids = prompt_token_ids
                top_logprobs = prompt_logprobs
                output_text = prompt_text
            elif request.echo and request.max_tokens > 0:
                token_ids = prompt_token_ids + output.token_ids
                top_logprobs = prompt_logprobs + output.logprobs
                output_text = prompt_text + output.text
            else:
                token_ids = output.token_ids
                top_logprobs = output.logprobs
                output_text = output.text

            if request.logprobs is not None:
                logprobs = create_logprobs_fn(
                    token_ids=token_ids,
                    top_logprobs=top_logprobs,
                    num_output_top_logprobs=request.logprobs,
                )
            else:
                logprobs = None

            choice_data = CompletionResponseChoice(
                index=len(choices),
                text=output_text,
                logprobs=logprobs,
                finish_reason=output.finish_reason,
            )
            choices.append(choice_data)

        num_prompt_tokens += len(prompt_token_ids)
        num_generated_tokens += sum(
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


def merge_async_iterators(*iterators):
    """Merge multiple asynchronous iterators into a single iterator.

    This method handle the case where some iterators finish before others.
    When it yields, it yields a tuple (i, item) where i is the index of the
    iterator that yields the item.
    """
    queue = asyncio.Queue()

    finished = [False] * len(iterators)

    async def producer(i, iterator):
        async for item in iterator:
            await queue.put((i, item))
        finished[i] = True

    _tasks = [
        asyncio.create_task(producer(i, iterator))
        for i, iterator in enumerate(iterators)
    ]

    async def consumer():
        while not all(finished) or not queue.empty():
            item = await queue.get()
            yield item
        await asyncio.gather(*_tasks)

    return consumer()


class OpenAIServingCompletion(OpenAIServing):

    def __init__(self,
                 engine: AsyncLLMEngine,
                 served_model: str,
                 lora_modules: Optional[List[LoRA]] = None):
        super().__init__(engine=engine,
                         served_model=served_model,
                         lora_modules=lora_modules)

    async def create_completion(self, request: CompletionRequest,
                                raw_request: Request):
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/completions/create
        for the API specification. This API mimics the OpenAI Completion API.

        NOTE: Currently we do not support the following feature:
            - suffix (the language models we currently support do not support
            suffix)
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # Return error for unsupported features.
        if request.suffix is not None:
            return self.create_error_response(
                "suffix is not currently supported")

        model_name = request.model
        request_id = f"cmpl-{random_uuid()}"
        created_time = int(time.monotonic())

        # Schedule the request and get the result generator.
        generators = []
        try:
            sampling_params = request.to_sampling_params()
            lora_request = self._maybe_get_lora(request)
            guided_decode_logit_processor = (
                await get_guided_decoding_logits_processor(
                    request, self.engine.get_tokenizer()))
            if guided_decode_logit_processor is not None:
                if sampling_params.logits_processors is None:
                    sampling_params.logits_processors = []
                sampling_params.logits_processors.append(
                    guided_decode_logit_processor)
            prompt_is_tokens, prompts = parse_prompt_format(request.prompt)

            for i, prompt in enumerate(prompts):
                if prompt_is_tokens:
                    input_ids = self._validate_prompt_and_tokenize(
                        request, prompt_ids=prompt)
                else:
                    input_ids = self._validate_prompt_and_tokenize(
                        request, prompt=prompt)

                generators.append(
                    self.engine.generate(prompt,
                                         sampling_params,
                                         f"{request_id}-{i}",
                                         prompt_token_ids=input_ids,
                                         lora_request=lora_request))
        except ValueError as e:
            return self.create_error_response(str(e))

        result_generator: AsyncIterator[Tuple[
            int, RequestOutput]] = merge_async_iterators(*generators)

        # Similar to the OpenAI API, when n != best_of, we do not stream the
        # results. In addition, we do not stream the results when use beam search.
        stream = (request.stream
                  and (request.best_of is None or request.n == request.best_of)
                  and not request.use_beam_search)

        # Streaming response
        if stream:
            return completion_stream_generator(request,
                                               raw_request,
                                               self.engine.abort,
                                               result_generator,
                                               self._create_logprobs,
                                               request_id,
                                               created_time,
                                               model_name,
                                               num_prompts=len(prompts))

        # Non-streaming response
        final_res_batch: RequestOutput = [None] * len(prompts)
        async for i, res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(f"{request_id}-{i}")
                return self.create_error_response("Client disconnected")
            final_res_batch[i] = res
        response = request_output_to_completion_response(
            final_res_batch, request, self._create_logprobs, request_id,
            created_time, model_name)

        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        if request.stream:
            response_json = response.model_dump_json()

            async def fake_stream_generator() -> AsyncGenerator[str, None]:
                yield f"data: {response_json}\n\n"
                yield "data: [DONE]\n\n"

            return fake_stream_generator()

        return response
