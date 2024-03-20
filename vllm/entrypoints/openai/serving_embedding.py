import asyncio
import time
from fastapi import Request
from typing import AsyncIterator, List, Tuple
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmeddingResponseData,
    UsageInfo,
)
from vllm.outputs import EmbeddingRequestOutput
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)

TypeTokenIDs = List[int]


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


def request_output_to_embedding_response(
    final_res_batch: List[EmbeddingRequestOutput],
    request_id: str,
    created_time: int,
    model_name: str,
) -> EmbeddingResponse:
    data = []
    num_prompt_tokens = 0
    for idx, final_res in enumerate(final_res_batch):
        assert final_res is not None
        prompt_token_ids = final_res.prompt_token_ids

        embedding_data = EmeddingResponseData(
            index=idx, embedding=final_res.outputs.embedding)
        data.append(embedding_data)

        num_prompt_tokens += len(prompt_token_ids)

    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        total_tokens=num_prompt_tokens,
    )

    return EmbeddingResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        data=data,
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


class OpenAIServingEmbedding(OpenAIServing):

    def __init__(self, engine: AsyncLLMEngine, served_model: str):
        super().__init__(engine=engine,
                         served_model=served_model,
                         lora_modules=None)

    async def create_embedding(self, request: EmbeddingRequest,
                               raw_request: Request):
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/embeddings/create
        for the API specification. This API mimics the OpenAI Embedding API.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # Return error for unsupported features.
        if request.encoding_format == "base64":
            return self.create_error_response(
                "base64 encoding is not currently supported")
        if request.dimensions is not None:
            return self.create_error_response(
                "dimensions is currently not supported")

        model_name = request.model
        request_id = f"cmpl-{random_uuid()}"
        created_time = int(time.monotonic())

        # Schedule the request and get the result generator.
        generators = []
        try:
            prompt_is_tokens, prompts = parse_prompt_format(request.input)

            for i, prompt in enumerate(prompts):
                if prompt_is_tokens:
                    input_ids = self._validate_prompt_and_tokenize(
                        request, prompt_ids=prompt)
                else:
                    input_ids = self._validate_prompt_and_tokenize(
                        request, prompt=prompt)

                generators.append(
                    self.engine.generate(prompt,
                                         SamplingParams(),
                                         f"{request_id}-{i}",
                                         prompt_token_ids=input_ids))
        except ValueError as e:
            return self.create_error_response(str(e))

        result_generator: AsyncIterator[Tuple[
            int, EmbeddingRequestOutput]] = merge_async_iterators(*generators)

        # Non-streaming response
        final_res_batch: EmbeddingRequestOutput = [None] * len(prompts)
        async for i, res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(f"{request_id}-{i}")
                return self.create_error_response("Client disconnected")
            final_res_batch[i] = res
        response = request_output_to_embedding_response(
            final_res_batch, request_id, created_time, model_name)

        return response
