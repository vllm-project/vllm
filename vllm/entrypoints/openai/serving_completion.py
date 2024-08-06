import time
from typing import (AsyncGenerator, AsyncIterator, Callable, Dict, List,
                    Optional)
from typing import Sequence as GenericSequence
from typing import Tuple, cast

from fastapi import Request
from transformers import PreTrainedTokenizer

from vllm.config import ModelConfig
from vllm.engine.protocol import AsyncEngineClient
from vllm.entrypoints.logger import RequestLogger
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (CompletionLogProbs,
                                              CompletionRequest,
                                              CompletionResponse,
                                              CompletionResponseChoice,
                                              CompletionResponseStreamChoice,
                                              CompletionStreamResponse,
                                              UsageInfo)
# yapf: enable
from vllm.entrypoints.openai.serving_engine import (LoRAModulePath,
                                                    OpenAIServing,
                                                    PromptAdapterPath)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sequence import Logprob
from vllm.tracing import (contains_trace_headers, extract_trace_headers,
                          log_tracing_disabled_warning)
from vllm.utils import merge_async_iterators, random_uuid

logger = init_logger(__name__)

TypeTokenIDs = List[int]
TypeTopLogProbs = List[Optional[Dict[int, float]]]
TypeCreateLogProbsFn = Callable[
    [TypeTokenIDs, TypeTopLogProbs, Optional[int], int], CompletionLogProbs]


class OpenAIServingCompletion(OpenAIServing):

    def __init__(
        self,
        async_engine_client: AsyncEngineClient,
        model_config: ModelConfig,
        served_model_names: List[str],
        *,
        lora_modules: Optional[List[LoRAModulePath]],
        prompt_adapters: Optional[List[PromptAdapterPath]],
        request_logger: Optional[RequestLogger],
        return_tokens_as_token_ids: bool = False,
    ):
        super().__init__(async_engine_client=async_engine_client,
                         model_config=model_config,
                         served_model_names=served_model_names,
                         lora_modules=lora_modules,
                         prompt_adapters=prompt_adapters,
                         request_logger=request_logger,
                         return_tokens_as_token_ids=return_tokens_as_token_ids)

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

        model_name = self.served_model_names[0]
        request_id = f"cmpl-{random_uuid()}"
        created_time = int(time.time())

        # Schedule the request and get the result generator.
        generators: List[AsyncIterator[RequestOutput]] = []
        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            tokenizer = await self.async_engine_client.get_tokenizer(
                lora_request)

            guided_decode_logits_processor = (
                await self._guided_decode_logits_processor(request, tokenizer))
            prompts = list(
                self._tokenize_prompt_input_or_inputs(
                    request,
                    tokenizer,
                    request.prompt,
                    truncate_prompt_tokens=request.truncate_prompt_tokens,
                    add_special_tokens=request.add_special_tokens,
                ))

            for i, prompt_inputs in enumerate(prompts):
                sampling_params = request.to_sampling_params(
                    tokenizer,
                    guided_decode_logits_processor,
                    default_max_tokens=self.max_model_len -
                    len(prompt_inputs["prompt_token_ids"]))

                request_id_item = f"{request_id}-{i}"

                self._log_inputs(request_id_item,
                                 prompt_inputs,
                                 params=sampling_params,
                                 lora_request=lora_request,
                                 prompt_adapter_request=prompt_adapter_request)

                is_tracing_enabled = (
                    await self.async_engine_client.is_tracing_enabled())
                trace_headers = None
                if is_tracing_enabled:
                    trace_headers = extract_trace_headers(raw_request.headers)
                if not is_tracing_enabled and contains_trace_headers(
                        raw_request.headers):
                    log_tracing_disabled_warning()

                generator = self.async_engine_client.generate(
                    {"prompt_token_ids": prompt_inputs["prompt_token_ids"]},
                    sampling_params,
                    request_id_item,
                    lora_request=lora_request,
                    prompt_adapter_request=prompt_adapter_request,
                    trace_headers=trace_headers,
                )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        result_generator: AsyncIterator[Tuple[
            int, RequestOutput]] = merge_async_iterators(*generators)

        # Similar to the OpenAI API, when n != best_of, we do not stream the
        # results. In addition, we do not stream the results when use
        # beam search.
        stream = (request.stream
                  and (request.best_of is None or request.n == request.best_of)
                  and not request.use_beam_search)

        # Streaming response
        if stream:
            return self.completion_stream_generator(request,
                                                    raw_request,
                                                    result_generator,
                                                    request_id,
                                                    created_time,
                                                    model_name,
                                                    num_prompts=len(prompts),
                                                    tokenizer=tokenizer)

        # Non-streaming response
        final_res_batch: List[Optional[RequestOutput]] = [None] * len(prompts)
        try:
            async for i, res in result_generator:
                if await raw_request.is_disconnected():
                    # Abort the request if the client disconnects.
                    await self.async_engine_client.abort(f"{request_id}-{i}")
                    return self.create_error_response("Client disconnected")
                final_res_batch[i] = res

            for i, final_res in enumerate(final_res_batch):
                assert final_res is not None

                # The output should contain the input text
                # We did not pass it into vLLM engine to avoid being redundant
                # with the inputs token IDs
                if final_res.prompt is None:
                    final_res.prompt = prompts[i]["prompt"]

            final_res_batch_checked = cast(List[RequestOutput],
                                           final_res_batch)

            response = self.request_output_to_completion_response(
                final_res_batch_checked,
                request,
                request_id,
                created_time,
                model_name,
                tokenizer,
            )
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        if request.stream:
            response_json = response.model_dump_json()

            async def fake_stream_generator() -> AsyncGenerator[str, None]:
                yield f"data: {response_json}\n\n"
                yield "data: [DONE]\n\n"

            return fake_stream_generator()

        return response

    async def completion_stream_generator(
        self,
        request: CompletionRequest,
        raw_request: Request,
        result_generator: AsyncIterator[Tuple[int, RequestOutput]],
        request_id: str,
        created_time: int,
        model_name: str,
        num_prompts: int,
        tokenizer: PreTrainedTokenizer,
    ) -> AsyncGenerator[str, None]:
        num_choices = 1 if request.n is None else request.n
        previous_texts = [""] * num_choices * num_prompts
        previous_num_tokens = [0] * num_choices * num_prompts
        has_echoed = [False] * num_choices * num_prompts

        try:
            async for prompt_idx, res in result_generator:

                # Abort the request if the client disconnects.
                if await raw_request.is_disconnected():
                    await self.async_engine_client.abort(
                        f"{request_id}-{prompt_idx}")
                    raise StopAsyncIteration()

                for output in res.outputs:
                    i = output.index + prompt_idx * num_choices
                    # TODO(simon): optimize the performance by avoiding full
                    # text O(n^2) sending.

                    assert request.max_tokens is not None
                    if request.echo and request.max_tokens == 0:
                        # only return the prompt
                        delta_text = res.prompt
                        delta_token_ids = res.prompt_token_ids
                        out_logprobs = res.prompt_logprobs
                        has_echoed[i] = True
                    elif (request.echo and request.max_tokens > 0
                          and not has_echoed[i]):
                        # echo the prompt and first token
                        delta_text = res.prompt + output.text
                        delta_token_ids = (res.prompt_token_ids +
                                           output.token_ids)
                        out_logprobs = res.prompt_logprobs + (output.logprobs
                                                              or [])
                        has_echoed[i] = True
                    else:
                        # return just the delta
                        delta_text = output.text[len(previous_texts[i]):]
                        delta_token_ids = output.token_ids[
                            previous_num_tokens[i]:]
                        out_logprobs = output.logprobs[previous_num_tokens[
                            i]:] if output.logprobs else None

                    if request.logprobs is not None:
                        assert out_logprobs is not None, (
                            "Did not output logprobs")
                        logprobs = self._create_completion_logprobs(
                            token_ids=delta_token_ids,
                            top_logprobs=out_logprobs,
                            num_output_top_logprobs=request.logprobs,
                            tokenizer=tokenizer,
                            initial_text_offset=len(previous_texts[i]),
                        )
                    else:
                        logprobs = None

                    previous_texts[i] = output.text
                    previous_num_tokens[i] = len(output.token_ids)
                    finish_reason = output.finish_reason
                    stop_reason = output.stop_reason

                    chunk = CompletionStreamResponse(
                        id=request_id,
                        created=created_time,
                        model=model_name,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=i,
                                text=delta_text,
                                logprobs=logprobs,
                                finish_reason=finish_reason,
                                stop_reason=stop_reason,
                            )
                        ])
                    if (request.stream_options
                            and request.stream_options.include_usage):
                        if (request.stream_options.continuous_usage_stats
                                or output.finish_reason is not None):
                            prompt_tokens = len(res.prompt_token_ids)
                            completion_tokens = len(output.token_ids)
                            usage = UsageInfo(
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                total_tokens=prompt_tokens + completion_tokens,
                            )
                        if request.stream_options.continuous_usage_stats:
                            chunk.usage = usage
                        else:
                            chunk.usage = None

                    response_json = chunk.model_dump_json(exclude_unset=False)
                    yield f"data: {response_json}\n\n"

            if (request.stream_options
                    and request.stream_options.include_usage):
                final_usage_chunk = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[],
                    usage=usage,
                )
                final_usage_data = (final_usage_chunk.model_dump_json(
                    exclude_unset=False, exclude_none=True))
                yield f"data: {final_usage_data}\n\n"

        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    def request_output_to_completion_response(
        self,
        final_res_batch: List[RequestOutput],
        request: CompletionRequest,
        request_id: str,
        created_time: int,
        model_name: str,
        tokenizer: PreTrainedTokenizer,
    ) -> CompletionResponse:
        choices: List[CompletionResponseChoice] = []
        num_prompt_tokens = 0
        num_generated_tokens = 0

        for final_res in final_res_batch:
            prompt_token_ids = final_res.prompt_token_ids
            prompt_logprobs = final_res.prompt_logprobs
            prompt_text = final_res.prompt

            for output in final_res.outputs:
                assert request.max_tokens is not None
                if request.echo and request.max_tokens == 0:
                    token_ids = prompt_token_ids
                    out_logprobs = prompt_logprobs
                    output_text = prompt_text
                elif request.echo and request.max_tokens > 0:
                    token_ids = prompt_token_ids + list(output.token_ids)
                    out_logprobs = (prompt_logprobs + output.logprobs
                                    if request.logprobs is not None else None)
                    output_text = prompt_text + output.text
                else:
                    token_ids = output.token_ids
                    out_logprobs = output.logprobs
                    output_text = output.text

                if request.logprobs is not None:
                    assert out_logprobs is not None, "Did not output logprobs"
                    logprobs = self._create_completion_logprobs(
                        token_ids=token_ids,
                        top_logprobs=out_logprobs,
                        tokenizer=tokenizer,
                        num_output_top_logprobs=request.logprobs,
                    )
                else:
                    logprobs = None

                choice_data = CompletionResponseChoice(
                    index=len(choices),
                    text=output_text,
                    logprobs=logprobs,
                    finish_reason=output.finish_reason,
                    stop_reason=output.stop_reason,
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

    def _create_completion_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[Optional[Dict[int, Logprob]]],
        num_output_top_logprobs: int,
        tokenizer: PreTrainedTokenizer,
        initial_text_offset: int = 0,
    ) -> CompletionLogProbs:
        """Create logprobs for OpenAI Completion API."""
        out_text_offset: List[int] = []
        out_token_logprobs: List[Optional[float]] = []
        out_tokens: List[str] = []
        out_top_logprobs: List[Optional[Dict[str, float]]] = []

        last_token_len = 0

        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None:
                token = tokenizer.decode(token_id)
                if self.return_tokens_as_token_ids:
                    token = f"token_id:{token_id}"
                out_tokens.append(token)
                out_token_logprobs.append(None)
                out_top_logprobs.append(None)
            else:
                token = self._get_decoded_token(
                    step_top_logprobs[token_id],
                    token_id,
                    tokenizer,
                    return_as_token_id=self.return_tokens_as_token_ids)
                token_logprob = max(step_top_logprobs[token_id].logprob,
                                    -9999.0)
                out_tokens.append(token)
                out_token_logprobs.append(token_logprob)

                # makes sure to add the top num_output_top_logprobs + 1
                # logprobs, as defined in the openai API
                # (cf. https://github.com/openai/openai-openapi/blob/
                # 893ba52242dbd5387a97b96444ee1c742cfce9bd/openapi.yaml#L7153)
                out_top_logprobs.append({
                    # Convert float("-inf") to the
                    # JSON-serializable float that OpenAI uses
                    self._get_decoded_token(
                        top_lp[1],
                        top_lp[0],
                        tokenizer,
                        return_as_token_id=self.return_tokens_as_token_ids):
                    max(top_lp[1].logprob, -9999.0)
                    for i, top_lp in enumerate(step_top_logprobs.items())
                    if num_output_top_logprobs >= i
                })

            if len(out_text_offset) == 0:
                out_text_offset.append(initial_text_offset)
            else:
                out_text_offset.append(out_text_offset[-1] + last_token_len)
            last_token_len = len(token)

        return CompletionLogProbs(
            text_offset=out_text_offset,
            token_logprobs=out_token_logprobs,
            tokens=out_tokens,
            top_logprobs=out_top_logprobs,
        )
