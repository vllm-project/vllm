import time
from fastapi import Request
from typing import AsyncGenerator, Optional
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.engine.async_llm_engine import AsyncLLMEngine
from .protocol import (CompletionRequest, CompletionResponse,
                       CompletionResponseChoice,
                       CompletionResponseStreamChoice,
                       CompletionStreamResponse, LogProbs, UsageInfo)
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.entrypoints.openai.serving_engine import OpenAIServing

logger = init_logger(__name__)


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

        if request.suffix is not None:
            # The language models we currently support do not support suffix.
            return self.create_error_response(
                "suffix is not currently supported")

        if request.logit_bias is not None and len(request.logit_bias) > 0:
            # TODO: support logit_bias in vLLM engine.
            return self.create_error_response(
                "logit_bias is not currently supported")

        model_name = request.model
        request_id = f"cmpl-{random_uuid()}"

        use_token_ids = False
        if isinstance(request.prompt, list):
            if len(request.prompt) == 0:
                return self.create_error_response(
                    "please provide at least one prompt")
            first_element = request.prompt[0]
            if isinstance(first_element, int):
                use_token_ids = True
                prompt = request.prompt
            elif isinstance(first_element, (str, list)):
                # TODO: handles multiple prompt case in list[list[int]]
                if len(request.prompt) > 1:
                    return self.create_error_response(
                        "multiple prompts in a batch is not currently supported"
                    )
                use_token_ids = not isinstance(first_element, str)
                prompt = request.prompt[0]
        else:
            prompt = request.prompt

        if use_token_ids:
            _, error_check_ret = await self._check_length(request,
                                                          prompt_ids=prompt)
        else:
            token_ids, error_check_ret = await self._check_length(
                request, prompt=prompt)
        if error_check_ret is not None:
            return error_check_ret

        created_time = int(time.monotonic())
        try:
            spaces_between_special_tokens = request.spaces_between_special_tokens
            sampling_params = SamplingParams(
                n=request.n,
                best_of=request.best_of,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                repetition_penalty=request.repetition_penalty,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                min_p=request.min_p,
                stop=request.stop,
                stop_token_ids=request.stop_token_ids,
                ignore_eos=request.ignore_eos,
                max_tokens=request.max_tokens
                if not echo_without_generation else 1,
                logprobs=request.logprobs,
                use_beam_search=request.use_beam_search,
                prompt_logprobs=request.logprobs if request.echo else None,
                skip_special_tokens=request.skip_special_tokens,
                spaces_between_special_tokens=spaces_between_special_tokens,
            )
        except ValueError as e:
            return self.create_error_response(str(e))

        if use_token_ids:
            result_generator = self.engine.generate(None,
                                                    sampling_params,
                                                    request_id,
                                                    prompt_token_ids=prompt)
        else:
            result_generator = self.engine.generate(prompt, sampling_params,
                                                    request_id, token_ids)

        # Similar to the OpenAI API, when n != best_of, we do not stream the
        # results. In addition, we do not stream the results when use beam search.
        stream = (request.stream
                  and (request.best_of is None or request.n == request.best_of)
                  and not request.use_beam_search)

        def create_stream_response_json(
            index: int,
            text: str,
            logprobs: Optional[LogProbs] = None,
            finish_reason: Optional[str] = None,
            usage: Optional[UsageInfo] = None,
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
            if usage is not None:
                response.usage = usage
            response_json = response.json(exclude_unset=True,
                                          ensure_ascii=False)

            return response_json

        async def completion_stream_generator() -> AsyncGenerator[str, None]:
            previous_texts = [""] * request.n
            previous_num_tokens = [0] * request.n
            has_echoed = [False] * request.n
            async for res in result_generator:
                res: RequestOutput
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
                        logprobs = self._create_logprobs(
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
                    response_json = create_stream_response_json(
                        index=i,
                        text=delta_text,
                        logprobs=logprobs,
                        finish_reason=finish_reason,
                    )
                    yield f"data: {response_json}\n\n"
                    if output.finish_reason is not None:
                        logprobs = (LogProbs()
                                    if request.logprobs is not None else None)
                        prompt_tokens = len(res.prompt_token_ids)
                        completion_tokens = len(output.token_ids)
                        final_usage = UsageInfo(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        )
                        response_json = create_stream_response_json(
                            index=i,
                            text="",
                            logprobs=logprobs,
                            finish_reason=output.finish_reason,
                            usage=final_usage,
                        )
                        yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        # Streaming response
        if stream:
            return completion_stream_generator()

        # Non-streaming response
        final_res: RequestOutput = None
        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return self.create_error_response("Client disconnected")
            final_res = res
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
                logprobs = self._create_logprobs(
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

            return fake_stream_generator()

        return response
