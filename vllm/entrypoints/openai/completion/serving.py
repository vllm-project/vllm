# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import hashlib
import os
import time
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from http import HTTPStatus
from typing import TYPE_CHECKING, cast

from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.completion.protocol import (
    CompletionLogProbs,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import (
    GenerationError,
    OpenAIServing,
    clamp_prompt_logprobs,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.utils import get_max_tokens, should_include_usage
from vllm.exceptions import VLLMValidationError
from vllm.inputs import EngineInput
from vllm.l0_usdt import fire_l0_probe
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.utils.async_utils import merge_async_iterators
from vllm.utils.collection_utils import as_list

if TYPE_CHECKING:
    from vllm.entrypoints.serve.render.serving import OpenAIServingRender

logger = init_logger(__name__)

_L0_E6_EXPERIMENT_ID = "L0_E6_reject_error_path_validation"


def _l0_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.lower() in {"1", "true", "yes"}


def _l0_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _l0_e6_enabled() -> bool:
    return os.environ.get("VLLM_L0_EXPERIMENT_ID") == _L0_E6_EXPERIMENT_ID


def _l0_digest(value: object) -> str:
    text = str(value)
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def _l0_headers_extra(raw_request: Request | None, request: CompletionRequest) -> dict:
    headers = {}
    if raw_request is not None:
        headers = {str(k).lower(): str(v) for k, v in raw_request.headers.items()}
    extra = {
        "path": "online",
        "endpoint": "completions",
        "stream": bool(request.stream),
    }
    for header_name, extra_name in (
        ("x-l0-case-name", "case_name"),
        ("x-l0-failure-mode", "failure_mode"),
        ("x-l0-prompt-type", "prompt_type"),
        ("x-l0-benchmark-group", "benchmark_group"),
    ):
        value = headers.get(header_name)
        if value is not None:
            extra[extra_name] = value
    stream = _l0_bool(headers.get("x-l0-stream"))
    if stream is not None:
        extra["stream"] = stream
    for header_name, extra_name in (
        ("x-l0-client-request-index", "client_request_index"),
        ("x-l0-concurrency", "concurrency"),
    ):
        value = _l0_int(headers.get(header_name))
        if value is not None:
            extra[extra_name] = value
    return extra


class OpenAIServingCompletion(OpenAIServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        openai_serving_render: "OpenAIServingRender",
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        preserve_completion_request_id: bool = False,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )

        self.openai_serving_render = openai_serving_render
        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_force_include_usage = enable_force_include_usage
        self.preserve_completion_request_id = preserve_completion_request_id

        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        mc = self.model_config
        self.override_max_tokens = (
            self.default_sampling_params.get("max_tokens")
            if mc.generation_config not in ("auto", "vllm")
            else getattr(mc, "override_generation_config", {}).get("max_new_tokens")
        )

    def _l0_e6_request_id(
        self,
        raw_request: Request | None,
        request: CompletionRequest,
    ) -> str:
        return f"cmpl-{self._base_request_id(raw_request, request.request_id)}-0"

    @staticmethod
    def _response_request_id(base_request_id: str, preserve_canonical: bool) -> str:
        """Select the external completion ID without changing ingress identity.

        The default preserves OpenAI-compatible ``cmpl-`` IDs.  The opt-in
        branch is deliberately exact: no prefix, suffix, or alternate identity
        may be introduced between the canonical ingress ID and the response.
        """
        return base_request_id if preserve_canonical else f"cmpl-{base_request_id}"

    def _l0_e6_extra(
        self,
        raw_request: Request | None,
        request: CompletionRequest,
    ) -> dict:
        return _l0_headers_extra(raw_request, request)

    def _l0_e6_should_trace_failure(
        self,
        raw_request: Request | None,
        request: CompletionRequest,
    ) -> bool:
        if not _l0_e6_enabled():
            return False
        return bool(self._l0_e6_extra(raw_request, request).get("failure_mode"))

    def _l0_e6_fire_arrival(
        self,
        raw_request: Request | None,
        request: CompletionRequest,
        request_id_item: str,
    ) -> None:
        fire_l0_probe(
            "request_arrival",
            request_id_item,
            **self._l0_e6_extra(raw_request, request),
        )

    def _l0_e6_fire_reject(
        self,
        raw_request: Request | None,
        request: CompletionRequest,
        request_id_item: str,
        error_response: ErrorResponse,
        reject_reason: str | None = None,
    ) -> None:
        extra = self._l0_e6_extra(raw_request, request)
        reason = reject_reason or str(extra.get("failure_mode") or "request_reject")
        fire_l0_probe(
            "request_reject",
            request_id_item,
            **extra,
            finish_reason="reject",
            reject_reason=reason,
            http_status=error_response.error.code,
            error_type=error_response.error.type,
            error_message_digest=_l0_digest(error_response.error.message),
        )

    def _l0_e6_fire_error(
        self,
        raw_request: Request | None,
        request: CompletionRequest,
        request_id_item: str,
        error_type: str,
        message: object,
        http_status: int = 500,
    ) -> None:
        fire_l0_probe(
            "request_error",
            request_id_item,
            **self._l0_e6_extra(raw_request, request),
            finish_reason="error",
            error_type=error_type,
            error_message_digest=_l0_digest(message),
            http_status=http_status,
        )

    async def render_completion_request(
        self,
        request: CompletionRequest,
    ) -> list[EngineInput] | ErrorResponse:
        """
        Validate the model and preprocess a completion request.

        Delegates preprocessing logic to OpenAIServingRender, adding the
        engine-aware checks (LoRA model validation, engine health).

        Returns:
            A list of engine_inputs on success, or an ErrorResponse on failure.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        return await self.openai_serving_render.render_completion(request)

    async def create_completion(
        self,
        request: CompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | CompletionResponse | ErrorResponse:
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/completions/create
        for the API specification. This API mimics the OpenAI Completion API.

        NOTE: Currently we do not support the following feature:
            - suffix (the language models we currently support do not support
            suffix)
        """
        if request.stream and request.use_beam_search:
            return self.create_error_response(
                "Streaming is not currently supported with beam search"
            )

        l0_e6_trace_failure = self._l0_e6_should_trace_failure(raw_request, request)
        l0_e6_request_id_item = self._l0_e6_request_id(raw_request, request)
        if l0_e6_trace_failure:
            self._l0_e6_fire_arrival(raw_request, request, l0_e6_request_id_item)

        try:
            result = await self.render_completion_request(request)
        except VLLMValidationError as exc:
            if not l0_e6_trace_failure:
                raise
            error_response = self.create_error_response(
                str(exc),
                err_type=exc.__class__.__name__,
                status_code=HTTPStatus.BAD_REQUEST,
            )
            self._l0_e6_fire_reject(
                raw_request,
                request,
                l0_e6_request_id_item,
                error_response,
                str(self._l0_e6_extra(raw_request, request).get("failure_mode") or "validation_error"),
            )
            return error_response
        if isinstance(result, ErrorResponse):
            if l0_e6_trace_failure:
                self._l0_e6_fire_reject(
                    raw_request,
                    request,
                    l0_e6_request_id_item,
                    result,
                )
            return result

        engine_inputs = result

        request_id = self._response_request_id(
            self._base_request_id(raw_request, request.request_id),
            self.preserve_completion_request_id,
        )
        created_time = int(time.time())

        failure_mode = self._l0_e6_extra(raw_request, request).get("failure_mode")
        if (
            failure_mode == "runtime_error_mock"
            and os.environ.get("VLLM_L0_E6_INJECT_RUNTIME_ERROR") == "1"
        ):
            fire_l0_probe(
                "request_id_assigned",
                l0_e6_request_id_item,
                **self._l0_e6_extra(raw_request, request),
                external_request_id=request_id,
                internal_request_id=l0_e6_request_id_item,
            )
            self._l0_e6_fire_error(
                raw_request,
                request,
                l0_e6_request_id_item,
                "InjectedRuntimeError",
                "L0_E6 injected runtime error",
                HTTPStatus.INTERNAL_SERVER_ERROR.value,
            )
            return self.create_error_response(
                "L0_E6 injected runtime error",
                err_type="InjectedRuntimeError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        lora_request = self._maybe_get_adapters(request)

        # Extract data_parallel_rank from header (router can inject it)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)

        # Schedule the request and get the result generator.
        max_model_len = self.model_config.max_model_len
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        for i, engine_input in enumerate(engine_inputs):
            max_tokens = get_max_tokens(
                max_model_len,
                request.max_tokens,
                self._extract_prompt_len(engine_input),
                self.default_sampling_params,
                self.override_max_tokens,
            )

            sampling_params: SamplingParams | BeamSearchParams
            try:
                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(
                        max_tokens, self.default_sampling_params
                    )
                else:
                    sampling_params = request.to_sampling_params(
                        max_tokens,
                        self.default_sampling_params,
                    )
            except Exception as exc:
                error_response = self.create_error_response(
                    str(exc),
                    err_type=exc.__class__.__name__,
                    status_code=HTTPStatus.BAD_REQUEST,
                )
                if l0_e6_trace_failure:
                    self._l0_e6_fire_reject(
                        raw_request,
                        request,
                        l0_e6_request_id_item,
                        error_response,
                    )
                return error_response

            request_id_item = f"{request_id}-{i}"

            self._log_inputs(
                request_id_item,
                engine_input,
                params=sampling_params,
                lora_request=lora_request,
            )

            trace_headers = (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )

            if isinstance(sampling_params, BeamSearchParams):
                generator = self.beam_search(
                    prompt=engine_input,
                    request_id=request_id,
                    params=sampling_params,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                )
            else:
                generator = self.engine_client.generate(
                    engine_input,
                    sampling_params,
                    request_id_item,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                    data_parallel_rank=data_parallel_rank,
                )

            generators.append(generator)

        result_generator = merge_async_iterators(*generators)

        model_name = self.models.model_name(lora_request)
        num_prompts = len(engine_inputs)

        # Streaming response
        tokenizer = self.renderer.tokenizer

        if request.stream:
            return self.completion_stream_generator(
                request,
                engine_inputs,
                result_generator,
                request_id,
                created_time,
                model_name,
                num_prompts=num_prompts,
                tokenizer=tokenizer,
                request_metadata=request_metadata,
                raw_request=raw_request,
            )

        # Non-streaming response
        final_res_batch: list[RequestOutput | None] = [None] * num_prompts
        try:
            async for i, res in result_generator:
                final_res_batch[i] = res

            for i, final_res in enumerate(final_res_batch):
                assert final_res is not None

                # The output should contain the input text
                # We did not pass it into vLLM engine to avoid being redundant
                # with the inputs token IDs
                if final_res.prompt is None:
                    final_res.prompt = self._extract_prompt_text(engine_inputs[i])

            final_res_batch_checked = cast(list[RequestOutput], final_res_batch)

            response = self.request_output_to_completion_response(
                final_res_batch_checked,
                request,
                request_id,
                created_time,
                model_name,
                tokenizer,
                request_metadata,
            )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

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
        engine_inputs: list[EngineInput],
        result_generator: AsyncIterator[tuple[int, RequestOutput]],
        request_id: str,
        created_time: int,
        model_name: str,
        num_prompts: int,
        tokenizer: TokenizerLike | None,
        request_metadata: RequestResponseMetadata,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None]:
        num_choices = 1 if request.n is None else request.n
        previous_text_lens = [0] * num_choices * num_prompts
        previous_num_tokens = [0] * num_choices * num_prompts
        has_echoed = [False] * num_choices * num_prompts
        num_prompt_tokens = [0] * num_prompts
        num_cached_tokens = None
        first_iteration = True

        stream_options = request.stream_options
        include_usage, include_continuous_usage = should_include_usage(
            stream_options, self.enable_force_include_usage
        )

        try:
            async for prompt_idx, res in result_generator:
                prompt_token_ids = res.prompt_token_ids
                prompt_logprobs = res.prompt_logprobs

                if first_iteration:
                    num_cached_tokens = res.num_cached_tokens
                    first_iteration = False

                prompt_text = res.prompt
                if prompt_text is None:
                    engine_input = engine_inputs[prompt_idx]
                    prompt_text = self._extract_prompt_text(engine_input)

                # Prompt details are excluded from later streamed outputs
                if prompt_token_ids is not None:
                    num_prompt_tokens[prompt_idx] = len(prompt_token_ids)

                delta_token_ids: GenericSequence[int]
                out_logprobs: GenericSequence[dict[int, Logprob] | None] | None

                for output in res.outputs:
                    i = output.index + prompt_idx * num_choices

                    # Useful when request.return_token_ids is True
                    # Returning prompt token IDs shares the same logic
                    # with the echo implementation.
                    prompt_token_ids_to_return: list[int] | None = None

                    assert request.max_tokens is not None
                    if request.echo and not has_echoed[i]:
                        assert prompt_token_ids is not None
                        if request.return_token_ids:
                            prompt_text = ""
                        assert prompt_text is not None
                        if request.max_tokens == 0:
                            # only return the prompt
                            delta_text = prompt_text
                            delta_token_ids = prompt_token_ids
                            out_logprobs = prompt_logprobs
                        else:
                            # echo the prompt and first token
                            delta_text = prompt_text + output.text
                            delta_token_ids = [
                                *prompt_token_ids,
                                *output.token_ids,
                            ]
                            out_logprobs = [
                                *(prompt_logprobs or []),
                                *(output.logprobs or []),
                            ]
                        prompt_token_ids_to_return = prompt_token_ids
                        has_echoed[i] = True
                    else:
                        # return just the delta
                        delta_text = output.text
                        delta_token_ids = output.token_ids
                        out_logprobs = output.logprobs

                        # has_echoed[i] is reused here to indicate whether
                        # we have already returned the prompt token IDs.
                        if not has_echoed[i] and request.return_token_ids:
                            prompt_token_ids_to_return = prompt_token_ids
                            has_echoed[i] = True

                        if (
                            not delta_text
                            and not delta_token_ids
                            and not previous_num_tokens[i]
                        ):
                            # Chunked prefill case, don't return empty chunks
                            continue

                    if request.logprobs is not None:
                        assert out_logprobs is not None, "Did not output logprobs"
                        logprobs = self._create_completion_logprobs(
                            token_ids=delta_token_ids,
                            top_logprobs=out_logprobs,
                            num_output_top_logprobs=request.logprobs,
                            tokenizer=tokenizer,
                            initial_text_offset=previous_text_lens[i],
                            return_as_token_id=request.return_tokens_as_token_ids,
                        )
                    else:
                        logprobs = None

                    previous_text_lens[i] += len(output.text)
                    previous_num_tokens[i] += len(output.token_ids)
                    finish_reason = output.finish_reason
                    stop_reason = output.stop_reason

                    self._raise_if_error(finish_reason, request_id)

                    chunk = CompletionStreamResponse(
                        id=request_id,
                        object="text_completion",
                        created=created_time,
                        model=model_name,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=i,
                                text=delta_text,
                                logprobs=logprobs,
                                finish_reason=finish_reason,
                                stop_reason=stop_reason,
                                prompt_token_ids=prompt_token_ids_to_return,
                                token_ids=(
                                    as_list(output.token_ids)
                                    if request.return_token_ids
                                    else None
                                ),
                            )
                        ],
                    )
                    # Stamp on terminal chunk only when no trailing usage chunk
                    # will follow (that one is the true final message).
                    if (
                        not include_usage
                        and self.system_fingerprint is not None
                        and finish_reason is not None
                    ):
                        chunk.system_fingerprint = self.system_fingerprint
                    if include_continuous_usage:
                        prompt_tokens = num_prompt_tokens[prompt_idx]
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        )

                    response_json = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {response_json}\n\n"

            total_prompt_tokens = sum(num_prompt_tokens)
            total_completion_tokens = sum(previous_num_tokens)
            final_usage_info = UsageInfo(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            )

            if self.enable_prompt_tokens_details and num_cached_tokens:
                final_usage_info.prompt_tokens_details = PromptTokenUsageInfo(
                    cached_tokens=num_cached_tokens
                )

            if include_usage:
                final_usage_chunk = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[],
                    usage=final_usage_info,
                    system_fingerprint=self.system_fingerprint,
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=False, exclude_none=True
                )
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            request_metadata.final_usage_info = final_usage_info

        except asyncio.CancelledError:
            for prompt_idx in range(num_prompts):
                fire_l0_probe(
                    "request_terminal",
                    f"{request_id}-{prompt_idx}",
                    **self._l0_e6_extra(raw_request, request),
                    terminal_type="aborted",
                    terminal_source="openai_completion_stream_cancelled",
                    terminal_confidence="inferred",
                    raw_finish_reason="client_disconnected",
                    raw_terminal_code="asyncio_cancelled",
                    finish_reason="client_disconnected",
                )
            raise
        except GeneratorExit:
            for prompt_idx in range(num_prompts):
                fire_l0_probe(
                    "request_terminal",
                    f"{request_id}-{prompt_idx}",
                    **self._l0_e6_extra(raw_request, request),
                    terminal_type="aborted",
                    terminal_source="openai_completion_stream_closed",
                    terminal_confidence="inferred",
                    raw_finish_reason="client_disconnected",
                    raw_terminal_code="generator_exit",
                    finish_reason="client_disconnected",
                )
            raise
        except GenerationError as e:
            yield f"data: {self._convert_generation_error_to_streaming_response(e)}\n\n"
        except Exception as e:
            logger.exception("Error in completion stream generator.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    def request_output_to_completion_response(
        self,
        final_res_batch: list[RequestOutput],
        request: CompletionRequest,
        request_id: str,
        created_time: int,
        model_name: str,
        tokenizer: TokenizerLike | None,
        request_metadata: RequestResponseMetadata,
    ) -> CompletionResponse:
        choices: list[CompletionResponseChoice] = []
        num_prompt_tokens = 0
        num_generated_tokens = 0
        kv_transfer_params = None
        last_final_res = None
        for final_res in final_res_batch:
            last_final_res = final_res
            prompt_token_ids = final_res.prompt_token_ids
            assert prompt_token_ids is not None
            prompt_logprobs = clamp_prompt_logprobs(final_res.prompt_logprobs)
            prompt_text = final_res.prompt

            token_ids: GenericSequence[int]
            out_logprobs: GenericSequence[dict[int, Logprob] | None] | None

            for output in final_res.outputs:
                self._raise_if_error(output.finish_reason, request_id)

                assert request.max_tokens is not None
                if request.echo:
                    if request.return_token_ids:
                        prompt_text = ""
                    assert prompt_text is not None
                    if request.max_tokens == 0:
                        token_ids = prompt_token_ids
                        out_logprobs = prompt_logprobs
                        output_text = prompt_text
                    else:
                        token_ids = [*prompt_token_ids, *output.token_ids]

                        if request.logprobs is None:
                            out_logprobs = None
                        else:
                            assert prompt_logprobs is not None
                            assert output.logprobs is not None
                            out_logprobs = [
                                *prompt_logprobs,
                                *output.logprobs,
                            ]

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
                        return_as_token_id=request.return_tokens_as_token_ids,
                    )
                else:
                    logprobs = None

                choice_data = CompletionResponseChoice(
                    index=len(choices),
                    text=output_text,
                    logprobs=logprobs,
                    finish_reason=output.finish_reason,
                    stop_reason=output.stop_reason,
                    prompt_logprobs=final_res.prompt_logprobs,
                    prompt_token_ids=(
                        prompt_token_ids if request.return_token_ids else None
                    ),
                    token_ids=(
                        as_list(output.token_ids) if request.return_token_ids else None
                    ),
                )
                choices.append(choice_data)

                num_generated_tokens += len(output.token_ids)

            num_prompt_tokens += len(prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

        if (
            self.enable_prompt_tokens_details
            and last_final_res
            and last_final_res.num_cached_tokens
        ):
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=last_final_res.num_cached_tokens
            )

        request_metadata.final_usage_info = usage
        if final_res_batch:
            kv_transfer_params = final_res_batch[0].kv_transfer_params
        return CompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            system_fingerprint=self.system_fingerprint,
            kv_transfer_params=kv_transfer_params,
        )

    def _create_completion_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[dict[int, Logprob] | None],
        num_output_top_logprobs: int,
        tokenizer: TokenizerLike | None,
        initial_text_offset: int = 0,
        return_as_token_id: bool | None = None,
    ) -> CompletionLogProbs:
        """Create logprobs for OpenAI Completion API."""
        out_text_offset: list[int] = []
        out_token_logprobs: list[float | None] = []
        out_tokens: list[str] = []
        out_top_logprobs: list[dict[str, float] | None] = []

        last_token_len = 0

        should_return_as_token_id = (
            return_as_token_id
            if return_as_token_id is not None
            else self.return_tokens_as_token_ids
        )
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None:
                if should_return_as_token_id:
                    token = f"token_id:{token_id}"
                else:
                    if tokenizer is None:
                        raise VLLMValidationError(
                            "Unable to get tokenizer because "
                            "`skip_tokenizer_init=True`",
                            parameter="skip_tokenizer_init",
                            value=True,
                        )

                    token = tokenizer.decode(token_id)

                out_tokens.append(token)
                out_token_logprobs.append(None)
                out_top_logprobs.append(None)
            else:
                step_token = step_top_logprobs[token_id]

                token = self._get_decoded_token(
                    step_token,
                    token_id,
                    tokenizer,
                    return_as_token_id=should_return_as_token_id,
                )
                token_logprob = max(step_token.logprob, -9999.0)

                out_tokens.append(token)
                out_token_logprobs.append(token_logprob)

                # makes sure to add the top num_output_top_logprobs + 1
                # logprobs, as defined in the openai API
                # (cf. https://github.com/openai/openai-openapi/blob/
                # 893ba52242dbd5387a97b96444ee1c742cfce9bd/openapi.yaml#L7153)
                out_top_logprobs.append(
                    {
                        # Convert float("-inf") to the
                        # JSON-serializable float that OpenAI uses
                        self._get_decoded_token(
                            top_lp[1],
                            top_lp[0],
                            tokenizer,
                            return_as_token_id=should_return_as_token_id,
                        ): max(top_lp[1].logprob, -9999.0)
                        for i, top_lp in enumerate(step_top_logprobs.items())
                        if num_output_top_logprobs >= i
                    }
                )

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
