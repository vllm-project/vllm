# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Callable, Final, Optional, Union

import jinja2
from fastapi import Request
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (ErrorResponse,
                                              PromptTokenUsageInfo,
                                              RequestResponseMetadata,
                                              ResponseReasoningItem,
                                              ResponsesRequest,
                                              ResponsesResponse, UsageInfo)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)


class OpenAIServingResponses(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
        chat_template: Optional[str],
        chat_template_content_format: ChatTemplateContentFormatOption,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        expand_tools_even_if_tool_choice_none: bool = False,
        tool_parser: Optional[str] = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            model_config=model_config,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            enable_force_include_usage=enable_force_include_usage,
        )

        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format

        self.reasoning_parser: Optional[Callable[[AnyTokenizer],
                                                 ReasoningParser]] = None
        if reasoning_parser:
            try:
                self.reasoning_parser = (
                    ReasoningParserManager.get_reasoning_parser(
                        reasoning_parser))
                assert self.reasoning_parser is not None
            except Exception as e:
                raise TypeError(
                    f"{reasoning_parser=} has not been registered") from e

        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_force_include_usage = enable_force_include_usage
        self.default_sampling_params = (
            self.model_config.get_diff_sampling_param())
        if self.default_sampling_params:
            source = self.model_config.generation_config
            source = "model" if source == "auto" else source
            logger.info("Using default chat sampling params from %s: %s",
                        source, self.default_sampling_params)

        # HACK(woosuk): This is a hack. We should use a better store.
        # FIXME: This causes a memory leak since we never remove responses
        # from the store.
        self.response_store: dict[str, ResponsesResponse] = {}

    async def create_responses(
        self,
        request: ResponsesRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], ResponsesResponse, ErrorResponse]:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)
            model_name = self._get_model_name(request.model, lora_request)
            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            # Reponses API supports simple text inputs without chat format.
            if isinstance(request.input, str):
                text_input = request.input
                messages = [{"role": "user", "content": text_input}]
            else:
                messages = request.input

            _, request_prompts, engine_prompts = await self._preprocess_chat(
                request,
                tokenizer,
                messages,
                chat_template=self.chat_template,
                chat_template_content_format=self.chat_template_content_format,
            )
        except (ValueError, TypeError, RuntimeError,
                jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        request_metadata = RequestResponseMetadata(
            request_id=request.request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                default_max_tokens = self.max_model_len - len(
                    engine_prompt["prompt_token_ids"])
                sampling_params = request.to_sampling_params(
                    default_max_tokens, self.default_sampling_params)

                self._log_inputs(request.request_id,
                                 request_prompts[i],
                                 params=sampling_params,
                                 lora_request=lora_request,
                                 prompt_adapter_request=prompt_adapter_request)

                trace_headers = (None if raw_request is None else await
                                 self._get_trace_headers(raw_request.headers))

                generator = self.engine_client.generate(
                    engine_prompt,
                    sampling_params,
                    request.request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    prompt_adapter_request=prompt_adapter_request,
                    priority=request.priority,
                )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert len(generators) == 1
        result_generator, = generators

        if request.background:
            created_time = int(time.time())
            response = ResponsesResponse.from_request(
                request,
                sampling_params,
                model_name=model_name,
                created_time=created_time,
                output=[],
                status="queued",
                usage=None,
            )
            self.response_store[response.id] = response
            asyncio.create_task(
                self.responses_full_generator(
                    request,
                    sampling_params,
                    result_generator,
                    model_name,
                    tokenizer,
                    request_metadata,
                    created_time,
                    is_background=True,
                ))
            return response

        if request.stream:
            raise NotImplementedError("Streaming responses are not supported")

        try:
            return await self.responses_full_generator(
                request,
                sampling_params,
                result_generator,
                model_name,
                tokenizer,
                request_metadata,
            )
        except Exception as e:
            return self.create_error_response(str(e))

    async def responses_full_generator(
        self,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        result_generator: AsyncIterator[RequestOutput],
        model_name: str,
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
        is_background: bool = False,
    ) -> Union[ErrorResponse, ResponsesResponse]:
        if created_time is None:
            created_time = int(time.time())
        final_res: Optional[RequestOutput] = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert final_res is not None
        assert len(final_res.outputs) == 1
        final_output = final_res.outputs[0]

        if self.reasoning_parser:
            try:
                reasoning_parser = self.reasoning_parser(tokenizer)
            except RuntimeError as e:
                logger.exception("Error in reasoning parser creation.")
                return self.create_error_response(str(e))

            reasoning_content, content = (
                reasoning_parser.extract_reasoning_content(final_output.text,
                                                           request=request))
        else:
            reasoning_content = None
            content = final_output.text

        output = []
        if reasoning_content:
            reasoning_item = ResponseReasoningItem(
                text=reasoning_content,
                status=None,  # NOTE: Only the last output item has status.
            )
            output.append(reasoning_item)
        if content:
            output_text = ResponseOutputText(
                text=content,
                annotations=[],  # TODO
                type="output_text",
                logprobs=None,  # TODO
            )
            message = ResponseOutputMessage(
                id=f"msg_{random_uuid()}",
                content=[output_text],
                role="assistant",
                status="completed",
                type="message",
            )
            output.append(message)

        # Calculate usage.
        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = len(final_output.token_ids)
        usage = UsageInfo(prompt_tokens=num_prompt_tokens,
                          completion_tokens=num_generated_tokens,
                          total_tokens=num_prompt_tokens +
                          num_generated_tokens)
        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=final_res.num_cached_tokens)
        request_metadata.final_usage_info = usage

        response = ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time,
            output=output,
            status="completed",
            usage=usage,
        )

        response_id = response.id
        if is_background:
            # Background request. The response must be in the store.
            assert response_id in self.response_store
            # Update the response in the store.
            self.response_store[response_id] = response
        else:
            # Not a background request. The response must not be in the store.
            assert response_id not in self.response_store
            if request.store:
                self.response_store[response_id] = response
        return response

    async def retrieve_responses(
        self,
        response_id: str,
    ) -> Union[ErrorResponse, ResponsesResponse]:
        if not response_id.startswith("resp_"):
            return self.create_error_response(
                err_type="invalid_request_error",
                message=(f"Invalid 'response_id': '{response_id}'. "
                         "Expected an ID that begins with 'resp'."),
            )
        if response_id not in self.response_store:
            return self.create_error_response(
                err_type="invalid_request_error",
                message=f"Response with id '{response_id}' not found. ",
            )
        return self.response_store[response_id]
