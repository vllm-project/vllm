# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from collections.abc import AsyncGenerator
from http import HTTPStatus

from fastapi import Request

from vllm.entrypoints.chat_utils import ConversationMessage
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    RequestResponseMetadata,
    UsageInfo,
)
from vllm.entrypoints.utils import get_max_tokens
from vllm.inputs import EngineInput
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.reasoning import ReasoningParser
from vllm.tokenizers import TokenizerLike
from vllm.utils.async_utils import merge_async_iterators
from vllm.utils.collection_utils import as_list

logger = init_logger(__name__)


class OpenAIServingChatBatch(OpenAIServingChat):
    """Extends OpenAIServingChat with the /v1/chat/completions/batch endpoint.

    Processes N conversations from a single request concurrently and returns
    one choice per conversation indexed 0, 1, ..., N-1.
    """

    async def render_batch_chat_request(
        self,
        request: BatchChatCompletionRequest,
    ) -> tuple[list[list[ConversationMessage]], list[EngineInput]] | ErrorResponse:
        """Validate the model and preprocess a batched chat completion request.

        Performs engine-aware checks then delegates per-conversation
        preprocessing to OpenAIServingRender, validating the chat template
        once for the whole batch.

        Returns:
            A tuple of (all_conversations, engine_prompts) on success — one
            entry per conversation — or an ErrorResponse on failure.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        if self.engine_client.errored:
            raise self.engine_client.dead_error

        render = self.openai_serving_render

        if not render.use_harmony:
            # Common case: validate the chat template once for the whole batch.
            error_check_ret = render.validate_chat_template(
                request_chat_template=request.chat_template,
                chat_template_kwargs=request.chat_template_kwargs,
                trust_request_chat_template=render.trust_request_chat_template,
            )
            if error_check_ret is not None:
                return error_check_ret

        tool_parser = render.tool_parser
        tool_dicts: list[dict] | None = None

        all_conversations: list[list[ConversationMessage]] = []
        all_engine_prompts: list[EngineInput] = []

        for messages in request.messages:
            single_request = request.to_chat_completion_request(messages)
            if render.use_harmony:
                conversation, engine_prompts = render._make_request_with_harmony(
                    single_request, should_include_tools=tool_dicts is not None
                )
            else:
                conversation, engine_prompts = await render.preprocess_chat(
                    single_request,
                    messages,
                    default_template=render.chat_template,
                    default_template_content_format=render.chat_template_content_format,
                    default_template_kwargs=render.default_chat_template_kwargs,
                    tool_dicts=tool_dicts,
                    tool_parser=tool_parser,
                )
            all_conversations.append(conversation)
            all_engine_prompts.append(engine_prompts[0])

        return all_conversations, all_engine_prompts

    async def create_batch_chat_completion(
        self,
        request: BatchChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> ChatCompletionResponse | ErrorResponse:
        """Batch Chat Completion endpoint (/v1/chat/completions/batch).

        Processes N conversations from a single request concurrently and
        returns one choice per conversation indexed 0, 1, ..., N-1.
        Streaming, tool use, and beam search are not supported.
        """
        tokenizer = self.renderer.tokenizer
        assert tokenizer is not None

        reasoning_parser: ReasoningParser | None = None
        if self.reasoning_parser_cls:
            chat_template_kwargs = self._prepare_extra_chat_template_kwargs(
                request.chat_template_kwargs,
                self.default_chat_template_kwargs,
            )
            reasoning_parser = self.reasoning_parser_cls(
                tokenizer,
                chat_template_kwargs=chat_template_kwargs,  # type: ignore[call-arg]
            )

        render_result = await self.render_batch_chat_request(request)
        if isinstance(render_result, ErrorResponse):
            return render_result
        all_conversations, engine_prompts = render_result

        request_id = (
            f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
        )
        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)
        model_name = self.models.model_name(lora_request)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)
        max_model_len = self.model_config.max_model_len

        generators: list[AsyncGenerator[RequestOutput, None]] = []
        for i, engine_prompt in enumerate(engine_prompts):
            sub_request_id = f"{request_id}_{i}"
            max_tokens = get_max_tokens(
                max_model_len,
                request.max_completion_tokens
                if request.max_completion_tokens is not None
                else request.max_tokens,
                self._extract_prompt_len(engine_prompt),
                self.default_sampling_params,
                self.override_max_tokens,
            )
            single_request = request.to_chat_completion_request(request.messages[i])
            sampling_params = single_request.to_sampling_params(
                max_tokens, self.default_sampling_params
            )
            self._log_inputs(
                sub_request_id,
                engine_prompt,
                params=sampling_params,
                lora_request=lora_request,
            )
            trace_headers = (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )
            generators.append(
                self.engine_client.generate(
                    engine_prompt,
                    sampling_params,
                    sub_request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority if hasattr(request, "priority") else 0,
                    data_parallel_rank=data_parallel_rank,
                    reasoning_ended=None,
                )
            )

        return await self.chat_completion_full_generator_batch(
            request,  # type: ignore[arg-type]
            generators,
            request_id,
            model_name,
            all_conversations,
            tokenizer,
            request_metadata,
            reasoning_parser,
        )

    async def chat_completion_full_generator_batch(
        self,
        request: BatchChatCompletionRequest,  # type: ignore[override]
        generators: list[AsyncGenerator[RequestOutput, None]],
        request_id: str,
        model_name: str,
        all_conversations: list[list[ConversationMessage]],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        reasoning_parser: ReasoningParser | None = None,
    ) -> ErrorResponse | ChatCompletionResponse:
        """Handle batched (non-streaming) chat completions.

        Fans out N generators (one per conversation in the batch), collects
        the final output for each, and assembles a single
        ``ChatCompletionResponse`` whose ``choices`` are indexed 0,...,N-1.

        Tool-use and streaming are rejected upstream by the
        ``check_batch_mode`` validator, so neither needs to be handled here.
        """
        created_time = int(time.time())
        role = self.get_chat_request_role(request)  # type: ignore[arg-type]

        final_results: dict[int, RequestOutput] = {}
        try:
            async for prompt_idx, res in merge_async_iterators(*generators):
                final_results[prompt_idx] = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        choices: list[ChatCompletionResponseChoice] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for prompt_idx in range(len(generators)):
            final_res = final_results.get(prompt_idx)
            if final_res is None:
                return self.create_error_response(
                    f"No output received from the engine for prompt {prompt_idx}.",
                    err_type="InternalServerError",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

            assert final_res.prompt_token_ids is not None
            num_prompt_tokens = len(final_res.prompt_token_ids)
            if final_res.encoder_prompt_token_ids is not None:
                num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
            total_prompt_tokens += num_prompt_tokens
            total_completion_tokens += sum(
                len(output.token_ids) for output in final_res.outputs
            )

            for output in final_res.outputs:
                self._raise_if_error(output.finish_reason, request_id)

                if request.logprobs and request.top_logprobs is not None:
                    assert output.logprobs is not None, "Did not output logprobs"
                    logprobs = self._create_chat_logprobs(
                        token_ids=output.token_ids,
                        top_logprobs=output.logprobs,
                        num_output_top_logprobs=request.top_logprobs,
                        tokenizer=tokenizer,
                        return_as_token_id=request.return_token_ids,
                    )
                else:
                    logprobs = None

                if reasoning_parser:
                    reasoning, content = reasoning_parser.extract_reasoning(
                        output.text,
                        request=request,  # type: ignore[arg-type]
                    )
                    if not getattr(request, "include_reasoning", True):
                        reasoning = None
                else:
                    reasoning = None
                    content = output.text

                message = ChatMessage(role=role, reasoning=reasoning, content=content)

                if request.echo:
                    conversation = all_conversations[prompt_idx]
                    last_msg_content: str | list[dict[str, str]] = ""
                    if conversation and "content" in conversation[-1]:
                        last_msg_content = conversation[-1]["content"] or ""
                    if isinstance(last_msg_content, list):
                        last_msg_content = "\n".join(
                            msg["text"] for msg in last_msg_content
                        )
                    message.content = last_msg_content + (message.content or "")

                choice_data = ChatCompletionResponseChoice(
                    index=prompt_idx,
                    message=message,
                    logprobs=logprobs,
                    finish_reason=output.finish_reason
                    if output.finish_reason
                    else "stop",
                    stop_reason=output.stop_reason,
                    token_ids=(
                        as_list(output.token_ids) if request.return_token_ids else None
                    ),
                )
                choices.append(choice_data)

        usage = UsageInfo(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        )
        request_metadata.final_usage_info = usage

        choices.sort(key=lambda c: c.index)

        return ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )
