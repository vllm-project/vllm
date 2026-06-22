# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from http import HTTPStatus
from typing import Any, cast

from openai_harmony import Message as OpenAIMessage

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProbs,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionLogProbs,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    ToolCall,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import resolve_token_id_placeholder
from vllm.entrypoints.openai.parser.harmony_utils import (
    build_harmony_preamble,
    extract_instructions_from_messages,
    parse_chat_inputs_to_harmony_messages,
    render_for_completion,
)
from vllm.entrypoints.serve.disagg.mm_serde import encode_mm_kwargs_item
from vllm.entrypoints.serve.disagg.protocol import (
    DerenderChatRequest,
    DerenderCompletionRequest,
    GenerateRequest,
    MultiModalFeatures,
    PlaceholderRangeInfo,
)
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens
from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.inputs import (
    EngineInput,
    MultiModalHashes,
    MultiModalInput,
    MultiModalPlaceholders,
    tokens_input,
)
from vllm.logger import init_logger
from vllm.renderers.inputs.preprocess import (
    extract_prompt_components,
    extract_prompt_len,
)
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.tokenizers import TokenizerLike
from vllm.utils import random_uuid
from vllm.utils.mistral import mt as _mt

logger = init_logger(__name__)


def _parse_token_id_placeholder(token: str) -> int | None:
    """Extract token ID from a 'token_id:N' placeholder string."""
    if not token.startswith("token_id:"):
        return None
    try:
        return int(token[len("token_id:") :])
    except ValueError:
        return None


def _correct_decoded_token(
    token_id: int, context_token_ids: list[int], tokenizer: TokenizerLike
) -> str:
    """Use preceding tokens as context to fix U+FFFD from byte-fallback.

    Mirrors LogprobsProcessor._correct_decoded_token in v1/engine/logprobs.py.
    """
    max_ctx = min(len(context_token_ids), 4)

    for num_ctx in range(1, max_ctx + 1):
        context = context_token_ids[-num_ctx:]
        full_decoded = tokenizer.decode(context + [token_id])

        if full_decoded.endswith("�"):
            continue

        clean_end = len(context)
        for j in range(len(context) - 1, -1, -1):
            if tokenizer.decode([context[j]]).endswith("�"):
                clean_end = j
            else:
                break

        clean_prefix = tokenizer.decode(context[:clean_end]) if clean_end > 0 else ""

        if full_decoded.startswith(clean_prefix):
            return full_decoded[len(clean_prefix) :]

        common_len = 0
        for a, b in zip(clean_prefix, full_decoded):
            if a != b:
                break
            common_len += 1
        return full_decoded[common_len:]

    return ""


def _resolve_logprobs(
    logprobs: ChatCompletionLogProbs, tokenizer: TokenizerLike
) -> ChatCompletionLogProbs:
    """Resolve token_id:N placeholders in a ChatCompletionLogProbs object."""
    if logprobs.content is None:
        return logprobs

    context_token_ids: list[int] = []
    resolved_content = []

    for entry in logprobs.content:
        token_str, token_bytes = resolve_token_id_placeholder(entry.token, tokenizer)
        sampled_id = _parse_token_id_placeholder(entry.token)

        if token_str.endswith("�") and sampled_id is not None:
            token_str = _correct_decoded_token(sampled_id, context_token_ids, tokenizer)
            token_bytes = list(token_str.encode("utf-8"))

        resolved_top = []
        for top in entry.top_logprobs:
            top_str, top_bytes = resolve_token_id_placeholder(top.token, tokenizer)
            top_id = _parse_token_id_placeholder(top.token)
            if top_str.endswith("�") and top_id is not None:
                top_str = _correct_decoded_token(top_id, context_token_ids, tokenizer)
                top_bytes = list(top_str.encode("utf-8"))
            resolved_top.append(
                top.model_copy(update={"token": top_str, "bytes": top_bytes})
            )

        resolved_content.append(
            entry.model_copy(
                update={
                    "token": token_str,
                    "bytes": token_bytes,
                    "top_logprobs": resolved_top,
                }
            )
        )

        if sampled_id is not None:
            context_token_ids.append(sampled_id)

    return ChatCompletionLogProbs(content=resolved_content)


def _convert_chat_logprobs_to_completion_logprobs(
    logprobs: ChatCompletionLogProbs,
) -> CompletionLogProbs:
    """Convert ChatCompletionLogProbs (per-token objects) to CompletionLogProbs
    (parallel flat lists) as required by the /v1/completions response schema."""
    if logprobs.content is None:
        return CompletionLogProbs()

    tokens: list[str] = []
    token_logprobs: list[float | None] = []
    top_logprobs_list: list[dict[str, float] | None] = []
    text_offset: list[int] = []

    offset = 0
    for entry in logprobs.content:
        text_offset.append(offset)
        tokens.append(entry.token)
        token_logprobs.append(entry.logprob)
        top_logprobs_list.append(
            {t.token: t.logprob for t in entry.top_logprobs}
            if entry.top_logprobs
            else None
        )
        offset += len(entry.token)

    return CompletionLogProbs(
        text_offset=text_offset,
        token_logprobs=token_logprobs,
        tokens=tokens,
        top_logprobs=top_logprobs_list,
    )


class OpenAIServingRender:
    def __init__(
        self,
        online_renderer: "OnlineRenderer",
    ) -> None:
        self.model_config = online_renderer.model_config
        self.online_renderer = online_renderer

        self.renderer = online_renderer.renderer
        self.model_registry = online_renderer.model_registry

        self.supports_browsing = self.online_renderer.supports_browsing
        self.supports_code_interpreter = self.online_renderer.supports_code_interpreter

        self.default_sampling_params = (
            online_renderer.model_config.get_diff_sampling_param()
        )
        mc = online_renderer.model_config
        self.override_max_tokens = (
            self.default_sampling_params.get("max_tokens")
            if mc.generation_config not in ("auto", "vllm")
            else getattr(mc, "override_generation_config", {}).get("max_new_tokens")
        )

    async def render_chat_request(
        self,
        request: ChatCompletionRequest,
    ) -> GenerateRequest | ErrorResponse:
        """Validate the model and preprocess a chat completion request.

        This is the authoritative implementation used directly by the
        GPU-less render server and delegated to by OpenAIServingChat.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        if request.use_beam_search:
            return self.create_error_response(
                "Beam search is not supported by the render endpoint"
            )

        result = await self.online_renderer.render_chat(request, skip_mm_cache=True)
        if isinstance(result, ErrorResponse):
            return result

        _, engine_inputs = result

        if len(engine_inputs) != 1:
            return self.create_error_response(
                f"Expected exactly 1 engine prompt, got {len(engine_inputs)}"
            )

        engine_input = engine_inputs[0]

        prompt_components = extract_prompt_components(self.model_config, engine_input)
        token_ids = prompt_components.token_ids
        if not token_ids:
            return self.create_error_response("No token_ids rendered")
        token_ids = list(token_ids)

        input_length = extract_prompt_len(self.model_config, engine_input)
        max_tokens = get_max_tokens(
            self.model_config.max_model_len,
            request.max_completion_tokens
            if request.max_completion_tokens is not None
            else request.max_tokens,
            input_length,
            self.default_sampling_params,
            self.override_max_tokens,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
        )
        params = request.to_sampling_params(max_tokens, self.default_sampling_params)

        request_id = f"chatcmpl-{random_uuid()}"

        return GenerateRequest(
            request_id=request_id,
            token_ids=token_ids,
            features=self._extract_mm_features(engine_input),
            sampling_params=params,
            model=request.model,
            stream=bool(request.stream),
            stream_options=(request.stream_options if request.stream else None),
            cache_salt=request.cache_salt,
            priority=request.priority,
        )

    async def render_completion_request(
        self,
        request: CompletionRequest,
    ) -> list[GenerateRequest] | ErrorResponse:
        """Validate the model and preprocess a completion request.

        This is the authoritative implementation used directly by the
        GPU-less render server and delegated to by OpenAIServingCompletion.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret
        result = await self.online_renderer.render_completion(
            request, skip_mm_cache=True
        )
        if isinstance(result, ErrorResponse):
            return result
        generate_requests: list[GenerateRequest] = []
        for engine_input in result:
            prompt_components = extract_prompt_components(
                self.model_config, engine_input
            )
            token_ids = prompt_components.token_ids
            if not token_ids:
                return self.create_error_response("No token_ids rendered")
            token_ids = list(token_ids)

            input_length = extract_prompt_len(self.model_config, engine_input)
            max_tokens = get_max_tokens(
                self.model_config.max_model_len,
                request.max_tokens,
                input_length,
                self.default_sampling_params,
                self.override_max_tokens,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
            )
            params = request.to_sampling_params(
                max_tokens, self.default_sampling_params
            )

            request_id = f"cmpl-{random_uuid()}"

            generate_requests.append(
                GenerateRequest(
                    request_id=request_id,
                    token_ids=token_ids,
                    features=self._extract_mm_features(engine_input),
                    sampling_params=params,
                    model=request.model,
                    stream=bool(request.stream),
                    stream_options=(request.stream_options if request.stream else None),
                    cache_salt=request.cache_salt,
                    priority=request.priority,
                )
            )

        return generate_requests

    @staticmethod
    def _extract_mm_features(
        engine_input: EngineInput,
    ) -> MultiModalFeatures | None:
        """Extract multimodal metadata from a rendered engine prompt.

        Returns ``None`` for text-only prompts.
        """
        if engine_input.get("type") != "multimodal":
            return None

        # At this point engine_input is a MultiModalInput TypedDict.
        mm_engine_input = cast(MultiModalInput, engine_input)
        mm_hashes: MultiModalHashes = mm_engine_input["mm_hashes"]
        raw_placeholders: MultiModalPlaceholders = mm_engine_input["mm_placeholders"]

        mm_placeholders = {
            modality: [
                PlaceholderRangeInfo(offset=p.offset, length=p.length) for p in ranges
            ]
            for modality, ranges in raw_placeholders.items()
        }

        # Serialize tensor data per modality.
        kwargs_data: dict[str, list[str | None]] | None = None
        if raw_mm_kwargs := mm_engine_input.get("mm_kwargs"):
            kwargs_data = {}
            for modality, items in raw_mm_kwargs.items():
                kwargs_data[modality] = [
                    encode_mm_kwargs_item(item) if item is not None else None
                    for item in items
                ]

        return MultiModalFeatures(
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
            kwargs_data=kwargs_data,
        )

    def _make_request_with_harmony(
        self,
        request: ChatCompletionRequest,
        should_include_tools: bool = True,
    ):
        """Build Harmony (GPT-OSS) messages and engine prompt from a chat request."""
        messages: list[OpenAIMessage] = []

        # because of issues with pydantic we need to potentially
        # re-serialize the tool_calls field of the request
        # for more info: see comment in `maybe_serialize_tool_calls`
        _mt.maybe_serialize_tool_calls(request)  # type: ignore[arg-type]

        chat_messages = list(request.messages)
        instructions, chat_messages = extract_instructions_from_messages(chat_messages)

        # Add system message.
        # NOTE: In Chat Completion API, browsing is enabled by default
        # if the model supports it. TODO: Support browsing.
        assert not self.supports_browsing
        assert not self.supports_code_interpreter
        if (reasoning_effort := request.reasoning_effort) == "none":
            raise ValueError(f"Harmony does not support {reasoning_effort=}")
        tools = request.tools if should_include_tools else None
        messages.extend(
            build_harmony_preamble(
                instructions=instructions,
                tools=tools,  # type: ignore[arg-type]
                reasoning_effort=reasoning_effort,
                with_custom_tools=should_include_tools,
            )
        )

        # Add remaining conversation messages.
        messages.extend(parse_chat_inputs_to_harmony_messages(chat_messages))

        # Render prompt token ids.
        prompt_token_ids = render_for_completion(messages)
        engine_input = tokens_input(prompt_token_ids, cache_salt=request.cache_salt)

        return messages, [engine_input]

    async def derender_chat_response(
        self,
        request: DerenderChatRequest,
    ) -> ChatCompletionResponse | ErrorResponse:
        """Postprocess a GenerateResponse into a ChatCompletionResponse.

        Non-streaming only: expects the complete GenerateResponse with all
        token IDs present.  Uses ``parser.parse()`` for one-shot extraction.

        When ``request.chat_request`` is provided, the parser splits the
        output into (reasoning, content, tool_calls).  Otherwise falls
        back to plain detokenization.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        tokenizer = self.renderer.get_tokenizer()
        gen = request.generate_response
        chat_request = request.chat_request
        choices: list[ChatCompletionResponseChoice] = []

        try:
            for choice in gen.choices:
                if not choice.token_ids:
                    raise ValueError(
                        f"choice {choice.index} has empty or null token_ids"
                    )

                resolved_logprobs = (
                    _resolve_logprobs(choice.logprobs, tokenizer)
                    if choice.logprobs is not None
                    else None
                )

                if self.parser is not None and chat_request is not None:
                    # Parser path: decode with special tokens preserved
                    # so the parser can see markers like </think>,
                    # <tool_call>, or Harmony channel tokens.
                    decoded_text = tokenizer.decode(
                        choice.token_ids, skip_special_tokens=False
                    )

                    chat_template_kwargs: dict[str, Any] = {}
                    if not self.use_harmony:
                        chat_template_kwargs = (
                            chat_request.build_chat_params(
                                self.chat_template,
                                self.chat_template_content_format,
                            )
                            .with_defaults(self.default_chat_template_kwargs)
                            .chat_template_kwargs
                        )

                    parser = self.parser(
                        tokenizer,
                        chat_request.tools,
                        chat_template_kwargs=chat_template_kwargs,
                    )
                    reasoning, content, tool_calls = parser.parse(
                        decoded_text,
                        chat_request,
                        enable_auto_tools=self.enable_auto_tools,
                        model_output_token_ids=choice.token_ids,
                    )

                    if not getattr(chat_request, "include_reasoning", True):
                        reasoning = None

                    tc_items = (
                        [
                            ToolCall(
                                id=random_uuid(),
                                function=tc,
                            )
                            for tc in tool_calls
                        ]
                        if tool_calls
                        else []
                    )

                    message = ChatMessage(
                        role="assistant",
                        reasoning=reasoning,
                        content=content,
                        tool_calls=tc_items,
                    )
                else:
                    # No parser: plain detokenization.
                    decoded_text = tokenizer.decode(
                        choice.token_ids, skip_special_tokens=True
                    )
                    message = ChatMessage(role="assistant", content=decoded_text)

                choices.append(
                    ChatCompletionResponseChoice(
                        index=choice.index,
                        message=message,
                        logprobs=resolved_logprobs,
                        finish_reason=choice.finish_reason,
                    )
                )
        except ValueError as exc:
            return self.create_error_response(str(exc))

        prompt_tokens = (
            request.prompt_tokens if request.prompt_tokens is not None else 0
        )
        completion_tokens = sum(len(ch.token_ids) for ch in gen.choices if ch.token_ids)
        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        logger.debug(
            "derender_chat request_id=%s model=%s choices=%d completion_tokens=%d",
            gen.request_id,
            request.model,
            len(choices),
            completion_tokens,
        )
        return ChatCompletionResponse(
            id=gen.request_id,
            model=request.model,
            created=int(time.time()),
            choices=choices,
            usage=usage,
            prompt_logprobs=gen.prompt_logprobs,
            kv_transfer_params=gen.kv_transfer_params,
        )

    async def derender_completion_response(
        self,
        request: DerenderCompletionRequest,
    ) -> CompletionResponse | ErrorResponse:
        """Postprocess a list of GenerateResponses into a CompletionResponse.

        Non-streaming only.  Mirrors the multi-prompt completions case: one
        GenerateResponse per prompt, parallel to the list[GenerateRequest]
        from /v1/completions/render.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        n = len(request.generate_responses)
        prompt_tokens_list: list[int] = (
            request.prompt_tokens if request.prompt_tokens is not None else [0] * n
        )

        tokenizer = self.renderer.get_tokenizer()
        choices: list[CompletionResponseChoice] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        index = 0

        for gen, pt in zip(request.generate_responses, prompt_tokens_list):
            for choice in gen.choices:
                if not choice.token_ids:
                    return self.create_error_response(
                        f"choice {choice.index} in response {gen.request_id} "
                        "has empty or null token_ids"
                    )
                decoded_text = tokenizer.decode(
                    choice.token_ids, skip_special_tokens=True
                )
                completion_logprobs = None
                if choice.logprobs is not None:
                    resolved = _resolve_logprobs(choice.logprobs, tokenizer)
                    completion_logprobs = _convert_chat_logprobs_to_completion_logprobs(
                        resolved
                    )
                choices.append(
                    CompletionResponseChoice(
                        index=index,
                        text=decoded_text,
                        finish_reason=choice.finish_reason,
                        logprobs=completion_logprobs,
                    )
                )
                total_completion_tokens += len(choice.token_ids)
                index += 1
            total_prompt_tokens += pt

        if not request.generate_responses:
            return self.create_error_response("generate_responses must not be empty")

        first = request.generate_responses[0]
        kv_params = first.kv_transfer_params
        if any(
            r.kv_transfer_params != kv_params for r in request.generate_responses[1:]
        ):
            logger.warning(
                "derender_completion: kv_transfer_params differ across responses; "
                "setting to None on the aggregated response"
            )
            kv_params = None

        usage = UsageInfo(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        )

        logger.debug(
            "derender_completion request_id=%s model=%s choices=%d"
            " completion_tokens=%d",
            first.request_id,
            request.model,
            len(choices),
            total_completion_tokens,
        )
        return CompletionResponse(
            id=first.request_id,
            model=request.model,
            created=int(time.time()),
            choices=choices,
            usage=usage,
            kv_transfer_params=kv_params,
        )

    def create_error_response(
        self,
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> ErrorResponse:
        return create_error_response(message, err_type, status_code, param)

    async def _check_model(
        self,
        request: Any,
    ) -> ErrorResponse | None:
        return await self.model_registry.check_model(request.model)
