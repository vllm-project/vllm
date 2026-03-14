# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Sequence
from http import HTTPStatus
from typing import Any

from openai_harmony import Message as OpenAIMessage

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    ConversationMessage,
)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
)
from vllm.entrypoints.openai.models.serving import OpenAIModelRegistry
from vllm.entrypoints.openai.parser.harmony_utils import (
    get_developer_message,
    get_system_message,
    parse_chat_inputs_to_harmony_messages,
    render_for_completion,
)
from vllm.entrypoints.serve.disagg.protocol import (
    GenerateRequest,
    MultiModalFeatures,
    PlaceholderRangeInfo,
)
from vllm.entrypoints.utils import (
    create_error_response,
    get_max_tokens,
)
from vllm.inputs.data import ProcessorInputs, PromptType, SingletonPrompt, TokensPrompt
from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalHashes, MultiModalPlaceholderDict
from vllm.parser import ParserManager
from vllm.renderers import BaseRenderer, merge_kwargs
from vllm.renderers.inputs.preprocess import (
    extract_prompt_components,
    extract_prompt_len,
    parse_model_prompt,
    prompt_to_seq,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser
from vllm.utils import random_uuid
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.utils.mistral import mt as _mt

logger = init_logger(__name__)


class OpenAIServingRender:
    def __init__(
        self,
        model_config: ModelConfig,
        renderer: BaseRenderer,
        io_processor: Any,
        model_registry: OpenAIModelRegistry,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
        enable_auto_tools: bool = False,
        exclude_tools_when_tool_choice_none: bool = False,
        tool_parser: str | None = None,
        default_chat_template_kwargs: dict[str, Any] | None = None,
        log_error_stack: bool = False,
    ) -> None:
        self.model_config = model_config
        self.renderer = renderer
        self.io_processor = io_processor
        self.model_registry = model_registry
        self.request_logger = request_logger
        self.chat_template = chat_template
        self.chat_template_content_format: ChatTemplateContentFormatOption = (
            chat_template_content_format
        )
        self.trust_request_chat_template = trust_request_chat_template
        self.enable_auto_tools = enable_auto_tools
        self.exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none
        self.tool_parser: Callable[[TokenizerLike], ToolParser] | None = (
            ParserManager.get_tool_parser(
                tool_parser_name=tool_parser,
                enable_auto_tools=enable_auto_tools,
                model_name=model_config.model,
            )
        )
        self.default_chat_template_kwargs: dict[str, Any] = (
            default_chat_template_kwargs or {}
        )
        self.log_error_stack = log_error_stack
        self.use_harmony = model_config.hf_config.model_type == "gpt_oss"
        self.supports_browsing = False
        self.supports_code_interpreter = False

        self.default_sampling_params = model_config.get_diff_sampling_param()
        mc = model_config
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

        result = await self.render_chat(request)
        if isinstance(result, ErrorResponse):
            return result

        _, engine_prompts = result

        if len(engine_prompts) != 1:
            return self.create_error_response(
                f"Expected exactly 1 engine prompt, got {len(engine_prompts)}"
            )

        engine_prompt = engine_prompts[0]

        prompt_components = extract_prompt_components(self.model_config, engine_prompt)
        token_ids = prompt_components.token_ids
        if not token_ids:
            return self.create_error_response("No token_ids rendered")
        token_ids = list(token_ids)

        input_length = extract_prompt_len(self.model_config, engine_prompt)
        max_tokens = get_max_tokens(
            self.model_config.max_model_len,
            request.max_completion_tokens
            if request.max_completion_tokens is not None
            else request.max_tokens,
            input_length,
            self.default_sampling_params,
            self.override_max_tokens,
        )
        params = request.to_sampling_params(max_tokens, self.default_sampling_params)

        request_id = f"chatcmpl-{random_uuid()}"

        return GenerateRequest(
            request_id=request_id,
            token_ids=token_ids,
            features=self._extract_mm_features(engine_prompt),
            sampling_params=params,
            model=request.model,
            stream=bool(request.stream),
            stream_options=(request.stream_options if request.stream else None),
            cache_salt=request.cache_salt,
            priority=request.priority,
        )

    async def render_chat(
        self,
        request: ChatCompletionRequest,
    ) -> tuple[list[ConversationMessage], list[ProcessorInputs]] | ErrorResponse:
        """Core preprocessing logic for chat requests (no model/engine check).

        Called directly by render_chat_request and delegated to by
        OpenAIServingChat.render_chat_request after its engine-aware checks.
        """
        tokenizer = self.renderer.tokenizer

        tool_parser = self.tool_parser

        if is_mistral_tokenizer(tokenizer):
            # because of issues with pydantic we need to potentially
            # re-serialize the tool_calls field of the request
            # for more info: see comment in `maybe_serialize_tool_calls`
            _mt.maybe_serialize_tool_calls(request)  # type: ignore[arg-type]
            _mt.truncate_tool_call_ids(request)  # type: ignore[arg-type]
            _mt.validate_request_params(request)

        # Check if tool parsing is unavailable (common condition)
        tool_parsing_unavailable = (
            tool_parser is None
            and not is_mistral_tokenizer(tokenizer)
            and not self.use_harmony
        )

        # Validate tool_choice when tool parsing is required but unavailable
        if tool_parsing_unavailable and request.tool_choice not in (
            None,
            "none",
        ):
            if request.tool_choice == "auto" and not self.enable_auto_tools:
                # for hf tokenizers, "auto" tools requires
                # --enable-auto-tool-choice and --tool-call-parser
                return self.create_error_response(
                    '"auto" tool choice requires '
                    "--enable-auto-tool-choice and --tool-call-parser to be set"
                )
            elif request.tool_choice != "auto":
                # "required" or named tool requires tool parser
                return self.create_error_response(
                    f'tool_choice="{request.tool_choice}" requires '
                    "--tool-call-parser to be set"
                )

        if request.tools is None or (
            request.tool_choice == "none" and self.exclude_tools_when_tool_choice_none
        ):
            tool_dicts = None
        else:
            tool_dicts = [tool.model_dump() for tool in request.tools]

        if not self.use_harmony:
            # Common case.
            error_check_ret = self._validate_chat_template(
                request_chat_template=request.chat_template,
                chat_template_kwargs=request.chat_template_kwargs,
                trust_request_chat_template=self.trust_request_chat_template,
            )
            if error_check_ret is not None:
                return error_check_ret

            conversation, engine_prompts = await self._preprocess_chat(
                request,
                request.messages,
                default_template=self.chat_template,
                default_template_content_format=self.chat_template_content_format,
                default_template_kwargs=self.default_chat_template_kwargs,
                tool_dicts=tool_dicts,
                tool_parser=tool_parser,
            )
        else:
            # For GPT-OSS.
            should_include_tools = tool_dicts is not None
            conversation, engine_prompts = self._make_request_with_harmony(
                request, should_include_tools
            )

        return conversation, engine_prompts

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
        result = await self.render_completion(request)
        if isinstance(result, ErrorResponse):
            return result
        generate_requests: list[GenerateRequest] = []
        for engine_prompt in result:
            prompt_components = extract_prompt_components(
                self.model_config, engine_prompt
            )
            token_ids = prompt_components.token_ids
            if not token_ids:
                return self.create_error_response("No token_ids rendered")
            token_ids = list(token_ids)

            input_length = extract_prompt_len(self.model_config, engine_prompt)
            max_tokens = get_max_tokens(
                self.model_config.max_model_len,
                request.max_tokens,
                input_length,
                self.default_sampling_params,
                self.override_max_tokens,
            )
            params = request.to_sampling_params(
                max_tokens, self.default_sampling_params
            )

            request_id = f"cmpl-{random_uuid()}"

            generate_requests.append(
                GenerateRequest(
                    request_id=request_id,
                    token_ids=token_ids,
                    features=self._extract_mm_features(engine_prompt),
                    sampling_params=params,
                    model=request.model,
                    stream=bool(request.stream),
                    stream_options=(request.stream_options if request.stream else None),
                    cache_salt=request.cache_salt,
                    priority=request.priority,
                )
            )

        return generate_requests

    async def render_completion(
        self,
        request: CompletionRequest,
    ) -> list[ProcessorInputs] | ErrorResponse:
        """Core preprocessing logic for completion requests (no model/engine check).

        Called directly by render_completion_request and delegated to by
        OpenAIServingCompletion.render_completion_request after its engine-aware checks.
        """
        # Return error for unsupported features.
        if request.suffix is not None:
            return self.create_error_response("suffix is not currently supported")

        if request.echo and request.prompt_embeds is not None:
            return self.create_error_response("Echo is unsupported with prompt embeds.")

        if request.prompt_logprobs is not None and request.prompt_embeds is not None:
            return self.create_error_response(
                "prompt_logprobs is not compatible with prompt embeds."
            )

        engine_prompts = await self._preprocess_completion(
            request,
            prompt_input=request.prompt,
            prompt_embeds=request.prompt_embeds,
        )

        return engine_prompts

    @staticmethod
    def _extract_mm_features(
        engine_prompt: ProcessorInputs,
    ) -> MultiModalFeatures | None:
        """Extract multimodal metadata from a rendered engine prompt.

        Returns ``None`` for text-only prompts.
        """
        if engine_prompt.get("type") != "multimodal":
            return None

        # At this point engine_prompt is a MultiModalInputs TypedDict.
        mm_hashes: MultiModalHashes = engine_prompt["mm_hashes"]  # type: ignore[typeddict-item]
        raw_placeholders: MultiModalPlaceholderDict = engine_prompt["mm_placeholders"]  # type: ignore[typeddict-item]

        mm_placeholders = {
            modality: [
                PlaceholderRangeInfo(offset=p.offset, length=p.length) for p in ranges
            ]
            for modality, ranges in raw_placeholders.items()
        }

        return MultiModalFeatures(
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
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

        # Add system message.
        # NOTE: In Chat Completion API, browsing is enabled by default
        # if the model supports it. TODO: Support browsing.
        assert not self.supports_browsing
        assert not self.supports_code_interpreter
        if (reasoning_effort := request.reasoning_effort) == "none":
            raise ValueError(f"Harmony does not support {reasoning_effort=}")

        # Extract client-provided system message content so it can be
        # passed as structured instructions rather than appended as a raw
        # system message (which the Harmony parser cannot handle).
        non_system_messages = []
        system_instructions_parts: list[str] = []
        for msg in request.messages:
            msg_dict = (
                msg if isinstance(msg, dict) else msg.model_dump(exclude_none=True)
            )
            if msg_dict.get("role") == "system":
                content = msg_dict.get("content") or ""
                if isinstance(content, list):
                    content = "".join(
                        c.get("text", "")
                        for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    )
                if content:
                    system_instructions_parts.append(content)
            else:
                non_system_messages.append(msg)
        instructions = "\n".join(system_instructions_parts) or None

        sys_msg = get_system_message(
            reasoning_effort=reasoning_effort,
            browser_description=None,
            python_description=None,
            with_custom_tools=should_include_tools,
            instructions=instructions,
        )
        messages.append(sys_msg)

        # Add developer message.
        if request.tools or instructions:
            dev_msg = get_developer_message(
                instructions=instructions,
                tools=request.tools if should_include_tools else None,  # type: ignore[arg-type]
            )
            messages.append(dev_msg)

        # Add user message (system messages already extracted above).
        messages.extend(parse_chat_inputs_to_harmony_messages(non_system_messages))

        # Render prompt token ids.
        prompt_token_ids = render_for_completion(messages)
        engine_prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

        # Add cache_salt if provided in the request
        if request.cache_salt is not None:
            engine_prompt["cache_salt"] = request.cache_salt

        return messages, [engine_prompt]

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

    def _validate_chat_template(
        self,
        request_chat_template: str | None,
        chat_template_kwargs: dict[str, Any] | None,
        trust_request_chat_template: bool,
    ) -> ErrorResponse | None:
        """Copied from OpenAIServing._validate_chat_template."""
        if not trust_request_chat_template and (
            request_chat_template is not None
            or (
                chat_template_kwargs
                and chat_template_kwargs.get("chat_template") is not None
            )
        ):
            return self.create_error_response(
                "Chat template is passed with request, but "
                "--trust-request-chat-template is not set. "
                "Refused request with untrusted chat template."
            )
        return None

    async def _preprocess_completion(
        self,
        request: Any,
        prompt_input: str | list[str] | list[int] | list[list[int]] | None,
        prompt_embeds: bytes | list[bytes] | None,
    ) -> list[ProcessorInputs]:
        """Copied from OpenAIServing._preprocess_completion."""
        prompts = list[SingletonPrompt | bytes]()
        if prompt_embeds is not None:  # embeds take higher priority
            prompts.extend(prompt_to_seq(prompt_embeds))
        if prompt_input is not None:
            prompts.extend(prompt_to_seq(prompt_input))
        return await self._preprocess_cmpl(request, prompts)

    async def _preprocess_cmpl(
        self,
        request: Any,
        prompts: Sequence[PromptType | bytes],
    ) -> list[ProcessorInputs]:
        """Copied from OpenAIServing._preprocess_cmpl."""
        renderer = self.renderer
        model_config = self.model_config

        parsed_prompts = [
            (
                prompt
                if isinstance(prompt, bytes)
                else parse_model_prompt(model_config, prompt)
            )
            for prompt in prompts
        ]
        tok_params = request.build_tok_params(model_config)

        return await renderer.render_cmpl_async(
            parsed_prompts,
            tok_params,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(request, k, None)) is not None
            },
        )

    async def _preprocess_chat(
        self,
        request: Any,
        messages: list[Any],
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
        default_template_kwargs: dict[str, Any] | None,
        tool_dicts: list[dict[str, Any]] | None = None,
        tool_parser: Callable[[TokenizerLike], ToolParser] | None = None,
    ) -> tuple[list[ConversationMessage], list[ProcessorInputs]]:
        """Copied from OpenAIServing._preprocess_chat.

        Differences: isinstance check is ChatCompletionRequest-only
        (ResponsesRequest not supported here); TODO comment dropped accordingly.
        """
        renderer = self.renderer

        default_template_kwargs = merge_kwargs(
            default_template_kwargs,
            dict(
                tools=tool_dicts,
                tokenize=is_mistral_tokenizer(renderer.tokenizer),
            ),
        )

        tok_params = request.build_tok_params(self.model_config)
        chat_params = request.build_chat_params(
            default_template, default_template_content_format
        ).with_defaults(default_template_kwargs)

        (conversation,), (engine_prompt,) = await renderer.render_chat_async(
            [messages],
            chat_params,
            tok_params,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(request, k, None)) is not None
            },
        )

        # tool parsing is done only if a tool_parser has been set and if
        # tool_choice is not "none" (if tool_choice is "none" but a tool_parser
        # is set, we want to prevent parsing a tool_call hallucinated by the LLM
        if tool_parser is not None:
            tool_choice = getattr(request, "tool_choice", "none")
            if tool_choice != "none":
                if not isinstance(request, ChatCompletionRequest):
                    msg = (
                        "Tool usage is only supported "
                        " for ChatCompletionRequest, but got "
                        f"{type(request).__name__}"
                    )
                    raise NotImplementedError(msg)
                tokenizer = renderer.get_tokenizer()
                request = tool_parser(tokenizer).adjust_request(request=request)  # type: ignore[arg-type]

        return conversation, [engine_prompt]
