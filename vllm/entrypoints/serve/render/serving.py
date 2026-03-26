# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import Sequence
from copy import copy
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from openai.types.responses import ResponseFunctionToolCall, ResponseOutputItem
from openai.types.responses.tool import Mcp, Tool
from openai_harmony import Message as OpenAIMessage

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
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
    get_user_message,
    has_custom_tools,
    parse_chat_inputs_to_harmony_messages,
    render_for_completion,
)
from vllm.entrypoints.openai.responses.harmony import (
    construct_harmony_previous_input_messages,
    response_input_to_harmony,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    ResponsesResponse,
)
from vllm.entrypoints.openai.responses.utils import (
    construct_input_messages,
    construct_tool_dicts,
    extract_tool_types,
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
from vllm.inputs import (
    EngineInput,
    MultiModalHashes,
    MultiModalPlaceholders,
    PromptType,
    SingletonPrompt,
    tokens_input,
)
from vllm.logger import init_logger
from vllm.parser import ParserManager
from vllm.renderers import BaseRenderer, merge_kwargs
from vllm.renderers.inputs.preprocess import (
    extract_prompt_components,
    extract_prompt_len,
    parse_model_prompt,
    prompt_to_seq,
)
from vllm.tool_parsers import ToolParser
from vllm.utils import random_uuid
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.utils.mistral import mt as _mt

if TYPE_CHECKING:
    from vllm.entrypoints.mcp.tool_server import ToolServer

logger = init_logger(__name__)


def _extract_allowed_tools_from_mcp_requests(
    tools: list[Tool],
) -> dict[str, list[str] | None]:
    """Extract allowed_tools mapping from MCP tool requests."""
    allowed_tools_map: dict[str, list[str] | None] = {}
    for tool in tools:
        if not isinstance(tool, Mcp):
            continue

        allowed_tools_val = None
        if tool.allowed_tools is not None:
            if isinstance(tool.allowed_tools, list):
                allowed_tools_val = tool.allowed_tools
            elif hasattr(tool.allowed_tools, "tool_names"):
                allowed_tools_val = tool.allowed_tools.tool_names

        # Normalize "*" to None (both mean "allow all tools")
        if allowed_tools_val is not None and "*" in allowed_tools_val:
            allowed_tools_val = None

        allowed_tools_map[tool.server_label] = allowed_tools_val
    return allowed_tools_map


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
        self.tool_parser: type[ToolParser] | None = ParserManager.get_tool_parser(
            tool_parser_name=tool_parser,
            enable_auto_tools=enable_auto_tools,
            model_name=model_config.model,
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

        # Responses API support.
        # These are set after init by generate/api_router.py when an engine
        # is available.  GPU-less render servers use the defaults (None / {}).
        self.tool_server: ToolServer | None = None

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

    async def render_chat(
        self,
        request: ChatCompletionRequest,
    ) -> tuple[list[ConversationMessage], list[EngineInput]] | ErrorResponse:
        """Core preprocessing logic for chat requests (no model/engine check).

        Called directly by render_chat_request and delegated to by
        OpenAIServingChat.render_chat_request after its engine-aware checks.
        """
        tokenizer = self.renderer.tokenizer

        tool_parser = self.tool_parser

        if is_mistral_tokenizer(tokenizer):
            # because of issues with pydantic we need to potentially
            # re-serialize the tool_calls field of the request
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
            error_check_ret = self.validate_chat_template(
                request_chat_template=request.chat_template,
                chat_template_kwargs=request.chat_template_kwargs,
                trust_request_chat_template=self.trust_request_chat_template,
            )
            if error_check_ret is not None:
                return error_check_ret

            conversation, engine_inputs = await self.preprocess_chat(
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
            conversation, engine_inputs = self._make_request_with_harmony(
                request, should_include_tools
            )

        return conversation, engine_inputs

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

    async def render_completion(
        self,
        request: CompletionRequest,
    ) -> list[EngineInput] | ErrorResponse:
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

        engine_inputs = await self.preprocess_completion(
            request,
            prompt_input=request.prompt,
            prompt_embeds=request.prompt_embeds,
        )

        return engine_inputs

    async def render_responses_request(
        self,
        request: ResponsesRequest,
    ) -> GenerateRequest | ErrorResponse:
        """Validate the model and preprocess a responses request.

        This is the authoritative implementation used directly by the
        GPU-less render server and delegated to by OpenAIServingResponses.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # GPU-less render server has no store; only new conversations.
        result = await self.render_responses(
            request, prev_response=None, prev_messages=None
        )
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
            request.max_output_tokens,
            input_length,
            self.default_sampling_params,
            self.override_max_tokens,
        )
        params = request.to_sampling_params(max_tokens, self.default_sampling_params)

        request_id = f"resp-{random_uuid()}"

        return GenerateRequest(
            request_id=request_id,
            token_ids=token_ids,
            features=self._extract_mm_features(engine_input),
            sampling_params=params,
            model=request.model,
            stream=bool(request.stream),
            stream_options=None,
            cache_salt=request.cache_salt,
            priority=request.priority,
        )

    async def render_responses(
        self,
        request: ResponsesRequest,
        prev_response: ResponsesResponse | None,
        prev_messages: list[ChatCompletionMessageParam] | None,
    ) -> tuple[list[ChatCompletionMessageParam], list[EngineInput]] | ErrorResponse:
        """Core preprocessing logic for responses requests (no model/engine
        check).

        State lookups (response_store, msg_store) are performed by the
        caller (OpenAIServingResponses) and passed in as parameters.
        """
        if self.use_harmony:
            return self._make_responses_request_with_harmony(
                request, prev_response, prev_messages
            )

        # Non-harmony path.
        tool_dicts = construct_tool_dicts(request.tools, request.tool_choice)
        messages = construct_input_messages(
            request_instructions=request.instructions,
            request_input=request.input,
            prev_msg=prev_messages,
            prev_response_output=prev_response.output if prev_response else None,
        )

        _, engine_inputs = await self.preprocess_chat(
            request,
            messages,
            default_template=self.chat_template,
            default_template_content_format=self.chat_template_content_format,
            default_template_kwargs=None,
            tool_dicts=tool_dicts,
            tool_parser=self.tool_parser,
        )
        return messages, engine_inputs

    # ------------------------------------------------------------------
    # Responses API: Harmony helpers (moved from OpenAIServingResponses)
    # ------------------------------------------------------------------

    def _make_responses_request_with_harmony(
        self,
        request: ResponsesRequest,
        prev_response: ResponsesResponse | None,
        prev_messages: list[ChatCompletionMessageParam] | None,
    ) -> tuple[list[OpenAIMessage], list[EngineInput]]:
        if request.tool_choice != "auto":
            raise NotImplementedError(
                "Only 'auto' tool_choice is supported in response API with Harmony"
            )

        arrival_time = time.time()
        messages = self._construct_responses_input_with_harmony(
            request, prev_response, prev_messages
        )
        prompt_token_ids = render_for_completion(messages)
        engine_input = tokens_input(prompt_token_ids)
        engine_input["arrival_time"] = arrival_time

        # Add cache_salt if provided in the request.
        if request.cache_salt is not None:
            engine_input["cache_salt"] = request.cache_salt

        return messages, [engine_input]

    def _construct_responses_input_with_harmony(
        self,
        request: ResponsesRequest,
        prev_response: ResponsesResponse | None,
        prev_messages: list[ChatCompletionMessageParam] | None,
    ) -> list[OpenAIMessage]:
        messages: list[OpenAIMessage] = []
        if prev_response is None:
            # New conversation.
            tool_types = extract_tool_types(request.tools)
            with_custom_tools = has_custom_tools(tool_types)

            sys_msg = self._construct_responses_harmony_system_message(
                request, with_custom_tools, tool_types
            )
            messages.append(sys_msg)
            if with_custom_tools:
                dev_msg = get_developer_message(
                    instructions=request.instructions, tools=request.tools
                )
                messages.append(dev_msg)
            messages += construct_harmony_previous_input_messages(request)

        else:
            # Continue the previous conversation.
            # FIXME(woosuk): Currently, request params like reasoning and
            # instructions are ignored.
            assert prev_messages is not None
            prev_msgs = prev_messages

            # FIXME(woosuk): The slice-delete-reappend cycle below is
            # currently a no-op --- it removes messages then puts them all
            # back unfiltered.  It may be intentionally deferred (see FIXME
            # above) or redundant if the Harmony encoder already strips
            # analysis messages at render time.  If analysis messages need
            # to be dropped here, add a channel != "analysis" filter when
            # re-appending, similar to auto_drop_analysis_messages in
            # harmony_utils.py.
            if len(prev_msgs) > 0:
                last_msg = prev_msgs[-1]
                assert isinstance(last_msg, OpenAIMessage)
                if last_msg.channel == "final":
                    prev_final_msg_idx = -1
                    for i in range(len(prev_msgs) - 2, -1, -1):
                        prev_msg_i = prev_msgs[i]
                        assert isinstance(prev_msg_i, OpenAIMessage)
                        if prev_msg_i.channel == "final":
                            prev_final_msg_idx = i
                            break
                    recent_turn_msgs = prev_msgs[prev_final_msg_idx + 1 :]
                    del prev_msgs[prev_final_msg_idx + 1 :]
                    for msg in recent_turn_msgs:
                        assert isinstance(msg, OpenAIMessage)
                        prev_msgs.append(msg)
            messages.extend(prev_msgs)
        # Append the new input.
        # Responses API supports simple text inputs without chat format.
        if isinstance(request.input, str):
            # Skip empty string input when previous_input_messages supplies
            # the full conversation history --- an empty trailing user message
            # confuses the model into thinking nothing was sent.
            if request.input or not request.previous_input_messages:
                messages.append(get_user_message(request.input))
        else:
            prev_outputs: list[ResponseOutputItem] = (
                copy(prev_response.output) if prev_response is not None else []
            )
            for response_msg in request.input:
                new_msg = response_input_to_harmony(response_msg, prev_outputs)
                if new_msg is not None and new_msg.author.role != "system":
                    messages.append(new_msg)

                # User passes in a tool call request and its output.  We need
                # to add the tool call request to prev_outputs so that
                # response_input_to_harmony can find the tool call request
                # when parsing the tool call output.
                if isinstance(response_msg, ResponseFunctionToolCall):
                    prev_outputs.append(response_msg)
        return messages

    def _construct_responses_harmony_system_message(
        self,
        request: ResponsesRequest,
        with_custom_tools: bool,
        tool_types: set[str],
    ) -> OpenAIMessage:
        model_identity = self._extract_responses_system_message(request)

        reasoning_effort = request.reasoning.effort if request.reasoning else None

        # Extract allowed_tools from MCP tool requests.
        allowed_tools_map = _extract_allowed_tools_from_mcp_requests(request.tools)

        # Get filtered tool descriptions first.
        browser_description = (
            self.tool_server.get_tool_description(
                "browser", allowed_tools_map.get("web_search_preview")
            )
            if "web_search_preview" in tool_types
            and self.tool_server is not None
            and self.tool_server.has_tool("browser")
            else None
        )
        python_description = (
            self.tool_server.get_tool_description(
                "python", allowed_tools_map.get("code_interpreter")
            )
            if "code_interpreter" in tool_types
            and self.tool_server is not None
            and self.tool_server.has_tool("python")
            else None
        )
        container_description = (
            self.tool_server.get_tool_description(
                "container", allowed_tools_map.get("container")
            )
            if "container" in tool_types
            and self.tool_server is not None
            and self.tool_server.has_tool("container")
            else None
        )

        sys_msg = get_system_message(
            model_identity=model_identity,
            reasoning_effort=reasoning_effort,
            browser_description=browser_description,
            python_description=python_description,
            container_description=container_description,
            instructions=request.instructions,
            with_custom_tools=with_custom_tools,
        )
        return sys_msg

    @staticmethod
    def _extract_responses_system_message(
        request: ResponsesRequest,
    ) -> str | None:
        system_msg = None
        if not isinstance(request.input, str):
            for response_msg in request.input:
                if (
                    isinstance(response_msg, dict)
                    and response_msg.get("role") == "system"
                ):
                    content = response_msg.get("content")
                    if isinstance(content, str):
                        system_msg = content
                    elif isinstance(content, list):
                        for param in content:
                            if (
                                isinstance(param, dict)
                                and param.get("type") == "input_text"
                            ):
                                system_msg = param.get("text")
                                break
                    break
        return system_msg

    @staticmethod
    def _extract_mm_features(
        engine_input: EngineInput,
    ) -> MultiModalFeatures | None:
        """Extract multimodal metadata from a rendered engine prompt.

        Returns ``None`` for text-only prompts.
        """
        if engine_input.get("type") != "multimodal":
            return None

        # At this point engine_input is a MultiModalInputs TypedDict.
        mm_hashes: MultiModalHashes = engine_input["mm_hashes"]  # type: ignore[typeddict-item]
        raw_placeholders: MultiModalPlaceholders = engine_input["mm_placeholders"]  # type: ignore[typeddict-item]

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
        sys_msg = get_system_message(
            reasoning_effort=reasoning_effort,
            browser_description=None,
            python_description=None,
            with_custom_tools=should_include_tools,
        )
        messages.append(sys_msg)

        # Add developer message.
        if request.tools:
            dev_msg = get_developer_message(
                tools=request.tools if should_include_tools else None  # type: ignore[arg-type]
            )
            messages.append(dev_msg)

        # Add user message.
        messages.extend(parse_chat_inputs_to_harmony_messages(request.messages))

        # Render prompt token ids.
        prompt_token_ids = render_for_completion(messages)
        engine_input = tokens_input(prompt_token_ids, cache_salt=request.cache_salt)

        return messages, [engine_input]

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

    def validate_chat_template(
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

    async def preprocess_completion(
        self,
        request: Any,
        prompt_input: str | list[str] | list[int] | list[list[int]] | None,
        prompt_embeds: bytes | list[bytes] | None,
    ) -> list[EngineInput]:
        """Copied from OpenAIServing._preprocess_completion."""
        prompts = list[SingletonPrompt | bytes]()
        if prompt_embeds is not None:  # embeds take higher priority
            prompts.extend(prompt_to_seq(prompt_embeds))
        if prompt_input is not None:
            prompts.extend(prompt_to_seq(prompt_input))
        return await self.preprocess_cmpl(request, prompts)

    async def preprocess_cmpl(
        self,
        request: Any,
        prompts: Sequence[PromptType | bytes],
    ) -> list[EngineInput]:
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

    async def preprocess_chat(
        self,
        request: Any,
        messages: list[Any],
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
        default_template_kwargs: dict[str, Any] | None,
        tool_dicts: list[dict[str, Any]] | None = None,
        tool_parser: type[ToolParser] | None = None,
    ) -> tuple[list[ConversationMessage], list[EngineInput]]:
        """Copied from OpenAIServing._preprocess_chat."""
        renderer = self.renderer
        mm_config = self.model_config.multimodal_config

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
        ).with_defaults(
            default_template_kwargs,
            default_media_io_kwargs=(mm_config.media_io_kwargs if mm_config else None),
            default_mm_processor_kwargs=getattr(request, "mm_processor_kwargs", None),
        )

        (conversation,), (engine_input,) = await renderer.render_chat_async(
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
                if not isinstance(request, ChatCompletionRequest | ResponsesRequest):
                    msg = (
                        "Tool usage is only supported "
                        "for Chat Completions API or Responses API requests, "
                        f"but got {type(request).__name__}"
                    )
                    raise NotImplementedError(msg)
                tokenizer = renderer.get_tokenizer()
                request = tool_parser(tokenizer, request.tools).adjust_request(
                    request=request  # type: ignore[arg-type]
                )

        return conversation, [engine_input]
