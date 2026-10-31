# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from collections.abc import Callable
from concurrent.futures import Executor
from typing import TYPE_CHECKING, Any, cast

import torch
from fastapi import Request

from vllm.distributed.io_transfer.io_connector.base import IOConnectorBase
from vllm.entrypoints.anthropic.protocol import (
    AnthropicDisabledThinkingEffortOption,
    AnthropicMessagesRequest,
)
from vllm.entrypoints.anthropic.serving import (
    AnthropicServingMessages,
    DisabledThinkingEffortResolver,
)
from vllm.entrypoints.generate.base.protocol import MMPayload
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.models.serving import (
    OpenAIModelRegistry,
    OpenAIServingModels,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.pooling.io_processor import PluginWithIOProcessorPlugins
from vllm.entrypoints.pooling.pooling.protocol import (
    DataLocation,
    DataObject,
    DataRef,
    GeneratePoolingRequest,
    IOProcessorRequest,
    MetadataRef,
)
from vllm.entrypoints.pooling.typing import PoolingServeContext
from vllm.entrypoints.scale_out.token_in_token_out.mm_features import (
    extract_mm_features,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    MultiModalFeatures,
    ParseContext,
    ReasoningParserKwargs,
)
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.inputs import EngineInput
from vllm.inputs.engine import MultiModalHashes, MultiModalInput
from vllm.logger import init_logger
from vllm.multimodal.parse import MultiModalDataParser
from vllm.parser.utils import count_history_tool_calls
from vllm.renderers.inputs.preprocess import (
    extract_prompt_components,
    extract_prompt_len,
)
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.utils.async_utils import make_async

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.entrypoints.mcp.tool_server import ToolServer


def _hoist_transfer_params(
    params: SamplingParams,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Move KV/EC transfer params out of `extra_args`.

    Render emits them only as top-level `GenerateRequest` fields. Generate
    copies them back into `extra_args` for the engine.
    """
    if not params.extra_args:
        return None, None
    # to_sampling_params aliases request.vllm_xargs, so copy before popping.
    extra = dict(params.extra_args)
    kv = extra.pop("kv_transfer_params", None)
    ec = extra.pop("ec_transfer_params", None)
    params.extra_args = extra or None
    return kv, ec


class ServingRender(BaseServing):
    def __init__(
        self,
        models: OpenAIServingModels | OpenAIModelRegistry,
        online_renderer: "OnlineRenderer",
        *,
        request_logger: RequestLogger | None = None,
        tool_server: "ToolServer | None" = None,
        disabled_thinking_effort: AnthropicDisabledThinkingEffortOption = "auto",
        disabled_thinking_effort_resolver: DisabledThinkingEffortResolver | None = None,
        io_processor: PoolingIOProcessor | None = None,
        io_connector: IOConnectorBase | None = None,
        io_executor: Executor | None = None,
    ) -> None:
        super().__init__(
            models=models,
            model_config=online_renderer.model_config,
            request_logger=request_logger,
        )

        self.online_renderer = online_renderer
        self.tool_server = tool_server
        # A shared resolver takes precedence over disabled_thinking_effort.
        self._disabled_thinking_effort_resolver = (
            disabled_thinking_effort_resolver
            if disabled_thinking_effort_resolver is not None
            else DisabledThinkingEffortResolver(
                online_renderer, disabled_thinking_effort
            )
        )

        self._merge_inline_system = (
            AnthropicServingMessages._should_merge_inline_system(online_renderer)
        )

        self._placeholder_metadata_parser: MultiModalDataParser | None = None
        self._placeholder_metadata_parser_failed = False

        self.default_sampling_params = (
            online_renderer.model_config.get_diff_sampling_param()
        )
        mc = online_renderer.model_config
        self.override_max_tokens = (
            self.default_sampling_params.get("max_tokens")
            if mc.generation_config not in ("auto", "vllm")
            else getattr(mc, "override_generation_config", {}).get("max_new_tokens")
        )

        self.io_processor = io_processor
        self.io_connector = io_connector

        # Dedicated executor for blocking connector calls (put_data); owned
        # and shut down by vllm.distributed.io_transfer.
        self._io_executor = io_executor

        self._preprocessing_async = make_async(
            self._preprocessing,
            executor=self.online_renderer.renderer._executor,
        )

    @torch.inference_mode()
    def _preprocessing(self, ctx: PoolingServeContext) -> None:
        """Run the IO processor plugin's pre_process and render the prompts.

        Runs on the renderer executor since both steps are CPU-bound.
        """
        assert isinstance(self.io_processor, PluginWithIOProcessorPlugins)
        requests = self.io_processor.get_request_factory_online(ctx)
        if len(requests) == 0:
            raise ValueError("You must pass at least one prompt")
        ctx.engine_inputs = [self.io_processor.render(r) for r in requests]

    async def render_chat_request(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
        engine_chat_template_kwargs: (
            Callable[[dict[str, Any]], dict[str, Any]] | None
        ) = None,
    ) -> GenerateRequest | ErrorResponse:
        """Validate the model and preprocess a chat completion request.

        This is the authoritative implementation used directly by the
        GPU-less render server and delegated to by OpenAIServingChat.

        Args:
            request: The chat completion request.
            raw_request: The incoming HTTP request, if any.
            engine_chat_template_kwargs: The calling handler's
                `_engine_chat_template_kwargs`, applied to the kwargs put in
                `parse_context` to drop API server only state.

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
        kv_transfer_params, ec_transfer_params = _hoist_transfer_params(params)

        # Resolve here what OpenAIServingChat passes to the engine:
        # `_grammar_from_parser` (set by `adjust_request`) is not serialized.
        chat_template_kwargs = self.online_renderer.effective_chat_template_kwargs(
            request
        )
        reasoning_parser_kwargs = self._reasoning_parser_kwargs(chat_template_kwargs)
        parser = None
        if reasoning_parser_kwargs is not None:
            assert self.online_renderer.parser is not None
            parser = self.online_renderer.parser(
                self.online_renderer.renderer.get_tokenizer(),
                request.tools,
                chat_template_kwargs=reasoning_parser_kwargs.chat_template_kwargs,
                model_config=self.model_config,
            )
        reasoning_ended = request.resolve_reasoning_ended(parser, token_ids)

        request_id = (
            f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
        )

        return GenerateRequest(
            request_id=request_id,
            token_ids=token_ids,
            features=self._extract_mm_features(
                engine_input, mm_payload=request.mm_payload
            ),
            sampling_params=params,
            kv_transfer_params=kv_transfer_params,
            ec_transfer_params=ec_transfer_params,
            model=request.model,
            reasoning_ended=reasoning_ended,
            reasoning_parser_kwargs=reasoning_parser_kwargs,
            stream=bool(request.stream),
            stream_options=(request.stream_options if request.stream else None),
            return_token_ids=request.return_token_ids,
            cache_salt=request.cache_salt,
            priority=request.priority,
            token_offsets=engine_input.get("prompt_token_offsets"),
            parse_context=self._build_parse_context(
                request,
                engine_chat_template_kwargs(chat_template_kwargs)
                if engine_chat_template_kwargs is not None
                else chat_template_kwargs,
            ),
        )

    def _build_parse_context(
        self, request: ChatCompletionRequest, chat_template_kwargs: dict[str, Any]
    ) -> ParseContext:
        return ParseContext(
            tools=request.tools,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
            include_reasoning=request.include_reasoning,
            # Like the derenderer, Harmony output is parsed without template
            # kwargs.
            chat_template_kwargs=(
                {} if self.online_renderer.use_harmony else chat_template_kwargs
            ),
            history_tool_call_cnt=count_history_tool_calls(request),
        )

    async def render_messages_request(
        self,
        request: AnthropicMessagesRequest,
        raw_request: Request | None = None,
    ) -> GenerateRequest | ErrorResponse:
        """Validate the model and preprocess an Anthropic Messages request.

        Converts the request to the OpenAI chat format using the same
        conversion as the /v1/messages server path, then delegates to
        render_chat_request so the rendered tokens match the server exactly.
        """
        disabled_thinking_effort = (
            await self._disabled_thinking_effort_resolver.resolve_for(request)
        )
        chat_req = AnthropicServingMessages.to_chat_completion_request(
            request,
            merge_inline_system=self._merge_inline_system,
            disabled_thinking_effort=disabled_thinking_effort,
        )
        chat_req.mm_payload = request.mm_payload
        return await self.render_chat_request(chat_req, raw_request)

    async def render_completion_request(
        self,
        request: CompletionRequest,
        raw_request: Request | None = None,
    ) -> list[GenerateRequest] | ErrorResponse:
        """Validate the model and preprocess a completion request.

        This is the authoritative implementation used directly by the
        GPU-less render server and delegated to by OpenAIServingCompletion.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        if request.use_beam_search:
            return self.create_error_response(
                "Beam search is not supported by the render endpoint"
            )

        result = await self.online_renderer.render_completion(
            request, skip_mm_cache=True
        )
        if isinstance(result, ErrorResponse):
            return result
        base_id = f"cmpl-{self._base_request_id(raw_request, request.request_id)}"
        generate_requests: list[GenerateRequest] = []
        for i, engine_input in enumerate(result):
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
            kv_transfer_params, ec_transfer_params = _hoist_transfer_params(params)

            request_id = base_id if len(result) == 1 else f"{base_id}-{i}"

            generate_requests.append(
                GenerateRequest(
                    request_id=request_id,
                    token_ids=token_ids,
                    features=self._extract_mm_features(
                        engine_input, mm_payload=request.mm_payload
                    ),
                    sampling_params=params,
                    kv_transfer_params=kv_transfer_params,
                    ec_transfer_params=ec_transfer_params,
                    model=request.model,
                    stream=bool(request.stream),
                    stream_options=(request.stream_options if request.stream else None),
                    return_token_ids=request.return_token_ids,
                    cache_salt=request.cache_salt,
                    priority=request.priority,
                    token_offsets=engine_input.get("prompt_token_offsets"),
                )
            )

        return generate_requests

    async def render_responses_request(
        self,
        request: ResponsesRequest,
    ) -> GenerateRequest | ErrorResponse:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret
        if request.previous_response_id is not None:
            return self.create_error_response(
                message=(
                    "previous_response_id is not supported by the stateless "
                    "render endpoint."
                ),
                err_type="invalid_request_error",
                param="previous_response_id",
            )

        result = await self.online_renderer.render_responses(
            request,
            previous_messages=None,
            previous_response_outputs=None,
            tool_server=self.tool_server,
            skip_mm_cache=True,
        )
        if isinstance(result, ErrorResponse):
            return result

        engine_input = result.engine_input
        prompt_components = extract_prompt_components(self.model_config, engine_input)
        token_ids = prompt_components.token_ids
        if not token_ids:
            return self.create_error_response("No token_ids rendered")

        input_length = extract_prompt_len(self.model_config, engine_input)
        max_tokens = get_max_tokens(
            self.model_config.max_model_len,
            request.max_output_tokens,
            input_length,
            self.default_sampling_params,
            self.override_max_tokens,
            truncate_prompt_tokens=(-1 if request.truncation != "disabled" else None),
        )
        params = request.to_sampling_params(max_tokens, self.default_sampling_params)
        kv_transfer_params, ec_transfer_params = _hoist_transfer_params(params)

        return GenerateRequest(
            request_id=request.request_id,
            token_ids=list(token_ids),
            features=self._extract_mm_features(
                engine_input, mm_payload=request.mm_payload
            ),
            sampling_params=params,
            model=request.model,
            reasoning_parser_kwargs=self._reasoning_parser_kwargs(
                self.online_renderer.effective_chat_template_kwargs(request)
            ),
            stream=bool(request.stream),
            cache_salt=request.cache_salt,
            priority=request.priority,
            kv_transfer_params=kv_transfer_params,
            ec_transfer_params=ec_transfer_params,
            token_offsets=engine_input.get("prompt_token_offsets"),
        )

    def _reasoning_parser_kwargs(
        self, chat_template_kwargs: dict[str, Any]
    ) -> ReasoningParserKwargs | None:
        parser_cls = self.online_renderer.parser
        if parser_cls is None or parser_cls.reasoning_parser_cls is None:
            return None
        return ReasoningParserKwargs(chat_template_kwargs=chat_template_kwargs)

    def _placeholder_metadata_fields(self, modality: str) -> set[str]:
        """Fields the EC consumer still requires after embeddings transfer."""
        if self._placeholder_metadata_parser_failed:
            return set()
        if self._placeholder_metadata_parser is None:
            try:
                from vllm.multimodal import MULTIMODAL_REGISTRY

                self._placeholder_metadata_parser = (
                    MULTIMODAL_REGISTRY.create_processor(
                        self.model_config
                    ).info.data_parser
                )
            except Exception:
                logger.debug(
                    "Could not load placeholder metadata fields; "
                    "mm_metadata will use keep_on_cpu fields only."
                )
                self._placeholder_metadata_parser_failed = True
                return set()
        return set(
            self._placeholder_metadata_parser.placeholder_metadata_fields(modality)
        )

    def _extract_mm_features(
        self,
        engine_input: EngineInput,
        *,
        mm_payload: MMPayload = "tensors",
    ) -> MultiModalFeatures | None:
        return extract_mm_features(
            engine_input,
            metadata_fields_for=self._placeholder_metadata_fields,
            mm_payload=mm_payload,
        )

    async def preprocess_pooling_request(
        self,
        request: IOProcessorRequest,
        raw_request: Request,
    ) -> list[GeneratePoolingRequest] | ErrorResponse:
        """Validate the model and preprocess a pooling request.

        This is the authoritative implementation used directly by the
        GPU-less render server.
        """
        model_name = request.model
        if not model_name:
            return self.create_error_response("No model was specified")

        base_request_id = random_uuid()
        if raw_request is not None and (
            (req_id := raw_request.headers.get("X-Request-Id")) is not None
        ):
            base_request_id = req_id

        request_id = f"pooling-{base_request_id}"
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        if not (
            self.io_processor
            and isinstance(self.io_processor, PluginWithIOProcessorPlugins)
        ):
            return self.create_error_response(
                "No IO processor was found "
                "--io-processor-plugin is not set. "
                "Refused request without IO processor."
            )

        pooling_params = self.io_processor.create_pooling_params(request)
        prompt_extras = {
            k: v
            for k in ("mm_processor_kwargs", "cache_salt")
            if (v := getattr(request, k, None)) is not None
        }
        ctx = PoolingServeContext(
            request=request,
            raw_request=raw_request,
            model_name=model_name,
            pooling_params=pooling_params,
            request_id=request_id,
            lora_request=None,
            priorities=request.priority,
            prompt_extras=prompt_extras,
        )

        await self._preprocessing_async(ctx)

        if ctx.engine_inputs is None:
            raise ValueError("Engine prompts not available")

        ctx.pooling_params.verify(self.model_config)

        if self.io_connector:
            io_connector = self.io_connector
            io_processor = self.io_processor.io_processor

            # Validate all inputs and collect put arguments synchronously
            # before firing any I/O, so the gather below starts all
            # puts in a single event-loop tick with no staggering.
            #
            # Each entry: (prompt_request_id, metadata, object_id, item)
            put_entries: list[tuple[str, dict, str, Any]] = []
            for i, pooling_input in enumerate(ctx.engine_inputs):
                engine_input = pooling_input["prompts"]
                prompt_request_id = (
                    f"{ctx.request_id}-{i}"
                    if ctx.prompt_request_ids is None
                    else ctx.prompt_request_ids[i]
                )
                if engine_input.get("type") != "multimodal":
                    return self.online_renderer.create_error_response(
                        f"Couldn't extract data for {prompt_request_id}."
                    )
                mm_engine_input = cast(MultiModalInput, engine_input)
                raw_mm_kwargs = mm_engine_input.get("mm_kwargs")
                mm_hashes: MultiModalHashes = mm_engine_input["mm_hashes"]
                if not raw_mm_kwargs:
                    return self.online_renderer.create_error_response(
                        f"Couldn't extract data for {prompt_request_id}."
                    )

                if sum(len(items) for items in raw_mm_kwargs.values()) != 1:
                    return self.online_renderer.create_error_response(
                        "IO disaggregation supports exactly one multimodal item "
                        f"per prompt ({prompt_request_id})."
                    )

                metadata = io_processor.get_request_metadata(prompt_request_id) or {}

                for modality, items in raw_mm_kwargs.items():
                    for j, item in enumerate(items):
                        if item is None:
                            return self.online_renderer.create_error_response(
                                "Couldn't find data."
                            )
                        put_entries.append(
                            (
                                prompt_request_id,
                                metadata,
                                mm_hashes[modality][j],
                                item,
                            )
                        )

            # Fire all puts in one gather.
            async def _put_metadata(
                prompt_request_id: str, metadata: dict
            ) -> tuple[str, MetadataRef]:
                ref = await io_connector.put_metadata_async(
                    prompt_request_id,
                    metadata,
                    executor=self._io_executor,
                )
                return prompt_request_id, ref

            async def _put_data(
                prompt_request_id: str, object_id: str, item: Any
            ) -> tuple[str, DataRef]:
                ref = await io_connector.put_data_async(
                    object_id,
                    item,
                    executor=self._io_executor,
                )
                return prompt_request_id, ref

            # Split put_entries into separate metadata and data coros, keeping
            # one put_metadata per unique prompt_request_id.
            seen_metadata: set[str] = set()
            metadata_coros = []
            data_coros: list[Any] = []
            for prompt_request_id, metadata, object_id, item in put_entries:
                if prompt_request_id not in seen_metadata:
                    seen_metadata.add(prompt_request_id)
                    metadata_coros.append(_put_metadata(prompt_request_id, metadata))
                data_coros.append(_put_data(prompt_request_id, object_id, item))
            n_metadata = len(metadata_coros)
            all_results = await asyncio.gather(*metadata_coros, *data_coros)

            # Index metadata_refs by prompt_request_id.
            metadata_refs: dict[str, MetadataRef] = {}
            for prompt_request_id, ref in all_results[:n_metadata]:
                metadata_refs[prompt_request_id] = ref

            # Build generate_pooling_requests in original order.
            generate_pooling_requests: list[GeneratePoolingRequest] = []
            for prompt_request_id, data_ref in all_results[n_metadata:]:
                generate_pooling_requests.append(
                    GeneratePoolingRequest(
                        request_id=prompt_request_id,
                        data_ref=data_ref,
                        metadata_ref=metadata_refs[prompt_request_id],
                        model=request.model,
                        cache_salt=request.cache_salt,
                        priority=request.priority,
                    )
                )
        else:
            generate_pooling_requests = []
            for i, pooling_input in enumerate(ctx.engine_inputs):
                prompt_request_id = (
                    f"{ctx.request_id}-{i}"
                    if ctx.prompt_request_ids is None
                    else ctx.prompt_request_ids[i]
                )
                features = self._extract_mm_features(pooling_input["prompts"])

                if features is None or features.kwargs_data is None:
                    return self.online_renderer.create_error_response(
                        f"Couldn't extract data for {prompt_request_id}."
                    )

                if sum(len(v) for v in features.kwargs_data.values()) != 1:
                    return self.online_renderer.create_error_response(
                        "IO disaggregation supports exactly one multimodal item "
                        f"per prompt ({prompt_request_id})."
                    )

                metadata = (
                    self.io_processor.io_processor.get_request_metadata(
                        prompt_request_id
                    )
                    or {}
                )
                for modality, encoded_items in features.kwargs_data.items():
                    for j, data in enumerate(encoded_items):
                        if data is None:
                            return self.online_renderer.create_error_response(
                                "Couldn't find data."
                            )

                        object_id = features.mm_hashes[modality][j]
                        data_ref = DataRef(
                            object_id=object_id, location=DataLocation(locator="")
                        )
                        data_obj = DataObject[str](data=data)

                        generate_pooling_requests.append(
                            GeneratePoolingRequest(
                                request_id=prompt_request_id,
                                data_ref=data_ref,
                                data=data_obj,
                                metadata=metadata,
                                model=request.model,
                                cache_salt=request.cache_salt,
                                priority=request.priority,
                            )
                        )

        return generate_pooling_requests
