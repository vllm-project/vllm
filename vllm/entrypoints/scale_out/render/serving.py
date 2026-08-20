# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.serving import (
    OpenAIModelRegistry,
    OpenAIServingModels,
)
from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import encode_mm_kwargs_item
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    MultiModalFeatures,
    PlaceholderRangeInfo,
)
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.inputs import (
    EngineInput,
    MultiModalHashes,
    MultiModalInput,
    MultiModalPlaceholders,
)
from vllm.logger import init_logger
from vllm.renderers.inputs.preprocess import (
    extract_prompt_components,
    extract_prompt_len,
)
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.utils import random_uuid

logger = init_logger(__name__)


class ServingRender(BaseServing):
    def __init__(
        self,
        models: OpenAIServingModels | OpenAIModelRegistry,
        online_renderer: "OnlineRenderer",
        *,
        request_logger: RequestLogger | None = None,
    ) -> None:
        super().__init__(
            models=models,
            model_config=online_renderer.model_config,
            request_logger=request_logger,
        )

        self.online_renderer = online_renderer

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

        assistant_tokens_mask: list[int] | None = engine_input.get(  # type: ignore[assignment]
            "assistant_tokens_mask"
        )
        if assistant_tokens_mask is not None and len(assistant_tokens_mask) != len(
            token_ids
        ):
            logger.warning(
                "assistant_tokens_mask length (%d) != token_ids length (%d); "
                "this can happen with multimodal inputs where "
                "placeholder expansion changes the token count. "
                "The mask may be positionally misaligned.",
                len(assistant_tokens_mask),
                len(token_ids),
            )
            if len(assistant_tokens_mask) < len(token_ids):
                assistant_tokens_mask.extend(
                    [0] * (len(token_ids) - len(assistant_tokens_mask))
                )
            else:
                assistant_tokens_mask = assistant_tokens_mask[: len(token_ids)]

        request_id = f"chatcmpl-{random_uuid()}"

        return GenerateRequest(
            request_id=request_id,
            token_ids=token_ids,
            assistant_tokens_mask=assistant_tokens_mask,
            features=self._extract_mm_features(engine_input),
            sampling_params=params,
            model=request.model,
            stream=bool(request.stream),
            stream_options=(request.stream_options if request.stream else None),
            cache_salt=request.cache_salt,
            priority=request.priority,
            token_offsets=engine_input.get("prompt_token_offsets"),
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
                    token_offsets=engine_input.get("prompt_token_offsets"),
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
