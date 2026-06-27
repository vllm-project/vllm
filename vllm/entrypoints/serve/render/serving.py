# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import cast

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    UsageInfo,
)
from vllm.entrypoints.openai.models.serving import (
    OpenAIModelRegistry,
    OpenAIServingModels,
)
from vllm.entrypoints.serve.disagg.mm_serde import encode_mm_kwargs_item
from vllm.entrypoints.serve.disagg.protocol import (
    DerenderChatRequest,
    DerenderCompletionRequest,
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
from vllm.renderers.online_derenderer import OnlineDerenderer
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.utils import random_uuid

logger = init_logger(__name__)


class ServingRender(BaseServing):
    def __init__(
        self,
        models: OpenAIServingModels | OpenAIModelRegistry,
        online_renderer: "OnlineRenderer",
        online_derenderer: "OnlineDerenderer",
        *,
        request_logger: RequestLogger | None = None,
    ) -> None:
        super().__init__(
            models=models,
            model_config=online_renderer.model_config,
            request_logger=request_logger,
        )

        self.online_renderer = online_renderer
        self.online_derenderer = online_derenderer

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

        try:
            choices = await self.online_derenderer.derender_chat(
                request.generate_response, request.chat_request
            )
        except ValueError as exc:
            return self.create_error_response(str(exc))

        prompt_tokens = (
            request.prompt_tokens if request.prompt_tokens is not None else 0
        )
        gen = request.generate_response
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

        (
            choices,
            total_prompt_tokens,
            total_completion_tokens,
        ) = await self.online_derenderer.derender_completion(
            request.generate_responses, request.prompt_tokens
        )

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
