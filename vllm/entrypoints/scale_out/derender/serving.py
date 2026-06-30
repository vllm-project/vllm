# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import cast

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionResponse
from vllm.entrypoints.openai.completion.protocol import CompletionResponse
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    UsageInfo,
)
from vllm.entrypoints.openai.models.serving import (
    OpenAIModelRegistry,
    OpenAIServingModels,
)
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.inputs import (
    EngineInput,
    MultiModalHashes,
    MultiModalInput,
    MultiModalPlaceholders,
)
from vllm.logger import init_logger
from vllm.renderers.online_derenderer import OnlineDerenderer

from ..token_in_token_out.mm_serde import encode_mm_kwargs_item
from ..token_in_token_out.protocol import (
    DerenderChatRequest,
    DerenderCompletionRequest,
    MultiModalFeatures,
    PlaceholderRangeInfo,
)

logger = init_logger(__name__)


class ServingDerender(BaseServing):
    def __init__(
        self,
        models: OpenAIServingModels | OpenAIModelRegistry,
        online_derenderer: "OnlineDerenderer",
        *,
        request_logger: RequestLogger | None = None,
    ) -> None:
        super().__init__(
            models=models,
            model_config=models.model_config,
            request_logger=request_logger,
        )

        self.online_derenderer = online_derenderer

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
