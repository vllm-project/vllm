# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import asyncio
import io
import time
from collections.abc import AsyncGenerator
from collections.abc import Sequence as GenericSequence

import numpy as np
import pybase64 as base64
from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    GenerationError,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import OpenAIServing, clamp_prompt_logprobs
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.disagg.mm_serde import decode_mm_kwargs_item
from vllm.entrypoints.serve.disagg.protocol import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChoice,
    GenerateResponseStreamChoice,
    GenerateStreamResponse,
)
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.entrypoints.utils import get_max_tokens, should_include_usage
from vllm.inputs import EngineInput, mm_input
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.multimodal.inputs import (
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.utils.collection_utils import as_list

logger = init_logger(__name__)


class ServingTokens(OpenAIServing):
    """Provides Tokens IN <> Tokens OUT functionality to vLLM API."""

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        openai_serving_render: OpenAIServingRender,
        *,
        request_logger: RequestLogger | None,
        force_no_detokenize: bool = False,
        return_tokens_as_token_ids: bool = False,
        enable_prompt_tokens_details: bool = False,
        enable_log_outputs: bool = False,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )
        self.openai_serving_render = openai_serving_render
        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_log_outputs = enable_log_outputs
        self.force_no_detokenize = force_no_detokenize
        if force_no_detokenize:
            logger.info(
                "Tokens-only mode is enabled, skipping detokenization "
                "step for incoming requests."
            )

        # Mirrors ``OpenAIServingChat`` so we can apply server-side
        # ``max_tokens`` defaulting when the client omits it. Without this,
        # ``SamplingParams.max_tokens`` falls back to its dataclass default
        # of 16 and silently truncates every generation.
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        mc = self.model_config
        self.override_max_tokens = (
            self.default_sampling_params.get("max_tokens")
            if mc.generation_config not in ("auto", "vllm")
            else getattr(mc, "override_generation_config", {}).get("max_new_tokens")
        )

    async def serve_tokens(
        self,
        request: GenerateRequest,
        raw_request: Request | None = None,
    ) -> GenerateResponse | ErrorResponse | AsyncGenerator[str, None]:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        lora_request = None
        lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)

        model_name = self.models.model_name(lora_request)

        request_id = (
            f"generate-tokens-{self._base_request_id(raw_request, request.request_id)}"
        )

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        engine_input: EngineInput
        if features := request.features:
            # Convert PlaceholderRangeInfo → PlaceholderRange per modality.
            mm_placeholders: dict[str, list[PlaceholderRange]] = {
                modality: [
                    PlaceholderRange(offset=p.offset, length=p.length) for p in ranges
                ]
                for modality, ranges in features.mm_placeholders.items()
            }

            # Deserialize tensor data when present; None → cache hit.
            mm_kwargs: dict[str, list[MultiModalKwargsItem | None]] = {}
            if features.kwargs_data is not None:
                for modality, items in features.kwargs_data.items():
                    mm_kwargs[modality] = [
                        decode_mm_kwargs_item(item) if item is not None else None
                        for item in items
                    ]
            else:
                for modality, hashes in features.mm_hashes.items():
                    mm_kwargs[modality] = [None] * len(hashes)

            engine_input = mm_input(
                prompt_token_ids=request.token_ids,
                mm_kwargs=MultiModalKwargsItems(mm_kwargs),
                mm_hashes=features.mm_hashes,
                mm_placeholders=mm_placeholders,
                cache_salt=request.cache_salt,
            )
        else:
            (engine_input,) = await self.openai_serving_render.preprocess_completion(
                request,
                prompt_input=request.token_ids,
                prompt_embeds=None,
                skip_mm_cache=True,
            )

        # Schedule the request and get the result generator.
        result_generator: AsyncGenerator[RequestOutput, None] | None = None
        sampling_params = request.sampling_params

        # Apply server-side ``max_tokens`` defaulting when the client did
        # not set it, matching the OpenAI-compat endpoints. ``SamplingParams``
        # defaults ``max_tokens`` to 16, which would otherwise silently cap
        # every generation that omits the field.
        if not request.is_sampling_param_provided("max_tokens"):
            sampling_params.max_tokens = get_max_tokens(
                max_model_len=self.model_config.max_model_len,
                max_tokens=None,
                input_length=self._extract_prompt_len(engine_input),
                default_sampling_params=self.default_sampling_params,
                override_max_tokens=self.override_max_tokens,
            )

        if self.force_no_detokenize:
            sampling_params.detokenize = False
        if request.stream:
            sampling_params.output_kind = RequestOutputKind.DELTA

        self._log_inputs(
            request_id,
            engine_input,
            params=sampling_params,
            lora_request=lora_request,
        )

        trace_headers = (
            None
            if raw_request is None
            else await self._get_trace_headers(raw_request.headers)
        )

        result_generator = self.engine_client.generate(
            engine_input,
            sampling_params,
            request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=request.priority,
        )

        assert result_generator is not None

        if request.stream:
            return self.serve_tokens_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                request_metadata,
            )

        return await self.serve_tokens_full_generator(
            request, result_generator, request_id, model_name, request_metadata
        )

    async def serve_tokens_full_generator(
        self,
        request: GenerateRequest,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        model_name: str,
        request_metadata: RequestResponseMetadata,
    ) -> ErrorResponse | GenerateResponse:
        created_time = int(time.time())
        final_res: RequestOutput | None = None
        sampling_params: SamplingParams = request.sampling_params

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        assert final_res is not None

        choices: list[GenerateResponseChoice] = []
        num_generated_tokens = 0
        for output in final_res.outputs:
            token_ids = output.token_ids
            out_logprobs = output.logprobs

            # This is top_logprobs in completions API
            if sampling_params.logprobs is not None:
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_tokens_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    num_output_top_logprobs=sampling_params.logprobs,
                )
            else:
                logprobs = None

            # Encode routed_experts for transport. JSON can't carry raw
            # bytes, so we write the ndarray as a ``.npy`` byte stream
            # and base64-encode it. ``pybase64`` is ~3x faster than the
            # stdlib ``base64`` on large payloads thanks to SIMD.
            # This is the only base64 hop in the pipeline -- the
            # engine<->API-server link is binary msgpack + zmq.
            routed_experts_b64 = None
            if output.routed_experts is not None:
                buf = io.BytesIO()
                np.save(buf, output.routed_experts)
                routed_experts_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            choice_data = GenerateResponseChoice(
                index=output.index,
                logprobs=logprobs,
                finish_reason=output.finish_reason if output.finish_reason else "stop",
                token_ids=as_list(output.token_ids),
                routed_experts=routed_experts_b64,
            )

            choices.append(choice_data)
            num_generated_tokens += len(output.token_ids)

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:
            # This info is not available at the /coordinator level
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=final_res.num_cached_tokens
            )

        request_metadata.final_usage_info = usage

        response = GenerateResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
            kv_transfer_params=final_res.kv_transfer_params,
        )

        # Log complete response if output logging is enabled
        if self.enable_log_outputs and self.request_logger:
            for choice in choices:
                # Get the corresponding output token IDs
                output_token_ids = None
                if choice.index < len(final_res.outputs):
                    output_token_ids = final_res.outputs[choice.index].token_ids

                if output_token_ids:
                    # Log token_ids only.
                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs="",
                        output_token_ids=output_token_ids,
                        finish_reason=choice.finish_reason,
                        is_streaming=False,
                        delta=False,
                    )

        return response

    async def serve_tokens_stream_generator(
        self,
        request: GenerateRequest,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        model_name: str,
        request_metadata: RequestResponseMetadata,
    ) -> AsyncGenerator[str, None]:
        num_prompt_tokens = 0
        num_generated_tokens: list[int] = []
        first_iteration = True
        num_cached_tokens = None
        sampling_params: SamplingParams = request.sampling_params

        include_usage, include_continuous_usage = should_include_usage(
            request.stream_options, False
        )

        try:
            async for res in result_generator:
                if first_iteration:
                    if res.prompt_token_ids is not None:
                        num_prompt_tokens = len(res.prompt_token_ids)
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)
                    num_cached_tokens = res.num_cached_tokens
                    num_generated_tokens = [0] * len(res.outputs)
                    first_iteration = False

                for output in res.outputs:
                    i = output.index
                    delta_token_ids = output.token_ids
                    num_generated_tokens[i] += len(delta_token_ids)

                    finish_reason = output.finish_reason
                    self._raise_if_error(finish_reason, request_id)

                    if not delta_token_ids:
                        continue

                    if sampling_params.logprobs is not None:
                        out_logprobs = output.logprobs
                        assert out_logprobs is not None, "Did not output logprobs"
                        logprobs = self._create_tokens_logprobs(
                            token_ids=delta_token_ids,
                            top_logprobs=out_logprobs,
                            num_output_top_logprobs=sampling_params.logprobs,
                        )
                    else:
                        logprobs = None

                    chunk = GenerateStreamResponse(
                        request_id=request_id,
                        choices=[
                            GenerateResponseStreamChoice(
                                index=i,
                                logprobs=logprobs,
                                finish_reason=finish_reason,
                                token_ids=as_list(delta_token_ids),
                            )
                        ],
                    )
                    if include_continuous_usage:
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=num_generated_tokens[i],
                            total_tokens=(num_prompt_tokens + num_generated_tokens[i]),
                        )

                    yield f"data: {chunk.model_dump_json()}\n\n"

            total_completion_tokens = sum(num_generated_tokens)
            final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=num_prompt_tokens + total_completion_tokens,
            )

            if self.enable_prompt_tokens_details and num_cached_tokens:
                final_usage_info.prompt_tokens_details = PromptTokenUsageInfo(
                    cached_tokens=num_cached_tokens
                )

            if include_usage:
                final_chunk = GenerateStreamResponse(
                    request_id=request_id,
                    choices=[],
                    usage=final_usage_info,
                )
                yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"

            request_metadata.final_usage_info = final_usage_info

        except GenerationError as e:
            yield (
                f"data: {self._convert_generation_error_to_streaming_response(e)}\n\n"
            )
        except Exception as e:
            logger.exception("Error in token generation stream.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    def _create_tokens_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[dict[int, Logprob] | None],
        num_output_top_logprobs: int | None = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""
        logprobs_content: list[ChatCompletionLogProbsContent] = []

        for i, token_id in enumerate(token_ids):
            token = f"token_id:{token_id}"
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None or step_top_logprobs.get(token_id) is None:
                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                    )
                )
            else:
                step_token = step_top_logprobs[token_id]

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                        logprob=max(step_token.logprob, -9999.0),
                        top_logprobs=[
                            ChatCompletionLogProb(
                                token=token,
                                logprob=max(p[1].logprob, -9999.0),
                            )
                            for i, p in enumerate(step_top_logprobs.items())
                            if num_output_top_logprobs is not None
                            and i < max(num_output_top_logprobs, 1)
                        ],
                    )
                )

        return ChatCompletionLogProbs(content=logprobs_content)
