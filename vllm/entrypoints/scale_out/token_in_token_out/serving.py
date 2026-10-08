# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import asyncio
import concurrent.futures
import contextvars
import functools
import math
import time
from collections.abc import AsyncGenerator, Callable
from collections.abc import Sequence as GenericSequence
from typing import Any, TypeVar

import msgspec
from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import AsyncMultiModalItemTracker
from vllm.entrypoints.generate.base.protocol import (
    PerRequestMetrics,
    RequestResponseMetadata,
)
from vllm.entrypoints.generate.base.serving import (
    GenerateBaseServing,
    build_spec_decoding_metrics,
    clamp_prompt_logprobs,
    format_token_id_placeholder,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.engine.protocol import (
    ErrorResponse,
    PromptTokenUsageInfo,
    UsageInfo,
)
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens, should_include_usage
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.exceptions import GenerationError
from vllm.inputs import EngineInput, TokensPrompt, mm_input
from vllm.logger import init_logger
from vllm.logprobs import FlatLogprobs, Logprob
from vllm.multimodal.inputs import (
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.outputs import RequestOutput
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.utils.collection_utils import as_list
from vllm.utils.serial_utils import numpy2base64

from .logprobs_render import (
    append_sampled_logprobs,
    render_json_with_fragments,
    render_tokens_logprobs,
    render_top_k_logprobs,
)
from .mm_features import (
    mm_kwargs_from_features,
    placeholder_ranges_from_engine_input,
)
from .protocol import (
    GenerateLogProb,
    GenerateLogProbs,
    GenerateLogProbsContent,
    GenerateRequest,
    GenerateResponse,
    GenerateTextChoice,
    GenerateTextResponse,
    GenerateTextStreamChoice,
    GenerateTextStreamResponse,
    GenerateTokensChoice,
    GenerateTokensResponse,
    GenerateTokensStreamChoice,
    GenerateTokensStreamResponse,
    RenderedGenerateResponse,
)

logger = init_logger(__name__)

T = TypeVar("T")

# Logprob entries (positions x slots) from which a response is built in a
# worker thread rather than on the event loop (about 0.2 us per entry).
OFFLOAD_MIN_LOGPROB_ENTRIES = 1 << 15
# Two threads: a build is mostly GIL-bound and a second thread overlaps its
# GIL-free part; more threads only add memory.
_RESPONSE_BUILDER = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="generate-response"
)


async def build_response_off_loop(entries: int, build: Callable[[], T]) -> T:
    """Run ``build`` (CPU only, no shared-state side effects) inline for
    fewer than ``OFFLOAD_MIN_LOGPROB_ENTRIES`` logprob entries, else in a
    worker thread, so the event loop keeps serving other requests."""
    if entries < OFFLOAD_MIN_LOGPROB_ENTRIES:
        return build()
    return await asyncio.get_running_loop().run_in_executor(
        _RESPONSE_BUILDER, functools.partial(contextvars.copy_context().run, build)
    )


def _flat_logprob_entries(final_res: RequestOutput) -> int:
    """Logprob entries (positions x candidates) kept as FlatLogprobs."""
    return sum(
        output.logprobs.num_entries
        for output in final_res.outputs
        if isinstance(output.logprobs, FlatLogprobs)
    )


def _clamp_logprob(logprob: float) -> float:
    """The OpenAI shapes' floor for a logprob: ``-inf`` and NaN become ``-9999.0``.

    ``max(nan, -9999.0)`` is NaN in Python, so NaN needs its own check; NaN
    would otherwise fail ``JSONResponse`` or reach the client as ``null``.
    """
    return -9999.0 if math.isnan(logprob) else max(logprob, -9999.0)


def _logprob_token(
    token_id: int, logprob: Logprob | None, tokenizer: TokenizerLike | None
) -> tuple[str, list[int] | None]:
    """Token string and UTF-8 bytes for one logprob entry.

    Without a tokenizer the token is a ``token_id:N`` placeholder with no bytes.
    """
    if tokenizer is None:
        return format_token_id_placeholder(token_id), None
    token = (
        tokenizer.decode([token_id])
        if logprob is None
        else GenerateBaseServing._get_decoded_token(logprob, token_id, tokenizer)
    )
    return token, list(token.encode("utf-8", errors="replace"))


def _top_logprob(
    token_id: int, logprob: Logprob, tokenizer: TokenizerLike | None
) -> ChatCompletionLogProb:
    token, token_bytes = _logprob_token(token_id, logprob, tokenizer)
    return ChatCompletionLogProb(
        token=token, logprob=_clamp_logprob(logprob.logprob), bytes=token_bytes
    )


class ServingTokens(GenerateBaseServing):
    """Provides Tokens IN <> Tokens OUT functionality to vLLM API."""

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        online_renderer: OnlineRenderer,
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
        self.online_renderer = online_renderer
        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_log_outputs = enable_log_outputs
        self.force_no_detokenize = force_no_detokenize
        self.has_tokenizer = (
            not force_no_detokenize and self.renderer.tokenizer is not None
        )
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

    def _validate_mm_cache_handles(
        self,
        mm_kwargs: dict[str, list[MultiModalKwargsItem | None]],
        mm_hashes: dict[str, list[str]],
    ) -> ErrorResponse | None:
        cache = self.online_renderer.renderer.mm_processor_cache
        if cache is None:
            return None
        try:
            for modality, items in mm_kwargs.items():
                for mm_hash, item in zip(mm_hashes[modality], items, strict=True):
                    if item is not None:
                        cache.validate_input_item(item, mm_hash)
        except ValueError as error:
            return self.create_error_response(error)
        return None

    async def serve_tokens(
        self,
        request: GenerateRequest,
        raw_request: Request | None = None,
    ) -> (
        GenerateResponse
        | RenderedGenerateResponse
        | ErrorResponse
        | AsyncGenerator[str, None]
    ):
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        self._preflight()

        lora_request = None
        lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)

        model_name = self.models.model_name(lora_request)

        request_id = (
            f"generate-tokens-{self._base_request_id(raw_request, request.request_id)}"
        )

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        sampling_params = request.sampling_params
        if request.return_token_logprobs:
            if request.output_mode != "tokens":
                # ``sampled`` lives on the tokens-mode ``GenerateLogProbs``.
                return self.create_error_response(
                    "return_token_logprobs requires output_mode='tokens'"
                )
            if request.stream:
                return self.create_error_response(
                    "return_token_logprobs is not supported with stream=True"
                )
            if self.model_config.logprobs_mode not in (
                "raw_logprobs",
                "processed_logprobs",
            ):
                # The field is named for log-probabilities; do not hand back
                # logits under that name.
                return self.create_error_response(
                    "return_token_logprobs requires --logprobs-mode raw_logprobs "
                    "or processed_logprobs (server runs "
                    f"{self.model_config.logprobs_mode})"
                )
            if sampling_params.logprobs is None:
                sampling_params.logprobs = 0
            if sampling_params.logprobs == 0:
                # Only the sampled token's logprob is needed: transport one
                # float per token from the scheduler and skip per-token
                # Logprob entries and detokenization entirely.
                sampling_params._sampled_logprobs_only = True
            else:
                # Top logprobs were also requested: keep the object path and
                # read the sampled column from the flat representation.
                sampling_params.flat_logprobs = True
        max_num_seqs = self.engine_client.vllm_config.scheduler_config.max_num_seqs
        if sampling_params.n > max_num_seqs:
            return self.create_error_response(
                f"sampling_params.n must be at most the server's max_num_seqs "
                f"({max_num_seqs}), got {sampling_params.n}."
            )
        # The stream schema has no field for the scores.
        if request.stream and sampling_params.prompt_logprob_token_ids is not None:
            return self.create_error_response(
                "prompt_logprob_token_ids are not available when stream=true."
            )
        if self.force_no_detokenize and sampling_params.stop:
            # SamplingParams rejects stop with detokenize=False at request
            # validation, but this server forces detokenize=False afterwards,
            # so the combination must be rejected here or stop strings are
            # silently never applied.
            return self.create_error_response(
                "stop strings are not supported on a --tokens-only server "
                "because detokenization is disabled. Check stop strings on "
                "the coordinator, or use stop_token_ids."
            )
        if request.output_mode != "tokens" and not self.has_tokenizer:
            # The no-op detokenizer returns "", so without this check the
            # request would succeed with empty text.
            return self.create_error_response(
                f"output_mode={request.output_mode!r} requires a tokenizer, but "
                "this server does not load one (--tokens-only or "
                "--skip-tokenizer-init). Send the request to a server that "
                "loads a tokenizer, or use output_mode='tokens'."
            )
        try:
            msgspec.msgpack.encode(
                (
                    sampling_params,
                    request.kv_transfer_params,
                    request.ec_transfer_params,
                )
            )
        except (OverflowError, TypeError, ValueError) as e:
            return self.create_error_response(e)

        engine_input: EngineInput
        if request.content_parts:
            tracker = AsyncMultiModalItemTracker(self.model_config)
            mm_parser = tracker.create_parser()
            for part in request.content_parts:
                ptype = part.get("type", "")
                url = part.get("url")
                uuid = part.get("uuid")
                if ptype == "image_url":
                    mm_parser.parse_image(url, uuid)
                elif ptype == "audio_url":
                    mm_parser.parse_audio(url, uuid)
                elif ptype == "video_url":
                    mm_parser.parse_video(url, uuid)
            mm_data, mm_uuids = await tracker.resolve_items()
            prompt = TokensPrompt(prompt_token_ids=request.token_ids)
            if request.cache_salt is not None:
                prompt["cache_salt"] = request.cache_salt
            if mm_data:
                prompt["multi_modal_data"] = mm_data
            if mm_uuids:
                prompt["multi_modal_uuids"] = mm_uuids
            (engine_input,) = await self.online_renderer.renderer.render_cmpl_async(
                [prompt]
            )
        elif features := request.features:
            # Convert PlaceholderRangeInfo → PlaceholderRange per modality.
            mm_placeholders: dict[str, list[PlaceholderRange]] = {
                modality: [
                    PlaceholderRange(offset=p.offset, length=p.length) for p in ranges
                ]
                for modality, ranges in features.mm_placeholders.items()
            }

            # Deserialize full tensor data and optional metadata-only data.
            # Metadata-only items are valid when ec_transfer_params is set.
            mm_kwargs = mm_kwargs_from_features(features)
            if error := self._validate_mm_cache_handles(mm_kwargs, features.mm_hashes):
                return error

            engine_input = mm_input(
                prompt_token_ids=request.token_ids,
                mm_kwargs=MultiModalKwargsItems(mm_kwargs),
                mm_hashes=features.mm_hashes,
                mm_placeholders=mm_placeholders,
                cache_salt=request.cache_salt,
            )
        else:
            (engine_input,) = await self.online_renderer.preprocess_completion(
                request,
                prompt_input=request.token_ids,
                prompt_embeds=None,
                skip_mm_cache=True,
            )

        # Offsets are relative to the decoder prompt, so they are not
        # meaningful for encoder-decoder models.
        request._response_mm_placeholders = (
            placeholder_ranges_from_engine_input(engine_input)
            if request.return_token_ids and not self.model_config.is_encoder_decoder
            else None
        )

        # Schedule the request and get the result generator.
        result_generator: AsyncGenerator[RequestOutput, None] | None = None

        # Pass disaggregated-serving parameters through to the engine.
        if request.kv_transfer_params is not None:
            extra = sampling_params.extra_args or {}
            extra["kv_transfer_params"] = request.kv_transfer_params
            sampling_params.extra_args = extra
        if request.ec_transfer_params is not None:
            extra = sampling_params.extra_args or {}
            extra["ec_transfer_params"] = request.ec_transfer_params
            sampling_params.extra_args = extra

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
        sampling_params.output_kind = (
            RequestOutputKind.DELTA if request.stream else RequestOutputKind.FINAL_ONLY
        )
        if (
            not request.stream
            and request.output_mode == "tokens"
            and sampling_params.logprobs is not None
            and sampling_params.logprobs >= 0
            and sampling_params.prompt_logprobs is None
        ):
            # Keep the top-k rows as numpy columns without decoded tokens: the
            # response is rendered from them without a Python object per
            # entry. Prompt logprobs stay lists, as the response returns them.
            sampling_params.flat_logprobs = True
            sampling_params._detokenize_logprobs = False

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

        # Extract data_parallel_rank from header (router can inject it)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)
        session_id = self._get_session_id_from_headers(raw_request)

        result_generator = self.engine_client.generate(
            engine_input,
            sampling_params,
            request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=request.priority,
            data_parallel_rank=data_parallel_rank,
            session_id=session_id,
            reasoning_ended=request.reasoning_ended,
            reasoning_parser_kwargs=(
                request.reasoning_parser_kwargs.model_dump()
                if request.reasoning_parser_kwargs is not None
                else None
            ),
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
    ) -> ErrorResponse | GenerateResponse | RenderedGenerateResponse:
        created_time = int(time.time())
        final_res: RequestOutput | None = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        assert final_res is not None

        response, usage, choice_meta = await build_response_off_loop(
            _flat_logprob_entries(final_res),
            functools.partial(
                self._build_full_response,
                request,
                final_res,
                request_id,
                model_name,
                created_time,
            ),
        )
        request_metadata.final_usage_info = usage

        # Log complete response if output logging is enabled
        if self.enable_log_outputs and self.request_logger:
            for index, finish_reason in choice_meta:
                # Get the corresponding output token IDs
                output_token_ids = None
                if index < len(final_res.outputs):
                    output_token_ids = final_res.outputs[index].token_ids

                if output_token_ids:
                    # Log token_ids only.
                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs="",
                        output_token_ids=output_token_ids,
                        finish_reason=finish_reason,
                        is_streaming=False,
                        delta=False,
                    )

        return response

    def _build_full_response(
        self,
        request: GenerateRequest,
        final_res: RequestOutput,
        request_id: str,
        model_name: str,
        created_time: int,
    ) -> tuple[
        GenerateResponse | RenderedGenerateResponse,
        UsageInfo,
        list[tuple[int, str | None]],
    ]:
        """The response, its usage and (index, finish_reason) per choice.

        CPU only, no shared-state side effects: may run in a worker thread.
        """
        sampling_params: SamplingParams = request.sampling_params
        text_mode = request.output_mode == "text"
        tokenizer = self._logprobs_tokenizer(text_mode)
        # choice position -> pre-rendered JSON of its logprobs
        fragments: dict[int, list[bytes]] = {}
        tokens_choices: list[GenerateTokensChoice] = []
        text_choices: list[GenerateTextChoice] = []
        num_generated_tokens = 0
        for output in final_res.outputs:
            self._raise_if_error(output.finish_reason, request_id)

            token_ids = output.token_ids
            out_logprobs = output.logprobs

            sampled = None
            if request.return_token_logprobs:
                if output.sampled_logprobs is not None:
                    sampled = [_clamp_logprob(x) for x in output.sampled_logprobs]
                else:
                    assert isinstance(out_logprobs, FlatLogprobs), (
                        "Did not output logprobs"
                    )
                    sampled = self._sampled_token_logprobs(out_logprobs)

            # This is top_logprobs in completions API. With
            # return_token_logprobs the objects are only built when the
            # caller also asked for top logprobs.
            logprobs: GenerateLogProbs | ChatCompletionLogProbs | None = None
            rendered = None
            if sampling_params.logprobs is not None and not (
                request.return_token_logprobs and sampling_params.logprobs == 0
            ):
                assert out_logprobs is not None, "Did not output logprobs"
                top_logprobs: GenericSequence[dict[int, Logprob] | None] = out_logprobs
                if request.return_top_k_logprobs:
                    rendered = render_top_k_logprobs(
                        out_logprobs, len(token_ids), sampling_params.logprobs, sampled
                    )
                elif isinstance(out_logprobs, FlatLogprobs) and not text_mode:
                    rendered = render_tokens_logprobs(
                        token_ids, out_logprobs, sampling_params.logprobs
                    )
                    if rendered is not None and sampled is not None:
                        rendered = append_sampled_logprobs(rendered, sampled)
                    if rendered is None:
                        # Irregular rows: the per-entry path, in one pass.
                        top_logprobs = list(out_logprobs)
                if rendered is not None:
                    fragments[len(tokens_choices)] = rendered
                elif text_mode:
                    logprobs = self._create_text_logprobs(
                        token_ids=token_ids,
                        top_logprobs=out_logprobs,
                        num_output_top_logprobs=sampling_params.logprobs,
                        tokenizer=tokenizer,
                    )
                else:
                    logprobs = self._create_tokens_logprobs(
                        token_ids=token_ids,
                        top_logprobs=top_logprobs,
                        num_output_top_logprobs=sampling_params.logprobs,
                    )
            if sampled is not None and rendered is None:
                # Tokens mode only (checked above). With logprobs=0 no content
                # entries were built: content stays None, only sampled is set.
                if logprobs is None:
                    logprobs = GenerateLogProbs(sampled=sampled)
                else:
                    assert isinstance(logprobs, GenerateLogProbs)
                    logprobs.sampled = sampled

            routed_experts_b64 = (
                numpy2base64(output.routed_experts)
                if output.routed_experts is not None
                else None
            )

            sampling_mask = None
            if output.sampling_mask is not None:
                sampling_mask = output.sampling_mask.token_ids

            choice_fields: dict[str, Any] = dict(
                index=output.index,
                logprobs=logprobs,
                finish_reason=output.finish_reason if output.finish_reason else "stop",
                token_ids=as_list(output.token_ids),
                routed_experts=routed_experts_b64,
                sampling_mask=sampling_mask,
            )
            if text_mode:
                text_choices.append(
                    GenerateTextChoice(text=output.text, **choice_fields)
                )
            else:
                tokens_choices.append(GenerateTokensChoice(**choice_fields))
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
        if (
            self.enable_prompt_tokens_details
            and final_res.num_cached_tokens is not None
        ):
            # This info is not available at the /coordinator level
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=final_res.num_cached_tokens
            )

        per_request_metrics = None
        if request.sampling_params.n == 1:
            spec_stats = build_spec_decoding_metrics(final_res)
            if spec_stats is not None:
                per_request_metrics = PerRequestMetrics(speculative_decoding=spec_stats)
        response_fields: dict[str, Any] = dict(
            request_id=request_id,
            created=created_time,
            model=model_name,
            usage=usage,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
            prompt_token_id_logprobs=(
                numpy2base64(final_res.prompt_token_id_logprobs)
                if final_res.prompt_token_id_logprobs is not None
                else None
            ),
            prompt_token_ids=(
                final_res.prompt_token_ids if request.return_token_ids else None
            ),
            mm_placeholders=request._response_mm_placeholders,
            metrics=per_request_metrics,
            kv_transfer_params=final_res.kv_transfer_params,
            ec_transfer_params=final_res.ec_transfer_params,
        )
        response: GenerateTokensResponse | GenerateTextResponse
        if text_mode:
            response = GenerateTextResponse(choices=text_choices, **response_fields)
        else:
            response = GenerateTokensResponse(choices=tokens_choices, **response_fields)

        choice_meta = [
            (choice.index, choice.finish_reason) for choice in response.choices
        ]
        if fragments:
            parts = render_json_with_fragments(
                response.model_dump(), "logprobs", fragments
            )
            # Joined here, so a large body is never copied on the event loop.
            return RenderedGenerateResponse(b"".join(parts)), usage, choice_meta
        return response, usage, choice_meta

    async def serve_tokens_stream_generator(
        self,
        request: GenerateRequest,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        model_name: str,
        request_metadata: RequestResponseMetadata,
    ) -> AsyncGenerator[str, None]:
        num_prompt_tokens = 0
        first_iteration = True
        prompt_token_ids: list[int] | None = None
        num_cached_tokens = None
        sampling_params: SamplingParams = request.sampling_params
        # With n > 1, the first output may not include every choice yet.
        num_generated_tokens = [0] * sampling_params.n
        last_res: RequestOutput | None = None
        text_mode = request.output_mode == "text"
        tokenizer = self._logprobs_tokenizer(text_mode)

        include_usage, include_continuous_usage = should_include_usage(
            request.stream_options, False
        )

        try:
            async for res in result_generator:
                last_res = res
                if first_iteration:
                    if res.prompt_token_ids is not None:
                        num_prompt_tokens = len(res.prompt_token_ids)
                        if request.return_token_ids:
                            prompt_token_ids = res.prompt_token_ids
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)
                    num_cached_tokens = res.num_cached_tokens
                    first_iteration = False

                for output in res.outputs:
                    i = output.index
                    delta_token_ids = output.token_ids
                    num_generated_tokens[i] += len(delta_token_ids)

                    finish_reason = output.finish_reason
                    self._raise_if_error(finish_reason, request_id)

                    # Terminal outputs are always emitted so the client sees
                    # the finish reason, e.g. an abort, which has no new
                    # tokens. Text mode also emits text held back for stop
                    # string matching, which can arrive without token IDs.
                    if not (
                        delta_token_ids
                        or finish_reason is not None
                        or (text_mode and output.text)
                    ):
                        continue

                    if sampling_params.logprobs is not None:
                        out_logprobs = output.logprobs
                        assert out_logprobs is not None, "Did not output logprobs"
                        logprobs: GenerateLogProbs | ChatCompletionLogProbs | None
                        if text_mode:
                            logprobs = self._create_text_logprobs(
                                token_ids=delta_token_ids,
                                top_logprobs=out_logprobs,
                                num_output_top_logprobs=sampling_params.logprobs,
                                tokenizer=tokenizer,
                            )
                        else:
                            logprobs = self._create_tokens_logprobs(
                                token_ids=delta_token_ids,
                                top_logprobs=out_logprobs,
                                num_output_top_logprobs=sampling_params.logprobs,
                            )
                    else:
                        logprobs = None

                    routed_experts_b64 = (
                        numpy2base64(output.routed_experts)
                        if output.routed_experts is not None
                        else None
                    )

                    sampling_mask = None
                    if output.sampling_mask is not None:
                        sampling_mask = output.sampling_mask.token_ids
                    choice_fields: dict[str, Any] = dict(
                        index=i,
                        logprobs=logprobs,
                        finish_reason=finish_reason,
                        token_ids=as_list(delta_token_ids),
                        routed_experts=routed_experts_b64,
                        sampling_mask=sampling_mask,
                    )
                    chunk: GenerateTokensStreamResponse | GenerateTextStreamResponse
                    if text_mode:
                        chunk = GenerateTextStreamResponse(
                            request_id=request_id,
                            choices=[
                                GenerateTextStreamChoice(
                                    text=output.text, **choice_fields
                                )
                            ],
                        )
                    else:
                        chunk = GenerateTokensStreamResponse(
                            request_id=request_id,
                            choices=[GenerateTokensStreamChoice(**choice_fields)],
                        )

                    if prompt_token_ids is not None:
                        chunk.prompt_token_ids = prompt_token_ids
                        chunk.mm_placeholders = request._response_mm_placeholders
                        prompt_token_ids = None
                    if include_continuous_usage:
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=num_generated_tokens[i],
                            total_tokens=(num_prompt_tokens + num_generated_tokens[i]),
                        )

                    # Omit fields that are absent from token-bearing chunks.
                    exclude = {
                        name
                        for name in ("prompt_token_ids", "mm_placeholders", "metrics")
                        if getattr(chunk, name) is None
                    }
                    yield f"data: {chunk.model_dump_json(exclude=exclude)}\n\n"

            total_completion_tokens = sum(num_generated_tokens)
            final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=num_prompt_tokens + total_completion_tokens,
            )

            if self.enable_prompt_tokens_details and num_cached_tokens is not None:
                final_usage_info.prompt_tokens_details = PromptTokenUsageInfo(
                    cached_tokens=num_cached_tokens
                )

            if include_usage:
                per_request_metrics = None
                if sampling_params.n == 1:
                    spec_stats = build_spec_decoding_metrics(last_res)
                    if spec_stats is not None:
                        per_request_metrics = PerRequestMetrics(
                            speculative_decoding=spec_stats
                        )
                final_chunk: GenerateTokensStreamResponse | GenerateTextStreamResponse
                if text_mode:
                    final_chunk = GenerateTextStreamResponse(
                        request_id=request_id,
                        choices=[],
                        usage=final_usage_info,
                        metrics=per_request_metrics,
                    )
                else:
                    final_chunk = GenerateTokensStreamResponse(
                        request_id=request_id,
                        choices=[],
                        usage=final_usage_info,
                        metrics=per_request_metrics,
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

    def _logprobs_tokenizer(self, text_mode: bool) -> TokenizerLike | None:
        """Tokenizer that resolves logprob tokens, or None for placeholders.

        ``--return-tokens-as-token-ids`` keeps placeholders at every level, as
        it does on ``/v1/completions``.
        """
        if not text_mode or self.return_tokens_as_token_ids:
            return None
        return self.renderer.tokenizer

    @staticmethod
    def _sampled_token_logprobs(flat: FlatLogprobs) -> list[float]:
        """Sampled-token logprob per position from the flat representation.

        The sampler stores the sampled token first at every position, so its
        logprob is each position's first entry. Every position carries at
        least that entry whenever ``logprobs`` is requested. Values go through
        ``_clamp_logprob`` (``-inf`` and NaN become ``-9999.0``), so they stay
        JSON-representable.
        """
        return [_clamp_logprob(v) for v in flat.sampled_logprobs()]

    def _create_text_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[dict[int, Logprob] | None],
        num_output_top_logprobs: int | None = None,
        tokenizer: TokenizerLike | None = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs for ``output_mode="text"``.

        With a ``tokenizer`` the tokens are decoded strings and carry ``bytes``;
        without one (``--return-tokens-as-token-ids``) they are ``token_id:N``
        placeholders.
        """
        logprobs_content: list[ChatCompletionLogProbsContent] = []

        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None or step_top_logprobs.get(token_id) is None:
                token, token_bytes = _logprob_token(token_id, None, tokenizer)
                logprobs_content.append(
                    ChatCompletionLogProbsContent(token=token, bytes=token_bytes)
                )
            else:
                step_token = step_top_logprobs[token_id]
                token, token_bytes = _logprob_token(token_id, step_token, tokenizer)

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                        logprob=_clamp_logprob(step_token.logprob),
                        bytes=token_bytes,
                        top_logprobs=[
                            _top_logprob(top_id, logprob, tokenizer)
                            for i, (top_id, logprob) in enumerate(
                                step_top_logprobs.items()
                            )
                            if num_output_top_logprobs is not None
                            and (
                                num_output_top_logprobs == -1
                                or i < max(num_output_top_logprobs, 1)
                            )
                        ],
                    )
                )

        return ChatCompletionLogProbs(content=logprobs_content)

    def _create_tokens_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[dict[int, Logprob] | None],
        num_output_top_logprobs: int | None = None,
    ) -> GenerateLogProbs:
        """Create generate-shaped logprobs (integer token ids, no tokenizer).

        The engine reports rank 0 for a sampled token whose logprob is NaN; that
        is not a rank, so it is sent as ``None`` (the logprob is clamped).
        """
        logprobs_content: list[GenerateLogProbsContent] = []

        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None or step_top_logprobs.get(token_id) is None:
                # Same sentinel the OpenAI shapes use when the sampled token
                # has no entry in the engine's top-k map.
                logprobs_content.append(
                    GenerateLogProbsContent(token_id=token_id, logprob=-9999.0)
                )
            else:
                step_token = step_top_logprobs[token_id]

                logprobs_content.append(
                    GenerateLogProbsContent(
                        token_id=token_id,
                        logprob=_clamp_logprob(step_token.logprob),
                        rank=step_token.rank or None,
                        top_logprobs=[
                            GenerateLogProb(
                                token_id=top_token_id,
                                logprob=_clamp_logprob(top_logprob.logprob),
                                rank=top_logprob.rank or None,
                            )
                            for rank_index, (top_token_id, top_logprob) in enumerate(
                                step_top_logprobs.items()
                            )
                            if num_output_top_logprobs is not None
                            and (
                                num_output_top_logprobs == -1
                                or rank_index < max(num_output_top_logprobs, 1)
                            )
                        ],
                    )
                )

        return GenerateLogProbs(content=logprobs_content)
