# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
import time
from collections.abc import AsyncGenerator, Awaitable, Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, ClassVar, Generic, Protocol, TypeAlias, TypeVar

import numpy as np
from fastapi import Request
from pydantic import ConfigDict
from starlette.datastructures import Headers

import vllm.envs as envs
from vllm import CompletionOutput, PoolingRequestOutput, RequestOutput
from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.beam_search_utils import (
    get_beam_allowed_token_ids,
    get_trie_allowed_token_ids,
    init_beam_search_so_backend,
)
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.generate.beam_search.online import BeamSearchOnlineMixin
from vllm.entrypoints.generate.beam_search.utils import (
    BeamSearchSequence,
    create_sort_beams_key_function,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    GenerationError,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.serve.disagg.protocol import GenerateRequest, GenerateResponse
from vllm.entrypoints.serve.tokenize.protocol import (
    DetokenizeRequest,
    TokenizeChatRequest,
    TokenizeCompletionRequest,
    TokenizeResponse,
)
from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
    TranscriptionResponse,
)
from vllm.entrypoints.speech_to_text.translation.protocol import TranslationRequest
from vllm.inputs import EngineInput, PromptType
from vllm.logger import init_logger
from vllm.logprobs import Logprob, PromptLogprobs
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.renderers import ChatParams, TokenizeParams
from vllm.renderers.inputs.preprocess import (
    extract_prompt_components,
    extract_prompt_len,
)
from vllm.sampling_params import (
    BeamSearchParams,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.tokenizers import TokenizerLike
from vllm.tracing import (
    contains_trace_headers,
    extract_trace_headers,
    log_tracing_disabled_warning,
)
from vllm.utils import random_uuid
from vllm.utils.async_utils import (
    collect_from_async_generator,
    merge_async_iterators,
)
from vllm.v1.structured_output.backend_types import StructuredOutputBackend

logger = init_logger(__name__)


class RendererRequest(Protocol):
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        raise NotImplementedError


class RendererChatRequest(RendererRequest, Protocol):
    def build_chat_params(
        self,
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
    ) -> ChatParams:
        raise NotImplementedError


CompletionLikeRequest: TypeAlias = (
    CompletionRequest | TokenizeCompletionRequest | DetokenizeRequest
)

ChatLikeRequest: TypeAlias = (
    ChatCompletionRequest | BatchChatCompletionRequest | TokenizeChatRequest
)

SpeechToTextRequest: TypeAlias = TranscriptionRequest | TranslationRequest

AnyRequest: TypeAlias = (
    CompletionLikeRequest
    | ChatLikeRequest
    | SpeechToTextRequest
    | ResponsesRequest
    | GenerateRequest
)

AnyResponse: TypeAlias = (
    CompletionResponse
    | ChatCompletionResponse
    | TranscriptionResponse
    | TokenizeResponse
    | GenerateResponse
)

RequestT = TypeVar("RequestT", bound=AnyRequest)
_T = TypeVar("_T")


@dataclass(kw_only=True)
class ServeContext(Generic[RequestT]):
    request: RequestT
    raw_request: Request | None = None
    model_name: str
    request_id: str
    created_time: int = field(default_factory=lambda: int(time.time()))
    lora_request: LoRARequest | None = None
    engine_inputs: list[EngineInput] | None = None
    result_generator: AsyncGenerator[tuple[int, PoolingRequestOutput], None] | None = (
        None
    )
    final_res_batch: list[PoolingRequestOutput] | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class OpenAIServing(BeamSearchOnlineMixin):
    request_id_prefix: ClassVar[str] = """
    A short string prepended to every request’s ID.
    """

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
    ):
        super().__init__()

        self.engine_client = engine_client
        self.models = models

        self.request_logger = request_logger
        self.return_tokens_as_token_ids = return_tokens_as_token_ids

        self.model_config = engine_client.model_config
        self.renderer = engine_client.renderer
        self.input_processor = engine_client.input_processor
        vllm_config = getattr(engine_client, "vllm_config", None)
        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
        self.has_kv_connector = kv_transfer_config is not None

        # Computed once at startup (cached by ``vllm_config`` identity) and
        # stamped on non-streaming responses. Streaming chunks deliberately
        # omit it to avoid per-chunk overhead.
        from vllm.entrypoints.serve.utils.fingerprint import get_system_fingerprint

        try:
            self.system_fingerprint: str | None = get_system_fingerprint(
                engine_client.vllm_config
            )
        except Exception:
            # Never fail server startup over the fingerprint.
            self.system_fingerprint = None

    def _init_beam_search_so_backend(
        self,
        structured_outputs: StructuredOutputsParams,
    ) -> tuple:
        """Initialize a structured output backend for beam search.

        Delegates to the shared utility in
        :mod:`vllm.entrypoints.beam_search_utils`.
        """
        return init_beam_search_so_backend(
            vllm_config=self.engine_client.vllm_config,
            tokenizer=self.renderer.get_tokenizer(),
            vocab_size=self.model_config.get_vocab_size(),
            structured_outputs=structured_outputs,
        )

    @staticmethod
    def _get_beam_allowed_token_ids(
        beam: BeamSearchSequence,
        backend: StructuredOutputBackend,
        so_key: tuple,
        bitmask: Any,
        vocab_size: int,
    ) -> list[int] | None:
        """Compute the set of grammar-allowed token IDs for a beam.

        Delegates to the shared utility in
        :mod:`vllm.entrypoints.beam_search_utils`.
        """
        return get_beam_allowed_token_ids(beam, backend, so_key, bitmask, vocab_size)

    async def beam_search(
        self,
        prompt: EngineInput,
        request_id: str,
        params: BeamSearchParams,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        beam_width = params.beam_width
        max_tokens = params.max_tokens
        ignore_eos = params.ignore_eos
        temperature = params.temperature
        length_penalty = params.length_penalty
        include_stop_str_in_output = params.include_stop_str_in_output

        tokenizer = self.renderer.get_tokenizer()
        eos_token_id = tokenizer.eos_token_id
        sort_beams_key = create_sort_beams_key_function(eos_token_id, length_penalty)

        if prompt["type"] == "embeds":
            raise NotImplementedError("Embedding prompt not supported for beam search")

        decoder_prompt = (
            prompt if prompt["type"] != "enc_dec" else prompt["decoder_prompt"]
        )
        prompt_text = decoder_prompt.get("prompt")
        prompt_token_ids = decoder_prompt["prompt_token_ids"]
        tokenized_length = len(prompt_token_ids)

        logprobs_num = 2 * beam_width
        sampling_params = SamplingParams(
            logprobs=logprobs_num,
            max_tokens=1,
            temperature=temperature,
        )
        all_beams = [
            BeamSearchSequence(
                orig_prompt=prompt,
                tokens=prompt_token_ids,
                cum_logprob=0,
                logprobs=[],
                lora_request=lora_request,
            )
        ]
        completed: list[BeamSearchSequence] = []

        # Initialize structured output backend if requested.
        so_backend: StructuredOutputBackend | None = None
        so_key: tuple | None = None
        so_bitmask = None
        so_trie = None
        if params.structured_outputs is not None:
            so_backend, so_key, so_bitmask, so_trie = self._init_beam_search_so_backend(
                params.structured_outputs
            )

        try:
            for _ in range(max_tokens):
                # -- Build per-beam sampling params ----------------------
                if so_trie is not None:
                    # Fast path: re-walk trie from root each step (stateless,
                    # no per-beam pointer to maintain or clone).
                    beam_sp_list: list[SamplingParams] = []
                    active_beams: list[BeamSearchSequence] = []

                    for beam in all_beams:
                        allowed_ids = get_trie_allowed_token_ids(beam, so_trie)
                        if not allowed_ids:
                            # None = trie terminal (after EOS); [] = off-trie.
                            # Both cases: drop beam (EOS already handled below).
                            pass
                        else:
                            beam_sp_list.append(
                                SamplingParams(
                                    logprobs=logprobs_num,
                                    max_tokens=1,
                                    temperature=temperature,
                                    allowed_token_ids=allowed_ids,
                                )
                            )
                            active_beams.append(beam)

                    if not active_beams:
                        break
                elif so_backend is not None:
                    assert so_key is not None and so_bitmask is not None
                    vocab_size = self.model_config.get_vocab_size()
                    beam_sp_list = []
                    active_beams = []

                    for beam in all_beams:
                        allowed_ids = self._get_beam_allowed_token_ids(
                            beam,
                            so_backend,
                            so_key,
                            so_bitmask,
                            vocab_size,
                        )
                        if allowed_ids is None:
                            # Grammar terminated → mark beam completed.
                            completed.append(
                                BeamSearchSequence(
                                    orig_prompt=prompt,
                                    tokens=beam.tokens,
                                    logprobs=beam.logprobs,
                                    cum_logprob=beam.cum_logprob,
                                    lora_request=beam.lora_request,
                                    finish_reason="stop",
                                )
                            )
                        else:
                            beam_sp_list.append(
                                SamplingParams(
                                    logprobs=logprobs_num,
                                    max_tokens=1,
                                    temperature=temperature,
                                    allowed_token_ids=allowed_ids,
                                )
                            )
                            active_beams.append(beam)

                    if not active_beams:
                        break  # All beams terminated by grammar.
                else:
                    active_beams = all_beams
                    beam_sp_list = [sampling_params] * len(all_beams)

                # -- Launch inference for each active beam ---------------
                tasks = []
                request_id_batch = f"{request_id}-{random_uuid()}"

                for i, beam in enumerate(active_beams):
                    prompt_item = beam.get_prompt()
                    lora_request_item = beam.lora_request
                    request_id_item = f"{request_id_batch}-beam-{i}"
                    task = asyncio.create_task(
                        collect_from_async_generator(
                            self.engine_client.generate(
                                prompt_item,
                                beam_sp_list[i],
                                request_id_item,
                                lora_request=lora_request_item,
                                trace_headers=trace_headers,
                            )
                        )
                    )
                    tasks.append(task)

                output = [x[0] for x in await asyncio.gather(*tasks)]

                # -- Collect logprobs from each beam result --------------
                new_beams: list[BeamSearchSequence] = []
                all_beams_token_id: list[int] = []
                all_beams_logprob: list[float] = []

                # Build per-beam allowed-id sets for logprob filtering.
                # When structured output is active, logprobs are computed
                # from raw logits *before* the allowed_token_ids mask is
                # applied, so they may include disallowed tokens.
                allowed_sets: list[set[int] | None] = []
                if so_backend is not None or so_trie is not None:
                    for sp in beam_sp_list:
                        if sp.allowed_token_ids:
                            allowed_sets.append(set(sp.allowed_token_ids))
                        else:
                            allowed_sets.append(None)
                else:
                    allowed_sets = [None] * len(active_beams)

                for i, result in enumerate(output):
                    current_beam = active_beams[i]

                    # Check for error finish reason and abort.
                    if result.outputs[0].finish_reason == "error":
                        yield RequestOutput(
                            request_id=request_id,
                            prompt=prompt_text,
                            outputs=[
                                CompletionOutput(
                                    index=0,
                                    text="",
                                    token_ids=[],
                                    cumulative_logprob=None,
                                    logprobs=None,
                                    finish_reason="error",
                                )
                            ],
                            finished=True,
                            prompt_token_ids=prompt_token_ids,
                            prompt_logprobs=None,
                        )
                        return

                    if result.outputs[0].logprobs is not None:
                        logprobs = result.outputs[0].logprobs[0]
                        allowed = allowed_sets[i]
                        for token_id, logprob_obj in logprobs.items():
                            if allowed is not None and token_id not in allowed:
                                continue
                            all_beams_token_id.append(token_id)
                            all_beams_logprob.append(
                                current_beam.cum_logprob + logprob_obj.logprob
                            )

                # Handle EOS tokens.
                all_beams_token_id_np = np.array(all_beams_token_id)
                all_beams_logprob_np = np.array(all_beams_logprob)

                if not ignore_eos:
                    eos_idx = np.where(all_beams_token_id_np == eos_token_id)[0]
                    for idx in eos_idx:
                        # Map flat index back to parent beam.
                        parent_beam_idx = self._flat_idx_to_beam(
                            int(idx),
                            output,
                            active_beams,
                            allowed_sets,
                        )
                        current_beam = active_beams[parent_beam_idx]
                        _lp = output[parent_beam_idx].outputs[0].logprobs
                        assert _lp is not None
                        logprobs_entry = _lp[0]
                        completed.append(
                            BeamSearchSequence(
                                orig_prompt=prompt,
                                tokens=current_beam.tokens + [eos_token_id]
                                if include_stop_str_in_output
                                else current_beam.tokens,
                                logprobs=current_beam.logprobs + [logprobs_entry],
                                cum_logprob=float(all_beams_logprob_np[idx]),
                                finish_reason="stop",
                                stop_reason=eos_token_id,
                            )
                        )
                    all_beams_logprob_np[eos_idx] = -np.inf

                if len(all_beams_logprob_np) == 0:
                    break

                # -- Select top beam_width candidates, deduplicating trie paths.
                n_candidates = len(all_beams_logprob_np)
                k = min(beam_width, n_candidates)
                if k >= n_candidates:
                    topn_idx = np.arange(n_candidates)
                else:
                    topn_idx = np.argpartition(np.negative(all_beams_logprob_np), k)[:k]
                # Sort so we pick highest-prob first (important for trie dedup).
                topn_idx = topn_idx[np.argsort(all_beams_logprob_np[topn_idx])[::-1]]

                seen_trie_sequences: set[tuple] = set()
                for idx in topn_idx:
                    parent_beam_idx = self._flat_idx_to_beam(
                        int(idx),
                        output,
                        active_beams,
                        allowed_sets,
                    )
                    current_beam = active_beams[parent_beam_idx]
                    token_id = int(all_beams_token_id_np[idx])
                    _lp = output[parent_beam_idx].outputs[0].logprobs
                    assert _lp is not None
                    logprobs_entry = _lp[0]
                    new_tokens = current_beam.tokens + [token_id]
                    # Dedup by generated token sequence when using trie — two
                    # beams with identical generated tokens are identical choices.
                    if so_trie is not None:
                        orig = current_beam.orig_prompt
                        if orig["type"] == "enc_dec":
                            prompt_len = len(orig["decoder_prompt"]["prompt_token_ids"])
                        else:
                            prompt_len = len(orig["prompt_token_ids"])
                        gen_key = tuple(new_tokens[prompt_len:])
                        if gen_key in seen_trie_sequences:
                            continue
                        seen_trie_sequences.add(gen_key)
                    new_beams.append(
                        BeamSearchSequence(
                            orig_prompt=prompt,
                            tokens=new_tokens,
                            logprobs=current_beam.logprobs + [logprobs_entry],
                            lora_request=current_beam.lora_request,
                            cum_logprob=float(all_beams_logprob_np[idx]),
                        )
                    )

                all_beams = new_beams

        finally:
            if so_backend is not None:
                so_backend.destroy()

        completed.extend(all_beams)
        sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]

        for beam in best_beams:
            if beam.tokens[-1] == eos_token_id and not ignore_eos:
                tokens = beam.tokens[tokenized_length:-1]
            else:
                tokens = beam.tokens[tokenized_length:]
            beam.text = tokenizer.decode(tokens)

        yield RequestOutput(
            request_id=request_id,
            prompt=prompt_text,
            outputs=[
                CompletionOutput(
                    text=beam.text,  # type: ignore
                    cumulative_logprob=beam.cum_logprob,
                    token_ids=beam.tokens[tokenized_length:],
                    index=i,
                    logprobs=beam.logprobs,
                    finish_reason=beam.finish_reason
                    if beam.finish_reason is not None
                    else "length",
                    stop_reason=beam.stop_reason,
                )
                for (i, beam) in enumerate(best_beams)
            ],
            finished=True,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None,
        )

    @staticmethod
    def _flat_idx_to_beam(
        flat_idx: int,
        output: list[RequestOutput],
        active_beams: list[BeamSearchSequence],
        allowed_sets: list[set[int] | None],
    ) -> int:
        """Map a flat token index back to its parent beam index.

        When logprobs are filtered (structured output), the number of
        tokens contributed by each beam varies.  This helper walks
        through the per-beam contribution counts to find the parent.
        """
        cumulative = 0
        for beam_idx, result in enumerate(output):
            if result.outputs[0].logprobs is not None:
                logprobs = result.outputs[0].logprobs[0]
                allowed = allowed_sets[beam_idx]
                if allowed is not None:
                    count = sum(1 for tid in logprobs if tid in allowed)
                else:
                    count = len(logprobs)
                if flat_idx < cumulative + count:
                    return beam_idx
                cumulative += count
        # Should not be reached; raise to surface indexing bugs.
        raise RuntimeError(
            f"_flat_idx_to_beam: flat_idx {flat_idx} could not be mapped "
            f"to any beam (cumulative={cumulative})"
        )

    async def _preprocess(
        self,
        ctx: ServeContext,
    ) -> ErrorResponse | None:
        """
        Default preprocessing hook. Subclasses may override to prepare `ctx`.
        """
        return None

    def _build_response(
        self,
        ctx: ServeContext,
    ) -> AnyResponse | ErrorResponse:
        """
        Default response builder. Subclass may override this method
        to return the appropriate response object.
        """
        return self.create_error_response("unimplemented endpoint")

    async def handle(
        self,
        ctx: ServeContext,
    ) -> AnyResponse | ErrorResponse:
        async for response in self._pipeline(ctx):
            return response

        return self.create_error_response("No response yielded from pipeline")

    async def _pipeline(
        self,
        ctx: ServeContext,
    ) -> AsyncGenerator[AnyResponse | ErrorResponse, None]:
        """Execute the request processing pipeline yielding responses."""
        if error := await self._check_model(ctx.request):
            yield error
        if error := self._validate_request(ctx):
            yield error

        preprocess_ret = await self._preprocess(ctx)
        if isinstance(preprocess_ret, ErrorResponse):
            yield preprocess_ret

        generators_ret = await self._prepare_generators(ctx)
        if isinstance(generators_ret, ErrorResponse):
            yield generators_ret

        collect_ret = await self._collect_batch(ctx)
        if isinstance(collect_ret, ErrorResponse):
            yield collect_ret

        yield self._build_response(ctx)

    def _validate_request(self, ctx: ServeContext) -> ErrorResponse | None:
        truncate_prompt_tokens = getattr(ctx.request, "truncate_prompt_tokens", None)

        if (
            truncate_prompt_tokens is not None
            and truncate_prompt_tokens > self.model_config.max_model_len
        ):
            return self.create_error_response(
                "truncate_prompt_tokens value is "
                "greater than max_model_len."
                " Please request a smaller truncation size."
            )
        return None

    def _create_pooling_params(
        self,
        ctx: ServeContext,
    ) -> PoolingParams | ErrorResponse:
        if not hasattr(ctx.request, "to_pooling_params"):
            return self.create_error_response(
                "Request type does not support pooling parameters"
            )

        return ctx.request.to_pooling_params()

    async def _prepare_generators(
        self,
        ctx: ServeContext,
    ) -> ErrorResponse | None:
        """Schedule the request and get the result generator."""
        generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []

        trace_headers = (
            None
            if ctx.raw_request is None
            else await self._get_trace_headers(ctx.raw_request.headers)
        )

        pooling_params = self._create_pooling_params(ctx)
        if isinstance(pooling_params, ErrorResponse):
            return pooling_params

        if ctx.engine_inputs is None:
            return self.create_error_response("Engine prompts not available")

        for i, engine_input in enumerate(ctx.engine_inputs):
            request_id_item = f"{ctx.request_id}-{i}"

            self._log_inputs(
                request_id_item,
                engine_input,
                params=pooling_params,
                lora_request=ctx.lora_request,
            )

            generator = self.engine_client.encode(
                engine_input,
                pooling_params,
                request_id_item,
                lora_request=ctx.lora_request,
                trace_headers=trace_headers,
                priority=getattr(ctx.request, "priority", 0),
            )

            generators.append(generator)

        ctx.result_generator = merge_async_iterators(*generators)

        return None

    async def _collect_batch(
        self,
        ctx: ServeContext,
    ) -> ErrorResponse | None:
        """Collect batch results from the result generator."""
        if ctx.engine_inputs is None:
            return self.create_error_response("Engine prompts not available")

        num_prompts = len(ctx.engine_inputs)
        final_res_batch: list[PoolingRequestOutput | None]
        final_res_batch = [None] * num_prompts

        if ctx.result_generator is None:
            return self.create_error_response("Result generator not available")

        async for i, res in ctx.result_generator:
            final_res_batch[i] = res

        if None in final_res_batch:
            return self.create_error_response(
                "Failed to generate results for all prompts"
            )

        ctx.final_res_batch = [res for res in final_res_batch if res is not None]

        return None

    @staticmethod
    def create_error_response(
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> ErrorResponse:
        return create_error_response(message, err_type, status_code, param)

    def create_streaming_error_response(
        self,
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> str:
        json_str = json.dumps(
            self.create_error_response(
                message=message,
                err_type=err_type,
                status_code=status_code,
                param=param,
            ).model_dump()
        )
        return json_str

    def _raise_if_error(self, finish_reason: str | None, request_id: str) -> None:
        """Raise GenerationError if finish_reason indicates an error."""
        if finish_reason == "error":
            logger.error(
                "Request %s failed with an internal error during generation",
                request_id,
            )
            raise GenerationError("Internal server error")

    def _convert_generation_error_to_streaming_response(
        self, e: GenerationError
    ) -> str:
        """Convert GenerationError to streaming error response."""
        return self.create_streaming_error_response(
            str(e),
            err_type="InternalServerError",
            status_code=e.status_code,
        )

    async def _check_model(
        self,
        request: AnyRequest,
    ) -> ErrorResponse | None:
        error_response = None

        if self._is_model_supported(request.model):
            return None
        if request.model in self.models.lora_requests:
            return None
        if (
            envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING
            and request.model
            and (load_result := await self.models.resolve_lora(request.model))
        ):
            if isinstance(load_result, LoRARequest):
                return None
            if (
                isinstance(load_result, ErrorResponse)
                and load_result.error.code == HTTPStatus.BAD_REQUEST.value
            ):
                error_response = load_result

        return error_response or self.create_error_response(
            message=f"The model `{request.model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
            param="model",
        )

    def _get_active_default_mm_loras(self, request: AnyRequest) -> LoRARequest | None:
        """Determine if there are any active default multimodal loras."""
        # TODO: Currently this is only enabled for chat completions
        # to be better aligned with only being enabled for .generate
        # when run offline. It would be nice to support additional
        # tasks types in the future.
        message_types = self._get_message_types(request)
        default_mm_loras = set()

        for lora in self.models.lora_requests.values():
            # Best effort match for default multimodal lora adapters;
            # There is probably a better way to do this, but currently
            # this matches against the set of 'types' in any content lists
            # up until '_', e.g., to match audio_url -> audio
            if lora.lora_name in message_types:
                default_mm_loras.add(lora)

        # Currently only support default modality specific loras if
        # we have exactly one lora matched on the request.
        if len(default_mm_loras) == 1:
            return default_mm_loras.pop()
        return None

    def _maybe_get_adapters(
        self,
        request: AnyRequest,
        supports_default_mm_loras: bool = False,
    ) -> LoRARequest | None:
        if request.model in self.models.lora_requests:
            return self.models.lora_requests[request.model]

        # Currently only support default modality specific loras
        # if we have exactly one lora matched on the request.
        if supports_default_mm_loras:
            default_mm_lora = self._get_active_default_mm_loras(request)
            if default_mm_lora is not None:
                return default_mm_lora

        if self._is_model_supported(request.model):
            return None

        # if _check_model has been called earlier, this will be unreachable
        raise ValueError(f"The model `{request.model}` does not exist.")

    def _get_message_types(self, request: AnyRequest) -> set[str]:
        """Retrieve the set of types from message content dicts up
        until `_`; we use this to match potential multimodal data
        with default per modality loras.
        """
        message_types: set[str] = set()

        if not hasattr(request, "messages"):
            return message_types

        messages = request.messages
        if messages is None or isinstance(messages, (str, bytes)):
            return message_types

        for message in messages:
            if (
                isinstance(message, dict)
                and "content" in message
                and isinstance(message["content"], list)
            ):
                for content_dict in message["content"]:
                    if "type" in content_dict:
                        message_types.add(content_dict["type"].split("_")[0])
        return message_types

    def _validate_chat_template(
        self,
        request_chat_template: str | None,
        chat_template_kwargs: dict[str, Any] | None,
        trust_request_chat_template: bool,
    ) -> ErrorResponse | None:
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

    @staticmethod
    def _prepare_extra_chat_template_kwargs(
        request_chat_template_kwargs: dict[str, Any] | None = None,
        default_chat_template_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Helper to merge server-default and request-specific chat template kwargs."""
        request_chat_template_kwargs = request_chat_template_kwargs or {}
        if default_chat_template_kwargs is None:
            return request_chat_template_kwargs
        # Apply server defaults first, then request kwargs override.
        return default_chat_template_kwargs | request_chat_template_kwargs

    def _extract_prompt_components(self, prompt: PromptType | EngineInput):
        return extract_prompt_components(self.model_config, prompt)

    def _extract_prompt_text(self, prompt: PromptType | EngineInput):
        return self._extract_prompt_components(prompt).text

    def _extract_prompt_len(self, prompt: EngineInput):
        return extract_prompt_len(self.model_config, prompt)

    def _log_inputs(
        self,
        request_id: str,
        inputs: PromptType | EngineInput,
        params: SamplingParams | BeamSearchParams | None,
        lora_request: LoRARequest | None,
    ) -> None:
        if self.request_logger is None:
            return

        components = self._extract_prompt_components(inputs)

        self.request_logger.log_inputs(
            request_id,
            components.text,
            components.token_ids,
            components.embeds,
            params=params,
            lora_request=lora_request,
        )

    async def _get_trace_headers(
        self,
        headers: Headers,
    ) -> Mapping[str, str] | None:
        is_tracing_enabled = await self.engine_client.is_tracing_enabled()

        if is_tracing_enabled:
            return extract_trace_headers(headers)

        if contains_trace_headers(headers):
            log_tracing_disabled_warning()

        return None

    @staticmethod
    def _base_request_id(
        raw_request: Request | None, default: str | None = None
    ) -> str | None:
        """Pulls the request id to use from a header, if provided"""
        if raw_request is not None and (
            (req_id := raw_request.headers.get("X-Request-Id")) is not None
        ):
            return req_id

        return random_uuid() if default is None else default

    @staticmethod
    def _get_data_parallel_rank(raw_request: Request | None) -> int | None:
        """Pulls the data parallel rank from a header, if provided"""
        if raw_request is None:
            return None

        rank_str = raw_request.headers.get("X-data-parallel-rank")
        if rank_str is None:
            return None

        try:
            return int(rank_str)
        except ValueError:
            return None

    async def _with_kv_transfer_rejection_cleanup(
        self,
        awaitable: Awaitable[_T],
        request: ChatCompletionRequest | CompletionRequest | ResponsesRequest,
        raw_request: Request | None,
    ) -> _T:
        """Wrap a `create_*` coroutine so that, if it raises or returns an
        ErrorResponse (i.e. the request never reached the engine), the KV
        connector is notified to free any pinned remote-prefill blocks."""
        kv_transfer_params = self.has_kv_connector and request.kv_transfer_params
        if not kv_transfer_params or not kv_transfer_params.get("do_remote_prefill"):
            return await awaitable

        notify = True
        try:
            result = await awaitable
            if not isinstance(result, ErrorResponse):
                notify = False
            return result
        finally:
            if notify:
                try:
                    await self.engine_client.notify_kv_transfer_request_rejected(
                        request.request_id,
                        kv_transfer_params,
                        data_parallel_rank=self._get_data_parallel_rank(raw_request),
                    )
                except Exception:
                    logger.warning(
                        "Failed to notify KV connector about rejected request %s",
                        request.request_id,
                        exc_info=True,
                    )

    @staticmethod
    def _get_decoded_token(
        logprob: Logprob,
        token_id: int,
        tokenizer: TokenizerLike | None,
        return_as_token_id: bool = False,
    ) -> str:
        if return_as_token_id:
            return format_token_id_placeholder(token_id)

        if logprob.decoded_token is not None:
            return logprob.decoded_token

        if tokenizer is None:
            raise ValueError(
                "Unable to get tokenizer because `skip_tokenizer_init=True`"
            )

        return tokenizer.decode([token_id])

    def _is_model_supported(self, model_name: str | None) -> bool:
        if not model_name:
            return True
        if envs.VLLM_SKIP_MODEL_NAME_VALIDATION:
            return True
        return self.models.is_base_model(model_name)


def format_token_id_placeholder(token_id: int) -> str:
    return f"token_id:{token_id}"


def resolve_token_id_placeholder(
    token: str, tokenizer: TokenizerLike
) -> tuple[str, list[int] | None]:
    """Decode a 'token_id:N' placeholder back to a token string and UTF-8 bytes.

    Returns (token, None) unchanged if token is not a placeholder.
    This is the inverse of format_token_id_placeholder / _get_decoded_token
    when return_as_token_id=True.
    """
    suffix = token.removeprefix("token_id:")
    if suffix == token:
        return token, None
    try:
        token_id = int(suffix)
    except ValueError:
        return token, None
    token_repr = tokenizer.convert_ids_to_tokens([token_id])[0]
    if token_repr is None:
        logger.warning_once(
            "resolve_token_id_placeholder: token_id %d has no vocab entry; "
            "substituting empty string",
            token_id,
        )
        return "", None
    token_str = tokenizer.convert_tokens_to_string([token_repr])
    return token_str, list(token_str.encode("utf-8", errors="replace"))


def clamp_prompt_logprobs(
    prompt_logprobs: PromptLogprobs | None,
) -> PromptLogprobs | None:
    if prompt_logprobs is None:
        return prompt_logprobs

    for logprob_dict in prompt_logprobs:
        if logprob_dict is None:
            continue
        for logprob_values in logprob_dict.values():
            if logprob_values.logprob == float("-inf"):
                logprob_values.logprob = -9999.0
    return prompt_logprobs
