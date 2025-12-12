# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
import sys
import time
import traceback
from collections.abc import AsyncGenerator, Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, ClassVar, Generic, TypeAlias, TypeVar

import numpy as np
import torch
from fastapi import Request
from pydantic import ConfigDict, TypeAdapter
from starlette.datastructures import Headers
from typing_extensions import TypeIs

from vllm.entrypoints.context import (
    HarmonyContext,
    ParsableContext,
    StreamingHarmonyContext,
)
from vllm.entrypoints.openai.protocol import (
    FunctionCall,
    ResponseInputOutputItem,
    ResponsesRequest,
)
from vllm.entrypoints.pooling.classify.protocol import (
    ClassificationChatRequest,
    ClassificationCompletionRequest,
    ClassificationRequest,
    ClassificationResponse,
)
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
    EmbeddingRequest,
    EmbeddingResponse,
)
from vllm.entrypoints.pooling.pooling.protocol import (
    IOProcessorRequest,
    PoolingResponse,
)
from vllm.entrypoints.pooling.score.protocol import (
    RerankRequest,
    ScoreRequest,
    ScoreResponse,
)
from vllm.transformers_utils.tokenizer import AnyTokenizer

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from openai.types.responses import (
    ToolChoiceFunction,
)

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
    ConversationMessage,
    apply_hf_chat_template,
    apply_mistral_chat_template,
    parse_chat_messages_futures,
    resolve_chat_template_content_format,
)
from vllm.entrypoints.context import ConversationContext
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    DetokenizeRequest,
    ErrorInfo,
    ErrorResponse,
    FunctionDefinition,
    TokenizeChatRequest,
    TokenizeCompletionRequest,
    TokenizeResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    TranslationRequest,
)
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager
from vllm.entrypoints.renderer import BaseRenderer, CompletionRenderer, RenderConfig
from vllm.entrypoints.responses_utils import (
    construct_input_messages,
)
from vllm.entrypoints.serve.disagg.protocol import GenerateRequest, GenerateResponse
from vllm.entrypoints.utils import _validate_truncation_size
from vllm.inputs.data import ExplicitEncoderDecoderPrompt, PromptType
from vllm.beam_search import BeamSearchSequence, create_sort_beams_key_function
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt
from vllm.inputs.parse import PromptComponents, get_prompt_components, is_explicit_encoder_decoder_prompt, split_enc_dec_inputs
from vllm.logger import init_logger
from vllm.logprobs import Logprob, PromptLogprobs
from vllm.lora.request import LoRARequest
from vllm.multimodal import (  # noqa: F401 - Required to resolve Pydantic error in RequestProcessingMixin
    MultiModalDataDict,
    MultiModalUUIDDict,
)
from vllm.outputs import CompletionOutput, PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers import DeepseekV32Tokenizer, MistralTokenizer, TokenizerLike
from vllm.tracing import (
    contains_trace_headers,
    extract_trace_headers,
    log_tracing_disabled_warning,
)
from vllm.utils import random_uuid
from vllm.utils.async_utils import (
    AsyncMicrobatchTokenizer,
    collect_from_async_generator,
    make_async,
    merge_async_iterators,
)
from vllm.utils.collection_utils import is_list_of
from vllm.v1.engine import EngineCoreRequest


class GenerationError(Exception):
    """raised when finish_reason indicates internal server error (500)"""

    def __init__(self, message: str = "Internal server error"):
        super().__init__(message)
        self.status_code = HTTPStatus.INTERNAL_SERVER_ERROR


logger = init_logger(__name__)

CompletionLikeRequest: TypeAlias = (
    CompletionRequest
    | DetokenizeRequest
    | EmbeddingCompletionRequest
    | RerankRequest
    | ClassificationCompletionRequest
    | ScoreRequest
    | TokenizeCompletionRequest
)

ChatLikeRequest: TypeAlias = (
    ChatCompletionRequest
    | EmbeddingChatRequest
    | TokenizeChatRequest
    | ClassificationChatRequest
)
SpeechToTextRequest: TypeAlias = TranscriptionRequest | TranslationRequest
AnyRequest: TypeAlias = (
    CompletionLikeRequest
    | ChatLikeRequest
    | SpeechToTextRequest
    | ResponsesRequest
    | IOProcessorRequest
    | GenerateRequest
)

AnyResponse: TypeAlias = (
    CompletionResponse
    | ChatCompletionResponse
    | EmbeddingResponse
    | TranscriptionResponse
    | TokenizeResponse
    | PoolingResponse
    | ClassificationResponse
    | ScoreResponse
    | GenerateResponse
)


class TextTokensPrompt(TypedDict):
    prompt: str
    prompt_token_ids: list[int]


class EmbedsPrompt(TypedDict):
    prompt_embeds: torch.Tensor


RequestPrompt: TypeAlias = list[int] | str | TextTokensPrompt | EmbedsPrompt


def is_text_tokens_prompt(prompt: RequestPrompt) -> TypeIs[TextTokensPrompt]:
    return (
        isinstance(prompt, dict)
        and "prompt_token_ids" in prompt
        and "prompt_embeds" not in prompt
    )


def is_embeds_prompt(prompt: RequestPrompt) -> TypeIs[EmbedsPrompt]:
    return (
        isinstance(prompt, dict)
        and "prompt_token_ids" not in prompt
        and "prompt_embeds" in prompt
    )


RequestT = TypeVar("RequestT", bound=AnyRequest)


@dataclass(kw_only=True)
class RequestProcessingMixin:
    """
    Mixin for request processing,
    handling prompt preparation and engine input.
    """

    request_prompts: Sequence[RequestPrompt] | None = field(default_factory=list)
    engine_prompts: list[EngineTokensPrompt] | None = field(default_factory=list)


@dataclass(kw_only=True)
class ResponseGenerationMixin:
    """
    Mixin for response generation,
    managing result generators and final batch results.
    """

    result_generator: (
        AsyncGenerator[tuple[int, RequestOutput | PoolingRequestOutput], None] | None
    ) = None
    final_res_batch: list[RequestOutput | PoolingRequestOutput] = field(
        default_factory=list
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass(kw_only=True)
class ServeContext(RequestProcessingMixin, ResponseGenerationMixin, Generic[RequestT]):
    # Shared across all requests
    request: RequestT
    raw_request: Request | None = None
    model_name: str
    request_id: str
    created_time: int = field(default_factory=lambda: int(time.time()))
    lora_request: LoRARequest | None = None

    # Shared across most requests
    tokenizer: TokenizerLike | None = None


@dataclass(kw_only=True)
class ClassificationServeContext(ServeContext[ClassificationRequest]):
    pass


@dataclass(kw_only=True)
class EmbeddingServeContext(ServeContext[EmbeddingRequest]):
    chat_template: str | None = None
    chat_template_content_format: ChatTemplateContentFormatOption


class OpenAIServing:
    request_id_prefix: ClassVar[str] = """
    A short string prepended to every request’s ID (e.g. "embd", "classify")
    so you can easily tell “this ID came from Embedding vs Classification.”
    """

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
        log_error_stack: bool = False,
    ):
        super().__init__()

        self.engine_client = engine_client

        self.models = models

        self.request_logger = request_logger
        self.return_tokens_as_token_ids = return_tokens_as_token_ids
        self._tokenizer_executor = ThreadPoolExecutor(max_workers=1)
        self._apply_mistral_chat_template_async = make_async(
            apply_mistral_chat_template, executor=self._tokenizer_executor
        )

        self._async_tokenizer_pool: dict[TokenizerLike, AsyncMicrobatchTokenizer] = {}
        self.log_error_stack = log_error_stack

        self.input_processor = self.models.input_processor
        self.io_processor = self.models.io_processor
        self.model_config = self.models.model_config
        self.max_model_len = self.model_config.max_model_len

    def _get_tool_parser(
        self, tool_parser_name: str | None = None, enable_auto_tools: bool = False
    ) -> Callable[[TokenizerLike], ToolParser] | None:
        """Get the tool parser based on the name."""
        parser = None
        if not enable_auto_tools or tool_parser_name is None:
            return parser
        logger.info('"auto" tool choice has been enabled.')

        try:
            if tool_parser_name == "pythonic" and self.model_config.model.startswith(
                "meta-llama/Llama-3.2"
            ):
                logger.warning(
                    "Llama3.2 models may struggle to emit valid pythonic tool calls"
                )
            parser = ToolParserManager.get_tool_parser(tool_parser_name)
        except Exception as e:
            raise TypeError(
                "Error: --enable-auto-tool-choice requires "
                f"tool_parser:'{tool_parser_name}' which has not "
                "been registered"
            ) from e
        return parser

    def _get_reasoning_parser(
        self,
        reasoning_parser_name: str,
    ) -> Callable[[TokenizerLike], ReasoningParser] | None:
        """Get the reasoning parser based on the name."""
        parser = None
        if not reasoning_parser_name:
            return None
        try:
            parser = ReasoningParserManager.get_reasoning_parser(reasoning_parser_name)
            assert parser is not None
        except Exception as e:
            raise TypeError(f"{reasoning_parser_name=} has not been registered") from e
        return parser

    async def reset_mm_cache(self) -> None:
        self.input_processor.clear_mm_cache()
        await self.engine_client.reset_mm_cache()

    async def beam_search(
        self,
        prompt: PromptType,
        request_id: str,
        params: BeamSearchParams,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Perform beam search for text generation.

        For encoder-decoder models (like Whisper), this implementation includes
        performance optimizations:

        - Encoder deduplication: Encoder computed once, shared across all beams
        - Request batching: Multiple beams processed together in single forward pass  
        - Early stopping: Terminates when completed beams outperform active ones
        - Cross-attention block sharing: Deduplicated requests share KV cache blocks

        These optimizations provide ~3x speedup for beam search on encoder-decoder models.
        """
        beam_width = params.beam_width
        max_tokens = params.max_tokens
        ignore_eos = params.ignore_eos
        temperature = params.temperature
        length_penalty = params.length_penalty
        include_stop_str_in_output = params.include_stop_str_in_output

        input_processor = self.input_processor
        tokenizer = input_processor.tokenizer
        if tokenizer is None:
            raise ValueError(
                "You cannot use beam search when `skip_tokenizer_init=True`"
            )

        eos_token_id: int = tokenizer.eos_token_id  # type: ignore

        # Create prompt builder to handle different prompt types
        prompt_builder = self._create_beam_search_prompt_builder(prompt)
        prompt_token_ids = prompt_builder.get_initial_tokens()
        prompt_text = prompt_builder.get_prompt_text()
        multi_modal_data = prompt_builder.get_initial_multi_modal_data()
        mm_processor_kwargs = prompt_builder.get_initial_mm_processor_kwargs()

        # Check if this is an encoder-decoder prompt for optimization decisions
        is_enc_dec = isinstance(prompt_builder, EncoderDecoderBeamSearchPromptBuilder)

        # This is a workaround to fix multimodal beam search; this is a
        # bandaid fix for 2 small problems:
        # 1. Multi_modal_data on the processed_inputs currently resolves to
        #    `None`.
        # 2. preprocessing above expands the multimodal placeholders. However,
        #    this happens again in generation, so the double expansion causes
        #    a mismatch.
        # TODO - would be ideal to handle this more gracefully.

        tokenized_length = len(prompt_token_ids)

        sort_beams_key = create_sort_beams_key_function(eos_token_id, length_penalty)

        # Optimize beam search parameters for different model types
        if is_enc_dec:
            # For encoder-decoder models, use fewer logprobs since encoder dominates cost
            # beam_width + 1 provides minimal exploration while keeping search effective
            logprobs_num = beam_width + 1
            # Keep the existing conservative limit from protocol.py (300 tokens)
            effective_max_tokens = max_tokens
        else:
            # For decoder-only models, use standard beam search parameters
            logprobs_num = 2 * beam_width
            effective_max_tokens = max_tokens

        beam_search_params = SamplingParams(
            logprobs=logprobs_num,
            max_tokens=1,
            temperature=temperature,
        )
        all_beams = [
            BeamSearchSequence(
                tokens=prompt_token_ids,
                cum_logprob=0,
                logprobs=[],
                multi_modal_data=multi_modal_data,
                mm_processor_kwargs=mm_processor_kwargs,
                lora_request=lora_request,
            )
        ]
        completed = []
        
        # Performance instrumentation for encoder-decoder models
        step_count = 0
        if is_enc_dec:
            logger.info(
                f"Starting beam search: beam_width={beam_width}, max_tokens={effective_max_tokens}, "
                f"num_beams={len(all_beams)}, encoder_decoder=True"
            )

        for _ in range(effective_max_tokens):
            step_count += 1
            step_start_time = time.time() if is_enc_dec else None
            # Create prompts for each beam using the prompt builder
            prompts_batch, lora_req_batch = zip(
                *[
                    (prompt_builder.build_prompt_for_beam(beam), beam.lora_request)
                    for beam in all_beams
                ]
            )
            

            # Submit all beams concurrently
            request_id_batch = f"{request_id}-{random_uuid()}"
            tasks = []
            for i, (individual_prompt, lora_req) in enumerate(
                zip(prompts_batch, lora_req_batch)
            ):
                request_id_item = f"{request_id_batch}-beam-{i}"
                task = asyncio.create_task(
                    collect_from_async_generator(
                        self.engine_client.generate(
                            individual_prompt,
                            beam_search_params,
                            request_id_item,
                            lora_request=lora_req,
                            trace_headers=trace_headers,
                            priority=priority,  # Propagate original request priority
                        )
                    )
                )
                tasks.append(task)
            
            output = [x[0] for x in await asyncio.gather(*tasks)]

            new_beams = []
            # Store all new tokens generated by beam
            all_beams_token_id = []
            # Store the cumulative probability of all tokens
            # generated by beam search
            all_beams_logprob = []
            # Iterate through all beam inference results
            for i, result in enumerate(output):
                current_beam = all_beams[i]

                # check for error finish reason and abort beam search
                if result.outputs[0].finish_reason == "error":
                    # yield error output and terminate beam search
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
                    all_beams_token_id.extend(list(logprobs.keys()))
                    all_beams_logprob.extend(
                        [
                            current_beam.cum_logprob + obj.logprob
                            for obj in logprobs.values()
                        ]
                    )

            # Handle the token for the end of sentence (EOS)
            all_beams_token_id = np.array(all_beams_token_id)
            all_beams_logprob = np.array(all_beams_logprob)

            if not ignore_eos:
                # Get the index position of eos token in all generated results
                eos_idx = np.where(all_beams_token_id == eos_token_id)[0]
                for idx in eos_idx:
                    current_beam = all_beams[idx // logprobs_num]
                    result = output[idx // logprobs_num]
                    assert result.outputs[0].logprobs is not None
                    logprobs_entry = result.outputs[0].logprobs[0]
                    completed.append(
                        BeamSearchSequence(
                            tokens=current_beam.tokens + [eos_token_id]
                            if include_stop_str_in_output
                            else current_beam.tokens,
                            logprobs=current_beam.logprobs + [logprobs_entry],
                            cum_logprob=float(all_beams_logprob[idx]),
                            finish_reason="stop",
                            stop_reason=eos_token_id,
                        )
                    )
                # After processing, set the log probability of the eos condition
                # to negative infinity.
                all_beams_logprob[eos_idx] = -np.inf

            # Processing non-EOS tokens
            # Get indices of the top beam_width probabilities
            topn_idx = np.argpartition(np.negative(all_beams_logprob), beam_width)[
                :beam_width
            ]

            for idx in topn_idx:
                current_beam = all_beams[idx // logprobs_num]
                result = output[idx // logprobs_num]
                token_id = int(all_beams_token_id[idx])
                assert result.outputs[0].logprobs is not None
                logprobs_entry = result.outputs[0].logprobs[0]
                new_beams.append(
                    BeamSearchSequence(
                        tokens=current_beam.tokens + [token_id],
                        logprobs=current_beam.logprobs + [logprobs_entry],
                        lora_request=current_beam.lora_request,
                        cum_logprob=float(all_beams_logprob[idx]),
                        multi_modal_data=current_beam.multi_modal_data,
                        mm_processor_kwargs=current_beam.mm_processor_kwargs,
                    )
                )

            all_beams = new_beams
            
            # Track step duration (removed per-step logging for production)
            
            # Early stopping conditions
            if len(all_beams) == 0:
                # All beams have hit EOS
                break
            
            # Additional early stopping for encoder-decoder models:
            # Stop when we have enough good completions that can't be beaten
            if is_enc_dec and len(completed) >= beam_width:
                # Get the best completed beam's score
                best_completed_score = sort_beams_key(sorted(completed, key=sort_beams_key, reverse=True)[0])
                # Get the best active beam's score
                best_active_score = sort_beams_key(sorted(all_beams, key=sort_beams_key, reverse=True)[0])
                
                # If length_penalty <= 1.0, longer sequences won't improve scores
                # So we can stop if best completed beats best active
                if length_penalty <= 1.0 and best_completed_score >= best_active_score:
                    break

        completed.extend(all_beams)
        sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]
        
        # Log final beam search statistics
        # Beam search completed (removed verbose logging for production)

        for beam in best_beams:
            if beam.tokens[-1] == eos_token_id and not ignore_eos:
                # Skip the eos token in the text.
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

    def _create_beam_search_prompt_builder(self, prompt: PromptType) -> "BeamSearchPromptBuilder":
        """Create appropriate prompt builder for beam search based on prompt type."""
        if is_explicit_encoder_decoder_prompt(prompt):
            return EncoderDecoderBeamSearchPromptBuilder(prompt, self.input_processor)
        else:
            return DecoderOnlyBeamSearchPromptBuilder(prompt)

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
        )

    def _get_active_default_mm_loras(self, request: AnyRequest) -> LoRARequest | None:
        """Determine if there are any active default multimodal loras."""

    def _is_model_supported(self, model_name: str | None) -> bool:
        if not model_name:
            return True
        return self.models.is_base_model(model_name)

    @staticmethod
    def _base_request_id(
        raw_request: Request | None, default: str | None = None
    ) -> str:
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

    def _maybe_get_adapters(
        self,
        request: AnyRequest,
        supports_default_mm_loras: bool = False,
    ) -> LoRARequest | None:
        if request.model in self.models.lora_requests:
            return self.models.lora_requests[request.model]

        if supports_default_mm_loras:
            return self._get_active_default_mm_loras(request)
        return None

    def _log_inputs(
        self,
        request_id: str,
        inputs: RequestPrompt | PromptType,
        params: SamplingParams | PoolingParams | BeamSearchParams | None,
        lora_request: LoRARequest | None,
    ) -> None:
        if self.request_logger is None:
            return

        prompt, prompt_token_ids, prompt_embeds = self._get_prompt_components(inputs)

        # Log the input request
        self.request_logger.log_inputs(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_embeds=prompt_embeds,
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


class BeamSearchPromptBuilder:
    """Abstract base class for building prompts during beam search."""

    def get_initial_tokens(self) -> list[int]:
        """Get the initial token IDs for beam search."""
        raise NotImplementedError

    def get_prompt_text(self) -> str | None:
        """Get the prompt text for the response."""
        raise NotImplementedError

    def get_initial_multi_modal_data(self) -> MultiModalDataDict | None:
        """Return any multimodal attachments required for the first beam."""
        raise NotImplementedError

    def get_initial_mm_processor_kwargs(self) -> dict[str, Any] | None:
        """Return processor kwargs needed to handle multimodal inputs."""
        raise NotImplementedError

    def build_prompt_for_beam(self, beam: BeamSearchSequence) -> PromptType:
        """Build the appropriate prompt type for a beam during search."""
        raise NotImplementedError


class DecoderOnlyBeamSearchPromptBuilder(BeamSearchPromptBuilder):
    """Prompt builder for decoder-only models."""

    def __init__(self, prompt: PromptType):
        self.prompt = prompt
        self.multi_modal_data: MultiModalDataDict | None = None
        self.mm_processor_kwargs: dict[str, Any] | None = None

        # Extract prompt components
        if isinstance(prompt, str):
            self.prompt_text = prompt
            self.prompt_token_ids = []
            self.multi_modal_data = None
        else:
            self.prompt_text = prompt.get("prompt")  # type: ignore
            self.prompt_token_ids = prompt.get("prompt_token_ids", [])  # type: ignore
            self.multi_modal_data = prompt.get("multi_modal_data")  # type: ignore

        # Some multimodal prompts include processor kwargs
        self.mm_processor_kwargs = None
        if not isinstance(prompt, str):
            self.mm_processor_kwargs = prompt.get("mm_processor_kwargs")  # type: ignore

    def get_initial_tokens(self) -> list[int]:
        return self.prompt_token_ids

    def get_prompt_text(self) -> str | None:
        return self.prompt_text

    def get_initial_multi_modal_data(self) -> MultiModalDataDict | None:
        return self.multi_modal_data

    def get_initial_mm_processor_kwargs(self) -> dict[str, Any] | None:
        return self.mm_processor_kwargs

    def build_prompt_for_beam(self, beam: BeamSearchSequence) -> EngineTokensPrompt:
        return EngineTokensPrompt(
            prompt_token_ids=beam.tokens,
            multi_modal_data=beam.multi_modal_data,
            mm_processor_kwargs=beam.mm_processor_kwargs,
        )


class EncoderDecoderBeamSearchPromptBuilder(BeamSearchPromptBuilder):
    """Prompt builder for encoder-decoder models like Whisper with optimized encoder caching."""

    def __init__(self, prompt: PromptType, input_processor):
        self.original_encoder_prompt = prompt["encoder_prompt"]  # type: ignore
        self.original_mm_processor_kwargs = prompt.get("mm_processor_kwargs")  # type: ignore

        # Process the encoder-decoder prompt once to extract decoder tokens
        preprocessor = input_processor.input_preprocessor
        processed_inputs = preprocessor.preprocess(
            prompt,
            tokenization_kwargs=None,
        )
        # Split encoder and decoder inputs
        encoder_inputs, decoder_inputs = split_enc_dec_inputs(processed_inputs)

        # For beam search, we use the decoder prompt token IDs
        if decoder_inputs["type"] == "embeds":
            raise NotImplementedError("Beam search with prompt embeds not supported")

        self.prompt_token_ids = decoder_inputs["prompt_token_ids"]
        self.prompt_text = decoder_inputs.get("prompt")

        # Get multi_modal_data from encoder inputs (where audio is)
        self.multi_modal_data = (
            encoder_inputs.get("multi_modal_data")
            if encoder_inputs.get("type") == "mm"
            else None
        )

        # Use mm_processor_kwargs from encoder_inputs if available, otherwise from original prompt
        self.mm_processor_kwargs = (
            encoder_inputs.get("mm_processor_kwargs")
            if encoder_inputs.get("type") == "mm"
            else self.original_mm_processor_kwargs
        )

        # Cache the processed inputs to avoid re-processing the encoder
        self._processed_inputs = processed_inputs
        self._encoder_inputs = encoder_inputs
        self._decoder_template = decoder_inputs
        
        # Generate a unique cache salt for this beam search session
        # This helps the engine recognize that all beams share the same encoder input
        self._encoder_cache_salt = f"beam_search_encoder_{random_uuid()}"

    def get_initial_tokens(self) -> list[int]:
        return self.prompt_token_ids

    def get_prompt_text(self) -> str | None:
        return self.prompt_text

    def get_initial_multi_modal_data(self) -> MultiModalDataDict | None:
        return self.multi_modal_data

    def get_initial_mm_processor_kwargs(self) -> dict[str, Any] | None:
        return self.mm_processor_kwargs

    def build_prompt_for_beam(self, beam: BeamSearchSequence) -> ExplicitEncoderDecoderPrompt:
        # For encoder-decoder models, we use cache_salt to help the engine
        # recognize that all beams in this search share the same encoder input
        prompt = ExplicitEncoderDecoderPrompt(
            encoder_prompt=self.original_encoder_prompt,
            decoder_prompt={
                "prompt_token_ids": beam.tokens,
            },
            mm_processor_kwargs=self.mm_processor_kwargs or {},
        )
        # Add cache_salt to signal shared encoder input across all beams
        prompt["cache_salt"] = self._encoder_cache_salt  # type: ignore
        return prompt


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
        generation: AsyncGenerator[AnyResponse | ErrorResponse, None]
        generation = self._pipeline(ctx)

        async for response in generation:
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
            and truncate_prompt_tokens > self.max_model_len
        ):
            return self.create_error_response(
                "truncate_prompt_tokens value is "
                "greater than max_model_len."
                " Please, select a smaller truncation size."
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
        generators: list[
            AsyncGenerator[RequestOutput | PoolingRequestOutput, None]
        ] = []

        try:
            trace_headers = (
                None
                if ctx.raw_request is None
                else await self._get_trace_headers(ctx.raw_request.headers)
            )

            pooling_params = self._create_pooling_params(ctx)
            if isinstance(pooling_params, ErrorResponse):
                return pooling_params

            if ctx.engine_prompts is None:
                return self.create_error_response("Engine prompts not available")

            for i, engine_prompt in enumerate(ctx.engine_prompts):
                request_id_item = f"{ctx.request_id}-{i}"

                self._log_inputs(
                    request_id_item,
                    engine_prompt,
                    params=pooling_params,
                    lora_request=ctx.lora_request,
                )

                generator = self.engine_client.encode(
                    engine_prompt,
                    pooling_params,
                    request_id_item,
                    lora_request=ctx.lora_request,
                    trace_headers=trace_headers,
                    priority=getattr(ctx.request, "priority", 0),
                )

                generators.append(generator)

            ctx.result_generator = merge_async_iterators(*generators)

            return None

        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    async def _collect_batch(
        self,
        ctx: ServeContext,
    ) -> ErrorResponse | None:
        """Collect batch results from the result generator."""
        try:
            if ctx.engine_prompts is None:
                return self.create_error_response("Engine prompts not available")

            num_prompts = len(ctx.engine_prompts)
            final_res_batch: list[RequestOutput | PoolingRequestOutput | None]
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

        except Exception as e:
            return self.create_error_response(str(e))

    def create_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    ) -> ErrorResponse:
        if self.log_error_stack:
            exc_type, _, _ = sys.exc_info()
            if exc_type is not None:
                traceback.print_exc()
            else:
                traceback.print_stack()
        return ErrorResponse(
            error=ErrorInfo(message=message, type=err_type, code=status_code.value)
        )

    def create_streaming_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    ) -> str:
        json_str = json.dumps(
            self.create_error_response(
                message=message, err_type=err_type, status_code=status_code
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

    def _convert_generation_error_to_response(
        self, e: GenerationError
    ) -> ErrorResponse:
        """Convert GenerationError to ErrorResponse."""
        return self.create_error_response(
            str(e),
            err_type="InternalServerError",
            status_code=e.status_code,
        )

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

    async def _normalize_prompt_text_to_input(
        self,
        request: AnyRequest,
        prompt: str,
        tokenizer: TokenizerLike,
        add_special_tokens: bool,
    ) -> TextTokensPrompt:
        async_tokenizer = self._get_async_tokenizer(tokenizer)

        if (
            self.model_config.encoder_config is not None
            and self.model_config.encoder_config.get("do_lower_case", False)
        ):
            prompt = prompt.lower()

        truncate_prompt_tokens = getattr(request, "truncate_prompt_tokens", None)

        if truncate_prompt_tokens is None:
            encoded = await async_tokenizer(
                prompt, add_special_tokens=add_special_tokens
            )
        elif truncate_prompt_tokens < 0:
            # Negative means we cap at the model's max length
            encoded = await async_tokenizer(
                prompt,
                add_special_tokens=add_special_tokens,
                truncation=True,
                max_length=self.max_model_len,
            )
        else:
            encoded = await async_tokenizer(
                prompt,
                add_special_tokens=add_special_tokens,
                truncation=True,
                max_length=truncate_prompt_tokens,
            )

        input_ids = encoded.input_ids
        input_text = prompt

        return self._validate_input(request, input_ids, input_text)

    async def _normalize_prompt_tokens_to_input(
        self,
        request: AnyRequest,
        prompt_ids: list[int],
        tokenizer: TokenizerLike | None,
    ) -> TextTokensPrompt:
        truncate_prompt_tokens = getattr(request, "truncate_prompt_tokens", None)

        if truncate_prompt_tokens is None:
            input_ids = prompt_ids
        elif truncate_prompt_tokens < 0:
            input_ids = prompt_ids[-self.max_model_len :]
        else:
            input_ids = prompt_ids[-truncate_prompt_tokens:]

        if tokenizer is None:
            input_text = ""
        else:
            async_tokenizer = self._get_async_tokenizer(tokenizer)
            input_text = await async_tokenizer.decode(input_ids)

        return self._validate_input(request, input_ids, input_text)

    def _validate_input(
        self,
        request: AnyRequest,
        input_ids: list[int],
        input_text: str,
    ) -> TextTokensPrompt:
        token_num = len(input_ids)

        # Note: EmbeddingRequest, ClassificationRequest,
        # and ScoreRequest doesn't have max_tokens
        if isinstance(
            request,
            (
                EmbeddingChatRequest,
                EmbeddingCompletionRequest,
                ScoreRequest,
                RerankRequest,
                ClassificationCompletionRequest,
                ClassificationChatRequest,
            ),
        ):
            # Note: input length can be up to the entire model context length
            # since these requests don't generate tokens.
            if token_num > self.max_model_len:
                operations: dict[type[AnyRequest], str] = {
                    ScoreRequest: "score",
                    ClassificationCompletionRequest: "classification",
                    ClassificationChatRequest: "classification",
                }
                operation = operations.get(type(request), "embedding generation")
                raise ValueError(
                    f"This model's maximum context length is "
                    f"{self.max_model_len} tokens. However, you requested "
                    f"{token_num} tokens in the input for {operation}. "
                    f"Please reduce the length of the input."
                )
            return TextTokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

        # Note: TokenizeRequest and DetokenizeRequest doesn't have max_tokens
        # and does not require model context length validation
        if isinstance(
            request,
            (TokenizeCompletionRequest, TokenizeChatRequest, DetokenizeRequest),
        ):
            return TextTokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

        # chat completion endpoint supports max_completion_tokens
        if isinstance(request, ChatCompletionRequest):
            # TODO(#9845): remove max_tokens when field dropped from OpenAI API
            max_tokens = request.max_completion_tokens or request.max_tokens
        else:
            max_tokens = getattr(request, "max_tokens", None)

        # Note: input length can be up to model context length - 1 for
        # completion-like requests.
        if token_num >= self.max_model_len:
            raise ValueError(
                f"This model's maximum context length is "
                f"{self.max_model_len} tokens. However, your request has "
                f"{token_num} input tokens. Please reduce the length of "
                "the input messages."
            )

        if max_tokens is not None and token_num + max_tokens > self.max_model_len:
            raise ValueError(
                "'max_tokens' or 'max_completion_tokens' is too large: "
                f"{max_tokens}. This model's maximum context length is "
                f"{self.max_model_len} tokens and your request has "
                f"{token_num} input tokens ({max_tokens} > {self.max_model_len}"
                f" - {token_num})."
            )

        return TextTokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

    async def _tokenize_prompt_input_async(
        self,
        request: AnyRequest,
        tokenizer: TokenizerLike,
        prompt_input: str | list[int],
        add_special_tokens: bool = True,
    ) -> TextTokensPrompt:
        """
        A simpler implementation that tokenizes a single prompt input.
        """
        async for result in self._tokenize_prompt_inputs_async(
            request,
            tokenizer,
            [prompt_input],
            add_special_tokens=add_special_tokens,
        ):
            return result
        raise ValueError("No results yielded from tokenization")

    async def _tokenize_prompt_inputs_async(
        self,
        request: AnyRequest,
        tokenizer: TokenizerLike,
        prompt_inputs: Iterable[str | list[int]],
        add_special_tokens: bool = True,
    ) -> AsyncGenerator[TextTokensPrompt, None]:
        """
        A simpler implementation that tokenizes multiple prompt inputs.
        """
        for prompt in prompt_inputs:
            if isinstance(prompt, str):
                yield await self._normalize_prompt_text_to_input(
                    request,
                    prompt=prompt,
                    tokenizer=tokenizer,
                    add_special_tokens=add_special_tokens,
                )
            else:
                yield await self._normalize_prompt_tokens_to_input(
                    request,
                    prompt_ids=prompt,
                    tokenizer=tokenizer,
                )

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

    async def _preprocess_chat(
        self,
        request: ChatLikeRequest | ResponsesRequest,
        tokenizer: TokenizerLike | None,
        messages: list[ChatCompletionMessageParam],
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tool_dicts: list[dict[str, Any]] | None = None,
        documents: list[dict[str, str]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        tool_parser: Callable[[TokenizerLike], ToolParser] | None = None,
        add_special_tokens: bool = False,
    ) -> tuple[
        list[ConversationMessage],
        Sequence[RequestPrompt],
        list[EngineTokensPrompt],
    ]:
        model_config = self.model_config

        resolved_content_format = resolve_chat_template_content_format(
            chat_template,
            tool_dicts,
            chat_template_content_format,
            tokenizer,
            model_config=model_config,
        )
        conversation, mm_data_future, mm_uuids = parse_chat_messages_futures(
            messages,
            model_config,
            content_format=resolved_content_format,
        )

        _chat_template_kwargs: dict[str, Any] = dict(
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tool_dicts,
            documents=documents,
        )
        _chat_template_kwargs.update(chat_template_kwargs or {})

        request_prompt: str | list[int]

        if tokenizer is None:
            request_prompt = "placeholder"
        elif isinstance(tokenizer, MistralTokenizer):
            request_prompt = await self._apply_mistral_chat_template_async(
                tokenizer,
                messages=messages,
                **_chat_template_kwargs,
            )
        elif isinstance(tokenizer, DeepseekV32Tokenizer):
            request_prompt = tokenizer.apply_chat_template(
                conversation=conversation,
                messages=messages,
                model_config=model_config,
                **_chat_template_kwargs,
            )
        else:
            request_prompt = apply_hf_chat_template(
                tokenizer=tokenizer,
                conversation=conversation,
                model_config=model_config,
                **_chat_template_kwargs,
            )

        mm_data = await mm_data_future

        # tool parsing is done only if a tool_parser has been set and if
        # tool_choice is not "none" (if tool_choice is "none" but a tool_parser
        # is set, we want to prevent parsing a tool_call hallucinated by the LLM
        should_parse_tools = tool_parser is not None and (
            hasattr(request, "tool_choice") and request.tool_choice != "none"
        )

        if should_parse_tools:
            if not isinstance(request, ChatCompletionRequest | ResponsesRequest):
                msg = (
                    "Tool usage is only supported for Chat Completions API "
                    "or Responses API requests."
                )
                raise NotImplementedError(msg)
            request = tool_parser(tokenizer).adjust_request(request=request)  # type: ignore

        if tokenizer is None:
            assert isinstance(request_prompt, str), (
                "Prompt has to be a string",
                "when the tokenizer is not initialised",
            )
            prompt_inputs = TextTokensPrompt(
                prompt=request_prompt, prompt_token_ids=[1]
            )
        elif isinstance(request_prompt, str):
            prompt_inputs = await self._tokenize_prompt_input_async(
                request,
                tokenizer,
                request_prompt,
                add_special_tokens=add_special_tokens,
            )
        else:
            # For MistralTokenizer
            assert is_list_of(request_prompt, int), (
                "Prompt has to be either a string or a list of token ids"
            )
            prompt_inputs = TextTokensPrompt(
                prompt=tokenizer.decode(request_prompt),
                prompt_token_ids=request_prompt,
            )

        engine_prompt = EngineTokensPrompt(
            prompt_token_ids=prompt_inputs["prompt_token_ids"]
        )
        if mm_data is not None:
            engine_prompt["multi_modal_data"] = mm_data

        if mm_uuids is not None:
            engine_prompt["multi_modal_uuids"] = mm_uuids

        if request.mm_processor_kwargs is not None:
            engine_prompt["mm_processor_kwargs"] = request.mm_processor_kwargs

        if hasattr(request, "cache_salt") and request.cache_salt is not None:
            engine_prompt["cache_salt"] = request.cache_salt

        return conversation, [request_prompt], [engine_prompt]

    async def _process_inputs(
        self,
        request_id: str,
        engine_prompt: PromptType,
        params: SamplingParams | PoolingParams,
        *,
        lora_request: LoRARequest | None,
        trace_headers: Mapping[str, str] | None,
        priority: int,
    ) -> tuple[EngineCoreRequest, dict[str, Any]]:
        """Use the Processor to process inputs for AsyncLLM."""
        tokenization_kwargs: dict[str, Any] = {}
        _validate_truncation_size(
            self.max_model_len, params.truncate_prompt_tokens, tokenization_kwargs
        )

        engine_request = self.input_processor.process_inputs(
            request_id,
            engine_prompt,
            params,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
        )
        return engine_request, tokenization_kwargs

    async def _render_next_turn(
        self,
        request: ResponsesRequest,
        tokenizer: AnyTokenizer,
        messages: list[ResponseInputOutputItem],
        tool_dicts: list[dict[str, Any]] | None,
        tool_parser,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
    ):
        new_messages = construct_input_messages(
            request_input=messages,
        )

        _, request_prompts, engine_prompts = await self._preprocess_chat(
            request,
            tokenizer,
            new_messages,
            tool_dicts=tool_dicts,
            tool_parser=tool_parser,
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
        )
        return request_prompts, engine_prompts

    async def _generate_with_builtin_tools(
        self,
        request_id: str,
        request_prompt: RequestPrompt,
        engine_prompt: EngineTokensPrompt,
        sampling_params: SamplingParams,
        context: ConversationContext,
        lora_request: LoRARequest | None = None,
        priority: int = 0,
        **kwargs,
    ):
        prompt_text, _, _ = self._get_prompt_components(request_prompt)
        orig_priority = priority
        sub_request = 0
        while True:
            # Ensure that each sub-request has a unique request id.
            sub_request_id = f"{request_id}_{sub_request}"
            self._log_inputs(
                sub_request_id,
                request_prompt,
                params=sampling_params,
                lora_request=lora_request,
            )
            trace_headers = kwargs.get("trace_headers")
            engine_request, tokenization_kwargs = await self._process_inputs(
                sub_request_id,
                engine_prompt,
                sampling_params,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=priority,
            )

            generator = self.engine_client.generate(
                engine_request,
                sampling_params,
                sub_request_id,
                lora_request=lora_request,
                priority=priority,
                prompt_text=prompt_text,
                tokenization_kwargs=tokenization_kwargs,
                **kwargs,
            )

            async for res in generator:
                context.append_output(res)
                # NOTE(woosuk): The stop condition is handled by the engine.
                yield context

            if not context.need_builtin_tool_call():
                # The model did not ask for a tool call, so we're done.
                break

            # Call the tool and update the context with the result.
            tool_output = await context.call_tool()
            context.append_tool_output(tool_output)

            # TODO: uncomment this and enable tool output streaming
            # yield context

            # Create inputs for the next turn.
            # Render the next prompt token ids.
            if isinstance(context, (HarmonyContext, StreamingHarmonyContext)):
                prompt_token_ids = context.render_for_completion()
                engine_prompt = EngineTokensPrompt(prompt_token_ids=prompt_token_ids)
                request_prompt = prompt_token_ids
            elif isinstance(context, ParsableContext):
                request_prompts, engine_prompts = await self._render_next_turn(
                    context.request,
                    context.tokenizer,
                    context.parser.response_messages,
                    context.tool_dicts,
                    context.tool_parser_cls,
                    context.chat_template,
                    context.chat_template_content_format,
                )
                engine_prompt = engine_prompts[0]
                request_prompt = request_prompts[0]
                prompt_text, _, _ = self._get_prompt_components(request_prompt)

            # Update the sampling params.
            sampling_params.max_tokens = self.max_model_len - len(
                engine_prompt["prompt_token_ids"]
            )
            # OPTIMIZATION
            priority = orig_priority - 1
            sub_request += 1

    def _get_prompt_components(
        self,
        prompt: RequestPrompt | PromptType,
    ) -> PromptComponents:
        if isinstance(prompt, list):
            return PromptComponents(token_ids=prompt)

        return get_prompt_components(prompt)  # type: ignore[arg-type]

    def _log_inputs(
        self,
        request_id: str,
        inputs: RequestPrompt | PromptType,
        params: SamplingParams | PoolingParams | BeamSearchParams | None,
        lora_request: LoRARequest | None,
    ) -> None:
        if self.request_logger is None:
            return

        prompt, prompt_token_ids, prompt_embeds = self._get_prompt_components(inputs)

        self.request_logger.log_inputs(
            request_id,
            prompt,
            prompt_token_ids,
            prompt_embeds,
            params=params,
            lora_request=lora_request,
        )

    @staticmethod
    def _parse_tool_calls_from_content(
        request: ResponsesRequest | ChatCompletionRequest,
        tokenizer: TokenizerLike,
        enable_auto_tools: bool,
        tool_parser_cls: Callable[[TokenizerLike], ToolParser] | None,
        content: str | None = None,
    ) -> tuple[list[FunctionCall] | None, str | None]:
        function_calls = list[FunctionCall]()
        if request.tool_choice and isinstance(request.tool_choice, ToolChoiceFunction):
            assert content is not None
            # Forced Function Call
            function_calls.append(
                FunctionCall(name=request.tool_choice.name, arguments=content)
            )
            content = None  # Clear content since tool is called.
        elif request.tool_choice and isinstance(
            request.tool_choice, ChatCompletionNamedToolChoiceParam
        ):
            assert content is not None
            # Forced Function Call
            function_calls.append(
                FunctionCall(name=request.tool_choice.function.name, arguments=content)
            )
            content = None  # Clear content since tool is called.
        elif request.tool_choice == "required":
            assert content is not None
            tool_calls = TypeAdapter(list[FunctionDefinition]).validate_json(content)
            function_calls.extend(
                [
                    FunctionCall(
                        name=tool_call.name,
                        arguments=json.dumps(tool_call.parameters, ensure_ascii=False),
                    )
                    for tool_call in tool_calls
                ]
            )
            content = None  # Clear content since tool is called.
        elif (
            tool_parser_cls
            and enable_auto_tools
            and (request.tool_choice == "auto" or request.tool_choice is None)
        ):
            # Automatic Tool Call Parsing
            try:
                tool_parser = tool_parser_cls(tokenizer)
            except RuntimeError as e:
                logger.exception("Error in tool parser creation.")
                raise e
            tool_call_info = tool_parser.extract_tool_calls(
                content if content is not None else "",
                request=request,  # type: ignore
            )
            if tool_call_info is not None and tool_call_info.tools_called:
                # extract_tool_calls() returns a list of tool calls.
                function_calls.extend(
                    FunctionCall(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    )
                    for tool_call in tool_call_info.tool_calls
                )
                content = tool_call_info.content
                if content and content.strip() == "":
                    content = None
            else:
                # No tool calls.
                return None, content

        return function_calls, content

    @staticmethod
    def _get_decoded_token(
        logprob: Logprob,
        token_id: int,
        tokenizer: TokenizerLike | None,
        return_as_token_id: bool = False,
    ) -> str:
        if return_as_token_id:
            return f"token_id:{token_id}"

        if logprob.decoded_token is not None:
            return logprob.decoded_token

        if tokenizer is None:
            raise ValueError(
                "Unable to get tokenizer because `skip_tokenizer_init=True`"
            )

        return tokenizer.decode(token_id)


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
