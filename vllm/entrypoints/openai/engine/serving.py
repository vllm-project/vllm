# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
import sys
import time
import traceback
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, ClassVar, Generic, Protocol, TypeAlias, TypeVar

import numpy as np
from fastapi import Request
from openai.types.responses import (
    ToolChoiceFunction,
)
from pydantic import ConfigDict, TypeAdapter
from starlette.datastructures import Headers

import vllm.envs as envs
from vllm.beam_search import BeamSearchSequence, create_sort_beams_key_function
from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
    ConversationMessage,
)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorInfo,
    ErrorResponse,
    FunctionCall,
    FunctionDefinition,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.responses.context import (
    ConversationContext,
    HarmonyContext,
    ParsableContext,
    StreamingHarmonyContext,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponseInputOutputItem,
    ResponsesRequest,
)
from vllm.entrypoints.openai.responses.utils import (
    construct_input_messages,
)
from vllm.entrypoints.openai.translations.protocol import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranslationRequest,
)
from vllm.entrypoints.pooling.classify.protocol import (
    ClassificationChatRequest,
    ClassificationCompletionRequest,
    ClassificationResponse,
)
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingBytesResponse,
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
    EmbeddingResponse,
)
from vllm.entrypoints.pooling.pooling.protocol import (
    IOProcessorRequest,
    PoolingChatRequest,
    PoolingCompletionRequest,
    PoolingResponse,
)
from vllm.entrypoints.pooling.score.protocol import (
    RerankRequest,
    ScoreDataRequest,
    ScoreQueriesDocumentsRequest,
    ScoreRequest,
    ScoreResponse,
    ScoreTextRequest,
)
from vllm.entrypoints.serve.disagg.protocol import GenerateRequest, GenerateResponse
from vllm.entrypoints.serve.tokenize.protocol import (
    DetokenizeRequest,
    TokenizeChatRequest,
    TokenizeCompletionRequest,
    TokenizeResponse,
)
from vllm.entrypoints.utils import get_max_tokens, sanitize_message
from vllm.exceptions import VLLMValidationError
from vllm.inputs.data import EmbedsPrompt, PromptType, TokensPrompt
from vllm.inputs.parse import (
    get_prompt_components,
    is_explicit_encoder_decoder_prompt,
)
from vllm.logger import init_logger
from vllm.logprobs import Logprob, PromptLogprobs
from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalDataDict
from vllm.outputs import CompletionOutput, PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.renderers import ChatParams, TokenizeParams, merge_kwargs
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser
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


class GenerationError(Exception):
    """raised when finish_reason indicates internal server error (500)"""

    def __init__(self, message: str = "Internal server error"):
        super().__init__(message)
        self.status_code = HTTPStatus.INTERNAL_SERVER_ERROR


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
    CompletionRequest
    | TokenizeCompletionRequest
    | DetokenizeRequest
    | EmbeddingCompletionRequest
    | ClassificationCompletionRequest
    | RerankRequest
    | ScoreRequest
    | PoolingCompletionRequest
)

ChatLikeRequest: TypeAlias = (
    ChatCompletionRequest
    | TokenizeChatRequest
    | EmbeddingChatRequest
    | ClassificationChatRequest
    | PoolingChatRequest
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
    | EmbeddingBytesResponse
    | TranscriptionResponse
    | TokenizeResponse
    | PoolingResponse
    | ClassificationResponse
    | ScoreResponse
    | GenerateResponse
)


RequestT = TypeVar("RequestT", bound=AnyRequest)


@dataclass(kw_only=True)
class ServeContext(Generic[RequestT]):
    request: RequestT
    raw_request: Request | None = None
    model_name: str
    request_id: str
    created_time: int = field(default_factory=lambda: int(time.time()))
    lora_request: LoRARequest | None = None
    engine_prompts: list[TokensPrompt | EmbedsPrompt] | None = None

    result_generator: AsyncGenerator[tuple[int, PoolingRequestOutput], None] | None = (
        None
    )
    final_res_batch: list[PoolingRequestOutput] = field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


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

        self.log_error_stack = log_error_stack

        self.input_processor = self.models.input_processor
        self.io_processor = self.models.io_processor
        self.renderer = self.models.renderer
        self.model_config = self.models.model_config
        self.max_model_len = self.model_config.max_model_len

    async def beam_search(
        self,
        prompt: PromptType,
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

        input_processor = self.input_processor
        tokenizer = input_processor.tokenizer
        if tokenizer is None:
            raise VLLMValidationError(
                "You cannot use beam search when `skip_tokenizer_init=True`",
                parameter="skip_tokenizer_init",
                value=True,
            )

        eos_token_id: int = tokenizer.eos_token_id  # type: ignore

        if is_explicit_encoder_decoder_prompt(prompt):
            raise NotImplementedError

        prompt_text: str | None
        prompt_token_ids: list[int]
        multi_modal_data: MultiModalDataDict | None
        if isinstance(prompt, str):
            prompt_text = prompt
            prompt_token_ids = []
            multi_modal_data = None
        else:
            prompt_text = prompt.get("prompt")  # type: ignore
            prompt_token_ids = prompt.get("prompt_token_ids", [])  # type: ignore
            multi_modal_data = prompt.get("multi_modal_data")  # type: ignore

        mm_processor_kwargs: dict[str, Any] | None = None

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

        logprobs_num = 2 * beam_width
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

        for _ in range(max_tokens):
            prompts_batch, lora_req_batch = zip(
                *[
                    (
                        TokensPrompt(
                            prompt_token_ids=beam.tokens,
                            multi_modal_data=beam.multi_modal_data,
                            mm_processor_kwargs=beam.mm_processor_kwargs,
                        ),
                        beam.lora_request,
                    )
                    for beam in all_beams
                ]
            )

            tasks = []
            request_id_batch = f"{request_id}-{random_uuid()}"

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

        completed.extend(all_beams)
        sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]

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

    async def _preprocess(
        self,
        ctx: ServeContext,
    ) -> ErrorResponse | None:
        """
        Default preprocessing hook. Subclasses may override
        to prepare `ctx` (classification, embedding, etc.).
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
        generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []

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
            return self.create_error_response(e)

    async def _collect_batch(
        self,
        ctx: ServeContext,
    ) -> ErrorResponse | None:
        """Collect batch results from the result generator."""
        try:
            if ctx.engine_prompts is None:
                return self.create_error_response("Engine prompts not available")

            num_prompts = len(ctx.engine_prompts)
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

        except Exception as e:
            return self.create_error_response(e)

    def create_error_response(
        self,
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> ErrorResponse:
        exc: Exception | None = None

        if isinstance(message, Exception):
            exc = message

            from vllm.exceptions import VLLMValidationError

            if isinstance(exc, VLLMValidationError):
                err_type = "BadRequestError"
                status_code = HTTPStatus.BAD_REQUEST
                param = exc.parameter
            elif isinstance(exc, (ValueError, TypeError, RuntimeError, OverflowError)):
                # Common validation errors from user input
                err_type = "BadRequestError"
                status_code = HTTPStatus.BAD_REQUEST
                param = None
            elif isinstance(exc, NotImplementedError):
                err_type = "NotImplementedError"
                status_code = HTTPStatus.NOT_IMPLEMENTED
                param = None
            elif exc.__class__.__name__ == "TemplateError":
                # jinja2.TemplateError (avoid importing jinja2)
                err_type = "BadRequestError"
                status_code = HTTPStatus.BAD_REQUEST
                param = None
            else:
                err_type = "InternalServerError"
                status_code = HTTPStatus.INTERNAL_SERVER_ERROR
                param = None

            message = str(exc)

        if self.log_error_stack:
            exc_type, _, _ = sys.exc_info()
            if exc_type is not None:
                traceback.print_exc()
            else:
                traceback.print_stack()

        return ErrorResponse(
            error=ErrorInfo(
                message=sanitize_message(message),
                type=err_type,
                code=status_code.value,
                param=param,
            )
        )

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

    def _validate_input(
        self,
        request: object,
        input_ids: list[int],
        input_text: str,
    ) -> TokensPrompt:
        token_num = len(input_ids)

        # Note: EmbeddingRequest, ClassificationRequest,
        # and ScoreRequest doesn't have max_tokens
        if isinstance(
            request,
            (
                EmbeddingChatRequest,
                EmbeddingCompletionRequest,
                ScoreDataRequest,
                ScoreTextRequest,
                ScoreQueriesDocumentsRequest,
                RerankRequest,
                ClassificationCompletionRequest,
                ClassificationChatRequest,
            ),
        ):
            # Note: input length can be up to the entire model context length
            # since these requests don't generate tokens.
            if token_num > self.max_model_len:
                operations: dict[type[AnyRequest], str] = {
                    ScoreDataRequest: "score",
                    ScoreTextRequest: "score",
                    ScoreQueriesDocumentsRequest: "score",
                    ClassificationCompletionRequest: "classification",
                    ClassificationChatRequest: "classification",
                }
                operation = operations.get(type(request), "embedding generation")
                raise VLLMValidationError(
                    f"This model's maximum context length is "
                    f"{self.max_model_len} tokens. However, you requested "
                    f"{token_num} tokens in the input for {operation}. "
                    f"Please reduce the length of the input.",
                    parameter="input_tokens",
                    value=token_num,
                )
            return TokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

        # Note: TokenizeRequest and DetokenizeRequest doesn't have max_tokens
        # and does not require model context length validation
        if isinstance(
            request,
            (TokenizeCompletionRequest, TokenizeChatRequest, DetokenizeRequest),
        ):
            return TokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

        # chat completion endpoint supports max_completion_tokens
        if isinstance(request, ChatCompletionRequest):
            # TODO(#9845): remove max_tokens when field dropped from OpenAI API
            max_tokens = request.max_completion_tokens or request.max_tokens
        else:
            max_tokens = getattr(request, "max_tokens", None)

        # Note: input length can be up to model context length - 1 for
        # completion-like requests.
        if token_num >= self.max_model_len:
            raise VLLMValidationError(
                f"This model's maximum context length is "
                f"{self.max_model_len} tokens. However, your request has "
                f"{token_num} input tokens. Please reduce the length of "
                "the input messages.",
                parameter="input_tokens",
                value=token_num,
            )

        if max_tokens is not None and token_num + max_tokens > self.max_model_len:
            raise VLLMValidationError(
                "'max_tokens' or 'max_completion_tokens' is too large: "
                f"{max_tokens}. This model's maximum context length is "
                f"{self.max_model_len} tokens and your request has "
                f"{token_num} input tokens ({max_tokens} > {self.max_model_len}"
                f" - {token_num}).",
                parameter="max_tokens",
                value=max_tokens,
            )

        return TokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

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

    async def _preprocess_completion(
        self,
        request: RendererRequest,
        prompt_input: str | list[str] | list[int] | list[list[int]] | None,
        prompt_embeds: bytes | list[bytes] | None,
    ) -> list[TokensPrompt | EmbedsPrompt]:
        renderer = self.renderer
        tok_params = request.build_tok_params(self.model_config)

        in_prompts = await renderer.render_completions_async(
            prompt_input, prompt_embeds
        )
        engine_prompts = await renderer.tokenize_prompts_async(in_prompts, tok_params)

        extra_items = {
            k: v
            for k in ("mm_processor_kwargs", "cache_salt")
            if (v := getattr(request, k, None)) is not None
        }
        for prompt in engine_prompts:
            prompt.update(extra_items)  # type: ignore

        return engine_prompts

    async def _preprocess_chat(
        self,
        request: RendererChatRequest,
        messages: list[ChatCompletionMessageParam],
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
        default_template_kwargs: dict[str, Any] | None,
        tool_dicts: list[dict[str, Any]] | None = None,
        tool_parser: Callable[[TokenizerLike], ToolParser] | None = None,
    ) -> tuple[list[ConversationMessage], list[TokensPrompt | EmbedsPrompt]]:
        from vllm.tokenizers.mistral import MistralTokenizer

        renderer = self.renderer

        default_template_kwargs = merge_kwargs(
            default_template_kwargs,
            dict(
                tools=tool_dicts,
                tokenize=isinstance(renderer.tokenizer, MistralTokenizer),
            ),
        )

        tok_params = request.build_tok_params(self.model_config)
        chat_params = request.build_chat_params(
            default_template, default_template_content_format
        ).with_defaults(default_template_kwargs)

        conversation, prompt = await renderer.render_messages_async(
            messages, chat_params
        )
        engine_prompt = await renderer.tokenize_prompt_async(prompt, tok_params)

        extra_items = {
            k: v
            for k in ("mm_processor_kwargs", "cache_salt")
            if (v := getattr(request, k, None)) is not None
        }
        engine_prompt.update(extra_items)  # type: ignore

        # tool parsing is done only if a tool_parser has been set and if
        # tool_choice is not "none" (if tool_choice is "none" but a tool_parser
        # is set, we want to prevent parsing a tool_call hallucinated by the LLM
        if tool_parser is not None:
            tool_choice = getattr(request, "tool_choice", "none")
            if tool_choice != "none":
                if not isinstance(request, ChatCompletionRequest | ResponsesRequest):
                    msg = (
                        "Tool usage is only supported for Chat Completions API "
                        "or Responses API requests."
                    )
                    raise NotImplementedError(msg)

                # TODO: Update adjust_request to accept ResponsesRequest
                tokenizer = renderer.get_tokenizer()
                request = tool_parser(tokenizer).adjust_request(request=request)  # type: ignore[arg-type]

        return conversation, [engine_prompt]

    async def _render_next_turn(
        self,
        request: ResponsesRequest,
        messages: list[ResponseInputOutputItem],
        tool_dicts: list[dict[str, Any]] | None,
        tool_parser: Callable[[TokenizerLike], ToolParser] | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
    ):
        new_messages = construct_input_messages(
            request_input=messages,
        )

        _, engine_prompts = await self._preprocess_chat(
            request,
            new_messages,
            default_template=chat_template,
            default_template_content_format=chat_template_content_format,
            default_template_kwargs=None,
            tool_dicts=tool_dicts,
            tool_parser=tool_parser,
        )
        return engine_prompts

    async def _generate_with_builtin_tools(
        self,
        request_id: str,
        engine_prompt: TokensPrompt | EmbedsPrompt,
        sampling_params: SamplingParams,
        tok_params: TokenizeParams,
        context: ConversationContext,
        lora_request: LoRARequest | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
    ):
        prompt_text = engine_prompt.get("prompt")

        orig_priority = priority
        sub_request = 0
        while True:
            # Ensure that each sub-request has a unique request id.
            sub_request_id = f"{request_id}_{sub_request}"

            self._log_inputs(
                sub_request_id,
                engine_prompt,
                params=sampling_params,
                lora_request=lora_request,
            )

            tokenization_kwargs = tok_params.get_encode_kwargs()
            engine_request = self.input_processor.process_inputs(
                sub_request_id,
                engine_prompt,
                sampling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
            )

            generator = self.engine_client.generate(
                engine_request,
                sampling_params,
                sub_request_id,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=priority,
                prompt_text=prompt_text,
                tokenization_kwargs=tokenization_kwargs,
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
            # Render the next prompt token ids and update sampling_params.
            if isinstance(context, (HarmonyContext, StreamingHarmonyContext)):
                token_ids = context.render_for_completion()
                engine_prompt = TokensPrompt(prompt_token_ids=token_ids)

                sampling_params.max_tokens = self.max_model_len - len(token_ids)
            elif isinstance(context, ParsableContext):
                engine_prompts = await self._render_next_turn(
                    context.request,
                    context.parser.response_messages,
                    context.tool_dicts,
                    context.tool_parser_cls,
                    context.chat_template,
                    context.chat_template_content_format,
                )
                engine_prompt = engine_prompts[0]
                prompt_text = engine_prompt.get("prompt")

                sampling_params.max_tokens = get_max_tokens(
                    self.max_model_len,
                    context.request,
                    engine_prompt,
                    self.default_sampling_params,  # type: ignore
                )

            # OPTIMIZATION
            priority = orig_priority - 1
            sub_request += 1

    def _log_inputs(
        self,
        request_id: str,
        inputs: PromptType,
        params: SamplingParams | PoolingParams | BeamSearchParams | None,
        lora_request: LoRARequest | None,
    ) -> None:
        if self.request_logger is None:
            return

        prompt, prompt_token_ids, prompt_embeds = get_prompt_components(inputs)

        self.request_logger.log_inputs(
            request_id,
            prompt,
            prompt_token_ids,
            prompt_embeds,
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

    @staticmethod
    def _parse_tool_calls_from_content(
        request: ResponsesRequest | ChatCompletionRequest,
        tokenizer: TokenizerLike | None,
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
            if tokenizer is None:
                raise ValueError(
                    "Tokenizer not available when `skip_tokenizer_init=True`"
                )

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
                        id=tool_call.id,
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

        return tokenizer.decode([token_id])

    def _is_model_supported(self, model_name: str | None) -> bool:
        if not model_name:
            return True
        return self.models.is_base_model(model_name)


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
