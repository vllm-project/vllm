# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
import sys
import time
import traceback
from collections.abc import AsyncGenerator, Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Any, ClassVar, Generic, TypeAlias, TypeVar

import torch
from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from starlette.datastructures import Headers
from typing_extensions import TypeIs

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from openai.types.responses import (
    ToolChoiceFunction,
)

import vllm.envs as envs
from vllm.beam_search import BeamSearchSequence, create_sort_beams_key_function
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
    ClassificationRequest,
    ClassificationResponse,
    CompletionRequest,
    CompletionResponse,
    DetokenizeRequest,
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorInfo,
    ErrorResponse,
    FunctionCall,
    FunctionDefinition,
    IOProcessorRequest,
    PoolingResponse,
    RerankRequest,
    ResponsesRequest,
    ScoreRequest,
    ScoreResponse,
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
from vllm.entrypoints.utils import _validate_truncation_size
from vllm.inputs.data import PromptType
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt
from vllm.inputs.parse import (
    PromptComponents,
    get_prompt_components,
    is_explicit_encoder_decoder_prompt,
)
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
from vllm.tracing import (
    contains_trace_headers,
    extract_trace_headers,
    log_tracing_disabled_warning,
)
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.utils import random_uuid
from vllm.utils.async_utils import (
    AsyncMicrobatchTokenizer,
    collect_from_async_generator,
    make_async,
    merge_async_iterators,
)
from vllm.utils.collection_utils import is_list_of
from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)

CompletionLikeRequest: TypeAlias = (
    CompletionRequest
    | DetokenizeRequest
    | EmbeddingCompletionRequest
    | RerankRequest
    | ClassificationRequest
    | ScoreRequest
    | TokenizeCompletionRequest
)

ChatLikeRequest: TypeAlias = (
    ChatCompletionRequest | EmbeddingChatRequest | TokenizeChatRequest
)
SpeechToTextRequest: TypeAlias = TranscriptionRequest | TranslationRequest
AnyRequest: TypeAlias = (
    CompletionLikeRequest
    | ChatLikeRequest
    | SpeechToTextRequest
    | ResponsesRequest
    | IOProcessorRequest
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


class RequestProcessingMixin(BaseModel):
    """
    Mixin for request processing,
    handling prompt preparation and engine input.
    """

    request_prompts: Sequence[RequestPrompt] | None = []
    engine_prompts: list[EngineTokensPrompt] | None = []

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ResponseGenerationMixin(BaseModel):
    """
    Mixin for response generation,
    managing result generators and final batch results.
    """

    result_generator: (
        AsyncGenerator[tuple[int, RequestOutput | PoolingRequestOutput], None] | None
    ) = None
    final_res_batch: list[RequestOutput | PoolingRequestOutput] = Field(
        default_factory=list
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ServeContext(
    RequestProcessingMixin,
    ResponseGenerationMixin,
    BaseModel,
    Generic[RequestT],
):
    # Shared across all requests
    request: RequestT
    raw_request: Request | None = None
    model_name: str
    request_id: str
    created_time: int = Field(default_factory=lambda: int(time.time()))
    lora_request: LoRARequest | None = None

    # Shared across most requests
    tokenizer: AnyTokenizer | None = None

    # `protected_namespaces` resolves Pydantic v2's warning
    # on conflict with protected namespace "model_"
    model_config = ConfigDict(
        protected_namespaces=(),
        arbitrary_types_allowed=True,
    )


ClassificationServeContext = ServeContext[ClassificationRequest]


class EmbeddingServeContext(ServeContext[EmbeddingRequest]):
    chat_template: str | None = None
    chat_template_content_format: ChatTemplateContentFormatOption


# Used to resolve the Pydantic error related to
# forward reference of MultiModalDataDict in TokensPrompt
RequestProcessingMixin.model_rebuild()
ServeContext.model_rebuild()
ClassificationServeContext.model_rebuild()
EmbeddingServeContext.model_rebuild()


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

        self._async_tokenizer_pool: dict[AnyTokenizer, AsyncMicrobatchTokenizer] = {}
        self.log_error_stack = log_error_stack

        self.processor = self.models.processor
        self.io_processor = self.models.io_processor
        self.model_config = self.models.model_config
        self.max_model_len = self.model_config.max_model_len

    def _get_tool_parser(
        self, tool_parser_name: str | None = None, enable_auto_tools: bool = False
    ) -> Callable[[AnyTokenizer], ToolParser] | None:
        """Get the tool parser based on the name."""
        parser = None
        if not enable_auto_tools or tool_parser_name is None:
            return parser
        logger.info(
            '"auto" tool choice has been enabled please note that while'
            " the parallel_tool_calls client option is preset for "
            "compatibility reasons, it will be ignored."
        )

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
    ) -> Callable[[AnyTokenizer], ReasoningParser] | None:
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
        self.processor.clear_mm_cache()
        await self.engine_client.reset_mm_cache()

    async def beam_search(
        self,
        prompt: PromptType,
        request_id: str,
        params: BeamSearchParams,
        lora_request: LoRARequest | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        beam_width = params.beam_width
        max_tokens = params.max_tokens
        ignore_eos = params.ignore_eos
        temperature = params.temperature
        length_penalty = params.length_penalty
        include_stop_str_in_output = params.include_stop_str_in_output

        processor = self.processor
        tokenizer = processor.tokenizer
        if tokenizer is None:
            raise ValueError(
                "You cannot use beam search when `skip_tokenizer_init` is True"
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

        beam_search_params = SamplingParams(
            logprobs=2 * beam_width,
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
                        EngineTokensPrompt(
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
                        )
                    )
                )
                tasks.append(task)

            output = [x[0] for x in await asyncio.gather(*tasks)]

            new_beams = []
            for i, current_beam in enumerate(all_beams):
                result = output[i]

                if result.outputs[0].logprobs is not None:
                    logprobs = result.outputs[0].logprobs[0]
                    for token_id, logprob_obj in logprobs.items():
                        if token_id == eos_token_id and not ignore_eos:
                            completed.append(
                                BeamSearchSequence(
                                    tokens=current_beam.tokens + [token_id]
                                    if include_stop_str_in_output
                                    else current_beam.tokens,
                                    logprobs=current_beam.logprobs + [logprobs],
                                    cum_logprob=current_beam.cum_logprob
                                    + logprob_obj.logprob,
                                    finish_reason="stop",
                                    stop_reason=eos_token_id,
                                )
                            )
                        else:
                            new_beams.append(
                                BeamSearchSequence(
                                    tokens=current_beam.tokens + [token_id],
                                    logprobs=current_beam.logprobs + [logprobs],
                                    lora_request=current_beam.lora_request,
                                    cum_logprob=current_beam.cum_logprob
                                    + logprob_obj.logprob,
                                    multi_modal_data=current_beam.multi_modal_data,
                                    mm_processor_kwargs=current_beam.mm_processor_kwargs,
                                )
                            )

            sorted_beams = sorted(new_beams, key=sort_beams_key, reverse=True)
            all_beams = sorted_beams[:beam_width]

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

    def _get_renderer(self, tokenizer: AnyTokenizer | None) -> BaseRenderer:
        """
        Get a Renderer instance with the provided tokenizer.
        Uses shared async tokenizer pool for efficiency.
        """
        return CompletionRenderer(
            model_config=self.model_config,
            tokenizer=tokenizer,
            async_tokenizer_pool=self._async_tokenizer_pool,
        )

    def _build_render_config(
        self,
        request: Any,
    ) -> RenderConfig:
        """
        Build and return a `RenderConfig` for an endpoint.

        Used by the renderer to control how prompts are prepared
        (e.g., tokenization and length handling). Endpoints should
        implement this with logic appropriate to their request type.
        """
        raise NotImplementedError

    def _get_async_tokenizer(self, tokenizer) -> AsyncMicrobatchTokenizer:
        """
        Return (and cache) an `AsyncMicrobatchTokenizer` bound to the
        given tokenizer.
        """
        async_tokenizer = self._async_tokenizer_pool.get(tokenizer)
        if async_tokenizer is None:
            async_tokenizer = AsyncMicrobatchTokenizer(tokenizer)
            self._async_tokenizer_pool[tokenizer] = async_tokenizer
        return async_tokenizer

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

        for message in request.messages:
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
        tokenizer: AnyTokenizer,
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
        tokenizer: AnyTokenizer | None,
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
                ClassificationRequest,
            ),
        ):
            # Note: input length can be up to the entire model context length
            # since these requests don't generate tokens.
            if token_num > self.max_model_len:
                operations: dict[type[AnyRequest], str] = {
                    ScoreRequest: "score",
                    ClassificationRequest: "classification",
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
        tokenizer: AnyTokenizer,
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
        tokenizer: AnyTokenizer,
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
        tokenizer: AnyTokenizer,
        messages: list[ChatCompletionMessageParam],
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tool_dicts: list[dict[str, Any]] | None = None,
        documents: list[dict[str, str]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        tool_parser: Callable[[AnyTokenizer], ToolParser] | None = None,
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
            tokenizer,
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

        engine_request = self.processor.process_inputs(
            request_id,
            engine_prompt,
            params,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
        )
        return engine_request, tokenization_kwargs

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
        while True:
            self._log_inputs(
                request_id,
                request_prompt,
                params=sampling_params,
                lora_request=lora_request,
            )
            trace_headers = kwargs.get("trace_headers")
            engine_request, tokenization_kwargs = await self._process_inputs(
                request_id,
                engine_prompt,
                sampling_params,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=priority,
            )

            generator = self.engine_client.generate(
                engine_request,
                sampling_params,
                request_id,
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
            prompt_token_ids = context.render_for_completion()
            engine_prompt = EngineTokensPrompt(prompt_token_ids=prompt_token_ids)
            request_prompt = prompt_token_ids
            # Update the sampling params.
            sampling_params.max_tokens = self.max_model_len - len(prompt_token_ids)
            # OPTIMIZATION
            priority = orig_priority - 1

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
        default = default or random_uuid()
        if raw_request is None:
            return default

        return raw_request.headers.get("X-Request-Id", default)

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
        tokenizer: AnyTokenizer,
        enable_auto_tools: bool,
        tool_parser_cls: Callable[[AnyTokenizer], ToolParser] | None,
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
        tokenizer: AnyTokenizer,
        return_as_token_id: bool = False,
    ) -> str:
        if return_as_token_id:
            return f"token_id:{token_id}"

        if logprob.decoded_token is not None:
            return logprob.decoded_token
        return tokenizer.decode(token_id)

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
