# SPDX-License-Identifier: Apache-2.0
import base64
import io
import json
import sys
import time
from collections.abc import (AsyncGenerator, Iterable, Iterator, Mapping,
                             Sequence)
from concurrent.futures.thread import ThreadPoolExecutor
from http import HTTPStatus
from typing import (Annotated, Any, Callable, ClassVar, Generic, Optional,
                    TypeVar, Union, cast, overload)

import torch
from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import Headers
from typing_extensions import TypeIs

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         ChatTemplateContentFormatOption,
                                         ConversationMessage,
                                         apply_hf_chat_template,
                                         apply_mistral_chat_template,
                                         parse_chat_messages_futures,
                                         resolve_chat_template_content_format)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              ClassificationRequest,
                                              ClassificationResponse,
                                              CompletionRequest,
                                              CompletionResponse,
                                              DetokenizeRequest,
                                              EmbeddingChatRequest,
                                              EmbeddingCompletionRequest,
                                              EmbeddingRequest,
                                              EmbeddingResponse, ErrorResponse,
                                              PoolingResponse, RerankRequest,
                                              ScoreRequest, ScoreResponse,
                                              TokenizeChatRequest,
                                              TokenizeCompletionRequest,
                                              TokenizeResponse,
                                              TranscriptionRequest,
                                              TranscriptionResponse)
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.openai.tool_parsers import ToolParser
# yapf: enable
from vllm.inputs.data import EmbedsPrompt as EngineEmbedsPrompt
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt
from vllm.inputs.parse import parse_and_batch_prompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import (  # noqa: F401 - Required to resolve Pydantic error in RequestProcessingMixin
    MultiModalDataDict)
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.sequence import Logprob, PromptLogprobs
from vllm.tracing import (contains_trace_headers, extract_trace_headers,
                          log_tracing_disabled_warning)
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.utils import (is_list_of, make_async, merge_async_iterators,
                        random_uuid)

logger = init_logger(__name__)

CompletionLikeRequest = Union[CompletionRequest, DetokenizeRequest,
                              EmbeddingCompletionRequest, RerankRequest,
                              ClassificationRequest, ScoreRequest,
                              TokenizeCompletionRequest]

ChatLikeRequest = Union[ChatCompletionRequest, EmbeddingChatRequest,
                        TokenizeChatRequest]

AnyRequest = Union[CompletionLikeRequest, ChatLikeRequest,
                   TranscriptionRequest]

AnyResponse = Union[
    CompletionResponse,
    ChatCompletionResponse,
    EmbeddingResponse,
    TranscriptionResponse,
    TokenizeResponse,
    PoolingResponse,
    ClassificationResponse,
    ScoreResponse,
]


class TextTokensPrompt(TypedDict):
    prompt: str
    prompt_token_ids: list[int]


class EmbedsPrompt(TypedDict):
    prompt_embeds: torch.Tensor


RequestPrompt = Union[list[int], str, TextTokensPrompt, EmbedsPrompt]


def is_text_tokens_prompt(prompt: RequestPrompt) -> TypeIs[TextTokensPrompt]:
    return (isinstance(prompt, dict) and "prompt_token_ids" in prompt
            and "prompt_embeds" not in prompt)


def is_embeds_prompt(prompt: RequestPrompt) -> TypeIs[EmbedsPrompt]:
    return (isinstance(prompt, dict) and "prompt_token_ids" not in prompt
            and "prompt_embeds" in prompt)


RequestT = TypeVar("RequestT", bound=AnyRequest)


class RequestProcessingMixin(BaseModel):
    """
    Mixin for request processing, 
    handling prompt preparation and engine input.
    """
    request_prompts: Optional[Sequence[RequestPrompt]] = \
                            Field(default_factory=list)
    engine_prompts: Optional[Union[list[EngineTokensPrompt],
                                   list[EngineEmbedsPrompt]]] = Field(
                                       default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ResponseGenerationMixin(BaseModel):
    """
    Mixin for response generation, 
    managing result generators and final batch results.
    """
    result_generator: Optional[AsyncGenerator[tuple[int, Union[
        RequestOutput, PoolingRequestOutput]], None]] = None
    final_res_batch: list[Union[RequestOutput, PoolingRequestOutput]] = Field(
        default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ServeContext(RequestProcessingMixin, ResponseGenerationMixin, BaseModel,
                   Generic[RequestT]):
    # Shared across all requests
    request: RequestT
    raw_request: Optional[Request] = None
    model_name: str
    request_id: str
    created_time: int = Field(default_factory=lambda: int(time.time()))
    lora_request: Optional[LoRARequest] = None
    prompt_adapter_request: Optional[PromptAdapterRequest] = None

    # Shared across most requests
    tokenizer: Optional[AnyTokenizer] = None
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None

    # `protected_namespaces` resolves Pydantic v2's warning
    # on conflict with protected namespace "model_"
    model_config = ConfigDict(
        protected_namespaces=(),
        arbitrary_types_allowed=True,
    )


ClassificationServeContext = ServeContext[ClassificationRequest]


class EmbeddingServeContext(ServeContext[EmbeddingRequest]):
    chat_template: Optional[str] = None
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
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
        return_tokens_as_token_ids: bool = False,
    ):
        super().__init__()

        self.engine_client = engine_client
        self.model_config = model_config
        self.max_model_len = model_config.max_model_len

        self.models = models

        self.request_logger = request_logger
        self.return_tokens_as_token_ids = return_tokens_as_token_ids

        self._tokenizer_executor = ThreadPoolExecutor(max_workers=1)

        self._tokenize_prompt_input_async = make_async(
            self._tokenize_prompt_input, executor=self._tokenizer_executor)
        self._tokenize_prompt_input_or_inputs_async = make_async(
            self._tokenize_prompt_input_or_inputs,
            executor=self._tokenizer_executor)

    async def _preprocess(
        self,
        ctx: ServeContext,
    ) -> Optional[ErrorResponse]:
        """
        Default preprocessing hook. Subclasses may override
        to prepare `ctx` (classification, embedding, etc.).
        """
        return None

    def _build_response(
        self,
        ctx: ServeContext,
    ) -> Union[AnyResponse, ErrorResponse]:
        """
        Default response builder. Subclass may override this method
        to return the appropriate response object.
        """
        return self.create_error_response("unimplemented endpoint")

    async def handle(
        self,
        ctx: ServeContext,
    ) -> Union[AnyResponse, ErrorResponse]:
        generation: AsyncGenerator[Union[AnyResponse, ErrorResponse], None]
        generation = self._pipeline(ctx)

        async for response in generation:
            return response

        return self.create_error_response("No response yielded from pipeline")

    async def _pipeline(
        self,
        ctx: ServeContext,
    ) -> AsyncGenerator[Union[AnyResponse, ErrorResponse], None]:
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

    def _validate_request(self, ctx: ServeContext) -> Optional[ErrorResponse]:
        truncate_prompt_tokens = getattr(ctx.request, "truncate_prompt_tokens",
                                         None)

        if truncate_prompt_tokens is not None:
            if truncate_prompt_tokens <= self.max_model_len:
                ctx.truncate_prompt_tokens = truncate_prompt_tokens
            else:
                return self.create_error_response(
                    "truncate_prompt_tokens value is "
                    "greater than max_model_len."
                    " Please, select a smaller truncation size.")
        return None

    async def _prepare_generators(
        self,
        ctx: ServeContext,
    ) -> Optional[ErrorResponse]:
        """Schedule the request and get the result generator."""
        generators: list[AsyncGenerator[Union[RequestOutput,
                                              PoolingRequestOutput],
                                        None]] = []

        try:
            trace_headers = (None if ctx.raw_request is None else await
                             self._get_trace_headers(ctx.raw_request.headers))

            if not hasattr(ctx.request, "to_pooling_params"):
                return self.create_error_response(
                    "Request type does not support pooling parameters")

            pooling_params = ctx.request.to_pooling_params()

            if ctx.engine_prompts is None:
                return self.create_error_response(
                    "Engine prompts not available")

            for i, engine_prompt in enumerate(ctx.engine_prompts):
                request_id_item = f"{ctx.request_id}-{i}"

                if ctx.request_prompts is None:
                    return self.create_error_response(
                        "Request prompts not available")

                self._log_inputs(
                    request_id_item,
                    ctx.request_prompts[i],
                    params=pooling_params,
                    lora_request=ctx.lora_request,
                    prompt_adapter_request=ctx.prompt_adapter_request)

                # Mypy has an existing bug related to inferring the variance of
                # TypedDicts with `builtins.enumerate`:
                # https://github.com/python/mypy/issues/8586#issuecomment-2867698435
                engine_prompt = cast(
                    Union[EngineTokensPrompt, EngineEmbedsPrompt],
                    engine_prompt)
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
    ) -> Optional[ErrorResponse]:
        """Collect batch results from the result generator."""
        try:
            if ctx.engine_prompts is None:
                return self.create_error_response(
                    "Engine prompts not available")

            num_prompts = len(ctx.engine_prompts)
            final_res_batch: list[Optional[Union[RequestOutput,
                                                 PoolingRequestOutput]]]
            final_res_batch = [None] * num_prompts

            if ctx.result_generator is None:
                return self.create_error_response(
                    "Result generator not available")

            async for i, res in ctx.result_generator:
                final_res_batch[i] = res

            if None in final_res_batch:
                return self.create_error_response(
                    "Failed to generate results for all prompts")

            ctx.final_res_batch = [
                res for res in final_res_batch if res is not None
            ]

            return None

        except Exception as e:
            return self.create_error_response(str(e))

    def create_error_response(
            self,
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
        return ErrorResponse(message=message,
                             type=err_type,
                             code=status_code.value)

    def create_streaming_error_response(
            self,
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> str:
        json_str = json.dumps({
            "error":
            self.create_error_response(message=message,
                                       err_type=err_type,
                                       status_code=status_code).model_dump()
        })
        return json_str

    async def _check_model(
        self,
        request: AnyRequest,
    ) -> Optional[ErrorResponse]:

        error_response = None

        if self._is_model_supported(request.model):
            return None
        if request.model in [
                lora.lora_name for lora in self.models.lora_requests
        ]:
            return None
        if envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING and request.model and (
                load_result := await self.models.resolve_lora(request.model)):
            if isinstance(load_result, LoRARequest):
                return None
            if isinstance(load_result, ErrorResponse) and \
                load_result.code == HTTPStatus.BAD_REQUEST.value:
                error_response = load_result
        if request.model in [
                prompt_adapter.prompt_adapter_name
                for prompt_adapter in self.models.prompt_adapter_requests
        ]:
            return None

        return error_response or self.create_error_response(
            message=f"The model `{request.model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND)

    def _maybe_get_adapters(
        self, request: AnyRequest
    ) -> Union[tuple[None, None], tuple[LoRARequest, None], tuple[
            None, PromptAdapterRequest]]:
        if self._is_model_supported(request.model):
            return None, None
        for lora in self.models.lora_requests:
            if request.model == lora.lora_name:
                return lora, None
        for prompt_adapter in self.models.prompt_adapter_requests:
            if request.model == prompt_adapter.prompt_adapter_name:
                return None, prompt_adapter
        # if _check_model has been called earlier, this will be unreachable
        raise ValueError(f"The model `{request.model}` does not exist.")

    def _normalize_prompt_text_to_input(
        self,
        request: AnyRequest,
        tokenizer: AnyTokenizer,
        prompt: str,
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=-1)]],
        add_special_tokens: bool,
    ) -> TextTokensPrompt:
        if (self.model_config.encoder_config is not None
                and self.model_config.encoder_config.get(
                    "do_lower_case", False)):
            prompt = prompt.lower()

        if truncate_prompt_tokens is None:
            encoded = tokenizer(prompt, add_special_tokens=add_special_tokens)
        elif truncate_prompt_tokens < 0:
            # Negative means we cap at the model's max length
            encoded = tokenizer(prompt,
                                add_special_tokens=add_special_tokens,
                                truncation=True,
                                max_length=self.max_model_len)
        else:
            encoded = tokenizer(prompt,
                                add_special_tokens=add_special_tokens,
                                truncation=True,
                                max_length=truncate_prompt_tokens)

        input_ids = encoded.input_ids

        input_text = prompt

        return self._validate_input(request, input_ids, input_text)

    def _normalize_prompt_tokens_to_input(
        self,
        request: AnyRequest,
        tokenizer: AnyTokenizer,
        prompt_ids: list[int],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]],
    ) -> TextTokensPrompt:
        if truncate_prompt_tokens is None:
            input_ids = prompt_ids
        elif truncate_prompt_tokens < 0:
            input_ids = prompt_ids[-self.max_model_len:]
        else:
            input_ids = prompt_ids[-truncate_prompt_tokens:]

        input_text = tokenizer.decode(input_ids)

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
        if isinstance(request,
                      (EmbeddingChatRequest, EmbeddingCompletionRequest,
                       ScoreRequest, RerankRequest, ClassificationRequest)):
            operation = {
                ScoreRequest: "score",
                ClassificationRequest: "classification"
            }.get(type(request), "embedding generation")

            if token_num > self.max_model_len:
                raise ValueError(
                    f"This model's maximum context length is "
                    f"{self.max_model_len} tokens. However, you requested "
                    f"{token_num} tokens in the input for {operation}. "
                    f"Please reduce the length of the input.")
            return TextTokensPrompt(prompt=input_text,
                                    prompt_token_ids=input_ids)

        # Note: TokenizeRequest and DetokenizeRequest doesn't have max_tokens
        # and does not require model context length validation
        if isinstance(request, (TokenizeCompletionRequest, TokenizeChatRequest,
                                DetokenizeRequest)):
            return TextTokensPrompt(prompt=input_text,
                                    prompt_token_ids=input_ids)

        # chat completion endpoint supports max_completion_tokens
        if isinstance(request, ChatCompletionRequest):
            # TODO(#9845): remove max_tokens when field dropped from OpenAI API
            max_tokens = request.max_completion_tokens or request.max_tokens
        else:
            max_tokens = getattr(request, "max_tokens", None)
        if max_tokens is None:
            if token_num >= self.max_model_len:
                raise ValueError(
                    f"This model's maximum context length is "
                    f"{self.max_model_len} tokens. However, you requested "
                    f"{token_num} tokens in the messages, "
                    f"Please reduce the length of the messages.")
        elif token_num + max_tokens > self.max_model_len:
            raise ValueError(
                f"This model's maximum context length is "
                f"{self.max_model_len} tokens. However, you requested "
                f"{max_tokens + token_num} tokens "
                f"({token_num} in the messages, "
                f"{max_tokens} in the completion). "
                f"Please reduce the length of the messages or completion.")

        return TextTokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

    def _tokenize_prompt_input(
        self,
        request: AnyRequest,
        tokenizer: AnyTokenizer,
        prompt_input: Union[str, list[int]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=-1)]] = None,
        add_special_tokens: bool = True,
    ) -> TextTokensPrompt:
        """
        A simpler implementation of {meth}`_tokenize_prompt_input_or_inputs`
        that assumes single input.
        """
        return next(
            self._tokenize_prompt_inputs(
                request,
                tokenizer,
                [prompt_input],
                truncate_prompt_tokens=truncate_prompt_tokens,
                add_special_tokens=add_special_tokens,
            ))

    def _tokenize_prompt_inputs(
        self,
        request: AnyRequest,
        tokenizer: AnyTokenizer,
        prompt_inputs: Iterable[Union[str, list[int]]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=-1)]] = None,
        add_special_tokens: bool = True,
    ) -> Iterator[TextTokensPrompt]:
        """
        A simpler implementation of {meth}`_tokenize_prompt_input_or_inputs`
        that assumes multiple inputs.
        """
        for text in prompt_inputs:
            if isinstance(text, str):
                yield self._normalize_prompt_text_to_input(
                    request,
                    tokenizer,
                    prompt=text,
                    truncate_prompt_tokens=truncate_prompt_tokens,
                    add_special_tokens=add_special_tokens,
                )
            else:
                yield self._normalize_prompt_tokens_to_input(
                    request,
                    tokenizer,
                    prompt_ids=text,
                    truncate_prompt_tokens=truncate_prompt_tokens,
                )

    def _tokenize_prompt_input_or_inputs(
        self,
        request: AnyRequest,
        tokenizer: AnyTokenizer,
        input_or_inputs: Optional[Union[str, list[str], list[int],
                                        list[list[int]]]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=-1)]] = None,
        add_special_tokens: bool = True,
    ) -> tuple[list[TextTokensPrompt], list[EmbedsPrompt]]:
        """
        Tokenize/detokenize depending on the input format.

        According to `OpenAI API <https://platform.openai.com/docs/api-reference/embeddings/create>`_
        , each input can be a string or array of tokens. Note that each request
        can pass one or more inputs.
        """
        inputs_embeds = list[EmbedsPrompt]()
        inputs_text = list[TextTokensPrompt]()

        if (isinstance(request, CompletionRequest)
                and request.prompt_embeds is not None):
            inputs_embeds.extend(
                self._load_prompt_embeds(request.prompt_embeds,
                                         truncate_prompt_tokens))

        # Empty prompts are okay as long as there are prompt embeddings
        if input_or_inputs is None or (inputs_embeds
                                       and input_or_inputs == ""):
            return [], inputs_embeds

        # Although our type checking is based on mypy,
        # VSCode Pyright extension should still work properly
        # "is False" is required for Pyright to perform type narrowing
        # See: https://github.com/microsoft/pyright/issues/7672
        inputs_text.extend([
            self._normalize_prompt_text_to_input(
                request,
                tokenizer,
                prompt=prompt_input["content"],
                truncate_prompt_tokens=truncate_prompt_tokens,
                add_special_tokens=add_special_tokens)
            if prompt_input["is_tokens"] is False else
            self._normalize_prompt_tokens_to_input(
                request,
                tokenizer,
                prompt_ids=prompt_input["content"],
                truncate_prompt_tokens=truncate_prompt_tokens)
            for prompt_input in parse_and_batch_prompt(input_or_inputs)
        ])

        return inputs_text, inputs_embeds

    @overload
    async def _preprocess_completion(
        self,
        request: Union[DetokenizeRequest, EmbeddingCompletionRequest,
                       RerankRequest, ClassificationRequest, ScoreRequest,
                       TokenizeCompletionRequest],
        tokenizer: AnyTokenizer,
        input_or_inputs: Union[str, list[str], list[int], list[list[int]]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=-1)]] = ...,
        add_special_tokens: bool = ...,
    ) -> tuple[list[TextTokensPrompt], list[EngineTokensPrompt]]:
        ...

    @overload
    async def _preprocess_completion(
        self,
        request: CompletionRequest,
        tokenizer: AnyTokenizer,
        input_or_inputs: Optional[Union[str, list[str], list[int],
                                        list[list[int]]]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=-1)]] = ...,
        add_special_tokens: bool = ...,
    ) -> tuple[list[Union[TextTokensPrompt, EmbedsPrompt]], list[Union[
            EngineTokensPrompt, EngineEmbedsPrompt]]]:
        ...

    async def _preprocess_completion(
        self,
        request: CompletionLikeRequest,
        tokenizer: AnyTokenizer,
        input_or_inputs: Optional[Union[str, list[str], list[int],
                                        list[list[int]]]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=-1)]] = None,
        add_special_tokens: bool = True,
    ) -> tuple[Union[list[TextTokensPrompt], list[Union[
            TextTokensPrompt, EmbedsPrompt]]], Union[
                list[EngineTokensPrompt], list[Union[EngineTokensPrompt,
                                                     EngineEmbedsPrompt]]]]:
        if not isinstance(request,
                          CompletionRequest) and input_or_inputs is None:
            raise ValueError(
                "Prompt embeds with non-completion requests is not"
                " currently supported.")

        (request_prompts_text, request_prompts_embeds
         ) = await self._tokenize_prompt_input_or_inputs_async(
             request,
             tokenizer,
             input_or_inputs,
             truncate_prompt_tokens=truncate_prompt_tokens,
             add_special_tokens=add_special_tokens,
         )

        engine_prompts_text = [
            EngineTokensPrompt(
                prompt_token_ids=request_prompt_text["prompt_token_ids"])
            for request_prompt_text in request_prompts_text
        ]

        # This check is equivalent to simply checking if
        # `request_prompts_embeds` is empty, but it's difficult to propagate
        # overloads to the private helper functions to enable this check.
        # This overload is needed because only TextPrompts are allowed for
        # non-completion requests and if we don't add the overload here,
        # everywhere this function is used outside of serving_completion will
        # need logic asserting that only text prompts are in the request.
        if not isinstance(request,
                          CompletionRequest) and input_or_inputs is not None:
            return request_prompts_text, engine_prompts_text

        engine_prompts_embeds = [
            EngineEmbedsPrompt(
                prompt_embeds=request_prompt_embeds["prompt_embeds"])
            for request_prompt_embeds in request_prompts_embeds
        ]

        request_prompts = request_prompts_embeds + request_prompts_text
        engine_prompts = engine_prompts_embeds + engine_prompts_text
        return request_prompts, engine_prompts

    async def _preprocess_chat(
        self,
        request: ChatLikeRequest,
        tokenizer: AnyTokenizer,
        messages: list[ChatCompletionMessageParam],
        chat_template: Optional[str],
        chat_template_content_format: ChatTemplateContentFormatOption,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tool_dicts: Optional[list[dict[str, Any]]] = None,
        documents: Optional[list[dict[str, str]]] = None,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
        tool_parser: Optional[Callable[[AnyTokenizer], ToolParser]] = None,
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
        add_special_tokens: bool = False,
    ) -> tuple[list[ConversationMessage], Sequence[RequestPrompt],
               list[EngineTokensPrompt]]:
        model_config = self.model_config

        resolved_content_format = resolve_chat_template_content_format(
            chat_template,
            tool_dicts,
            chat_template_content_format,
            tokenizer,
            model_config=model_config,
        )
        conversation, mm_data_future = parse_chat_messages_futures(
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

        request_prompt: Union[str, list[int]]
        if isinstance(tokenizer, MistralTokenizer):
            request_prompt = apply_mistral_chat_template(
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
        should_parse_tools = tool_parser is not None and (hasattr(
            request, "tool_choice") and request.tool_choice != "none")

        if should_parse_tools:
            if not isinstance(request, ChatCompletionRequest):
                msg = "Tool usage is only supported for Chat Completions API"
                raise NotImplementedError(msg)

            request = tool_parser(tokenizer).adjust_request(  # type: ignore
                request=request)

        if isinstance(request_prompt, str):
            prompt_inputs = await self._tokenize_prompt_input_async(
                request,
                tokenizer,
                request_prompt,
                truncate_prompt_tokens=truncate_prompt_tokens,
                add_special_tokens=add_special_tokens,
            )
        else:
            # For MistralTokenizer
            assert is_list_of(request_prompt, int), (
                "Prompt has to be either a string or a list of token ids")
            prompt_inputs = TextTokensPrompt(
                prompt=tokenizer.decode(request_prompt),
                prompt_token_ids=request_prompt)

        engine_prompt = EngineTokensPrompt(
            prompt_token_ids=prompt_inputs["prompt_token_ids"])
        if mm_data is not None:
            engine_prompt["multi_modal_data"] = mm_data
        if request.mm_processor_kwargs is not None:
            engine_prompt["mm_processor_kwargs"] = request.mm_processor_kwargs

        if hasattr(request, "cache_salt") and request.cache_salt is not None:
            engine_prompt["cache_salt"] = request.cache_salt

        return conversation, [request_prompt], [engine_prompt]

    def _load_prompt_embeds(
        self,
        prompt_embeds: Optional[Union[bytes, list[bytes]]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    ) -> list[EmbedsPrompt]:

        def _load_and_validate_embed(embed: bytes) -> EmbedsPrompt:
            tensor = torch.load(io.BytesIO(base64.b64decode(embed)),
                                weights_only=True)
            assert isinstance(
                tensor,
                (torch.FloatTensor, torch.BFloat16Tensor, torch.HalfTensor))
            if tensor.dim() > 2:
                tensor = tensor.squeeze(0)
                assert tensor.dim() == 2
            if truncate_prompt_tokens is not None:
                tensor = tensor[-truncate_prompt_tokens:]
            return {"prompt_embeds": tensor}

        if prompt_embeds:
            if isinstance(prompt_embeds, list):
                return [
                    _load_and_validate_embed(embed) for embed in prompt_embeds
                ]
            else:
                return [_load_and_validate_embed(prompt_embeds)]
        else:
            return []

    def _log_inputs(
        self,
        request_id: str,
        inputs: RequestPrompt,
        params: Optional[Union[SamplingParams, PoolingParams,
                               BeamSearchParams]],
        lora_request: Optional[LoRARequest],
        prompt_adapter_request: Optional[PromptAdapterRequest],
    ) -> None:
        if self.request_logger is None:
            return
        prompt, prompt_token_ids, prompt_embeds = None, None, None
        if isinstance(inputs, str):
            prompt = inputs
        elif isinstance(inputs, list):
            prompt_token_ids = inputs
        elif 'prompt_embeds' in inputs:
            prompt_embeds = inputs.get("prompt_embeds")
        else:
            prompt = inputs["prompt"]
            prompt_token_ids = inputs["prompt_token_ids"]

        self.request_logger.log_inputs(
            request_id,
            prompt,
            prompt_token_ids,
            prompt_embeds,
            params=params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
        )

    async def _get_trace_headers(
        self,
        headers: Headers,
    ) -> Optional[Mapping[str, str]]:
        is_tracing_enabled = await self.engine_client.is_tracing_enabled()

        if is_tracing_enabled:
            return extract_trace_headers(headers)

        if contains_trace_headers(headers):
            log_tracing_disabled_warning()

        return None

    @staticmethod
    def _base_request_id(raw_request: Optional[Request],
                         default: Optional[str] = None) -> Optional[str]:
        """Pulls the request id to use from a header, if provided"""
        default = default or random_uuid()
        if raw_request is None:
            return default

        return raw_request.headers.get("X-Request-Id", default)

    @staticmethod
    def _get_decoded_token(logprob: Logprob,
                           token_id: int,
                           tokenizer: AnyTokenizer,
                           return_as_token_id: bool = False) -> str:
        if return_as_token_id:
            return f"token_id:{token_id}"

        if logprob.decoded_token is not None:
            return logprob.decoded_token
        return tokenizer.decode(token_id)

    def _is_model_supported(self, model_name: Optional[str]) -> bool:
        if not model_name:
            return True
        return self.models.is_base_model(model_name)

    def _get_model_name(self,
                        model_name: Optional[str] = None,
                        lora_request: Optional[LoRARequest] = None) -> str:
        if lora_request:
            return lora_request.lora_name
        if not model_name:
            return self.models.base_model_paths[0].name
        return model_name


def clamp_prompt_logprobs(
    prompt_logprobs: Union[PromptLogprobs,
                           None]) -> Union[PromptLogprobs, None]:
    if prompt_logprobs is None:
        return prompt_logprobs

    for logprob_dict in prompt_logprobs:
        if logprob_dict is None:
            continue
        for logprob_values in logprob_dict.values():
            if logprob_values.logprob == float('-inf'):
                logprob_values.logprob = -9999.0
    return prompt_logprobs
