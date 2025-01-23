import json
from concurrent.futures.thread import ThreadPoolExecutor
from http import HTTPStatus
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Mapping,
                    Optional, Sequence, Tuple, TypedDict, Union)

from fastapi import Request
from pydantic import Field
from starlette.datastructures import Headers
from typing_extensions import Annotated

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
                                              CompletionRequest,
                                              DetokenizeRequest,
                                              EmbeddingChatRequest,
                                              EmbeddingCompletionRequest,
                                              ErrorResponse, ScoreRequest,
                                              TokenizeChatRequest,
                                              TokenizeCompletionRequest)
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.openai.tool_parsers import ToolParser
# yapf: enable
from vllm.inputs import TokensPrompt
from vllm.inputs.parse import parse_and_batch_prompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.sequence import Logprob
from vllm.tracing import (contains_trace_headers, extract_trace_headers,
                          log_tracing_disabled_warning)
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.utils import is_list_of, make_async, random_uuid

logger = init_logger(__name__)

CompletionLikeRequest = Union[CompletionRequest, DetokenizeRequest,
                              EmbeddingCompletionRequest, ScoreRequest,
                              TokenizeCompletionRequest]

ChatLikeRequest = Union[ChatCompletionRequest, EmbeddingChatRequest,
                        TokenizeChatRequest]

AnyRequest = Union[CompletionLikeRequest, ChatLikeRequest]


class TextTokensPrompt(TypedDict):
    prompt: str
    prompt_token_ids: List[int]


RequestPrompt = Union[List[int], str, TextTokensPrompt]


class OpenAIServing:

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
        if self._is_model_supported(request.model):
            return None
        if request.model in [
                lora.lora_name for lora in self.models.lora_requests
        ]:
            return None
        if request.model in [
                prompt_adapter.prompt_adapter_name
                for prompt_adapter in self.models.prompt_adapter_requests
        ]:
            return None
        return self.create_error_response(
            message=f"The model `{request.model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND)

    def _maybe_get_adapters(
        self, request: AnyRequest
    ) -> Union[Tuple[None, None], Tuple[LoRARequest, None], Tuple[
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
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]],
        add_special_tokens: bool,
    ) -> TextTokensPrompt:
        if (self.model_config.encoder_config is not None
                and self.model_config.encoder_config.get(
                    "do_lower_case", False)):
            prompt = prompt.lower()

        if truncate_prompt_tokens is None:
            encoded = tokenizer(prompt, add_special_tokens=add_special_tokens)
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
        prompt_ids: List[int],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]],
    ) -> TextTokensPrompt:
        if truncate_prompt_tokens is None:
            input_ids = prompt_ids
        else:
            input_ids = prompt_ids[-truncate_prompt_tokens:]

        input_text = tokenizer.decode(input_ids)

        return self._validate_input(request, input_ids, input_text)

    def _validate_input(
        self,
        request: AnyRequest,
        input_ids: List[int],
        input_text: str,
    ) -> TextTokensPrompt:
        token_num = len(input_ids)

        # Note: EmbeddingRequest and ScoreRequest doesn't have max_tokens
        if isinstance(
                request,
            (EmbeddingChatRequest, EmbeddingCompletionRequest, ScoreRequest)):

            operation = "score" if isinstance(request, ScoreRequest) \
                else "embedding generation"
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
            max_tokens = request.max_tokens
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
        prompt_input: Union[str, List[int]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
        add_special_tokens: bool = True,
    ) -> TextTokensPrompt:
        """
        A simpler implementation of :meth:`_tokenize_prompt_input_or_inputs`
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
        prompt_inputs: Iterable[Union[str, List[int]]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
        add_special_tokens: bool = True,
    ) -> Iterator[TextTokensPrompt]:
        """
        A simpler implementation of :meth:`_tokenize_prompt_input_or_inputs`
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
        input_or_inputs: Union[str, List[str], List[int], List[List[int]]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
        add_special_tokens: bool = True,
    ) -> List[TextTokensPrompt]:
        """
        Tokenize/detokenize depending on the input format.

        According to `OpenAI API <https://platform.openai.com/docs/api-reference/embeddings/create>`_
        , each input can be a string or array of tokens. Note that each request
        can pass one or more inputs.
        """
        # Although our type checking is based on mypy,
        # VSCode Pyright extension should still work properly
        # "is True" is required for Pyright to perform type narrowing
        # See: https://github.com/microsoft/pyright/issues/7672
        return [
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
        ]

    async def _preprocess_completion(
        self,
        request: CompletionLikeRequest,
        tokenizer: AnyTokenizer,
        input_or_inputs: Union[str, List[str], List[int], List[List[int]]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
        add_special_tokens: bool = True,
    ) -> Tuple[List[TextTokensPrompt], List[TokensPrompt]]:
        request_prompts = await self._tokenize_prompt_input_or_inputs_async(
            request,
            tokenizer,
            input_or_inputs,
            truncate_prompt_tokens=truncate_prompt_tokens,
            add_special_tokens=add_special_tokens,
        )

        engine_prompts = [
            TokensPrompt(prompt_token_ids=request_prompt["prompt_token_ids"])
            for request_prompt in request_prompts
        ]

        return request_prompts, engine_prompts

    async def _preprocess_chat(
        self,
        request: ChatLikeRequest,
        tokenizer: AnyTokenizer,
        messages: List[ChatCompletionMessageParam],
        chat_template: Optional[str],
        chat_template_content_format: ChatTemplateContentFormatOption,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tool_dicts: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[Dict[str, str]]] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        tool_parser: Optional[Callable[[AnyTokenizer], ToolParser]] = None,
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
        add_special_tokens: bool = False,
    ) -> Tuple[List[ConversationMessage], Sequence[RequestPrompt],
               List[TokensPrompt]]:
        resolved_content_format = resolve_chat_template_content_format(
            chat_template,
            chat_template_content_format,
            tokenizer,
        )
        conversation, mm_data_future = parse_chat_messages_futures(
            messages,
            self.model_config,
            tokenizer,
            content_format=resolved_content_format,
        )

        _chat_template_kwargs: Dict[str, Any] = dict(
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tool_dicts,
            documents=documents,
        )
        _chat_template_kwargs.update(chat_template_kwargs or {})

        request_prompt: Union[str, List[int]]
        is_mistral_tokenizer = isinstance(tokenizer, MistralTokenizer)
        if is_mistral_tokenizer:
            request_prompt = apply_mistral_chat_template(
                tokenizer,
                messages=messages,
                **_chat_template_kwargs,
            )
        else:
            request_prompt = apply_hf_chat_template(
                tokenizer,
                conversation=conversation,
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

        engine_prompt = TokensPrompt(
            prompt_token_ids=prompt_inputs["prompt_token_ids"])
        if mm_data is not None:
            engine_prompt["multi_modal_data"] = mm_data

        return conversation, [request_prompt], [engine_prompt]

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

        if isinstance(inputs, str):
            prompt = inputs
            prompt_token_ids = None
        elif isinstance(inputs, list):
            prompt = None
            prompt_token_ids = inputs
        else:
            prompt = inputs["prompt"]
            prompt_token_ids = inputs["prompt_token_ids"]

        self.request_logger.log_inputs(
            request_id,
            prompt,
            prompt_token_ids,
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

    def _is_model_supported(self, model_name):
        return self.models.is_base_model(model_name)
