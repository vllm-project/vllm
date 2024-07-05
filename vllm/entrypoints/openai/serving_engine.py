import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import Iterable, Iterator, List, Optional, TypedDict, Union

from pydantic import Field
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing_extensions import Annotated

from vllm.config import ModelConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest,
                                              DetokenizeRequest,
                                              EmbeddingRequest, ErrorResponse,
                                              ModelCard, ModelList,
                                              ModelPermission, TokenizeRequest)
from vllm.inputs import parse_and_batch_prompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.sequence import Logprob
from vllm.transformers_utils.tokenizer import get_tokenizer

logger = init_logger(__name__)


class TextTokensPrompt(TypedDict):
    prompt: str
    prompt_token_ids: List[int]


@dataclass
class LoRAModulePath:
    name: str
    local_path: str


class OpenAIServing:

    def __init__(
        self,
        engine: AsyncLLMEngine,
        model_config: ModelConfig,
        served_model_names: List[str],
        lora_modules: Optional[List[LoRAModulePath]],
        *,
        log_requests: bool,
        max_log_len: Optional[int],
    ):
        self.engine = engine
        self.model_config = model_config
        self.max_model_len = model_config.max_model_len

        # A separate tokenizer to map token IDs to strings.
        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            tokenizer_revision=model_config.tokenizer_revision,
            trust_remote_code=model_config.trust_remote_code,
            truncation_side="left")

        self.served_model_names = served_model_names

        if lora_modules is None:
            self.lora_requests = []
        else:
            self.lora_requests = [
                LoRARequest(
                    lora_name=lora.name,
                    lora_int_id=i,
                    lora_local_path=lora.local_path,
                ) for i, lora in enumerate(lora_modules, start=1)
            ]

        self.log_requests = log_requests
        self.max_log_len = max_log_len

    async def show_available_models(self) -> ModelList:
        """Show available models. Right now we only have one model."""
        model_cards = [
            ModelCard(id=served_model_name,
                      max_model_len=self.max_model_len,
                      root=self.served_model_names[0],
                      permission=[ModelPermission()])
            for served_model_name in self.served_model_names
        ]
        lora_cards = [
            ModelCard(id=lora.lora_name,
                      root=self.served_model_names[0],
                      permission=[ModelPermission()])
            for lora in self.lora_requests
        ]
        model_cards.extend(lora_cards)
        return ModelList(data=model_cards)

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
        self, request: Union[ChatCompletionRequest, CompletionRequest,
                             DetokenizeRequest, EmbeddingRequest,
                             TokenizeRequest]
    ) -> Optional[ErrorResponse]:
        if request.model in self.served_model_names:
            return None
        if request.model in [lora.lora_name for lora in self.lora_requests]:
            return None
        return self.create_error_response(
            message=f"The model `{request.model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND)

    def _maybe_get_lora(
        self, request: Union[CompletionRequest, ChatCompletionRequest,
                             DetokenizeRequest, EmbeddingRequest,
                             TokenizeRequest]
    ) -> Optional[LoRARequest]:
        if request.model in self.served_model_names:
            return None
        for lora in self.lora_requests:
            if request.model == lora.lora_name:
                return lora
        # if _check_model has been called earlier, this will be unreachable
        raise ValueError(f"The model `{request.model}` does not exist.")

    def _normalize_prompt_text_to_input(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest,
                       DetokenizeRequest, EmbeddingRequest, TokenizeRequest],
        prompt: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]],
        add_special_tokens: bool,
    ) -> TextTokensPrompt:
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
        request: Union[ChatCompletionRequest, CompletionRequest,
                       DetokenizeRequest, EmbeddingRequest, TokenizeRequest],
        prompt_ids: List[int],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]],
    ) -> TextTokensPrompt:
        if truncate_prompt_tokens is None:
            input_ids = prompt_ids
        else:
            input_ids = prompt_ids[-truncate_prompt_tokens:]

        input_text = tokenizer.decode(prompt_ids)

        return self._validate_input(request, input_ids, input_text)

    def _validate_input(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest,
                       DetokenizeRequest, EmbeddingRequest, TokenizeRequest],
        input_ids: List[int],
        input_text: str,
    ) -> TextTokensPrompt:
        token_num = len(input_ids)

        # Note: EmbeddingRequest doesn't have max_tokens
        if isinstance(request, EmbeddingRequest):
            if token_num > self.max_model_len:
                raise ValueError(
                    f"This model's maximum context length is "
                    f"{self.max_model_len} tokens. However, you requested "
                    f"{token_num} tokens in the input for embedding "
                    f"generation. Please reduce the length of the input.")
            return TextTokensPrompt(prompt=input_text,
                                    prompt_token_ids=input_ids)

        # Note: TokenizeRequest and DetokenizeRequest doesn't have max_tokens
        # and does not require model context length validation
        if isinstance(request, (TokenizeRequest, DetokenizeRequest)):
            return TextTokensPrompt(prompt=input_text,
                                    prompt_token_ids=input_ids)

        if request.max_tokens is None:
            if token_num >= self.max_model_len:
                raise ValueError(
                    f"This model's maximum context length is "
                    f"{self.max_model_len} tokens. However, you requested "
                    f"{token_num} tokens in the messages, "
                    f"Please reduce the length of the messages.")
            request.max_tokens = self.max_model_len - token_num

        if token_num + request.max_tokens > self.max_model_len:
            raise ValueError(
                f"This model's maximum context length is "
                f"{self.max_model_len} tokens. However, you requested "
                f"{request.max_tokens + token_num} tokens "
                f"({token_num} in the messages, "
                f"{request.max_tokens} in the completion). "
                f"Please reduce the length of the messages or completion.")

        return TextTokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

    def _tokenize_prompt_input(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest,
                       DetokenizeRequest, EmbeddingRequest, TokenizeRequest],
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
                [prompt_input],
                truncate_prompt_tokens=truncate_prompt_tokens,
                add_special_tokens=add_special_tokens,
            ))

    def _tokenize_prompt_inputs(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest,
                       DetokenizeRequest, EmbeddingRequest, TokenizeRequest],
        prompt_inputs: Iterable[Union[str, List[int]]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
        add_special_tokens: bool = True,
    ) -> Iterator[TextTokensPrompt]:
        """
        A simpler implementation of :meth:`_tokenize_prompt_input_or_inputs`
        that assumes multiple inputs.
        """
        tokenizer = self.tokenizer

        for text in prompt_inputs:
            if isinstance(text, str):
                yield self._normalize_prompt_text_to_input(
                    request,
                    prompt=text,
                    tokenizer=tokenizer,
                    truncate_prompt_tokens=truncate_prompt_tokens,
                    add_special_tokens=add_special_tokens,
                )
            else:
                yield self._normalize_prompt_tokens_to_input(
                    request,
                    prompt_ids=text,
                    tokenizer=tokenizer,
                    truncate_prompt_tokens=truncate_prompt_tokens,
                )

    def _tokenize_prompt_input_or_inputs(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest,
                       DetokenizeRequest, EmbeddingRequest, TokenizeRequest],
        input_or_inputs: Union[str, List[str], List[int], List[List[int]]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
        add_special_tokens: bool = True,
    ) -> Iterator[TextTokensPrompt]:
        """
        Tokenize/detokenize depending on the input format.

        According to `OpenAI API <https://platform.openai.com/docs/api-reference/embeddings/create>`_
        , each input can be a string or array of tokens. Note that each request
        can pass one or more inputs.
        """
        tokenizer = self.tokenizer

        for prompt_input in parse_and_batch_prompt(input_or_inputs):
            # Although our type checking is based on mypy,
            # VSCode Pyright extension should still work properly
            # "is True" is required for Pyright to perform type narrowing
            # See: https://github.com/microsoft/pyright/issues/7672
            if prompt_input["is_tokens"] is False:
                yield self._normalize_prompt_text_to_input(
                    request,
                    prompt=prompt_input["content"],
                    tokenizer=tokenizer,
                    truncate_prompt_tokens=truncate_prompt_tokens,
                    add_special_tokens=add_special_tokens,
                )
            else:
                yield self._normalize_prompt_tokens_to_input(
                    request,
                    prompt_ids=prompt_input["content"],
                    tokenizer=tokenizer,
                    truncate_prompt_tokens=truncate_prompt_tokens,
                )

    def _log_inputs(
        self,
        request_id: str,
        inputs: TextTokensPrompt,
        params: Union[SamplingParams, PoolingParams],
        lora_request: Optional[LoRARequest],
    ) -> None:
        if not self.log_requests:
            return

        shortened_prompt = inputs["prompt"]
        shortened_token_ids = inputs["prompt_token_ids"]

        max_log_len = self.max_log_len
        if max_log_len is not None:
            shortened_prompt = shortened_prompt[:max_log_len]
            shortened_token_ids = shortened_token_ids[:max_log_len]

        logger.info(
            "Received request %s: prompt: %r, "
            "params: %s, prompt_token_ids: %s, "
            "lora_request: %s.", request_id, shortened_prompt, params,
            shortened_token_ids, lora_request)

    def _get_decoded_token(self, logprob: Logprob, token_id: int) -> str:
        if logprob.decoded_token is not None:
            return logprob.decoded_token
        return self.tokenizer.decode(token_id)
