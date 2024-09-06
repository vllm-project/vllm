import json
import pathlib
from dataclasses import dataclass
from http import HTTPStatus
from typing import Iterable, Iterator, List, Optional, Tuple, TypedDict, Union

from pydantic import Field
from typing_extensions import Annotated

from vllm.config import ModelConfig
from vllm.engine.protocol import AsyncEngineClient
from vllm.entrypoints.logger import RequestLogger
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest,
                                              DetokenizeRequest,
                                              EmbeddingRequest, ErrorResponse,
                                              LoadLoraAdapterRequest,
                                              ModelCard, ModelList,
                                              ModelPermission,
                                              TokenizeChatRequest,
                                              TokenizeCompletionRequest,
                                              TokenizeRequest,
                                              UnloadLoraAdapterRequest)
# yapf: enable
from vllm.inputs.parse import parse_and_batch_prompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding import (
    get_guided_decoding_logits_processor)
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import LogitsProcessor, SamplingParams
from vllm.sequence import Logprob
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import AtomicCounter

logger = init_logger(__name__)


@dataclass
class PromptAdapterPath:
    name: str
    local_path: str


@dataclass
class LoRAModulePath:
    name: str
    path: str


AnyRequest = Union[ChatCompletionRequest, CompletionRequest, DetokenizeRequest,
                   EmbeddingRequest, TokenizeRequest]


class TextTokensPrompt(TypedDict):
    prompt: str
    prompt_token_ids: List[int]


class OpenAIServing:

    def __init__(
        self,
        async_engine_client: AsyncEngineClient,
        model_config: ModelConfig,
        served_model_names: List[str],
        *,
        lora_modules: Optional[List[LoRAModulePath]],
        prompt_adapters: Optional[List[PromptAdapterPath]],
        request_logger: Optional[RequestLogger],
        return_tokens_as_token_ids: bool = False,
    ):
        super().__init__()

        self.async_engine_client = async_engine_client
        self.model_config = model_config
        self.max_model_len = model_config.max_model_len

        self.served_model_names = served_model_names

        self.lora_id_counter = AtomicCounter(0)
        self.lora_requests = []
        if lora_modules is not None:
            self.lora_requests = [
                LoRARequest(
                    lora_name=lora.name,
                    lora_int_id=i,
                    lora_path=lora.path,
                ) for i, lora in enumerate(lora_modules, start=1)
            ]

        self.prompt_adapter_requests = []
        if prompt_adapters is not None:
            for i, prompt_adapter in enumerate(prompt_adapters, start=1):
                with pathlib.Path(prompt_adapter.local_path,
                                  "adapter_config.json").open() as f:
                    adapter_config = json.load(f)
                    num_virtual_tokens = adapter_config["num_virtual_tokens"]
                self.prompt_adapter_requests.append(
                    PromptAdapterRequest(
                        prompt_adapter_name=prompt_adapter.name,
                        prompt_adapter_id=i,
                        prompt_adapter_local_path=prompt_adapter.local_path,
                        prompt_adapter_num_virtual_tokens=num_virtual_tokens))

        self.request_logger = request_logger
        self.return_tokens_as_token_ids = return_tokens_as_token_ids

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
        prompt_adapter_cards = [
            ModelCard(id=prompt_adapter.prompt_adapter_name,
                      root=self.served_model_names[0],
                      permission=[ModelPermission()])
            for prompt_adapter in self.prompt_adapter_requests
        ]
        model_cards.extend(lora_cards)
        model_cards.extend(prompt_adapter_cards)
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

    async def _guided_decode_logits_processor(
            self, request: Union[ChatCompletionRequest, CompletionRequest],
            tokenizer: AnyTokenizer) -> Optional[LogitsProcessor]:
        decoding_config = await self.async_engine_client.get_decoding_config()
        guided_decoding_backend = request.guided_decoding_backend \
            or decoding_config.guided_decoding_backend
        return await get_guided_decoding_logits_processor(
            guided_decoding_backend, request, tokenizer)

    async def _check_model(
        self,
        request: AnyRequest,
    ) -> Optional[ErrorResponse]:
        if request.model in self.served_model_names:
            return None
        if request.model in [lora.lora_name for lora in self.lora_requests]:
            return None
        if request.model in [
                prompt_adapter.prompt_adapter_name
                for prompt_adapter in self.prompt_adapter_requests
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
        if request.model in self.served_model_names:
            return None, None
        for lora in self.lora_requests:
            if request.model == lora.lora_name:
                return lora, None
        for prompt_adapter in self.prompt_adapter_requests:
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
        if isinstance(request, (TokenizeCompletionRequest, TokenizeChatRequest,
                                DetokenizeRequest)):
            return TextTokensPrompt(prompt=input_text,
                                    prompt_token_ids=input_ids)

        if request.max_tokens is None:
            if token_num >= self.max_model_len:
                raise ValueError(
                    f"This model's maximum context length is "
                    f"{self.max_model_len} tokens. However, you requested "
                    f"{token_num} tokens in the messages, "
                    f"Please reduce the length of the messages.")
        elif token_num + request.max_tokens > self.max_model_len:
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
    ) -> Iterator[TextTokensPrompt]:
        """
        Tokenize/detokenize depending on the input format.

        According to `OpenAI API <https://platform.openai.com/docs/api-reference/embeddings/create>`_
        , each input can be a string or array of tokens. Note that each request
        can pass one or more inputs.
        """
        for prompt_input in parse_and_batch_prompt(input_or_inputs):
            # Although our type checking is based on mypy,
            # VSCode Pyright extension should still work properly
            # "is True" is required for Pyright to perform type narrowing
            # See: https://github.com/microsoft/pyright/issues/7672
            if prompt_input["is_tokens"] is False:
                yield self._normalize_prompt_text_to_input(
                    request,
                    tokenizer,
                    prompt=prompt_input["content"],
                    truncate_prompt_tokens=truncate_prompt_tokens,
                    add_special_tokens=add_special_tokens,
                )
            else:
                yield self._normalize_prompt_tokens_to_input(
                    request,
                    tokenizer,
                    prompt_ids=prompt_input["content"],
                    truncate_prompt_tokens=truncate_prompt_tokens,
                )

    def _log_inputs(
        self,
        request_id: str,
        inputs: Union[str, List[int], TextTokensPrompt],
        params: Optional[Union[SamplingParams, PoolingParams]],
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

    async def _check_load_lora_adapter_request(
            self, request: LoadLoraAdapterRequest) -> Optional[ErrorResponse]:
        # Check if both 'lora_name' and 'lora_path' are provided
        if not request.lora_name or not request.lora_path:
            return self.create_error_response(
                message="Both 'lora_name' and 'lora_path' must be provided.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST)

        # Check if the lora adapter with the given name already exists
        if any(lora_request.lora_name == request.lora_name
               for lora_request in self.lora_requests):
            return self.create_error_response(
                message=
                f"The lora adapter '{request.lora_name}' has already been"
                "loaded.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST)

        return None

    async def _check_unload_lora_adapter_request(
            self,
            request: UnloadLoraAdapterRequest) -> Optional[ErrorResponse]:
        # Check if either 'lora_name' or 'lora_int_id' is provided
        if not request.lora_name and not request.lora_int_id:
            return self.create_error_response(
                message=
                "either 'lora_name' and 'lora_int_id' needs to be provided.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST)

        # Check if the lora adapter with the given name exists
        if not any(lora_request.lora_name == request.lora_name
                   for lora_request in self.lora_requests):
            return self.create_error_response(
                message=
                f"The lora adapter '{request.lora_name}' cannot be found.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST)

        return None

    async def load_lora_adapter(
            self,
            request: LoadLoraAdapterRequest) -> Union[ErrorResponse, str]:
        error_check_ret = await self._check_load_lora_adapter_request(request)
        if error_check_ret is not None:
            return error_check_ret

        lora_name, lora_path = request.lora_name, request.lora_path
        unique_id = self.lora_id_counter.inc(1)
        self.lora_requests.append(
            LoRARequest(lora_name=lora_name,
                        lora_int_id=unique_id,
                        lora_path=lora_path))
        return f"Success: LoRA adapter '{lora_name}' added successfully."

    async def unload_lora_adapter(
            self,
            request: UnloadLoraAdapterRequest) -> Union[ErrorResponse, str]:
        error_check_ret = await self._check_unload_lora_adapter_request(request
                                                                        )
        if error_check_ret is not None:
            return error_check_ret

        lora_name = request.lora_name
        self.lora_requests = [
            lora_request for lora_request in self.lora_requests
            if lora_request.lora_name != lora_name
        ]
        return f"Success: LoRA adapter '{lora_name}' removed successfully."
