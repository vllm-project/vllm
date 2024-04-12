import asyncio
import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import (Dict, Iterable, Iterator, List, Literal, Optional, Tuple,
                    TypedDict, Union)

from pydantic import conint

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest, ErrorResponse,
                                              LogProbs, ModelCard, ModelList,
                                              ModelPermission)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import Logprob
from vllm.transformers_utils.tokenizer import get_tokenizer

logger = init_logger(__name__)


class InputStrings(TypedDict):
    input_text: str
    is_tokens: Literal[False]


class InputTokens(TypedDict):
    input_text: List[int]
    is_tokens: Literal[True]


@dataclass
class LoRAModulePath:
    name: str
    local_path: str


class OpenAIServing:

    def __init__(self, engine: AsyncLLMEngine, served_model: str,
                 lora_modules: Optional[List[LoRAModulePath]]):
        self.engine = engine
        self.served_model = served_model
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

        self.max_model_len = 0
        self.tokenizer = None

        try:
            event_loop = asyncio.get_running_loop()
        except RuntimeError:
            event_loop = None

        if event_loop is not None and event_loop.is_running():
            # If the current is instanced by Ray Serve,
            # there is already a running event loop
            event_loop.create_task(self._post_init())
        else:
            # When using single vLLM without engine_use_ray
            asyncio.run(self._post_init())

    async def _post_init(self):
        engine_model_config = await self.engine.get_model_config()
        self.max_model_len = engine_model_config.max_model_len

        # A separate tokenizer to map token IDs to strings.
        self.tokenizer = get_tokenizer(
            engine_model_config.tokenizer,
            tokenizer_mode=engine_model_config.tokenizer_mode,
            trust_remote_code=engine_model_config.trust_remote_code,
            truncation_side="left")

    async def show_available_models(self) -> ModelList:
        """Show available models. Right now we only have one model."""
        model_cards = [
            ModelCard(id=self.served_model,
                      root=self.served_model,
                      permission=[ModelPermission()])
        ]
        lora_cards = [
            ModelCard(id=lora.lora_name,
                      root=self.served_model,
                      permission=[ModelPermission()])
            for lora in self.lora_requests
        ]
        model_cards.extend(lora_cards)
        return ModelList(data=model_cards)

    def _create_logprobs(
        self,
        token_ids: List[int],
        top_logprobs: Optional[List[Optional[Dict[int, Logprob]]]] = None,
        num_output_top_logprobs: Optional[int] = None,
        initial_text_offset: int = 0,
    ) -> LogProbs:
        """Create OpenAI-style logprobs."""
        logprobs = LogProbs()
        last_token_len = 0
        if num_output_top_logprobs:
            logprobs.top_logprobs = []

        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None:
                token = self.tokenizer.decode(token_id)
                logprobs.tokens.append(token)
                logprobs.token_logprobs.append(None)
                logprobs.top_logprobs.append(None)
            else:
                token_logprob = step_top_logprobs[token_id].logprob
                token = step_top_logprobs[token_id].decoded_token
                logprobs.tokens.append(token)
                logprobs.token_logprobs.append(token_logprob)

                if num_output_top_logprobs:
                    logprobs.top_logprobs.append({
                        p.decoded_token: p.logprob
                        for i, p in step_top_logprobs.items()
                    } if step_top_logprobs else None)

            if len(logprobs.text_offset) == 0:
                logprobs.text_offset.append(initial_text_offset)
            else:
                logprobs.text_offset.append(logprobs.text_offset[-1] +
                                            last_token_len)
            last_token_len = len(token)
        return logprobs

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
        self, request: Union[CompletionRequest, ChatCompletionRequest]
    ) -> Optional[ErrorResponse]:
        if request.model == self.served_model:
            return None
        if request.model in [lora.lora_name for lora in self.lora_requests]:
            return None
        return self.create_error_response(
            message=f"The model `{request.model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND)

    def _maybe_get_lora(
        self, request: Union[CompletionRequest, ChatCompletionRequest]
    ) -> Optional[LoRARequest]:
        if request.model == self.served_model:
            return None
        for lora in self.lora_requests:
            if request.model == lora.lora_name:
                return lora
        # if _check_model has been called earlier, this will be unreachable
        raise ValueError("The model `{request.model}` does not exist.")

    def _validate_prompt_and_tokenize(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest],
        prompt: Optional[str] = None,
        prompt_ids: Optional[List[int]] = None,
        truncate_prompt_tokens: Optional[conint(ge=1)] = None
    ) -> Tuple[List[int], str]:
        if not (prompt or prompt_ids):
            raise ValueError("Either prompt or prompt_ids should be provided.")
        if (prompt and prompt_ids):
            raise ValueError(
                "Only one of prompt or prompt_ids should be provided.")

        if prompt_ids is None:
            tokenizer_kwargs = {} if truncate_prompt_tokens is None else {
                "truncation": True,
                "max_length": truncate_prompt_tokens,
            }
            input_ids = self.tokenizer(prompt, **tokenizer_kwargs).input_ids
        elif truncate_prompt_tokens is not None:
            input_ids = prompt_ids[-truncate_prompt_tokens:]
        else:
            input_ids = prompt_ids

        input_text = prompt if prompt is not None else self.tokenizer.decode(
            prompt_ids)
        token_num = len(input_ids)

        if request.max_tokens is None:
            request.max_tokens = self.max_model_len - token_num

        if token_num + request.max_tokens > self.max_model_len:
            raise ValueError(
                f"This model's maximum context length is "
                f"{self.max_model_len} tokens. However, you requested "
                f"{request.max_tokens + token_num} tokens "
                f"({token_num} in the messages, "
                f"{request.max_tokens} in the completion). "
                f"Please reduce the length of the messages or completion.", )
        else:
            return input_ids, input_text

    def _tokenize_input_text(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest],
        input_text: Union[str, List[int]],
        truncate_prompt_tokens: Optional[conint(ge=1)] = None,
    ) -> Tuple[List[int], str]:
        """A simpler implementation of
        :meth:`~vllm.entrypoints.openai.serving_engine.OpenAIServing._tokenize_input_text_or_texts`
        that assumes single input."""
        return next(
            self._tokenize_input_texts(
                request,
                [input_text],
                truncate_prompt_tokens=truncate_prompt_tokens,
            ))

    def _tokenize_input_texts(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest],
        input_texts: Iterable[Union[str, List[int]]],
        truncate_prompt_tokens: Optional[conint(ge=1)] = None,
    ) -> Iterator[Tuple[List[int], str]]:
        """A simpler implementation of
        :meth:`~vllm.entrypoints.openai.serving_engine.OpenAIServing._tokenize_input_text_or_texts`
        that assumes multiple input."""
        for input_text in input_texts:
            if isinstance(input_text, str):
                yield self._validate_prompt_and_tokenize(
                    request,
                    prompt=input_text,
                    truncate_prompt_tokens=truncate_prompt_tokens,
                )
            else:
                yield self._validate_prompt_and_tokenize(
                    request,
                    prompt_ids=input_text,
                    truncate_prompt_tokens=truncate_prompt_tokens,
                )

    def _parse_input_element(
        self,
        elem: Union[str, int, List[int]],
    ) -> Union[InputStrings, InputTokens]:
        if isinstance(elem, str):
            # case 2: array of strings
            return InputStrings(prompt=elem, is_tokens=False)
        if isinstance(elem, int):
            # case 3: array of tokens
            return InputTokens(prompt=[elem], is_tokens=True)
        if isinstance(elem, list):
            # case 4: array of token arrays
            return InputTokens(prompt=elem, is_tokens=True)

    def _parse_input_text_or_texts(
        self,
        input_text_or_texts: Union[str, List[str], List[int], List[List[int]]],
    ) -> List[Union[InputStrings, InputTokens]]:
        if isinstance(input_text_or_texts, str):
            # case 1: a string
            return [self._parse_input_element(input_text_or_texts)]

        if isinstance(input_text_or_texts, list):
            return [self._parse_input_element(e) for e in input_text_or_texts]

        raise ValueError("prompt must be a string, array of strings, "
                         "array of tokens, or array of token arrays")

    def _tokenize_input_text_or_texts(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest],
        input_text_or_texts: Union[str, List[str], List[int], List[List[int]]],
        truncate_prompt_tokens: Optional[conint(ge=1)] = None,
    ) -> Iterator[Tuple[List[int], str]]:
        """Tokenize/detokenize depending on the input format.
        
        According to `OpenAI API <https://platform.openai.com/docs/api-reference/embeddings/create>`_
        , each input can be a string or array of tokens. Note that each request
        can pass one or more inputs.
        """
        for input_ in self._parse_input_text_or_texts(input_text_or_texts):
            # Although our type checking is based on mypy,
            # VSCode Pyright extension should still work properly
            # "is True" is required for Pyright to perform type narrowing
            # See: https://github.com/microsoft/pyright/issues/7672
            if input_["is_tokens"] is True:
                yield self._validate_prompt_and_tokenize(
                    request,
                    prompt_ids=input_["input_text"],
                    truncate_prompt_tokens=truncate_prompt_tokens,
                )
            else:
                yield self._validate_prompt_and_tokenize(
                    request,
                    prompt=input_["input_text"],
                    truncate_prompt_tokens=truncate_prompt_tokens,
                )
