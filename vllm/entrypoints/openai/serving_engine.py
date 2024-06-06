import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import Field
from typing_extensions import Annotated

from vllm.config import ModelConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest,
                                              EmbeddingRequest, ErrorResponse,
                                              ModelCard, ModelList,
                                              ModelPermission)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import Logprob
from vllm.transformers_utils.tokenizer import get_tokenizer

logger = init_logger(__name__)


@dataclass
class LoRAModulePath:
    name: str
    local_path: str


class OpenAIServing:

    def __init__(self, engine: AsyncLLMEngine, model_config: ModelConfig,
                 served_model_names: List[str],
                 lora_modules: Optional[List[LoRAModulePath]]):
        super().__init__()

        self.engine = engine
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
        self, request: Union[CompletionRequest, ChatCompletionRequest,
                             EmbeddingRequest]
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
                             EmbeddingRequest]
    ) -> Optional[LoRARequest]:
        if request.model in self.served_model_names:
            return None
        for lora in self.lora_requests:
            if request.model == lora.lora_name:
                return lora
        # if _check_model has been called earlier, this will be unreachable
        raise ValueError(f"The model `{request.model}` does not exist.")

    def _validate_prompt_and_tokenize(
            self,
            request: Union[ChatCompletionRequest, CompletionRequest,
                           EmbeddingRequest],
            prompt: Optional[str] = None,
            prompt_ids: Optional[List[int]] = None,
            truncate_prompt_tokens: Optional[Annotated[int,
                                                       Field(ge=1)]] = None,
            add_special_tokens: Optional[bool] = True
    ) -> Tuple[List[int], str]:
        if not (prompt or prompt_ids):
            raise ValueError("Either prompt or prompt_ids should be provided.")
        if (prompt and prompt_ids):
            raise ValueError(
                "Only one of prompt or prompt_ids should be provided.")

        if prompt_ids is None:
            # When using OpenAIServingChat for chat completions, for
            # most models the special tokens (e.g., BOS) have already
            # been added by the chat template. Therefore, we do not
            # need to add them again.
            # Set add_special_tokens to False (by default) to avoid
            # adding the BOS tokens again.
            tokenizer_kwargs: Dict[str, Any] = {
                "add_special_tokens": add_special_tokens
            }
            if truncate_prompt_tokens is not None:
                tokenizer_kwargs.update({
                    "truncation": True,
                    "max_length": truncate_prompt_tokens,
                })
            input_ids = self.tokenizer(prompt, **tokenizer_kwargs).input_ids
        elif truncate_prompt_tokens is not None:
            input_ids = prompt_ids[-truncate_prompt_tokens:]
        else:
            input_ids = prompt_ids

        input_text = prompt if prompt is not None else self.tokenizer.decode(
            prompt_ids)
        token_num = len(input_ids)

        # Note: EmbeddingRequest doesn't have max_tokens
        if isinstance(request, EmbeddingRequest):
            if token_num > self.max_model_len:
                raise ValueError(
                    f"This model's maximum context length is "
                    f"{self.max_model_len} tokens. However, you requested "
                    f"{token_num} tokens in the input for embedding "
                    f"generation. Please reduce the length of the input.", )
            return input_ids, input_text

        if request.max_tokens is None:
            if token_num >= self.max_model_len:
                raise ValueError(
                    f"This model's maximum context length is "
                    f"{self.max_model_len} tokens. However, you requested "
                    f"{token_num} tokens in the messages, "
                    f"Please reduce the length of the messages.", )
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

    def _get_decoded_token(self, logprob: Logprob, token_id: int) -> str:
        if logprob.decoded_token is not None:
            return logprob.decoded_token
        return self.tokenizer.decode(token_id)
