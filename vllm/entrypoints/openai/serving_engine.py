import asyncio
import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import Dict, List, Optional, Tuple, Union

from pydantic import Field
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing_extensions import Annotated

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest, ErrorResponse,
                                              LogProbs, ModelCard, ModelList,
                                              ModelPermission, LoraAddRequest,
                                              LoraErrorResponse)

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

    def __init__(self, engine: AsyncLLMEngine, served_model_names: List[str],
                 lora_modules: Optional[List[LoRAModulePath]]):
        self.engine = engine
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

        self.max_model_len = 0
        # Lazy initialized
        self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

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
            tokenizer_revision=engine_model_config.tokenizer_revision,
            trust_remote_code=engine_model_config.trust_remote_code,
            truncation_side="left")

    async def show_available_models(self) -> ModelList:
        """Show available models. Right now we only have one model."""
        model_cards = [
            ModelCard(id=served_model_name,
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

    def _create_logprobs(
        self,
        token_ids: List[int],
        top_logprobs: List[Optional[Dict[int, Logprob]]],
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
                assert logprobs.top_logprobs is not None
                logprobs.top_logprobs.append(None)
            else:
                token_logprob = step_top_logprobs[token_id].logprob
                token = step_top_logprobs[token_id].decoded_token
                logprobs.tokens.append(token)
                logprobs.token_logprobs.append(token_logprob)

                if num_output_top_logprobs:
                    assert logprobs.top_logprobs is not None
                    logprobs.top_logprobs.append({
                        # Convert float("-inf") to the
                        # JSON-serializable float that OpenAI uses
                        p.decoded_token: max(p.logprob, -9999.0)
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

    def create_lora_error_response(
        self,
        message: str,
        lora_name: str,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    ) -> LoraErrorResponse:
        return LoraErrorResponse(message=message, type=err_type, code=status_code.value, errorLoraName=lora_name)

    async def _check_model(
        self, request: Union[CompletionRequest, ChatCompletionRequest]
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
        self, request: Union[CompletionRequest, ChatCompletionRequest]
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
        request: Union[ChatCompletionRequest, CompletionRequest],
        prompt: Optional[str] = None,
        prompt_ids: Optional[List[int]] = None,
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
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

    def _get_lora_id(self, model_name: str):
        lora = next((lora for lora in self.lora_requests if model_name == lora.lora_name), None)
        return lora.lora_int_id if lora else None

    def _remove_lora(self, model_name: str):
        lora = next((lora for lora in self.lora_requests if model_name == lora.lora_name), None)
        if lora:
            self.lora_requests.remove(lora)

    def _add_lora(self, model):
        existing_lora = next((lora for lora in self.lora_requests if model.name == lora.lora_name), None)
        if not existing_lora:
            lora = LoRAModulePath(name=model.name, local_path=model.path)
            lora_request = LoRARequest(
                lora_name=lora.name,
                lora_int_id=len(self.lora_requests) + 1,
                lora_local_path=lora.local_path,
            )
            self.lora_requests.append(lora_request)
        return self.lora_requests[-1]

    async def add_lora_module(self, lora_add_request: LoraAddRequest, model_name: str) -> Union[bool, ErrorResponse]:
        # if lora_add_request.model.name is not same as model_name, return error
        if lora_add_request.model.name != model_name:
            return self.create_lora_error_response(
                message=f"Adding lora {lora_add_request.model.name} failed: model name mismatch with payload {model_name}.",
                err_type="UnprocessableEntityError",
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                lora_name=lora_add_request.model.name,
            )
        lora_request = self._add_lora(lora_add_request.model)
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.engine.engine.model_executor.add_lora, lora_request
            )
        except RuntimeError as e:
            self._remove_lora(lora_request.lora_name)
            return self.create_lora_error_response(
                message=f"Adding lora {lora_request.lora_name} failed: {e}",
                err_type="UnprocessableEntityError",
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                lora_name=lora_request.lora_name,
            )
        except ValueError as e:
            self._remove_lora(lora_request.lora_name)
            return self.create_lora_error_response(
                message=f"Adding lora {lora_request.lora_name} failed: {e}",
                err_type="BadRequestError",
                status_code=HTTPStatus.BAD_REQUEST,
                lora_name=lora_request.lora_name,
            )
        except Exception:
            self._remove_lora(lora_request.lora_name)
            return self.create_lora_error_response(
                message=f"Adding lora {lora_request.lora_name} failed.",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                lora_name=lora_request.lora_name,
            )
        return result

    async def remove_lora_module(self, model_name) -> Union[bool, ErrorResponse]:
        lora_id = self._get_lora_id(model_name)
        if lora_id is None:
            # Return a successful response even if the model doesn't exist
            return True
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.engine.engine.model_executor.remove_lora, lora_id
            )
        except Exception:
            return self.create_lora_error_response(
                message=f"Removing lora {model_name} failed.",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                lora_name=model_name,
            )
        else:
            self._remove_lora(model_name)
        return result
