# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any, Final, Optional, Union, List
import hashlib
import array

import torch
import jinja2
from fastapi import Request

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.logger import RequestLogger
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (DetokenizeRequest,
                                              DetokenizeResponse,
                                              ErrorResponse,
                                              TokenizeChatRequest,
                                              TokenizeRequest,
                                              TokenizeResponse,
                                              TokenizerInfoResponse)
# yapf: enable
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


def compute_chunked_hashes(
    tokens: Union[torch.Tensor, List[int]],
    chunk_size: int,
    init_hash: str = ""
) -> List[str]:
    """
    Compute chunked hash chain for a sequence of tokens
    
    :param tokens: Input token sequence (PyTorch Tensor or List[int])
    :param chunk_size: Size of each chunk
    :param init_hash: Initial hash value (empty string by default)
    :return: List of hash values for all chunks, in order
    """
    def hash_chunk(chunk, prefix):
        if isinstance(chunk, torch.Tensor):
            chunk_bytes = chunk.cpu().to(torch.uint32).numpy().tobytes()
        else:
            chunk_bytes = array.array("I", chunk).tobytes()
        return hashlib.sha256(prefix.encode("ascii") + chunk_bytes).hexdigest()

    # Process in chunks
    chunks = [
        tokens[i:i + chunk_size] 
        for i in range(0, len(tokens), chunk_size)
    ]
    
    # Compute hash chain
    hash_chain = []
    current_hash = init_hash
    for chunk in chunks:
        current_hash = hash_chunk(chunk, current_hash)
        hash_chain.append(current_hash)
    
    return hash_chain


class OpenAIServingTokenization(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
        chat_template: Optional[str],
        chat_template_content_format: ChatTemplateContentFormatOption,
    ) -> None:
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger)

        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format

    async def create_tokenize(
        self,
        request: TokenizeRequest,
        raw_request: Request,
    ) -> Union[List[str], ErrorResponse]:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        request_id = f"tokn-{self._base_request_id(raw_request)}"

        try:
            lora_request = self._maybe_get_adapters(request)

            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            if isinstance(request, TokenizeChatRequest):
                tool_dicts = (None if request.tools is None else
                              [tool.model_dump() for tool in request.tools])
                (
                    _,
                    request_prompts,
                    engine_prompts,
                ) = await self._preprocess_chat(
                    request,
                    tokenizer,
                    request.messages,
                    tool_dicts=tool_dicts,
                    chat_template=request.chat_template or self.chat_template,
                    chat_template_content_format=self.
                    chat_template_content_format,
                    add_generation_prompt=request.add_generation_prompt,
                    continue_final_message=request.continue_final_message,
                    chat_template_kwargs=request.chat_template_kwargs,
                    add_special_tokens=request.add_special_tokens,
                )
            else:
                (request_prompts,
                 engine_prompts) = await self._preprocess_completion(
                     request,
                     tokenizer,
                     request.prompt,
                     add_special_tokens=request.add_special_tokens,
                 )
        except (ValueError, TypeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        input_ids: list[int] = []
        for i, engine_prompt in enumerate(engine_prompts):
            self._log_inputs(request_id,
                             request_prompts[i],
                             params=None,
                             lora_request=lora_request)

            if isinstance(engine_prompt,
                          dict) and "prompt_token_ids" in engine_prompt:
                input_ids.extend(engine_prompt["prompt_token_ids"])

        prefix_chunk_size = getattr(request, 'prefix_chunk_size', 256)
        bash_chain = compute_chunked_hashes(tokens=input_ids, chunk_size=prefix_chunk_size)
        return bash_chain

    async def create_detokenize(
        self,
        request: DetokenizeRequest,
        raw_request: Request,
    ) -> Union[DetokenizeResponse, ErrorResponse]:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        request_id = f"tokn-{self._base_request_id(raw_request)}"

        lora_request = self._maybe_get_adapters(request)

        tokenizer = await self.engine_client.get_tokenizer(lora_request)

        self._log_inputs(request_id,
                         request.tokens,
                         params=None,
                         lora_request=lora_request)

        prompt_input = await self._tokenize_prompt_input_async(
            request,
            tokenizer,
            request.tokens,
        )
        input_text = prompt_input["prompt"]

        return DetokenizeResponse(prompt=input_text)

    async def get_tokenizer_info(
        self, ) -> Union[TokenizerInfoResponse, ErrorResponse]:
        """Get comprehensive tokenizer information."""
        try:
            tokenizer = await self.engine_client.get_tokenizer()
            info = TokenizerInfo(tokenizer, self.chat_template).to_dict()
            return TokenizerInfoResponse(**info)
        except Exception as e:
            return self.create_error_response(
                f"Failed to get tokenizer info: {str(e)}")


@dataclass
class TokenizerInfo:
    tokenizer: AnyTokenizer
    chat_template: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        """Return the tokenizer configuration."""
        return self._get_tokenizer_config()

    def _get_tokenizer_config(self) -> dict[str, Any]:
        """Get tokenizer configuration directly from the tokenizer object."""
        config = dict(getattr(self.tokenizer, "init_kwargs", None) or {})

        # Remove file path fields
        config.pop("vocab_file", None)
        config.pop("merges_file", None)

        config = self._make_json_serializable(config)
        config["tokenizer_class"] = type(self.tokenizer).__name__
        if self.chat_template:
            config["chat_template"] = self.chat_template
        return config

    def _make_json_serializable(self, obj):
        """Convert any non-JSON-serializable objects to serializable format."""
        if hasattr(obj, "content"):
            return obj.content
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
